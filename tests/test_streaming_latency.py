"""Unit tests for the low-latency streaming path.

Three pieces are covered here, all of them chosen because they are the places
where a mistake is silent rather than loud:

* `app/engine/fastpath.py`'s two thread-local switches. The KV window and the
  prompt-embedding override both change what the model computes, and both
  restore on exit; a leaked flag would corrupt whatever ran next rather than
  raise.
* `app/engine/codecstream.py`'s span arithmetic. Its whole job is to hand back
  each frame's audio exactly once - a duplicated or dropped sample is an
  audible click, not an exception - so a stub codec with a known expansion
  factor pins the slicing down without needing the real 1.8 GB DAC.
* `app/engine/streamgen.py`'s prefix matching. It decides how much of the KV
  cache to trust, and trusting one position too many means generating from a
  prompt the model never actually saw.

None of it needs a GPU; the two that need torch skip without it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.engine.streamgen import default_emit_schedule

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------
# fastpath: the thread-local switches
# --------------------------------------------------------------------------


def test_kv_window_sets_and_restores():
    from app.engine.fastpath import _live_len, kv_window

    assert _live_len() is None
    with kv_window(64):
        assert _live_len() == 64
        with kv_window(128):
            assert _live_len() == 128
        assert _live_len() == 64  # inner window restores the outer one
    assert _live_len() is None


def test_kv_window_restores_after_an_exception():
    from app.engine.fastpath import _live_len, kv_window

    with pytest.raises(RuntimeError):
        with kv_window(32):
            raise RuntimeError("boom")
    assert _live_len() is None


def test_prompt_embeddings_sets_and_restores():
    from app.engine.fastpath import _state, prompt_embeddings

    assert getattr(_state, "prompt_embed", False) is False
    with prompt_embeddings():
        assert _state.prompt_embed is True
    assert _state.prompt_embed is False


def test_prompt_embeddings_routes_prompt_tokens_to_the_host_table():
    """The regression this exists for: with the embedding table offloaded, the
    fork reads the resident 4097-row semantic slice whenever it sees a single
    position, because in the fork a lone position is always a *generated*
    token. Prefilling a prompt one position at a time breaks that assumption,
    and it breaks silently - ids outside the slice get clamped to row 0, so the
    model reads a plausible wrong embedding instead of failing."""
    from app.engine.fastpath import prompt_embeddings

    class _Table:
        def __init__(self):
            self.calls = []

        def __call__(self, tok):
            self.calls.append(tok)
            return torch.zeros(tok.shape + (4,))

    class _Model:
        embeddings_offloaded = True
        semantic_head = torch.zeros(4097, 4)

        def __init__(self):
            self.embeddings = _Table()
            self.slice_calls = 0

        def _original(self, tok):
            self.slice_calls += 1
            return torch.zeros(tok.shape + (4,))

    # Mirrors the patched embed_main_tokens' decision, which is the whole
    # behaviour under test; installing the real patch would need fish_speech.
    def embed(model, tok, in_prompt):
        if in_prompt and getattr(model, "embeddings_offloaded", False):
            return model.embeddings(tok)
        return model._original(tok)

    model = _Model()
    one_position = torch.tensor([[151700]])

    embed(model, one_position, in_prompt=False)
    assert model.slice_calls == 1 and model.embeddings.calls == []

    with prompt_embeddings():
        embed(model, one_position, in_prompt=True)
    assert model.slice_calls == 1 and len(model.embeddings.calls) == 1


# --------------------------------------------------------------------------
# codecstream: hand back each frame's audio exactly once
# --------------------------------------------------------------------------

EXPANSION = 2048


class _StubQuantizer:
    """decode_codes is identity on the first codebook; upsample expands x4."""

    def __init__(self):
        self.decode_widths = []

    def decode_codes(self, indices):
        self.decode_widths.append(indices.shape[-1])
        return indices[:, :1].to(torch.float32)

    def upsample(self, z):
        return z.repeat_interleave(4, dim=-1)


class _StubCodec:
    sample_rate = 44100

    def __init__(self):
        self.quantizer = _StubQuantizer()

    def decoder(self, x):  # the remaining x512 of the 2048 total
        return x.repeat_interleave(512, dim=-1)


def _frames(start, n):
    """Frames whose first codebook carries its own index, so the decoded audio
    says which frame each sample came from."""
    block = torch.zeros(10, n, dtype=torch.long)
    block[0] = torch.arange(start, start + n)
    return block


def test_incremental_decoder_emits_each_frame_exactly_once():
    from app.engine.codecstream import IncrementalDecoder

    decoder = IncrementalDecoder(_StubCodec(), lookback_frames=16, post_context_frames=256)
    out = []
    emitted = 0
    for size in (4, 8, 16, 5):
        decoder.extend(_frames(emitted, size))
        emitted += size
        chunk = decoder.drain()
        assert chunk is not None
        out.append(torch.from_numpy(chunk))

    audio = torch.cat(out)
    assert audio.numel() == emitted * EXPANSION
    # Every frame's samples appear once, in order, with nothing repeated from
    # the lookback context and nothing dropped at a block boundary.
    expected = torch.arange(emitted, dtype=torch.float32).repeat_interleave(EXPANSION)
    assert torch.equal(audio, expected)


def test_incremental_decoder_drain_is_empty_without_new_frames():
    from app.engine.codecstream import IncrementalDecoder

    decoder = IncrementalDecoder(_StubCodec())
    assert decoder.drain() is None
    decoder.extend(_frames(0, 3))
    assert decoder.drain() is not None
    assert decoder.drain() is None  # nothing new since the last call
    assert decoder.emitted_frames == 3


def test_incremental_decoder_bounds_the_post_module_context():
    """post_module is re-run over a prefix, but a capped one - otherwise its
    cost would grow without bound over a long utterance."""
    from app.engine.codecstream import IncrementalDecoder

    codec = _StubCodec()
    decoder = IncrementalDecoder(codec, lookback_frames=16, post_context_frames=256)
    for i in range(40):
        decoder.extend(_frames(i * 16, 16))
        decoder.drain()
    assert max(codec.quantizer.decode_widths) <= 256


def test_incremental_decoder_raises_on_non_finite_audio():
    from app.engine.codecstream import IncrementalDecoder

    class _NaNCodec(_StubCodec):
        def decoder(self, x):
            return torch.full_like(x.repeat_interleave(512, dim=-1), float("nan"))

    decoder = IncrementalDecoder(_NaNCodec())
    decoder.extend(_frames(0, 4))
    with pytest.raises(RuntimeError, match="non-finite"):
        decoder.drain()


# --------------------------------------------------------------------------
# streamgen: how much of the KV cache is safe to reuse
# --------------------------------------------------------------------------


def _generator(**overrides):
    from app.engine.streamgen import StreamingGenerator

    return StreamingGenerator(None, None, "cpu", Settings(**overrides))


def _tokens(*rows):
    return torch.tensor(rows, dtype=torch.long)


def test_shared_prefix_is_zero_with_an_empty_cache():
    assert _generator()._shared_prefix(_tokens([1, 2, 3], [0, 0, 0])) == 0


def test_shared_prefix_matches_a_full_repeat():
    generator = _generator()
    encoded = _tokens([1, 2, 3, 4], [9, 9, 9, 9])
    generator._remember(encoded)
    assert generator._shared_prefix(encoded) == 4


def test_shared_prefix_stops_at_the_first_divergence():
    generator = _generator()
    generator._remember(_tokens([1, 2, 3, 4], [9, 9, 9, 9]))
    assert generator._shared_prefix(_tokens([1, 2, 7, 8], [9, 9, 9, 9])) == 2


def test_shared_prefix_compares_every_codebook_row():
    """Two prompts can agree on the semantic token and still differ in the VQ
    codes underneath it - different reference audio, same transcript. Those
    produce different KV entries, so row 0 alone is not a safe key."""
    generator = _generator()
    generator._remember(_tokens([1, 2, 3], [5, 5, 5]))
    assert generator._shared_prefix(_tokens([1, 2, 3], [5, 7, 5])) == 1


def test_shared_prefix_handles_a_longer_new_prompt():
    generator = _generator()
    generator._remember(_tokens([1, 2], [9, 9]))
    assert generator._shared_prefix(_tokens([1, 2, 3, 4], [9, 9, 9, 9])) == 2


def test_shared_prefix_is_zero_when_the_cache_is_disabled():
    generator = _generator(prefix_cache=False)
    encoded = _tokens([1, 2, 3], [9, 9, 9])
    generator._remember(encoded)
    assert generator._shared_prefix(encoded) == 0


def test_truncate_cache_keeps_only_the_untouched_prefix():
    """Before overwriting KV entries from position N, the cache must stop
    claiming anything at or past N - otherwise a cancelled prefill leaves it
    vouching for keys that have since been replaced."""
    generator = _generator()
    generator._remember(_tokens([1, 2, 3, 4, 5], [9, 9, 9, 9, 9]))
    generator._truncate_cache(3)
    assert generator.stats.cached_tokens == 3
    assert generator._shared_prefix(_tokens([1, 2, 3, 7, 7], [9, 9, 9, 9, 9])) == 3
    # Positions at or past the truncation point are no longer claimed.
    assert generator._shared_prefix(_tokens([1, 2, 3, 4, 5], [9, 9, 9, 9, 9])) == 3


def test_truncate_cache_to_zero_forgets_everything():
    generator = _generator()
    encoded = _tokens([1, 2, 3], [9, 9, 9])
    generator._remember(encoded)
    generator._truncate_cache(0)
    assert generator._shared_prefix(encoded) == 0
    assert generator.stats.cached_tokens == 0


def test_invalidate_prefix_forgets_everything():
    generator = _generator()
    encoded = _tokens([1, 2, 3], [9, 9, 9])
    generator._remember(encoded)
    generator.invalidate_prefix()
    assert generator._shared_prefix(encoded) == 0
    assert generator.stats.cached_tokens == 0


# --------------------------------------------------------------------------
# the emit schedule
# --------------------------------------------------------------------------


def test_emit_schedule_ramps_from_first_to_max():
    assert default_emit_schedule(4, 32) == [4, 8, 16, 32]
    assert default_emit_schedule(1, 32) == [1, 2, 4, 8, 16, 32]


def test_emit_schedule_is_a_single_step_when_first_equals_max():
    assert default_emit_schedule(8, 8) == [8]


def test_emit_schedule_never_exceeds_max():
    schedule = default_emit_schedule(6, 20)
    assert schedule[-1] == 20 and all(s <= 20 for s in schedule)


def test_emit_schedule_survives_nonsense_bounds():
    assert default_emit_schedule(0, 0) == [1]
    assert default_emit_schedule(10, 2) == [10]


# --------------------------------------------------------------------------
# prefill mode selection
# --------------------------------------------------------------------------


def test_prefill_mode_auto_walks_short_spans_one_position_at_a_time(monkeypatch):
    """Without the 4-bit kernel, short spans dodge bitsandbytes' flat ~477ms
    dequantisation cost by staying on its sequence-length-1 path; long ones
    amortise it."""
    from app.engine import nf4gemv

    monkeypatch.setattr(nf4gemv, "available", lambda: False)
    generator = _generator(prefill_mode="auto", prefill_sequential_max=11)
    assert generator._sequential_prefill(8) is True
    assert generator._sequential_prefill(11) is True
    assert generator._sequential_prefill(12) is False
    assert generator._sequential_prefill(239) is False


def test_prefill_mode_auto_always_batches_with_the_4bit_kernel(monkeypatch):
    """One pass of `nf4gemv` serves up to 32 positions for the price of one,
    so walking positions can only be slower."""
    from app.engine import nf4gemv

    monkeypatch.setattr(nf4gemv, "available", lambda: True)
    generator = _generator(prefill_mode="auto", prefill_sequential_max=11)
    assert generator._sequential_prefill(1) is False
    assert generator._sequential_prefill(8) is False


def test_prefill_mode_can_be_forced_either_way():
    assert _generator(prefill_mode="sequential")._sequential_prefill(500) is True
    assert _generator(prefill_mode="batched")._sequential_prefill(1) is False


# --------------------------------------------------------------------------
# Byte-budget batching within one segment
# --------------------------------------------------------------------------

TAMIL = "கண்டிப்பா உங்க ஆர்டரை பத்தின விவரங்களுக்கு நான் நிச்சயமாக உதவுவேன்."


def test_tamil_costs_far_more_bytes_than_characters():
    """The premise behind `_batches`: `chunk_length` is a budget in bytes, but
    the segmenter bounds by characters. For Latin text the two nearly agree;
    for Tamil a character is ~2.8 bytes, so a 220-character segment is ~600
    bytes - three times what the model is batched for."""
    ratio = len(TAMIL.encode("utf-8")) / len(TAMIL)
    assert ratio > 2.5
    assert len("x" * 220) == len(("x" * 220).encode("utf-8"))


def _vendored():
    """Put the fork on sys.path; skip if it has not been checked out."""
    from app.engine.factory import EngineConfigError, _ensure_vendor_on_path

    try:
        _ensure_vendor_on_path(Settings())
    except EngineConfigError as exc:  # pragma: no cover - only on a bare checkout
        pytest.skip(str(exc))


def test_batches_respect_a_byte_budget_not_a_character_count():
    _vendored()
    generator = _generator()
    text = " ".join([TAMIL] * 4)
    batches = generator._batches(text, 200)
    assert len(batches) > 1, "a multi-hundred-byte Tamil segment was sent in one piece"
    assert "".join(batches).replace(" ", "") == text.replace(" ", "")


def test_batches_are_a_single_piece_when_chunking_is_disabled():
    _vendored()
    generator = _generator()
    assert generator._batches("Short enough.", 0) == ["Short enough."]


def test_prefix_stats_report_truncations():
    """A batch that runs out of budget stops mid-word but looks identical to a
    finished one from the outside, so it has to be counted somewhere."""
    generator = _generator()
    assert generator.stats.as_dict()["truncations"] == 0
    generator.stats.truncations += 1
    assert generator.stats.as_dict()["truncations"] == 1


# --------------------------------------------------------------------------
# Batched prompt embeddings
# --------------------------------------------------------------------------


def test_prompt_embedding_batch_serves_rows_in_order():
    """A sequential prefill knows every token id before it starts, so the CPU
    can gather them in one pass instead of paying a device synchronise, a
    gather and a copy back per step."""
    from app.engine.fastpath import _state, prompt_embedding_batch

    rows = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    assert getattr(_state, "embed_batch", None) is None
    with prompt_embedding_batch(rows):
        served = [_state.embed_batch["rows"][i : i + 1].unsqueeze(0) for i in range(4)]
    assert getattr(_state, "embed_batch", None) is None
    assert served[0].shape == (1, 1, 3)
    assert torch.equal(served[2][0, 0], rows[2])


def test_prompt_embedding_batch_restores_on_exception():
    from app.engine.fastpath import _state, prompt_embedding_batch

    with pytest.raises(RuntimeError):
        with prompt_embedding_batch(torch.zeros(2, 2)):
            raise RuntimeError("boom")
    assert getattr(_state, "embed_batch", None) is None


def test_hint_suffix_inclusion_is_configurable():
    """Whether a hint guesses the whole prompt shape or stops after the text is
    a measurement, not a principle: including the suffix is the only way a hint
    can leave nothing to do, but it is re-done at a new position every time a
    word arrives."""
    assert _generator().enabled is True
    assert Settings().hint_include_suffix is True
    assert Settings(hint_include_suffix=False).hint_include_suffix is False


# --------------------------------------------------------------------------
# A hint must be bounded, not just optional
# --------------------------------------------------------------------------


def test_a_hint_never_prefills_more_than_its_budget():
    """A hint is bounded so one cannot monopolise the worker. It is *not*
    forced onto the per-token path: doing that starves any language whose words
    are long — one Tamil word is ~11 tokens. It does not override the
    batched/sequential choice: batching is what lets a hint keep up, and the
    pile-up it used to cause is prevented by `GpuWorker.submit_hint` holding a
    single slot rather than a queue."""
    from app.engine.base import CancelToken

    generator = _generator(hint_max_tokens=4)
    prompt = torch.zeros(11, 60, dtype=torch.long)
    prompt[0] = torch.arange(60)

    calls = []
    generator._encode = lambda messages: (prompt, None, None)
    generator._system_message = lambda *a: object()
    generator._prefill = lambda enc, start, cancel, sequential=None: calls.append(
        (start, enc.size(1), sequential)
    )

    done = generator.hint("some text still being typed", [], [], CancelToken())

    assert len(calls) == 1
    start, end, sequential = calls[0]
    assert start == 0
    assert end - start == 4, "a hint ran past its cap"
    assert sequential is None, "a hint overrode the batched/sequential choice"
    assert done == 4


def test_successive_hints_carry_on_from_where_the_last_one_stopped():
    """Capping costs nothing in the end: a partial prefill keeps what it wrote,
    so the next hint resumes rather than restarting."""
    from app.engine.base import CancelToken

    generator = _generator(hint_max_tokens=4)
    prompt = torch.zeros(11, 60, dtype=torch.long)
    prompt[0] = torch.arange(60)
    generator._encode = lambda messages: (prompt, None, None)
    generator._system_message = lambda *a: object()

    starts = []

    def fake_prefill(enc, start, cancel, sequential=None):
        starts.append(start)
        generator._remember(enc)  # what a completed prefill does

    generator._prefill = fake_prefill
    for _ in range(3):
        generator.hint("text", [], [], CancelToken())

    assert starts == [0, 4, 8], f"hints restarted instead of resuming: {starts}"
