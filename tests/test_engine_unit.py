"""Unit tests for the engine package that run with no GPU and no vendored
fish-speech checkout.

Everything here exercises pure-Python logic: `CancelToken`, the worker's job
loop driven with stubbed generation (standing in for `decode_one_token` /
`generate_long`, which live in the vendored fork and are not importable in
this environment), the reference cache, and factory validation. Nothing
imports `torch` or `fish_speech` at module scope - `app/engine/worker.py`,
`app/engine/references.py`, and `app/engine/factory.py` are all written so
that importing them, and calling their non-GPU code paths, works without
either dependency present. GPU-dependent behaviour (a real model load, a
real synthesis) is marked and skipped below rather than faked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.engine.base import CancelToken, Cancelled
from app.engine.factory import EngineConfigError, _validate_checkpoint, _validate_precision, _ensure_vendor_on_path
from app.engine.references import ReferenceManager, _BoundedCache
from app.engine.worker import ChunkMessage, DoneMessage, ErrorMessage, GpuWorker, _Job
from app.protocol import ReferenceAudio, TTSParams


# --------------------------------------------------------------------------
# CancelToken
# --------------------------------------------------------------------------


def test_cancel_token_starts_uncancelled():
    token = CancelToken()
    assert not token.cancelled
    token.raise_if_cancelled()  # must not raise


def test_cancel_token_raises_once_cancelled():
    token = CancelToken()
    token.cancel()
    assert token.cancelled
    with pytest.raises(Cancelled):
        token.raise_if_cancelled()


def test_cancel_token_cancel_is_idempotent():
    token = CancelToken()
    token.cancel()
    token.cancel()  # must not raise or toggle back
    assert token.cancelled


# --------------------------------------------------------------------------
# GpuWorker._wrap_cancellable: the per-token cancellation hook
# --------------------------------------------------------------------------


def _worker(**overrides) -> GpuWorker:
    settings = Settings(device="cuda", half=True, **overrides)
    return GpuWorker(settings)


def test_wrap_cancellable_passes_through_when_not_cancelled():
    worker = _worker()
    calls = []

    def stub_decode(*args, **kwargs):
        calls.append((args, kwargs))
        return "a-token"

    wrapped = worker._wrap_cancellable(stub_decode)
    worker._current_cancel = CancelToken()

    assert wrapped(1, x=2) == "a-token"
    assert calls == [((1,), {"x": 2})]


def test_wrap_cancellable_raises_before_reaching_decode_step():
    """The core barge-in trick: cancellation is caught *before* the wrapped
    callable runs, not after - so a cancelled job never samples another
    token, only ever finishes the one already in flight."""
    worker = _worker()
    calls = []

    def stub_decode(*args, **kwargs):
        calls.append(1)
        return "a-token"

    wrapped = worker._wrap_cancellable(stub_decode)
    token = CancelToken()
    worker._current_cancel = token
    token.cancel()

    with pytest.raises(Cancelled):
        wrapped()
    assert calls == []  # never called the real decode step


def test_wrap_cancellable_ignores_tokens_from_other_jobs():
    """Only `self._current_cancel` (the in-flight job) gates decoding; a
    cancelled token belonging to some other, unrelated job must not stop
    this one - relevant because worker.submit() hands each job its own
    CancelToken and the worker only ever tracks the one it's running."""
    worker = _worker()
    stale_token = CancelToken()
    stale_token.cancel()

    wrapped = worker._wrap_cancellable(lambda: "ok")
    worker._current_cancel = CancelToken()  # a fresh, uncancelled token

    assert wrapped() == "ok"


# --------------------------------------------------------------------------
# GpuWorker._handle: the job loop, with generation stubbed out
# --------------------------------------------------------------------------


class _FakeCodes:
    """Stands in for the (num_codebooks, n) tensor of code frames that
    `StreamingGenerator.generate` yields; only `.shape` is read by `_handle`."""

    def __init__(self, frames: int):
        self.shape = (4, frames)


class _StubDecoder:
    """Stands in for `IncrementalDecoder`: every block becomes a fixed 0.5s of
    audio, so the job loop's metrics are checkable without a codec."""

    SAMPLE_RATE = 44100

    def __init__(self):
        self.extended = []

    def extend(self, block):
        self.extended.append(block)

    def drain(self):
        import numpy as np

        return np.zeros(self.SAMPLE_RATE // 2, dtype=np.float32)

    def drain_async(self):
        samples = self.drain()

        class _Ready:
            def ready(self):
                return True

            def result(self):
                return samples

        return _Ready()


def _make_job(cancel: CancelToken | None = None) -> tuple[_Job, "list"]:
    import queue

    params = TTSParams(text="hello")
    out: "queue.Queue[object]" = queue.Queue()
    job = _Job(text="hello world", params=params, cancel=cancel or CancelToken(), out=out)
    return job, out


def _drain(out) -> list:
    import queue

    items = []
    while True:
        try:
            items.append(out.get_nowait())
        except queue.Empty:
            break
    return items


class _StubReferences:
    def __init__(self):
        self.resolve_calls = 0

    def resolve(self, references, reference_id):
        self.resolve_calls += 1
        return [], []


def test_handle_streams_chunks_then_done():
    worker = _worker()
    worker._references = _StubReferences()
    worker.sample_rate = _StubDecoder.SAMPLE_RATE
    worker._make_decoder = lambda: _StubDecoder()
    worker._stream_blocks = lambda *a, **k: iter([_FakeCodes(10), _FakeCodes(5)])

    job, out = _make_job()
    worker._handle(job)

    messages = _drain(out)
    assert [type(m) for m in messages] == [ChunkMessage, ChunkMessage, DoneMessage]
    done = messages[-1]
    assert done.metrics.segments == 2
    assert done.metrics.token_frames == 15
    assert done.metrics.audio_seconds == pytest.approx(1.0)
    assert not done.metrics.cancelled
    assert done.metrics.error is None
    assert worker._current_cancel is None  # cleared, ready for the next job


def test_handle_mid_stream_cancellation_yields_partial_audio_then_done():
    """Simulates cancellation arriving while `generate_long` is mid-batch:
    one segment already decoded and queued, then Cancelled unwinds out of
    the (stubbed) generator exactly as it would out of the real
    decode_one_token -> decode_n_tokens -> generate -> generate_long chain."""
    worker = _worker()
    worker._references = _StubReferences()

    def fake_stream(*a, **k):
        yield _FakeCodes(10)
        raise Cancelled()

    worker.sample_rate = _StubDecoder.SAMPLE_RATE
    worker._make_decoder = lambda: _StubDecoder()
    worker._stream_blocks = fake_stream

    job, out = _make_job()
    worker._handle(job)

    messages = _drain(out)
    assert [type(m) for m in messages] == [ChunkMessage, DoneMessage]
    done = messages[-1]
    assert done.metrics.cancelled is True
    assert done.metrics.segments == 1  # the chunk decoded before cancellation lands
    assert worker._current_cancel is None


def test_handle_pre_cancelled_job_never_starts_generation():
    """A job cancelled before it was even dequeued must drain cleanly
    without touching references or generation at all."""
    worker = _worker()
    refs = _StubReferences()
    worker._references = refs
    worker._stream_blocks = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not be called for a pre-cancelled job")
    )

    cancel = CancelToken()
    cancel.cancel()
    job, out = _make_job(cancel=cancel)
    worker._handle(job)

    messages = _drain(out)
    assert [type(m) for m in messages] == [DoneMessage]
    assert messages[0].metrics.cancelled is True
    assert refs.resolve_calls == 0


def test_handle_reports_ordinary_exceptions_as_error_not_cancelled():
    worker = _worker()
    worker._references = _StubReferences()

    def fake_stream(*a, **k):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this a generator function

    worker._make_decoder = lambda: _StubDecoder()
    worker._stream_blocks = fake_stream

    job, out = _make_job()
    worker._handle(job)

    messages = _drain(out)
    assert [type(m) for m in messages] == [ErrorMessage]
    assert "boom" in str(messages[0].error)


def test_worker_ready_for_next_job_after_cancellation():
    """A cancelled job must not poison the worker: the very next job runs
    to completion normally."""
    worker = _worker()
    worker._references = _StubReferences()
    worker.sample_rate = _StubDecoder.SAMPLE_RATE
    worker._make_decoder = lambda: _StubDecoder()

    cancelled_job, out1 = _make_job()

    def cancel_then_raise(*a, **k):
        yield _FakeCodes(1)
        raise Cancelled()

    worker._stream_blocks = cancel_then_raise
    worker._handle(cancelled_job)
    assert any(isinstance(m, DoneMessage) and m.metrics.cancelled for m in _drain(out1))

    next_job, out2 = _make_job()
    worker._stream_blocks = lambda *a, **k: iter([_FakeCodes(1)])
    worker._handle(next_job)

    messages = _drain(out2)
    assert [type(m) for m in messages] == [ChunkMessage, DoneMessage]
    assert not messages[-1].metrics.cancelled


# --------------------------------------------------------------------------
# references.py: _BoundedCache eviction/keying
# --------------------------------------------------------------------------


def test_bounded_cache_basic_get_put():
    cache = _BoundedCache(max_entries=2)
    assert cache.get("a") is None
    cache.put("a", "value-a")
    assert cache.get("a") == "value-a"
    assert len(cache) == 1


def test_bounded_cache_evicts_least_recently_used():
    cache = _BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a" (least recently touched)

    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    assert len(cache) == 2


def test_bounded_cache_get_refreshes_recency():
    cache = _BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # touch "a" so "b" becomes the least recently used
    cache.put("c", 3)  # should evict "b", not "a"

    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_bounded_cache_put_overwrite_does_not_grow():
    cache = _BoundedCache(max_entries=2)
    cache.put("a", 1)
    cache.put("a", 2)
    assert len(cache) == 1
    assert cache.get("a") == 2


# --------------------------------------------------------------------------
# references.py: ReferenceManager content-hash caching
# --------------------------------------------------------------------------


class _FakeResolved:
    def __init__(self, text):
        self.tokens = f"tokens-for-{text}"
        self.text = text


def test_reference_manager_caches_by_content_hash():
    settings = Settings(codec_decode_only=True)
    manager = ReferenceManager(settings, gpu_codec=None, gpu_device="cpu")

    encode_calls = []

    def fake_encode(audio_bytes, text):
        encode_calls.append((audio_bytes, text))
        return _FakeResolved(text)

    manager._encode = fake_encode

    clip = ReferenceAudio(audio=b"same-bytes", text="hello")
    tokens1, texts1 = manager.resolve([clip], None)
    tokens2, texts2 = manager.resolve([clip], None)  # identical bytes again

    assert encode_calls == [(b"same-bytes", "hello")]  # encoded exactly once
    assert tokens1 == tokens2 == ["tokens-for-hello"]
    assert texts1 == texts2 == ["hello"]


def test_reference_manager_distinct_clips_encode_separately():
    settings = Settings(codec_decode_only=True)
    manager = ReferenceManager(settings, gpu_codec=None, gpu_device="cpu")

    encode_calls = []
    manager._encode = lambda audio_bytes, text: encode_calls.append(audio_bytes) or _FakeResolved(text)

    a = ReferenceAudio(audio=b"clip-a", text="a")
    b = ReferenceAudio(audio=b"clip-b", text="b")
    manager.resolve([a, b], None)

    assert encode_calls == [b"clip-a", b"clip-b"]


def test_reference_manager_no_reference_returns_empty():
    settings = Settings(codec_decode_only=True)
    manager = ReferenceManager(settings, gpu_codec=None, gpu_device="cpu")
    tokens, texts = manager.resolve([], None)
    assert tokens == []
    assert texts == []


def test_reference_manager_unknown_reference_id_raises_without_fish_speech():
    """The existence check happens before importing fish_speech, so this
    error path works even though fish_speech is not installed here."""
    settings = Settings(codec_decode_only=True)
    manager = ReferenceManager(settings, gpu_codec=None, gpu_device="cpu")
    with pytest.raises(ValueError, match="unknown reference_id"):
        manager.resolve([], "definitely-not-a-real-id")


# --------------------------------------------------------------------------
# factory.py: validation errors
# --------------------------------------------------------------------------


def test_ensure_vendor_on_path_missing_dir(tmp_path):
    settings = Settings(vendor_dir=tmp_path / "does-not-exist")
    with pytest.raises(EngineConfigError, match="vendor_dir"):
        _ensure_vendor_on_path(settings)


def test_ensure_vendor_on_path_missing_fish_speech_package(tmp_path):
    vendor_dir = tmp_path / "vendor"
    vendor_dir.mkdir()
    settings = Settings(vendor_dir=vendor_dir)
    with pytest.raises(EngineConfigError, match="fish_speech"):
        _ensure_vendor_on_path(settings)


def test_ensure_vendor_on_path_success_adds_to_sys_path(tmp_path):
    vendor_dir = tmp_path / "vendor"
    (vendor_dir / "fish_speech").mkdir(parents=True)
    settings = Settings(vendor_dir=vendor_dir)

    before = list(sys.path)
    try:
        _ensure_vendor_on_path(settings)
        assert str(vendor_dir) in sys.path
    finally:
        sys.path[:] = before


def test_validate_checkpoint_missing_dir(tmp_path):
    settings = Settings(checkpoint_dir=tmp_path / "nope")
    with pytest.raises(EngineConfigError, match="checkpoint_dir"):
        _validate_checkpoint(settings)


def test_validate_checkpoint_missing_config_json(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    settings = Settings(checkpoint_dir=ckpt)
    with pytest.raises(EngineConfigError, match="config.json"):
        _validate_checkpoint(settings)


def test_validate_checkpoint_missing_weights(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    settings = Settings(checkpoint_dir=ckpt)
    with pytest.raises(EngineConfigError, match="model.pth"):
        _validate_checkpoint(settings)


def test_validate_checkpoint_missing_codec(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.pth").write_bytes(b"")
    settings = Settings(checkpoint_dir=ckpt)
    with pytest.raises(EngineConfigError, match="codec.pth"):
        _validate_checkpoint(settings)


def test_validate_checkpoint_complete_passes(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    (ckpt / "model.pth").write_bytes(b"")
    (ckpt / "codec.pth").write_bytes(b"")
    settings = Settings(checkpoint_dir=ckpt)
    _validate_checkpoint(settings)  # must not raise


def test_validate_precision_refuses_bf16_on_sm75():
    settings = Settings(device="cuda", half=False)
    with pytest.raises(EngineConfigError, match="sm_75"):
        _validate_precision(settings, capability=(7, 5))


def test_validate_precision_allows_bf16_on_ampere():
    settings = Settings(device="cuda", half=False)
    _validate_precision(settings, capability=(8, 0))  # must not raise


def test_validate_precision_skips_check_when_half_requested():
    settings = Settings(device="cuda", half=True)
    # capability=(7, 5) would fail the check if it were even consulted
    _validate_precision(settings, capability=(7, 5))


def test_validate_precision_skips_check_off_cuda():
    settings = Settings(device="cpu", half=False)
    _validate_precision(settings, capability=(7, 5))


# --------------------------------------------------------------------------
# GPU-dependent behaviour lives in tests/test_integration_gpu.py, which runs
# against a live server with the real checkpoint loaded rather than faking it.
# --------------------------------------------------------------------------
