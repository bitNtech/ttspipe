"""Frame-level streaming generation with a cached system prefix.

`generate_long` is built to return a finished utterance: it yields once per
text batch, after every token of that batch has been sampled. For a voice
agent that is the wrong granularity twice over.

**Time to first audio is the whole batch.** Nothing reaches the client until
the last frame of the batch is sampled, so a 5-second reply costs 5 seconds of
generation before a single sample plays. `generate()` here yields code frames
as they are sampled, on a ramped schedule (`emit_schedule`) that emits the
first handful almost immediately and then grows the block size to amortise the
codec's fixed per-call cost once audio is already playing.

**The voice prompt is re-prefilled every request.** The system message holds
the reference clip's VQ codes, and a 10-second reference is ~220 tokens of
prompt. Measured on a GTX 1650, prefilling 245 tokens costs 5.7 s - against
475 ms for the 17 tokens of an actual user turn - and `app/engine/references.py`
already notes that those tokens are "replayed on *every* subsequent request
that reuses the voice". They do not have to be. The system message is byte for
byte identical across requests using the same voice, attention is causal, and
`KVCache` is a plain position-indexed buffer, so the prefix's entries stay
valid until something overwrites them - and nothing does, because every
request writes only from position S upwards. Prefilling it once and starting
each later request at S reproduces the full prefill's logits exactly (measured:
identical argmax, max|delta| 0.0078, which is fp16 rounding).

Rather than special-casing the system message, the cache holds the exact token
matrix currently materialised in the KV buffer and prefills only what a new
prompt does not already share with it - the longest common prefix, compared
across every codebook row, not just the semantic one. The system message falls
out of that as the common case (it is identical, so it always matches), and
being keyed on tokens rather than on a voice id means any change to the
reference clip, its transcript, or the tokenizer's output invalidates it by
construction rather than by someone remembering to.

Everything here runs on `GpuWorker`'s single thread, which is what makes a
process-wide KV cache safe to treat as this object's own.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from app.engine.base import CancelToken
from app.engine.fastpath import kv_window, prompt_embedding_batch, prompt_embeddings

if TYPE_CHECKING:
    import torch

    from app.protocol import TTSParams

logger = logging.getLogger(__name__)

RAS_WIN_SIZE = 10


class PromptTooLong(ValueError):
    """The prompt plus its generation budget does not fit in the context."""


@dataclass(slots=True)
class PrefixStats:
    hits: int = 0
    misses: int = 0
    cached_tokens: int = 0
    last_reused_tokens: int = 0
    last_prefilled_tokens: int = 0
    last_prefill_ms: float = 0.0
    #: Batches that ran out of token budget instead of ending on their own.
    #: Anything above zero means someone heard a sentence stop mid-word.
    truncations: int = 0

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "cached_tokens": self.cached_tokens,
            "last_reused_tokens": self.last_reused_tokens,
            "last_prefilled_tokens": self.last_prefilled_tokens,
            "last_prefill_ms": round(self.last_prefill_ms, 1),
            "truncations": self.truncations,
        }


#: How fast the block size grows on the way to `stream_max_frames`.
#:
#: With generation running at about realtime, a block reaches the client only
#: once every frame of it has been generated, so the buffer a player needs is
#: about one block; growing blocks (the old 1, 2, ... 32 ramp for a server at
#: 0.4x realtime) made each one arrive a whole previous block late. The ramp
#: now tops out at a few frames (`stream_max_frames`), and doubling on the way
#: there only matters for the first three blocks.
EMIT_GROWTH = 2


def default_emit_schedule(first: int, maximum: int) -> list[int]:
    """Block sizes to emit, smallest first, growing up to `maximum`.

    The first block sets time-to-first-audio, so it is one frame. Later
    blocks grow to `maximum` because each codec call has a fixed cost
    (~5 ms of the streaming codec's GPU time), and no further (see
    `EMIT_GROWTH`).
    """
    first = max(1, int(first))
    maximum = max(first, int(maximum))
    schedule = []
    size = first
    while size < maximum:
        schedule.append(size)
        size *= EMIT_GROWTH
    schedule.append(maximum)
    return schedule


class StreamingGenerator:
    """Owns the prompt construction, the KV prefix cache, and the decode loop."""

    def __init__(self, model, decode_one_token: Callable, device, settings, graphs=None) -> None:
        self._model = model
        #: `app/engine/graphs.DecodeGraphs`, or None to decode eagerly.
        self.graphs = graphs
        self._decode_one_token = decode_one_token
        self._device = device
        self._settings = settings

        self._cached: "torch.Tensor | None" = None  # (codebook_dim, N) on CPU
        self.enabled = bool(getattr(settings, "prefix_cache", True))
        self.stats = PrefixStats()

        self._logit_bias: "torch.Tensor | None" = None
        self._im_end_id: int | None = None

    # --- prompt construction (mirrors generate_long's, deliberately) ---

    def _system_parts(self, prompt_texts: Sequence[str], prompt_tokens: Sequence["torch.Tensor"]):
        import re

        import torch
        from fish_speech.content_sequence import TextPart, VQPart

        if not (prompt_texts and prompt_tokens):
            return [TextPart(text="convert the provided text to speech", cal_loss=False)]

        tagged = []
        for i, t in enumerate(prompt_texts):
            tagged.append(t if re.search(r"<\|speaker:\d+\|>", t) else f"<|speaker:{i}|>{t}")

        return [
            TextPart(
                text="convert the provided text to speech reference to the following:\n\nText:\n",
                cal_loss=False,
            ),
            TextPart(text="\n".join(tagged), cal_loss=False),
            TextPart(text="\n\nSpeech:\n", cal_loss=False),
            VQPart(codes=torch.cat([c.cpu() for c in prompt_tokens], dim=1), cal_loss=False),
        ]

    def _system_message(self, prompt_texts, prompt_tokens):
        from fish_speech.conversation import Message

        return Message(
            role="system",
            parts=self._system_parts(prompt_texts, prompt_tokens),
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )

    def _encode(self, messages):
        from fish_speech.conversation import Conversation

        conversation = Conversation()
        for message in messages:
            conversation.append(message)
        return conversation.encode_for_inference(
            self._model.tokenizer, num_codebooks=self._model.config.num_codebooks
        )

    # --- prefix cache ---

    def invalidate_prefix(self) -> None:
        """Forget what the KV buffer holds. The next request prefills in full."""
        self._cached = None
        self.stats.cached_tokens = 0

    def _shared_prefix(self, encoded: "torch.Tensor") -> int:
        """How many leading positions of `encoded` the KV cache already holds.

        Compares every codebook row: two prompts agreeing on the semantic token
        but not on the VQ codes beneath it produce different KV entries, so row
        0 alone is not a safe key.
        """
        import torch

        if not self.enabled or self._cached is None:
            return 0
        n = min(self._cached.size(1), encoded.size(1))
        if n == 0:
            return 0
        equal = (self._cached[:, :n] == encoded[:, :n]).all(dim=0)
        mismatches = (~equal).nonzero()
        if mismatches.numel() == 0:
            return int(n)
        return int(mismatches[0].item())

    def _sequential_prefill(self, span: int) -> bool:
        """Whether to walk this many prompt positions one at a time.

        Normally a batched prefill wins by a mile - it is one GEMM instead of
        N GEMVs - and for a long span it still does. For a short one it does
        not, because of how bitsandbytes NF4 behaves: only sequence length 1
        reaches its fused `gemv_4bit` kernel, and anything longer dequantises
        every weight into a dense GEMM first. Measured on a GTX 1650 that
        dequantisation is a flat ~477 ms however short the span - prefilling 8
        prompt tokens and 19 prompt tokens both cost ~575 ms - while walking
        positions one at a time costs ~48 ms each and skips it. So the crossover
        is around 11 tokens, which is squarely in the range a cached prefix
        leaves behind, and the 239-token prefix of a fresh voice is well past it.

        Either way the result is the same: attention is causal and the KV cache
        is indexed by position, so a position cannot tell whether its
        predecessors arrived alongside it or before it.
        """
        mode = getattr(self._settings, "prefill_mode", "auto")
        if mode == "sequential":
            return True
        if mode == "batched":
            return False
        from app.engine import nf4gemv

        # `app/engine/nf4gemv.py` serves several positions in one pass over the
        # weights, so a batch of 8 costs what one position does and walking
        # never wins.
        if nf4gemv.available():
            return False
        return span <= getattr(self._settings, "prefill_sequential_max", 32)

    def _truncate_cache(self, length: int) -> None:
        """Forget everything the cache claims from `length` onwards.

        Called before writing KV entries there. Without it, a prefill that is
        cancelled - or that raises - part way through would leave the cache
        still vouching for a prompt whose entries have just been overwritten,
        and the next request would reuse positions holding someone else's keys.
        Positions below `length` are untouched and stay valid.
        """
        if self._cached is None:
            return
        self._cached = self._cached[:, :length] if length > 0 else None
        self.stats.cached_tokens = 0 if self._cached is None else self._cached.size(1)

    def _gather_prompt_rows(self, encoded: "torch.Tensor", start: int, total: int):
        """Every embedding a sequential prefill will need, gathered once.

        Falls back to None - meaning "do it the per-step way" - whenever the
        model is not holding its table on the host, since then there is no round
        trip to save.
        """
        import torch

        model = self._model
        if not getattr(model, "embeddings_offloaded", False):
            return None
        ids = encoded[0, start:total].to("cpu")
        with torch.inference_mode():
            rows = model.embeddings(ids)
        return rows.to(self._device, non_blocking=True)

    def _prefill(
        self,
        encoded: "torch.Tensor",
        start: int,
        cancel: CancelToken,
        sequential: bool | None = None,
    ) -> None:
        """Populate KV entries for positions [start, len) of this prompt.

        `sequential` overrides the per-span choice; hints force it so they can
        never take the batched path's flat cost.
        """
        import time

        import torch

        total = encoded.size(1)
        if start >= total:
            return
        cancel.raise_if_cancelled()
        self._truncate_cache(start)

        codebook_dim = self._model.config.num_codebooks + 1
        prompt = encoded[:, start:].to(device=self._device)
        t0 = time.perf_counter()
        walk = self._sequential_prefill(total - start) if sequential is None else sequential
        if walk:
            # Position by position, recording progress as it goes. A cancelled
            # hint has still really written the positions it got through, and
            # throwing that away would make speculative prefill worth much less
            # than it is - the next request would redo work already on the GPU.
            done = start
            rows = self._gather_prompt_rows(encoded, start, total)
            # `prompt_embedding_batch` serves rows we already gathered;
            # `prompt_embeddings` falls back to one host round trip per step.
            embedding_source = (
                prompt_embeddings() if rows is None else prompt_embedding_batch(rows)
            )
            try:
                with torch.inference_mode(), embedding_source:
                    for offset in range(total - start):
                        cancel.raise_if_cancelled()
                        position = start + offset
                        input_pos = torch.tensor(
                            [position], device=self._device, dtype=torch.long
                        )
                        with kv_window(position + 1):
                            self._model.forward_generate(
                                prompt[:, offset : offset + 1].view(1, codebook_dim, -1),
                                input_pos,
                            )
                        done = position + 1
            finally:
                self._remember(encoded[:, :done])
                del rows
        elif self.graphs is not None and self.graphs.prefill(encoded[:, start:], start):
            self._remember(encoded)
        else:
            # One call, so there is no partial state to keep: it either lands
            # whole or the cache stays truncated at `start`.
            with torch.inference_mode(), prompt_embeddings():
                input_pos = torch.arange(start, total, device=self._device, dtype=torch.long)
                with kv_window(total):
                    self._model.forward_generate(
                        prompt.view(1, codebook_dim, -1), input_pos
                    )
            self._remember(encoded)
        self.stats.last_prefill_ms = (time.perf_counter() - t0) * 1000.0

    def _remember(self, encoded: "torch.Tensor") -> None:
        self._cached = encoded.detach().cpu().clone() if self.enabled else None
        self.stats.cached_tokens = 0 if self._cached is None else self._cached.size(1)

    def hint(self, text: str, prompt_texts, prompt_tokens, cancel: CancelToken) -> int:
        """Prefill the prompt for a segment that has not been cut yet.

        In a voice agent the text arrives from an upstream LLM a few tokens at
        a time, and `app/segmenter.py` holds it back until it is a speakable
        segment. The GPU is idle for that whole stretch, and then has to prefill
        the prompt the moment the segment is cut - work that could have happened
        while the LLM was still writing.

        The prompt built here is the *complete* shape a real request would use -
        closing `im_end`, assistant turn and all - not just the text so far. An
        earlier version stopped after the text, which meant the trailing
        `<|im_end|><|im_start|>assistant\n<|voice|>` was always left to pay for:
        about 8 tokens, and on this card ~430 ms of the ~450 ms that remained.
        Guessing the whole shape is what makes a full hit possible, and the
        caller guesses the sentence's final punctuation for the same reason (see
        `Session._maybe_hint`).

        Guessing is safe because nothing here is trusted later. Re-tokenising a
        longer string can legitimately disagree with what a shorter one produced
        ("a"+"bc" need not tokenise like "abc"), and the sentence may simply not
        end where we guessed - so the real request compares tokens position by
        position and keeps only the prefix that actually matches. A wrong guess
        costs idle GPU time, never a wrong voice, and even a wrong guess still
        lands every token up to the point where it diverged.
        """
        from fish_speech.content_sequence import TextPart
        from fish_speech.conversation import Message

        if not text.strip():
            return 0
        with_suffix = bool(getattr(self._settings, "hint_include_suffix", True))
        messages = [
            self._system_message(prompt_texts, prompt_tokens),
            Message(
                role="user",
                parts=[TextPart(text=text, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=with_suffix,
            ),
        ]
        if with_suffix:
            messages.append(self._assistant_opener())
        encoded, masks, parts = self._encode(messages)
        if masks is not None or parts is not None:
            return 0

        shared = self._shared_prefix(encoded)
        # The last position is deliberately left alone: a real request has to
        # run it through the model anyway to sample its first frame, and
        # prefilling it here would only have to be redone.
        end = encoded.size(1) - 1
        if shared >= end:
            return 0  # already prefilled this exact prompt

        # Bounded, but free to batch. Both halves were arrived at by
        # measurement.
        #
        # Batching matters most exactly where there is most to do: one Tamil
        # word is ~11 tokens and a hint gets ~125ms between an LLM's words, so
        # walking positions at ~43ms each leaves the hint permanently behind -
        # 40-42 tokens outstanding at cut time against 17-24 when it may batch.
        #
        # What made batching dangerous was never the batch; it was that a
        # batched prefill is a single forward call, so it cannot be abandoned,
        # and hints used to accumulate one per word until the real segment was
        # queued behind all of them (measured: 7.5-10.8s to first audio). That
        # is fixed where it belongs, in `GpuWorker.submit_hint`, which holds one
        # slot rather than a queue - so at most one uncancellable call can ever
        # be in front of a segment.
        end = min(end, shared + max(1, self._settings.hint_max_tokens))
        self._prefill(encoded[:, :end], shared, cancel)
        return end - shared

    def warm(self, prompt_texts, prompt_tokens, cancel: CancelToken) -> int:
        """Prefill just the system message for this voice and cache it.

        Called when the active voice changes, so the ~220 prompt tokens of a
        reference clip are paid once, off the critical path, instead of landing
        on whichever user turn happens to be first.
        """
        encoded, masks, parts = self._encode([self._system_message(prompt_texts, prompt_tokens)])
        if masks is not None or parts is not None:
            return 0
        shared = self._shared_prefix(encoded)
        if shared >= encoded.size(1):
            return encoded.size(1)
        self._prefill(encoded, shared, cancel)
        logger.info(
            "warmed voice prefix: %d tokens in %.0f ms", encoded.size(1), self.stats.last_prefill_ms
        )
        return encoded.size(1)

    # --- sampling scaffolding ---

    def _ensure_bias(self):
        import torch
        from fish_speech.tokenizer import IM_END_TOKEN

        if self._logit_bias is not None:
            return self._logit_bias, self._im_end_id

        config = self._model.config
        dtype = next(self._model.parameters()).dtype
        im_end = self._model.tokenizer.get_token_id(IM_END_TOKEN)
        bias = torch.full((1, 1, config.vocab_size), float("-inf"), device=self._device, dtype=dtype)
        bias[0, 0, config.semantic_begin_id : config.semantic_end_id + 1] = 0.0
        bias[0, 0, im_end] = 0.0
        self._logit_bias, self._im_end_id = bias, im_end
        return bias, im_end

    # --- the streaming loop ---

    # --- batching within one request ---

    def _batches(self, text: str, chunk_length: int) -> list[str]:
        """Split one segment the way the model was trained to receive it.

        `chunk_length` is a budget in **bytes**, not characters, and that
        distinction is the whole reason this exists. `app/segmenter.py` bounds a
        segment by characters (`max_segment_chars`, 220), which for Latin text
        lands comfortably under the 200-byte default - but Tamil runs about 2.76
        bytes per character, so the same 220-character segment is ~600 bytes,
        three times the budget, in a single turn the model never saw in
        training. Handing that over in one piece is what makes a voice come
        apart on non-Latin scripts.

        The fork's own splitters are reused rather than reimplemented so the
        boundaries land exactly where `generate_long` would have put them.
        """
        from fish_speech.models.text2semantic.inference import (
            group_turns_into_batches,
            split_plain_text_into_batches,
            split_text_by_speaker,
        )

        turns = split_text_by_speaker(text)
        if turns:
            return group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)
        if chunk_length > 0:
            return split_plain_text_into_batches(text, max_bytes=chunk_length)
        return [text]

    @staticmethod
    def _user_message(text: str):
        from fish_speech.content_sequence import TextPart
        from fish_speech.conversation import Message

        return Message(
            role="user",
            parts=[TextPart(text=text, cal_loss=False)],
            cal_loss=False,
            add_im_start=True,
            add_im_end=True,
        )

    @staticmethod
    def _assistant_opener():
        from fish_speech.conversation import Message

        return Message(
            role="assistant",
            parts=[],
            cal_loss=False,
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        )

    @staticmethod
    def _assistant_message(codes: "torch.Tensor"):
        """A finished batch, fed back so the next one can hear it.

        `generate_long` does this deliberately - "each one is generated with the
        previously generated codes in context, so prosody carries across the
        boundary" (fish_speech/models/text2semantic/inference.py). Dropping it
        makes consecutive batches of the same sentence sound like separate
        takes spliced together.
        """
        from fish_speech.content_sequence import VQPart
        from fish_speech.conversation import Message

        return Message(
            role="assistant",
            parts=[VQPart(codes=codes.cpu(), cal_loss=False)],
            cal_loss=False,
            modality="voice",
            add_im_start=True,
            add_im_end=True,
        )

    # --- the streaming loop ---

    def generate(
        self,
        text: str,
        params: "TTSParams",
        prompt_texts: Sequence[str],
        prompt_tokens: Sequence["torch.Tensor"],
        cancel: CancelToken,
        emit_schedule: Sequence[int],
    ) -> Iterator["torch.Tensor"]:
        """Yield blocks of code frames, shape (num_codebooks, n), as sampled."""
        import torch

        cancel.raise_if_cancelled()

        history = [self._system_message(prompt_texts, prompt_tokens)]
        # The ramp is per *request*, not per batch: once audio is playing, a
        # later batch should keep emitting big blocks rather than drop back to
        # single frames it no longer needs.
        state = {"blocks": 0}

        for batch_text in self._batches(text, params.chunk_length):
            history.append(self._user_message(batch_text))
            collected: list["torch.Tensor"] = []
            yield from self._run_batch(history, params, cancel, emit_schedule, state, collected)
            if collected:
                history.append(self._assistant_message(torch.cat(collected, dim=-1)))
            else:
                # Nothing came back for this batch; leaving a user turn with no
                # reply in the context would misdescribe the conversation to the
                # next one.
                history.pop()

    def _first_token_eager(
        self, encoded, prompt, shared, total_len, params, cancel, bias, audio_masks, audio_parts
    ):
        import torch
        from fish_speech.models.text2semantic.inference import decode_one_token_ar
        from torch.nn.attention import SDPBackend, sdpa_kernel

        model = self._model
        codebook_dim = model.config.num_codebooks + 1
        dtype = next(model.parameters()).dtype
        cacheable = audio_masks is None and audio_parts is None
        # Everything up to the last prompt position only needs its KV entries
        # written; just the last one has to produce logits, and it is the frame
        # the client is waiting on. Splitting them lets the bulk go through
        # `_prefill` and keeps this sampling call at sequence length 1.
        last = total_len - 1
        if cacheable and last > shared:
            self._prefill(encoded[:, :last], shared, cancel)
            head = last
        else:
            head = shared  # audio-part prompts must go through in one piece
            self._truncate_cache(shared)

        input_pos = torch.arange(head, total_len, device=self._device, dtype=torch.long)
        with torch.inference_mode(), kv_window(total_len), prompt_embeddings(), sdpa_kernel(
            SDPBackend.MATH
        ):
            return decode_one_token_ar(
                model,
                prompt[:, head:].view(1, codebook_dim, -1),
                input_pos,
                torch.tensor(params.temperature, device=self._device, dtype=dtype),
                torch.tensor(params.top_p, device=self._device, dtype=dtype),
                params.top_k,
                bias,
                audio_masks,
                audio_parts,
            )

    def _decode_graphed(
        self, graphs, token, total_len, max_new_tokens, cancel, emit_schedule, state,
        collected, im_end_id,
    ):
        """The decode loop as graph replays, kept one step ahead of the CPU.

        Deciding whether a turn has ended needs the sampled token on the host,
        which is a synchronisation per frame; waiting on step k with step k+1
        already queued means the GPU never idles while Python catches up. If
        step k turns out to be the end, step k+1 was wasted work at a position
        past the prompt, which the next request overwrites before reading.
        """
        import torch

        # See the eager loop in `_run_batch`: a turn can end on its first
        # token, and decoding the codebooks sampled alongside im_end is a click.
        if int(token[0, 0]) == im_end_id:
            return
        steps = max_new_tokens - 1
        codes = torch.empty((token.shape[0], steps + 1), dtype=token.dtype, device=token.device)
        codes[:, :1].copy_(token)
        main = torch.empty(steps + 1, dtype=token.dtype, pin_memory=True)
        done = [torch.cuda.Event(), torch.cuda.Event()]

        def launch(i: int) -> None:
            out = graphs.decode(total_len + i - 1)
            codes[:, i : i + 1].copy_(out)
            main[i : i + 1].copy_(out[0], non_blocking=True)
            done[i % 2].record()

        kept, emitted = 1, 0
        # The first block goes out before step 1 is queued: its codec work
        # would otherwise wait behind a whole frame of generation, and it is
        # the block the listener is waiting on.
        if kept >= emit_schedule[min(state["blocks"], len(emit_schedule) - 1)]:
            yield codes[1:, :kept]
            emitted = kept
            state["blocks"] += 1
        if steps > 0:
            launch(1)
        ended_naturally = False
        for i in range(1, steps + 1):
            if i < steps:
                launch(i + 1)
            done[i % 2].synchronize()
            cancel.raise_if_cancelled()
            if int(main[i]) == im_end_id:
                ended_naturally = True
                break
            kept = i + 1
            block = emit_schedule[min(state["blocks"], len(emit_schedule) - 1)]
            if kept - emitted >= block:
                yield codes[1:, emitted:kept]
                emitted = kept
                state["blocks"] += 1

        if kept > emitted:
            yield codes[1:, emitted:kept]
            state["blocks"] += 1
        collected.append(codes[1:, :kept].clone())
        if not ended_naturally:
            self._note_truncation(kept, total_len, max_new_tokens)

    def _note_truncation(self, frames: int, total_len: int, max_new_tokens: int) -> None:
        # Running out of budget looks exactly like a finished sentence from the
        # outside: same frames, same clean finish, just stopping mid-word. Say
        # so, or a truncated voice is indistinguishable from a short one.
        self.stats.truncations += 1
        logger.warning(
            "segment hit its token budget after %d frames (%.1fs of audio) without "
            "the model ending the turn - speech is cut off. prompt=%d tokens, "
            "budget=%d, context=%d. Raise TTS_MAX_SEQ_LEN or max_new_tokens, or "
            "shorten the reference clip.",
            frames,
            frames * 2048 / 44100.0,
            total_len,
            max_new_tokens,
            self._model.config.max_seq_len,
        )

    def _run_batch(
        self,
        history: list,
        params: "TTSParams",
        cancel: CancelToken,
        emit_schedule: Sequence[int],
        state: dict,
        collected: list,
    ) -> Iterator["torch.Tensor"]:
        import time

        import torch
        from torch.nn.attention import SDPBackend, sdpa_kernel

        model = self._model
        config = model.config
        codebook_dim = config.num_codebooks + 1
        bias, im_end_id = self._ensure_bias()
        dtype = next(model.parameters()).dtype

        encoded, audio_masks, audio_parts = self._encode(history + [self._assistant_opener()])
        total_len = encoded.size(1)

        # An audio-part prompt is embedded through audio_projector rather than
        # from token ids, so its KV entries are not reproducible from `encoded`
        # alone and the cache cannot vouch for them.
        cacheable = audio_masks is None and audio_parts is None
        shared = self._shared_prefix(encoded) if cacheable else 0
        if shared:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        # The last prompt position has to run through the model here anyway, to
        # sample the first frame, so never let the prefix swallow it.
        shared = min(shared, total_len - 1)
        self.stats.last_reused_tokens = shared
        self.stats.last_prefilled_tokens = total_len - shared

        max_length = config.max_seq_len
        max_new_tokens = params.max_new_tokens or (max_length - total_len)
        max_new_tokens = min(max_new_tokens, max_length - total_len)
        if max_new_tokens <= 0:
            raise PromptTooLong(
                f"prompt is {total_len} tokens and the context is {max_length}; "
                "shorten the reference clip or the text"
            )

        prompt = encoded.to(device=self._device)
        t0 = time.perf_counter()
        graphs = self.graphs if cacheable else None
        if graphs is not None and graphs.max_top_k is not None and params.top_k > graphs.max_top_k:
            graphs = None

        if graphs is not None:
            # Everything not yet cached - the last position included - in one
            # forward: the 4-bit kernel reads each weight once for up to 32
            # positions, so the prompt costs about what a single frame does.
            # Its last position's logits then go straight into the captured
            # first-frame sampler.
            self._truncate_cache(shared)
            graphs.begin(params, total_len)
            if graphs.prefill(encoded[:, shared:], shared):
                token = graphs.first()
            else:
                input_pos = torch.arange(shared, total_len, device=self._device, dtype=torch.long)
                with torch.inference_mode(), kv_window(total_len), prompt_embeddings():
                    result = model.forward_generate(
                        prompt[:, shared:].view(1, codebook_dim, -1), input_pos
                    )
                token = graphs.first(result.logits, result.hidden_states)
                del result
        else:
            token = self._first_token_eager(
                encoded, prompt, shared, total_len, params, cancel, bias, audio_masks, audio_parts
            )
        self.stats.last_prefill_ms = (time.perf_counter() - t0) * 1000.0
        # Only now is the whole prompt materialised, so only now may the cache
        # claim it. The frames generated from here on are not part of any future
        # prompt, so the cache stops at `total_len` and generation is free to
        # write past it.
        if cacheable:
            self._remember(encoded)
        else:
            self.invalidate_prefix()

        if graphs is not None:
            yield from self._decode_graphed(
                graphs, token, total_len, max_new_tokens, cancel, emit_schedule, state,
                collected, im_end_id,
            )
            return

        temperature = torch.tensor(params.temperature, device=self._device, dtype=dtype)
        top_p = torch.tensor(params.top_p, device=self._device, dtype=dtype)
        previous_tokens = torch.zeros(
            (codebook_dim, RAS_WIN_SIZE), dtype=torch.int, device=self._device
        )

        pending: list["torch.Tensor"] = []
        cur_token = token.view(1, codebook_dim, -1)

        # The model can end the turn on its very first token - a batch that is
        # only punctuation, or whatever `flush()` drained. The loop below checks
        # for that before keeping a frame, but this first one arrives outside
        # it, and `decode_one_token_ar` samples all ten fine codebooks off a
        # clamped, meaningless index when the main token is `im_end`. Decoding
        # that produces ~46 ms of noise: a click where a tiny segment should be.
        if cur_token[0, 0, -1] == im_end_id:
            return

        first = cur_token.view(codebook_dim, -1)[1:].clone()
        pending.append(first)
        collected.append(first)

        position = total_len
        input_pos = torch.tensor([position], device=self._device, dtype=torch.int)

        ended_naturally = False
        for _ in range(max_new_tokens - 1):
            cancel.raise_if_cancelled()

            block = emit_schedule[min(state["blocks"], len(emit_schedule) - 1)]
            if len(pending) >= block:
                yield torch.cat(pending, dim=-1)
                pending = []
                state["blocks"] += 1

            with torch.inference_mode(), kv_window(position + 1), sdpa_kernel(SDPBackend.MATH):
                next_token = self._decode_one_token(
                    model=model,
                    x=cur_token,
                    input_pos=input_pos,
                    previous_tokens=previous_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=params.top_k,
                    semantic_logit_bias=bias,
                    audio_masks=None,
                    audio_parts=None,
                ).clone()

            input_pos += 1
            position += 1
            cur_token = next_token.view(1, codebook_dim, -1)
            previous_tokens = previous_tokens.roll(-1, dims=1)
            previous_tokens[:, -1] = next_token.view(codebook_dim, -1)[:, 0]

            if cur_token[0, 0, -1] == im_end_id:
                ended_naturally = True
                break
            frame = cur_token.view(codebook_dim, -1)[1:].clone()
            pending.append(frame)
            collected.append(frame)

        # Whatever did not fill a whole block still has to reach the client -
        # dropping it would clip the end of every batch.
        if pending:
            yield torch.cat(pending, dim=-1)
            state["blocks"] += 1

        if not ended_naturally:
            self._note_truncation(len(collected), total_len, max_new_tokens)
