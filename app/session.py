"""Per-connection state machine that drives one conversation end to end.

One `Session` owns everything downstream of "a decoded client event arrived":
feeding the segmenter, turning ready segments into `SynthesisRequest`s, running
them through the engine one at a time, encoding the resulting audio, and
queuing the encoded wire frames for the WebSocket layer to send. It never
touches the socket itself -- `app.ws.router` owns that -- which is what lets
the receive loop stay unblocked while a segment is mid-synthesis.

Concurrency model: `handle_event()` (called from the receive task) only ever
does synchronous, non-blocking work -- it decodes text through the segmenter
and pushes segment strings onto an internal queue. A single background
worker task drains that queue and does the actual (potentially slow)
`engine.synthesize()` calls. That split is what makes barge-in responsive:
`cancel` flips a `CancelToken` and drops queued segments synchronously, from
the event-handling coroutine, without waiting for whatever the worker is
doing right now.

Event-order contract: `start` must come first; `text`/`flush`/`stop`/`cancel`
before a `start` (or after a `stop` has finished the turn) is a
`ProtocolError`, same family raised by `decode_client_frame`, so the router
can handle both with one code path. `ping` is legal at any time.

`cancel` vs `stop`: both end the current audio stream with a `finish` frame,
but only `stop` ends the *turn* -- the session returns to "awaiting start".
`cancel` sends `finish(reason="cancel")` as a status update and leaves the
session `active`, so the client can keep sending `text` under the same
voice/seed without a new `start` (barge-in: interrupt, then keep talking).

Speculative prefill: while the segmenter is holding text back waiting for a
sentence to finish, the GPU has nothing to do and the prompt for that sentence
is already most of the way known. `_maybe_hint` spends that gap prefilling it
(`TTSEngine.prefill_hint`), so the segment that eventually gets cut has little
or no prompt left to pay for. It guesses twice over - that the sentence ends
here, and that it ends with a full stop - because a guess that lands turns the
next turn's prefill into nothing at all, and a guess that misses still leaves
every token before the divergence prefilled.

It is strictly an optimisation. At most one is in flight; it is cancelled the
instant a real segment is queued, since the worker runs one job at a time and a
hint ahead of real work would delay exactly what it exists to speed up; and the
engine re-checks the guess token by token before reusing any of it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
import time
from typing import Any

from app.audio import StreamEncoder, resolve_format
from app.config import Settings
from app.engine.base import (
    AudioChunk,
    CancelToken,
    EngineError,
    EngineNotReady,
    SynthesisRequest,
    TTSEngine,
)
from app.protocol import (
    CancelEvent,
    ClientEvent,
    FlushEvent,
    PingEvent,
    ProtocolError,
    ReferenceAudio,
    StartEvent,
    StopEvent,
    TextEvent,
    audio_frame,
    with_server_defaults,
    finish_frame,
    log_frame,
    pong_frame,
    started_frame,
)
from app.segmenter import SENTENCE_END, Segmenter
from app.voice import VoiceStore

logger = logging.getLogger("tts.session")

_STOP = object()  # sentinel: drain done, emit finish(stop), go idle
_SHUTDOWN = object()  # sentinel: tear the worker down now, no draining


def _draw_seed() -> int:
    return random.getrandbits(31)


def _guess_completed_segment(pending: str) -> str:
    """What the buffered text will most likely look like once it is a segment.

    The segmenter cuts at sentence-ending punctuation, so buffered text never
    *contains* a finished sentence - which means a speculative prefill of the
    raw buffer can never match the prompt that eventually gets built. Adding the
    punctuation the sentence is probably about to end with is what turns a
    partial hit into a total one, and the full stop is the right guess because
    that is how most sentences end.

    Nothing downstream trusts it: a miss diverges at the guessed character and
    everything before it still counts (see `StreamingGenerator.hint`).
    """
    text = pending.rstrip()
    if not text:
        return text
    if text[-1] in SENTENCE_END or text[-1] in ",;:-\u2014":
        return text
    return text + "."


class Session:
    """Drives one WebSocket connection's worth of TTS conversation."""

    def __init__(
        self,
        engine: TTSEngine,
        settings: Settings,
        session_id: str,
        voices: VoiceStore | None = None,
    ):
        self.engine = engine
        self.settings = settings
        self.session_id = session_id
        self.voices = voices
        self.outbound: asyncio.Queue[bytes] = asyncio.Queue()

        self._state = "idle"  # "idle" | "active"
        self._segmenter: Segmenter | None = None
        self._params: Any = None
        self._encoder: StreamEncoder | None = None
        self._header_sent = False
        self._elapsed_ms = 0.0
        self._cancel_token = CancelToken()
        self._segment_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._turn_done = asyncio.Event()
        self._turn_done.set()
        self._closed = False
        self._hint_task: asyncio.Task[None] | None = None
        self._hint_cancel: CancelToken | None = None
        self._hinted_text = ""
        self._last_hint_at = 0.0
        self._last_chunk_at = 0.0
        self._worker_task: asyncio.Task[None] = asyncio.ensure_future(self._worker_loop())

    @property
    def state(self) -> str:
        return self._state

    @property
    def closed(self) -> bool:
        return self._closed

    # -- event handling (called from the receive loop; never blocks) --

    async def handle_event(self, event: ClientEvent) -> None:
        if isinstance(event, PingEvent):
            await self.outbound.put(pong_frame())
            return

        if isinstance(event, StartEvent):
            if self._state == "stopping":
                # The previous turn is still draining. Waiting is friendlier
                # than rejecting a client that pipelines its turns.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._turn_done.wait(), timeout=30.0)
            if self._state != "idle":
                raise ProtocolError("received 'start' while a turn is already active")
            await self._handle_start(event)
            return

        if isinstance(event, CancelEvent):
            # Barge-in must work whenever anything is in flight, including
            # after `stop` -- "the client finished the sentence and the server
            # is still speaking it" is the single most common interruption.
            # Only an idle session has nothing to cancel, and a redundant
            # barge-in there is harmless, not a protocol violation.
            if self._state != "idle":
                self._handle_cancel()
            return

        if self._state != "active":
            raise ProtocolError(f"received {event.event!r} before 'start'")

        if isinstance(event, TextEvent):
            self._enqueue_segments(self._segmenter.push(event.text))
            self._maybe_hint()
            self.engine.wake()
        elif isinstance(event, FlushEvent):
            self._enqueue_segments(self._segmenter.flush())
        elif isinstance(event, StopEvent):
            self._cancel_hint()
            self._enqueue_segments(self._segmenter.close())
            self._segment_queue.put_nowait(_STOP)
            self._state = "stopping"
        else:  # pragma: no cover - decode_client_frame limits ClientEvent to the above
            raise ProtocolError(f"unhandled event type {type(event).__name__}")

    async def aclose(self) -> None:
        """Stop the worker and abandon any in-flight synthesis. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self._cancel_token.cancel()
        self._discard_encoder()
        self._turn_done.set()
        self._segment_queue.put_nowait(_SHUTDOWN)
        self._cancel_hint()
        if self._hint_task is not None:
            self._hint_task.cancel()
        self._worker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker_task

    # -- start / cancel: synchronous, immediate --

    def _resolve_voice(self, params: Any) -> tuple[Any, str | None]:
        """Apply the server's active voice, unless this frame named its own.

        Precedence is explicit-beats-ambient: a frame carrying `references` or
        a `reference_id` is answering the voice question itself and is left
        alone. Only a frame that says nothing inherits whatever
        `POST /v1/voice` last set, and `use_active_voice: false` opts out of
        even that without having to clear the server-wide voice for everyone.
        """
        if params.references:
            return params, "inline"
        if params.reference_id:
            return params, params.reference_id
        if not params.use_active_voice or self.voices is None:
            return params, None

        voice = self.voices.get()
        if voice is None:
            return params, None
        reference = ReferenceAudio(audio=voice.audio, text=voice.text)
        return params.model_copy(update={"references": [reference]}), voice.voice_id

    async def _handle_start(self, event: StartEvent) -> None:
        seed = event.request.seed if event.request.seed is not None else _draw_seed()
        request = with_server_defaults(
            event.request,
            chunk_length=self.settings.chunk_length,
            format=self.settings.default_format,
        )
        params, voice_id = self._resolve_voice(request.model_copy(update={"seed": seed}))
        self._params = params
        self._segmenter = Segmenter(self.settings.min_segment_chars, self.settings.max_segment_chars)
        self._elapsed_ms = 0.0
        self._header_sent = False
        self._cancel_token = CancelToken()
        self._hinted_text = ""
        self._last_hint_at = 0.0
        self._drain_segment_queue()
        self._discard_encoder()
        self._turn_done.clear()

        fmt, encoder, warning = resolve_format(params.format, self.engine.sample_rate)
        self._encoder = encoder
        if warning:
            await self.outbound.put(log_frame(warning))
        await self.outbound.put(
            started_frame(self.session_id, self.engine.sample_rate, fmt, seed, voice_id)
        )
        self._state = "active"
        # The client's LLM is about to start writing; spend that time getting
        # an idle GPU back up to full clocks (GpuWorker.wake).
        self.engine.wake()

    def _handle_cancel(self) -> None:
        # Order matters: stop the token *before* swapping it so the worker's
        # already-captured reference (for whatever it's mid-processing) is
        # the one that ends up cancelled, not the fresh one.
        was_stopping = self._state == "stopping"
        self._cancel_hint()
        self._cancel_token.cancel()
        self._cancel_token = CancelToken()
        self._drain_segment_queue()
        if self._segmenter is not None:
            self._segmenter.flush()  # discard whatever hadn't been cut yet
        self.outbound.put_nowait(finish_frame("cancel"))

        if was_stopping:
            # The turn was already draining, and draining the queue just threw
            # away the _STOP sentinel that would have ended it. Close the turn
            # here, or it would sit in "stopping" forever and stall the next
            # `start`.
            self._discard_encoder()
            self._end_turn()
        else:
            self._restart_stream()

    def _restart_stream(self) -> None:
        """Begin a fresh audio stream for what the client says next.

        A barge-in ends the audio the client was playing, and it has thrown
        that audio away. Carrying the same encoder and the same clock into the
        next thing said would hand it a container whose header it has already
        discarded - headerless bytes it believes are WAV, or an mp3 frame
        sequence continuing from nothing - and timestamps that still count the
        interrupted speech, which a client scheduling on `time` would place
        seconds in the future. Both sound like silence or static after every
        interruption, which is exactly when a voice agent can least afford it.
        """
        self._discard_encoder()
        self._header_sent = False
        self._elapsed_ms = 0.0
        fmt = getattr(self._params, "format", None)
        if fmt is None:
            return
        _fmt, encoder, _warning = resolve_format(fmt, self.engine.sample_rate)
        self._encoder = encoder

    def _maybe_hint(self) -> None:
        """Prefill the prompt for the half-finished segment, if it is free to.

        Skipped whenever a segment is already waiting: the worker queue is
        serial, so a hint enqueued ahead of real work would just delay it.

        A hint already in flight is *replaced*, not waited for. Waiting leaves
        it prefilling text that is already stale, and costs nothing to avoid:
        hints walk positions one at a time, so cancelling one takes effect
        within ~48ms and every position it wrote is kept, which means the
        replacement resumes rather than restarting.

        Serialising them instead was measured and does not help - see
        `StreamingGenerator.hint` for why a hint must stay interruptible.
        """
        if self._closed or self._params is None or self._segmenter is None:
            return
        if not self._segment_queue.empty():
            return

        pending = self._segmenter.pending.strip()
        # Below a few characters there is no prompt worth prefilling, and
        # re-hinting a guess we already made would only re-tokenise it.
        if len(pending) < self.settings.min_hint_chars:
            return
        guess = _guess_completed_segment(pending)
        if guess == self._hinted_text:
            return
        # No throttle here: `GpuWorker.submit_hint` keeps a single slot, so a
        # hint issued now replaces one that has not started rather than
        # queueing behind it. Rate-limiting instead of coalescing was tried and
        # is worse - it bounds the pile-up by starving the hint, which costs
        # token-dense languages most, and they are the ones with the most
        # prompt left to prefill.
        now = time.monotonic()
        if (now - self._last_hint_at) * 1000.0 < self.settings.hint_min_interval_ms:
            return
        self._last_hint_at = now

        self._hinted_text = guess
        self._hint_task = asyncio.ensure_future(self._run_hint(guess, self._params))

    async def _run_hint(self, text: str, params: Any) -> None:
        """Retire the previous hint before starting this one.

        Only one hint may be outstanding at a time, and the bound has to be
        enforced here rather than by declining to start new ones. Each hint
        occupies a worker-queue slot *and* an event-loop executor thread parked
        on that slot's result, so firing one per word - which is the point -
        piles up both, and the real segment ends up queued behind all of them.
        Measured on the first turn after a language switch, when every hint has
        real work to do: 7.2s to first audio against 0.29s once the prefix is
        shared.

        Cancelling first is what makes waiting cheap: hints walk positions one
        at a time, so the previous one returns within about one position, ~48ms,
        having kept everything it wrote.
        """
        previous = self._hint_task
        self._cancel_hint()
        if previous is not None and not previous.done():
            with contextlib.suppress(Exception):
                await previous

        if self._closed or self._params is None:
            return
        cancel = CancelToken()
        self._hint_cancel = cancel
        with contextlib.suppress(Exception):  # an optimisation must never fail a turn
            await self.engine.prefill_hint(text, params, cancel)

    def _cancel_hint(self) -> None:
        """Take the GPU back from a speculative prefill. Keeps what it wrote."""
        if self._hint_cancel is not None:
            self._hint_cancel.cancel()
            self._hint_cancel = None

    def _enqueue_segments(self, segments: list[str]) -> None:
        if segments:
            # Real work has arrived, so the gap the hint was filling is over.
            self._cancel_hint()
        for segment in segments:
            self._segment_queue.put_nowait(segment)

    def _drain_segment_queue(self) -> None:
        while True:
            try:
                self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    # -- worker: the only place that calls the engine --

    async def _worker_loop(self) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            while True:
                item = await self._segment_queue.get()
                if item is _SHUTDOWN:
                    return
                if item is _STOP:
                    await self._flush_encoder()
                    await self.outbound.put(finish_frame("stop"))
                    self._end_turn()
                    continue
                await self._run_segment(item, self._cancel_token)

    async def _run_segment(self, text: str, token: CancelToken) -> None:
        assert self._encoder is not None and self._params is not None
        request = SynthesisRequest(text=text, params=self._params, session_id=self.session_id)
        # Wall time from the end of the previous segment's audio to the start of
        # this one's is dead air in the middle of a sentence-to-sentence join,
        # and it is invisible in any per-segment number. Recording it here is
        # what makes the join measurable at all.
        t0 = time.perf_counter()
        idle_ms = (t0 - self._last_chunk_at) * 1000.0 if self._last_chunk_at else 0.0
        first_at: float | None = None
        try:
            async for chunk in self.engine.synthesize(request, token):
                if token.cancelled:
                    break
                if first_at is None:
                    first_at = time.perf_counter()
                await self._emit(chunk)
        except EngineNotReady as exc:
            # The backend is unusable, so nothing else in this turn can work
            # either. End it properly rather than failing one segment at a time.
            await self.outbound.put(log_frame(f"synthesis error: {exc}"))
            self._discard_encoder()
            await self.outbound.put(finish_frame("error", str(exc)))
            self._end_turn()
            self._drain_segment_queue()
        except EngineError as exc:
            # One segment failed; the rest of the reply has not. Ending the turn
            # here used to discard every remaining segment *and* return the
            # session to `idle`, so the next `text` frame was a protocol error
            # and the socket closed - one bad sentence took the whole answer and
            # the connection with it. Report it and carry on; a turn where every
            # segment fails still ends normally on `stop`, having said so each
            # time.
            logger.warning("session %s: skipping a failed segment: %s", self.session_id, exc)
            await self.outbound.put(log_frame(f"skipped a segment: {exc}"))
        finally:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "session %s segment: idle_before_ms=%.0f ttfa_ms=%s total_ms=%.0f chars=%d",
                    self.session_id,
                    idle_ms,
                    f"{(first_at - t0) * 1000.0:.0f}" if first_at else "-",
                    (time.perf_counter() - t0) * 1000.0,
                    len(text),
                )

    async def _flush_encoder(self) -> None:
        """Emit the encoder's trailing bytes. Only mp3/flac have any.

        libsndfile writes the last frames (and, for flac, the STREAMINFO
        fix-ups) at close(). Skipping this truncates the end of the audio,
        which is inaudible in a short clip and very audible in a long one.
        """
        if self._encoder is None:
            return
        try:
            tail = self._encoder.finalize()
        except Exception as exc:  # a broken encoder must not kill the turn
            await self.outbound.put(log_frame(f"encoder finalize failed: {exc}"))
            tail = b""
        finally:
            self._encoder = None
        if tail:
            await self.outbound.put(audio_frame(tail, self._elapsed_ms))

    def _discard_encoder(self) -> None:
        """Close the encoder without sending anything (aborted turn, teardown)."""
        if self._encoder is None:
            return
        with contextlib.suppress(Exception):
            self._encoder.finalize()
        self._encoder = None

    def _end_turn(self) -> None:
        self._state = "idle"
        self._encoder = None
        self._header_sent = False
        self._elapsed_ms = 0.0
        self._turn_done.set()

    async def _emit(self, chunk: AudioChunk) -> None:
        assert self._encoder is not None
        payload = b""
        if not self._header_sent:
            header = self._encoder.header()
            if header:
                payload += header
            self._header_sent = True
        payload += self._encoder.encode(chunk)
        self._elapsed_ms += chunk.duration_s * 1000.0
        self._last_chunk_at = time.perf_counter()
        if payload:
            await self.outbound.put(audio_frame(payload, self._elapsed_ms))
