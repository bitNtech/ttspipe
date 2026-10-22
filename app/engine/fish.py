"""`FishSpeechEngine`: the async seam between the WebSocket transport and
`app/engine/worker.py`'s single GPU thread.

Everything GPU-shaped - model load, token sampling, codec decode - happens
synchronously on that one background thread (see worker.py's docstring for
why one thread is correct on a 4 GB card, not a limitation). This module's
only job is to keep the asyncio event loop free while that thread works:
`load()` runs the blocking load in an executor so `/health` keeps responding
mid-load, and `synthesize()` bridges the worker's plain `queue.Queue` results
back into an async generator via `run_in_executor` on the blocking `get()` -
one of the two bridging patterns the task spec calls out explicitly, chosen
over a `call_soon_threadsafe` + `asyncio.Queue` pair because the worker
already hands back a queue per job and there is nothing to gain from a second
queue type here.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import time
from typing import AsyncIterator

from app.engine.base import (
    AudioChunk,
    CancelToken,
    EngineError,
    EngineNotReady,
    EngineStatus,
    SynthesisRequest,
    TTSEngine,
)
from app.engine.worker import (
    ChunkMessage,
    DoneMessage,
    ErrorMessage,
    GpuWorker,
    WarmedMessage,
)

logger = logging.getLogger(__name__)


class FishSpeechEngine(TTSEngine):
    def __init__(self, settings) -> None:
        self._settings = settings
        self._worker = GpuWorker(settings)

        self._load_lock = asyncio.Lock()
        self._loaded = False
        self._loading = False
        self._load_seconds: float | None = None
        self._last_error: str | None = None

        self._total_requests = 0
        self._total_cancelled = 0
        self._total_failed = 0
        self._total_audio_seconds = 0.0
        self._last_ttfa_ms: float | None = None
        self._active_jobs = 0

    # --- TTSEngine ---

    async def load(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            self._loading = True
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            try:
                self._worker.start()
                await loop.run_in_executor(None, self._worker.wait_ready)
            except BaseException as exc:  # noqa: BLE001 - surfaced, not swallowed
                self._last_error = str(exc)
                self._loading = False
                raise EngineError(f"failed to load FishSpeech engine: {exc}") from exc

            self.sample_rate = self._worker.sample_rate
            self._load_seconds = time.perf_counter() - t0
            self._loaded = True
            self._loading = False
            logger.info(
                "FishSpeech engine ready in %.2fs (sample_rate=%d)",
                self._load_seconds,
                self.sample_rate,
            )

        if self._settings.warmup:
            await self._warmup()

    async def _warmup(self) -> None:
        """Pay the first-call CUDA cost (kernel autotune, cache allocation)
        before real traffic sees it, with a request too small to matter."""
        from app.protocol import TTSParams

        text = "Warming up the model."
        request = SynthesisRequest(
            text=text,
            params=TTSParams(text=text, max_new_tokens=64, chunk_length=200),
            session_id="__warmup__",
        )
        cancel = CancelToken()
        try:
            t0 = time.perf_counter()
            async for _ in self.synthesize(request, cancel):
                pass
            logger.info("warm-up synthesis complete in %.2fs", time.perf_counter() - t0)
        except Exception:  # noqa: BLE001 - warm-up is best-effort
            logger.warning("warm-up synthesis failed; serving traffic anyway", exc_info=True)

    async def warm_voice(self, params) -> int:
        """Prefill this voice's system prompt so no user turn has to pay for it.

        A 10-second reference clip is ~220 prompt tokens, and prefilling them
        costs seconds on a 4 GB card - a cost that lands on whichever request
        happens to be first unless it is paid here instead. Best-effort: a
        failure is logged and swallowed, because a cold cache only costs
        latency, and the next synthesis will prefill it anyway.
        """
        if not self._loaded:
            return 0
        loop = asyncio.get_running_loop()
        cancel = CancelToken()
        out_queue = self._worker.submit_warm(params, cancel)
        message = await loop.run_in_executor(None, out_queue.get)
        if isinstance(message, WarmedMessage):
            if message.prefix_tokens:
                logger.info(
                    "voice prefix warmed: %d tokens in %.0f ms",
                    message.prefix_tokens,
                    message.elapsed_ms,
                )
            return message.prefix_tokens
        if isinstance(message, ErrorMessage):
            logger.warning("voice prefix warm-up failed: %s", message.error)
        return 0

    def wake(self) -> None:
        if self._loaded:
            self._worker.wake()

    async def prefill_hint(self, text: str, params, cancel: CancelToken | None = None) -> int:
        """Prefill the prompt for a segment the client is still sending.

        Runs on the same worker queue as synthesis, so it can only ever occupy
        a gap between jobs - exactly the gap a voice agent leaves while its LLM
        is still writing the sentence. `cancel` is how the caller takes that gap
        back the moment it turns out not to be a gap after all: a hint is
        checked before every prefilled position and keeps whatever it had
        already written, so cutting one short costs nothing.
        """
        if not self._loaded:
            return 0
        loop = asyncio.get_running_loop()
        out_queue = self._worker.submit_hint(text, params, cancel or CancelToken())
        message = await loop.run_in_executor(None, out_queue.get)
        return message.prefix_tokens if isinstance(message, WarmedMessage) else 0

    async def aclose(self) -> None:
        if not self._loaded and not self._loading:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._worker.shutdown)
        self._loaded = False

    @property
    def ready(self) -> bool:
        return self._loaded and not self.worker_dead

    @property
    def worker_dead(self) -> bool:
        """The one thread that can run the model is gone; nothing will recover it.

        Its loop never raises by design, so this means something no handler
        anticipated. Reported through /health as a failure so a supervisor
        restarts the process instead of every request hanging on a queue
        nobody reads.
        """
        return self._loaded and not self._worker.alive

    def status(self) -> EngineStatus:
        return EngineStatus(
            ready=self._loaded,
            loading=self._loading,
            device=self._settings.device,
            backend="fish-speech",
            model_repo=self._settings.model_repo,
            sample_rate=self.sample_rate,
            precision="fp16" if self._settings.half else "bf16",
            quantization="nf4" if self._settings.bnb4 else "none",
            load_seconds=self._load_seconds,
            last_error=self._last_error,
            queue_depth=self._worker.queue_depth,
            active_jobs=self._active_jobs,
            total_requests=self._total_requests,
            total_cancelled=self._total_cancelled,
            total_failed=self._total_failed,
            total_audio_seconds=self._total_audio_seconds,
            last_ttfa_ms=self._last_ttfa_ms,
            vram=self._vram_stats(),
            prefix_cache=self._worker.prefix_stats(),
            kernels=self._worker.kernel_stats(),
        )

    def _vram_stats(self) -> dict[str, float]:
        if self._settings.device != "cuda":
            return {}
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            free, total = torch.cuda.mem_get_info()
            mib = 1024.0**2
            return {
                "free_mib": free / mib,
                "total_mib": total / mib,
                "used_mib": (total - free) / mib,
                "allocated_mib": torch.cuda.memory_allocated() / mib,
                "reserved_mib": torch.cuda.memory_reserved() / mib,
            }
        except Exception:  # noqa: BLE001 - status must never raise
            return {}

    async def synthesize(
        self, request: SynthesisRequest, cancel: CancelToken
    ) -> AsyncIterator[AudioChunk]:
        if not self._loaded:
            raise EngineNotReady("FishSpeech engine has not finished loading")

        loop = asyncio.get_running_loop()
        # Our own token, so a timeout can abandon this segment without
        # cancelling the rest of the caller's turn.
        job_cancel = cancel.child()
        out_queue = self._worker.submit(request.text, request.params, job_cancel)
        timeout = self._settings.queue_timeout_s or None

        self._total_requests += 1
        self._active_jobs += 1
        completed = False
        try:
            while True:
                try:
                    message = await loop.run_in_executor(None, out_queue.get, True, timeout)
                except queue.Empty:
                    job_cancel.cancel()
                    self._total_failed += 1
                    raise EngineError(
                        f"engine produced nothing for {timeout:.0f}s "
                        f"(queue depth {self._worker.queue_depth}); segment abandoned"
                    ) from None
                if isinstance(message, ChunkMessage):
                    yield message.chunk
                elif isinstance(message, DoneMessage):
                    completed = True
                    self._record_metrics(message.metrics)
                    break
                elif isinstance(message, ErrorMessage):
                    self._total_failed += 1
                    raise EngineError(str(message.error)) from message.error
                else:  # pragma: no cover - defensive, worker only sends the above
                    raise EngineError(f"unexpected worker message: {message!r}")
        finally:
            self._active_jobs -= 1
            # A cancelled consumer stops iterating, which closes this generator
            # at the `yield` above -- so the worker's DoneMessage is never read
            # and `_record_metrics` never runs. Count it here instead, or every
            # barge-in and every dropped client would vanish from /status.
            if not completed and cancel.cancelled:
                self._total_cancelled += 1

    def _record_metrics(self, metrics) -> None:
        if metrics.cancelled:
            self._total_cancelled += 1
        if metrics.ttfa_ms is not None:
            self._last_ttfa_ms = metrics.ttfa_ms
        self._total_audio_seconds += metrics.audio_seconds
