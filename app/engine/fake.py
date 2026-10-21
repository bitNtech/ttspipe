"""No-GPU stand-in for the real engine.

Generates a short, deterministic tone-plus-noise burst per segment, sized to
the text it was asked to speak, and streams it out in several chunks with
real (if tiny) `asyncio.sleep`s between them -- so anything that exercises
streaming or cancellation against a real engine exercises identically
against this one, just without a GPU or model weights. Selected with
`TTS_ENGINE=fake`; this is what makes the whole server testable in CI.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from app.engine.base import (
    AudioChunk,
    CancelToken,
    EngineNotReady,
    EngineStatus,
    SynthesisRequest,
    TTSEngine,
)

_SAMPLE_RATE = 24000
_CHARS_PER_SECOND = 16.0  # a plausible speaking rate, used only to size fake audio
_MIN_DURATION_S = 0.15
_MAX_DURATION_S = 6.0
_CHUNK_S = 0.12  # ~120ms per streamed chunk -- enough chunks to make cancellation land mid-utterance
_STEP_SLEEP_S = 0.01  # small enough not to slow tests, large enough to be a real yield point


def _seed_for(text: str, seed: int | None) -> int:
    """A deterministic 31-bit seed from (text, seed): same inputs, same waveform."""
    base = seed if seed is not None else 0
    h = 0
    for ch in text:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return (base ^ h) & 0x7FFFFFFF


class FakeEngine(TTSEngine):
    """Deterministic, GPU-free `TTSEngine` used for tests and local dev."""

    def __init__(self) -> None:
        self.sample_rate = _SAMPLE_RATE
        self._ready = False
        self._loading = False
        self._load_seconds: float | None = None
        self._last_error: str | None = None
        self._active_jobs = 0
        self._total_requests = 0
        self._total_cancelled = 0
        self._total_failed = 0
        self._total_audio_seconds = 0.0
        self._last_ttfa_ms: float | None = None

    async def load(self) -> None:
        if self._ready or self._loading:
            return
        self._loading = True
        start = time.monotonic()
        await asyncio.sleep(0.05)  # pretend there's a model to warm up
        self._load_seconds = time.monotonic() - start
        self._loading = False
        self._ready = True

    async def aclose(self) -> None:
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def status(self) -> EngineStatus:
        return EngineStatus(
            ready=self._ready,
            loading=self._loading,
            device="cpu",
            backend="fake",
            model_repo="fake://tone-generator",
            sample_rate=self.sample_rate,
            precision="float32",
            quantization="none",
            load_seconds=self._load_seconds,
            last_error=self._last_error,
            queue_depth=0,
            active_jobs=self._active_jobs,
            total_requests=self._total_requests,
            total_cancelled=self._total_cancelled,
            total_failed=self._total_failed,
            total_audio_seconds=self._total_audio_seconds,
            last_ttfa_ms=self._last_ttfa_ms,
            vram={},
        )

    async def synthesize(
        self, request: SynthesisRequest, cancel: CancelToken
    ) -> AsyncIterator[AudioChunk]:
        if not self._ready:
            raise EngineNotReady("fake engine has not finished loading")

        self._total_requests += 1
        self._active_jobs += 1
        start = time.monotonic()
        first_chunk = True
        was_cancelled = False
        try:
            text = request.text or ""
            duration_s = min(
                _MAX_DURATION_S, max(_MIN_DURATION_S, len(text) / _CHARS_PER_SECOND)
            )
            seed = _seed_for(text, request.params.seed if request.params else None)
            rng = np.random.default_rng(seed)
            freq = 180.0 + (seed % 240)  # 180-420 Hz, distinct per seed/text

            total_samples = int(duration_s * self.sample_rate)
            chunk_samples = max(1, int(_CHUNK_S * self.sample_rate))
            emitted = 0
            while emitted < total_samples:
                if cancel.cancelled:
                    was_cancelled = True
                    return
                n = min(chunk_samples, total_samples - emitted)
                t = np.arange(emitted, emitted + n, dtype=np.float32)
                tone = np.sin(2.0 * np.pi * freq * t / self.sample_rate).astype(np.float32)
                noise = rng.standard_normal(n).astype(np.float32)
                samples = (0.25 * tone + 0.03 * noise).astype(np.float32)
                emitted += n

                await asyncio.sleep(_STEP_SLEEP_S)
                if cancel.cancelled:
                    was_cancelled = True
                    return

                if first_chunk:
                    self._last_ttfa_ms = (time.monotonic() - start) * 1000.0
                    first_chunk = False
                self._total_audio_seconds += n / self.sample_rate
                yield AudioChunk(samples=samples, sample_rate=self.sample_rate)
        finally:
            self._active_jobs = max(0, self._active_jobs - 1)
            if was_cancelled:
                self._total_cancelled += 1
