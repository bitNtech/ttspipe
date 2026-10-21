"""The seam between the transport layer and the GPU.

The WebSocket layer only ever sees `TTSEngine`. Everything that knows about
torch, bitsandbytes or the fish_speech package lives behind this interface, so
the server can be exercised end-to-end on a machine with no CUDA at all by
swapping in the fake engine.

Audio crosses this boundary as mono float32 in [-1, 1] at `sample_rate`, one
numpy array per decoded segment. Encoding to wav/pcm/mp3/flac is the transport
layer's job; the engine never builds container formats.
"""

from __future__ import annotations

import abc
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.protocol import TTSParams


class EngineError(RuntimeError):
    """Inference failed for a reason worth reporting to the client verbatim."""


class EngineNotReady(EngineError):
    """A request arrived before the weights finished loading."""


class Cancelled(Exception):
    """Raised inside the worker to unwind a generation that was aborted."""


class CancelToken:
    """One-way flag, set from the event loop and polled from the GPU thread.

    Cancellation is checked between sampled tokens, so barge-in latency is one
    decode step (single-digit milliseconds) rather than one audio segment.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise Cancelled()


@dataclass(slots=True)
class AudioChunk:
    """One decoded segment of speech."""

    samples: np.ndarray  # float32, mono, shape (n,)
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return len(self.samples) / float(self.sample_rate)


@dataclass(slots=True)
class SynthesisRequest:
    """One speakable segment of text plus the voice and sampling settings."""

    text: str
    params: "TTSParams"
    session_id: str = ""


@dataclass
class EngineStatus:
    """What `/status` reports. Plain types only — this gets JSON-encoded."""

    ready: bool = False
    loading: bool = False
    device: str = "cpu"
    backend: str = "unknown"
    model_repo: str = ""
    sample_rate: int = 0
    precision: str = ""
    quantization: str = ""
    load_seconds: float | None = None
    last_error: str | None = None
    queue_depth: int = 0
    active_jobs: int = 0
    total_requests: int = 0
    total_cancelled: int = 0
    total_failed: int = 0
    total_audio_seconds: float = 0.0
    last_ttfa_ms: float | None = None
    vram: dict[str, float] = field(default_factory=dict)
    #: Prompt-prefix reuse, the single biggest lever on time-to-first-audio
    #: when a reference voice is in play. See app/engine/streamgen.py.
    prefix_cache: dict = field(default_factory=dict)
    #: How many calls the hand-written kernels served and how many went back to
    #: bitsandbytes or to the fork. A fallback is silent and merely slower, so
    #: a rising `fallback` is the only way to notice one. See
    #: app/engine/nf4gemv.py and app/engine/fusedops.py.
    kernels: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


class TTSEngine(abc.ABC):
    """Abstract text-to-speech backend."""

    #: Native output rate of the codec. Valid once `load()` has returned.
    sample_rate: int = 0

    @abc.abstractmethod
    async def load(self) -> None:
        """Bring the model up. Idempotent; safe to call concurrently."""

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Release the worker thread and any GPU memory."""

    @property
    @abc.abstractmethod
    def ready(self) -> bool: ...

    @abc.abstractmethod
    def status(self) -> EngineStatus: ...

    async def warm_voice(self, params) -> int:
        """Prefill whatever prompt state this voice implies, off the hot path.

        Returns the number of prompt tokens now resident, 0 if the backend has
        no such state. Not abstract: a backend with no prompt cache is correct
        to do nothing here, and callers treat it as best-effort.
        """
        return 0

    def wake(self) -> None:
        """Hint that work is about to arrive. Optional; a no-op by default."""

    async def prefill_hint(self, text: str, params, cancel: "CancelToken | None" = None) -> int:
        """Speculatively prepare state for text that is still arriving.

        Returns the number of prompt tokens prefilled, 0 if the backend has no
        such state or declined. Purely an optimisation: callers must not depend
        on it having happened, and a backend is free to make it a no-op.
        """
        return 0

    @abc.abstractmethod
    def synthesize(
        self, request: SynthesisRequest, cancel: CancelToken
    ) -> AsyncIterator[AudioChunk]:
        """Stream audio for one segment.

        Implementations are async generators. They must yield the first chunk as
        soon as it exists rather than accumulating, must poll `cancel` often
        enough that abandonment is felt immediately, and must stop cleanly (no
        exception) once cancelled.
        """


async def drain(
    engine: TTSEngine, request: SynthesisRequest, cancel: CancelToken
) -> np.ndarray:
    """Collect a whole segment into one array. Test and warm-up helper."""
    parts: list[np.ndarray] = []
    async for chunk in engine.synthesize(request, cancel):
        parts.append(chunk.samples)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)
