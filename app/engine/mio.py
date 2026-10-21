"""Indic-Mio engine (SPRINGLab/Indic-Mio): a Qwen3-0.6B LM that emits MioCodec
speech tokens, decoded to audio by MioCodec. Selected with `TTS_ENGINE=mio`.

Voice = the codec's global embedding, taken from the first reference clip of
the request (so `POST /v1/voice` and per-frame references both work) or, with
none, from `TTS_MIO_DEFAULT_REFERENCE`. The reference transcript is unused:
Mio conditions on the speaker embedding, not on a text prompt.

ponytail: a segment is generated whole, then decoded once and sliced into
chunks, so time-to-first-audio is the full segment's generation time. At
RTF < 0.1 and ~220-char segments that is well under a second; stream
token-chunks through the codec only if measurement says it matters.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator

import numpy as np

from app.engine.base import (
    AudioChunk,
    CancelToken,
    EngineError,
    EngineNotReady,
    EngineStatus,
    SynthesisRequest,
    TTSEngine,
)

logger = logging.getLogger(__name__)

_SPEECH_OFFSET = 151669  # first speech token id in the LM vocab
_SPEECH_CODES = 12800
_CHUNK_S = 0.25


class MioEngine(TTSEngine):
    def __init__(self, settings) -> None:
        self._s = settings
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mio")  # one GPU, one thread
        self._load_lock = asyncio.Lock()
        self._loaded = False
        self._loading = False
        self._load_seconds: float | None = None
        self._last_error: str | None = None
        self._active_jobs = 0
        self._total_requests = 0
        self._total_cancelled = 0
        self._total_failed = 0
        self._total_audio_seconds = 0.0
        self._last_ttfa_ms: float | None = None
        self._embeddings: dict[str, object] = {}  # clip sha1 -> global embedding

    # --- lifecycle ---
    async def load(self) -> None:
        async with self._load_lock:
            if self._loaded:
                return
            self._loading = True
            start = time.monotonic()
            try:
                await asyncio.get_running_loop().run_in_executor(self._pool, self._load_sync)
            except Exception as exc:
                self._last_error = str(exc)
                raise
            finally:
                self._loading = False
            self._load_seconds = time.monotonic() - start
            self._loaded = True

    def _load_sync(self) -> None:
        import torch
        from miocodec import MioCodecModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        s = self._s
        dtype = torch.float16 if s.half else torch.bfloat16
        self._tok = AutoTokenizer.from_pretrained(s.mio_model_repo, trust_remote_code=True)
        self._lm = AutoModelForCausalLM.from_pretrained(
            s.mio_model_repo, torch_dtype=dtype, trust_remote_code=True
        ).to(s.device).eval()
        self._codec = MioCodecModel.from_pretrained(s.mio_codec_repo).eval().to(s.device)
        self.sample_rate = int(self._codec.config.sample_rate)
        if s.mio_default_reference:
            self._default_embedding = self._embed(open(s.mio_default_reference, "rb").read())
        else:
            self._default_embedding = None
        if s.warmup and self._default_embedding is not None:
            self._generate("Hello.", self._default_embedding, 0.9, 0.9, 7, 64, CancelToken())

    async def aclose(self) -> None:
        self._loaded = False
        self._pool.shutdown(wait=True)

    @property
    def ready(self) -> bool:
        return self._loaded

    def status(self) -> EngineStatus:
        vram: dict[str, float] = {}
        try:
            import torch

            if self._s.device == "cuda" and torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                mib = 1024.0**2
                vram = {
                    "free_mib": free / mib,
                    "total_mib": total / mib,
                    "used_mib": (total - free) / mib,
                    "allocated_mib": torch.cuda.memory_allocated() / mib,
                    "reserved_mib": torch.cuda.memory_reserved() / mib,
                }
        except Exception:  # noqa: BLE001 - status must never raise
            pass
        return EngineStatus(
            ready=self._loaded,
            loading=self._loading,
            device=self._s.device,
            backend="indic-mio",
            model_repo=self._s.mio_model_repo,
            sample_rate=self.sample_rate,
            precision="fp16" if self._s.half else "bf16",
            quantization="none",
            load_seconds=self._load_seconds,
            last_error=self._last_error,
            active_jobs=self._active_jobs,
            total_requests=self._total_requests,
            total_cancelled=self._total_cancelled,
            total_failed=self._total_failed,
            total_audio_seconds=self._total_audio_seconds,
            last_ttfa_ms=self._last_ttfa_ms,
            vram=vram,
        )

    # --- inference (GPU thread) ---
    def _embed(self, audio: bytes):
        """Reference clip bytes -> codec global embedding, cached per clip."""
        key = hashlib.sha1(audio).hexdigest()
        if key not in self._embeddings:
            import soundfile as sf
            import torch
            import torchaudio

            wav, sr = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav.mean(axis=1))
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            wav = (wav / (wav.abs().max() + 1e-8)).to(self._s.device)  # as miocodec.load_audio
            self._embeddings[key] = self._codec.encode(wav, return_content=False).global_embedding
        return self._embeddings[key]

    def _generate(self, text, embedding, temperature, top_p, seed, max_new, cancel) -> np.ndarray:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _Cancel(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                return torch.full((input_ids.shape[0],), cancel.cancelled, dtype=torch.bool,
                                  device=input_ids.device)

        if seed is not None:
            torch.manual_seed(seed)
        prompt = self._tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._s.device)
        with torch.inference_mode():
            out = self._lm.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                stopping_criteria=StoppingCriteriaList([_Cancel()]),
            )
            ids = out[0, inputs["input_ids"].shape[1]:]
            codes = ids[(ids >= _SPEECH_OFFSET) & (ids < _SPEECH_OFFSET + _SPEECH_CODES)] - _SPEECH_OFFSET
            if cancel.cancelled or codes.numel() == 0:
                return np.zeros(0, dtype=np.float32)
            wav = self._codec.decode(content_token_indices=codes, global_embedding=embedding)
        return wav.float().clamp(-1, 1).cpu().numpy()

    def _synth_sync(self, request: SynthesisRequest, cancel: CancelToken) -> np.ndarray:
        p = request.params
        refs = p.references
        embedding = self._embed(refs[0].audio) if refs else self._default_embedding
        if embedding is None:
            raise EngineError(
                "Indic-Mio needs a voice: send a reference clip, set one with POST /v1/voice, "
                "or set TTS_MIO_DEFAULT_REFERENCE"
            )
        # `max_new_tokens=0` means "no explicit cap"; 25 tokens/s, so this is ~80 s.
        max_new = p.max_new_tokens or 2048
        return self._generate(request.text, embedding, p.temperature, p.top_p, p.seed, max_new, cancel)

    async def synthesize(
        self, request: SynthesisRequest, cancel: CancelToken
    ) -> AsyncIterator[AudioChunk]:
        if not self._loaded:
            raise EngineNotReady("Indic-Mio engine has not finished loading")
        self._total_requests += 1
        self._active_jobs += 1
        start = time.monotonic()
        try:
            try:
                samples = await asyncio.get_running_loop().run_in_executor(
                    self._pool, self._synth_sync, request, cancel
                )
            except Exception:
                self._total_failed += 1
                raise
            step = int(_CHUNK_S * self.sample_rate)
            for i in range(0, len(samples), step):
                if cancel.cancelled:
                    break
                chunk = samples[i : i + step]
                if i == 0:
                    self._last_ttfa_ms = (time.monotonic() - start) * 1000.0
                self._total_audio_seconds += len(chunk) / self.sample_rate
                yield AudioChunk(samples=chunk, sample_rate=self.sample_rate)
        finally:
            self._active_jobs -= 1
            if cancel.cancelled:
                self._total_cancelled += 1
