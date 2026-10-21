"""Voice-cloning input: reference audio + transcript -> prompt VQ codes.

`generate_long` conditions generation on a prompt built from previously
*encoded* speech (VQ codes) plus its transcript, not from raw audio. Turning
a client's uploaded clip into that prompt means running it through the
codec's encoder once, which is where this module earns its keep: encoding is
comparatively expensive (see `_get_cpu_codec` below) and the identical clip
often arrives again either resent verbatim by a naive client, or looked up
by `reference_id` across many requests for the same cloned voice - so results
are cached and only ever computed once per distinct clip.

The GPU codec on a 4 GB card is normally loaded `decode_only=True`
(`app/config.py: codec_decode_only`), which drops the encoder entirely to
save ~0.96 GiB of VRAM (fish_speech/models/dac/inference.py). That trades
away the ability to encode reference audio on the GPU at all, so this module
builds a second, CPU-resident, *full* codec lazily the first time a reference
actually needs encoding - paying a one-off latency cost instead of a
permanent VRAM one. When the GPU codec is not decode-only (small models /
plenty of headroom), that same GPU codec is reused for encoding instead and
nothing extra is loaded.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch

    from app.config import Settings

logger = logging.getLogger(__name__)

# The model card measures 220 prompt tokens for a 10.2s reference clip versus
# 683 for 31.7s - and every one of those tokens is replayed on *every*
# subsequent request that reuses the voice, not paid once like the encode.
_TARGET_CLIP_SECONDS = 10.0
_WARN_CLIP_SECONDS = 20.0


class _AudioLike(Protocol):
    audio: bytes
    text: str


@dataclass(slots=True)
class ResolvedReference:
    """One encoded reference clip, ready to hand to `generate_long`."""

    tokens: "torch.Tensor"  # CPU tensor, shape (num_codebooks, T)
    text: str


class _BoundedCache:
    """A trivial LRU keyed by string, holding whatever callers put in it.

    Used for two different key spaces (content hash, reference_id) with two
    separate instances rather than one shared namespace, so a hash collision
    between the two schemes is structurally impossible rather than merely
    unlikely.
    """

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._order: list[str] = []
        self._entries: dict[str, object] = {}

    def get(self, key: str):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._order.remove(key)
                self._order.append(key)
            return entry

    def put(self, key: str, value: object) -> None:
        with self._lock:
            if key in self._entries:
                self._order.remove(key)
            elif len(self._entries) >= self._max_entries:
                oldest = self._order.pop(0)
                del self._entries[oldest]
            self._entries[key] = value
            self._order.append(key)

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._entries


class ReferenceManager:
    """Resolves `TTSParams.references` / `reference_id` into prompt VQ codes.

    Every public method is safe to call from the GPU worker thread (the only
    caller in practice - see app/engine/worker.py) or, defensively, from any
    other thread: the caches and the lazily-built CPU codec are lock-guarded
    even though today's call pattern never contends.
    """

    def __init__(self, settings: "Settings", gpu_codec, gpu_device: str) -> None:
        self._settings = settings
        self._gpu_codec = gpu_codec
        self._gpu_device = gpu_device

        self._hash_cache = _BoundedCache(max_entries=128)
        self._id_cache = _BoundedCache(max_entries=32)

        self._cpu_codec = None
        self._cpu_codec_build_lock = threading.Lock()
        self._cpu_codec_use_lock = threading.Lock()

    # --- public API ---

    def resolve(
        self, references: "list[_AudioLike]", reference_id: str | None
    ) -> "tuple[list[torch.Tensor], list[str]]":
        """-> (prompt_tokens, prompt_texts) in speaker order, both lists.

        Empty lists mean "no reference" - `generate_long` treats that as
        unconditioned generation, which is what an empty `references` and no
        `reference_id` should mean.
        """
        if reference_id:
            resolved = self._resolve_by_id(reference_id)
        elif references:
            resolved = self._resolve_by_hash(references)
        else:
            return [], []
        return [r.tokens for r in resolved], [r.text for r in resolved]

    # --- content-hash path ---

    def _resolve_by_hash(self, references: "list[_AudioLike]") -> list[ResolvedReference]:
        import hashlib

        out: list[ResolvedReference] = []
        for ref in references:
            key = hashlib.sha256(ref.audio).hexdigest()
            cached = self._hash_cache.get(key)
            if cached is None:
                cached = self._encode(ref.audio, ref.text)
                self._hash_cache.put(key, cached)
            out.append(cached)
        return out

    # --- reference_id path ---

    def _resolve_by_id(self, reference_id: str) -> list[ResolvedReference]:
        cached = self._id_cache.get(reference_id)
        if cached is not None:
            return cached

        resolved = self._load_reference_dir(reference_id)
        self._id_cache.put(reference_id, resolved)
        return resolved

    def _load_reference_dir(self, reference_id: str) -> list[ResolvedReference]:
        """references/<id>/*.{wav,...} + matching .lab transcripts.

        Mirrors the fork's ReferenceLoader.load_by_id convention (an audio
        file plus a same-stem .lab transcript per clip) but resolves against
        the repo root rather than the process cwd, and additionally caches
        each clip's encoded tokens next to it on disk so a second process (or
        a restarted one) skips the encode too, not just this process's
        in-memory cache.
        """
        from app.config import REPO_ROOT

        ref_dir = REPO_ROOT / "references" / reference_id
        if not ref_dir.is_dir():
            # Checked before importing fish_speech so a bad id fails fast and
            # cleanly even if, e.g., the vendored fork has import problems.
            raise ValueError(f"unknown reference_id {reference_id!r}: {ref_dir} does not exist")

        from fish_speech.utils.file import AUDIO_EXTENSIONS, list_files, read_ref_text

        audio_files = list_files(ref_dir, AUDIO_EXTENSIONS, recursive=False, sort=True)
        if not audio_files:
            raise ValueError(f"reference_id {reference_id!r} has no audio clips in {ref_dir}")

        out: list[ResolvedReference] = []
        for audio_path in audio_files:
            text_path = audio_path.with_suffix(".lab")
            text = read_ref_text(str(text_path)) if text_path.exists() else ""

            tokens_cache = audio_path.with_suffix(".tokens.pt")
            tokens = self._load_or_encode_cached(audio_path, tokens_cache)
            out.append(ResolvedReference(tokens=tokens, text=text))
        return out

    def _load_or_encode_cached(self, audio_path, tokens_cache) -> "torch.Tensor":
        import torch

        if tokens_cache.exists():
            logger.debug("loaded precomputed reference tokens: %s", tokens_cache)
            return torch.load(tokens_cache, map_location="cpu", weights_only=True)

        tokens = self._encode_from_path(audio_path)
        try:
            torch.save(tokens, tokens_cache)
        except OSError:
            # References may live on a read-only mount; the encode still
            # succeeded, only the on-disk cache write is skipped.
            logger.warning("could not write reference token cache %s", tokens_cache)
        return tokens

    def _encode_from_path(self, audio_path) -> "torch.Tensor":
        return self._encode(audio_path.read_bytes(), "").tokens

    # --- encoding ---

    def _encode(self, audio_bytes: bytes, text: str) -> ResolvedReference:
        import torch
        import torchaudio

        codec = self._encoding_codec()
        codec_device = next(codec.parameters()).device
        sample_rate = codec.sample_rate

        waveform, orig_sr = torchaudio.load(io.BytesIO(audio_bytes))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_sr, sample_rate)

        duration_s = waveform.shape[-1] / sample_rate
        if duration_s > _WARN_CLIP_SECONDS:
            logger.warning(
                "reference clip is %.1fs (target ~%.0fs); its prompt token "
                "count scales with length and is paid on every request that "
                "reuses this voice, not just this one",
                duration_s,
                _TARGET_CLIP_SECONDS,
            )

        audio = waveform.to(codec_device)[None]  # (1, 1, T)
        audio_lengths = torch.tensor([audio.shape[-1]], device=codec_device, dtype=torch.long)

        use_lock = self._cpu_codec_use_lock if codec is self._cpu_codec else None
        if use_lock is not None:
            use_lock.acquire()
        try:
            with torch.inference_mode():
                indices, feature_lengths = codec.encode(audio, audio_lengths)
            tokens = indices[0, :, : feature_lengths[0]].detach().to("cpu")
        finally:
            if use_lock is not None:
                use_lock.release()

        return ResolvedReference(tokens=tokens, text=text)

    def _encoding_codec(self):
        if not self._settings.codec_decode_only:
            return self._gpu_codec
        return self._get_cpu_codec()

    def _get_cpu_codec(self):
        if self._cpu_codec is None:
            with self._cpu_codec_build_lock:
                if self._cpu_codec is None:
                    self._cpu_codec = self._build_cpu_codec()
        return self._cpu_codec

    def _build_cpu_codec(self):
        """A lazily-built, CPU-resident, encode-capable codec.

        Built once, on first use, so a server that never receives a raw
        reference clip (only ever uses `reference_id`s with precomputed
        tokens) never pays this cost at all.
        """
        import time

        from fish_speech.models.dac.inference import load_model as load_codec

        logger.info(
            "building CPU-resident codec for reference encoding "
            "(one-off; the GPU codec is decode_only and cannot encode)"
        )
        t0 = time.perf_counter()
        codec = load_codec(
            config_name="modded_dac_vq",
            checkpoint_path=str(self._settings.checkpoint_dir / "codec.pth"),
            device="cpu",
            decode_only=False,
            precision=None,
        )
        logger.info("CPU reference codec ready in %.2fs", time.perf_counter() - t0)
        return codec
