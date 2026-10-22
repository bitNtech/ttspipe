"""Runtime configuration, entirely driven by environment variables.

Every setting has a working default so the server starts with no env at all.
`run.sh` exports the same names, so the shell and the app never disagree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: str) -> Path:
    raw = os.environ.get(name, default)
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    # --- server ---
    host: str = field(default_factory=lambda: _str("TTS_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _int("TTS_PORT", 8080))
    log_level: str = field(default_factory=lambda: _str("TTS_LOG_LEVEL", "INFO").upper())
    # "text" for humans, "json" (one object per line) for a log pipeline.
    log_format: str = field(default_factory=lambda: _str("TTS_LOG_FORMAT", "text").lower())
    # Comma-separated API keys. Unset means no authentication, which is right
    # on a loopback-only box and wrong on anything reachable. When set, every
    # route except /health and /ready wants one (see app/security.py).
    api_keys: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            k.strip() for k in _str("TTS_API_KEYS", "").split(",") if k.strip()
        )
    )
    # Comma-separated origins allowed to call the HTTP API from a browser.
    # Empty: no CORS headers at all (same-origin only). "*" allows any.
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            o.strip() for o in _str("TTS_CORS_ORIGINS", "").split(",") if o.strip()
        )
    )

    # --- engine selection ---
    # "fish" (FishSpeech S2-Pro NF4) or "mio" (SPRINGLab/Indic-Mio). Every
    # fish-only setting below is ignored under "mio".
    engine: str = field(default_factory=lambda: _str("TTS_ENGINE", "fish").strip().lower())
    mio_model_repo: str = field(
        default_factory=lambda: _str("TTS_MIO_MODEL_REPO", "SPRINGLab/Indic-Mio")
    )
    mio_codec_repo: str = field(
        default_factory=lambda: _str("TTS_MIO_CODEC_REPO", "Aratako/MioCodec-25Hz-24kHz")
    )
    # Voice used when a request carries no reference clip. Mio has no built-in
    # voice, so without this (or an active voice) such a request is an error.
    mio_default_reference: str = field(
        default_factory=lambda: _str("TTS_MIO_DEFAULT_REFERENCE", "")
    )

    # --- model location ---
    model_repo: str = field(
        default_factory=lambda: _str("TTS_MODEL_REPO", "scarxity/fish-speech-s2-pro-nf4")
    )
    checkpoint_dir: Path = field(
        default_factory=lambda: _path("TTS_CHECKPOINT_DIR", "checkpoints/s2-pro-nf4")
    )
    vendor_dir: Path = field(
        default_factory=lambda: _path("TTS_VENDOR_DIR", "vendor/fish-speech")
    )

    # --- inference / VRAM ---
    device: str = field(default_factory=lambda: _str("TTS_DEVICE", "cuda"))
    bnb4: bool = field(default_factory=lambda: _bool("TTS_BNB4", True))
    half: bool = field(default_factory=lambda: _bool("TTS_HALF", True))
    compile_model: bool = field(default_factory=lambda: _bool("TTS_COMPILE", False))
    max_seq_len: int = field(default_factory=lambda: _int("TTS_MAX_SEQ_LEN", 1536))
    # The decode-only codec drops the analysis half (~0.96 GiB of weights).
    # Reference clips are then encoded by a lazily built CPU copy instead.
    codec_decode_only: bool = field(
        default_factory=lambda: _bool("TTS_CODEC_DECODE_ONLY", True)
    )
    # The 4 GB profile. The embedding table is 155776 x 2560 fp16 (~0.72 GiB);
    # keeping it in host memory and reading only the resident 4097-row slice is
    # what makes S2-Pro fit on a 4 GB card at all. It requires the sliced
    # lm_head, so `full_lm_head` must stay off alongside it.
    offload_embeddings: bool = field(
        default_factory=lambda: _bool("TTS_OFFLOAD_EMBEDDINGS", True)
    )
    full_lm_head: bool = field(default_factory=lambda: _bool("TTS_FULL_LM_HEAD", False))
    # Codec decode is the peak-memory step: one token frame becomes 2048 audio
    # samples. Chunking it stops activations scaling with utterance length --
    # bit-exact, ~15% slower on long utterances, ~250 MiB of headroom.
    decode_chunk_frames: int = field(
        default_factory=lambda: _int("TTS_DECODE_CHUNK_FRAMES", 64)
    )
    decode_overlap_frames: int = field(
        default_factory=lambda: _int("TTS_DECODE_OVERLAP_FRAMES", 32)
    )
    # The DAC decoder's convolution stack overflows to NaN in fp16 on Turing
    # (sm_75), which has no tensor cores -- measured: every fp16 combination
    # yields all-NaN audio, fp32 without autocast is clean. The fork's fp16
    # default was validated on Ampere+. "auto" picks fp32 below sm_80.
    codec_precision: str = field(
        default_factory=lambda: _str("TTS_CODEC_PRECISION", "auto").lower()
    )
    warmup: bool = field(default_factory=lambda: _bool("TTS_WARMUP", True))

    # --- latency ---
    # Prefill the reference-voice system prompt once and reuse its KV entries.
    # A 10s reference is ~220 prompt tokens, and re-prefilling them costs ~5.2s
    # of every single request's time-to-first-audio.
    prefix_cache: bool = field(default_factory=lambda: _bool("TTS_PREFIX_CACHE", True))
    # Serve single-vector 4-bit matmuls from app/engine/nf4gemv.py instead of
    # bitsandbytes. Decoding a frame is nothing but 76 of those calls, and
    # bitsandbytes' NF4 GEMV reads weights at 35-48 GB/s on a card that copies
    # at 158 - a row-blocked kernel gets 80-111. Off falls back to
    # bitsandbytes, which is correct and about twice as slow per frame.
    fast_gemv: bool = field(default_factory=lambda: _bool("TTS_FAST_GEMV", True))
    # Run RMS norm and rotary embedding as one kernel each instead of the six
    # to nine elementwise operations PyTorch spells them as. There are 387 of
    # them per frame and none touches more than a few kilobytes, so what this
    # removes is ~2900 kernel launches a frame, not arithmetic.
    fused_ops: bool = field(default_factory=lambda: _bool("TTS_FUSED_OPS", True))
    # Replay each decode step as a captured CUDA graph instead of launching its
    # ~4000 kernels one by one. On Windows a launch costs ~30 us of CPU, so an
    # eager frame measured 152 ms against 53 ms of GPU work. Costs a few
    # seconds at start-up to capture one graph per KV-window bucket.
    cuda_graphs: bool = field(default_factory=lambda: _bool("TTS_CUDA_GRAPHS", True))
    # Keep the 200 MiB VQ codebook embedding table in pinned host memory and
    # gather the ten rows a frame needs over PCIe (app/engine/hostembed.py).
    # On a 4 GB card this is the difference between headroom and the Windows
    # driver silently paging GPU memory to system RAM.
    host_codebook_embeddings: bool = field(
        default_factory=lambda: _bool("TTS_HOST_CODEBOOK_EMBEDDINGS", True)
    )
    # VRAM the allocator must leave alone for the CUDA context, cuDNN and the
    # captured graphs. 0 disables the cap. See GpuWorker._load for why a cap
    # is needed at all on Windows.
    vram_headroom_mb: int = field(default_factory=lambda: _int("TTS_VRAM_HEADROOM_MB", 640))
    # --- CPU ---
    # On CPU, bitsandbytes' NF4 matmul dequantises the weight on every call and
    # measured ~4x slower than a dense bf16 one (AMX/oneDNN). With the RAM to
    # spare, expand the NF4 checkpoint to dense weights once at load: same
    # quantised values, ~8 GB resident instead of ~2.5. Ignored on CUDA.
    cpu_dense: bool = field(default_factory=lambda: _bool("TTS_CPU_DENSE", True))
    # How the rebuilt layers are served: "nf4" (exact values, AVX-512 and AMX
    # kernels in app/engine/nf4cpu.py; falls back to bf16 without
    # AVX512-BF16) or "bf16" (a dense copy, served by oneDNN).
    cpu_weights: str = field(default_factory=lambda: _str("TTS_CPU_WEIGHTS", "nf4").lower())
    # Run each decode position through a block as one C++ call (norms, rotary,
    # KV write, attention, residuals and the NF4 matmuls) - see
    # app/engine/cpu.py:install_decode_blocks. Needs cpu_weights=nf4.
    cpu_fused_decode: bool = field(default_factory=lambda: _bool("TTS_CPU_FUSED_DECODE", True))
    # Intra-op threads for torch on CPU. 0 = one per logical CPU (hyperthreads
    # measured faster than physical cores only; see app/engine/cpu.py).
    cpu_threads: int = field(default_factory=lambda: _int("TTS_CPU_THREADS", 0))
    # Frames per emitted audio block, ramping from the first to the largest.
    # One frame is 2048 samples (46.4ms of audio at 44.1kHz) and costs ~43ms
    # to generate on an RTX 2050, so the first block is the biggest term in
    # time-to-first-audio. Later blocks stay small on purpose: generation runs
    # at about realtime, so a block only reaches the client once all of it is
    # generated, and a player holding one block of audio can ride out the
    # next. The streaming codec's per-call cost (~5 ms plus ~1.7 ms a frame)
    # is what stops them being single frames. Simulated and measured, a
    # constant 4 needs the least client buffering (docs/LATENCY.md).
    stream_first_frames: int = field(
        default_factory=lambda: _int("TTS_STREAM_FIRST_FRAMES", 1)
    )
    stream_max_frames: int = field(default_factory=lambda: _int("TTS_STREAM_MAX_FRAMES", 4))
    # Shortest half-finished segment worth speculatively prefilling. Below this
    # the prompt is nearly all system prefix, which is already cached.
    min_hint_chars: int = field(default_factory=lambda: _int("TTS_MIN_HINT_CHARS", 8))
    # Whether a speculative prefill guesses the whole prompt shape (closing
    # im_end + assistant turn) or stops after the text. Including it is the only
    # way a hint can ever leave nothing to do, but it is also re-done at a new
    # position every time a word arrives, so it can crowd out forward progress
    # on the text itself. Which wins is a measurement, not a principle - see
    # docs/LATENCY.md.
    #
    # Unset means auto: on for CUDA, where a batch of positions costs about one
    # and a full hit is worth it; off for CPU, where every new word invalidates
    # the guessed suffix and redoing it (~5 tokens) costs more than the word
    # interval - measured 52-56 ms of queueing per segment with it, 0-3 without,
    # and ~195 ms engine time to first audio either way.
    hint_include_suffix: bool | None = field(
        default_factory=lambda: (
            None
            if os.environ.get("TTS_HINT_INCLUDE_SUFFIX", "").strip() == ""
            else _bool("TTS_HINT_INCLUDE_SUFFIX", True)
        )
    )
    # Most prompt positions one speculative prefill may write. Hints always walk
    # positions one at a time, never batched: a batched prefill is a single
    # forward call and therefore cannot be abandoned once it starts, so a run of
    # them queues ahead of the real segment (measured: 9.2s to first audio).
    # Walking is cancellable every ~48ms, which makes a generous budget safe -
    # and it needs to be generous, because one Tamil word is ~11 tokens.
    hint_max_tokens: int = field(default_factory=lambda: _int("TTS_HINT_MAX_TOKENS", 32))
    # Positions a hint prefills per forward call, checking for cancellation
    # between calls. 0 = auto: 4 on CPU (see StreamingGenerator._hint_chunk),
    # one batched call on CUDA, where the 4-bit kernel makes a batch of 32 cost
    # about one position. -1 forces the single batched call everywhere.
    hint_chunk_tokens: int = field(default_factory=lambda: _int("TTS_HINT_CHUNK_TOKENS", 0))
    # Floor on how often a speculative prefill may be issued. This is only a
    # guard against re-tokenising the same text many times a second; what
    # actually bounds the work is `GpuWorker.submit_hint`, which holds one slot
    # rather than a queue so a newer hint replaces a waiting one. Rate-limiting
    # was tried as the bound instead and is worse - it starves the hint, which
    # costs token-dense languages most (English held at 429ms, Tamil pushed to
    # 1130ms, against 276ms and 830ms with coalescing).
    hint_min_interval_ms: int = field(
        default_factory=lambda: _int("TTS_HINT_MIN_INTERVAL_MS", 60)
    )
    # How the prompt's KV entries get populated. bitsandbytes' NF4 kernels only
    # take the fused `gemv_4bit` path at sequence length 1; a longer sequence
    # dequantises every weight into a dense GEMM first, which measures as a
    # ~477ms fixed cost per call on a GTX 1650 *whatever* the span - 8 tokens
    # and 19 tokens both cost ~575ms. Walking positions one at a time avoids
    # that entirely at ~48ms each, so short spans are cheaper sequentially and
    # long ones are not. Measured crossover: ~11 tokens.
    prefill_mode: str = field(
        default_factory=lambda: _str("TTS_PREFILL_MODE", "auto").lower()
    )
    prefill_sequential_max: int = field(
        default_factory=lambda: _int("TTS_PREFILL_SEQUENTIAL_MAX", 11)
    )
    # These two only govern the recomputing codec path, used when CUDA graphs
    # are off or a span cannot stream; the streaming codec
    # (app/engine/codecstream.StreamingCodec) carries its state instead.
    #
    # Left context for the codec's causal convolutions when decoding a block on
    # its own. Its receptive field is 10 frames, so 10 is exact; every extra
    # frame is convolution work whose audio is thrown away.
    codec_lookback_frames: int = field(
        default_factory=lambda: _int("TTS_CODEC_LOOKBACK_FRAMES", 10)
    )
    # How far back post_module is re-run per block. Its 8 layers of 128-frame
    # windows reach back ~1000 frames. Measured with the fp16 codec against a
    # single full pass: 512 is 48 dB SNR (the fp16 floor - a full pass over a
    # shorter prefix differs from a longer one by the same 47 dB), 256 only 32.
    codec_post_context_frames: int = field(
        default_factory=lambda: _int("TTS_CODEC_POST_CONTEXT_FRAMES", 512)
    )

    # --- streaming behaviour ---
    # Text is held back until a segment looks speakable; these bound that wait.
    min_segment_chars: int = field(default_factory=lambda: _int("TTS_MIN_SEGMENT_CHARS", 24))
    # ...except a reply's first segment, which is cut at the first finished
    # sentence of at least this many characters: it alone decides when the
    # caller hears anything, so "Sure." should not wait for the next sentence.
    first_segment_min_chars: int = field(
        default_factory=lambda: _int("TTS_FIRST_SEGMENT_MIN_CHARS", 2)
    )
    max_segment_chars: int = field(default_factory=lambda: _int("TTS_MAX_SEGMENT_CHARS", 220))
    chunk_length: int = field(default_factory=lambda: _int("TTS_CHUNK_LENGTH", 200))
    default_format: str = field(default_factory=lambda: _str("TTS_DEFAULT_FORMAT", "wav"))
    # Output rate for requests that name none. 0 keeps the engine's native
    # rate (44.1 kHz for S2-Pro). Set 16000 or 24000 to match what the agent's
    # audio pipeline consumes, so no client has to ask.
    default_sample_rate: int = field(default_factory=lambda: _int("TTS_DEFAULT_SAMPLE_RATE", 0))

    # --- concurrency ---
    max_sessions: int = field(default_factory=lambda: _int("TTS_MAX_SESSIONS", 8))
    # Longest the engine may go without producing anything for a segment -
    # waiting in the queue behind other sessions included - before the segment
    # is abandoned with an error instead of hanging the turn. 0 disables.
    queue_timeout_s: float = field(default_factory=lambda: _float("TTS_QUEUE_TIMEOUT_S", 120.0))
    # Close a WebSocket that has sent nothing for this long while no audio is
    # in flight, so a client that vanished without a FIN cannot hold one of
    # `max_sessions` forever. A `ping` frame counts as activity. 0 disables.
    session_idle_timeout_s: float = field(
        default_factory=lambda: _float("TTS_SESSION_IDLE_TIMEOUT_S", 300.0)
    )

    def as_dict(self) -> dict:
        """JSON-safe, and safe to show: API keys are reported as a count."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "api_keys":
                out["api_keys_configured"] = len(v)
                continue
            out[f.name] = str(v) if isinstance(v, Path) else list(v) if isinstance(v, tuple) else v
        return out


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings_cache() -> None:
    """Test hook: force the next get_settings() to re-read the environment."""
    global _settings
    _settings = None
