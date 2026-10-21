"""The single GPU worker thread.

A GTX 1650 has 4096 MiB of VRAM total; it cannot hold state for two
generations in flight at once (the KV cache alone is sized for
`max_batch_size=1` - see `_load`). So instead of fighting that limit, one
daemon thread owns the model end to end - semantic transformer, codec, and
the CPU fallback codec in `references.py` - and every request is a job on a
`queue.Queue` that thread drains one at a time. That serialisation is the
correct design for this card, not a stopgap: the wall-clock cost of a
request is dominated by token decode, and two jobs time-sliced on the same
GPU would just take turns anyway, with the added risk of blowing the VRAM
budget the moment they overlap.

Cancellation (barge-in) is the one thing that has to cut across that
serial queue with low latency, so it does not wait for a job to reach the
front of the queue or even finish stepping through a batch: `_wrap_cancellable`
closes over `decode_one_token` (the per-token sampling callable `init_model`
hands back) and checks the in-flight job's `CancelToken` before every single
token. Because `generate_long` is an ordinary Python generator built out of
plain function calls (not `asyncio`), a `Cancelled` raised inside that
closure unwinds through `decode_n_tokens` -> `generate` -> `generate_long`
exactly like any exception unwinds through nested calls, landing back in
`_handle`'s `try/except Cancelled` within roughly one decode step - typically
single-digit milliseconds - rather than having to wait out the rest of the
current audio segment.

The fork's own `launch_thread_safe_queue` (fish_speech/models/text2semantic/
inference.py) has no such hook, which is why this module exists instead of
using it directly - though its init_event/init_error pattern for surfacing a
load failure to the caller, instead of hanging forever, is worth keeping and
is reproduced in `wait_ready`.

Latency: this thread does not call `generate_long`. That function yields once
per text batch, after the batch's last token is sampled, so time-to-first-audio
is the cost of generating the entire batch - and it re-prefills the reference
voice's ~220 prompt tokens every request on top. `app/engine/streamgen.py`
replaces it with a loop that yields code frames as they are sampled and
prefills the voice prompt once, and `app/engine/codecstream.py` turns those
frames into audio a block at a time. Together with the attention fast path in
`app/engine/fastpath.py` that is what takes first audio from ~20.8s to under a
second on a GTX 1650.
"""

from __future__ import annotations

import contextlib
import logging
import queue
from collections import deque
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from app.engine.base import AudioChunk, Cancelled, CancelToken

if TYPE_CHECKING:
    import torch

    from app.config import Settings
    from app.protocol import TTSParams

logger = logging.getLogger(__name__)

_SENTINEL = object()
#: Placed on the job queue to say "a hint is waiting"; the request itself lives
#: in a single slot so a newer one can replace it without queueing twice.
_HINT_MARKER = object()
#: Placed on the job queue to say "keep the GPU clocked up for a moment".
_WAKE_MARKER = object()

#: How long one `wake()` keeps the GPU busy if no work arrives. The card drops
#: to idle clocks within a second or two of no work and needs a few hundred ms
#: of load to come back. Measured on an RTX 2050, first audio after 5 s idle:
#: 387 ms with no wake, 260 ms after 0.15 s of it, 107 ms after 0.3 s, 56 ms
#: (the warm figure) after 0.6 s.
WAKE_SECONDS = 0.6


@dataclass(slots=True)
class JobMetrics:
    """Per-job numbers surfaced through `EngineStatus` by app/engine/fish.py."""

    queued_at: float
    started_at: float | None = None
    first_segment_at: float | None = None
    finished_at: float | None = None
    token_frames: int = 0  # sum of generated codebook-frame counts, all segments
    segments: int = 0
    audio_seconds: float = 0.0
    #: Wall time inside the codec, i.e. turning code frames into samples. It
    #: shares the GPU thread with token generation, so every millisecond here
    #: is a millisecond the transformer is not sampling - which is what makes
    #: it worth separating from the rest of `finished_at - started_at`.
    decode_seconds: float = 0.0
    cancelled: bool = False
    error: str | None = None

    @property
    def ttfa_ms(self) -> float | None:
        """Time from job start to first decoded segment - the number that
        matters for perceived latency, since audio starts playing then."""
        if self.started_at is None or self.first_segment_at is None:
            return None
        return (self.first_segment_at - self.started_at) * 1000.0


@dataclass(slots=True)
class _Job:
    text: str
    params: "TTSParams"
    cancel: CancelToken
    out: "queue.Queue[object]"
    metrics: JobMetrics = field(default_factory=lambda: JobMetrics(queued_at=time.perf_counter()))


@dataclass(slots=True)
class _WarmJob:
    """Prefill a voice's system prompt and stop there.

    Goes through the same queue as synthesis so it cannot race a job that is
    mid-generation for the KV cache both of them write.
    """

    params: "TTSParams"
    cancel: CancelToken
    out: "queue.Queue[object]"


@dataclass(slots=True)
class _HintJob:
    """Prefill the prompt for text the client is still sending.

    Shares the job queue with synthesis, which is what makes it safe: the KV
    buffer has one writer at a time, so a hint can never land in the middle of
    a generation that is filling positions past the prompt.
    """

    text: str
    params: "TTSParams"
    cancel: CancelToken
    out: "queue.Queue[object]"


@dataclass(slots=True)
class WarmedMessage:
    prefix_tokens: int
    elapsed_ms: float


@dataclass(slots=True)
class ChunkMessage:
    chunk: AudioChunk


@dataclass(slots=True)
class DoneMessage:
    metrics: JobMetrics


@dataclass(slots=True)
class ErrorMessage:
    error: BaseException


@contextlib.contextmanager
def _skip_random_init(bnb4: bool):
    """Build the model without filling 4B parameters with random numbers first.

    The fork constructs the whole model in fp32 on the CPU, initialises every
    weight, then overwrites all of it from the checkpoint - and with a
    prequantized checkpoint it also copies each fp32 Linear into a Linear4bit
    whose weight is replaced moments later. On a 16 GB machine that is ~16 GB
    of pages touched for nothing, and loading took five minutes of swapping.
    Left untouched, the allocations are never committed to physical memory.
    Every key is loaded from the checkpoint (the fork logs "All keys matched"),
    so nothing is left uninitialised.
    """
    import torch.nn as nn
    from fish_speech.models.text2semantic import llama

    saved = (
        nn.Linear.reset_parameters,
        nn.Embedding.reset_parameters,
        llama.BaseTransformer._init_weights,
        llama._convert_linear_layers_to_bnb4,
    )

    def convert(module, compute_dtype=None):
        import bitsandbytes as bnb

        for name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                convert(child, compute_dtype)
                continue
            quantized = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=compute_dtype,
                quant_type="nf4",
                compress_statistics=True,
            )
            if child.bias is not None:  # the weight is replaced from the checkpoint
                quantized.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, quantized)

    nn.Linear.reset_parameters = lambda self: None
    nn.Embedding.reset_parameters = lambda self: None
    llama.BaseTransformer._init_weights = lambda self, module: None
    if bnb4:
        llama._convert_linear_layers_to_bnb4 = convert
    try:
        yield
    finally:
        (
            nn.Linear.reset_parameters,
            nn.Embedding.reset_parameters,
            llama.BaseTransformer._init_weights,
            llama._convert_linear_layers_to_bnb4,
        ) = saved


class GpuWorker:
    """Owns the model; everything else talks to it through `submit()`."""

    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._jobs: "queue.Queue[object]" = queue.Queue()
        self._ready = threading.Event()
        self._load_error: list[BaseException] = []

        self._model = None
        self._decode_one_token: Callable | None = None
        self._codec = None
        self._references = None
        self._device = settings.device
        self._precision = None

        self.sample_rate = 0
        self._current_cancel: CancelToken | None = None
        self._codec_precision = None
        self._codec_stream = None
        self._streamer = None
        self._emit_schedule: list[int] = []
        # At most one speculative prefill is ever queued; a newer one replaces
        # the pending request rather than joining it. See `submit_hint`.
        self._hint_lock = threading.Lock()
        self._pending_hint: "_HintJob | None" = None
        self._hint_queued = False
        self._wake_until = 0.0
        self._wake_queued = False
        self._wake_buf = None

        self._thread = threading.Thread(target=self._run, name="fish-gpu-worker", daemon=True)

    # --- lifecycle, called from fish.py ---

    def start(self) -> None:
        self._thread.start()

    def wait_ready(self, timeout: float | None = None) -> None:
        """Block until load succeeds or fails; raise on failure.

        Without this, a load that raises inside the thread would otherwise
        leave a caller blocked on `submit()` forever with no error to report -
        exactly the failure mode `launch_thread_safe_queue` avoids with its
        own init_event/init_error pair.
        """
        ready = self._ready.wait(timeout)
        if not ready:
            raise TimeoutError("FishSpeech worker did not finish loading in time")
        if self._load_error:
            raise self._load_error[0]

    @property
    def loaded(self) -> bool:
        return self._ready.is_set() and not self._load_error

    @property
    def queue_depth(self) -> int:
        return self._jobs.qsize()

    def submit(self, text: str, params: "TTSParams", cancel: CancelToken) -> "queue.Queue[object]":
        """Enqueue one segment; returns the queue its results arrive on.

        The result queue is private to this job (freshly created here), so
        concurrent callers never see each other's messages even though every
        job funnels through the same single worker thread.
        """
        out: "queue.Queue[object]" = queue.Queue()
        self._jobs.put(_Job(text=text, params=params, cancel=cancel, out=out))
        return out

    def submit_hint(
        self, text: str, params: "TTSParams", cancel: CancelToken
    ) -> "queue.Queue[object]":
        """Ask for a speculative prefill, replacing any request not yet started.

        An LLM emits words faster than this card prefills them, so a hint per
        word would queue work that is obsolete before it runs - and the real
        segment would wait behind every one of it. Measured when they were
        allowed to accumulate: 7.5-10.8s to first audio against 0.43s.

        So there is one slot, not a queue. A newer request overwrites the one
        waiting, which is strictly what you want: it is the same prompt plus
        more text, so the older one was going to be superseded anyway. The
        replaced caller is answered immediately rather than left waiting, since
        it is holding an event-loop executor thread until it hears back.
        """
        out: "queue.Queue[object]" = queue.Queue()
        job = _HintJob(text=text, params=params, cancel=cancel, out=out)
        with self._hint_lock:
            superseded, self._pending_hint = self._pending_hint, job
            queued, self._hint_queued = self._hint_queued, True
        if superseded is not None:
            superseded.out.put(WarmedMessage(0, 0.0))
        if not queued:
            self._jobs.put(_HINT_MARKER)
        return out

    def wake(self) -> None:
        """Bring the GPU up to speed before work arrives. Cheap to call often.

        A laptop GPU left idle for a second or two drops to its lowest clocks,
        and the first request after that pays for the climb back - measured,
        first audio went from ~60 ms to 200-400 ms. Calling this when a client
        starts a turn spends the time its LLM takes to produce the first
        sentence keeping the card busy with a tiny kernel, which is abandoned
        the moment real work is queued.
        """
        with self._hint_lock:
            self._wake_until = time.perf_counter() + WAKE_SECONDS
            queued, self._wake_queued = self._wake_queued, True
        if not queued:
            self._jobs.put(_WAKE_MARKER)

    def _handle_wake(self) -> None:
        import torch

        with self._hint_lock:
            self._wake_queued = False
            until = self._wake_until
        if self._wake_buf is None:
            self._wake_buf = torch.zeros((512, 512), dtype=torch.float16, device=self._device)
        buf = self._wake_buf
        # Back to back rather than paced: the clock governor reacts to
        # utilisation, and each burst is short enough (well under a
        # millisecond) that a job queued meanwhile waits for at most one.
        while time.perf_counter() < until and self._jobs.empty():
            torch.mm(buf, buf)
            torch.cuda.synchronize()

    def submit_warm(self, params: "TTSParams", cancel: CancelToken) -> "queue.Queue[object]":
        """Enqueue a prefix warm-up; results arrive on the returned queue."""
        out: "queue.Queue[object]" = queue.Queue()
        self._jobs.put(_WarmJob(params=params, cancel=cancel, out=out))
        return out

    def shutdown(self) -> None:
        self._jobs.put(_SENTINEL)
        # Unstick a job that's mid-generation so shutdown doesn't wait out a
        # full decode; harmless no-op if nothing is in flight.
        if self._current_cancel is not None:
            self._current_cancel.cancel()
        self._thread.join(timeout=10.0)
        self._release_gpu_state()

    def _release_gpu_state(self) -> None:
        self._model = None
        self._decode_one_token = None
        self._codec = None
        self._references = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass

    # --- the thread body ---

    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:  # noqa: BLE001 - re-raised via wait_ready()
            logger.error("FishSpeech worker failed to load: %s", traceback.format_exc())
            self._load_error.append(exc)
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._jobs.get()
            if item is _SENTINEL:
                break
            if isinstance(item, _WarmJob):
                self._handle_warm(item)  # never raises; failures go out on job.out
            elif item is _WAKE_MARKER:
                self._handle_wake()
            elif item is _HINT_MARKER:
                with self._hint_lock:
                    hint, self._pending_hint = self._pending_hint, None
                    self._hint_queued = False
                if hint is not None:
                    self._handle_hint(hint)  # never raises; failures go out on job.out
            else:
                self._handle(item)  # never raises; failures go out on job.out

    def _resolve_codec_precision(self):
        """fp16 codec decode is silently wrong on pre-Ampere hardware.

        Measured on a GTX 1650 (sm_75): fp16 codec weights, and fp32 weights
        under an fp16 autocast, both produce 100% NaN samples, which reach the
        client as digital silence rather than an error. fp32 without autocast
        is clean. Ampere and later are fine in fp16, so this only costs memory
        on the cards that need it.
        """
        import torch

        choice = (self._settings.codec_precision or "auto").lower()
        if choice == "auto":
            major = 0
            if torch.cuda.is_available() and torch.device(self._settings.device).type == "cuda":
                major = torch.cuda.get_device_capability()[0]
            cap = (
                torch.cuda.get_device_capability()
                if torch.cuda.is_available()
                else (0, 0)
            )
            choice = "fp16" if cap[0] >= 8 else "fp32"
            logger.info(
                "codec precision auto-selected: %s (sm_%d%d)", choice, cap[0], cap[1]
            )
        if choice not in ("fp16", "fp32"):
            raise RuntimeError(f"TTS_CODEC_PRECISION must be auto/fp16/fp32, got {choice!r}")
        return torch.float16 if choice == "fp16" else torch.float32

    def _apply_memory_profile(self) -> None:
        """Export the fork's FISH_* switches from Settings.

        Without `FISH_OFFLOAD_EMBEDDINGS` the 155776x2560 fp16 embedding table
        (~0.72 GiB) lands in VRAM and a 4 GB card runs out while still moving
        weights onto the device -- before the KV cache is even allocated, which
        is why lowering max_seq_len does not rescue that failure.
        """
        import os

        settings = self._settings
        if settings.offload_embeddings and settings.full_lm_head:
            raise RuntimeError(
                "TTS_OFFLOAD_EMBEDDINGS requires the sliced lm_head; "
                "unset TTS_FULL_LM_HEAD or disable embedding offload"
            )

        os.environ["FISH_OFFLOAD_EMBEDDINGS"] = "1" if settings.offload_embeddings else "0"
        os.environ["FISH_FULL_LM_HEAD"] = "1" if settings.full_lm_head else "0"
        os.environ["FISH_DECODE_CHUNK_FRAMES"] = str(settings.decode_chunk_frames)
        os.environ["FISH_DECODE_OVERLAP_FRAMES"] = str(settings.decode_overlap_frames)
        logger.info(
            "memory profile: offload_embeddings=%s full_lm_head=%s "
            "codec_decode_only=%s decode_chunk_frames=%d max_seq_len=%d",
            settings.offload_embeddings,
            settings.full_lm_head,
            settings.codec_decode_only,
            settings.decode_chunk_frames,
            settings.max_seq_len,
        )

    def _load(self) -> None:
        import torch

        settings = self._settings
        device = settings.device
        precision = torch.float16 if settings.half else torch.bfloat16

        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            # On Windows the driver does not fail an allocation that exceeds
            # VRAM, it pages it to system memory - measured: new allocations
            # past ~3.8 GB on a 4 GB card are written at 6.5 GB/s instead of
            # 105, and the codec slowed from ~40 ms to 2.5 s a call with no
            # error anywhere. Capping the allocator makes it free its cache
            # and retry instead, which is merely a little slower.
            # Headroom is for what the allocator does not see: the CUDA
            # context, cuDNN, graph executables - ~320 MiB measured with this
            # server's graphs - plus a margin below where paging starts.
            index = torch.device(device).index or 0
            total = torch.cuda.get_device_properties(index).total_memory
            headroom = settings.vram_headroom_mb * 2**20
            if 0 < headroom < total:
                limit = total - headroom
                torch.cuda.set_per_process_memory_fraction(limit / total, index)
                logger.info("VRAM cap for this process: %.0f MiB", limit / 2**20)

        # The fork reads its memory-profile switches straight from the
        # environment inside init_model, so they have to be set before the
        # call rather than passed as arguments. Driving them from Settings
        # keeps one source of truth instead of two.
        self._apply_memory_profile()

        # fish_speech only exists once app.engine.factory has put vendor_dir
        # on sys.path; importing it at module scope would make merely
        # `import app.engine.worker` require that to already be true.
        from fish_speech.models.dac.inference import load_model as load_codec
        from fish_speech.models.text2semantic.inference import init_model

        from app.engine.references import ReferenceManager

        # Must land before the model is built so every Attention instance picks
        # the patched forward up; it patches the class, not the instances.
        # Not optional: `app/engine/streamgen.py` drives the model through
        # `kv_window` and `prompt_embeddings`, and both are defined by this
        # patch. Running the fork's own attention against a narrowed mask, or
        # its own embedding lookup against a one-position prompt, does not fall
        # back gracefully - it mismatches shapes or reads the wrong rows.
        from app.engine.fastpath import install as install_fastpath

        install_fastpath()

        with _skip_random_init(settings.bnb4):
            model, decode_one_token = init_model(
                checkpoint_path=str(settings.checkpoint_dir),
                device=device,
                precision=precision,
                compile=settings.compile_model,
                max_length=settings.max_seq_len,
                bnb4=settings.bnb4,
            )
        if torch.device(device).type == "cuda" and settings.host_codebook_embeddings:
            from app.engine.hostembed import install as install_hostembed

            install_hostembed(model)

        # Every millisecond of a frame is inside one of this model's 76 4-bit
        # linear layers, and on this card bitsandbytes reads their weights at a
        # quarter of what the memory bus can do. Routing the single-vector case
        # through our own kernel roughly halves the per-frame cost; if it
        # cannot be built it says so and bitsandbytes serves every call as
        # before.
        if settings.bnb4 and settings.fast_gemv:
            from app.engine.nf4gemv import install as install_gemv

            install_gemv(model)

        # What is left of a frame once the matmuls are fast is 235 RMS norms
        # and 152 rotary embeddings, each written as six to nine separate
        # elementwise operations on a few kilobytes. One kernel each.
        if settings.fused_ops:
            from app.engine.fusedops import install as install_fused

            install_fused()
            # A decode position through a whole block in nine kernels
            # (app/engine/fusedlayer.py); needs the 4-bit kernel for its matmuls.
            if settings.bnb4 and settings.fast_gemv:
                from app.engine.fusedlayer import install as install_block

                install_block()

        with torch.device(device):
            model.setup_caches(
                max_batch_size=1,
                max_seq_len=model.config.max_seq_len,
                dtype=next(model.parameters()).dtype,
            )

        codec_precision = self._resolve_codec_precision()
        self._codec_precision = codec_precision
        codec = load_codec(
            config_name="modded_dac_vq",
            checkpoint_path=str(settings.checkpoint_dir / "codec.pth"),
            device=device,
            decode_only=settings.codec_decode_only,
            # precision casting is only valid alongside decode_only=True (see
            # fish_speech/models/dac/inference.py:load_model docstring); the
            # full encode-capable codec stays fp32. Passing None leaves the
            # weights fp32, which is what pre-Ampere cards need.
            precision=(
                codec_precision
                if (settings.codec_decode_only and codec_precision is torch.float16)
                else None
            ),
        )

        self._model = model
        self._decode_one_token = self._wrap_cancellable(decode_one_token)
        self._codec = codec
        self._device = device
        self._precision = precision
        self.sample_rate = codec.sample_rate
        self._references = ReferenceManager(settings, codec, device)

        from app.engine.streamgen import StreamingGenerator, default_emit_schedule

        graphs = None
        if settings.cuda_graphs and torch.device(device).type == "cuda":
            from app.engine.graphs import DecodeGraphs

            bias = StreamingGenerator(model, None, device, settings)._ensure_bias()[0]
            from app.engine import fusedlayer

            # The fused block's attention reads only up to the live position
            # and its chunks past it exit at once, so one graph over the whole
            # cache costs what a tight window would. Without it, attention
            # really does cover the window, and windows are bucketed.
            step = model.max_seq_len if fusedlayer.stats().get("ready") else 128
            graphs = DecodeGraphs(model, device, bias, step=step)
            graphs.capture_all()
        self._streamer = StreamingGenerator(
            model, self._decode_one_token, device, settings, graphs=graphs
        )
        self._emit_schedule = default_emit_schedule(
            settings.stream_first_frames, settings.stream_max_frames
        )
        self._codec_stream = None
        if graphs is not None:
            from app.engine.codecstream import StreamingCodec

            sizes = sorted({1 << i for i in range(max(self._emit_schedule).bit_length())})
            self._codec_stream = StreamingCodec(codec, max_frames=settings.max_seq_len, sizes=sizes)
            self._codec_stream.warm(model.config.num_codebooks)
            torch.cuda.empty_cache()

        logger.info(
            "FishSpeech worker loaded (device=%s precision=%s bnb4=%s codec_decode_only=%s)",
            device,
            precision,
            settings.bnb4,
            settings.codec_decode_only,
        )

    def _wrap_cancellable(self, decode_one_token: Callable) -> Callable:
        """Check the in-flight job's cancel token before every sampled token.

        `self._current_cancel` is set in `_handle` right before generation
        starts for that job and cleared in its `finally`; the worker
        processes exactly one job at a time, so this always polls the token
        for whichever job is actually running.
        """

        def _checked(*args, **kwargs):
            token = self._current_cancel
            if token is not None:
                token.raise_if_cancelled()
            return decode_one_token(*args, **kwargs)

        return _checked

    def _handle(self, job: _Job) -> None:
        metrics = job.metrics
        metrics.started_at = time.perf_counter()
        self._current_cancel = job.cancel
        try:
            job.cancel.raise_if_cancelled()  # cancelled before it was even dequeued

            prompt_tokens, prompt_texts = self._references.resolve(
                job.params.references, job.params.reference_id
            )
            job.cancel.raise_if_cancelled()

            if job.params.seed is not None:
                from fish_speech.utils import set_seed

                set_seed(job.params.seed)

            decoder = self._make_decoder()

            def emit(samples, frames: int) -> None:
                if samples is None or samples.size == 0:
                    return
                chunk = AudioChunk(samples=samples, sample_rate=self.sample_rate)
                metrics.segments += 1
                metrics.token_frames += frames
                metrics.audio_seconds += chunk.duration_s
                if metrics.first_segment_at is None:
                    metrics.first_segment_at = time.perf_counter()
                job.out.put(ChunkMessage(chunk))

            # Audio is collected one or two blocks behind the codec, so the
            # GPU is never idle waiting for the host to take it - except for
            # the first block, which is what the listener is waiting on.
            waiting: deque = deque()
            for block in self._stream_blocks(job, prompt_texts, prompt_tokens):
                job.cancel.raise_if_cancelled()
                decode_t0 = time.perf_counter()
                decoder.extend(block)
                pending = decoder.drain_async()
                if pending is not None:
                    if metrics.first_segment_at is None:
                        emit(pending.result(), int(block.shape[-1]))
                    else:
                        waiting.append((pending, int(block.shape[-1])))
                while waiting and waiting[0][0].ready():
                    done, frames = waiting.popleft()
                    emit(done.result(), frames)
                metrics.decode_seconds += time.perf_counter() - decode_t0
            while waiting:
                done, frames = waiting.popleft()
                emit(done.result(), frames)

            metrics.finished_at = time.perf_counter()
            self._log_job(job, "ok")
            job.out.put(DoneMessage(metrics))

        except Cancelled:
            # No cleanup needed beyond this: the KV cache is sized once at
            # load time and never cleared between jobs by design (see
            # `_load`); a cancelled job's half-written cache entries live at
            # positions [0, however-far-it-got), and the next job overwrites
            # exactly those positions from its own position 0 as it runs its
            # own prefill and decode - a job never reads cache content it
            # did not itself just write. So "dirty" leftovers are inert, not
            # poisoned, and the model is immediately ready for the next job.
            metrics.cancelled = True
            metrics.finished_at = time.perf_counter()
            job.out.put(DoneMessage(metrics))

        except Exception as exc:  # noqa: BLE001 - reported, not raised, in this thread
            logger.error("FishSpeech job failed: %s", traceback.format_exc())
            metrics.error = str(exc)
            metrics.finished_at = time.perf_counter()
            job.out.put(ErrorMessage(exc))

        finally:
            self._current_cancel = None

    def _log_job(self, job: _Job, outcome: str) -> None:
        """One line per segment, at DEBUG, accounting for every millisecond.

        A turn that breaks up does so in the joins between segments as often as
        inside one, and from the outside the two look identical: audio simply
        stops arriving. `queue_ms` separates them - it is how long this segment
        sat behind other work on the single GPU thread (a hint, a voice warm-up,
        the previous segment) before it started, which is time no amount of
        faster generation would recover.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        m = job.metrics
        started = m.started_at or m.queued_at
        finished = m.finished_at or time.perf_counter()
        audio = m.audio_seconds or 0.0
        logger.debug(
            "job %s: queue_ms=%.0f ttfa_ms=%s gen_ms=%.0f codec_ms=%.0f (%.0f%%) "
            "audio_s=%.2f rtf=%.2f frames=%d blocks=%d ms_per_frame=%.1f chars=%d",
            outcome,
            (started - m.queued_at) * 1000.0,
            f"{m.ttfa_ms:.0f}" if m.ttfa_ms is not None else "-",
            (finished - started) * 1000.0,
            m.decode_seconds * 1000.0,
            100.0 * m.decode_seconds / (finished - started) if finished > started else 0.0,
            audio,
            ((finished - started) / audio) if audio else float("nan"),
            m.token_frames,
            m.segments,
            (finished - started) * 1000.0 / m.token_frames if m.token_frames else float("nan"),
            len(job.text),
        )

    def _handle_warm(self, job: "_WarmJob") -> None:
        t0 = time.perf_counter()
        self._current_cancel = job.cancel
        try:
            prompt_tokens, prompt_texts = self._references.resolve(
                job.params.references, job.params.reference_id
            )
            tokens = self._streamer.warm(prompt_texts, prompt_tokens, job.cancel)
            job.out.put(WarmedMessage(tokens, (time.perf_counter() - t0) * 1000.0))
        except Cancelled:
            job.out.put(WarmedMessage(0, (time.perf_counter() - t0) * 1000.0))
        except Exception as exc:  # noqa: BLE001 - a failed warm-up must not kill the worker
            logger.warning("voice prefix warm-up failed: %s", exc, exc_info=True)
            job.out.put(ErrorMessage(exc))
        finally:
            self._current_cancel = None

    def _handle_hint(self, job: "_HintJob") -> None:
        t0 = time.perf_counter()
        self._current_cancel = job.cancel
        try:
            prompt_tokens, prompt_texts = self._references.resolve(
                job.params.references, job.params.reference_id
            )
            tokens = self._streamer.hint(job.text, prompt_texts, prompt_tokens, job.cancel)
            logger.debug(
                "hint: prefilled=%d tokens in %.0f ms for %d chars",
                tokens,
                (time.perf_counter() - t0) * 1000.0,
                len(job.text),
            )
            job.out.put(WarmedMessage(tokens, (time.perf_counter() - t0) * 1000.0))
        except Cancelled:
            job.out.put(WarmedMessage(0, (time.perf_counter() - t0) * 1000.0))
        except Exception as exc:  # noqa: BLE001 - a hint is an optimisation, never a failure
            logger.debug("prompt hint failed (harmless): %s", exc)
            job.out.put(WarmedMessage(0, (time.perf_counter() - t0) * 1000.0))
        finally:
            self._current_cancel = None

    def prefix_stats(self) -> dict:
        streamer = self._streamer
        return {} if streamer is None else streamer.stats.as_dict()

    def kernel_stats(self) -> dict:
        """Hand-written kernel coverage, for `/status`.

        Imported lazily and defensively: neither module is required for the
        server to work, and a build that failed has already said so in the log.
        """
        out: dict = {}
        try:
            from app.engine import nf4gemv

            out["nf4_gemv"] = nf4gemv.stats()
        except Exception:  # noqa: BLE001 - status must never raise
            pass
        try:
            from app.engine import fusedops

            out["fused_ops"] = fusedops.stats()
        except Exception:  # noqa: BLE001
            pass
        return out

    def _make_decoder(self):
        """The per-job incremental codec decoder. A seam: unit tests stand in
        for it so the job loop can be exercised with no GPU and no codec."""
        from app.engine.codecstream import IncrementalDecoder

        return IncrementalDecoder(
            self._codec,
            lookback_frames=self._settings.codec_lookback_frames,
            post_context_frames=self._settings.codec_post_context_frames,
            use_autocast=self._use_codec_autocast(),
            stream=self._codec_stream,
        )

    def _stream_blocks(self, job: _Job, prompt_texts, prompt_tokens):
        """Blocks of code frames as they are sampled. The other test seam."""
        return self._streamer.generate(
            job.text,
            job.params,
            prompt_texts,
            prompt_tokens,
            job.cancel,
            self._emit_schedule,
        )

    def _use_codec_autocast(self) -> bool:
        """fp16 autocast is only correct when the codec weights really are fp16;
        see `_resolve_codec_precision` for why that is not a given here."""
        import torch

        return getattr(self, "_codec_precision", torch.float32) is torch.float16
