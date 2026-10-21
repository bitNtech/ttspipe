"""Incremental codec decode: audio for the frames generated so far.

`DAC.from_indices` decodes a finished utterance. Streaming needs the opposite
shape - "here are 8 more frames, give me their audio now" - and the naive way
to get it, calling `from_indices` on each slice in turn, is audibly wrong.
Measured on 220 random frames against a single full pass: 1.6 dB SNR with no
left context, and still only 6.9 dB with 32 frames of it.

The reason is that decoding is two halves with very different needs, and
`rvq.decode_codes` documents both:

  * `quantizer.decode_codes` (codebook lookup + `post_module`) is 8 stacked
    window-limited attention layers reaching back ~1000 frames. Slicing its
    *input* is what destroyed the SNR above. It is causal, though - a prefix's
    output matches the full sequence's to 1.7e-7 relative - so re-running it
    over everything generated so far returns stable values for the frames
    already emitted. It is also the cheap half, running at the 21.5 Hz token
    rate: ~31 ms even for a long utterance, and `post_context_frames` caps
    that by exploiting the 512-frame attention window.

  * `quantizer.upsample` + `decoder` is causal convolution with a receptive
    field of 10 frames. Feeding it a few frames of real left context and
    discarding their audio reproduces a full pass exactly - the same argument
    `DAC._decode_chunked` already relies on.

Doing it that way measures 109 dB SNR at `lookback_frames=10` and above, which
is the numerical floor rather than an approximation. 4 frames of lookback
gives 43 dB, which is why the default is not tuned below 16.

One instance per utterance; `extend()` must be called with frames in order.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

#: Samples of audio per token frame (quantizer.upsample x4, decoder x512).
SAMPLES_PER_FRAME = 2048


class PendingAudio:
    """Samples on their way to the host; `result()` waits for them."""

    def __init__(self, waveform) -> None:
        import torch

        self._device = waveform  # keeps the source alive until the copy lands
        if waveform.device.type == "cuda":
            self._host = torch.empty(waveform.shape, dtype=torch.float32, pin_memory=True)
            self._host.copy_(waveform, non_blocking=True)
            self._event = torch.cuda.Event()
            self._event.record()
        else:
            self._host, self._event = waveform, None

    def ready(self) -> bool:
        return self._event is None or self._event.query()

    def result(self) -> "np.ndarray":
        if self._event is not None:
            self._event.synchronize()
        samples = self._host.numpy().astype(np.float32, copy=True)
        self._device = None
        if not np.isfinite(samples).all():
            raise RuntimeError(
                "codec decode produced non-finite samples; the codec precision "
                "is wrong for this GPU (try TTS_CODEC_PRECISION=fp32)"
            )
        return samples


class IncrementalDecoder:
    """Turns a growing sequence of code frames into contiguous audio."""

    def __init__(
        self,
        codec,
        *,
        lookback_frames: int = 16,
        post_context_frames: int = 512,
        use_autocast: bool = False,
        stream: "StreamingCodec | None" = None,
    ) -> None:
        self._codec = codec
        # Carries state across calls, so it is only valid while every frame
        # goes through it in order; the first span it cannot take ends its use
        # for this utterance and the recomputing path below takes over.
        self._stream = stream
        if stream is not None:
            stream.reset()
        self._quantizer = codec.quantizer
        self._lookback = max(int(lookback_frames), 10)
        self._post_context = max(int(post_context_frames), 256)
        self._use_autocast = use_autocast

        self._codes: "torch.Tensor | None" = None  # (1, num_codebooks, T) on device
        self._emitted = 0  # frames already returned as audio

    @property
    def emitted_frames(self) -> int:
        return self._emitted

    @property
    def pending_frames(self) -> int:
        return 0 if self._codes is None else self._codes.shape[-1] - self._emitted

    def extend(self, codes: "torch.Tensor") -> None:
        """Append newly generated frames, shape (num_codebooks, n)."""
        import torch

        block = codes if codes.dim() == 3 else codes[None]
        block = block.to(dtype=torch.long)
        self._codes = block if self._codes is None else torch.cat([self._codes, block], dim=-1)

    def drain(self) -> "np.ndarray | None":
        """Decode every frame appended but not yet returned. None if there are none."""
        pending = self.drain_async()
        return None if pending is None else pending.result()

    def drain_async(self) -> "PendingAudio | None":
        """`drain`, without waiting: the samples reach the host behind the GPU.

        Waiting for them synchronises the whole stream - including generation
        queued after the codec - and the GPU then sits idle while the host
        catches up. Collecting the result later keeps it fed.
        """
        import torch

        if self._codes is None:
            return None
        total = self._codes.shape[-1]
        start = self._emitted
        if total <= start:
            return None

        device_type = self._codes.device.type
        with torch.inference_mode():
            if self._use_autocast:
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    waveform = self._decode_span(start, total)
            else:
                waveform = self._decode_span(start, total)

        self._emitted = total
        return PendingAudio(waveform.float().contiguous())

    def _decode_span(self, start: int, end: int) -> "torch.Tensor":
        if self._stream is not None:
            waveform = self._stream.decode(self._codes, start, end)
            if waveform is not None:
                return waveform
            self._stream = None
        # post_module is causal but window-limited, so it only ever needs the
        # most recent `post_context` frames to produce identical values here.
        post_start = max(0, end - self._post_context)
        # decode_codes clamps its argument in place, so it never sees ours.
        z_post = self._quantizer.decode_codes(self._codes[..., post_start:end].clone())

        conv_start = max(post_start, start - self._lookback)
        context = start - conv_start
        piece = z_post[..., conv_start - post_start : end - post_start]
        audio = self._codec.decoder(self._quantizer.upsample(piece))
        return audio[0].squeeze()[context * SAMPLES_PER_FRAME :]


_SNAKE_SOURCE = r"""
extern "C" __global__ void snake(const half_t* __restrict__ x, const half_t* __restrict__ alpha,
                                 half_t* __restrict__ y, long long n, int C, int T) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = h2f(alpha[(int)((i / T) % C)]);
  const float v = h2f(x[i]);
  const float s = __sinf(a * v);
  y[i] = f2h(v + s * s / (a + 1e-9f));
}
"""


def _install_fused_snake() -> None:
    """The DAC decoder's Snake activation as one kernel, for fp16 inputs.

    `x + sin(alpha * x)^2 / alpha` is several elementwise kernels as written,
    each a full pass over activations that run to 96 channels x 88k samples a
    block. Fused it is one read and one write, in fp32 arithmetic.
    """
    import ctypes

    import dac.nn.layers as layers
    import torch

    from app.engine import nvrtc

    if getattr(layers, "_fused_snake", False):
        return
    try:
        kernel = nvrtc.build(
            nvrtc.PRELUDE + _SNAKE_SOURCE, ["snake"], fast_math=False, label="snake"
        )["snake"]
    except Exception as exc:  # noqa: BLE001 - the scripted original still works
        logger.warning("fused snake unavailable (%s)", exc)
        return
    original = layers.snake

    def snake(x, alpha):
        if (
            x.dtype != torch.float16
            or alpha.dtype != torch.float16
            or x.dim() != 3
            or not x.is_contiguous()
            or x.device.type != "cuda"
        ):
            return original(x, alpha)
        y = torch.empty_like(x)
        n = x.numel()
        argv, _keep = nvrtc.pack(
            [
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(alpha.data_ptr()),
                ctypes.c_void_p(y.data_ptr()),
                ctypes.c_longlong(n),
                ctypes.c_int(x.shape[1]),
                ctypes.c_int(x.shape[2]),
            ]
        )
        nvrtc.launch(kernel, (n + 255) // 256, 256, argv)
        return y

    layers.snake = snake
    layers._fused_snake = True


class StreamingCodec:
    """The codec decoder run as a stream: each call decodes only new frames.

    Every layer below the VQ lookup is causal, so the output for frames
    [start, end) depends only on frames before `end` - which means the state
    those frames need can be *carried* from call to call instead of being
    recomputed from a lookback:

    - each causal convolution keeps the last `padding` samples of its input,
      which is exactly the zero padding a single full pass would have seen,
      replaced by real history;
    - each transposed convolution keeps the input frames that overlap into
      the next block's output;
    - `post_module`'s eight window-limited attention layers keep a KV cache,
      at the absolute positions a full pass uses.

    So a call costs what its own frames cost, with no re-run of the last 512
    frames through `post_module` and no convolution lookback - the two fixed
    costs that made even a 1-frame block ~45 ms of GPU time and forced blocks
    to grow to 64 frames. With those gone a block can be a few frames, which
    is what lets a stream run close to realtime without the client having to
    buffer a whole large block first.

    Each block size is one captured CUDA graph; any span decomposes into
    powers of two, and the result is the same either way.
    """

    def __init__(self, codec, *, max_frames: int, sizes=(1, 2, 4)) -> None:
        import torch
        _install_fused_snake()
        _install_streaming_convs()
        self._codec = codec
        self._quantizer = codec.quantizer
        self._post = codec.quantizer.post_module
        self._sizes = tuple(sorted(sizes))
        self._graphs: dict[int, tuple] = {}
        # Shared by this instance's graphs: they replay one at a time, and all
        # state that outlives a replay lives outside the pool.
        self._pool = torch.cuda.graph_pool_handle()
        self._device = next(codec.parameters()).device
        self._dtype = next(codec.parameters()).dtype
        self._states: list = []
        for module in codec.modules():  # state belongs to one instance at a time
            if hasattr(module, "_stream_state"):
                del module._stream_state
        self._num_codebooks = None
        for module in list(codec.quantizer.upsample.modules()) + list(codec.decoder.modules()):
            if isinstance(module, _conv_classes()[0]) and module.stride != 1:
                raise RuntimeError("streaming decode assumes stride-1 causal convolutions")
        # cuDNN runs these convolutions NHWC in fp16 and, given NCW weights,
        # converts every weight tensor on every call - ~3 ms a call of pure
        # layout shuffling, measured, whatever the block size. Stored with
        # channels-last strides they are used as they are.
        for module in codec.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                weight = module.weight.data
                module.weight.data = (
                    weight.unsqueeze(2).contiguous(memory_format=torch.channels_last).squeeze(2)
                )

        cfg = self._post.config
        n_local = cfg.n_local_heads if cfg.n_local_heads > 0 else cfg.n_head
        head_dim = cfg.dim // cfg.n_head
        length = min(int(max_frames), cfg.block_size)
        if n_local != cfg.n_head:
            raise RuntimeError("streaming decode assumes post_module has no grouped heads")
        self._kv = [
            (
                torch.zeros((1, n_local, length, head_dim), dtype=self._dtype, device=self._device),
                torch.zeros((1, n_local, length, head_dim), dtype=self._dtype, device=self._device),
            )
            for _ in self._post.layers
        ]
        self._window = int(self._post.window_size or length)
        self._length = length

    @property
    def max_frames(self) -> int:
        return self._length

    # --- state ---

    def _conv_state(self, module, x):
        import torch

        state = getattr(module, "_stream_state", None)
        if state is None:
            if hasattr(module, "padding"):  # causal conv: its left padding
                width = module.padding
            else:  # transposed: input frames whose output spills into the next block
                width = -(-module.kernel_size // module.stride) - 1
            state = torch.zeros((x.shape[0], x.shape[1], width), dtype=x.dtype, device=x.device)
            module._stream_state = state
            self._states.append(state)
        return state

    def reset(self) -> None:
        """Start a new utterance: all carried state back to a full pass's zeros."""
        import torch

        with torch.inference_mode():  # the states were made inside it
            for state in self._states:
                state.zero_()

    def _run(self, codes, pos):
        """The captured body: codes (1, num_codebooks, n) at positions `pos`."""
        import torch

        q = self._quantizer
        indices = codes.clone()
        indices[:, 0] = torch.clamp(indices[:, 0], max=q.semantic_quantizer.codebook_size - 1)
        indices[:, 1:] = torch.clamp(indices[:, 1:], max=q.quantizer.codebook_size - 1)
        z = (
            q.semantic_quantizer.from_codes(indices[:, :1])[0]
            + q.quantizer.from_codes(indices[:, 1:])[0]
        )

        x = self._post_forward(z.transpose(1, 2) if self._post.channels_first else z, pos)
        z = x.transpose(1, 2) if self._post.channels_first else x

        _streaming.codec = self
        try:
            audio = self._codec.decoder(q.upsample(z))
        finally:
            _streaming.codec = None
        return audio[0].reshape(-1).float()

    def _post_forward(self, x, pos):
        """`post_module` over n new frames, attending through the carried KV cache.

        Each position sees the `window` positions ending at itself, so only
        those are gathered - `window + n - 1` keys at most - rather than
        attending over the whole cache with a mask. Heads are not grouped in
        this module, so there is no expansion to copy either.
        """
        import torch
        import torch.nn.functional as F
        from fish_speech.models.dac.modded_dac import apply_rotary_emb

        post = self._post
        n = x.shape[1]
        x = post.look_ahead_conv(post.input_proj(x))
        freqs = post.freqs_cis[pos]
        span = self._window + n - 1
        key_pos = pos[:1] - (self._window - 1) + torch.arange(span, device=x.device)
        index = key_pos.clamp(min=0)
        rows = pos.view(-1, 1)
        mask = (key_pos.view(1, -1) <= rows) & (key_pos.view(1, -1) > rows - self._window)
        mask = (mask & (key_pos >= 0).view(1, -1))[None, None]
        for layer, (kc, vc) in zip(post.layers, self._kv):
            attn = layer.attention
            h = layer.attention_norm(x)
            size = attn.n_local_heads * attn.head_dim
            q, k, v = attn.wqkv(h).split([size, size, size], dim=-1)
            q = apply_rotary_emb(q.view(1, n, attn.n_head, attn.head_dim), freqs).transpose(1, 2)
            k = apply_rotary_emb(k.view(1, n, attn.n_local_heads, attn.head_dim), freqs)
            kc[:, :, pos] = k.transpose(1, 2)
            vc[:, :, pos] = v.view(1, n, attn.n_local_heads, attn.head_dim).transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q, kc.index_select(2, index), vc.index_select(2, index), attn_mask=mask
            )
            y = attn.wo(y.transpose(1, 2).reshape(1, n, attn.n_head * attn.head_dim))
            x = x + layer.attention_layer_scale(y)
            x = x + layer.ffn_layer_scale(layer.feed_forward(layer.ffn_norm(x)))
        return post.output_proj(post.norm(x))

    def _capture(self, n: int) -> None:
        import torch

        codes = torch.zeros((1, self._num_codebooks, n), dtype=torch.long, device=self._device)
        pos = torch.arange(n, device=self._device)
        torch.cuda.empty_cache()
        with torch.inference_mode():
            self._run(codes, pos)  # creates the conv states, lets cuDNN pick algorithms
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._pool):
            out = self._run(codes, pos)
        self._graphs[n] = (graph, codes, pos, out)

    def warm(self, num_codebooks: int) -> None:
        """Capture one graph per block size. Call once, before first use."""
        import time

        t0 = time.perf_counter()
        self._num_codebooks = num_codebooks
        for n in self._sizes:
            self._capture(n)
        self.reset()
        logger.info(
            "captured %d streaming codec graphs in %.1fs",
            len(self._graphs),
            time.perf_counter() - t0,
        )

    def decode(self, codes, start: int, end: int):
        """Audio for frames [start, end) of `codes`, or None if it cannot.

        Frames before `start` must already have gone through this instance
        since the last `reset()`, in order.
        """
        import torch

        if end > self._length or not self._graphs:
            return None
        pieces = []
        with torch.inference_mode():
            while start < end:
                n = max(size for size in self._sizes if size <= end - start)
                graph, buf, pos, out = self._graphs[n]
                buf.copy_(codes[..., start : start + n])
                torch.arange(start, start + n, out=pos)
                graph.replay()
                pieces.append(out.clone())
                start += n
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


class _StreamingFlag:
    codec: "StreamingCodec | None" = None


_streaming = _StreamingFlag()


def _conv_classes():
    """The causal conv classes, from both modules that define their own copy.

    `rvq` (the quantizer's upsampler) and `modded_dac` (the decoder) each
    declare a `CausalConvNet` and a `CausalTransConvNet`; they are identical
    but distinct, and patching only one leaves half the stack unstreamed.
    """
    from fish_speech.models.dac import modded_dac, rvq

    return (
        (rvq.CausalConvNet, modded_dac.CausalConvNet),
        (rvq.CausalTransConvNet, modded_dac.CausalTransConvNet),
    )


def _install_streaming_convs() -> None:
    """Let causal convolutions carry state while a StreamingCodec is decoding."""
    import torch

    convs, transposed = _conv_classes()
    for conv_cls in convs:
        if getattr(conv_cls, "_streaming_patched", False):
            continue

        def conv(self, x, _original=conv_cls.forward):
            stream = _streaming.codec
            if stream is None:
                return _original(self, x)
            if self.padding == 0:
                return self.conv(x).contiguous()
            state = stream._conv_state(self, x)
            full = torch.cat([state, x], dim=-1)
            state.copy_(full[..., -self.padding :])
            return self.conv(full).contiguous()

        conv_cls.forward = conv
        conv_cls._streaming_patched = True

    for trans_cls in transposed:
        if getattr(trans_cls, "_streaming_patched", False):
            continue

        def trans(self, x, _original=trans_cls.forward):
            stream = _streaming.codec
            if stream is None:
                return _original(self, x)
            state = stream._conv_state(self, x)
            keep = state.shape[-1]
            full = torch.cat([state, x], dim=-1) if keep else x
            if keep:
                state.copy_(full[..., -keep:])
            y = self.conv(full)
            return y[..., keep * self.stride : (keep + x.shape[-1]) * self.stride].contiguous()

        trans_cls.forward = trans
        trans_cls._streaming_patched = True
