"""The two elementwise chains that surround every matmul, as one kernel each.

Once `app/engine/nf4gemv.py` has the 4-bit matmuls down to about two thirds of
a frame, what is left is not one expensive thing. It is 235 RMS norms and 152
rotary embeddings, none of which touches more than a few kilobytes, each
written in PyTorch as a chain of six to nine separate operations:

```python
def _norm(self, x):
    return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

def forward(self, x):
    return self._norm(x.float()).type_as(x) * self.weight
```

That is a cast, a multiply, a mean, an add, a reciprocal square root, a
multiply, a cast back and a multiply - eight kernels, eight launches, and seven
round trips to memory for a tensor of 2560 halves that fits in L1. `profile()`
counted 1393 `aten::mul` and 1075 `aten::copy_` calls in a single frame, for
3.5 ms of GPU time and 4.5 ms of CPU time that is almost entirely the cost of
asking rather than the cost of doing. Rotary embedding is the same shape of
problem: a cast, four multiplies, an add, a subtract, a `stack` and a cast.

Fusing each into one kernel replaces roughly 3200 launches a frame with 390.
The arithmetic is unchanged and done in the same precision the fork does it in
- fp32 for the normalisation, the result rounded to fp16 before the weight
multiply, exactly as `_norm(x.float()).type_as(x) * self.weight` does. Rotary
embedding, having no reduction in it, comes back bit for bit identical. The
norm sums its squares in this kernel's order rather than PyTorch's tree
reduction, so about three elements in a thousand round the other way; that is
the only difference, and `tests/test_fusedops.py` pins it against the fork's
own output at thresholds an fp16 accumulation or a dropped intermediate
rounding would fail.

Compiled without `--use_fast_math`, unlike the GEMV: everything downstream of a
norm is scaled by its reciprocal square root, which is not a place to accept an
approximate reciprocal for a percent of throughput.

Both patches are class-level and cover the depth transformer as well as the
slow one, which matters more than it sounds - 4 fast layers running ten times a
frame are 80 of the 152 rotary calls. Either can be turned off with
`TTS_FUSED_OPS=0`, and both fall back per call, not just at start-up: a shape
this does not handle goes to the fork's own code and comes back identical.
"""

from __future__ import annotations

import ctypes
import logging

from app.engine import nvrtc

logger = logging.getLogger(__name__)

_SOURCE = (
    nvrtc.PRELUDE
    + r"""
//: One block per row. Rows are 128 (a head) or 2560 (the model width) wide, so
//: a block never needs more than a few hundred threads, and the reduction is
//: one warp shuffle chain plus one across warps.
extern "C" __global__ void rms_norm(
    const half_t* __restrict__ X, const half_t* __restrict__ Wt,
    half_t* __restrict__ Y, int D, float eps)
{
  __shared__ float partial[32];
  const size_t base = (size_t)blockIdx.x * D;

  float acc = 0.f;
  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    const float v = h2f(X[base + i]);
    acc += v * v;
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = acc;
  __syncthreads();

  if (threadIdx.x < 32) {
    const int warps = (blockDim.x + 31) >> 5;
    float v = (threadIdx.x < warps) ? partial[threadIdx.x] : 0.f;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0) partial[0] = rsqrtf(v / (float)D + eps);
  }
  __syncthreads();
  const float scale = partial[0];

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    // Rounded to fp16 before the weight multiply, because `.type_as(x)` is
    // where the fork rounds it and the weight is fp16 too.
    const float normed = h2f(f2h(h2f(X[base + i]) * scale));
    Y[base + i] = f2h(normed * h2f(Wt[i]));
  }
}

//: `freqs_cis` is registered as bfloat16 and may have been cast to fp16 or
//: fp32 by a `.to()` on the model, so the element type travels as an argument
//: rather than being assumed. The branch is uniform across the grid.
#define F_FP32 0
#define F_FP16 1
#define F_BF16 2

extern "C" __global__ void rope(
    const half_t* __restrict__ X, const void* __restrict__ F,
    half_t* __restrict__ Y, int pairs, int half_dim, int heads, int seq, int ftype)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs) return;

  const int p = i % half_dim;
  const int s = (i / half_dim / heads) % seq;
  const int f = (s * half_dim + p) * 2;

  float fr, fi;
  if (ftype == F_FP32) {
    fr = ((const float*)F)[f]; fi = ((const float*)F)[f + 1];
  } else if (ftype == F_FP16) {
    fr = h2f(((const half_t*)F)[f]); fi = h2f(((const half_t*)F)[f + 1]);
  } else {
    fr = bf2f(((const half_t*)F)[f]); fi = bf2f(((const half_t*)F)[f + 1]);
  }

  // A pair is two adjacent elements of the last axis, so the pair index is the
  // element index halved - no reshape, no stack, no copy back.
  const float x0 = h2f(X[2 * i]);
  const float x1 = h2f(X[2 * i + 1]);
  // Spelled with the rounding intrinsics so the compiler cannot contract these
  // into fused multiply-adds. An FMA here is more accurate, not less, but the
  // fork writes two multiplies and a subtract and rounds between them, and the
  // point of this kernel is to be the same function rather than a better one.
  Y[2 * i]     = f2h(__fsub_rn(__fmul_rn(x0, fr), __fmul_rn(x1, fi)));
  Y[2 * i + 1] = f2h(__fadd_rn(__fmul_rn(x1, fr), __fmul_rn(x0, fi)));
}
"""
)

_kernels: dict | None = None
_failed = False
enabled = True

_stats = {"norm": 0, "norm_fallback": 0, "rope": 0, "rope_fallback": 0}

#: Threads per block for the norm. 256 covers a 2560-wide row in ten strides
#: and a 128-wide one in one, and keeps `partial` inside a single warp's worth
#: of slots.
NORM_THREADS = 256
ROPE_THREADS = 256

_FTYPE = {"torch.float32": 0, "torch.float16": 1, "torch.bfloat16": 2}


def stats() -> dict:
    return dict(_stats)


def _built():
    global _kernels, _failed
    if _kernels is not None or _failed:
        return _kernels
    try:
        _kernels = nvrtc.build(_SOURCE, ["rms_norm", "rope"], fast_math=False, label="fusedops")
        logger.info("fused rms_norm and rope kernels compiled")
    except Exception as exc:  # noqa: BLE001 - the fork's own code still works
        _failed = True
        logger.warning("fused ops unavailable (%s); using the fork's elementwise chains", exc)
    return _kernels


def rms_norm(x, weight, eps: float):
    """`x * rsqrt(mean(x*x) + eps) * weight` over the last axis, or None."""
    import torch

    kernels = _built()
    if kernels is None or not enabled:
        return None
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        return None
    if x.device.type != "cuda" or not x.is_contiguous() or not weight.is_contiguous():
        return None
    width = x.shape[-1]
    if width != weight.numel() or width == 0:
        return None

    out = torch.empty_like(x)
    argv, _keep = nvrtc.pack(
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(width),
            ctypes.c_float(eps),
        ]
    )
    nvrtc.launch(kernels["rms_norm"], x.numel() // width, NORM_THREADS, argv)
    _stats["norm"] += 1
    return out


def rope(x, freqs_cis):
    """Rotary embedding of `x` (B, S, H, D) against `freqs_cis` (S, D/2, 2)."""
    import torch

    kernels = _built()
    if kernels is None or not enabled:
        return None
    if x.dtype != torch.float16 or x.device.type != "cuda" or not x.is_contiguous():
        return None
    if not freqs_cis.is_contiguous() or x.dim() != 4:
        return None
    ftype = _FTYPE.get(str(freqs_cis.dtype))
    if ftype is None:
        return None
    bsz, seq, heads, head_dim = x.shape
    half_dim = head_dim // 2
    if head_dim % 2 or freqs_cis.numel() != seq * half_dim * 2:
        return None

    out = torch.empty_like(x)
    pairs = x.numel() // 2
    argv, _keep = nvrtc.pack(
        [
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(freqs_cis.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(pairs),
            ctypes.c_int(half_dim),
            ctypes.c_int(heads),
            ctypes.c_int(seq),
            ctypes.c_int(ftype),
        ]
    )
    nvrtc.launch(
        kernels["rope"], (pairs + ROPE_THREADS - 1) // ROPE_THREADS, ROPE_THREADS, argv
    )
    _stats["rope"] += 1
    return out


def note_rope_fallback() -> None:
    """Called by `app/engine/fastpath.py` when it had to use the fork's rotary."""
    _stats["rope_fallback"] += 1


def install() -> bool:
    """Route `RMSNorm.forward` through the fused kernel. Idempotent.

    Rotary is not patched here: `app/engine/fastpath.py` binds
    `apply_rotary_emb` into its own attention forward, so it calls `rope()`
    directly and this would not reach it.
    """
    from fish_speech.models.text2semantic.llama import RMSNorm

    if getattr(RMSNorm, "_fused_patched", False):
        return _built() is not None
    if _built() is None:
        return False

    original_forward = RMSNorm.forward

    def forward(self, x):
        out = rms_norm(x, self.weight, self.eps)
        if out is not None:
            return out
        _stats["norm_fallback"] += 1
        return original_forward(self, x)

    RMSNorm.forward = forward
    RMSNorm._fused_patched = True
    RMSNorm._fused_original_forward = original_forward
    return True
