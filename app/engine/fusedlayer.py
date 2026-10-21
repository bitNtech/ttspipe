"""One decode position through a transformer block in nine kernels instead of ~30.

With the 4-bit matmuls at ~90% of the card's memory bandwidth
(`app/engine/nf4gemv.py`), what is left of a frame is the glue between them:
76 block passes a frame (36 slow layers, 4 fast layers x 10 codebooks), each
~30 kernels as the fork and PyTorch spell it - the qkv split and views, two
head norms of five kernels each, two rotaries, two KV-cache scatters, eight
attention ops, the SiLU, the gate multiply, two residual adds. Inside a CUDA
graph a kernel costs no CPU, but each still costs a few microseconds of GPU
launch and tail, and ~2300 of them were ~7 ms of a 47 ms frame.

A decode block here is:

    rms_norm -> wqkv -> qk_rope_kv -> attn_partial -> attn_combine
             -> wo(+x) -> rms_norm -> w1 -> w3(silu(w1) * .) -> w2(+h)

`qk_rope_kv` does the head norms, rotary and the KV-cache write for every head
at once; attention is split-K over 32-position chunks (flash-decoding) so the
8 KV heads still fill the card; the residual adds and the SwiGLU gate ride on
the matmul's store (`nf4gemv.EPI_*`). Folding the norms into the matmul's
input loads was tried and is slower: every block has to finish the row
reduction before its first weight load, which costs more than the launch it
saves. Rounding follows the fork: every place
it produces an fp16 tensor, this rounds to fp16 too.

Only a single position (decode) takes this path, and only when every piece is
available; anything else goes to the fork's own block unchanged.
"""

from __future__ import annotations

import ctypes
import logging

from app.engine import nvrtc

logger = logging.getLogger(__name__)

HEAD_DIM = 128
CHUNK = 32

_SOURCE = nvrtc.PRELUDE + r"""
#define D 128
#define CHUNK __CHUNK__
#define F_FP32 0
#define F_FP16 1
#define F_BF16 2

__device__ __forceinline__ float ld_f(const void* F, int i, int ftype) {
  if (ftype == F_FP32) return ((const float*)F)[i];
  if (ftype == F_FP16) return h2f(((const half_t*)F)[i]);
  return bf2f(((const half_t*)F)[i]);
}

// One block per head (query, key, value), one thread per dimension.
extern "C" __global__ void qk_rope_kv(
    const half_t* __restrict__ QKV, const half_t* __restrict__ QW, const half_t* __restrict__ KW,
    float eps, const void* __restrict__ F, int ftype, const long long* __restrict__ POS,
    half_t* __restrict__ Q, half_t* __restrict__ KC, half_t* __restrict__ VC,
    int nh, int nkv, int max_len)
{
  __shared__ float sx[D];
  __shared__ float red[D / 32];
  const int head = blockIdx.x, d = threadIdx.x;
  const long long pos = POS[0];

  if (head >= nh + nkv) {                       // value: copy into the cache
    const int vh = head - nh - nkv;
    VC[((long long)vh * max_len + pos) * D + d] = QKV[(nh + nkv + vh) * D + d];
    return;
  }
  const bool is_q = head < nh;
  float x = h2f(QKV[head * D + d]);
  const half_t* w = is_q ? QW : KW;
  if (w != 0) {                                 // torch RMSNorm over the head
    float v = x * x;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    if ((d & 31) == 0) red[d >> 5] = v;
    __syncthreads();
    const float ss = red[0] + red[1] + red[2] + red[3];
    x = h2f(f2h(x * rsqrtf(ss / (float)D + eps) * h2f(w[d])));
  }
  sx[d] = x;
  __syncthreads();
  const float x0 = sx[d & ~1], x1 = sx[d | 1];
  const float fr = ld_f(F, d & ~1, ftype), fi = ld_f(F, d | 1, ftype);
  const float y = (d & 1) ? __fadd_rn(__fmul_rn(x1, fr), __fmul_rn(x0, fi))
                          : __fsub_rn(__fmul_rn(x0, fr), __fmul_rn(x1, fi));
  if (is_q) Q[head * D + d] = f2h(y);
  else KC[((long long)(head - nh) * max_len + pos) * D + d] = f2h(y);
}

// Grid (nkv, chunks); one warp per query head of the group; each lane owns 4 dims.
extern "C" __global__ void attn_partial(
    const half_t* __restrict__ Q, const half_t* __restrict__ KC, const half_t* __restrict__ VC,
    const long long* __restrict__ POS, float* __restrict__ PM, float* __restrict__ PL,
    float* __restrict__ PACC, int groups, int max_len, int nchunk, float scale)
{
  const int kh = blockIdx.x, c = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qh = kh * groups + warp;
  const long long L = POS[0] + 1;
  const long long j0 = (long long)c * CHUNK;
  const size_t slot = (size_t)qh * nchunk + c;
  if (j0 >= L) {
    if (lane == 0) { PM[slot] = -1e30f; PL[slot] = 0.f; }
    return;
  }
  const long long j1 = min(L, j0 + CHUNK);
  float q[4];
  {
    const uint2 u = *reinterpret_cast<const uint2*>(Q + qh * D + lane * 4);
    const half_t* h = reinterpret_cast<const half_t*>(&u);
    #pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = h2f(h[i]) * scale;
  }
  float m = -1e30f, l = 0.f, acc[4] = {0.f, 0.f, 0.f, 0.f};
  const half_t* kb = KC + (size_t)kh * max_len * D + lane * 4;
  const half_t* vb = VC + (size_t)kh * max_len * D + lane * 4;
  for (long long j = j0; j < j1; ++j) {
    const uint2 ku = *reinterpret_cast<const uint2*>(kb + j * D);
    const half_t* kh4 = reinterpret_cast<const half_t*>(&ku);
    float s = q[0] * h2f(kh4[0]) + q[1] * h2f(kh4[1]) + q[2] * h2f(kh4[2]) + q[3] * h2f(kh4[3]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
    const float mn = fmaxf(m, s);
    const float a = __expf(m - mn), p = __expf(s - mn);
    const uint2 vu = *reinterpret_cast<const uint2*>(vb + j * D);
    const half_t* vh4 = reinterpret_cast<const half_t*>(&vu);
    l = l * a + p;
    #pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = acc[i] * a + p * h2f(vh4[i]);
    m = mn;
  }
  if (lane == 0) { PM[slot] = m; PL[slot] = l; }
  float* out = PACC + slot * D + lane * 4;
  #pragma unroll
  for (int i = 0; i < 4; ++i) out[i] = acc[i];
}

// One block per query head, one thread per dimension.
extern "C" __global__ void attn_combine(
    const float* __restrict__ PM, const float* __restrict__ PL, const float* __restrict__ PACC,
    half_t* __restrict__ Y, int nchunk)
{
  const int qh = blockIdx.x, d = threadIdx.x;
  float M = -1e30f;
  for (int c = 0; c < nchunk; ++c) M = fmaxf(M, PM[qh * nchunk + c]);
  float num = 0.f, den = 0.f;
  for (int c = 0; c < nchunk; ++c) {
    const float l = PL[qh * nchunk + c];
    if (l == 0.f) continue;
    const float w = __expf(PM[qh * nchunk + c] - M);
    den += l * w;
    num += PACC[((size_t)qh * nchunk + c) * D + d] * w;
  }
  Y[qh * D + d] = f2h(num / den);
}
"""

_FTYPE = {"torch.float32": 0, "torch.float16": 1, "torch.bfloat16": 2}
_kernels = None
_installed = False
_stats = {"fused": 0, "fallback": 0}


def stats() -> dict:
    return {**_stats, "ready": bool(_kernels) and _installed}


def _built():
    global _kernels
    if _kernels is None:
        try:
            _kernels = nvrtc.build(
                _SOURCE.replace("__CHUNK__", str(CHUNK)),
                ["qk_rope_kv", "attn_partial", "attn_combine"],
                fast_math=False,
                label="fusedlayer",
            )
        except Exception as exc:  # noqa: BLE001 - the fork's block still works
            logger.warning("fused decode block unavailable (%s)", exc)
            _kernels = False
    return _kernels or None


def _ptr(t) -> ctypes.c_void_p:
    return ctypes.c_void_p(0 if t is None else t.data_ptr())


def _eligible(block, x, mask, input_pos) -> bool:
    import torch

    from app.engine import nf4gemv

    a, ff = block.attention, block.feed_forward
    if x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1 or x.dtype != torch.float16:
        return False
    if mask is None or input_pos is None or input_pos.numel() != 1 or a.kv_cache is None:
        return False
    if a.head_dim != HEAD_DIM or a.n_head % a.n_local_heads or a.n_head // a.n_local_heads > 32:
        return False
    if input_pos.dtype != torch.long:
        return False
    for linear in (a.wqkv, a.wo, ff.w1, ff.w2, ff.w3):
        if getattr(linear, "bias", None) is not None or not nf4gemv.prepare(linear):
            return False
    return True


def decode_block(block, x, freqs_cis, mask, input_pos):
    """`TransformerBlock.forward` for one position, or None if it cannot."""
    import torch

    from app.engine import fusedops, nf4gemv

    kernels = _built()
    if kernels is None or not _eligible(block, x, mask, input_pos):
        return None
    a, ff = block.attention, block.feed_forward
    xn = fusedops.rms_norm(x, block.attention_norm.weight, block.attention_norm.eps)
    if xn is None or str(freqs_cis.dtype) not in _FTYPE or not freqs_cis.is_contiguous():
        return None
    qkv = nf4gemv.matmul(a.wqkv, xn)
    if qkv is None:
        return None

    nh, nkv, max_len = a.n_head, a.n_local_heads, a.kv_cache.k_cache.shape[2]
    qn = getattr(a, "q_norm", None) if a.attention_qk_norm else None
    kn = getattr(a, "k_norm", None) if a.attention_qk_norm else None
    eps = float(qn.eps if qn is not None and qn.eps is not None else 1e-6)
    q = torch.empty(nh * HEAD_DIM, dtype=torch.float16, device=x.device)
    argv, _k1 = nvrtc.pack(
        [
            _ptr(qkv), _ptr(None if qn is None else qn.weight), _ptr(None if kn is None else kn.weight),
            ctypes.c_float(eps), _ptr(freqs_cis), ctypes.c_int(_FTYPE[str(freqs_cis.dtype)]),
            _ptr(input_pos), _ptr(q), _ptr(a.kv_cache.k_cache), _ptr(a.kv_cache.v_cache),
            ctypes.c_int(nh), ctypes.c_int(nkv), ctypes.c_int(max_len),
        ]
    )
    nvrtc.launch(kernels["qk_rope_kv"], nh + 2 * nkv, HEAD_DIM, argv)

    # The graph is captured per window, so the chunk count is fixed by the
    # mask's width; chunks past the live position exit at once.
    nchunk = -(-mask.shape[-1] // CHUNK)
    pm = torch.empty(nh * nchunk, dtype=torch.float32, device=x.device)
    pl = torch.empty_like(pm)
    pacc = torch.empty(nh * nchunk * HEAD_DIM, dtype=torch.float32, device=x.device)
    groups = nh // nkv
    argv, _k2 = nvrtc.pack(
        [
            _ptr(q), _ptr(a.kv_cache.k_cache), _ptr(a.kv_cache.v_cache), _ptr(input_pos),
            _ptr(pm), _ptr(pl), _ptr(pacc), ctypes.c_int(groups), ctypes.c_int(max_len),
            ctypes.c_int(nchunk), ctypes.c_float(HEAD_DIM**-0.5),
        ]
    )
    _launch2(kernels["attn_partial"], nkv, nchunk, 32 * groups, argv)
    y = torch.empty((1, 1, nh * HEAD_DIM), dtype=torch.float16, device=x.device)
    argv, _k3 = nvrtc.pack([_ptr(pm), _ptr(pl), _ptr(pacc), _ptr(y), ctypes.c_int(nchunk)])
    nvrtc.launch(kernels["attn_combine"], nh, HEAD_DIM, argv)

    h = nf4gemv.matmul(a.wo, y, epi=nf4gemv.EPI_RESIDUAL, operand=x)
    hn = fusedops.rms_norm(h, block.ffn_norm.weight, block.ffn_norm.eps)
    g1 = nf4gemv.matmul(ff.w1, hn)
    g = nf4gemv.matmul(ff.w3, hn, epi=nf4gemv.EPI_SILU_GATE, operand=g1)
    out = nf4gemv.matmul(ff.w2, g, epi=nf4gemv.EPI_RESIDUAL, operand=h)
    _stats["fused"] += 1
    return out


def _launch2(function, gx: int, gy: int, block: int, argv) -> None:
    import torch

    status = nvrtc.driver().cuLaunchKernel(
        function, gx, gy, 1, block, 1, 1, 0,
        ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), argv, None,
    )
    if status:
        raise RuntimeError(f"cuLaunchKernel failed: {status}")


def install() -> bool:
    """Route single-position `TransformerBlock.forward` calls through `decode_block`."""
    from fish_speech.models.text2semantic.llama import TransformerBlock

    if getattr(TransformerBlock, "_fused_patched", False):
        return _built() is not None
    if _built() is None:
        return False
    original = TransformerBlock.forward

    def forward(self, x, freqs_cis, mask, input_pos=None):
        out = decode_block(self, x, freqs_cis, mask, input_pos)
        if out is not None:
            return out
        if x.shape[1] == 1:
            _stats["fallback"] += 1
        return original(self, x, freqs_cis, mask, input_pos)

    global _installed
    TransformerBlock.forward = forward
    TransformerBlock._fused_patched = True
    _installed = True
    TransformerBlock._fused_original_forward = original
    return True
