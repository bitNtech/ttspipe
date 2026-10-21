"""Top-k / top-p sampling of one logit row in one kernel.

A frame samples eleven times - the semantic token twice (the repetition-aware
sampler draws a normal and a high-temperature candidate) and each of the nine
residual codebooks once - and the fork's sampler is a full `torch.sort` of the
row, a softmax, a cumsum, a scatter back and a softmax again: ~15 kernels per
draw, ~0.9 ms a frame inside a CUDA graph for rows of 4096.

It only ever needs the top `top_k` entries, so this finds them with a two-pass
radix select on the fp16 bit patterns (histograms in shared memory), sorts
those few in one warp, and does the rest - the full-row softmax normaliser,
the top-p cut, the temperature and the draw - in the same block.

The draw is the fork's own: `argmax(probs / Exp(1))` with the exponentials
made from `torch.rand`, which is exact sampling from `probs`; each candidate
uses the random number of its own vocabulary index. Cumulative probabilities
are summed in fp32 where the fork's fp16 `cumsum` rounds, so a draw whose mass
lands exactly on the top-p boundary can differ; the distribution is the same.
"""

from __future__ import annotations

import ctypes
import logging

from app.engine import nvrtc

logger = logging.getLogger(__name__)

#: Largest `top_k` the kernel takes. Requests above it use the fork's sampler.
MAX_TOP_K = 64

_SOURCE = nvrtc.PRELUDE + r"""
#define KMAX __KMAX__
#define CAND 128          // candidate slots: KMAX plus room for ties at the threshold
#define THREADS 1024
#define INFINITY __int_as_float(0x7f800000)

__device__ __forceinline__ uint key16(half_t h) {
  // fp16 bits -> an unsigned key that orders like the float.
  return (h & 0x8000u) ? (uint)(~h & 0xFFFFu) : (uint)(h | 0x8000u);
}

extern "C" __global__ __launch_bounds__(THREADS) void sample_topk(
    const half_t* __restrict__ L, int V, const half_t* __restrict__ TEMP,
    const half_t* __restrict__ TOPP, const long long* __restrict__ TOPK,
    const float* __restrict__ U, int* __restrict__ OUT)
{
  __shared__ uint hist[256];
  __shared__ float red[32];
  __shared__ float cval[CAND];
  __shared__ int cidx[CAND];
  __shared__ int ccount, above, thr, need;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int K = (int)TOPK[0];
  K = K < 1 ? 1 : (K > KMAX ? KMAX : K);
  if (K > V) K = V;

  // max and the full-row softmax normaliser
  float m = -INFINITY;
  for (int i = tid; i < V; i += THREADS) m = fmaxf(m, h2f(L[i]));
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  if (lane == 0) red[warp] = m;
  __syncthreads();
  if (warp == 0) {
    float v = red[lane];
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  const float M = red[0];
  __syncthreads();
  float z = 0.f;
  for (int i = tid; i < V; i += THREADS) z += __expf(h2f(L[i]) - M);
  for (int o = 16; o; o >>= 1) z += __shfl_xor_sync(0xffffffffu, z, o);
  if (lane == 0) red[warp] = z;
  __syncthreads();
  if (warp == 0) {
    float v = red[lane];
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[0] = v;
  }

  // radix select: the K-th largest key, high byte then low byte
  for (int b = tid; b < 256; b += THREADS) hist[b] = 0;
  if (tid == 0) ccount = 0;
  __syncthreads();
  const float Z = red[0];
  for (int i = tid; i < V; i += THREADS) atomicAdd(&hist[key16(L[i]) >> 8], 1u);
  __syncthreads();
  if (tid == 0) {
    int acc = 0, b = 255;
    for (; b > 0 && acc + (int)hist[b] < K; --b) acc += hist[b];
    above = acc; thr = b;
  }
  __syncthreads();
  const uint hi = (uint)thr;
  const int above_hi = above;
  __syncthreads();
  for (int b = tid; b < 256; b += THREADS) hist[b] = 0;
  __syncthreads();
  for (int i = tid; i < V; i += THREADS) {
    const uint k = key16(L[i]);
    if ((k >> 8) == hi) atomicAdd(&hist[k & 0xFFu], 1u);
  }
  __syncthreads();
  if (tid == 0) {
    int acc = above_hi, b = 255;
    for (; b > 0 && acc + (int)hist[b] < K; --b) acc += hist[b];
    thr = (int)((hi << 8) | (uint)b);
    need = K - acc;  // how many entries equal to the threshold make the cut
  }
  __syncthreads();
  const uint T = (uint)thr;

  // everything strictly above the threshold, then ties by lowest index
  for (int i = tid; i < V; i += THREADS) {
    if (key16(L[i]) > T) {
      const int s = atomicAdd(&ccount, 1);
      if (s < CAND) { cval[s] = h2f(L[i]); cidx[s] = i; }
    }
  }
  __syncthreads();
  if (warp == 0) {
    int taken = 0;
    for (int base = 0; base < V && taken < need; base += 32) {
      const int i = base + lane;
      const bool eq = i < V && key16(L[i]) == T;
      const uint mask = __ballot_sync(0xffffffffu, eq);
      const int rank = __popc(mask & ((1u << lane) - 1u));
      if (eq && taken + rank < need) {
        const int s = ccount + taken + rank;
        if (s < CAND) { cval[s] = h2f(L[i]); cidx[s] = i; }
      }
      taken += __popc(mask);
    }
    if (lane == 0) ccount = min(K, CAND);
  }
  __syncthreads();

  // one warp: sort the candidates (descending value, then index), cut, draw
  if (warp == 0) {
    const int n = ccount;
    float v[CAND / 32]; int ix[CAND / 32];
    #pragma unroll
    for (int j = 0; j < CAND / 32; ++j) {
      const int s = j * 32 + lane;
      v[j] = s < n ? cval[s] : -INFINITY;
      ix[j] = s < n ? cidx[s] : 0x7fffffff;
    }
    // Bitonic sort over CAND slots, slot = j*32 + lane, descending by value
    // and then ascending by index. Fully unrolled so every register index is
    // a constant.
    #pragma unroll
    for (int size = 2; size <= CAND; size <<= 1) {
      #pragma unroll
      for (int stride = size >> 1; stride > 0; stride >>= 1) {
        if (stride >= 32) {
          const int d = stride >> 5;  // partner is register j ^ d, same lane
          #pragma unroll
          for (int j = 0; j < CAND / 32; ++j) {
            if (j & d) continue;
            const int s = j * 32 + lane;
            const bool desc = (s & size) == 0;
            const bool lower_better = (v[j] > v[j | d]) || (v[j] == v[j | d] && ix[j] < ix[j | d]);
            if (lower_better != desc) {
              const float tv = v[j]; v[j] = v[j | d]; v[j | d] = tv;
              const int ti = ix[j]; ix[j] = ix[j | d]; ix[j | d] = ti;
            }
          }
        } else {
          #pragma unroll
          for (int j = 0; j < CAND / 32; ++j) {
            const int s = j * 32 + lane;
            const float pv = __shfl_xor_sync(0xffffffffu, v[j], stride);
            const int pi = __shfl_xor_sync(0xffffffffu, ix[j], stride);
            const bool lower = (s & stride) == 0;
            const bool desc = (s & size) == 0;
            const bool mine_better = (v[j] > pv) || (v[j] == pv && ix[j] < pi);
            // The lower slot of a descending pair keeps the better entry.
            const bool keep = (lower == desc) ? mine_better : !mine_better;
            if (!keep) { v[j] = pv; ix[j] = pi; }
          }
        }
      }
    }
    // inclusive cumulative probability over the sorted order
    const float temp = fmaxf(h2f(TEMP[0]), 1e-5f);
    const float topp = h2f(TOPP[0]);
    const int topk = K;
    float best = -INFINITY; int best_i = ix[0];
    float carry = 0.f;
    #pragma unroll
    for (int j = 0; j < CAND / 32; ++j) {
      const int s = j * 32 + lane;
      float p = (s < n) ? __expf(v[j] - M) / Z : 0.f;
      float c = p;
      for (int o = 1; o < 32; o <<= 1) {
        const float t = __shfl_up_sync(0xffffffffu, c, o);
        if (lane >= o) c += t;
      }
      c += carry;
      carry = __shfl_sync(0xffffffffu, c, 31);
      const bool keep = s < n && (s == 0 || (c <= topp && s < topk));
      if (keep) {
        const float u = fmaxf(U[ix[j]], 1e-20f);
        const float score = v[j] / temp - __logf(-__logf(u));
        if (score > best) { best = score; best_i = ix[j]; }
      }
    }
    for (int o = 16; o; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
      if (ob > best || (ob == best && oi < best_i)) { best = ob; best_i = oi; }
    }
    if (lane == 0) OUT[0] = best_i;
  }
}
"""

_kernel = None
_failed = False


def available() -> bool:
    global _kernel, _failed
    if _kernel is None and not _failed:
        try:
            _kernel = nvrtc.build(
                _SOURCE.replace("__KMAX__", str(MAX_TOP_K)), ["sample_topk"], fast_math=False,
                label="fusedsample",
            )["sample_topk"]
        except Exception as exc:  # noqa: BLE001 - the fork's sampler still works
            _failed = True
            logger.warning("fused sampler unavailable (%s)", exc)
    return _kernel is not None


def sample(logits, temperature, top_p, top_k):
    """The fork's `sample(...)[0]`: an int32 tensor of shape (1,) holding the drawn index."""
    import torch

    row = logits.reshape(-1)
    if row.dtype != torch.float16 or not row.is_contiguous():
        row = row.to(torch.float16).contiguous()
    uniform = torch.rand(row.numel(), device=row.device, dtype=torch.float32)
    out = torch.empty(1, dtype=torch.int, device=row.device)
    argv, _keep = nvrtc.pack(
        [
            ctypes.c_void_p(row.data_ptr()), ctypes.c_int(row.numel()),
            ctypes.c_void_p(temperature.data_ptr()), ctypes.c_void_p(top_p.data_ptr()),
            ctypes.c_void_p(top_k.data_ptr()), ctypes.c_void_p(uniform.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
        ]
    )
    nvrtc.launch(_kernel, 1, 1024, argv)
    return out
