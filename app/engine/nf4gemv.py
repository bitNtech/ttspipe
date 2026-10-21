"""The 4-bit matmul for decode and prefill, on tensor cores, reading bitsandbytes' weights in place.

Every millisecond this server spends generating a frame is spent reading
weights: a frame is 390 `Linear4bit` calls - five per layer across 36 layers of
the slow transformer, plus 4 layers of the depth transformer run ten times -
about 4 GB of NF4 at batch size one. So the per-frame cost is essentially the
rate at which the card can stream 4-bit weights, and bitsandbytes' own NF4 GEMV
gets a fraction of it (35-48 GB/s on a GTX 1650).

Prefill is the other half of time-to-first-audio, and bitsandbytes has no
small-batch kernel at all: anything longer than one position dequantises every
weight into a dense fp16 GEMM first - ~490 ms flat on a 4 GB card whatever the
span, which is why the old code walked prompt tokens one GEMV at a time.

**One kernel covers both.** Each warp computes a 16-row x 8-column tile with
`mma.sync.m16n8k16`: weights dequantised into registers as the A operand, up to
8 input vectors as B. At N=1 seven of B's eight columns are zero - tensor cores
have throughput to spare, and the dequantisation is the same work a scalar GEMV
does. At N=8 the weights are still read exactly once, so 8 prompt positions
cost what 1 does. Measured on an RTX 2050 (112 GB/s peak), graph-timed:

| | frame of matmuls | effective |
| --- | ---: | ---: |
| scalar GEMV, 8 rows/warp (previous kernel) | ~46 ms | ~87 GB/s |
| this, N=1 | 40.1 ms | ~96 GB/s |
| this, N=8 | 42.6 ms | ~92 GB/s |
| this, N=16 | 46.2 ms | ~85 GB/s |
| this, N=32 | 73.6 ms | ~54 GB/s |

**No repacking.** A dot product does not care what order it sums in, as long as
the weights and the inputs agree. So rather than permuting bitsandbytes' layout
into the fragment layout `mma` wants, each thread loads 16 contiguous bytes of
two rows (32 codes, one uint4 each), and the *logical* k index of a fragment is
mapped onto those physical codes - the B fragment reads the inputs through the
same mapping. The weight buffer, the doubly-quantised scales and the NF4
codebook are all read where bitsandbytes put them, so this costs no VRAM, and
bitsandbytes' own forward keeps working for anything declined here.

Dequantisation is the arithmetic bitsandbytes does before its fp16 GEMM - code
x scale in fp32, rounded to fp16 - and accumulation is fp32, so results agree
with its dequantise-then-matmul reference to fp16 rounding
(`tests/test_nf4gemv.py`).

Larger tiles for more inputs: `x` is re-read once per row tile, so at N=32 the
inputs would cost 8x the weights' traffic through L2 if each warp only did one
row tile. Past N=8 a warp covers several (`_config`).

**It is compiled at runtime through NVRTC** - see `app/engine/nvrtc.py` for why
that rather than Triton. Anything this does not handle - a blocksize that is not
a power-of-two multiple of 64, `in_features` not a multiple of 128, fp32
activations, a misaligned pointer - falls through to bitsandbytes unchanged.
`/status` reports the split under `kernels.nf4_gemv`.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import TYPE_CHECKING

from app.engine import nvrtc

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

#: Most input vectors one launch takes (4 column tiles of 8). Longer prompts go
#: through in slices of this many.
MAX_N = 32

#: The NF4 codebook, in index order. These are the values bitsandbytes'
#: `kQuantMapNF4` holds; the kernel looks them up rather than computing them.
NF4_CODE = (
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
)

_SOURCE = (
    nvrtc.PRELUDE
    + r"""
#define NT __NT__          // 8-column tiles of inputs
#define MT __MT__          // 16-row tiles per warp
#define WARPS __WARPS__    // warps per block, splitting K between them
#define NESTED __NESTED__

__device__ __forceinline__ uint pack2(float lo, float hi) {
  uint r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

extern "C" __global__ __launch_bounds__(32 * WARPS) void nf4_mma(
    const half_t*        __restrict__ X,      // N x K
    const uint4*         __restrict__ W,      // M x K/32: bnb packed, 2 codes per byte
    const unsigned char* __restrict__ A8,     // nested: one int8 scale per code block
    const float*         __restrict__ AMAX,   // nested: scales of those; plain: the scales
    const float*         __restrict__ CODE2,  // nested: the 256-point scale codebook
    half_t*              __restrict__ Y,      // N x M
    const float*         __restrict__ LUT,    // 16 NF4 code points
    float offset, int M, int K, int N, int nblk, int bs_shift, int bs2_shift,
    const half_t* __restrict__ R,     // epilogue operand, N x M (unused when epi == 0)
    int epi)                          // 0: Y = XW; 1: Y = R + XW; 2: Y = silu(R) * XW
{
  // Sixteen entries in sixteen banks: every bank has one possible address, so
  // any mix of indices is served without a conflict.
  __shared__ float lut_s[16];
  __shared__ float red[MT][NT][4][32];
  if (threadIdx.x < 16) lut_s[threadIdx.x] = LUT[threadIdx.x];
  for (int i = threadIdx.x; i < MT * NT * 128; i += 32 * WARPS) (&red[0][0][0][0])[i] = 0.f;

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  __syncthreads();
  const int g = lane >> 2;          // mma "groupID": the row (and input column) this lane feeds
  const int t = lane & 3;           // mma "threadID_in_group"
  const int row0 = blockIdx.x * 16 * MT;
  const int kv = K >> 5;            // uint4 per weight row
  const int nchunk = K >> 7;        // a warp consumes 128 codes of K per step

  float acc[MT][NT][4];
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  for (int c = warp; c < nchunk; c += WARPS) {
    // Lane (g, t) owns physical codes [k0, k0 + 32) of rows g and g + 8.
    const int k0 = c * 128 + 32 * t;
    uint4 pa[MT], pb[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int ra = min(row0 + 16 * i + g, M - 1), rb = min(row0 + 16 * i + g + 8, M - 1);
      pa[i] = W[(size_t)ra * kv + c * 4 + t];
      pb[i] = W[(size_t)rb * kv + c * 4 + t];
    }
    // 32 codes never straddle two scale blocks: blocks are >= 64 codes and
    // both are powers of two.
    float sa[MT], sb[MT];
    #pragma unroll
    for (int i = 0; i < MT; ++i) {
      const int ra = min(row0 + 16 * i + g, M - 1), rb = min(row0 + 16 * i + g + 8, M - 1);
      const size_t ba = (size_t)ra * nblk + (k0 >> bs_shift);
      const size_t bb = (size_t)rb * nblk + (k0 >> bs_shift);
#if NESTED
      sa[i] = CODE2[A8[ba]] * AMAX[ba >> bs2_shift] + offset;
      sb[i] = CODE2[A8[bb]] * AMAX[bb >> bs2_shift] + offset;
#else
      sa[i] = AMAX[ba];
      sb[i] = AMAX[bb];
#endif
    }
    // Eight mma k-steps of 16. In step s, the fragment's logical columns
    // (2t, 2t+1) are physical codes k0 + 4s + {0, 1} and (2t+8, 2t+9) are
    // k0 + 4s + {2, 3}; the B fragment reads x through the same map.
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
      uint2 xr[NT];
      #pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int n = j * 8 + g;
        xr[j] = (n < N) ? *reinterpret_cast<const uint2*>(X + (size_t)n * K + k0 + 4 * s)
                        : make_uint2(0, 0);
      }
      #pragma unroll
      for (int i = 0; i < MT; ++i) {
        // Byte 2s holds codes 4s (high nibble) and 4s+1; byte 2s+1 holds 4s+2, 4s+3.
        const uint wa = (&pa[i].x)[s >> 1] >> (16 * (s & 1));
        const uint wb = (&pb[i].x)[s >> 1] >> (16 * (s & 1));
        const uint a0 = pack2(lut_s[(wa >> 4) & 15u] * sa[i], lut_s[wa & 15u] * sa[i]);
        const uint a2 = pack2(lut_s[(wa >> 12) & 15u] * sa[i], lut_s[(wa >> 8) & 15u] * sa[i]);
        const uint a1 = pack2(lut_s[(wb >> 4) & 15u] * sb[i], lut_s[wb & 15u] * sb[i]);
        const uint a3 = pack2(lut_s[(wb >> 12) & 15u] * sb[i], lut_s[(wb >> 8) & 15u] * sb[i]);
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
          asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]), "+f"(acc[i][j][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(xr[j].x), "r"(xr[j].y));
        }
      }
    }
  }

  // Split-K across the block's warps.
  #pragma unroll
  for (int i = 0; i < MT; ++i)
    #pragma unroll
    for (int j = 0; j < NT; ++j)
      #pragma unroll
      for (int e = 0; e < 4; ++e) atomicAdd(&red[i][j][e][lane], acc[i][j][e]);
  __syncthreads();
  for (int idx = threadIdx.x; idx < MT * NT * 128; idx += 32 * WARPS) {
    const int i = idx / (NT * 128), j = (idx / 128) % NT, e = (idx >> 5) & 3, l = idx & 31;
    // Accumulator e of lane l: row (l >> 2) (+8 for e >= 2), column 2(l & 3) + (e & 1).
    const int row = row0 + 16 * i + (l >> 2) + ((e < 2) ? 0 : 8);
    const int n = j * 8 + 2 * (l & 3) + (e & 1);
    if (n < N && row < M) {
      const size_t o = (size_t)n * M + row;
      // Rounded wherever the unfused fork rounds: the matmul's own output is
      // fp16, and so is each elementwise op after it.
      const float y = h2f(f2h(red[i][j][e][l]));
      if (epi == 1) {
        Y[o] = f2h(h2f(R[o]) + y);
      } else if (epi == 2) {
        const float a = h2f(R[o]);
        Y[o] = f2h(h2f(f2h(a / (1.f + __expf(-a)))) * y);
      } else {
        Y[o] = f2h(y);
      }
    }
  }
}
"""
)


def _config(n: int) -> tuple[int, int, int]:
    """(column tiles, row tiles per warp, warps per block) for n inputs.

    Measured on an RTX 2050 across this model's five weight shapes, clocks held
    up between runs: one input wants the most warps in flight; more inputs want
    each warp to cover more rows, so the inputs it loads are reused.
    """
    nt = (n + 7) // 8
    if n == 1:
        return 1, 1, 8
    if nt <= 2:
        return nt, 2, 4
    return nt, 4, 4


_lock = threading.Lock()
_kernels: dict[tuple, object] = {}  # (nested, nt, mt, warps) -> CUfunction
_built = False
_cuda = None  # libcuda handle
_lut = None  # 16 fp32 code points, on device
_unavailable: str | None = None

_stats = {"fast": 0, "fallback": 0, "layers": 0}


def stats() -> dict:
    """Calls served by the kernel, calls handed back to bitsandbytes, layers covered."""
    return dict(_stats)


def _variants():
    for nested in (True, False):
        for n in (1, 8, 16, 24, 32):
            yield (nested, *_config(n))


def available() -> bool:
    """Whether the kernel is built and usable. Compiles every variant on the first call."""
    global _cuda, _lut, _unavailable, _built

    if _built:
        return True
    if _unavailable is not None:
        return False

    with _lock:
        if _built:
            return True
        if _unavailable is not None:
            return False
        try:
            import torch

            # Separate builds rather than runtime branches: the tile shape sizes
            # the register file, and whether scales are doubly quantised is a
            # property of the checkpoint, not of the call.
            for key in set(_variants()):
                nested, nt, mt, warps = key
                source = (
                    _SOURCE.replace("__NT__", str(nt))
                    .replace("__MT__", str(mt))
                    .replace("__WARPS__", str(warps))
                    .replace("__NESTED__", "1" if nested else "0")
                )
                _kernels[key] = nvrtc.build(source, ["nf4_mma"], label="nf4_mma")["nf4_mma"]
            _cuda = nvrtc.driver()
            _lut = torch.tensor(NF4_CODE, dtype=torch.float32, device="cuda")
            _built = True
            logger.info("nf4 tensor-core matmul compiled (%d variants)", len(_kernels))
            return True
        except Exception as exc:  # noqa: BLE001 - a missing fast path is not fatal
            _kernels.clear()
            _unavailable = str(exc)
            logger.warning(
                "nf4 kernel unavailable (%s); falling back to bitsandbytes, "
                "which is ~2x slower per frame and far slower to prefill",
                exc,
            )
            return False


def prepare(module) -> bool:
    """Attach what the kernel needs to one `Linear4bit`, or say it cannot serve it.

    Idempotent, and the result is cached on the module, so the eligibility
    checks below run once per layer rather than once per frame. Nothing
    bitsandbytes owns is modified or copied.
    """
    import torch

    cached = getattr(module, "_nf4_fast", -1)
    if cached != -1:
        return cached is not None

    def reject(why: str) -> bool:
        module._nf4_fast = None
        logger.debug("nf4 kernel declines %s: %s", type(module).__name__, why)
        return False

    weight = getattr(module, "weight", None)
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return reject("weight is not quantised yet")
    if getattr(quant_state, "quant_type", None) != "nf4":
        return reject(f"quant_type={getattr(quant_state, 'quant_type', None)}")
    if weight.device.type != "cuda":
        return reject("weight is not on a CUDA device")

    out_features, in_features = tuple(quant_state.shape)
    blocksize = int(quant_state.blocksize)
    if blocksize < 64 or blocksize & (blocksize - 1):
        return reject(f"blocksize {blocksize} is not a power of two >= 64")
    if in_features % blocksize or in_features % 128:
        return reject(f"in_features {in_features} is not a multiple of 128 and {blocksize}")
    if weight.data_ptr() % 16:
        return reject("packed weight is not 16-byte aligned")

    nblk = in_features // blocksize
    nested = bool(getattr(quant_state, "nested", False))
    if nested:
        state2 = quant_state.state2
        bs2 = int(state2.blocksize)
        if bs2 & (bs2 - 1):
            return reject(f"second-level blocksize {bs2} is not a power of two")
        if quant_state.absmax.dtype != torch.uint8 or state2.absmax.dtype != torch.float32:
            return reject("second-level scales are not the int8/fp32 pair the kernel reads")
        scale_ptrs = (
            quant_state.absmax.data_ptr(),
            state2.absmax.data_ptr(),
            state2.code.data_ptr(),
        )
        offset = float(quant_state.offset)
        bs2_shift = bs2.bit_length() - 1
        held = (quant_state.absmax, state2.absmax, state2.code)
    else:
        if quant_state.absmax.dtype != torch.float32:
            return reject(f"scales are {quant_state.absmax.dtype}, not float32")
        scale_ptrs = (0, quant_state.absmax.data_ptr(), 0)
        offset = 0.0
        bs2_shift = 0
        held = (quant_state.absmax,)
    if quant_state.absmax.numel() != out_features * nblk:
        return reject("scales do not form a rectangle over the weight rows")

    # Built once and kept: rebuilding a dozen ctypes objects per call costs
    # more than some of the calls. Only the pointers and N change.
    x_ptr = ctypes.c_void_p(0)
    y_ptr = ctypes.c_void_p(0)
    n_arg = ctypes.c_int(1)
    r_ptr = ctypes.c_void_p(0)
    epi_arg = ctypes.c_int(0)
    values = [
        x_ptr,
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(scale_ptrs[0]),
        ctypes.c_void_p(scale_ptrs[1]),
        ctypes.c_void_p(scale_ptrs[2]),
        y_ptr,
        ctypes.c_void_p(_lut.data_ptr()),
        ctypes.c_float(offset),
        ctypes.c_int(out_features),
        ctypes.c_int(in_features),
        n_arg,
        ctypes.c_int(nblk),
        ctypes.c_int(blocksize.bit_length() - 1),
        ctypes.c_int(bs2_shift),
        r_ptr,
        epi_arg,
    ]
    argv = (ctypes.c_void_p * len(values))(
        *[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in values]
    )
    module._nf4_fast = {
        "nested": nested,
        "held": held,  # so nothing the pointers above name can be freed
        "values": values,
        "argv": argv,
        "x_ptr": x_ptr,
        "y_ptr": y_ptr,
        "n": n_arg,
        "r_ptr": r_ptr,
        "epi": epi_arg,
        "M": out_features,
        "K": in_features,
    }
    _stats["layers"] += 1
    return True


#: Epilogues `matmul` can fold into the kernel's store.
EPI_NONE, EPI_RESIDUAL, EPI_SILU_GATE = 0, 1, 2


def matmul(module, x: "torch.Tensor", *, epi: int = EPI_NONE, operand=None) -> "torch.Tensor | None":
    """`x @ module.weight.T` for any number of input rows, or None if not applicable.

    `epi` fuses what the fork does right after: `operand + xW` (a residual
    add) or `silu(operand) * xW` (the SwiGLU gate). `operand` has the output's
    shape and must be contiguous fp16 on the same device.
    """
    import torch

    fast = getattr(module, "_nf4_fast", None)
    if fast is None:
        return None
    if x.dtype != torch.float16 or x.device.type != "cuda":
        return None
    K, M = fast["K"], fast["M"]
    if x.shape[-1] != K:
        return None
    flat = x.reshape(-1, K)
    if not flat.is_contiguous() or flat.data_ptr() % 16:
        flat = flat.contiguous()
        if flat.data_ptr() % 16:
            return None

    rows = flat.shape[0]
    if epi != EPI_NONE:
        if operand is None or operand.numel() != rows * M or operand.dtype != torch.float16:
            return None
        operand = operand.reshape(rows, M)
        if not operand.is_contiguous():
            return None
    out = torch.empty((rows, M), dtype=torch.float16, device=x.device)
    fast["epi"].value = epi
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    for start in range(0, rows, MAX_N):
        n = min(MAX_N, rows - start)
        nt, mt, warps = _config(n)
        fast["x_ptr"].value = flat.data_ptr() + start * K * 2
        fast["y_ptr"].value = out.data_ptr() + start * M * 2
        fast["n"].value = n
        fast["r_ptr"].value = operand.data_ptr() + start * M * 2 if epi != EPI_NONE else 0
        status = _cuda.cuLaunchKernel(
            _kernels[(fast["nested"], nt, mt, warps)],
            -(-M // (16 * mt)), 1, 1,
            32 * warps, 1, 1,
            0,
            stream,
            fast["argv"],
            None,
        )
        if status:
            raise RuntimeError(f"cuLaunchKernel failed for nf4_mma: {status}")
    _stats["fast"] += 1
    return out.view(*x.shape[:-1], M)


def install(model=None) -> bool:
    """Route fp16 `Linear4bit` calls through the kernel.

    Patches the class, so it covers every instance including any the model
    builds later. Returns whether the fast path is live; a False is a
    performance regression, never a failure, and is logged as such.
    """
    import bitsandbytes as bnb

    if getattr(bnb.nn.Linear4bit, "_nf4_patched", False):
        if model is not None:
            _prepare_all(model)
        return available()

    if not available():
        return False

    original_forward = bnb.nn.Linear4bit.forward

    def forward(self, x):
        if prepare(self):
            out = matmul(self, x)
            if out is not None:
                if self.bias is not None:
                    out = out + self.bias
                return out
        _stats["fallback"] += 1
        return original_forward(self, x)

    bnb.nn.Linear4bit.forward = forward
    bnb.nn.Linear4bit._nf4_patched = True
    bnb.nn.Linear4bit._nf4_original_forward = original_forward

    if model is not None:
        _prepare_all(model)
    return True


def _prepare_all(model) -> None:
    """Build every layer's launch arguments now, so no frame pays for it mid-utterance."""
    import bitsandbytes as bnb

    served = declined = 0
    for module in model.modules():
        if isinstance(module, bnb.nn.Linear4bit):
            if prepare(module):
                served += 1
            else:
                declined += 1
    logger.info("nf4 kernel ready for %d of %d 4-bit layers", served, served + declined)
