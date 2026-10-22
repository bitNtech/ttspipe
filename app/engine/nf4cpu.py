"""NF4 matrix-vector product for CPU, on bitsandbytes' weights as stored.

The CPU counterpart of `app/engine/nf4gemv.py`. A decode step is nothing but
single-position matmuls, and on CPU those are bound by how many weight bytes
cross the memory bus. Dense bf16 reads 2 bytes a weight; NF4 reads half a
byte plus one fp32 absmax per 64. bitsandbytes' own CPU path dequantises the
whole weight before multiplying, which throws that away; regrouping onto a
uniform int4 GEMM keeps the bytes but adds ~8% error per layer, because NF4's
sixteen levels are not evenly spaced.

This kernel keeps the exact NF4 values: the sixteen-entry code table fits in
one AVX-512 register, so a nibble becomes its value with a single
`vpermps`. Each 64-weight block is 32 bytes; the high nibbles are the even
positions and the low ones the odd positions, so the activation is split into
even and odd halves once per call and each nibble lane multiplies straight
against its half. Rows are spread over torch's intra-op threads.

Up to `MAX_ROWS` activation rows share one pass over the weights (a
speculative prefill walks four at a time). Past that the vector units become
the bound, and `nf4_amx` takes over to `AMX_MAX_ROWS`: the same NF4 bytes
decoded into bf16 tiles for the AMX tile unit. Longer prompts (a fresh
voice's ~350-token prefix) go to the dense bf16 copy on oneDNN.

`decode_block` and `prefill_block` run a whole transformer block - norms,
rotary, KV write, attention, residuals and the four matmuls - for one
position or a short run of them, in one call (see app/engine/cpu.py).

Measured on a c4-standard-16 (Xeon 8581C, ~91 GB/s): single-position matmuls
at 60-75 GB/s (was ~30 before rows were register-blocked), a frame of the
36+4-layer model in ~78 ms (was ~150).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ROWS = 6

#: Up to here a call goes to the AMX kernel when the CPU has one: it decodes
#: the NF4 bytes into tiles, so a short prompt reads half a byte a weight
#: instead of the dense copy's two. Measured per 36 layers: ~80 ms from 7 to
#: 16 positions, against ~100-107 dense and 95-197 on the vector kernel.
#: Longer prompts are faster dense.
AMX_MAX_ROWS = 16

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

typedef uint16_t bf16_t;

static inline float bf2f(bf16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline bf16_t f2bf(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    bits += 0x7FFF + ((bits >> 16) & 1);  // round to nearest even
    return (bf16_t)(bits >> 16);
}

static inline float round_bf(float f) { return bf2f(f2bf(f)); }

// Sixteen bf16 values widened to fp32 (exact: bf16 is fp32's top half).
static inline __m512 bf16x16(const bf16_t* p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*)p)), 16));
}

// y[m, n] = sum_k x[m, k] * code[q[n, k]] * absmax[n, k / 64]: bf16 in and
// out, fp32 accumulation, rows spread over torch's intra-op threads.
//
// Per 64-weight block (32 bytes): widen the bytes to 16-bit lanes, split high
// (even k) and low (odd k) nibbles, look each up in the bf16 code table with
// one vpermw, and dot against the matching half of x with vdpbf16ps, which
// multiplies 32 bf16 pairs into 16 fp32 sums. The block's absmax is applied
// once to the fp32 partial. `xs` is x already split into even / odd halves:
// (M, 2, K/2).
//
// One activation row (every decode step) takes `gemv_rows1`: R rows of the
// weight per pass, each with its own accumulator held in a register. The
// general loop below keeps `acc[m]` in a runtime-indexed array the compiler
// leaves in memory, which makes every block of a row wait on a store-to-load
// round trip - measured ~30 GB/s on a machine that streams ~90. Each row
// still sums its blocks in the same order with the same operations, so the
// result is bit-identical; only independent rows are interleaved.
template <int R>
static inline void rows1(const bf16_t* xs, int64_t H, int64_t NB, const uint8_t* qp,
                         const bf16_t* ap, int64_t n, __m512i lut, __m512i mask,
                         bf16_t* yp) {
    __m512 acc[R];
    const uint8_t* row[R];
    const bf16_t* am[R];
    for (int r = 0; r < R; ++r) {
        acc[r] = _mm512_setzero_ps();
        row[r] = qp + (n + r) * H;
        am[r] = ap + (n + r) * NB;
    }
    for (int64_t b = 0; b < NB; ++b) {
        const int64_t j = b * 32;
        const __m512bh xe = (__m512bh)_mm512_loadu_si512(xs + j);
        const __m512bh xo = (__m512bh)_mm512_loadu_si512(xs + H + j);
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) {
            _mm_prefetch((const char*)(row[r] + j + 512), _MM_HINT_T0);
            __m512i v = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row[r] + j)));
            __m512bh we = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(v, 4), lut);
            __m512bh wo = (__m512bh)_mm512_permutexvar_epi16(_mm512_and_si512(v, mask), lut);
            __m512 p = _mm512_dpbf16_ps(_mm512_setzero_ps(), we, xe);
            p = _mm512_dpbf16_ps(p, wo, xo);
            acc[r] = _mm512_fmadd_ps(p, _mm512_set1_ps(bf2f(am[r][b])), acc[r]);
        }
    }
    for (int r = 0; r < R; ++r) yp[n + r] = f2bf(_mm512_reduce_add_ps(acc[r]));
}

static void gemv_rows1(const bf16_t* xs, int64_t K, const uint8_t* qp, const bf16_t* ap,
                       __m512i lut, __m512i mask, int64_t n_out, bf16_t* yp) {
    const int64_t H = K / 2, NB = K / 64;
    at::parallel_for(0, (n_out + 3) / 4, 4, [&](int64_t g0, int64_t g1) {
        for (int64_t g = g0; g < g1; ++g) {
            const int64_t n = g * 4;
            if (n + 4 <= n_out) {
                rows1<4>(xs, H, NB, qp, ap, n, lut, mask, yp);
            } else {
                for (int64_t t = n; t < n_out; ++t) rows1<1>(xs, H, NB, qp, ap, t, lut, mask, yp);
            }
        }
    });
}

// Several activation rows (a prompt prefill): MB of them share each decoded
// weight block, each with its own register accumulator - the weights cross
// the memory bus once for all of them. Same per-row operations as above, so
// a position's result does not depend on how many others it was batched with.
template <int MB>
static inline void rowsM(const bf16_t* xs, int64_t H, int64_t NB, const uint8_t* row,
                         const bf16_t* am, __m512i lut, __m512i mask, int64_t m0,
                         bf16_t* yp, int64_t ldy, int64_t n) {
    __m512 acc[MB];
    for (int m = 0; m < MB; ++m) acc[m] = _mm512_setzero_ps();
    for (int64_t b = 0; b < NB; ++b) {
        __m512i v = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)(row + b * 32)));
        __m512bh we = (__m512bh)_mm512_permutexvar_epi16(_mm512_srli_epi16(v, 4), lut);
        __m512bh wo = (__m512bh)_mm512_permutexvar_epi16(_mm512_and_si512(v, mask), lut);
        const __m512 s = _mm512_set1_ps(bf2f(am[b]));
        const int64_t j = b * 32;
#pragma GCC unroll 8
        for (int m = 0; m < MB; ++m) {
            const bf16_t* xm = xs + (m0 + m) * 2 * H + j;
            __m512bh xe = (__m512bh)_mm512_loadu_si512(xm);
            __m512bh xo = (__m512bh)_mm512_loadu_si512(xm + H);
            __m512 p = _mm512_dpbf16_ps(_mm512_setzero_ps(), we, xe);
            p = _mm512_dpbf16_ps(p, wo, xo);
            acc[m] = _mm512_fmadd_ps(p, s, acc[m]);
        }
    }
    for (int m = 0; m < MB; ++m) yp[(m0 + m) * ldy + n] = f2bf(_mm512_reduce_add_ps(acc[m]));
}

static void gemv_raw(const bf16_t* xs, int64_t M, int64_t K, const uint8_t* qp,
                     const bf16_t* ap, const bf16_t* code, int64_t n_out, bf16_t* yp,
                     int64_t ldy) {
    const int64_t H = K / 2, NB = K / 64;
    bf16_t lut_h[32];
    for (int i = 0; i < 32; ++i) lut_h[i] = code[i & 15];
    const __m512i lut = _mm512_loadu_si512(lut_h);
    const __m512i mask = _mm512_set1_epi16(15);
    if (M == 1 && ldy == n_out) {
        gemv_rows1(xs, K, qp, ap, lut, mask, n_out, yp);
        return;
    }
    at::parallel_for(0, n_out, 8, [&](int64_t n0, int64_t n1) {
        for (int64_t n = n0; n < n1; ++n) {
            const uint8_t* row = qp + n * H;
            const bf16_t* am = ap + n * NB;
            // The row stays in L1 across chunks; only the first pass is DRAM.
            for (int64_t m0 = 0; m0 < M; m0 += 8) {
                switch (std::min<int64_t>(8, M - m0)) {
                    case 8: rowsM<8>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 7: rowsM<7>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 6: rowsM<6>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 5: rowsM<5>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 4: rowsM<4>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 3: rowsM<3>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    case 2: rowsM<2>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                    default: rowsM<1>(xs, H, NB, row, am, lut, mask, m0, yp, ldy, n); break;
                }
            }
        }
    });
}

static void split_even_odd(const bf16_t* x, int64_t K, bf16_t* xs) {
    const int64_t H = K / 2;
    for (int64_t j = 0; j < H; ++j) {
        xs[j] = x[2 * j];
        xs[H + j] = x[2 * j + 1];
    }
}

// fp32 -> bf16 (round to nearest even, as f2bf) split into even / odd halves
// in one pass: what `f2bf` then `split_even_odd` do, 32 values at a time.
static void f32_to_split_bf16(const float* x, int64_t K, bf16_t* xs) {
    const int64_t H = K / 2;
    alignas(64) static const uint16_t idx[32] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                                                 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31};
    const __m512i perm = _mm512_load_si512(idx);
    int64_t j = 0;
    for (; j + 32 <= K; j += 32) {
        __m512i b = (__m512i)_mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + j + 16), _mm512_loadu_ps(x + j));
        b = _mm512_permutexvar_epi16(perm, b);
        _mm256_storeu_si256((__m256i*)(xs + j / 2), _mm512_castsi512_si256(b));
        _mm256_storeu_si256((__m256i*)(xs + H + j / 2), _mm512_extracti64x4_epi64(b, 1));
    }
    for (; j < K; j += 2) {
        xs[j / 2] = f2bf(x[j]);
        xs[H + j / 2] = f2bf(x[j + 1]);
    }
}

// bf16 -> fp32, 16 at a time (exact).
static void bf16_to_f32(const bf16_t* y, int64_t n, float* out) {
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) _mm512_storeu_ps(out + i, bf16x16(y + i));
    for (; i < n; ++i) out[i] = bf2f(y[i]);
}

// silu(a) * b with the fork's rounding points, over threads: a scalar exp per
// element was ~100 us of a ~800 us block at the 9728-wide MLP.
static void silu_mul(const float* a, const float* b, int64_t n, float* out) {
    at::parallel_for(0, n, 1024, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) {
            const float silu = round_bf(a[i] / (1.0f + std::exp(-a[i])));
            out[i] = round_bf(silu * b[i]);
        }
    });
}

torch::Tensor nf4_gemv(torch::Tensor x, torch::Tensor packed, torch::Tensor absmax,
                       torch::Tensor code16, int64_t n_out) {
    TORCH_CHECK(x.dim() == 2, "x must be (M, K)");
    const int64_t M = x.size(0), K = x.size(1), H = K / 2;
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
    auto xb = x.to(torch::kBFloat16).contiguous();
    auto xs = torch::empty({M, 2, H}, xb.options());
    const bf16_t* src = (const bf16_t*)xb.data_ptr();
    bf16_t* dst = (bf16_t*)xs.data_ptr();
    for (int64_t m = 0; m < M; ++m) split_even_odd(src + m * K, K, dst + m * 2 * H);
    auto y = torch::empty({M, n_out}, xb.options());
    gemv_raw(dst, M, K, packed.data_ptr<uint8_t>(), (const bf16_t*)absmax.data_ptr(),
             (const bf16_t*)code16.data_ptr(), n_out, (bf16_t*)y.data_ptr(), n_out);
    return y;
}

// ---------------------------------------------------------------------------
// Prompt prefill on AMX: many positions against the NF4 weights as stored.
//
// The AVX-512 kernel above is the right shape for one position - it is
// bandwidth-bound and reads half a byte a weight. For a prompt of more than a
// few positions it becomes compute-bound (each position costs the vector units
// again), and the alternative was a dense bf16 copy of every weight, 4x the
// bytes, served by oneDNN. Here the NF4 bytes are decoded once per call into
// bf16 tiles - code[q] * absmax rounded to bf16, the same value the dense copy
// holds - and multiplied on the AMX tile unit, where 16 positions cost about
// what one does. Computed as Y^T = W X^T so decoded weight rows load straight
// into A tiles; only X, which is small, is repacked (into AMX's pair layout).

struct AmxConfig {
    uint8_t palette;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
};

static bool amx_ready() {
    static int ok = -1;
    if (ok < 0) {
        // ARCH_REQ_XCOMP_PERM (0x1023) for XFEATURE_XTILEDATA (18): Linux will
        // not let a process touch tile registers until it has asked.
        ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0 ? 1 : 0;
    }
    return ok == 1;
}

bool amx_available() { return amx_ready(); }

// Tiles: C0-C3 = tmm0-3 (16 weight rows x mc positions, fp32), A = tmm4/5
// (16 weight rows x 32 k, bf16), B = tmm6 (16 k-pairs x mc positions).
static void amx_configure(int mc) {
    AmxConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.palette = 1;
    for (int t = 0; t < 4; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = (uint16_t)(mc * 4); }
    for (int t = 4; t < 6; ++t) { cfg.rows[t] = 16; cfg.colsb[t] = 64; }
    cfg.rows[6] = 16;
    cfg.colsb[6] = (uint16_t)(mc * 4);
    _tile_loadconfig(&cfg);
}

// One 64-weight block (32 bytes) of one weight row as two A-tile rows.
//
// AMX multiplies k in pairs, and a dot product does not care which pairs, as
// long as both operands agree. So rather than interleaving the even-k (high
// nibble) and odd-k (low nibble) halves back into k order, each half becomes a
// "virtual chunk" as it comes out of the table lookup: dword p of `we` is
// (w[4p], w[4p+2]) and of `wo` is (w[4p+1], w[4p+3]). X is packed to match,
// once per call, which is cheap - it is a few positions, not 100M weights.
static inline void decode64(const uint8_t* p, __m512i lut, bf16_t* we, bf16_t* wo) {
    const __m512i v = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i*)p));
    _mm512_storeu_si512(we, _mm512_permutexvar_epi16(_mm512_srli_epi16(v, 4), lut));
    _mm512_storeu_si512(wo, _mm512_permutexvar_epi16(_mm512_and_si512(v, _mm512_set1_epi16(15)), lut));
}

// The block's sixteen code values times its absmax, rounded to bf16 - the
// value a dense copy of the weight would hold.
static inline __m512i scaled_lut(__m512 lutv, bf16_t absmax) {
    return _mm512_castsi256_si512(
        (__m256i)_mm512_cvtneps_pbh(_mm512_mul_ps(lutv, _mm512_set1_ps(bf2f(absmax)))));
}

torch::Tensor nf4_amx(torch::Tensor x, torch::Tensor packed, torch::Tensor absmax,
                      torch::Tensor code16, int64_t n_out) {
    TORCH_CHECK(amx_ready(), "AMX is not available");
    TORCH_CHECK(x.dim() == 2, "x must be (M, K)");
    const int64_t M = x.size(0), K = x.size(1), H = K / 2, NB = K / 64, KC = K / 32;
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
    auto xb = x.to(torch::kBFloat16).contiguous();
    auto y = torch::empty({M, n_out}, xb.options());
    const bf16_t* xs16 = (const bf16_t*)xb.data_ptr();
    const uint8_t* qp = packed.data_ptr<uint8_t>();
    const bf16_t* ap = (const bf16_t*)absmax.data_ptr();
    const bf16_t* code = (const bf16_t*)code16.data_ptr();
    bf16_t* yp = (bf16_t*)y.data_ptr();

    // X^T in the virtual-chunk pair layout, one block per 16-position chunk:
    // chunk c, virtual chunk vc = 2b + h, row p, position m holds
    // (x[m][64b + 4p + h], x[m][64b + 4p + 2 + h]).
    const int64_t chunks = (M + 15) / 16;
    std::vector<bf16_t> xv(chunks * 16 * K);
    for (int64_t c = 0; c < chunks; ++c) {
        const int64_t m0 = c * 16, mc = std::min<int64_t>(16, M - m0);
        bf16_t* dst = xv.data() + c * 16 * K;
        for (int64_t vc = 0; vc < KC; ++vc) {
            const int64_t b = vc / 2, h = vc % 2;
            for (int64_t p = 0; p < 16; ++p)
                for (int64_t m = 0; m < mc; ++m) {
                    const bf16_t* xr = xs16 + (m0 + m) * K + 64 * b + 4 * p + h;
                    bf16_t* d = dst + ((vc * 16 + p) * mc + m) * 2;
                    d[0] = xr[0];
                    d[1] = xr[2];
                }
        }
    }

    float lut_f[16];
    for (int i = 0; i < 16; ++i) lut_f[i] = bf2f(code[i]);
    const __m512 lutv = _mm512_loadu_ps(lut_f);

    const int64_t n_full = n_out / 16 * 16;
    const int64_t groups = (n_full + 63) / 64;
    at::parallel_for(0, groups, 1, [&](int64_t g0, int64_t g1) {
        float cbuf[16 * 16];
        int configured = -1;
        // One position chunk (every prefill a hint leaves behind): decode
        // into a double buffer in L1, block b+1 while the tile unit multiplies
        // block b - a tile load straight after the stores that filled it
        // serialises the two, measured as decode + tiles with no overlap at
        // all. More chunks: decode each weight slab once into L2 and reuse it
        // for every chunk.
        thread_local std::vector<bf16_t> abuf;
        const bool stream = chunks == 1;
        abuf.resize(stream ? 2 * 4 * 2 * 16 * 32 : 4 * KC * 16 * 32);
        auto tile = [&](int bi, int64_t vc) -> bf16_t* {
            return stream ? abuf.data() + ((((vc >> 1) & 1) * 8 + bi * 2 + (vc & 1)) * 16) * 32
                          : abuf.data() + ((bi * KC + vc) * 16) * 32;
        };
        auto decode = [&](int64_t n0, int nb, int64_t b) {
            for (int bi = 0; bi < nb; ++bi)
                for (int r = 0; r < 16; ++r) {
                    const int64_t n = n0 + bi * 16 + r;
                    decode64(qp + n * H + b * 32, scaled_lut(lutv, ap[n * NB + b]),
                             tile(bi, 2 * b) + r * 32, tile(bi, 2 * b + 1) + r * 32);
                }
        };
        for (int64_t g = g0; g < g1; ++g) {
            const int64_t n0 = g * 64;
            const int nb = (int)std::min<int64_t>(4, (n_full - n0) / 16);
            if (!stream)
                for (int64_t b = 0; b < NB; ++b) decode(n0, nb, b);
            for (int64_t c = 0; c < chunks; ++c) {
                const int64_t m0 = c * 16;
                const int mc = (int)std::min<int64_t>(16, M - m0);
                if (mc != configured) { amx_configure(mc); configured = mc; }
                const bf16_t* xc = xv.data() + c * 16 * K;
                _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                if (stream) decode(n0, nb, 0);
                for (int64_t vc = 0; vc < KC; ++vc) {
                    if (stream && (vc & 1) == 0 && vc / 2 + 1 < NB) decode(n0, nb, vc / 2 + 1);
                    _tile_loadd(6, xc + vc * 16 * mc * 2, mc * 4);
#define NF4_AMX_STEP(I, T)                                    \
                    if (nb > I) {                            \
                        _tile_loadd(T, tile(I, vc), 64);     \
                        _tile_dpbf16ps(I, T, 6);             \
                    }
                    NF4_AMX_STEP(0, 4) NF4_AMX_STEP(1, 5) NF4_AMX_STEP(2, 4) NF4_AMX_STEP(3, 5)
#undef NF4_AMX_STEP
                }
#define NF4_AMX_OUT(I)                                                                       \
                if (nb > I) {                                                               \
                    _tile_stored(I, cbuf, mc * 4);                                          \
                    for (int r = 0; r < 16; ++r)                                            \
                        for (int m = 0; m < mc; ++m)                                        \
                            yp[(m0 + m) * n_out + n0 + I * 16 + r] = f2bf(cbuf[r * mc + m]); \
                }
                NF4_AMX_OUT(0) NF4_AMX_OUT(1) NF4_AMX_OUT(2) NF4_AMX_OUT(3)
#undef NF4_AMX_OUT
            }
        }
        if (configured >= 0) _tile_release();
    });

    if (n_full < n_out) {  // rows that do not fill a tile: the vector kernel
        std::vector<bf16_t> xs(M * 2 * H);
        for (int64_t m = 0; m < M; ++m)
            for (int64_t j = 0; j < H; ++j) {
                xs[m * 2 * H + j] = xs16[m * K + 2 * j];
                xs[m * 2 * H + H + j] = xs16[m * K + 2 * j + 1];
            }
        gemv_raw(xs.data(), M, K, qp + n_full * H, ap + n_full * NB, code, n_out - n_full,
                 yp + n_full, n_out);
    }
    return y;
}

// RMS norm of n values in fp32, rounded to bf16 the way the fork's
// `_norm(x.float()).type_as(x) * weight` rounds: after the norm and again
// after the weight.
static void rms_norm(const float* x, int64_t n, const bf16_t* w, float eps, float* out) {
    double ss = 0;
    for (int64_t i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const float r = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int64_t i = 0; i < n; ++i) out[i] = round_bf(round_bf(x[i] * r) * bf2f(w[i]));
}

static void gemv1(const float* x, int64_t K, const torch::Tensor& packed,
                  const torch::Tensor& absmax, const bf16_t* code, int64_t n_out, float* y,
                  std::vector<bf16_t>& xbuf, std::vector<bf16_t>& ybuf) {
    xbuf.resize(K);
    f32_to_split_bf16(x, K, xbuf.data());
    ybuf.resize(n_out);
    gemv_raw(xbuf.data(), 1, K, packed.data_ptr<uint8_t>(), (const bf16_t*)absmax.data_ptr(),
             code, n_out, ybuf.data(), n_out);
    bf16_to_f32(ybuf.data(), n_out, y);
}

// One decode position through a whole transformer block, everything but the
// four matmuls done here in plain loops instead of ~60 aten ops. Mirrors
// TransformerBlock.forward + app/engine/fastpath.py's attention exactly:
// qk-norm (if given), interleaved-pair rotary, KV write at `pos`, grouped
// attention over the first `mask.numel()` keys under the boolean mask.
torch::Tensor decode_block(
    torch::Tensor x, torch::Tensor attn_norm_w, torch::Tensor ffn_norm_w,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w, double eps, double qk_eps,
    torch::Tensor qkv_p, torch::Tensor qkv_a, torch::Tensor wo_p, torch::Tensor wo_a,
    torch::Tensor w13_p, torch::Tensor w13_a, torch::Tensor w2_p, torch::Tensor w2_a,
    torch::Tensor code16, torch::Tensor k_cache, torch::Tensor v_cache, int64_t pos,
    torch::Tensor freqs, torch::Tensor mask, int64_t n_head, int64_t n_local, int64_t head_dim,
    int64_t inter) {
    const int64_t D = x.size(-1);
    const int64_t qs = n_head * head_dim, kvs = n_local * head_dim;
    const bf16_t* code = (const bf16_t*)code16.data_ptr();
    std::vector<bf16_t> xbuf, ybuf;

    std::vector<float> xf(D), h(D), qkv(qs + 2 * kvs);
    const bf16_t* xin = (const bf16_t*)x.data_ptr();
    for (int64_t i = 0; i < D; ++i) xf[i] = bf2f(xin[i]);

    rms_norm(xf.data(), D, (const bf16_t*)attn_norm_w.data_ptr(), (float)eps, h.data());
    gemv1(h.data(), D, qkv_p, qkv_a, code, qs + 2 * kvs, qkv.data(), xbuf, ybuf);
    float* q = qkv.data();
    float* k = q + qs;
    float* v = k + kvs;

    if (q_norm_w.numel() > 0) {  // nn.RMSNorm over head_dim, per head
        std::vector<float> tmp(head_dim);
        auto norm_heads = [&](float* t, int64_t heads, const torch::Tensor& w) {
            const bf16_t* wp = (const bf16_t*)w.data_ptr();
            for (int64_t hd = 0; hd < heads; ++hd) {
                float* r = t + hd * head_dim;
                double ss = 0;
                for (int64_t i = 0; i < head_dim; ++i) ss += (double)r[i] * r[i];
                const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + (float)qk_eps);
                for (int64_t i = 0; i < head_dim; ++i) r[i] = round_bf(r[i] * inv * bf2f(wp[i]));
            }
        };
        norm_heads(q, n_head, q_norm_w);
        norm_heads(k, n_local, k_norm_w);
    }

    // Rotary on interleaved pairs; freqs is (head_dim/2, 2) = (cos, sin).
    auto fr = freqs.to(torch::kFloat32).contiguous();
    const float* f = fr.data_ptr<float>();
    auto rope = [&](float* t, int64_t heads) {
        for (int64_t hd = 0; hd < heads; ++hd) {
            float* r = t + hd * head_dim;
            for (int64_t i = 0; i < head_dim / 2; ++i) {
                const float a = r[2 * i], b = r[2 * i + 1], c = f[2 * i], s = f[2 * i + 1];
                r[2 * i] = round_bf(a * c - b * s);
                r[2 * i + 1] = round_bf(b * c + a * s);
            }
        }
    };
    rope(q, n_head);
    rope(k, n_local);

    // KV cache is (1, n_local, max_len, head_dim) bf16.
    const int64_t max_len = k_cache.size(2);
    bf16_t* kc = (bf16_t*)k_cache.data_ptr();
    bf16_t* vc = (bf16_t*)v_cache.data_ptr();
    for (int64_t hd = 0; hd < n_local; ++hd)
        for (int64_t i = 0; i < head_dim; ++i) {
            kc[(hd * max_len + pos) * head_dim + i] = f2bf(k[hd * head_dim + i]);
            vc[(hd * max_len + pos) * head_dim + i] = f2bf(v[hd * head_dim + i]);
        }
    // An empty `wo` asks for the cache write alone: a position whose output
    // nobody reads still has to leave its K/V behind for the ones after it.
    if (wo_p.numel() == 0) return x;

    const int64_t L = mask.numel();
    auto mk = mask.to(torch::kBool).contiguous();
    const bool* mp = mk.data_ptr<bool>();
    const int64_t groups = n_head / n_local;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const bool vec = head_dim % 32 == 0;
    std::vector<float> att(qs);
    at::parallel_for(0, n_head, 1, [&](int64_t h0, int64_t h1) {
        std::vector<float> sc(L);
        for (int64_t hq = h0; hq < h1; ++hq) {
            const int64_t hk = hq / groups;
            const float* qr = q + hq * head_dim;
            const bf16_t* kb = kc + hk * max_len * head_dim;
            const bf16_t* vb = vc + hk * max_len * head_dim;
            float mx = -INFINITY;
            for (int64_t l = 0; l < L; ++l) {
                if (!mp[l]) { sc[l] = -INFINITY; continue; }
                const bf16_t* kr = kb + l * head_dim;
                float d;
                if (vec) {  // two independent fp32 chains instead of one scalar one
                    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
                    for (int64_t i = 0; i < head_dim; i += 32) {
                        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(qr + i), bf16x16(kr + i), s0);
                        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(qr + i + 16), bf16x16(kr + i + 16), s1);
                    }
                    d = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
                } else {
                    d = 0;
                    for (int64_t i = 0; i < head_dim; ++i) d += qr[i] * bf2f(kr[i]);
                }
                sc[l] = round_bf(d) * scale;
                mx = std::max(mx, sc[l]);
            }
            double sum = 0;
            for (int64_t l = 0; l < L; ++l) {
                sc[l] = mp[l] ? std::exp(sc[l] - mx) : 0.0f;
                sum += sc[l];
            }
            float* o = att.data() + hq * head_dim;
            for (int64_t i = 0; i < head_dim; ++i) o[i] = 0;
            for (int64_t l = 0; l < L; ++l) {
                if (sc[l] == 0) continue;
                const float p = round_bf((float)(sc[l] / sum));
                const bf16_t* vr = vb + l * head_dim;
                if (vec) {  // same per-element order as the scalar loop
                    const __m512 pv = _mm512_set1_ps(p);
                    for (int64_t i = 0; i < head_dim; i += 16)
                        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(pv, bf16x16(vr + i), _mm512_loadu_ps(o + i)));
                } else {
                    for (int64_t i = 0; i < head_dim; ++i) o[i] += p * bf2f(vr[i]);
                }
            }
            for (int64_t i = 0; i < head_dim; ++i) o[i] = round_bf(o[i]);
        }
    });

    std::vector<float> y(D);
    gemv1(att.data(), qs, wo_p, wo_a, code, D, y.data(), xbuf, ybuf);
    for (int64_t i = 0; i < D; ++i) xf[i] = round_bf(xf[i] + y[i]);  // h = x + attn

    rms_norm(xf.data(), D, (const bf16_t*)ffn_norm_w.data_ptr(), (float)eps, h.data());
    std::vector<float> ab(2 * inter);
    gemv1(h.data(), D, w13_p, w13_a, code, 2 * inter, ab.data(), xbuf, ybuf);
    silu_mul(ab.data(), ab.data() + inter, inter, ab.data());
    gemv1(ab.data(), inter, w2_p, w2_a, code, D, y.data(), xbuf, ybuf);

    auto out = torch::empty_like(x);
    bf16_t* op = (bf16_t*)out.data_ptr();
    for (int64_t i = 0; i < D; ++i) op[i] = f2bf(xf[i] + y[i]);
    return out;
}

// M rows of fp32 activations through one NF4 weight, rounding the input to
// bf16 first exactly as `gemv1` does: the vector kernel for a few rows, AMX
// once there are enough rows for its fixed cost to pay off.
static void matmul_rows(const float* x, int64_t M, int64_t K, const torch::Tensor& packed,
                        const torch::Tensor& absmax, const torch::Tensor& code16, int64_t n_out,
                        float* y) {
    auto xb = torch::empty({M, K}, torch::kBFloat16);
    bf16_t* xp = (bf16_t*)xb.data_ptr();
    for (int64_t i = 0; i < M * K; ++i) xp[i] = f2bf(x[i]);
    torch::Tensor yt;
    if (M >= 7 && amx_ready()) {
        yt = nf4_amx(xb, packed, absmax, code16, n_out);
    } else {
        yt = nf4_gemv(xb, packed, absmax, code16, n_out);
    }
    const bf16_t* yp = (const bf16_t*)yt.data_ptr();
    bf16_to_f32(yp, M * n_out, y);
}

// `decode_block` for a short run of consecutive positions [pos, pos + M): a
// prompt prefill of up to a few dozen tokens, which is what a cached voice
// prefix and a speculative hint leave for each real segment. Same operations
// and rounding points per position as `decode_block`; position m attends
// under row m of `mask` (M, L). Replaces ~60 aten calls per layer per call,
// which cost as much as the matmuls themselves at these sizes.
torch::Tensor prefill_block(
    torch::Tensor x, torch::Tensor attn_norm_w, torch::Tensor ffn_norm_w,
    torch::Tensor q_norm_w, torch::Tensor k_norm_w, double eps, double qk_eps,
    torch::Tensor qkv_p, torch::Tensor qkv_a, torch::Tensor wo_p, torch::Tensor wo_a,
    torch::Tensor w13_p, torch::Tensor w13_a, torch::Tensor w2_p, torch::Tensor w2_a,
    torch::Tensor code16, torch::Tensor k_cache, torch::Tensor v_cache, int64_t pos,
    torch::Tensor freqs, torch::Tensor mask, int64_t n_head, int64_t n_local, int64_t head_dim,
    int64_t inter) {
    const int64_t M = x.size(-2), D = x.size(-1);
    const int64_t qs = n_head * head_dim, kvs = n_local * head_dim, QKV = qs + 2 * kvs;

    std::vector<float> xf(M * D), h(M * D), qkv(M * QKV);
    const bf16_t* xin = (const bf16_t*)x.contiguous().data_ptr();
    for (int64_t i = 0; i < M * D; ++i) xf[i] = bf2f(xin[i]);
    const bf16_t* anw = (const bf16_t*)attn_norm_w.data_ptr();
    for (int64_t m = 0; m < M; ++m) rms_norm(xf.data() + m * D, D, anw, (float)eps, h.data() + m * D);
    matmul_rows(h.data(), M, D, qkv_p, qkv_a, code16, QKV, qkv.data());

    auto fr = freqs.to(torch::kFloat32).contiguous();  // (M, head_dim/2, 2)
    const float* fbase = fr.data_ptr<float>();
    const int64_t max_len = k_cache.size(2);
    bf16_t* kc = (bf16_t*)k_cache.data_ptr();
    bf16_t* vc = (bf16_t*)v_cache.data_ptr();
    for (int64_t m = 0; m < M; ++m) {
        float* q = qkv.data() + m * QKV;
        float* k = q + qs;
        float* v = k + kvs;
        if (q_norm_w.numel() > 0) {
            auto norm_heads = [&](float* t, int64_t heads, const torch::Tensor& w) {
                const bf16_t* wp = (const bf16_t*)w.data_ptr();
                for (int64_t hd = 0; hd < heads; ++hd) {
                    float* r = t + hd * head_dim;
                    double ss = 0;
                    for (int64_t i = 0; i < head_dim; ++i) ss += (double)r[i] * r[i];
                    const float inv = 1.0f / std::sqrt((float)(ss / head_dim) + (float)qk_eps);
                    for (int64_t i = 0; i < head_dim; ++i) r[i] = round_bf(r[i] * inv * bf2f(wp[i]));
                }
            };
            norm_heads(q, n_head, q_norm_w);
            norm_heads(k, n_local, k_norm_w);
        }
        const float* f = fbase + m * head_dim;
        auto rope = [&](float* t, int64_t heads) {
            for (int64_t hd = 0; hd < heads; ++hd) {
                float* r = t + hd * head_dim;
                for (int64_t i = 0; i < head_dim / 2; ++i) {
                    const float a = r[2 * i], b = r[2 * i + 1], c = f[2 * i], sn = f[2 * i + 1];
                    r[2 * i] = round_bf(a * c - b * sn);
                    r[2 * i + 1] = round_bf(b * c + a * sn);
                }
            }
        };
        rope(q, n_head);
        rope(k, n_local);
        for (int64_t hd = 0; hd < n_local; ++hd)
            for (int64_t i = 0; i < head_dim; ++i) {
                kc[(hd * max_len + pos + m) * head_dim + i] = f2bf(k[hd * head_dim + i]);
                vc[(hd * max_len + pos + m) * head_dim + i] = f2bf(v[hd * head_dim + i]);
            }
    }

    const int64_t L = mask.size(-1);
    auto mk = mask.to(torch::kBool).contiguous();
    const bool* mbase = mk.data_ptr<bool>();
    const int64_t groups = n_head / n_local;
    const float scale = 1.0f / std::sqrt((float)head_dim);
    const bool vec = head_dim % 32 == 0;
    std::vector<float> att(M * qs);
    at::parallel_for(0, n_head * M, 1, [&](int64_t t0, int64_t t1) {
        std::vector<float> sc(L);
        for (int64_t t = t0; t < t1; ++t) {
            const int64_t hq = t / M, m = t % M;
            const bool* mp = mbase + m * L;
            const int64_t hk = hq / groups;
            const float* qr = qkv.data() + m * QKV + hq * head_dim;
            const bf16_t* kb = kc + hk * max_len * head_dim;
            const bf16_t* vb = vc + hk * max_len * head_dim;
            float mx = -INFINITY;
            for (int64_t l = 0; l < L; ++l) {
                if (!mp[l]) { sc[l] = -INFINITY; continue; }
                const bf16_t* kr = kb + l * head_dim;
                float d;
                if (vec) {
                    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
                    for (int64_t i = 0; i < head_dim; i += 32) {
                        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(qr + i), bf16x16(kr + i), s0);
                        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(qr + i + 16), bf16x16(kr + i + 16), s1);
                    }
                    d = _mm512_reduce_add_ps(_mm512_add_ps(s0, s1));
                } else {
                    d = 0;
                    for (int64_t i = 0; i < head_dim; ++i) d += qr[i] * bf2f(kr[i]);
                }
                sc[l] = round_bf(d) * scale;
                mx = std::max(mx, sc[l]);
            }
            double sum = 0;
            for (int64_t l = 0; l < L; ++l) {
                sc[l] = mp[l] ? std::exp(sc[l] - mx) : 0.0f;
                sum += sc[l];
            }
            float* o = att.data() + m * qs + hq * head_dim;
            for (int64_t i = 0; i < head_dim; ++i) o[i] = 0;
            for (int64_t l = 0; l < L; ++l) {
                if (sc[l] == 0) continue;
                const float pr = round_bf((float)(sc[l] / sum));
                const bf16_t* vr = vb + l * head_dim;
                if (vec) {
                    const __m512 pv = _mm512_set1_ps(pr);
                    for (int64_t i = 0; i < head_dim; i += 16)
                        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(pv, bf16x16(vr + i), _mm512_loadu_ps(o + i)));
                } else {
                    for (int64_t i = 0; i < head_dim; ++i) o[i] += pr * bf2f(vr[i]);
                }
            }
            for (int64_t i = 0; i < head_dim; ++i) o[i] = round_bf(o[i]);
        }
    });

    std::vector<float> y(M * D);
    matmul_rows(att.data(), M, qs, wo_p, wo_a, code16, D, y.data());
    for (int64_t i = 0; i < M * D; ++i) xf[i] = round_bf(xf[i] + y[i]);  // h = x + attn

    const bf16_t* fnw = (const bf16_t*)ffn_norm_w.data_ptr();
    for (int64_t m = 0; m < M; ++m) rms_norm(xf.data() + m * D, D, fnw, (float)eps, h.data() + m * D);
    std::vector<float> ab(M * 2 * inter), g(M * inter);
    matmul_rows(h.data(), M, D, w13_p, w13_a, code16, 2 * inter, ab.data());
    for (int64_t m = 0; m < M; ++m)
        silu_mul(ab.data() + m * 2 * inter, ab.data() + m * 2 * inter + inter, inter,
                 g.data() + m * inter);
    matmul_rows(g.data(), M, inter, w2_p, w2_a, code16, D, y.data());

    auto out = torch::empty_like(x);
    bf16_t* op = (bf16_t*)out.data_ptr();
    for (int64_t i = 0; i < M * D; ++i) op[i] = f2bf(xf[i] + y[i]);
    return out;
}
"""

_lock = threading.Lock()
_ext = None
_error: str | None = None
_amx: bool | None = None


def _build():
    global _ext, _error
    with _lock:
        if _ext is not None or _error is not None:
            return _ext
        try:
            flags = open("/proc/cpuinfo").read()
            if " avx512_bf16 " not in flags or " avx512bw " not in flags:
                raise RuntimeError("CPU lacks AVX512-BF16/BW")
            from torch.utils.cpp_extension import load_inline

            build_dir = Path(os.environ.get(
                "TTS_CPU_KERNEL_DIR", Path(__file__).resolve().parents[2] / ".kernels"
            ))
            build_dir.mkdir(parents=True, exist_ok=True)
            _ext = load_inline(
                name="nf4cpu_v12",
                cpp_sources=_SOURCE,
                functions=["nf4_gemv", "nf4_amx", "amx_available", "decode_block", "prefill_block"],
                extra_cflags=["-O3", "-mavx512f", "-mavx512bw", "-mavx512bf16",
                              "-mamx-tile", "-mamx-bf16", "-fopenmp"],
                extra_ldflags=["-fopenmp"],
                build_directory=str(build_dir),
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on toolchain
            _error = f"{type(exc).__name__}: {exc}"
            logger.warning("NF4 CPU kernel unavailable (%s); using dense weights", _error)
        return _ext


def available() -> bool:
    return _build() is not None


def amx_available() -> bool:
    """AMX tiles usable here: the CPU has them and Linux granted this process
    permission to use them (`arch_prctl(ARCH_REQ_XCOMP_PERM)`)."""
    global _amx
    if _amx is None:
        ext = _build()
        try:
            flags = open("/proc/cpuinfo").read()
            _amx = bool(ext is not None and " amx_bf16 " in flags and ext.amx_available())
        except Exception:  # noqa: BLE001 - an optimisation, never a failure
            _amx = False
    return _amx


def make_linear(child, dtype):
    """An nn.Module serving `child` (a bitsandbytes Linear4bit): single and
    few-position calls through the NF4 kernel, longer ones through a dense
    copy of the same values."""
    import bitsandbytes.functional as F
    import torch
    import torch.nn as nn

    ext = _build()
    w = child.weight
    qs = w.quant_state
    absmax = qs.absmax
    if qs.nested:
        absmax = F.dequantize_blockwise(qs.absmax, qs.state2) + qs.offset
    n, k = child.out_features, child.in_features

    class NF4Linear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features, self.out_features = k, n
            self.register_buffer("packed", w.data.reshape(n, k // 2).contiguous())
            self.register_buffer("absmax", absmax.to(dtype).reshape(n, k // 64).contiguous())
            self.register_buffer("code", qs.code.to(torch.bfloat16).contiguous())
            dense = F.dequantize_4bit(w.data, qs).to(dtype).contiguous()
            self.register_buffer("dense", dense)
            self.register_buffer(
                "bias", None if child.bias is None else child.bias.data.to(dtype)
            )

        def forward(self, x):
            shape = x.shape
            rows = x.reshape(-1, k)
            if rows.shape[0] <= MAX_ROWS:
                y = ext.nf4_gemv(rows, self.packed, self.absmax, self.code, self.out_features)
                if self.bias is not None:
                    y = y + self.bias
            elif rows.shape[0] <= AMX_MAX_ROWS and amx_available():
                y = ext.nf4_amx(rows, self.packed, self.absmax, self.code, self.out_features)
                if self.bias is not None:
                    y = y + self.bias
            else:
                y = torch.nn.functional.linear(rows.to(self.dense.dtype), self.dense, self.bias)
            return y.view(*shape[:-1], self.out_features)

    return NF4Linear()


def fuse_rows(first, second):
    """One NF4Linear computing `first` and `second` stacked along the output
    (same input): one pass over x and one call instead of two."""
    import copy

    import torch

    fused = copy.copy(first)
    fused._buffers = dict(first._buffers)
    for name in ("packed", "absmax", "dense"):
        fused._buffers[name] = torch.cat([first._buffers[name], second._buffers[name]]).contiguous()
    if first.bias is not None or second.bias is not None:
        raise RuntimeError("fuse_rows: biased layers are not fused")
    fused.out_features = first.out_features + second.out_features
    return fused
