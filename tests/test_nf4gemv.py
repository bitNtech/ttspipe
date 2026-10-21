"""The 4-bit GEMV and the grouped decode attention, against what they replace.

Both of these exist only to be faster, so the thing worth pinning is that they
did not change the answer. Every test here compares against the implementation
that was there before - bitsandbytes' own `Linear4bit.forward`, and
`scaled_dot_product_attention` with `enable_gqa=True` - rather than against
values recorded from a previous run, so an upstream change that moves the
reference moves the test with it.

The one agreement that is not merely numerical is the weight layout: the kernel
reads bitsandbytes' `Params4bit` buffer in place, which is only safe while NF4
means "two 4-bit indices per byte, high nibble first, over the weight flattened
row-major". `test_packing_layout_is_what_the_kernel_assumes` states that as a
property of bitsandbytes rather than of us, so a release that repacked would
fail here instead of quietly producing noise.

Skipped wholesale without CUDA: there is no CPU path to test, and the fallback
these guard is bitsandbytes itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the kernel and its fallback are both CUDA-only"
)


@pytest.fixture(scope="module")
def installed():
    from app.engine import nf4gemv

    if not nf4gemv.install():
        pytest.skip("nf4 gemv kernel could not be built on this device")
    return nf4gemv


def make_layer(out_features: int, in_features: int, *, nested: bool = True, bias: bool = False):
    layer = bnb.nn.Linear4bit(
        in_features,
        out_features,
        bias=bias,
        compute_dtype=torch.float16,
        quant_type="nf4",
        compress_statistics=nested,
    ).to("cuda")
    return layer


def relative_error(got, ref) -> float:
    scale = ref.float().abs().max().item()
    return (got.float() - ref.float()).abs().max().item() / max(scale, 1e-6)


# --------------------------------------------------------------------------
# the layout the kernel reads
# --------------------------------------------------------------------------


def test_packing_layout_is_what_the_kernel_assumes():
    """NF4 packs two indices per byte, high nibble first, row-major flattened."""
    import bitsandbytes.functional as BF

    from app.engine.nf4gemv import NF4_CODE

    out_features, in_features = 8, 256
    weight = torch.randn(out_features, in_features, device="cuda", dtype=torch.float16)
    packed, state = BF.quantize_4bit(
        weight, blocksize=64, compress_statistics=False, quant_type="nf4"
    )
    dequantized = BF.dequantize_4bit(packed, state).reshape(out_features, in_features)

    code = torch.tensor(NF4_CODE, device="cuda")
    scales = state.absmax.reshape(out_features, in_features // 64)
    ratio = dequantized.float().reshape(out_features, -1, 64) / scales.float().clamp(
        min=1e-20
    ).unsqueeze(-1)
    boundaries = (code[:-1] + code[1:]) / 2
    index = torch.bucketize(ratio.reshape(out_features, in_features), boundaries)

    flat = packed.reshape(-1)
    assert (flat >> 4).long().equal(index.reshape(-1)[0::2])
    assert (flat & 0xF).long().equal(index.reshape(-1)[1::2])


def test_nested_scales_are_what_the_kernel_reconstructs():
    """The kernel undoes bitsandbytes' double quantisation with code2[a8]*s + offset."""
    import bitsandbytes.functional as BF

    weight = torch.randn(64, 512, device="cuda", dtype=torch.float16)
    _, state = BF.quantize_4bit(
        weight, blocksize=64, compress_statistics=True, quant_type="nf4"
    )
    reference = BF.dequantize_blockwise(state.absmax, state.state2) + state.offset

    blocks = torch.arange(state.absmax.numel(), device="cuda")
    ours = (
        state.state2.code[state.absmax.long()]
        * state.state2.absmax[blocks // state.state2.blocksize]
        + state.offset
    )
    assert torch.equal(reference, ours)


# --------------------------------------------------------------------------
# the kernel against bitsandbytes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (6144, 2560),  # wqkv
        (2560, 4096),  # wo
        (9728, 2560),  # w1 / w3
        (2560, 9728),  # w2
        (1544, 2560),  # rows that do not divide into 16-row tiles
    ],
)
@pytest.mark.parametrize("nested", [True, False])
def test_matches_bitsandbytes_on_one_vector(installed, shape, nested):
    layer = make_layer(*shape, nested=nested)
    x = torch.randn(1, 1, shape[1], device="cuda", dtype=torch.float16) * 0.5

    ours = layer(x)
    theirs = bnb.nn.Linear4bit._nf4_original_forward(layer, x)

    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    # Both sum the same products in a different order, in fp16: agreement is to
    # rounding, not to the bit.
    assert relative_error(ours, theirs) < 5e-3


def test_bias_is_applied(installed):
    layer = make_layer(512, 2560, bias=True)
    with torch.no_grad():
        layer.bias.copy_(torch.randn_like(layer.bias))
    x = torch.randn(1, 1, 2560, device="cuda", dtype=torch.float16)

    ours = layer(x)
    theirs = bnb.nn.Linear4bit._nf4_original_forward(layer, x)
    assert relative_error(ours, theirs) < 5e-3


@pytest.mark.parametrize("shape", [(1, 1, 2560), (1, 2560), (2560,)])
def test_every_single_vector_shape_is_served(installed, shape):
    layer = make_layer(512, 2560)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)

    before = installed.stats()["fast"]
    ours = layer(x)
    assert installed.stats()["fast"] == before + 1

    theirs = bnb.nn.Linear4bit._nf4_original_forward(layer, x)
    assert ours.shape == theirs.shape
    assert relative_error(ours, theirs) < 5e-3


@pytest.mark.parametrize("rows", [2, 5, 8, 9, 16, 19, 32, 33, 70])
@pytest.mark.parametrize("shape", [(6144, 2560), (2560, 9728), (1544, 2560)])
def test_several_positions_match_bitsandbytes(installed, shape, rows):
    """A prefill goes through the same kernel, in slices of MAX_N past 32 rows.

    bitsandbytes' own path for this shape dequantises to fp16 and runs a dense
    GEMM, which is exactly the arithmetic the kernel reproduces tile by tile.
    """
    layer = make_layer(*shape)
    x = torch.randn(1, rows, shape[1], device="cuda", dtype=torch.float16) * 0.5

    before = installed.stats()["fast"]
    ours = layer(x)
    assert installed.stats()["fast"] == before + 1

    theirs = bnb.nn.Linear4bit._nf4_original_forward(layer, x)
    assert ours.shape == theirs.shape
    assert relative_error(ours, theirs) < 5e-3


@pytest.mark.parametrize("epi", ["residual", "gate"])
def test_fused_epilogues_match_the_unfused_ops(installed, epi):
    import torch.nn.functional as F

    layer = make_layer(2560, 9728)
    x = torch.randn(1, 1, 9728, device="cuda", dtype=torch.float16) * 0.5
    operand = torch.randn(1, 1, 2560, device="cuda", dtype=torch.float16)
    plain = bnb.nn.Linear4bit._nf4_original_forward(layer, x)
    assert installed.prepare(layer)
    if epi == "residual":
        ours = installed.matmul(layer, x, epi=installed.EPI_RESIDUAL, operand=operand)
        theirs = operand + plain
    else:
        ours = installed.matmul(layer, x, epi=installed.EPI_SILU_GATE, operand=operand)
        theirs = F.silu(operand) * plain
    assert relative_error(ours, theirs) < 5e-3


def test_unsupported_layer_is_declined_not_broken(installed):
    """An in_features that does not divide into scale blocks goes to bitsandbytes."""
    layer = make_layer(256, 2560)
    layer.weight.quant_state.blocksize = 96  # not a power of two

    x = torch.randn(1, 1, 2560, device="cuda", dtype=torch.float16)
    assert not installed.prepare(layer)
    before = installed.stats()["fallback"]
    layer(x)
    assert installed.stats()["fallback"] == before + 1


# --------------------------------------------------------------------------
# grouped decode attention
# --------------------------------------------------------------------------


@pytest.mark.parametrize("kv_len", [1, 10, 17, 401, 2048])
def test_decode_attention_matches_sdpa(kv_len):
    """The hand-written grouped path equals SDPA's own enable_gqa result."""
    import torch.nn.functional as F

    from app.engine.fastpath import decode_attention

    n_head, n_local_heads, head_dim = 32, 8, 128
    torch.manual_seed(0)
    q = torch.randn(1, n_head, 1, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    mask = torch.ones(1, 1, 1, kv_len, device="cuda", dtype=torch.bool)

    ours = decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim)
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    assert ours.shape == theirs.shape
    assert relative_error(ours, theirs) < 5e-3


@pytest.mark.parametrize("seqlen,prefix", [(2, 0), (8, 351), (19, 300), (64, 10)])
def test_prefill_attention_matches_sdpa(seqlen, prefix):
    """A short prefill, causal within itself, over an already-cached prefix."""
    import torch.nn.functional as F

    from app.engine.fastpath import decode_attention

    n_head, n_local_heads, head_dim = 32, 8, 128
    kv_len = prefix + seqlen
    torch.manual_seed(2)
    q = torch.randn(1, n_head, seqlen, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    causal = torch.ones(kv_len, kv_len, device="cuda", dtype=torch.bool).tril()
    mask = causal[prefix:][None, None]

    ours = decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim)
    theirs = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)

    assert ours.shape == theirs.shape
    assert relative_error(ours, theirs) < 5e-3


def test_decode_attention_honours_the_mask():
    """Masked-out keys contribute nothing, whatever they hold."""
    import torch.nn.functional as F

    from app.engine.fastpath import decode_attention

    n_head, n_local_heads, head_dim, kv_len = 32, 8, 128, 64
    torch.manual_seed(1)
    q = torch.randn(1, n_head, 1, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    mask = torch.ones(1, 1, 1, kv_len, device="cuda", dtype=torch.bool)
    mask[..., 40:] = False

    ours = decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim)

    # Whatever is behind the mask is not read: replacing it must not move the
    # answer, which a naive implementation that only zeroed the scores fails.
    k[:, :, 40:] = 1e3
    v[:, :, 40:] = -1e3
    again = decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim)
    assert torch.equal(ours, again)

    truncated = F.scaled_dot_product_attention(
        q, k[:, :, :40], v[:, :, :40], enable_gqa=True
    )
    assert relative_error(ours, truncated) < 5e-3


def test_query_heads_map_onto_the_right_kv_head():
    """Grouping is interleaved: query head i reads KV head i // groups.

    A reshape that got this backwards would still produce plausible audio, so
    it is checked directly rather than through a norm.
    """
    from app.engine.fastpath import decode_attention

    n_head, n_local_heads, head_dim, kv_len = 8, 2, 4, 3
    groups = n_head // n_local_heads
    q = torch.zeros(1, n_head, 1, head_dim, device="cuda", dtype=torch.float16)
    k = torch.zeros(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.zeros(1, n_local_heads, kv_len, head_dim, device="cuda", dtype=torch.float16)
    mask = torch.ones(1, 1, 1, kv_len, device="cuda", dtype=torch.bool)

    # Each KV head carries its own index as the value; attention over identical
    # keys is a uniform average, so each query head must read back its group's.
    for h in range(n_local_heads):
        v[0, h] = float(h + 1)

    out = decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim)
    for head in range(n_head):
        assert out[0, head, 0, 0].item() == pytest.approx(head // groups + 1)
