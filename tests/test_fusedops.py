"""The fused RMS norm and rotary embedding, against the fork's own versions.

Both of these replace a chain of PyTorch operations with one kernel doing the
same arithmetic in the same precision, so the bar here is tighter than "close
enough". Rotary embedding has no reduction in it and comes out bit for bit
identical, which `test_rope_is_bit_exact` asserts outright. The norm sums its
squares in the kernel's own order rather than PyTorch's tree reduction, so a
few elements per thousand can land on the other side of an fp16 rounding
boundary; `test_rms_norm_matches_the_fork` pins the agreement at 99.5% of
elements identical and no element more than a couple of ulps out.

Those thresholds are chosen to catch the shortcuts that would otherwise pass
unnoticed, because both still sound fine: accumulating the sum of squares in
fp16 leaves 94% of elements identical, and skipping the round to fp16 before
the weight multiply leaves 71%. This kernel leaves 99.7%.

The other half of the file is about what the kernels decline. Every shape they
cannot serve has to reach the fork's code and come back identical, because that
is what makes this an optimisation rather than a rewrite - a prefill of several
positions hands them a non-contiguous tensor, and the codec runs in fp32.

Needs CUDA and the vendored fork; skipped without either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="these kernels are CUDA-only"
)


@pytest.fixture(scope="module")
def fork():
    from app.config import get_settings
    from app.engine.factory import _ensure_vendor_on_path

    try:
        _ensure_vendor_on_path(get_settings())
        from fish_speech.models.text2semantic.llama import RMSNorm, apply_rotary_emb
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"vendored fish-speech not importable: {exc}")
    return RMSNorm, apply_rotary_emb


@pytest.fixture(scope="module")
def fused():
    from app.engine import fusedops

    if fusedops._built() is None:
        pytest.skip("fused kernels could not be built on this device")
    return fusedops


def make_norm(fork, width: int):
    RMSNorm, _ = fork
    norm = RMSNorm(width, eps=1e-6).to("cuda").half()
    with torch.no_grad():
        norm.weight.copy_(torch.randn(width, device="cuda").half())
    return norm


def reference_norm(fork, norm, x):
    RMSNorm, _ = fork
    original = getattr(RMSNorm, "_fused_original_forward", RMSNorm.forward)
    return original(norm, x)


# --------------------------------------------------------------------------
# RMS norm
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,width",
    [
        (1, 2560),   # attention_norm / ffn_norm, the model width
        (32, 128),   # q_norm, one row per query head
        (8, 128),    # k_norm, one row per KV head
        (1, 9728),   # a row wider than one block's stride
        (3, 2560),   # several positions at once
        (1, 40),     # narrower than a warp
    ],
)
def test_rms_norm_matches_the_fork(fork, fused, rows, width):
    torch.manual_seed(rows * 1000 + width)
    norm = make_norm(fork, width)
    x = torch.randn(1, rows, width, device="cuda", dtype=torch.float16)

    ours = fused.rms_norm(x, norm.weight, norm.eps)
    theirs = reference_norm(fork, norm, x)

    assert ours is not None
    assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
    identical = (ours == theirs).float().mean().item()
    scale = theirs.float().abs().clamp(min=1e-4)
    worst = ((ours.float() - theirs.float()).abs() / scale).max().item()
    assert identical > 0.995, f"only {identical:.4f} of elements identical"
    assert worst < 2e-3, f"worst relative difference {worst:.2e}"


def test_rms_norm_declines_fp32(fork, fused):
    """The codec runs in fp32 on this card and must keep getting the fork's."""
    norm = make_norm(fork, 256).float()
    x = torch.randn(1, 1, 256, device="cuda", dtype=torch.float32)
    assert fused.rms_norm(x, norm.weight, norm.eps) is None


def test_rms_norm_declines_a_non_contiguous_row(fork, fused):
    norm = make_norm(fork, 128)
    wide = torch.randn(1, 4, 256, device="cuda", dtype=torch.float16)
    assert fused.rms_norm(wide[..., :128], norm.weight, norm.eps) is None


def test_patched_module_falls_back_without_changing_the_answer(fork, fused):
    """Installing the patch must not change what an unsupported shape returns."""
    RMSNorm, _ = fork
    assert fused.install()
    norm = make_norm(fork, 256).float()
    x = torch.randn(1, 2, 256, device="cuda", dtype=torch.float32)

    before = fused.stats()["norm_fallback"]
    ours = norm(x)
    assert fused.stats()["norm_fallback"] == before + 1
    assert torch.equal(ours, RMSNorm._fused_original_forward(norm, x))


# --------------------------------------------------------------------------
# rotary embedding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 128),  # q at a decode step
        (1, 1, 8, 128),   # k at a decode step
        (1, 4, 32, 128),  # several positions
        (2, 3, 8, 64),    # batched, narrower heads
    ],
)
@pytest.mark.parametrize("freq_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_rope_is_bit_exact(fork, fused, shape, freq_dtype):
    _, apply_rotary_emb = fork
    bsz, seq, heads, head_dim = shape
    torch.manual_seed(hash(shape) % 9973)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    freqs = torch.randn(seq, head_dim // 2, 2, device="cuda", dtype=freq_dtype)

    ours = fused.rope(x, freqs)
    theirs = apply_rotary_emb(x, freqs)

    assert ours is not None
    assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
    assert torch.equal(ours, theirs)


def test_rope_uses_the_frequencies_of_its_own_position(fork, fused):
    """Position s must read row s, not row 0 broadcast across the sequence.

    Getting this wrong leaves the first position right and every later one
    subtly mis-rotated, which degrades a voice rather than breaking it.
    """
    _, apply_rotary_emb = fork
    seq, heads, head_dim = 5, 2, 8
    x = torch.randn(1, seq, heads, head_dim, device="cuda", dtype=torch.float16)
    freqs = torch.randn(seq, head_dim // 2, 2, device="cuda", dtype=torch.bfloat16)

    whole = fused.rope(x, freqs)
    for s in range(seq):
        alone = apply_rotary_emb(x[:, s : s + 1], freqs[s : s + 1])
        assert torch.equal(whole[:, s : s + 1], alone)


def test_rope_declines_a_non_contiguous_input(fused):
    """A prefill splits wqkv into views; those go to the fork's own code."""
    packed = torch.randn(1, 4, 32, 256, device="cuda", dtype=torch.float16)
    freqs = torch.randn(4, 64, 2, device="cuda", dtype=torch.bfloat16)
    assert fused.rope(packed[..., :128], freqs) is None


def test_rope_declines_mismatched_frequencies(fused):
    x = torch.randn(1, 4, 32, 128, device="cuda", dtype=torch.float16)
    short = torch.randn(2, 64, 2, device="cuda", dtype=torch.bfloat16)
    assert fused.rope(x, short) is None
