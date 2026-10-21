"""The fused decode block and the streaming codec, against what they replace.

Both exist only to be faster, so what is pinned is that they compute the same
thing as the fork's own code: `fusedlayer.decode_block` against the fork's
`TransformerBlock.forward` (bitsandbytes matmuls, its norms, rotary and
attention), and the codec's causal convolutions run as a stream against the
same stack run over the whole sequence at once.

CUDA only: every piece here is a CUDA kernel or a CUDA-graph path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA kernels")


def relative_error(got, ref) -> float:
    scale = ref.float().abs().max().item()
    return (got.float() - ref.float()).abs().max().item() / max(scale, 1e-6)


@pytest.fixture(scope="module")
def kernels():
    from app.engine import fastpath, fusedlayer, fusedops, nf4gemv

    if not (nf4gemv.install() and fusedops.install()):
        pytest.skip("kernels could not be built on this device")
    fastpath.install()
    if not fusedlayer.install():
        pytest.skip("fused decode block could not be built")
    return fusedlayer


def _block(qk_norm: bool):
    from fish_speech.models.text2semantic.llama import BaseModelArgs, KVCache, TransformerBlock

    config = BaseModelArgs(
        dim=512, n_head=4, n_local_heads=2, head_dim=128, intermediate_size=1024,
        attention_qk_norm=qk_norm, norm_eps=1e-6,
    )
    torch.manual_seed(0)
    block = TransformerBlock(config).half()
    for parent in (block.attention, block.feed_forward):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                q = bnb.nn.Linear4bit(
                    child.in_features, child.out_features, bias=False,
                    compute_dtype=torch.float16, quant_type="nf4", compress_statistics=True,
                )
                q.weight = bnb.nn.Params4bit(child.weight.data.float() * 0.05, requires_grad=False, quant_type="nf4")
                setattr(parent, name, q)
    block = block.cuda()
    for norm in [block.attention_norm, block.ffn_norm] + (
        [block.attention.q_norm, block.attention.k_norm] if qk_norm else []
    ):
        with torch.no_grad():
            norm.weight.copy_(torch.rand_like(norm.weight) + 0.5)
    block.attention.kv_cache = KVCache(1, 256, 2, 128, dtype=torch.float16).cuda()
    return block


@pytest.mark.parametrize("qk_norm", [True, False])
@pytest.mark.parametrize("pos,window", [(0, 1), (37, 64), (200, 256)])
def test_decode_block_matches_the_forks_block(kernels, qk_norm, pos, window):
    from fish_speech.models.text2semantic.llama import TransformerBlock, precompute_freqs_cis

    block = _block(qk_norm)
    cache = block.attention.kv_cache
    with torch.no_grad():
        cache.k_cache.normal_()
        cache.v_cache.normal_()
    freqs = precompute_freqs_cis(256, 128, 10000).cuda()
    x = torch.randn(1, 1, 512, device="cuda", dtype=torch.float16)
    ip = torch.tensor([pos], device="cuda", dtype=torch.long)
    mask = torch.ones(256, 256, dtype=torch.bool, device="cuda").tril()[None, None, ip, :window]
    f = freqs[ip]

    k0, v0 = cache.k_cache.clone(), cache.v_cache.clone()
    with torch.inference_mode():
        theirs = TransformerBlock._fused_original_forward(block, x, f, mask, ip)
        k_ref, v_ref = cache.k_cache.clone(), cache.v_cache.clone()
        cache.k_cache.copy_(k0)
        cache.v_cache.copy_(v0)
        ours = kernels.decode_block(block, x, f, mask, ip)

    assert ours is not None
    assert relative_error(ours, theirs) < 5e-3
    # The cache write is the one side effect, and it must be the same one - to
    # rounding, since the matmul's split-K sums land in no fixed order.
    assert relative_error(cache.k_cache[:, :, pos], k_ref[:, :, pos]) < 1e-3
    assert relative_error(cache.v_cache[:, :, pos], v_ref[:, :, pos]) < 1e-3


def test_decode_block_declines_a_prefill(kernels):
    block = _block(True)
    x = torch.randn(1, 3, 512, device="cuda", dtype=torch.float16)
    ip = torch.arange(3, device="cuda")
    mask = torch.ones(1, 1, 3, 3, dtype=torch.bool, device="cuda")
    assert kernels.decode_block(block, x, None, mask, ip) is None


# --------------------------------------------------------------------------
# the streaming codec's causal stack
# --------------------------------------------------------------------------


class _Stream:
    """What the patched convolutions need from a StreamingCodec: somewhere to keep state."""

    def __init__(self):
        self._states = []

    def _conv_state(self, module, x):
        from app.engine.codecstream import StreamingCodec

        return StreamingCodec._conv_state(self, module, x)


def test_streamed_causal_stack_equals_one_pass():
    """Carried state reproduces a single pass over the whole sequence, whatever the chunking."""
    from fish_speech.models.dac import modded_dac, rvq

    from app.engine import codecstream

    codecstream._install_streaming_convs()
    torch.manual_seed(0)
    stack = torch.nn.Sequential(
        modded_dac.CausalConvNet(8, 16, kernel_size=7),
        modded_dac.CausalTransConvNet(16, 8, kernel_size=8, stride=4),
        modded_dac.CausalConvNet(8, 8, kernel_size=7, dilation=3),
        rvq.CausalTransConvNet(8, 4, kernel_size=2, stride=2),
        rvq.CausalConvNet(4, 4, kernel_size=7, groups=4),
    ).cuda()
    x = torch.randn(1, 8, 40, device="cuda")
    with torch.no_grad():
        whole = stack(x)
        for chunks in ([1] * 40, [3, 1, 7, 13, 16], [40]):
            codecstream._streaming.codec = _Stream()
            try:
                pieces, start = [], 0
                for n in chunks:
                    pieces.append(stack(x[..., start : start + n]))
                    start += n
            finally:
                codecstream._streaming.codec = None
                for m in stack.modules():
                    if hasattr(m, "_stream_state"):
                        del m._stream_state
            assert torch.allclose(torch.cat(pieces, -1), whole, atol=1e-5), chunks


def test_unstreamed_convs_are_untouched():
    from fish_speech.models.dac import modded_dac

    from app.engine import codecstream

    codecstream._install_streaming_convs()
    conv = modded_dac.CausalConvNet(4, 4, kernel_size=7).cuda()
    x = torch.randn(1, 4, 20, device="cuda")
    with torch.no_grad():
        padded = conv.conv(torch.nn.functional.pad(x, (6, 0)))
        assert torch.equal(conv(x), padded)
