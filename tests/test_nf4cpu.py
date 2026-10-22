"""The CPU NF4 kernels (`app/engine/nf4cpu.py`) against what they replace.

Both exist only to be faster: `nf4_gemv` against the same product in fp32
from the dequantised weight, and `decode_block` against the fork's own
`TransformerBlock.forward` on the same NF4 weights. Skipped on a CPU without
AVX512-BF16, where the server serves dense bf16 weights instead.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "fish-speech"))

torch = pytest.importorskip("torch")
bnb = pytest.importorskip("bitsandbytes")
pytest.importorskip("fish_speech")

from app.engine import nf4cpu  # noqa: E402

pytestmark = pytest.mark.skipif(not nf4cpu.available(), reason="needs AVX512-BF16")


def _nf4(n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 255, (n, k // 2), dtype=torch.uint8, generator=g)
    absmax = (torch.rand(n, k // 64, generator=g) * 0.1).to(torch.bfloat16)
    return packed, absmax


def _dequantise(packed, absmax, code):
    n = packed.shape[0]
    q = torch.stack([(packed >> 4).long(), (packed & 15).long()], -1).reshape(n, -1)
    return code.float()[q] * absmax.float().repeat_interleave(64, 1)


@pytest.mark.parametrize(("n", "k"), [(6144, 2560), (2560, 9728), (4097, 2560), (13, 64)])
def test_gemv_single_row_matches_multi_row_and_fp32(n, k):
    ext = nf4cpu._build()
    code = torch.randn(16).sort().values.to(torch.bfloat16)
    packed, absmax = _nf4(n, k, seed=n + k)
    x = torch.randn(1, k).to(torch.bfloat16)

    one = ext.nf4_gemv(x, packed, absmax, code, n)
    # Two rows take the general path; the single-row fast path must agree
    # with it bit for bit, not merely closely.
    two = ext.nf4_gemv(torch.cat([x, x]), packed, absmax, code, n)
    assert torch.equal(one[0], two[0])

    ref = x.float() @ _dequantise(packed, absmax, code).T
    assert ((one.float() - ref).norm() / ref.norm()).item() < 5e-3


def _block(qk_norm: bool):
    from fish_speech.models.text2semantic.llama import BaseModelArgs, KVCache, TransformerBlock

    torch.manual_seed(0)
    cfg = BaseModelArgs(
        dim=256, n_head=4, n_local_heads=2, head_dim=64, intermediate_size=512,
        attention_qk_norm=qk_norm, max_seq_len=96,
    )
    block = TransformerBlock(cfg).to(torch.bfloat16)
    for parent in (block.attention, block.feed_forward):
        for name, child in list(parent.named_children()):
            if isinstance(child, torch.nn.Linear):
                lin = bnb.nn.Linear4bit(
                    child.in_features, child.out_features, bias=False,
                    quant_type="nf4", compute_dtype=torch.bfloat16,
                )
                lin.weight = bnb.nn.Params4bit(
                    (torch.randn_like(child.weight, dtype=torch.float32) * 0.05).to(torch.bfloat16),
                    requires_grad=False, quant_type="nf4",
                )
                setattr(parent, name, lin.to("cpu"))
    block.attention.kv_cache = KVCache(1, cfg.max_seq_len, cfg.n_local_heads, cfg.head_dim)
    return cfg, block


@pytest.mark.parametrize("qk_norm", [False, True])
def test_decode_block_matches_the_forks_block(qk_norm):
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis

    from app.engine.cpu import densify, install_decode_blocks

    cfg, ref = _block(qk_norm)
    fast = copy.deepcopy(ref)
    holder = torch.nn.ModuleList([fast])
    assert densify(holder, torch.bfloat16, "nf4") == 5  # wqkv, wo, w1, w2, w3 (w1+w3 then fused)
    assert install_decode_blocks(holder) == 1

    # Same history in both caches, then one decode step at `pos`.
    pos = 70
    for block in (ref, fast):
        cache = block.attention.kv_cache
        g = torch.Generator().manual_seed(1)
        cache.k_cache.copy_(torch.randn(cache.k_cache.shape, generator=g).to(torch.bfloat16))
        cache.v_cache.copy_(torch.randn(cache.v_cache.shape, generator=g).to(torch.bfloat16))

    input_pos = torch.tensor([pos])
    causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
    mask = causal[None, None, input_pos, :]
    freqs = precompute_freqs_cis(cfg.max_seq_len, cfg.head_dim)[input_pos]
    x = torch.randn(1, 1, cfg.dim).to(torch.bfloat16)

    with torch.inference_mode():
        want = ref(x, freqs, mask, input_pos)
        got = fast(x, freqs, mask, input_pos)

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(got, want) < 2e-2
    for name in ("k_cache", "v_cache"):
        a = getattr(fast.attention.kv_cache, name)[:, :, pos]
        b = getattr(ref.attention.kv_cache, name)[:, :, pos]
        assert rel(a, b) < 2e-2


@pytest.mark.skipif(not nf4cpu.amx_available(), reason="needs AMX")
@pytest.mark.parametrize(("n", "k", "m"), [(6144, 2560, 9), (2560, 4096, 16), (4097, 2560, 3), (48, 128, 33)])
def test_amx_matches_the_dense_weights(n, k, m):
    """The AMX path multiplies code[q] * absmax rounded to bf16 - the dense copy's values."""
    ext = nf4cpu._build()
    code = torch.randn(16).sort().values.to(torch.bfloat16)
    packed, absmax = _nf4(n, k, seed=n * k + m)
    x = torch.randn(m, k).to(torch.bfloat16)
    got = ext.nf4_amx(x, packed, absmax, code, n)
    ref = x.float() @ _dequantise(packed, absmax, code).to(torch.bfloat16).float().T
    assert ((got.float() - ref).norm() / ref.norm()).item() < 5e-3


@pytest.mark.parametrize("positions", [3, 9, 10])
def test_prefill_block_matches_the_forks_block(positions):
    """A short prefill through `prefill_block` against the fork's forward."""
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis

    from app.engine.cpu import densify, install_decode_blocks

    calls = []
    real = nf4cpu._build()

    class Spy:
        def __getattr__(self, name):
            return getattr(real, name)

        def prefill_block(self, *args):
            calls.append(1)
            return real.prefill_block(*args)

    cfg, ref = _block(qk_norm=True)
    fast = copy.deepcopy(ref)
    holder = torch.nn.ModuleList([fast])
    densify(holder, torch.bfloat16, "nf4")
    nf4cpu._ext = Spy()  # install_decode_blocks binds whatever _build() returns
    try:
        install_decode_blocks(holder)
    finally:
        nf4cpu._ext = real

    start = 50
    for block in (ref, fast):
        cache = block.attention.kv_cache
        g = torch.Generator().manual_seed(2)
        cache.k_cache.copy_(torch.randn(cache.k_cache.shape, generator=g).to(torch.bfloat16))
        cache.v_cache.copy_(torch.randn(cache.v_cache.shape, generator=g).to(torch.bfloat16))

    input_pos = torch.arange(start, start + positions)
    causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
    mask = causal[None, None, input_pos, :]
    freqs = precompute_freqs_cis(cfg.max_seq_len, cfg.head_dim)[input_pos]
    x = torch.randn(1, positions, cfg.dim).to(torch.bfloat16)

    with torch.inference_mode():
        want = ref(x, freqs, mask, input_pos)
        got = fast(x, freqs, mask, input_pos)
    assert calls, "the fused prefill path did not serve the call"

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    assert rel(got, want) < 2e-2
    for name in ("k_cache", "v_cache"):
        a = getattr(fast.attention.kv_cache, name)[:, :, start : start + positions]
        b = getattr(ref.attention.kv_cache, name)[:, :, start : start + positions]
        assert rel(a, b) < 2e-2


def test_write_kv_leaves_exactly_the_cache_a_full_step_does():
    """Depth position 0 runs its last layer as `write_kv` only; the K/V it
    leaves must be the full step's, bit for bit, since positions 1..9 read them."""
    from fish_speech.models.text2semantic.llama import precompute_freqs_cis

    from app.engine.cpu import densify, install_decode_blocks

    cfg, block = _block(qk_norm=False)
    holder = torch.nn.ModuleList([block])
    densify(holder, torch.bfloat16, "nf4")
    install_decode_blocks(holder)
    cache = block.attention.kv_cache

    input_pos = torch.tensor([0])
    causal = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len, dtype=torch.bool))
    mask = causal[None, None, input_pos, :]
    freqs = precompute_freqs_cis(cfg.max_seq_len, cfg.head_dim)[input_pos]
    x = torch.randn(1, 1, cfg.dim).to(torch.bfloat16)

    with torch.inference_mode():
        block(x, freqs, mask, input_pos)
        full = (cache.k_cache.clone(), cache.v_cache.clone())
        cache.k_cache.zero_(), cache.v_cache.zero_()
        block.write_kv(x, freqs, mask, input_pos)
    assert torch.equal(cache.k_cache, full[0])
    assert torch.equal(cache.v_cache, full[1])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_topk_sampler_draws_the_tokens_the_sorting_one_does(dtype):
    import fish_speech.models.text2semantic.inference as inf

    from app.engine.cpu import install_topk_sampler

    original = inf.sample
    try:
        install_topk_sampler()
        fast = inf.sample
        g = torch.Generator().manual_seed(3)
        differ = 0
        for trial in range(300):
            logits = (torch.randn(1, 1, 4096, generator=g) * (1 + trial % 7)).to(dtype)
            temperature = torch.tensor(0.3 + (trial % 5) * 0.2)
            top_p = torch.tensor(0.5 + (trial % 4) * 0.15)
            top_k = (30, 1, 100, 4096)[trial % 4]  # 4096 = no cut: the original serves it
            torch.manual_seed(trial)
            want, want_p = original(logits, temperature, top_p, top_k)
            torch.manual_seed(trial)
            got, got_p = fast(logits, temperature, top_p, top_k)
            # bf16 logits tie often. Which of two equal logits sits at the
            # top-k boundary is arbitrary in both (sort is not stable), and a
            # rounding difference can move the top-p boundary by one element.
            # Same distribution; a different token only on such a tie.
            assert int((got_p > 0).sum()) - int((want_p > 0).sum()) in (-1, 0, 1)
            differ += not torch.equal(got, want)
        assert differ <= 3
    finally:
        inf.sample = original
