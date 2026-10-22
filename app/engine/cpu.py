"""CPU inference profile: thread placement and dense weights.

Without a GPU the costs are different. Every frame is still ~76 single-vector
matmuls, but bitsandbytes' CPU NF4 path dequantises the weight on each call -
measured 3.5 ms for a 2560x9728 layer against 0.9 ms for the same layer dense
in bf16 (oneDNN uses AMX on Sapphire Rapids and later). So on CPU the NF4
checkpoint is expanded once at load and served dense: the values are the
quantised ones, only the storage changes.

Dense bf16 is still bandwidth-bound (~7 GB of weights per frame for the slow
stack alone, plus the 4-layer fast stack re-read ten times), so decode-time
matmuls go through `app/engine/nf4cpu.py` instead: the NF4 bytes as stored,
decoded in registers, ~2.7x faster per layer than dense at the same values.
Prefill (many positions) stays on the dense copy, where AMX makes it cheap.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def default_threads() -> int:
    """Every logical CPU, hyperthreads included. Decode is memory-bound, but
    the NF4 kernels decode weights in registers and a sibling thread hides
    load latency: measured on c4-standard-24, a frame takes ~77 ms on 24
    threads against ~85 ms on the 12 physical cores (and 80 vs 85 ms on 16
    against 8 on a c4-standard-16)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def configure_threads(requested: int) -> int:
    import torch

    threads = requested if requested > 0 else default_threads()
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)  # one worker thread drives the model
    except RuntimeError:
        pass  # already set once in this process
    torch.set_flush_denormal(True)
    logger.info("CPU threads: %d intra-op (%d logical CPUs)", threads, os.cpu_count() or 0)
    return threads


def densify(model, dtype, weights: str = "nf4") -> int:
    """Replace every bitsandbytes Linear4bit in `model` with a CPU-fast layer:
    the exact NF4 weights through `app/engine/nf4cpu.py` (`weights="nf4"`), or
    a dense nn.Linear of the same values (`"bf16"`, the fallback on a CPU
    without AVX512-BF16). Returns how many."""
    import bitsandbytes as bnb
    import bitsandbytes.functional as F
    import torch
    import torch.nn as nn

    if weights not in ("nf4", "bf16"):
        raise RuntimeError(f"TTS_CPU_WEIGHTS must be nf4 or bf16, got {weights!r}")
    if weights == "nf4":
        from app.engine import nf4cpu

        if not nf4cpu.available():
            weights = "bf16"
    count = 0

    def walk(module) -> None:
        nonlocal count
        for name, child in list(module.named_children()):
            if not isinstance(child, bnb.nn.Linear4bit):
                walk(child)
                continue
            if weights == "nf4":
                setattr(module, name, nf4cpu.make_linear(child, dtype))
                count += 1
                continue
            w = child.weight
            dense = F.dequantize_4bit(w.data, w.quant_state).to(dtype).contiguous()
            lin = nn.Linear(child.in_features, child.out_features,
                            bias=child.bias is not None, device="meta")
            lin.weight = nn.Parameter(dense, requires_grad=False)
            if child.bias is not None:
                lin.bias = nn.Parameter(child.bias.data.to(dtype), requires_grad=False)
            setattr(module, name, lin)
            count += 1

    with torch.no_grad():
        walk(model)
        if weights == "nf4":
            fuse_feed_forward(model)
    logger.info("CPU: rebuilt %d NF4 linear layers as %s", count, weights)
    return count


def fuse_feed_forward(model) -> int:
    """w1 and w3 read the same input: serve them as one NF4 matmul of twice
    the rows, then split. Saves a call and a pass over x per block."""
    import types

    import torch.nn.functional as F
    from fish_speech.models.text2semantic.llama import FeedForward

    from app.engine import nf4cpu

    count = 0
    for module in model.modules():
        if not isinstance(module, FeedForward):
            continue
        w1, w3 = module.w1, module.w3
        if type(w1).__name__ != "NF4Linear" or type(w3).__name__ != "NF4Linear":
            continue
        module.w13 = nf4cpu.fuse_rows(w1, w3)
        split = w1.out_features
        del module.w1, module.w3

        def forward(self, x, _split=split):
            a, b = self.w13(x).split([_split, self.w13.out_features - _split], dim=-1)
            return self.w2(F.silu(a) * b)

        module.forward = types.MethodType(forward, module)
        count += 1
    logger.info("CPU: fused w1/w3 in %d feed-forward blocks", count)
    return count


#: Longest run of positions served by `nf4cpu.prefill_block`: what a hint
#: leaves of a sentence's prompt. Measured over the 36-layer stack, fused
#: against the dense bf16 copy on oneDNN: 89 vs 98 ms at 4 positions, 108 vs
#: 113 at 8, 127 vs 121 at 12 - past ~10 the dense copy wins, and longer
#: prompts go through the original forward.
PREFILL_MAX_POSITIONS = 10


def install_decode_blocks(model) -> int:
    """Serve each transformer block's short calls in one C++ call each:
    `nf4cpu.decode_block` for a decode step, `nf4cpu.prefill_block` for a
    prefill of up to `PREFILL_MAX_POSITIONS` - the four NF4 matmuls plus every
    norm, rotary, KV-cache write, attention and residual, instead of ~60 aten
    ops at a few microseconds of dispatch each. Longer prefills and anything
    unexpected go through the original forward."""
    import types

    import torch
    from fish_speech.models.text2semantic.llama import TransformerBlock

    from app.engine import nf4cpu

    ext = nf4cpu._build()
    empty = torch.empty(0, dtype=torch.bfloat16)
    count = 0
    for block in model.modules():
        if not isinstance(block, TransformerBlock):
            continue
        attn, ff = block.attention, block.feed_forward
        layers = (attn.wqkv, attn.wo, getattr(ff, "w13", None), ff.w2)
        if any(type(l).__name__ != "NF4Linear" for l in layers):
            continue
        qk = bool(getattr(attn, "attention_qk_norm", False))
        q_w = attn.q_norm.weight.detach().to(torch.bfloat16).contiguous() if qk else empty
        k_w = attn.k_norm.weight.detach().to(torch.bfloat16).contiguous() if qk else empty
        qk_eps = (attn.q_norm.eps if qk and attn.q_norm.eps is not None
                  else torch.finfo(torch.bfloat16).eps)
        original = block.forward
        # Everything but x, the position, freqs and mask is fixed per block;
        # gathered once (and again only if the KV cache is reallocated) because
        # ~20 nn.Module attribute lookups a call were ~60 us of a ~800 us block.
        cached = [None, None, None]  # kv_cache, args before `start`, args after mask

        def static(self, _q=q_w, _k=k_w, _qk_eps=qk_eps, _cached=cached):
            a = self.attention
            if _cached[0] is not a.kv_cache:
                f = self.feed_forward
                _cached[1] = (
                    self.attention_norm.weight, self.ffn_norm.weight, _q, _k,
                    self.attention_norm.eps, _qk_eps,
                    a.wqkv.packed, a.wqkv.absmax, a.wo.packed, a.wo.absmax,
                    f.w13.packed, f.w13.absmax, f.w2.packed, f.w2.absmax,
                    a.wqkv.code, a.kv_cache.k_cache, a.kv_cache.v_cache,
                )
                _cached[2] = (a.n_head, a.n_local_heads, a.head_dim, f.w2.in_features)
                _cached[0] = a.kv_cache
            return _cached[1], _cached[2]

        def forward(self, x, freqs_cis, mask, input_pos=None, _orig=original, _static=static):
            n = x.shape[1]
            if (x.shape[0] != 1 or not 1 <= n <= PREFILL_MAX_POSITIONS or mask is None
                    or input_pos is None or input_pos.numel() != n
                    or self.attention.kv_cache is None
                    or x.dtype != torch.bfloat16 or mask.dtype != torch.bool):
                return _orig(x, freqs_cis, mask, input_pos)
            head, tail = _static(self)
            start = int(input_pos[0])
            if n == 1:
                return ext.decode_block(
                    x.contiguous(), *head, start, freqs_cis.reshape(-1, 2), mask.reshape(-1), *tail
                )
            if int(input_pos[-1]) - start != n - 1:  # not one consecutive run
                return _orig(x, freqs_cis, mask, input_pos)
            return ext.prefill_block(
                x.contiguous(), *head, start, freqs_cis.reshape(n, -1, 2), mask.reshape(n, -1), *tail
            )

        def kv_only(self, x, freqs_cis, mask, input_pos, _static=static):
            head, tail = _static(self)
            # An empty `wo` and MLP: decode_block writes the cache and returns.
            ext.decode_block(
                x.contiguous(), *head[:8], empty, empty, empty, empty, empty, empty, *head[14:],
                int(input_pos[0]), freqs_cis.reshape(-1, 2), mask.reshape(-1), *tail[:3], 0,
            )

        block.forward = types.MethodType(forward, block)
        block.write_kv = types.MethodType(kv_only, block)
        count += 1
    logger.info("CPU: fused decode path installed on %d transformer blocks", count)
    return count


def install_fast_first_position(model) -> bool:
    """Skip the work the decode loop throws away at depth position 0.

    Position 0 of the fast (depth) transformer takes the slow hidden state and
    its logits are discarded - codebook 0 comes from the slow head. Only the
    K/V it leaves in each layer's cache matter to positions 1..9, and the last
    layer's K/V depend on nothing past that layer's QKV projection. So the
    last layer's attention output, its MLP, the final norm and the 4096-row
    output head (~95M weights of the ~1B the fast stack reads per frame) are
    skipped there. Positions 1..9 see the identical cache either way.
    """
    import types

    import torch

    layers = getattr(model, "fast_layers", None)
    if not layers or getattr(model, "fast_student", None) is not None:
        return False
    if any(not hasattr(layer, "write_kv") for layer in layers):
        return False
    original = model.forward_generate_fast

    def forward_generate_fast(self, x, input_pos=None, _orig=original):
        if (input_pos is None or input_pos.numel() != 1 or int(input_pos[0]) != 0
                or x.shape[0] != 1 or x.dtype != torch.bfloat16):
            return _orig(x, input_pos)
        x = x.view(1, 1, -1)
        mask = self.causal_mask[None, None, input_pos, : self.config.num_codebooks]
        freqs_cis = self.fast_freqs_cis[input_pos]
        for layer in self.fast_layers[:-1]:
            x = layer(x, freqs_cis, mask, input_pos=input_pos)
        self.fast_layers[-1].write_kv(x, freqs_cis, mask, input_pos)
        return None  # nobody reads position 0's logits

    model.forward_generate_fast = types.MethodType(forward_generate_fast, model)
    logger.info("CPU: depth position 0 writes its last layer's K/V only")
    return True


def install_topk_sampler() -> bool:
    """`sample` with the top-k cut taken by `topk` instead of a full sort.

    fish-speech sorts all 4096 logits eleven times a frame (codebook 0 twice
    for repetition-aware sampling, then nine residual codebooks) to apply a
    top-k of 30 and a top-p inside it: ~0.2 ms a sort, ~2 ms a frame. Here the
    cumulative probabilities are taken from the same full softmax, gathered
    in the same descending order, and the kept set is scattered back into a
    full-width distribution before the same Gumbel-style draw - so a given
    seed draws the token the original draws. The exceptions are ties: equal
    bf16 logits on the top-k boundary are picked arbitrarily by both (sort is
    not stable), and the top-p boundary can move one element on a rounding
    difference. The distribution is the same.
    """
    import torch
    import fish_speech.models.text2semantic.inference as inf

    original = inf.sample
    if getattr(original, "_topk", False):
        return False

    def sample(logits, temperature, top_p, top_k):
        row = logits[0, -1]
        n = row.shape[-1]
        if not 0 < top_k < n:
            return original(logits, temperature, top_p, top_k)
        top, idx = torch.topk(row, top_k)
        cum = torch.cumsum(torch.nn.functional.softmax(row, dim=-1)[idx], dim=-1)
        keep = cum <= top_p
        keep[0] = True
        kept = top[keep] / torch.clip(temperature, min=1e-5)
        probs = torch.zeros_like(row)
        probs[idx[keep]] = torch.nn.functional.softmax(kept, dim=-1)
        return inf.multinomial_sample_one_no_sync(probs), probs

    sample._topk = True
    inf.sample = sample
    logger.info("CPU: top-k sampling without a full sort")
    return True
