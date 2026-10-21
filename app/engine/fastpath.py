"""Decode-time patches to the vendored fork's attention.

Two things in `fish_speech.models.text2semantic.llama.Attention.forward` cost
far more than they need to during single-token decoding, and both are
invisible until you measure a frame rather than a whole utterance.

**The KV cache is always read at its full allocated length.** `KVCache.update`
returns the whole `(1, n_local_heads, max_seq_len, head_dim)` buffer, and
`forward_generate` builds its mask as `causal_mask[..., input_pos, :max_seq_len]`,
so a frame at position 60 attends over 2048 positions - 1988 of which are
zeros the causal mask then throws away. The work is real even though the
result is not.

**GQA is materialised with `repeat_interleave`.** With `n_head=32` and
`n_local_heads=8`, every layer expands both K and V four-fold into fresh
tensors before calling SDPA: at `max_seq_len=2048` that is 16.8 MiB written
per tensor per layer, ~2.7 GiB of pure memory traffic per frame across 36
layers. PyTorch 2.5+ takes `enable_gqa=True` and broadcasts inside the kernel
instead, so the expansion never lands in memory.

Fixing both is bit-exact - measured `max|delta| == 0.0` against the original
on this checkpoint - because neither changes what is computed, only how much
redundant work surrounds it. Together they take one decode frame from 167 ms
to 91 ms on a GTX 1650.

This is installed unconditionally rather than behind a switch. It is not only
an optimisation: `app/engine/streamgen.py` steers the model through
`kv_window()` and `prompt_embeddings()`, and both are defined here. Turning the
patch off would leave the fork's own attention receiving a narrowed mask it
cannot broadcast against a full-length cache, and its own embedding lookup
reading the semantic slice for prompt tokens that are not in it.

The live length travels through `mask.shape[-1]` rather than a second
argument: `forward_generate` already derives the mask from the position, so
having attention trust the mask's width keeps one source of truth and leaves
the patched signature identical to the original. `kv_window()` is how callers
set that width - see `app/engine/streamgen.py`, which also needs it to prefill
a chunk of prompt against an already-populated prefix.

`prompt_embeddings()` fixes a second assumption the same caller breaks. With
`FISH_OFFLOAD_EMBEDDINGS` the 155776x2560 table lives in host memory and
`embed_main_tokens` reads the resident 4097-row semantic slice instead,
deciding which it needs from `tok.shape[-1] == 1` - one position means a
decode step, which can only have produced a semantic token or im_end. That
holds for the fork, where the prompt always goes through in a single call. It
stops holding the moment a prompt is prefilled a position at a time, or its
last position is split off to be sampled on its own: those are prompt tokens,
arbitrary vocabulary, and the slice does not contain them. Silently, too - the
clamp turns every out-of-range id into row 0 rather than failing, so the model
gets plausible garbage and stops after a frame or two. The context manager
says "these are prompt tokens" explicitly instead of inferring it from a shape.
"""

from __future__ import annotations

import contextlib
import logging
import threading

logger = logging.getLogger(__name__)

_installed = False
_state = threading.local()

#: Whether a one-position decode step takes the hand-written grouped-query path
#: below instead of SDPA. Off restores SDPA with `enable_gqa=True`, which is
#: correct and, on this card, materialises the 4x expansion again. It exists so
#: the two can be measured against each other on the same warm GPU, which is
#: the only way to compare them on a card that throttles.
grouped_decode = True


def _live_len() -> int | None:
    return getattr(_state, "kv_len", None)


@contextlib.contextmanager
def prompt_embeddings():
    """Embed from the full host-side table, whatever the sequence length.

    Wrap any forward pass over *prompt* positions in this. Generated tokens
    must not be wrapped: they are semantic ids, the resident slice covers them,
    and the host round-trip would only cost latency.
    """
    previous = getattr(_state, "prompt_embed", False)
    _state.prompt_embed = True
    try:
        yield
    finally:
        _state.prompt_embed = previous


@contextlib.contextmanager
def prompt_embedding_batch(rows):
    """Serve the next `n` prompt embeddings from one pre-gathered GPU buffer.

    `prompt_embeddings()` sends every prompt token to the host table one at a
    time, which is right when there is one of them and wasteful when a prefill
    is walking N positions: each step pays a device synchronise for the
    `.to("cpu")`, a gather, and a copy back, for a row of 2560 fp16 - about 5 ms
    of the ~48 ms a sequential prefill step costs. The ids are all known before
    the loop starts, so the CPU can gather them in a single pass and ship them
    across once.

    Rows are consumed in order, one per `embed_main_tokens` call, which is
    exactly one per `forward_generate`. Anything that would break that ordering
    must not run inside this block.
    """
    previous = getattr(_state, "embed_batch", None)
    _state.embed_batch = {"rows": rows, "next": 0}
    try:
        yield
    finally:
        _state.embed_batch = previous


@contextlib.contextmanager
def kv_window(length: int | None):
    """Attend over `[0, length)` instead of the whole allocated cache.

    `None` restores the fork's own behaviour (the full `max_seq_len`), which
    is what prefill from position 0 wants and what keeps this a no-op for any
    caller that has not opted in.
    """
    previous = getattr(_state, "kv_len", None)
    _state.kv_len = length
    try:
        yield
    finally:
        _state.kv_len = previous


def decode_attention(q, k, v, mask, n_head, n_local_heads, head_dim):
    """Grouped-query attention for a decode step or a short prefill.

    A decode step is one query position, which is not a matmul shape at all -
    it is a GEMV against the whole cache, and SDPA has no kernel for it here.
    Flash and the memory-efficient backend both refuse this shape on sm_75, so
    `app/engine/streamgen.py` pins the math backend, and the math backend
    implements `enable_gqa` by materialising exactly the expansion the
    `enable_gqa` argument was added to avoid: four copies of each of the 8 KV
    heads, written out and read back, in every one of the 36 layers of every
    frame.

    Measured, that is most of why a frame gets more expensive as an utterance
    goes on - 59.7 ms at position 50 against 97.2 ms at position 1200, for a
    model whose weights cost the same either way. Reading K and V once each,
    against the grouped layout they are already stored in, takes the same two
    points to 55.4 ms and 78.1 ms.

    A short prefill has the same problem at S query positions: the math
    backend is the only one that takes an explicit mask, and it expands K and
    V four-fold in every layer.

    `q` is (B, n_head, S, head_dim); `k` and `v` are
    (B, n_local_heads, L, head_dim); `mask` is a boolean (B, 1, S, L) where
    True means visible. Returns (B, n_head, S, head_dim).
    """
    import torch

    bsz, _, seqlen, _ = q.shape
    groups = n_head // n_local_heads
    # (B, n_head, S, D) -> (B, n_local_heads, groups * S, D). Query head i reads
    # KV head i // groups, which is the grouping repeat_interleave and
    # enable_gqa both use, so this reshape is the expansion rather than an
    # alternative to it.
    qg = q.reshape(bsz, n_local_heads, groups * seqlen, head_dim)
    scores = torch.matmul(qg, k.transpose(-2, -1)).float() * (head_dim**-0.5)
    scores = scores.view(bsz, n_local_heads, groups, seqlen, -1)
    # The mask is applied rather than assumed: nothing else here decides what
    # a caller's window means.
    scores = torch.where(mask.unsqueeze(2), scores, float("-inf"))
    probs = torch.softmax(scores, dim=-1).to(v.dtype).view(bsz, n_local_heads, groups * seqlen, -1)
    return torch.matmul(probs, v).view(bsz, n_head, seqlen, head_dim)


#: Longest query span the grouped path takes. Past this the fp32 score tensor
#: (32 heads x S x L) is large enough that SDPA's own kernels are the better
#: trade, and a span that long is a voice being warmed, off the critical path.
GROUPED_MAX_QUERY = 64


def install() -> None:
    """Patch the fork in place. Idempotent; safe to call from any thread."""
    global _installed
    if _installed:
        return

    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from fish_speech.models.text2semantic.llama import (
        Attention,
        BaseTransformer,
        apply_rotary_emb,
    )

    from app.engine import fusedops

    if not hasattr(F.scaled_dot_product_attention, "__call__"):  # pragma: no cover
        raise RuntimeError("torch SDPA unavailable")

    original_attention_forward = Attention.forward
    original_forward_generate = BaseTransformer.forward_generate

    def forward(self, x, freqs_cis, mask, input_pos=None):
        bsz, seqlen, _ = x.shape
        q_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim

        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        if self.attention_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # `app/engine/fusedops.py` does this in one kernel instead of nine.
        # It is called here rather than by patching `apply_rotary_emb`, because
        # that name is bound into this closure and a patch would not reach it.
        fused_q = fusedops.rope(q, freqs_cis)
        fused_k = fusedops.rope(k, freqs_cis) if fused_q is not None else None
        if fused_k is not None:
            q, k = fused_q, fused_k
        else:
            fusedops.note_rope_fallback()
            q = apply_rotary_emb(q, freqs_cis)
            k = apply_rotary_emb(k, freqs_cis)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        # The mask's key axis is the authoritative live length: everything at
        # or past it is either masked out or has never been written.
        if mask is not None and mask.shape[-1] < k.shape[2]:
            kv_len = mask.shape[-1]
            k = k[:, :, :kv_len]
            v = v[:, :, :kv_len]

        if grouped_decode and seqlen <= GROUPED_MAX_QUERY and mask is not None:
            y = decode_attention(
                q, k, v, mask, self.n_head, self.n_local_heads, self.head_dim
            )
            y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)
            return self.wo(y)

        # enable_gqa broadcasts the 8 KV heads across 32 query heads inside the
        # kernel; the repeat_interleave this replaces wrote both expansions out
        # to memory first.
        if mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=True,
                    enable_gqa=True,
                )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0,
                enable_gqa=True,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, q_size)
        return self.wo(y)

    def forward_generate(
        self, inp, input_pos=None, audio_masks=None, audio_parts=None, return_all=False
    ):
        window = _live_len()
        if window is None or input_pos is None:
            return original_forward_generate(
                self, inp, input_pos, audio_masks, audio_parts, return_all
            )
        # forward_generate reads self.max_seq_len to size the mask. Narrowing it
        # for the duration of the call is how the window reaches the mask
        # without touching the fork's own signature.
        saved = self.max_seq_len
        self.max_seq_len = window
        try:
            return original_forward_generate(
                self, inp, input_pos, audio_masks, audio_parts, return_all
            )
        finally:
            self.max_seq_len = saved

    original_embed_main_tokens = BaseTransformer.embed_main_tokens

    def embed_main_tokens(self, tok):
        batch = getattr(_state, "embed_batch", None)
        if batch is not None:
            index = batch["next"]
            batch["next"] = index + 1
            return batch["rows"][index : index + 1].unsqueeze(0)
        if getattr(_state, "prompt_embed", False) and getattr(
            self, "embeddings_offloaded", False
        ):
            # Prompt positions can name any token in the vocabulary, so they
            # have to come from the host table however few of them there are.
            # Pinned, it is read in place (app/engine/hostembed.py), which is
            # also what lets a prefill be a captured graph.
            host = getattr(self, "host_embeddings", None)
            if host is not None:
                return host(tok)
            out = self.embeddings(tok.to("cpu"))
            return out.to(self.semantic_head.device, non_blocking=True)
        return original_embed_main_tokens(self, tok)

    Attention.forward = forward
    BaseTransformer.forward_generate = forward_generate
    BaseTransformer.embed_main_tokens = embed_main_tokens

    Attention._fastpath_original_forward = original_attention_forward
    BaseTransformer._fastpath_original_forward_generate = original_forward_generate

    _installed = True
    logger.info(
        "fastpath installed: KV window + native GQA in attention, explicit prompt embedding"
    )
