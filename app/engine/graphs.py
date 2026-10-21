"""The decode step as CUDA graph replays.

A frame is one pass through the 36-layer slow transformer and ten through the
4-layer depth transformer, plus sampling: about 4000 kernel launches, almost
none of which runs for more than a few microseconds except the 4-bit matmuls.
On Windows (WDDM) every launch costs ~30 us of CPU, so an eager frame measured
152 ms of wall time against 53 ms of GPU time - two thirds of every frame was
the CPU asking. A captured graph is one launch.

What has to be static for that, and how each is handled:

- **The KV window.** `kv_window()` narrows attention to the live length, which
  changes every frame. Graphs are captured per window *bucket* (multiples of
  `step` positions) and a frame replays the smallest bucket that covers it;
  the few masked positions past the live length cost almost nothing.
- **Scalars created per call.** `decode_one_token_ar` builds `torch.tensor(...)`
  constants on the device each frame - a host-to-device copy, illegal while
  capturing. `sample_frame` below is the same computation with those constants
  made once.
- **top_k** reaches the fork's sampler as an int that would be baked in, so it
  is passed as a 0-dim tensor instead; `indices >= top_k` means the same thing
  either way and the graph then serves any request's value.
- **State between frames** (the token fed back, its position, the repetition
  window) lives in static buffers the graph advances itself, so the Python
  loop only replays and reads back the main token to see whether the turn
  ended.

Nothing about the arithmetic changes: the graph records the very kernels the
eager path launches, in the same order.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9


def _sample(logits, temperature, top_p, top_k):
    """One draw: `app/engine/fusedsample.py` when built, else `_sample_torch`."""
    from app.engine import fusedsample

    if fusedsample.available():
        return fusedsample.sample(logits, temperature, top_p, top_k)
    return _sample_torch(logits, temperature, top_p, top_k)


def _sample_torch(logits, temperature, top_p, top_k):
    """The fork's `sample` without its one host round trip.

    `logits_to_probs` keeps the most likely token with
    `sorted_indices_to_remove[0] = False`, which copies a host scalar into
    device memory - not allowed while capturing. Masking index 0 out of the
    removal set is the same thing.
    """
    import torch
    from fish_speech.models.text2semantic.inference import multinomial_sample_one_no_sync

    logits = logits[0, -1]
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
    indices = torch.arange(sorted_logits.shape[-1], device=sorted_logits.device)
    remove = ((cum_probs > top_p) | (indices >= top_k)) & (indices != 0)
    remove = remove.scatter(dim=-1, index=sorted_indices, src=remove)
    logits = torch.where(remove, float("-Inf"), logits)
    logits = logits / torch.clip(temperature, min=1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one_no_sync(probs)


def _fast_prime(model, hidden, pos) -> None:
    """Position 0 of the depth transformer, for its KV entries only.

    Its logits are thrown away (codebook 0 is the semantic token), so the last
    layer's feed-forward and the output head - which feed nothing but those
    logits - are skipped: ~42 MB of weights a frame not read.
    """
    x = hidden.view(1, 1, -1)  # already through fast_project_in
    mask = model.causal_mask[None, None, pos, : model.config.num_codebooks]
    freqs = model.fast_freqs_cis[pos]
    for layer in model.fast_layers[:-1]:
        x = layer(x, freqs, mask, input_pos=pos)
    last = model.fast_layers[-1]
    last.attention(last.attention_norm(x), freqs, mask, pos)


def sample_frame(model, logits, hidden, temperature, top_p, top_k, bias, prev, consts):
    """`decode_one_token_ar` after its slow forward, with no host round trips.

    Mirrors the fork line for line; see fish_speech/models/text2semantic/
    inference.py. `prev` is None for the first frame of a batch, which is how
    the fork disables repetition-aware sampling there too.
    """
    import torch
    from fish_speech.models.text2semantic.inference import _fast_embed, _fast_step

    sliced_head = getattr(model, "semantic_head", None) is not None
    biased = logits if sliced_head else logits + bias

    main = _sample(biased, temperature, top_p, top_k)
    if sliced_head:
        main = model.semantic_index_to_token_id(main)
    high = _sample(biased, consts["high_temp"], consts["high_top_p"], top_k)
    if sliced_head:
        high = model.semantic_index_to_token_id(high)
    if prev is not None:
        in_window = (prev[0] == main).any()
        is_semantic = (main >= model.config.semantic_begin_id) & (
            main <= model.config.semantic_end_id
        )
        main = torch.where(in_window & is_semantic, high, main)

    codebooks = [main]
    student = getattr(model, "fast_student", None)
    if student is None:
        _fast_prime(model, hidden, consts["fast_pos"][0])
    else:
        _fast_step(model, student, hidden, consts["fast_pos"][0], project=True)
    a = torch.clamp(main - model.config.semantic_begin_id, min=0, max=model.config.codebook_size - 1)
    h = _fast_embed(model, student, a)
    codebooks.append(a)
    for i in range(1, model.config.num_codebooks):
        out = _fast_step(model, student, h, consts["fast_pos"][i], project=False)
        a = _sample(out, temperature, top_p, top_k)
        h = _fast_embed(model, student, a)
        codebooks.append(a)
    return torch.stack(codebooks, dim=1).T  # (num_codebooks + 1, 1)


#: Padded lengths prompt spans are prefilled at. Longer spans go through in
#: slices of the largest.
PREFILL_BUCKETS = (1, 2, 4, 8, 16, 32)


class DecodeGraphs:
    """Captured decode steps, one per KV-window bucket, plus the first frame."""

    def __init__(self, model, device, bias, *, step: int = 128) -> None:
        import torch

        self.model = model
        self.device = device
        self.step = step
        cfg = model.config
        dtype = next(model.parameters()).dtype
        self.cd = cfg.num_codebooks + 1
        self.max_len = int(model.max_seq_len)
        self.bias = bias

        # Inputs and state the graphs read and advance.
        self.tok = torch.zeros((1, self.cd, 1), dtype=torch.int, device=device)
        self.pos = torch.zeros((1,), dtype=torch.long, device=device)
        self.prev = torch.zeros((self.cd, RAS_WIN_SIZE), dtype=torch.int, device=device)
        self.temperature = torch.tensor(0.8, dtype=dtype, device=device)
        self.top_p = torch.tensor(0.8, dtype=dtype, device=device)
        self.top_k = torch.tensor(30, dtype=torch.long, device=device)
        # Written by an eager prompt forward, read by the first-frame graph.
        width = model.semantic_head.shape[0] if getattr(model, "semantic_head", None) is not None else cfg.vocab_size
        self.logits = torch.zeros((1, 1, width), dtype=dtype, device=device)
        self.hidden = torch.zeros((1, 1, cfg.fast_dim), dtype=dtype, device=device)
        self.out = torch.zeros((self.cd, 1), dtype=torch.int, device=device)
        self.consts = {
            "high_temp": torch.tensor(RAS_HIGH_TEMP, dtype=dtype, device=device),
            "high_top_p": torch.tensor(RAS_HIGH_TOP_P, dtype=dtype, device=device),
            "fast_pos": [
                torch.tensor([i], dtype=torch.long, device=device)
                for i in range(cfg.num_codebooks)
            ],
        }
        # Prompt prefill, one graph per padded length. Each attends over the
        # whole cache rather than a window bucket: masked positions cost ~2 ms
        # at 1536, and one graph per length is 6 graphs instead of 72.
        self._prefill: dict[int, tuple] = {}
        for n in PREFILL_BUCKETS:
            self._prefill[n] = (
                None,
                torch.zeros((1, self.cd, n), dtype=torch.long, device=device),
                torch.zeros((n,), dtype=torch.long, device=device),
                torch.zeros((1,), dtype=torch.long, device=device),
            )
        self._decode: dict[int, "torch.cuda.CUDAGraph"] = {}
        from app.engine import fusedsample

        #: Requests asking for more candidates than the captured sampler takes
        #: must decode eagerly.
        self.max_top_k = fusedsample.MAX_TOP_K if fusedsample.available() else None
        self._first: "torch.cuda.CUDAGraph | None" = None
        self.pool = torch.cuda.graph_pool_handle()

    # --- the captured bodies ---

    def _advance(self, codes, *, push: bool) -> None:
        self.out.copy_(codes)
        self.tok.copy_(codes.view(1, self.cd, 1))
        if push:
            rolled = self.prev.roll(-1, dims=1)
            rolled[:, -1] = codes[:, 0]
            self.prev.copy_(rolled)

    def _decode_body(self, window: int) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        from app.engine.fastpath import kv_window

        with kv_window(window), sdpa_kernel(SDPBackend.MATH):
            result = self.model.forward_generate(self.tok, self.pos)
            codes = sample_frame(
                self.model, result.logits, result.hidden_states, self.temperature,
                self.top_p, self.top_k, self.bias, self.prev, self.consts,
            )
        self._advance(codes, push=True)
        self.pos.add_(1)

    def _prefill_body(self, n: int) -> None:
        from fish_speech.models.text2semantic.llama import BaseTransformer

        from app.engine.fastpath import kv_window, prompt_embeddings

        _, inp, pos, last = self._prefill[n]
        with kv_window(self.max_len), prompt_embeddings():
            # BaseTransformer's, for `return_all`: padding follows the real
            # positions, so the last real one has to be picked out by index.
            result = BaseTransformer.forward_generate(self.model, inp, pos, None, None, True)
        self.logits.copy_(result.logits.index_select(1, last))
        hidden = self.model.fast_project_in(result.hidden_states.index_select(1, last))
        self.hidden.copy_(hidden)

    def _first_body(self) -> None:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel(SDPBackend.MATH):
            codes = sample_frame(
                self.model, self.logits, self.hidden, self.temperature, self.top_p,
                self.top_k, self.bias, None, self.consts,
            )
        self._advance(codes, push=False)

    def _capture(self, body):
        import torch

        # Warm up on a side stream first, as torch.cuda.graph asks: the first
        # call of some kernels allocates (cuBLAS workspaces, lazy builds).
        saved = (self.tok.clone(), self.pos.clone(), self.prev.clone())
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.inference_mode():
            body()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.pool):
            body()
        self.tok.copy_(saved[0])
        self.pos.copy_(saved[1])
        self.prev.copy_(saved[2])
        return graph

    def capture_all(self) -> None:
        """Capture every bucket now, so no frame ever pays for a capture."""
        import torch

        t0 = time.perf_counter()
        # The warm-up runs really execute, so they write one KV entry. Put it at
        # the last position, which no cached prompt can reach and which any
        # generation writes itself before it reads.
        self.pos.fill_(self.max_len - 1)
        self.tok.zero_()
        self.tok[0, 0, 0] = self.model.config.semantic_begin_id
        for window in range(self.step, self.max_len + self.step, self.step):
            window = min(window, self.max_len)
            if window not in self._decode:
                self._decode[window] = self._capture(lambda w=window: self._decode_body(w))
        self._first = self._capture(self._first_body)
        # Prefill warm-ups write KV entries too, so they get the same treatment.
        for n, (_, inp, pos, last) in self._prefill.items():
            inp.zero_()
            pos.copy_(torch.arange(self.max_len - n, self.max_len, device=self.device))
            last.fill_(n - 1)
            graph = self._capture(lambda n=n: self._prefill_body(n))
            self._prefill[n] = (graph, inp, pos, last)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # the side-stream warm-ups' blocks are cached per stream
        logger.info(
            "captured %d decode and %d prefill graphs in %.1fs",
            len(self._decode) + 1,
            len(self._prefill),
            time.perf_counter() - t0,
        )

    # --- the per-request calls ---

    def begin(self, params, position: int) -> None:
        """Set the sampling knobs and where the first decode step writes."""
        self.temperature.fill_(params.temperature)
        self.top_p.fill_(params.top_p)
        self.top_k.fill_(params.top_k)
        self.pos.fill_(position)
        self.prev.zero_()

    def prefill(self, tokens, start: int) -> bool:
        """Write KV entries for `tokens` (codebook_dim, n) at [start, start + n).

        Leaves the last position's logits and hidden state where `first()`
        reads them. False, having done nothing, if the span cannot be padded
        to a bucket without running off the end of the cache.
        """
        import torch

        n = tokens.shape[-1]
        pieces = []
        while n - sum(pieces) > PREFILL_BUCKETS[-1]:
            pieces.append(PREFILL_BUCKETS[-1])
        pieces.append(n - sum(pieces))
        padded = sum(pieces[:-1]) + next(b for b in PREFILL_BUCKETS if b >= pieces[-1])
        if start + padded > self.max_len:
            return False
        tokens = tokens.to(self.device, non_blocking=True)
        offset = 0
        with torch.inference_mode():
            for size in pieces:
                bucket = next(b for b in PREFILL_BUCKETS if b >= size)
                graph, inp, pos, last = self._prefill[bucket]
                inp[0, :, :size].copy_(tokens[:, offset : offset + size])
                # Padding repeats the last real token; its KV entries land past
                # the prompt, where generation overwrites them before reading.
                inp[0, :, size:].copy_(tokens[:, offset + size - 1 : offset + size])
                torch.arange(start + offset, start + offset + bucket, out=pos)
                last.fill_(size - 1)
                graph.replay()
                offset += size
        return True

    def first(self, logits=None, hidden=None) -> "torch.Tensor":
        """Sample frame 1 from the prompt's last position. Returns `out`."""
        import torch

        if logits is not None:
            self.logits.copy_(logits)
            self.hidden.copy_(hidden)
        with torch.inference_mode():  # the captured RNG state is an inference tensor
            self._first.replay()
        return self.out

    def decode(self, position: int) -> "torch.Tensor":
        """One frame at `position` (which must match `pos`). Returns `out`."""
        import torch

        window = min(self.max_len, -(-(position + 1) // self.step) * self.step)
        with torch.inference_mode():
            self._decode[window].replay()
        return self.out
