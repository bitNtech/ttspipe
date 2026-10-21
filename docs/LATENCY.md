# Latency

How this server gets audio out fast, what it costs on a small GPU, and how to
drive it so a voice agent feels conversational.

The short version, on an RTX 2050 laptop GPU (4 GB, 112 GB/s, Windows 11)
with a 9-second cloned voice: first audio **54–66 ms** for text streamed in
as an LLM writes it, **~70–100 ms** for a new sentence sent whole, and a frame
of audio (46.4 ms) generated and decoded in **~46–49 ms** - about realtime,
where this machine started at 152 ms. Nothing was quantised below the
checkpoint's own 4 bits: the optimised path agrees with the fork's own code to
a KL divergence of 2.7e-5 on the semantic token, against 0.005 for merely
moving the sampling temperature from 0.8 to 0.75.

[Round two](#round-two-rtx-2050-on-windows) below is the current design and
its measurements. Everything after it is round one, on a GTX 1650 under Linux,
kept because the reasoning still holds; where round two replaced a mechanism,
it says so.

---

## Contents

- [Round two: RTX 2050 on Windows](#round-two-rtx-2050-on-windows)
- [Measured results](#measured-results) (round one)
- [Where the 20.8 seconds went](#where-the-208-seconds-went)
- [The five changes](#the-five-changes)
- [Driving it from a voice agent](#driving-it-from-a-voice-agent)
- [The throughput wall](#the-throughput-wall)
- [Playing it without breaking up](#playing-it-without-breaking-up)
- [Tuning knobs](#tuning-knobs)
- [Reproducing the measurements](#reproducing-the-measurements)

---

## Round two: RTX 2050 on Windows

### Measured results

RTX 2050 (4 GB GDDR6, 112 GB/s, sm_86, 16 SMs), Windows 11, S2-Pro NF4, the
active 8.8 s Tamil reference voice (349 prompt tokens), measured end to end
over the WebSocket API with `scripts/stream_probe.py` and in-process with
`scripts/bench_latency.py`.

| | Before this round | After |
| --- | ---: | ---: |
| Model load | 293 s | 35–50 s |
| First audio, prompt already prefilled (repeat, or fully hinted) | 103–181 ms | **54–58 ms** |
| First audio, text streamed in (server-side, from segment cut) | — | **62–66 ms** |
| First audio, new sentence sent whole, GPU warm | 264–299 ms | **70–100 ms** |
| First audio, text streamed in after 8 s idle | — | **62–64 ms** |
| First audio, sentence sent whole after idle | — | 250–400 ms (see [idle clocks](#an-idle-laptop-gpu-is-a-slow-one)) |
| One frame, generated and decoded (46.4 ms of audio) | 157 ms | **46–49 ms** |
| Client buffer for a gapless 14 s, 3-sentence reply | — | 0.5–1.0 s |

The spread in the last two rows is the card heating up: the same frame
measured 46.1 ms in-process from cold and 48–50 ms after a few minutes of
continuous decoding.

### The limit, and why nothing was quantised further

A frame is one pass through 36 transformer layers and ten through the 4-layer
depth transformer: **4.03 GB of NF4 weights**, every frame, at batch size one.
At the ~101 GB/s the kernel below reaches (90% of the bus), that is 40 ms of
the 46.4 ms a frame of audio lasts, before attention, sampling or the codec.
So on this card the model runs at about realtime and no faster, and the only
remaining lever of any size is fewer weight bytes.

That was measured and rejected. Requantising the depth transformer's MLPs
from NF4 to 3 bits (Lloyd-Max levels over the real weights, with and without
fixed endpoints) would save ~4 ms a frame, and teacher-forced on real
utterances it moves the residual-codebook distributions by a KL of **0.11–0.17**
with the top-1 token agreeing only **62–67%** of the time. For scale, the whole
optimised path below differs from the fork's own code by 1.6e-5 (99.4% top-1),
and shifting the sampling temperature from 0.8 to 0.75 is 0.005. Three bits
would be a different voice. NF4 for the codec's `post_module` was also tried
(11 dB SNR against fp16 - audibly broken) and dropped.

### What changed

**NVRTC on Windows** - `app/engine/nvrtc.py`. The hand-written kernels
looked for `libnvrtc.so` and `libcuda.so.1`, so on Windows every one of them
was silently off and bitsandbytes served every matmul. The torch wheel ships
`nvrtc64_*.dll`; the driver is `nvcuda.dll`.

**Model load: 293 s → 35 s** - `GpuWorker._load`. The fork builds the whole
4B-parameter model in fp32, randomly initialises every weight, copies each
fp32 `Linear` into a `Linear4bit`, and only then overwrites all of it from the
prequantised checkpoint. On a 16 GB machine that is 16 GB of pages touched for
nothing. `_skip_random_init` skips the initialisation and the copy; every key
is still loaded from the checkpoint.

**CUDA graphs** - `app/engine/graphs.py`. On Windows a kernel launch costs
~30 µs of CPU and a frame is ~4000 of them: **152 ms of wall time for 53 ms of
GPU work**. The decode step, the prompt prefill (one graph per padded length)
and the first-frame sampler are captured and replayed. What had to change for
that: the fork's sampler writes a host scalar into device memory
(`sorted_indices_to_remove[0] = False`), `decode_one_token_ar` builds device
constants per call, and `top_k` reaches the sampler as an int that would be
baked in - all replaced by static equivalents that compute the same thing.
One decode graph covers every position (see the fused attention below), so
the executables - ~36 MB of device memory each on this driver - cost little.

**A tensor-core 4-bit kernel for decode *and* prefill** -
`app/engine/nf4gemv.py`. Each warp computes a 16x8 tile with `mma.sync`,
dequantising bitsandbytes' own NF4 buffer into registers as it goes. Rather
than repacking the weights into the fragment layout `mma` wants, the logical
k index of each fragment is mapped onto the codes a thread already loaded,
and the inputs are read through the same map - a dot product does not care
about order. One input vector: 93–100 GB/s against 87 for the previous scalar
GEMV. Up to 8 prompt positions cost what one does, 16 about 15% more, so the
remaining prompt of a request goes through in one pass instead of the old
~48 ms per position (bitsandbytes dequantises the whole weight for anything
longer than one, ~490 ms flat). The residual adds and the SwiGLU gate are
fused into its store.

**A decode block in nine kernels instead of ~30** - `app/engine/fusedlayer.py`.
Head norms, rotary and both KV-cache writes in one kernel; attention as
split-K over 32-position chunks (flash-decoding) that reads only up to the
live position, so masked chunks cost nothing and one graph serves the whole
cache. 46.9 → 44.1 ms a step.

**One kernel per draw** - `app/engine/fusedsample.py`. A frame samples eleven
times, each a full sort of a 4096-entry row in the fork. A radix select finds
the top-k in shared memory, one warp sorts them, and the top-p cut and the
draw happen in the same block: 0.9 ms → 0.23 ms a frame. Its distribution
matches the fork's to sampling noise (total variation 0.01–0.02 over 20 000
draws, the same as two runs of the fork against each other).

**A streaming codec** - `StreamingCodec` in `app/engine/codecstream.py`. The
old incremental decoder re-ran the last 512 frames through `post_module` and
a 10-frame convolution lookback on every call: ~45 ms of GPU time whatever
the block size, which forced blocks to grow to 32 or 64 frames. Every layer is
causal, so the state a block needs can be carried instead: each causal
convolution keeps its last `padding` input samples, each transposed one the
input frame that overlaps the next block, and `post_module` a KV cache at the
absolute positions a full pass uses, attending over only its 128-frame
window. A call now costs ~5 ms plus ~1.7 ms a frame, and the output is
**48–49 dB SNR against one pass over the whole utterance** - the fp16 floor (a
full pass over a 400-frame prefix differs from one over 700 frames by the same
47 dB). Two things made most of that ~5 ms: cuDNN was converting every fp16
weight to NHWC on every call (~3 ms), fixed by storing them with channels-last
strides; and the Snake activation was five kernels, now one. Measured along
the way: `post_module` context of 256 frames is only 32 dB against a full
pass, so the recomputing fallback's default went back to 512.

**Small blocks, because realtime changes the arithmetic.** At 0.4x realtime a
client has to buffer anyway and growing blocks cost nothing. At about realtime,
a block reaches the client only once all of it is generated, so a player needs
roughly one block in hand - and a ramp that doubles makes every block arrive a
whole previous block late. Simulated against the measured costs (a 1-frame
first block at ~60 ms, 43 ms a frame, a codec call of 5 ms + 1.7 ms/frame),
the buffer a 28 s reply needs for zero stalls:

| emit schedule | buffer |
| --- | ---: |
| 1, 2, 4, 4, 4 ... | **140 ms** |
| 1, 2, 4, 6, 6 ... | 230 ms |
| 1, 2, 4, 8, 8 ... | 320 ms |

**The GPU never waits for the host.** The decode loop keeps one step queued
ahead of the one it is checking for end-of-turn, and codec audio is collected
a block behind (`drain_async`), so the stream is never drained just to copy
samples out. The step speculated past the end of a turn is wasted work at a
position the next request overwrites.

**VRAM.** On Windows an allocation past physical memory does not fail - the
driver pages it to system RAM, and measured, the codec went from ~40 ms to
**2.5 s** a call with no error anywhere. The card was there: 3.9 GB used. The
200 MiB codebook embedding table now lives in pinned host memory and is
gathered over PCIe inside the graphs (`app/engine/hostembed.py`; a frame reads
ten rows), the allocator is capped at `total - TTS_VRAM_HEADROOM_MB`, the KV
cache is 1536 positions instead of 2048, and one decode graph replaces
seventeen. Steady state is now ~3.2 GB.

### An idle laptop GPU is a slow one

A second or two after the last frame the card drops to ~250 MHz, and the
first request after that pays for the climb: a first frame that takes 24 ms
warm took 120–330 ms, and first audio after 5 s idle was 387 ms. So a session
now calls `GpuWorker.wake()` on `start` and on every `text` frame, which keeps
the GPU busy with a tiny kernel for 0.6 s or until real work arrives (it
checks between kernels, so a real job waits at most one). Measured after 5 s
idle: 387 ms with no wake, 107 ms after 0.3 s of it, 56 ms after 0.6 s.

That only helps when the server hears about a turn before its text is
complete. **Send `start` the moment the user stops speaking**, before the LLM
has produced anything, and stream text as it arrives: then the LLM's own
time to first token is spent waking the GPU and prefilling. A client that
waits and sends a whole sentence after a pause gets 250–400 ms.

### Checking that it still speaks the same

Teacher-forced means: a real utterance is generated, then its
prompt and frames are fed back one position at a time through both paths, and
the per-position distributions compared.

| | KL (T=0.8) | top-1 agreement |
| --- | ---: | ---: |
| semantic token, optimised vs the fork's own path | 2.7e-5 | 100% |
| codebooks 1–9, optimised vs the fork's own path | 1.6e-5 | 99.4% |
| *for scale:* the fork's logits at T=0.8 vs T=0.75 | 5.1e-3 | — |
| *rejected:* depth MLPs requantised to 3 bits | 0.11–0.17 | 62–67% |

The codec is held to 48–49 dB against a single full pass (above).
`tests/test_nf4gemv.py`, `tests/test_fusedops.py` and
`tests/test_fused_decode.py` pin each kernel against what it replaced.

### New knobs

| Variable | Default | What it does |
| --- | --- | --- |
| `TTS_CUDA_GRAPHS` | `1` | Replay decode, prefill and codec steps as captured graphs. |
| `TTS_HOST_CODEBOOK_EMBEDDINGS` | `1` | Keep the 200 MiB codebook table in pinned host memory. |
| `TTS_VRAM_HEADROOM_MB` | `640` | VRAM the allocator leaves alone; 0 disables the cap. |
| `TTS_STREAM_MAX_FRAMES` | `4` | Largest emitted block (was 32). |
| `TTS_MAX_SEQ_LEN` | `1536` | KV positions (was 2048). |
| `TTS_CODEC_LOOKBACK_FRAMES` | `10` | Recomputing codec path only (was 16). |
| `TTS_CODEC_POST_CONTEXT_FRAMES` | `512` | Recomputing codec path only (was 256). |

`TTS_PREFILL_MODE=auto` now always batches when the 4-bit kernel is built.

### Reproducing

```bash
# In-process, GPU to itself (stop the server first):
.venv/Scripts/python scripts/bench_latency.py

# End to end, against a running server:
.venv/Scripts/python scripts/stream_probe.py --text "..." --repeat 3
.venv/Scripts/python scripts/stream_probe.py --text "..." --trickle
```

---

## Measured results

Round one. GTX 1650 (4 GB, sm_75), S2-Pro NF4, a 9.1 s reference voice (239 prompt
tokens), measured end to end over the WebSocket API.

| Case | Before | After |
| --- | ---: | ---: |
| Prompt already prefilled (repeat / fully hinted) | — | **123 ms** |
| Latin text streamed in as an LLM emits it | — | **276 ms** |
| Latin sentence sent whole, cold prompt | — | ~600 ms |
| Tamil streamed in (token-dense script) | — | ~830 ms |
| First turn after setting a voice | 20 817 ms | as above — the voice prefix is prefilled in the background |
| Setting a new voice (background, off the hot path) | — | 4–6 s |

Time-to-first-audio decomposes cleanly, which is why the spread above is so
wide:

```
TTFA  =  (N-1) x 48 ms          prefill of prompt tokens not already cached
       +  97 ms                 the last prompt token, which also samples frame 1
       +  19 ms                 codec decode of that frame
```

`N` is `prefix_cache.last_prefilled_tokens` in `/status`. Everything below is
about driving `N` towards 1.

Per-frame decode cost, the number everything else is built on:

| | Before | After |
| --- | ---: | ---: |
| Slow (semantic) transformer, one frame | 119.9 ms | **42.9 ms** |
| Depth transformer, 10 steps | 44.9 ms | 44.9 ms |
| Sampling | 1.8 ms | 1.8 ms |
| **Total per frame** | **166.9 ms** | **~90 ms** |

One frame is 2048 samples — 46.4 ms of audio at 44.1 kHz.

That 90 ms was still almost twice what realtime needs, and all of it was inside
kernels PyTorch and bitsandbytes had no good version of for this card. Three
more changes — a 4-bit GEMV, grouped attention for a one-position decode, and
RMS norm and rotary as one kernel each — take it to about 58 ms. The reasoning,
the measurements and what is left are in
[the throughput wall](#the-throughput-wall); they are separated from the five
changes below because they buy a different thing. Everything above this line
makes audio *start* sooner. Everything there makes it *keep up*, which on a
reply of any length matters more.

---

## Where the 20.8 seconds went

Three separate problems, none of which were visible without measuring a single
frame rather than a whole utterance.

**Prefill dominated.** In-context voice cloning carries the reference clip's VQ
codes in the system prompt, so a 9 s clip is 239 prompt tokens — and they were
re-prefilled on every request. A 245-token prefill measured 5.7 s; a 400-token
one measured 11.6 s. `app/engine/references.py` already noted that those tokens
are "replayed on *every* subsequent request that reuses the voice". They were.

**Nothing was emitted until a batch finished.** `generate_long` yields once per
text batch, after its last token is sampled. A 3-second reply meant 3 seconds of
generation before a single sample reached the client.

**Attention read far more memory than it needed.** Every layer ran over the full
2048-position KV cache regardless of how far in the utterance actually was, and
materialised a 4× `repeat_interleave` GQA expansion of both K and V — ~75 MB per
layer, ~2.7 GB of pure memory traffic per frame across 36 layers.

---

## The five changes

### 1. Attention fast path — `app/engine/fastpath.py`

Slice the KV cache to the live length and let SDPA broadcast the 8 KV heads
across 32 query heads itself (`enable_gqa=True`) instead of writing the
expansion out. Installed unconditionally: the streaming generator steers the
model through the same module's `kv_window()` and `prompt_embeddings()`, so
this is load-bearing rather than merely faster. **Bit-exact** — measured `max|delta| == 0.0` against the fork's
own path — because neither change alters what is computed, only how much
redundant work surrounds it.

> 119.9 ms → 42.9 ms per frame for the slow transformer (2.8×).

`enable_gqa=True` turned out to be only half of that fix: the math backend,
which is the only SDPA backend that will take a one-position decode on sm_75,
implements it by materialising the expansion anyway. `decode_attention()` in
the same file does the two matmuls itself — see
[the throughput wall](#the-throughput-wall).

### 2. Prompt prefix cache — `app/engine/streamgen.py`

The KV cache is a plain position-indexed buffer and attention is causal, so a
prompt's entries stay valid until something overwrites them — and nothing does,
because every request writes only from its own start position upwards. The
generator keeps the exact token matrix currently resident and prefills only what
a new prompt does not already share with it: the longest common prefix, compared
across *every* codebook row, not just the semantic one.

The system message falls out as the common case (identical every time, so it
always matches). Reproducing the full prefill's logits exactly was verified:
identical argmax, `max|delta|` 0.0078, which is fp16 rounding.

> A typical turn reuses 251 of 259 prompt tokens.

### 3. Frame-level streaming — `app/engine/streamgen.py` + `app/engine/codecstream.py`

Code frames are yielded as they are sampled, on a ramped schedule — the first
block is small so audio starts immediately, later blocks grow to amortise the
codec's ~30 ms fixed cost per call once audio is already playing.

Decoding a block on its own is not as simple as calling `from_indices` on a
slice: that measures **1.6 dB SNR**. Decoding is two halves with different
needs, and `rvq.decode_codes` documents both. `quantizer.decode_codes`
(`post_module`) is window-limited attention reaching back ~1000 frames, so
slicing its *input* is what destroys the signal — but it is causal (a prefix
matches the full sequence to 1.7e-7 relative) and cheap, so it is re-run over
the frames so far. Everything below it is causal convolution with a 10-frame
receptive field, so it runs on just the new block plus a lookback that is then
discarded.

> Done that way: **109 dB SNR** against a single full pass — the numerical
> floor, not an approximation. A 256-frame cap on the `post_module` re-run keeps
> its cost flat over a long utterance.

### 4. Voice warm-up — `app/voice.py`

`POST /v1/voice` now schedules the reference encode and the 239-token prefill in
the background, and the same happens at startup for a voice that survived a
restart. Changing voice costs the change, not the next thing you say.

> Removed a one-off 17 s hit from whichever turn happened to be first.

### 5. Speculative prefill — `app/session.py`

A hint is bounded as well as optional, and getting that bound right took two
wrong answers first.

Left completely uncapped, a cold cache sends every hint down the batched
prefill path - ~490 ms of flat cost, once per word. A batched prefill is a
single forward call, so it cannot be abandoned once it starts; replacing one
mid-flight just queues another behind it, and the real segment waits behind all
of them. Measured: **9.2 s** to first audio on a turn that normally takes
0.28 s, triggered by nothing more exotic than an English sentence following a
Tamil one.

Forcing hints onto the interruptible per-token path fixes that and breaks
something else. One Tamil word is ~11 tokens and a hint gets ~125 ms between an
LLM's words, so ~2 tokens of progress against 11 tokens of new text means the
hint falls behind for the whole sentence: **39-41** tokens left at cut time,
against 17-24 when batching is allowed.

So hints keep the batched path for large spans, stop after `hint_max_tokens`
positions, and - the part that actually prevents the pile-up - never run two at
once. `Session._maybe_hint` holds the slot, and `_run_hint` calls back into it
the moment one finishes so the next starts against the freshest text rather
than waiting for another `text` frame that may never come.


While the segmenter holds text back waiting for a sentence to finish, the GPU is
idle and the prompt for that sentence is already most of the way known. The
session spends that gap prefilling it.

Re-tokenising a longer string can legitimately disagree with what a shorter one
produced (`"a" + "bc"` need not tokenise like `"abc"`), so nothing is assumed:
the real request still compares tokens position by position and keeps only what
actually matches. A wrong guess costs a re-prefill, never a wrong voice.

> ~600 ms → ~276 ms when Latin text arrives the way an LLM emits it, and
> ~830 ms for Tamil. Stable: 263-359 ms across five consecutive turns, including
> the first turn after a language switch.

Stability was the hard part, and three plausible fixes were wrong before the
right one. An LLM emits words faster than this card prefills them, so hinting on
every `text` frame queues work that is obsolete before it runs - and each queued
hint holds both a worker slot and an event-loop executor thread parked on its
result. The real segment then waits behind all of them: measured on a cold
prefix, **7.5-10.8 s** to first audio.

Capping how much one hint may prefill does not fix it. Forcing hints onto the
interruptible per-token path does not fix it either, and starves token-dense
languages - one Tamil word is ~11 tokens, so a hint that cannot finish a word
makes no progress, and Tamil went to 40-42 tokens outstanding against 17-24.
Rate-limiting bounds the pile-up by starving the hint, which costs those same
languages most: it held English at a stable 429 ms but pushed Tamil to 1130 ms.

The cause was never the batch, or the rate. It was that hints could *accumulate*
while a batched prefill - a single forward call, unabandonable once started -
sat in front of the segment. `GpuWorker.submit_hint` now holds one slot rather
than a queue: a newer request overwrites the one waiting, which is exactly right
because it is the same prompt plus more text, and the replaced caller is
answered immediately so it stops holding a thread. At most one uncancellable
call can ever be in front of a segment, which lets hints batch freely again.

Two details decide how well it works. The guessed prompt is the **complete**
shape a real request would use - closing `im_end`, assistant turn and all - not
just the text so far; an earlier version stopped after the text and always left
the trailing `<|im_end|><|im_start|>assistant\n<|voice|>` to pay for, about 5
tokens and most of what remained. And a hint already in flight is **replaced**,
not waited for: a hint takes longer than the gap between two LLM tokens, so
waiting let three or four words arrive unhinted while a stale guess held the
slot. Replacing is free, because a cancelled prefill keeps every position it
already wrote and the replacement resumes from there.

What it cannot do is outrun the text. Prefill manages about 21 tokens/second on
this card. Latin text at ~8 words/s is roughly 10 tokens/s, so hints keep up and
`N` lands at ~6. Tamil is ~2.76 bytes and 4-6 model tokens per word, so the same
word rate is ~40 tokens/s - twice what prefill can absorb - and `N` stays at
17-24, which is why the Tamil row above is ~830 ms. That gap is prefill
throughput, i.e. the NF4 kernel again, not the hinting logic.

### A bug this turned up

With `TTS_OFFLOAD_EMBEDDINGS` the embedding table lives in host memory and the
fork reads a resident 4097-row semantic slice instead, deciding which it needs
from `tok.shape[-1] == 1` — one position means a decode step, which can only
have produced a semantic token. That holds for the fork, where a prompt always
goes through in one call. It stops holding the moment a prompt is prefilled a
position at a time, and it fails *silently*: out-of-range ids get clamped to row
0, so the model reads a plausible wrong embedding and stops after a frame or
two. `prompt_embeddings()` states the intent instead of inferring it from a
shape. Covered by `tests/test_streaming_latency.py`.

---

## Driving it from a voice agent

**Set the voice once, at startup.** Not per turn. The 239-token prefix is
prefilled in the background when you `POST /v1/voice`, and every turn afterwards
reuses it.

**Stream text as your LLM produces it.** Send `text` frames as tokens arrive
rather than buffering the sentence yourself. That is what lets the server
prefill the prompt while the LLM is still writing — worth ~300 ms a turn. Sending
the finished sentence in one frame gives the server no gap to work in.

**Keep reference clips near 9–10 s.** Prompt length scales with clip length, and
while the prefix is cached, a longer one costs more on every cache miss and more
VRAM.

**Send `stop` as soon as the sentence is done.** The last text frame and `stop`
in the same breath; the server starts the moment the segment is speakable.

**Barge-in is free.** `cancel` is felt within one decode step (single-digit ms) —
it is checked before every sampled token. Use it the instant you detect speech.

**Expect to buffer, and buffer by the arithmetic rather than by a constant.**
See [Playing it without breaking up](#playing-it-without-breaking-up): how deep
the buffer has to be follows from the delivery rate and the length of the reply,
and `scripts/stream_probe.py` measures it for your hardware in one command.

---

## Keeping the voice intact

Streaming frame-by-frame only helps if what streams out is still speech. These
are the things that were breaking it, all found by auditing the new path rather
than by hearing them.

### Byte budgets, not character counts

`chunk_length` is a budget in **bytes**; `app/segmenter.py` bounds a segment in
**characters**. For Latin text the two nearly agree. Tamil runs about 2.76 bytes
per character, so a 220-character segment is ~600 bytes - three times what the
model is batched for - and the first streaming rewrite handed it over in one
piece, because it had dropped the fork's text batching entirely.

`StreamingGenerator._batches` restores it, reusing the fork's own splitters so
boundaries land exactly where `generate_long` put them. Each batch is also fed
the previous batch's codes, which the fork does deliberately: *"each one is
generated with the previously generated codes in context, so prosody carries
across the boundary."* Without that, consecutive batches of one sentence sound
like separate takes spliced together.

### An unclosed `[` used to silence the whole turn

Holding the cut back at an open bracket is right for `[laugh]`. For a bracket
that never closes - "the array [0", a tag an upstream LLM began and abandoned -
`_find_cut` froze its scan limit at that character, so no later boundary could
be found and the hard cut at `max_segment_chars` compared against the frozen
limit too. Measured before the fix: **950 characters buffered, zero segments
emitted**. Nothing is spoken until `flush`. `_OPEN_BRACKET_HOLD` bounds it at 64
characters, which no real tag approaches.

### A click where a short segment should be

The model can end a turn on its very first token. The decode loop checked for
that, but the *first* token arrives outside the loop, and `decode_one_token_ar`
samples all ten fine codebooks off a clamped, meaningless index when the main
token is `im_end`. Decoding it produced ~46 ms of noise. Now that frame is
dropped.

### Bounds that were never enforced

Every integer knob in `TTSParams` was declared `Annotated[int, conint(...)]`,
which is a **no-op under pydantic v2** - `conint()` builds a type, so as
`Annotated` metadata it is silently ignored. `max_new_tokens=-1` gave
`range(-2)`: an empty audio stream and a clean `finish`, indistinguishable from
a segment the model chose not to speak. The float bounds next to them used
`Field` and did work, which is why this went unnoticed. They are all `Field`
now.

### Truncation is no longer silent

Running out of token budget produced the same frames, the same clean finish, and
the same `finish(reason="stop")` as a sentence that ended properly - it just
stopped mid-word. The decode loop now records whether the model ended the turn
itself, logs a warning when it did not, and counts it in
`/status` → `prefix_cache.truncations`. Anything above zero means someone heard
a sentence cut off.

`max_new_tokens` now defaults to `0`, meaning "as much as the context allows",
so the budget is bounded by the context rather than by an invisible 1024-frame
ceiling. With a ~400-token prompt that is ~1650 frames, about 76 seconds - and a
220-character segment can physically produce at most ~400. The counter is there
to prove it never binds rather than to assume it.

### One bad segment no longer takes the answer with it

A synthesis error used to emit `finish(reason="error")`, discard every remaining
segment of the utterance, and return the session to `idle` - so the next `text`
frame was a protocol error and the socket closed. One bad sentence cost the rest
of the reply and the connection. A per-segment failure is now reported on the
`log` channel and skipped; only an unusable engine (`EngineNotReady`) ends the
turn.

### Barge-in leaves a stream the client can decode

`cancel` kept the same encoder and the same clock. The client, which has just
thrown away the audio it was playing, then received a container whose header it
had already discarded and timestamps that still counted the interrupted speech.
Both sound like silence or static right when a voice agent can least afford it.
A cancel now starts a fresh encoder, re-sends the header and restarts the clock.

### Punctuation stays with its sentence

Cutting at the first terminator split anything written with more than one
closing character, and the leftovers opened the *next* segment as text to
pronounce: `...` became a cut after the first dot and a segment beginning `..`;
`?!` left a bare `!`; `He said "Hello." Then...` cut before the quotation mark.
`_end_of_terminator` absorbs the run and any closing quote or bracket.

### What was checked and found sound

Frame accounting in the decode loop (no frame generated-but-unyielded, none
yielded twice, the final partial block flushed on both termination paths); KV
budget safety; the prefix cache's inability to serve a stale prefix; the NaN
guards on both decode paths; and the audio encoders' tail handling. Measured on
a real Tamil utterance: longest silent run 2 ms, and a max sample-to-sample
delta only 1.64x the 99.99th percentile, i.e. no clicks at block or batch seams.

---

## Why a cold sentence cannot reach 150 ms

Time-to-first-audio is `(N-1) x 48 ms + 97 ms + 19 ms`, so 150 ms needs `N <= 2`.
`N` is the number of prompt tokens not already prefilled when the client says
the sentence is over, and it has a hard floor that is nothing to do with how
clever the hinting is:

| | tokens |
| --- | ---: |
| `<\|im_end\|>` + `<\|im_start\|>assistant\n` + voice modality (fixed) | **6** |
| the sentence's last word — English | 2 |
| the sentence's last word — Tamil | **11** |

Both measured against this checkpoint's tokenizer. Nothing can prefill either
group before the last word exists, and the last word and the end-of-sentence
signal arrive together. So the best any speculation can do is `N = 8` for
English and `N = 17` for Tamil — **384 ms and 884 ms of prefill** at 48 ms a
token, before a single frame is sampled.

Measured `N` is 5-6 for English and 17-24 for Tamil, i.e. the hinting is already
at or slightly better than that floor (better, because the guessed full stop
sometimes lands and carries part of the suffix with it). There is no remaining
slack in the scheduling to recover; the only way down is to make a prefill token
cost less than 48 ms, which is the NF4 kernel again.

The 150 ms target *is* met whenever the prompt is already resident - a repeated
phrase, or any sentence the client finished sending early enough for a hint to
complete: **123 ms**, of which 97 ms is the single forward pass that samples
frame 1 and 19 ms is the codec.

### What the CPU can and cannot do here

Measured, not assumed:

| | GPU | CPU |
| --- | ---: | ---: |
| Codec decode, 1 frame (on the TTFA path) | **19 ms** | 72 ms |
| Codec decode, 32 frames | 130 ms | 1335 ms |

Moving the codec to the CPU would free ~720 MiB of VRAM but add ~52 ms to every
first frame, so it makes the number this document is about *worse*. It is also
not needed: the freed VRAM would buy a larger KV cache, and the context is not
the binding constraint (see the token budget section).

Where the CPU does help is the embedding table, which already lives in host
memory. A sequential prefill used to send each prompt token there on its own -
a device synchronise, a gather and a copy back per step. The ids are all known
before the loop starts, so `_gather_prompt_rows` now does one host gather and
one upload for the whole span. It is a small win against a 48 ms step, and it is
the only part of the hot path the CPU is actually better at.

---

## The throughput wall

Latency was the first problem and it is fixed. Throughput is the second one,
and unlike latency it is not about *when* work starts — it is the rate at which
frames come out at all, which is what decides whether a client has to buffer
before it can play anything. A frame is 46.4 ms of audio. Generate one in less
than that and playback starts on the first block; generate one in more and the
client must hold `(RTF - 1) x duration` seconds of audio before it dares start.
At the RTF of 2.63 this server shipped with, a 7.2-second reply meant **11.7
seconds of silence** before the first sample played. That, not
time-to-first-audio, is what a listener experiences as latency on a long reply.

### Where a frame goes

Every millisecond of a frame is inside the model, and almost all of it is inside
one of 390 4-bit matrix-vector products — 36 layers of the slow transformer at
five apiece, plus 4 layers of the depth transformer run once per codebook, ten
times a frame. Profiling one frame at position 400:

| | GPU ms/frame | calls/frame |
| --- | ---: | ---: |
| 4-bit linear layers | 39.5 | 390 |
| attention (`bmm`) | 7.9 | 152 |
| RMS norm + rotary, as elementwise chains | ~9.3 | ~3200 |
| everything else | ~4.8 | |

Three separate problems, and none of them is arithmetic.

### 1. bitsandbytes reads 4-bit weights at a quarter of the bus

| | measured |
| --- | ---: |
| Device-to-device copy, 256 MiB | 158 GB/s |
| `bnb.functional.gemv_4bit`, this model's shapes | 35–48 GB/s |
| `app/engine/nf4gemv.py`, same shapes | 74–111 GB/s |

Its NF4 GEMV gives one row block to a warp, which is the right shape on a card
with enough memory parallelism to hide the dependent chain. On 14 SMs it is not:
each warp has a single 16-byte weight load in flight and then thirty-two
dependent multiply-adds before the next one issues.

Accumulating **eight rows per warp** is most of the difference (55 → 100 GB/s,
measured by turning it off). Eight weight loads are independent, so eight
requests are in flight per thread instead of one, and the input vector — which
every row multiplies — is fetched once per eight rows rather than once per row.
Reading the block scales *before* the 256 multiply-adds rather than after is
worth another fifth (79 → 96 GB/s): they are two dependent loads deep, and
issuing them early lets that latency finish underneath work that does not need
it yet.

The kernel reads bitsandbytes' own `Params4bit` buffer in place and undoes its
double-quantised scales inline, so it costs no VRAM at all — which matters,
because the first version dequantised the scales into fp16 and the 121 MiB that
took was more than this card had left. Long turns failed with an allocator error
instead of merely running slower.

### 2. The math backend was re-expanding grouped attention

A decode step is one query position, which is not a matmul shape: it is a GEMV
against the whole cache. Flash and the memory-efficient SDPA backends both
refuse that shape on sm_75, so the generator pins the math backend — and the
math backend implements `enable_gqa` by materialising exactly the four-fold
expansion of the 8 KV heads that `enable_gqa` exists to avoid.

That was most of why a frame got more expensive as an utterance went on. Doing
the two matmuls against the grouped layout directly reads K and V once each:

| position in the utterance | before | after |
| --- | ---: | ---: |
| 50 | 59.7 ms | 55.4 ms |
| 400 | 68.9 ms | 63.2 ms |
| 1200 | 97.2 ms | 78.1 ms |

### 3. 3200 kernel launches a frame for two functions

What is left once the matmuls are fast is 235 RMS norms and 152 rotary
embeddings, none of which touches more than a few kilobytes, each written in
PyTorch as a chain of six to nine separate operations — a cast, a multiply, a
mean, an add, a reciprocal square root, a multiply, a cast back, a multiply.
The profiler counted 1393 `aten::mul` and 1075 `aten::copy_` calls in a single
frame. One kernel each:

| | before | after |
| --- | ---: | ---: |
| RMS norm, 2560 wide | 32.0 µs | 11.4 µs |
| rotary, 32 heads x 128 | 56.1 µs | 12.7 µs |

Rotary comes back bit for bit identical. The norm sums its squares in the
kernel's order rather than PyTorch's tree reduction, so about three elements in
a thousand round the other way — `tests/test_fusedops.py` pins that at
thresholds an fp16 accumulation (94% identical) or a dropped intermediate
rounding (71%) would fail.

### What that adds up to

Measured in one process, interleaved so the card's thermal drift cannot favour
either configuration, at a steady state reached after 90 seconds of continuous
decoding:

| | median ms/frame | RTF |
| --- | ---: | ---: |
| before | 144.8 | 3.12 |
| + the 4-bit kernel | 104.5 | 2.25 |
| + grouped decode attention | 97.9 | 2.11 |
| + fused norm and rotary | **92.9** | **2.00** |

**1.56x, and nothing about the output changed** — the kernels are held to
bitsandbytes' and the fork's own results, not to recorded ones.

End to end over the WebSocket API, with the server restarted between
configurations and run `off, on, off, on` so drift cannot favour either:

| | ms/frame | RTF |
| --- | ---: | ---: |
| off (first pair) | 170.8 | 3.63–4.20 |
| on (first pair) | 133.8 | 2.63–2.97 |
| off (second pair) | 144.9 | 3.01–3.83 |
| on (second pair) | **77.2** | **1.63–1.69** |

Pooled: **168.9 → 100.2 ms/frame, 1.69x**.

On a rested card, against the same turn this document measured before:

```
before:  ttfa_ms 675   rtf 2.634   audio_s 7.198   min_preroll_ms 11901.7
after:   ttfa_ms 553   rtf 1.667   audio_s 6.873   min_preroll_ms  5187.2
         ttfa_ms 565   rtf 1.640   audio_s 8.731   min_preroll_ms  6333.4
```

`min_preroll_ms` is the thing a listener actually waits through — the buffer a
client has to hold before it can play a turn to the end without running dry —
and per second of speech it goes from **1653 ms to 725–755 ms**. A 7.2-second
reply starts after about 5.4 s instead of 11.9 s. Per-segment, the server
reports `ms_per_frame` 118.9 → **75.8–78.8** and `rtf` 2.56 → **1.63–1.70**.

Interleaving matters more than it sounds. This card idles at 300 MHz, pulls to
about 1350 MHz of a 1785 MHz boost under sustained decoding, and the same
configuration measured 75.6 ms/frame cool and 138.0 ms/frame warm in the very
run above. An earlier server-level before/after appeared to show a 45%
*regression* from these changes — until the codec, which none of them touches,
turned out to have slowed by the same 45%. Any measurement here that is not
interleaved is measuring the temperature.

### What is still on the table

Realtime needs 46 ms/frame. At 93 ms of throttled steady state — about 60 ms
cool — the remaining candidates, in order of what they are worth:

- **A fused decode-attention kernel**, worth the 7.9 ms/frame that 152 tiny
  cuBLAS `bmm` calls now cost. Eighty of those are the depth transformer
  attending over ten positions, where the launch is the whole cost.
- **Pushing the 4-bit kernel further.** Removing the NF4 code lookup entirely
  measures 126 GB/s against the 96 the real kernel gets, so roughly a third of
  it is that lookup. It is two shared-memory reads per byte, conflict-free
  (sixteen entries in sixteen banks, so every bank has one possible address and
  broadcasts); a shuffle-based table was tried and lands in the same place,
  which suggests both are limited by the same pipe.
- **A distilled depth transformer.** The fork ships the scaffolding
  (`fast_student`, `tools/distill/`) precisely because 4 layers run ten times a
  frame cost as much as the 36-layer stack. No trained student exists here.
- **A newer GPU.** Ampere or later runs this faster than realtime as-is.

---

## Playing it without breaking up

Throughput being a wall makes the client's buffering the whole question, and it
is a question with an exact answer rather than a tuning exercise.

### The arithmetic

Playback consumes one second of audio per second. The server delivers at `1/RTF`.
So a buffer holding `B` seconds covers `t` seconds of playback exactly while
`B + t/RTF >= t`, and the binding case is the end of the turn:

```
B  >=  (1 - 1/RTF) x duration
```

At the measured RTF of 2.63, playing a 7.2 s reply without a single gap needs
**4.5 s of audio in hand before the first sample is played** - and, because that
4.5 s itself arrives at 0.38x realtime, about **11.9 s of waiting**. That is not
a property of any code here; it is what 2.63 means. `scripts/stream_probe.py`
reports it directly as `min_preroll_ms`, measured rather than derived:

```
$ python scripts/stream_probe.py --text "Hello. This should be coming out in the voice I just set. ..."
ttfa_ms 675   rtf 2.634   audio_s 7.198   min_preroll_ms 11901.7
```

`min_preroll_ms` is a boundary, and `tests/test_stream_probe.py` pins it as one:
at or above it a player never runs dry, below it it does.

### What the console was doing

The browser console scheduled each arriving chunk at
`max(currentTime + 0.06, nextAt)`. The 60 ms floor reads as jitter absorption
and is, while the server keeps ahead. The moment it falls behind - which at
0.38x realtime is after the first chunk - `nextAt` is permanently in the past,
every chunk lands at `currentTime + 0.06` instead, and the shortfall is inserted
as silence *between every pair of chunks*. Replayed against a recorded turn:

| | gaps | total silence inserted |
| --- | ---: | ---: |
| Before | **12** | **19.1 s** inside 7.2 s of speech |

Each gap is also a hard cut from a non-zero sample to silence and back, which is
a click either side.

### What it does now

`static/index.html` keeps a sample queue that is decoupled from the network
chunking, and applies two rules.

**The playout cursor only moves forward by audio actually written to it.** Two
consecutive buffers are sample-contiguous by construction, so a shortfall can no
longer become a gap per chunk - at worst it becomes one stall.

**Playback starts no sooner than the buffer can sustain it.** The player measures
the delivery rate from the stream itself and applies the arithmetic above. Given
an estimate of how long the turn will be (`player.expect()`, which the console
fits from characters per second of speech and re-fits after every turn), it
starts as late as it must and no later. Without one it aims at `minRun`, a
stretch of uninterrupted speech worth starting for, which grows after every stall
so a stream that keeps breaking converges on a buffer deep enough to play
through. When the server outruns playback none of it binds and playback starts on
the first chunk.

The stall that is genuinely unavoidable - a `minRun` that has not converged yet,
or a turn much longer than estimated - is at least made clean: a few
milliseconds are held back so the last thing heard is a fade to silence rather
than a cut mid-sample, and the resumption is faded in.

Replayed against the same recorded turns, with the player driven by a fake
`AudioContext` so the schedule can be inspected sample by sample:

| Mode | first sound | stalls | silence inserted |
| --- | ---: | ---: | ---: |
| Before | 0.74 s | 12 | 19.1 s |
| `lowest latency` | 0.94 s | 6 | 12.8 s |
| `smooth`, no duration estimate | 7.74 s | 1 | 6.0 s |
| `smooth` (default) | 14.4 s | **0** | **0** |
| `smooth`, server faster than realtime | 0.38 s | **0** | **0** |

The last row is the point of doing it this way rather than with a fixed pre-roll:
the same code costs nothing on hardware that can keep up, because the rule it
applies is the rate, not a constant.

`wav` now streams live alongside `pcm` (the 44-byte header is skipped wherever it
appears, which includes the fresh one a barge-in re-sends), so the console's two
uncompressed formats both play as they arrive rather than at the end.

### Where the time inside a segment goes

`scripts/bench_codec.py` breaks one incremental decode into its parts. On a
GTX 1650, decoding a block of `n` frames:

| | |
| --- | ---: |
| `quantizer.decode_codes` (`post_module`) | ~30 ms, flat |
| decoder convolution stack | ~4.2 ms per frame it runs over |
| device-to-host copy | ~0.1 ms |

The convolution stack runs over `n + codec_lookback_frames`, and the lookback's
audio is discarded - so a call costs **~97 ms before it decodes anything of its
own**, about two frames' worth of playback, whatever its size. Over a segment
that is ~8-9% of wall time, which `TTS_LOG_LEVEL=DEBUG` now reports per segment:

```
job ok: queue_ms=0 ttfa_ms=699 gen_ms=10102 codec_ms=760 (8%) audio_s=3.95
        rtf=2.56 frames=85 blocks=5 ms_per_frame=118.9 chars=57
```

That invites making the emit ramp coarser - quadrupling instead of doubling
reaches 32-frame blocks in four decode calls instead of six, worth ~190 ms a
segment. It was tried and **reverted**, because it makes the turn worse where it
counts. The client decides when to start from how much audio it holds, and it can
only re-evaluate that when a block lands; a coarser ramp gives it fewer chances
to notice it has enough, so it overshoots the threshold by half a block.
Measured over four turns, that delayed the start of gapless playback by 2.3 s
against the 190 ms of decode time it saved.

So the ramp is fine-grained for the same reason the first block is small: what
it is buying is resolution near the moment the client is deciding, and that is
worth more than the codec calls it costs.

### A knob that was not connected

`TTS_LOG_LEVEL` did nothing under uvicorn. `logging.basicConfig` is defined to
do nothing once the root logger has a handler, and uvicorn installs one before
the app's lifespan runs - so the level worked when `app.main` was the entry
point and was ignored in the only way the server is actually started. It sets
the levels on the loggers directly now, and `DEBUG` raises this application's
loggers while leaving the root at INFO, because what it is wanted for is the
per-segment timing above, not a megabyte of torch's.

---

## Tuning knobs

All environment variables; all have working defaults.

| Variable | Default | What it does |
| --- | --- | --- |
| `TTS_PREFIX_CACHE` | `1` | Reuse prompt KV entries across requests. |
| `TTS_STREAM_FIRST_FRAMES` | `1` | Frames in the first audio block — sets TTFA. |
| `TTS_STREAM_MAX_FRAMES` | `32` | Largest block, once audio is playing. |
| `TTS_PREFILL_MODE` | `auto` | `auto` / `batched` / `sequential`. |
| `TTS_PREFILL_SEQUENTIAL_MAX` | `11` | Span at or below which `auto` goes sequential. |
| `TTS_MIN_HINT_CHARS` | `8` | Shortest partial segment worth speculatively prefilling. |
| `TTS_HINT_MAX_TOKENS` | `32` | Most prompt positions one hint may write. |
| `TTS_HINT_MIN_INTERVAL_MS` | `60` | Floor on how often a hint may be issued. |
| `TTS_HINT_INCLUDE_SUFFIX` | `true` | Whether a hint guesses the whole prompt shape or stops after the text. |
| `TTS_CODEC_LOOKBACK_FRAMES` | `16` | Left context per decoded block (≥10 is exact). |
| `TTS_CODEC_POST_CONTEXT_FRAMES` | `512` | Cap on the `post_module` re-run. |
| `TTS_FAST_GEMV` | `1` | Serve single-vector 4-bit matmuls from `app/engine/nf4gemv.py`. |
| `TTS_FUSED_OPS` | `1` | RMS norm and rotary as one kernel each (`app/engine/fusedops.py`). |
| `TTS_NVRTC_PATH` | — | Override where `libnvrtc.so` is found. |

`TTS_FAST_GEMV` and `TTS_FUSED_OPS` are escape hatches, not tuning: off, every
call goes back to bitsandbytes and to the fork's own elementwise chains, which
are correct and about 1.6× slower per frame between them. They exist because a
hand-written kernel is a thing you want to be able to switch off on hardware it
has not been measured on — and because having both switchable in one process is
how [the throughput wall](#the-throughput-wall) measures them at all. Neither
needs a compiler installed; `app/engine/nvrtc.py` explains why.

`TTS_STREAM_FIRST_FRAMES` is the direct TTFA lever: each frame is ~90 ms of
generation, so the default of `1` is the lowest-latency setting. Raise it to `2`
or `4` if you would rather have fewer, larger audio frames on the wire — it
costs ~90 ms of first audio per extra frame.

`TTS_PREFILL_MODE` exists because the choice is not obvious and is worth
re-measuring on other hardware. bitsandbytes only takes its fused `gemv_4bit`
path at sequence length 1; longer sequences dequantise every weight into a dense
GEMM first, which costs a flat ~490 ms here *whatever* the span — 8 tokens and 19
tokens both measured ~575 ms — while walking positions one at a time costs ~48 ms
each and skips it entirely. Hence a crossover around 11 tokens rather than the
"batched always wins" the name suggests. It is also why Tamil lands on the
wrong side of it: with 17-24 tokens left to prefill, every turn pays that flat
cost.

---

## Reproducing the measurements

`/status` reports `last_ttfa_ms` and a `prefix_cache` block:

```json
{
  "last_ttfa_ms": 118.0,
  "prefix_cache": {
    "hits": 3, "misses": 1, "cached_tokens": 259,
    "last_reused_tokens": 251, "last_prefilled_tokens": 8,
    "last_prefill_ms": 100.2
  }
}
```

`last_reused_tokens` near `cached_tokens` means the prefix cache is doing its
job; a `last_prefilled_tokens` that stays high across turns of the same voice
means it is not, and the prompt is changing when you think it is not.

`TTS_LOG_LEVEL=DEBUG` adds one line per segment, which splits a turn's wall
clock into the parts that can be acted on separately - time spent waiting behind
other work on the single GPU thread, prompt prefill, token generation, and the
codec:

```
job ok: queue_ms=0 ttfa_ms=699 gen_ms=10102 codec_ms=760 (8%) audio_s=3.95
        rtf=2.56 frames=85 blocks=5 ms_per_frame=118.9 chars=57
session <id> segment: idle_before_ms=3 ttfa_ms=603 total_ms=8854 chars=62
```

`idle_before_ms` is the dead air at a sentence-to-sentence join, which no
per-segment number shows and which a listener hears as clearly as a stall.

Two scripts measure what a listener gets, rather than what the server did:

```bash
# Can this turn be played without breaking up, and how deep a buffer does it need?
.venv/bin/python scripts/stream_probe.py --text "..." --trickle --wav out.wav

# What does one incremental codec decode cost, by part and by block size?
# Needs the GPU to itself - stop the server first.
.venv/bin/python scripts/bench_codec.py
```

`stream_probe.py` reports `min_preroll_ms` (the buffer that makes playback
gapless), stall counts for any pre-roll you name, the sample-to-sample
discontinuity at every frame join against the distribution inside the frames,
and a per-frame timeline whose `wait_ms` column shows exactly where a turn
stops delivering.

### Comparing kernels on a card that throttles

Do not measure the hand-written kernels by starting the server twice and
comparing. This card's clocks move by a factor of two between a cold start and
a long session, which is larger than anything being measured; the last
configuration to run can look 45% slower purely for having gone last.

`TTS_FAST_GEMV` and `TTS_FUSED_OPS` are readable at runtime, so both
configurations can be driven from one process against one warm card:

```python
import bitsandbytes as bnb
import app.engine.fastpath as fastpath
import app.engine.fusedops as fusedops

fast, slow = bnb.nn.Linear4bit.forward, bnb.nn.Linear4bit._nf4_original_forward
for _ in range(rounds):                     # alternate, several times
    for use_ours in (False, True):
        bnb.nn.Linear4bit.forward = fast if use_ours else slow
        fastpath.grouped_decode = use_ours
        fusedops.enabled = use_ours
        record(time_n_frames())
```

Warm the card for 90 seconds of continuous decoding first, alternate for
several rounds, and take the median rather than the best — the best is whichever
configuration happened to run during a boost.

Unit coverage for everything above that does not need a GPU:

```bash
.venv/bin/python -m pytest tests/test_streaming_latency.py tests/test_stream_probe.py -v
```

The kernels themselves are held to what they replace, and need CUDA:

```bash
.venv/bin/python -m pytest tests/test_nf4gemv.py tests/test_fusedops.py -v
```
