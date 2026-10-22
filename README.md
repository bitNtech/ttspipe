# Fish Speech S2-Pro — Streaming WebSocket TTS Server

A production-oriented FastAPI WebSocket server around
[`scarxity/fish-speech-s2-pro-nf4`](https://huggingface.co/scarxity/fish-speech-s2-pro-nf4),
a bitsandbytes NF4 4-bit quantization of Fish Audio's 4B-parameter S2-Pro model.
It streams audio the way the official Fish Audio WebSocket service does: you push
text in as it becomes available, audio comes back as soon as the first segment is
decoded, and you can interrupt mid-utterance.

**Voice cloning from a ~10 s reference clip, ~50 languages, and inline
expressiveness tags (`[laugh]`, `[whispers]`, `[sighs]`) all work.**

Two HTTP routes sit alongside the socket. `POST /v1/voice` sets **the active
voice** — upload a clip and its exact transcript once, and every WebSocket turn
speaks in it until you change it. `POST /v1/tts/clone` is the one-shot version
for callers that don't want a streaming connection at all.

🖥 **<http://localhost:8080/>** is a static console for driving all of it by
hand: set a voice, stream text, hear it, barge in.

📖 **[docs/USAGE.md](docs/USAGE.md)** is the usage guide — every route and
field, worked examples, tuning and troubleshooting.
⚡ **[docs/LATENCY.md](docs/LATENCY.md)** is how first audio gets to ~55–100 ms, and
how to drive the server from a voice agent.
🎙 **[docs/AICA.md](docs/AICA.md)** is the production guide for the AICA voice
agent: the per-call protocol pattern, auth, output rates, metrics, draining,
and what hardware streams in realtime.

---

## Quick start

```bash
./run.sh
```

That is the whole setup. On a fresh Linux box with an NVIDIA GPU it provisions a
Python 3.12 virtualenv, installs a CUDA build of PyTorch, verifies CUDA and
bitsandbytes, vendors the inference code, downloads the ~4.9 GB checkpoint, runs
a preflight check, and starts the server on `0.0.0.0:8080`.

It is idempotent: the second run skips straight to launch.

```bash
./run.sh --preflight-only     # verify the environment, don't serve
./run.sh --skip-install       # trust the existing venv
./run.sh --skip-download      # trust the existing checkpoint
```

Then, in another shell:

```bash
.venv/bin/python examples/client.py \
    --text "Hello from Fish Speech. [laugh] This is streaming synthesis." \
    --out hello.wav
```

Or set the voice once and let every later request use it — `reference_text`
must be the clip's exact transcript:

```bash
curl -X POST http://localhost:8080/v1/voice \
    -F reference_audio=@voice.wav \
    -F reference_text="The quick brown fox jumps over the lazy dog."
```

```bash
# now any WebSocket turn is spoken in that voice
.venv/bin/python examples/client.py --text "And this is me, cloned." --out me.wav
```

Or clone a clip in a single HTTP call, without touching the active voice:

```bash
curl -X POST http://localhost:8080/v1/tts/clone \
    -F reference_audio=@voice.wav \
    -F reference_text="The quick brown fox jumps over the lazy dog." \
    -F text="Hello. This should sound like the clip I just uploaded." \
    -o cloned.wav
```

Or just open <http://localhost:8080/> and do all of it in the browser.

---

## Why the extra machinery

The published checkpoint contains `model.pth` and `codec.pth` and **no modeling
code** — `transformers.AutoModel` cannot load it. The architecture (a Dual-AR
transformer over a 10-codebook RVQ codec) is defined by the
[`fish-speech-int4-patch`](https://github.com/scarxity/fish-speech-int4-patch)
fork. `run.sh` therefore clones that fork at a pinned commit into `vendor/` and
puts it on `sys.path`, while installing only the ~25 runtime packages the
inference path actually imports rather than the fork's full training stack
(gradio, wandb, lightning, datasets and friends are not installed).

---

## Architecture

```
  WebSocket frame ──► protocol.py ──► session.py ──► segmenter.py
                       (msgpack)      (state machine)  (text → speakable segments)
                                           │
                                           ▼
                                    engine/fish.py  ── async generator
                                           │
                                    engine/worker.py  ── one daemon thread, owns the GPU
                                           │
                        init_model (NF4) ──┴── generate_long ──► DAC codec decode
                                                                       │
                                           audio.py ◄───────────────────┘
                                        (float32 → wav/pcm/mp3/flac, incremental)
```

| Module | Responsibility |
| --- | --- |
| `app/protocol.py` | Wire format. MessagePack frames, with JSON accepted for convenience. |
| `app/segmenter.py` | Cuts streamed text into speakable segments without splitting words, decimals, or `[tags]`. |
| `app/session.py` | Per-connection state machine: buffering, seed pinning, sequencing, barge-in. |
| `app/ws/router.py` | The `/v1/tts/live` endpoint. Independent receive/send tasks. |
| `app/http/clone.py` | The `/v1/tts/clone` endpoint. Multipart or JSON in, streamed audio out. |
| `app/http/voice.py` | The `/v1/voice` endpoints. The one voice the socket falls back to. |
| `app/http/ui.py` | Serves `static/index.html`, the browser console. |
| `app/voice.py` | The active voice: validation, and write-through to `references/_active/`. |
| `app/synthesis.py` | One-shot synthesis: the whole text up front, output still streamed. |
| `app/audio.py` | Incremental encoders and streaming sample-rate conversion. Never buffers a whole utterance. |
| `app/security.py` | API-key check for HTTP and WebSocket, before the handshake. |
| `app/metrics.py` | Prometheus `/metrics`, no dependency. |
| `app/engine/base.py` | The `TTSEngine` contract — the only thing the transport layer knows about. |
| `app/engine/worker.py` | Single GPU worker thread with token-level cancellation. |
| `app/engine/references.py` | Reference-clip encoding and caching for voice cloning. |
| `app/engine/fake.py` | A GPU-free engine (`TTS_ENGINE=fake`) so the server and its tests run anywhere. |

### Design choices that matter

**One GPU worker, not a pool.** S2-Pro in NF4 leaves little headroom on a small
card, and concurrent generations would contend for the same KV cache. Requests
from many connected clients are queued and served in order; concurrency lives in
the transport layer, not the GPU.

**Token-level cancellation.** The upstream generation loop has no cancellation
hook, so the `decode_one_token` callable is wrapped in a closure that checks a
`CancelToken` before each sampling step and raises. Barge-in unwinds within one
decode step rather than one audio segment.

**Minimal CPU↔GPU traffic.** Codes stay resident on the GPU from sampling
through codec decode; only the finished waveform crosses to host memory, once
per segment, as a single contiguous copy.

**The codec is loaded decode-only by default,** which drops roughly 0.96 GiB of
analysis-side weights that serving never touches. Encoding a *new* reference
clip needs that half, so it is done by a lazily built CPU copy of the codec and
cached by content hash — a one-off cost per voice, not per request.

---

## The WebSocket protocol

Connect to `ws://<host>:<port>/v1/tts/live`. Frames are MessagePack maps with an
`event` discriminator. JSON text frames are also accepted, which makes the
server reachable from a browser console or `websocat` with no msgpack codec.

### Client → server

| Event | Payload | Meaning |
| --- | --- | --- |
| `start` | `{"request": {...}, "debug": false, "context_id": null}` | Open an utterance. Sets the voice and sampling parameters for everything that follows. A `context_id` (≤128 chars) is echoed on this turn's `started`/`audio`/`finish` frames. |
| `text` | `{"text": "a fragment "}` | Append text. Send it as fast as you produce it. |
| `flush` | — | Speak whatever is buffered right now, without waiting for a sentence boundary. |
| `stop` | — | No more text. Drain the buffer, then finish. |
| `cancel` | — | Barge-in. Abandon in-flight audio immediately; the session stays open and ready for new text. |
| `ping` | — | Liveness. |

The `request` object of a `start` frame:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `text` | str | `""` | Optional initial text; usually you stream instead. |
| `language` | str \| null | `null` | Hint, e.g. `en`, `zh`, `ja`. |
| `format` | `wav`/`pcm`/`mp3`/`flac`/`mulaw` | `wav` | `pcm` is headerless int16 — the lowest-latency choice. `mulaw` is headerless G.711 for telephony. |
| `sample_rate` | int \| null | `null` | Output rate: 8000, 16000, 22050, 24000, 32000, 44100 or 48000. `null` is the native 44.1 kHz (or `TTS_DEFAULT_SAMPLE_RATE`). Converted server-side by a streaming resampler. |
| `references` | list of `{audio, text}` | `[]` | Voice cloning. `audio` is raw bytes, or base64 over JSON. `text` must be the clip's exact transcript. |
| `reference_id` | str \| null | `null` | Use a previously cached voice instead of resending the clip. |
| `seed` | int \| null | `null` | Drawn once per session when omitted, so every segment shares a voice. |
| `normalize` | bool | `true` | Text normalization; improves stability on numbers. |
| `chunk_length` | int 100–300 | `200` | Generation chunk size in bytes. Smaller ⇒ lower first-audio latency, slightly lower prosodic coherence. |
| `max_new_tokens` | int | `1024` | Per segment. |
| `top_p` | float 0.1–1.0 | `0.8` | |
| `top_k` | int | `30` | |
| `repetition_penalty` | float 0.9–2.0 | `1.1` | |
| `temperature` | float 0.1–1.0 | `0.8` | |

### Server → client

| Event | Payload |
| --- | --- |
| `started` | `{"session_id", "sample_rate", "format", "seed"}` |
| `audio` | `{"audio": <bytes>, "time": <ms of audio emitted so far>}` |
| `log` | `{"message"}` — only when `debug` was set |
| `finish` | `{"reason": "stop"\|"cancel"\|"error", "error": str\|null}` |
| `pong` | — |

For `format: "wav"` the first `audio` frame carries the RIFF header, as the
official service does. Concatenating every `audio` payload in order yields a
playable file.

### Minimal client

```python
import asyncio, ormsgpack, websockets

async def main():
    async with websockets.connect("ws://localhost:8080/v1/tts/live") as ws:
        await ws.send(ormsgpack.packb({"event": "start",
                                       "request": {"format": "wav"}}))
        for piece in ["Streaming ", "text ", "in, ", "audio out."]:
            await ws.send(ormsgpack.packb({"event": "text", "text": piece}))
        await ws.send(ormsgpack.packb({"event": "stop"}))

        with open("out.wav", "wb") as f:
            async for raw in ws:
                msg = ormsgpack.unpackb(raw)
                if msg["event"] == "audio":
                    f.write(msg["audio"])
                elif msg["event"] == "finish":
                    break

asyncio.run(main())
```

`examples/client.py` is the fuller version — it measures time-to-first-audio and
realtime factor, streams text word by word, and can demonstrate barge-in.

---

## HTTP endpoints

- `POST /v1/voice` — set the active voice from `reference_audio` +
  `reference_text`. Every `start` frame that names no voice of its own then
  uses it, and the `started` frame's `voice_id` confirms which one you got.
  `GET` it for metadata, `GET /v1/voice/audio` for the clip, `DELETE` to clear.
  It persists to `references/_active/`, so a restart keeps the voice.
- `GET /` — the browser console (`static/index.html`), including a streaming
  test panel that drives any `ws://` or `wss://` endpoint the way an agent
  does and reports first-audio latency, realtime factor, the buffer gapless
  playback needs, and barge-in timing.
- `POST /v1/tts` — speak `text` in the server's voice, audio streamed as it
  decodes. JSON or form body; takes every `start`-frame request field
  (`format`, `sample_rate`, `reference_id`, `seed`, ...). `X-Voice-Id` says
  which voice spoke.
- `POST /v1/audio/speech` — the same, OpenAI-compatible (`input`, `voice`,
  `response_format` of mp3/wav/flac/pcm), so stock OpenAI TTS clients in
  LiveKit, Pipecat or the `openai` SDK work with a base-URL change. OpenAI
  voice names mean the active voice; `"model"` means the uncloned model voice;
  anything else names a voice under `references/`. `pcm` is 24 kHz, as OpenAI
  defines it.
- `GET /v1/voices` — the active voice and every named voice under
  `references/<id>/` (a clip plus same-stem `.lab` transcript), usable as
  `reference_id`.
- `GET /v1/info` — formats, sample rates, defaults, limits and routes, for a
  client to configure itself from.
- `POST /v1/tts/clone` — one-shot voice cloning. `multipart/form-data` (the
  clip as a file part) or `application/json` (base64), with `reference_audio`,
  `reference_text` and `text`. The response body is the audio, streamed as it
  decodes; `X-Seed`, `X-Sample-Rate` and `X-Audio-Format` say what you got.
  `reference_text` must be the clip's **exact** transcript — see
  [docs/USAGE.md](docs/USAGE.md#the-transcript-must-be-exact) for why that
  matters more than it looks.
- `GET /health` — liveness. Returns **200 while the model is still
  loading** (with `"status": "loading"`) so orchestrators do not kill a container
  during a slow first start; 503 only on a genuine load failure or a dead
  engine worker thread.
- `GET /ready` — readiness: 200 only when a request would be served now; 503
  while loading, warming up or draining.
- `GET /metrics` — Prometheus text format: first-audio latency histograms,
  sessions, turns by outcome, segment failures, engine queue depth.
- `POST /admin/drain` — refuse new WebSocket sessions and let open ones finish,
  ahead of a restart. `scripts/drain.sh` wraps it.
- `GET /status` — engine state, device and VRAM figures, queue depth, request and
  cancellation counters, last measured time-to-first-audio, and the effective
  configuration.

When `TTS_API_KEYS` is set, every route except `/health` and `/ready` needs a
key: `Authorization: Bearer <key>`, `X-API-Key`, or `?api_key=` for browsers
(open the console as `/?api_key=<key>`).

**TLS / `wss://`:** set `TTS_SSL_CERTFILE` and `TTS_SSL_KEYFILE` and `run.sh`
serves `https://` and `wss://` directly; `scripts/make_dev_cert.sh` makes a
self-signed pair for testing. Behind a TLS-terminating proxy, leave them unset.

---

## Configuration

Everything is environment variables; `run.sh` and the app read the same names,
and a `.env` file at the repo root is sourced if present. See `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `TTS_HOST` / `TTS_PORT` | `0.0.0.0` / `8080` | Bind address. |
| `TTS_API_KEYS` | *(unset)* | Comma-separated keys. Unset disables auth. |
| `TTS_CORS_ORIGINS` | *(unset)* | Browser origins allowed cross-origin. |
| `TTS_LOG_FORMAT` | `text` | `json` for one object per line. |
| `TTS_DEFAULT_SAMPLE_RATE` | `0` | Output rate for requests that name none; `0` is native. |
| `TTS_SESSION_IDLE_TIMEOUT_S` | `300` | Close a socket silent this long with nothing in flight. `0` disables. |
| `TTS_GRACEFUL_SHUTDOWN_S` | `30` | uvicorn's grace period on SIGTERM (`run.sh`). |
| `TTS_ACCESS_LOG` | `0` | uvicorn access log; off because it records `?api_key=`. |
| `TTS_DEVICE` | `cuda` | |
| `TTS_MODEL_REPO` | `scarxity/fish-speech-s2-pro-nf4` | |
| `TTS_CHECKPOINT_DIR` | `checkpoints/s2-pro-nf4` | |
| `TTS_VENDOR_DIR` | `vendor/fish-speech` | Where the inference fork is cloned. |
| `TTS_BNB4` | `1` | Load the backbone as NF4 4-bit. |
| `TTS_HALF` | `1` | fp16. **Required on Turing (sm_75), which has no bf16.** |
| `TTS_MAX_SEQ_LEN` | `1536` | KV cache is pre-allocated to this. |
| `TTS_CODEC_DECODE_ONLY` | `1` | Drop the codec's analysis half (~0.96 GiB). |
| `TTS_OFFLOAD_EMBEDDINGS` | `1` | Keep the 155776x2560 embedding table in host memory (~0.72 GiB). **This is what makes 4 GB possible.** |
| `TTS_FULL_LM_HEAD` | `0` | Must stay off while embeddings are offloaded. |
| `TTS_CODEC_PRECISION` | `auto` | `auto` picks fp32 below sm_80. See hardware quirks. |
| `TTS_DECODE_CHUNK_FRAMES` | `64` | Bounds codec activation memory (~250 MiB). `0` disables. |
| `TTS_COMPILE` | `0` | `torch.compile`. Big first-call cost; only worth it for long-lived servers. |
| `TTS_WARMUP` | `1` | Synthesize a throwaway utterance at startup. |
| `TTS_PREFIX_CACHE` | `1` | Reuse prompt KV entries across requests instead of re-prefilling the voice. |
| `TTS_STREAM_FIRST_FRAMES` | `1` | Frames in the first audio block. **The direct TTFA lever** — each frame is ~90 ms. |
| `TTS_STREAM_MAX_FRAMES` | `4` | Largest block, once audio is already playing. Small on purpose: generation runs at about realtime. |
| `TTS_PREFILL_MODE` | `auto` | `auto` / `batched` / `sequential`. See [docs/LATENCY.md](docs/LATENCY.md#tuning-knobs). |
| `TTS_PREFILL_SEQUENTIAL_MAX` | `11` | Span at or below which `auto` prefills one position at a time. |
| `TTS_MIN_HINT_CHARS` | `8` | Shortest half-finished segment worth speculatively prefilling. |
| `TTS_CODEC_LOOKBACK_FRAMES` | `10` | Recomputing codec path only: left context per block. ≥10 is numerically exact. |
| `TTS_CODEC_POST_CONTEXT_FRAMES` | `512` | Recomputing codec path only: cap on the `post_module` re-run per block. |
| `TTS_CUDA_GRAPHS` | `true` | Replay decode, prefill and codec steps as captured CUDA graphs. |
| `TTS_HOST_CODEBOOK_EMBEDDINGS` | `true` | Keep the 200 MiB codebook embedding table in pinned host memory. |
| `TTS_VRAM_HEADROOM_MB` | `640` | VRAM the allocator leaves alone (Windows pages an over-full card silently). |
| `TTS_FAST_GEMV` | `1` | Serve single-vector 4-bit matmuls from `app/engine/nf4gemv.py` instead of bitsandbytes. |
| `TTS_FUSED_OPS` | `1` | RMS norm and rotary as one kernel each (`app/engine/fusedops.py`). |
| `TTS_NVRTC_PATH` | — | Where `libnvrtc.so` lives, if not beside `torch`. |
| `TTS_MIN_SEGMENT_CHARS` | `24` | Lower ⇒ faster first audio, choppier prosody. |
| `TTS_FIRST_SEGMENT_MIN_CHARS` | `2` | A reply's first segment is cut at its first finished sentence of this length, so an opener like "Sure." is spoken at once. |
| `TTS_HINT_INCLUDE_SUFFIX` | auto | Speculative prefill also guesses the end-of-turn markers. Auto: on for CUDA, off for CPU. |
| `TTS_HINT_CHUNK_TOKENS` | `0` (auto) | Positions a speculative prefill writes per call: 4 on CPU (preemptible), one batched call on CUDA. |
| `TTS_SSL_CERTFILE` / `TTS_SSL_KEYFILE` | *(unset)* | Serve `https://` / `wss://` directly (`run.sh`). |
| `TTS_MAX_SEGMENT_CHARS` | `220` | Hard cut for text that never punctuates. |
| `TTS_CHUNK_LENGTH` | `200` | **Bytes** per generated batch within a segment. Matters for non-Latin scripts: Tamil is ~2.76 bytes/char. |
| `TTS_DEFAULT_FORMAT` | `wav` | |
| `TTS_MAX_SESSIONS` | `8` | Concurrent WebSocket connections. |
| `TTS_QUEUE_TIMEOUT_S` | `120` | A segment that gets no audio for this long (queueing included) fails instead of hanging. `0` disables. |
| `TTS_CPU_THREADS` | `0` | CPU hosts: intra-op threads; `0` is one per logical CPU (hyperthreads included, which measured faster). |
| `TTS_LOG_LEVEL` | `INFO` | |
| `TTS_ENGINE` | *(unset)* | Set to `fake` for a GPU-free engine. |
| `TTS_INSTALL_DEV` | `1` | Install pytest and friends alongside the runtime deps. |
| `TTS_VENDOR_REF` | pinned SHA | Fork commit `run.sh` checks out. |
| `PYTHON_BIN`, `VENV_DIR`, `HF_HOME` | | Bootstrap overrides for `run.sh`. |

---

## Latency

On an RTX 2050 laptop GPU (4 GB) with a ~9 s cloned voice, first audio is
**54–66 ms** for text streamed in the way a voice agent produces it and
**~70–100 ms** for a new sentence sent whole, and audio is generated at about
realtime (~46–49 ms per 46.4 ms frame). That takes CUDA graphs, a tensor-core
4-bit kernel for decode and prefill, a fused decode block and sampler, and a
codec that streams with carried state - all held to the fork's own output
(KL 2.7e-5), no extra quantisation. **[docs/LATENCY.md](docs/LATENCY.md#round-two-rtx-2050-on-windows)**
has the measurements. Round one, on a GTX 1650:

| Case | Before | After |
| --- | ---: | ---: |
| Prompt already prefilled | — | **123 ms** |
| Latin text streamed in as an LLM emits it | — | **276 ms** |
| Latin sentence sent whole, cold prompt | — | ~600 ms |
| Tamil streamed in (token-dense script) | — | ~830 ms |
| First turn after setting a voice | 20 817 ms | as above |

Five things get it there, and `docs/LATENCY.md` has the measurements behind
each: a bit-exact attention fast path (167 ms → 90 ms per frame), a prompt
prefix cache that stops re-prefilling the reference clip's ~240–350 tokens every
request, frame-level streaming with an incremental codec decode that measures
109 dB SNR against a full pass, background prefill of the voice prompt when it is
set, and speculative prefill of each sentence while it is still arriving.

The same document's **[Keeping the voice intact](docs/LATENCY.md#keeping-the-voice-intact)**
section covers the separate problem of making sure what streams out is still
speech — byte-budget batching for non-Latin scripts, an unclosed `[` that used to
silence a whole turn, parameter bounds that were never enforced, and truncation
that used to be indistinguishable from a finished sentence.

To get the low number in your own client: **set the voice once at startup**, and
**stream `text` frames as your LLM emits them** rather than sending the finished
sentence — that gap is what the server prefills in.

**Throughput is a separate problem, and most of it has now been taken back.**
Generating a frame was 90 ms against the 46.4 ms of audio it carries, so audio
started in ~276 ms and then fell behind and clients had to buffer. Three
hand-written CUDA kernels — a 4-bit GEMV that reads weights at twice the rate
bitsandbytes manages on this card, grouped attention for a one-position decode,
and RMS norm and rotary as one kernel each — take a frame to 76 ms of a warm
card's time, and `stream_probe.py` puts the buffer a client must hold before a
7.2-second reply can play gaplessly at **5.4 s instead of 11.9 s**. See
[the throughput wall](docs/LATENCY.md#the-throughput-wall).

`/status` reports `last_ttfa_ms` and a `prefix_cache` block;
`examples/client.py` prints the former per run.

---

## Testing

```bash
.venv/bin/python -m pytest tests -v            # no GPU required
```

The suite runs against `TTS_ENGINE=fake`, so protocol handling, segmentation,
audio encoding, streaming, cancellation and the HTTP endpoints are all covered on
any machine. Tests that genuinely need CUDA are marked and skipped otherwise.

---

## Hardware

An NVIDIA GPU with compute capability 7.0+ is the serving target. A CPU-only
profile also exists (`TTS_DEVICE=cpu`, `app/engine/cpu.py`, with AVX-512 and
AMX kernels in `app/engine/nf4cpu.py`). On a GCP c4-standard-16 it loads in
~10 s and, with text streamed in at an LLM's pace, gives first audio ~180–240 ms
after a sentence ends, but it generates ~2.1× slower than realtime, so a
listener needs about half of each reply buffered before playback can run
gapless. That suits development and offline rendering, not live calls. See
[docs/AICA.md](docs/AICA.md#hardware-what-realtime-needs).

### Measured on a GTX 1650 (4 GB, sm_75, Turing)

Everything below was measured on this repo, not estimated.

| | |
| --- | --- |
| Model load | 52–82 s |
| VRAM at rest | 3454 MiB of 3716 MiB usable |
| VRAM peak under load | 3600 MiB |
| Realtime factor | **1.63–1.70×** slower than realtime (was 2.4–2.6×) |
| Time to first audio | **276 ms** streamed (Latin), 123 ms fully warm |
| Decode cost | **76–79 ms** per 46.4 ms audio frame (was ~119 ms) |
| Buffer needed for gapless playback | **725–755 ms** per second of speech (was 1653 ms) |
| Barge-in latency | **0–1 ms** |
| Progressive streaming | first audio block after 1 frame |
| Concurrent clients | 3 simultaneous, cleanly serialised, no VRAM growth |

The realtime factor is dominated by how fast 4-bit weights can be read. Both
transformer stacks are memory-bound, and bitsandbytes' NF4 GEMV reaches 35–48
GB/s of this card's measured 158 GB/s; `app/engine/nf4gemv.py` reaches 74–111
GB/s by accumulating eight output rows per warp, which is most of the 1.7×
above. The depth transformer alone still costs as much as the main model, since
its 4 layers run 10 times per frame. Ampere and later run this faster than
realtime. See [the throughput wall](docs/LATENCY.md#the-throughput-wall) for
what is left.

Note that this card is power-limited and its clocks move a lot: the same
configuration measured 75.6 ms/frame cool and 138.0 ms/frame warm in one run.
Comparisons here are interleaved for that reason, and single numbers from a
cold start will look better than a long session does.

### Hardware quirks worth knowing

**The codec must run in fp32 below sm_80.** On Turing, fp16 codec decode
produces 100% NaN samples, which reach the client as digital *silence* rather
than an error — measured across every combination:

| codec weights | autocast | result |
| --- | --- | --- |
| fp16 | fp16 | all NaN |
| fp16 | none | all NaN |
| fp32 | fp16 | all NaN |
| fp32 | none | clean |

`TTS_CODEC_PRECISION=auto` (the default) detects this and picks fp32 below
sm_80, at a cost of ~456 MiB. The engine also raises on non-finite decoder
output, so this can never again degrade silently. Ampere and later keep fp16.

**Embedding offload is not optional on 4 GB.** Without
`TTS_OFFLOAD_EMBEDDINGS=1`, the load fails while moving weights onto the device
— *before* the KV cache is allocated, which is why lowering `TTS_MAX_SEQ_LEN`
does not rescue that particular failure.

**Headroom is thin.** At rest the server sits ~260 MiB under the limit. Another
process using the GPU (a game, a compositor doing something ambitious) will push
it over. Reduce `TTS_MAX_SEQ_LEN` to 1024 if you need more room.

---

## License

The model weights and the vendored inference code are under the
**Fish Audio Research License** (see the
[upstream LICENSE](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE)).
Review it before any commercial or redistributive use. All credit for the model
and architecture belongs to [Fish Audio](https://fish.audio/); the NF4
quantization and 4 GB serving path come from
[`fish-speech-int4-patch`](https://github.com/scarxity/fish-speech-int4-patch).
