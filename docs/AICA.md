# Driving this server from the AICA voice agent

This page covers how a voice agent should use the server so the latency it was
built for actually reaches the caller, and how to run it in production. The
wire protocol itself is in [USAGE.md](USAGE.md). Where the milliseconds come
from is in [LATENCY.md](LATENCY.md).

## The shape of a call

Open **one WebSocket per call**, not one per reply. Each agent reply is a
*turn* on that socket:

```
connect  wss://tts.internal:8080/v1/tts/live      Authorization: Bearer <key>
  start  {"request": {"format": "pcm", "sample_rate": 16000}, "context_id": "turn-1"}
  text   "Sure, "             ← forward LLM tokens as they arrive
  text   "your order "
  text   "shipped on Friday."
  stop                         ← LLM finished; server drains and sends finish(stop)
  ... audio frames tagged context_id="turn-1" ...
  finish {"reason": "stop", "context_id": "turn-1"}

  start  {..., "context_id": "turn-2"}            ← next reply, same socket
  text   ...
  cancel                       ← caller barged in: audio stops within one decode step
  finish {"reason": "cancel", "context_id": "turn-2"}
  stop                         ← close the turn (or keep sending text under it)
```

What makes this fast:

1. **Set the voice once, at deploy time or call setup.** Use `POST /v1/voice`
   with the clip and its *exact* transcript. The server prefills that voice's
   prompt in the background. Every turn that names no voice of its own reuses
   it, so no turn pays the ~350-token prompt again.
2. **Stream LLM tokens straight in as `text` frames.** Don't wait for a
   sentence. While a sentence is still arriving, the server speculatively
   prefills it, and that hidden work is most of the gap between the "streamed"
   and "sent whole" first-audio figures in the README.
3. **Ask for the rate your pipeline plays.** Set `sample_rate: 16000` or
   `24000` (or set `TTS_DEFAULT_SAMPLE_RATE` server-side), and
   `format: "mulaw", sample_rate: 8000` for telephony. Conversion happens on
   the server with a streaming resampler. It adds a few milliseconds of filter
   delay once, not per chunk. `pcm` and `mulaw` are headerless and are the
   lowest-latency formats.
4. **Barge-in is `cancel`, not a disconnect.** The session stays open with the
   same voice and seed. Audio frames already in flight may still arrive before
   the `finish(cancel)` frame. Drop them by `context_id` when it no longer
   matches the turn you're playing.
5. **Use `flush` at clause boundaries only when you know more text is far
   off**, for example before a tool call. Otherwise the segmenter waits for
   punctuation, which keeps prosody natural.

`context_id` is optional and up to 128 characters. When a `start` carries one,
every `started`, `audio` and `finish` frame of that turn echoes it. Frames of
untagged turns are unchanged.

A reply's **first** segment is cut at its first finished sentence, however
short (`TTS_FIRST_SEGMENT_MIN_CHARS`). So "Sure." or "Okay." is spoken at once
instead of waiting for the next sentence. Later segments still wait for
`TTS_MIN_SEGMENT_CHARS`, which keeps prosody coherent.

## When the text is known up front

For greetings, hold messages and notifications, use HTTP instead of a socket.
The audio still streams as it decodes:

```bash
curl -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
     -d '{"text": "Thanks for calling. How can I help?", "format": "pcm", "sample_rate": 16000}' \
     https://tts.internal:8080/v1/tts -o greeting.pcm
```

`POST /v1/audio/speech` is the OpenAI-compatible form of the same route. An
agent framework's stock OpenAI TTS plugin works against it with only a base-URL
and key change:

| OpenAI field | Here |
| --- | --- |
| `input` | the text |
| `voice` | an OpenAI voice name means the active voice, `"model"` means the uncloned voice, any other value is a `references/<id>` voice |
| `response_format` | mp3, wav, flac or pcm (pcm is 24 kHz as OpenAI defines it). opus and aac get a 400. |
| `model`, `instructions`, `speed` | ignored. A `speed` other than 1 is flagged in `X-Warning`. |

`GET /v1/voices` lists the voices a request can name. `GET /v1/info` reports
formats, rates, limits and routes.

## Production settings

| Variable | Recommended | Why |
| --- | --- | --- |
| `TTS_API_KEYS` | one key per client, comma-separated | Without it, anyone who can reach the port can synthesize and replace the active voice. The server logs a warning at startup when it is unset on a non-loopback bind. |
| `TTS_DEFAULT_SAMPLE_RATE` | what the agent plays (e.g. `16000`) | Clients that don't ask still get the right rate. |
| `TTS_SESSION_IDLE_TIMEOUT_S` | `300` | Frees a `TTS_MAX_SESSIONS` slot held by a client that vanished without closing. A session that is still speaking is never idle. Send `ping` during long silences if a call can be quiet longer than this. |
| `TTS_QUEUE_TIMEOUT_S` | `30`–`120` | A segment that gets no audio for this long, queueing included, fails with an error instead of hanging the turn. |
| `TTS_MAX_SESSIONS` | ≈ concurrent calls this host can serve | There is one model worker. Sessions share it in arrival order. |
| `TTS_LOG_FORMAT` | `json` | One object per line for the log pipeline. |
| `TTS_CORS_ORIGINS` | only if a browser calls the HTTP API cross-origin | |
| `TTS_SSL_CERTFILE` / `TTS_SSL_KEYFILE` | set, unless a proxy terminates TLS | Serves `wss://`. For testing, `scripts/make_dev_cert.sh` makes a self-signed pair. |

`run.sh` launches uvicorn with WebSocket per-message deflate **off** (audio
doesn't compress; zlib only added latency) and the access log off (it would
record `?api_key=`). Run one process per accelerator. `--workers` would load
the model once per worker.

## Health, readiness, metrics

| Endpoint | Auth | Use |
| --- | --- | --- |
| `GET /health` | open | Liveness. Returns 200 while loading. Returns 503 when loading failed or the engine worker thread died, which means restart the process. |
| `GET /ready` | open | Readiness. Returns 200 only when a request would be served now. Returns 503 while loading, while the start-up warm-up and active-voice prefill run (`"reason": "warming"`), and while draining. |
| `GET /metrics` | key | Prometheus text format. |
| `GET /status` | key | Human-readable engine state, effective settings (keys redacted). |
| `POST /admin/drain` | key | Refuse new sessions and let open ones finish. |

Metrics worth alerting on:

- `tts_first_audio_seconds`: from a turn's first segment being ready to its
  first audio frame being queued. This is the server's whole share of the
  agent's response latency, including queueing behind other calls. Alert on
  p95.
- `tts_segment_failures_total`, `tts_turns_total{reason="error"}`: the engine
  is failing segments.
- `tts_sessions_rejected_total{reason="capacity"}`: you need another instance.
- `tts_engine_queue_depth`: sustained above 0 means calls are waiting on each
  other.
- `tts_ready == 0` for longer than a load takes.

## Deploying and restarting without cutting calls off

`deploy/aica-tts.service` is a systemd unit. On stop it runs
`scripts/drain.sh`. That script calls `/admin/drain` and waits for
`active_sessions` to reach 0, up to `TTS_DRAIN_TIMEOUT_S`. Then SIGTERM gives
the rest `TTS_GRACEFUL_SHUTDOWN_S`. Under Kubernetes, the same thing is a
`preStop` hook running `scripts/drain.sh`, a `readinessProbe` on `/ready`, a
`livenessProbe` on `/health`, and a `terminationGracePeriodSeconds` that covers
both timeouts.

### Running on boot on a public domain

`sudo deploy/install.sh` installs `aica-tts` as a systemd unit, enabled at
boot with `Restart=always`, and puts Caddy in front of it on ports 80 and 443.
The server itself listens on `127.0.0.1` only. Point the subdomain's A record
at the host's static IP, then set `TTS_PUBLIC_DOMAIN` in `.env` and rerun the
installer. Caddy gets a Let's Encrypt certificate and serves `https://` and
`wss://`. Keep `TTS_API_KEYS` set, because the endpoint is public. On GCP,
reserve the VM's external IP as static, or it changes on every stop and start.
The firewall must also allow tcp:80 and tcp:443 to the VM.

## Hardware: what realtime needs

Each 46.4 ms frame of audio reads every weight of the 36-layer main
transformer once, and the 4-layer depth transformer ten times. In NF4 that is
~3.9 GB of memory traffic per frame, so **sustained streaming is a
memory-bandwidth problem**.

| Host | Per frame | Realtime factor | Verdict for a live agent |
| --- | --- | --- | --- |
| GTX 1650 (4 GB) | 76–79 ms | ~1.65× | Buffer ~0.75 s per second of speech |
| RTX 2050 laptop | 46–49 ms | ~1.0× | About realtime |
| Ampere or newer (L4, A10, RTX 30/40) | below 46 ms | < 1× | Recommended |
| GCP c4-standard-16, CPU only (measured 2026-10-22) | ~78 ms + ~12 ms codec | ~2.1× | First audio 180–240 ms after a sentence ends (LLM-paced text), but about half of each reply must be buffered. Not suitable for live calls |
| GCP c4-standard-24, CPU only, 24 threads (measured 2026-10-22) | ~60 ms + ~10 ms codec | ~1.55× | First audio 210–240 ms for a whole sentence; with LLM-paced text, audio starts before the text ends. About 0.5 s of buffer per second of reply. Still not gapless |

On the CPU box, the measured read bandwidth is ~91 GB/s. That puts the floor
at ~43 ms per frame even with perfect kernels, barely under the 46.4 ms budget.
The CPU kernels run at 60–75 GB/s now (up from ~30). **Serve live calls from a
GPU.** The CPU profile is fine for development, tests and offline rendering.

### Measuring it yourself

The console at `/` has a **streaming test** panel. Point it at any `ws://` or
`wss://` endpoint (with `?api_key=` when keys are on). It streams a reply at
an LLM's pace and reports, per run:

- first audio after the first sentence
- realtime factor
- the pre-roll a player needs to never run dry
- barge-in-to-finish time

From a shell, `scripts/stream_probe.py --trickle` reports the same figures,
including `ttfa_after_text_ms`.
