# Usage

How to get speech out of this server: one-shot voice cloning over HTTP, live
streaming over WebSocket, and the knobs that matter for each.

The [README](../README.md) covers what the server *is* and how it is built.
This document covers how to use it.

---

## Contents

- [Start the server](#start-the-server)
- [The browser console](#the-browser-console)
- [Which API do I want?](#which-api-do-i-want)
- [Set the active voice — `POST /v1/voice`](#set-the-active-voice--post-v1voice)
- [Clone a voice — `POST /v1/tts/clone`](#clone-a-voice--post-v1ttsclone)
  - [The transcript must be exact](#the-transcript-must-be-exact)
  - [Making a good reference clip](#making-a-good-reference-clip)
  - [Examples](#examples)
  - [Request fields](#request-fields)
  - [Response](#response)
  - [Errors](#errors)
- [Stream speech — `WS /v1/tts/live`](#stream-speech--ws-v1ttslive)
- [Reusing a voice across requests](#reusing-a-voice-across-requests)
- [Audio formats](#audio-formats)
- [Expressive tags](#expressive-tags)
- [Tuning](#tuning)
- [Operations](#operations)
- [Troubleshooting](#troubleshooting)

---

## Start the server

```bash
./run.sh
```

First run provisions the venv, downloads the ~4.9 GB checkpoint and loads the
model; that takes a while. Subsequent runs go straight to launch and the model
load is 52–82 s on a GTX 1650.

Wait for readiness before sending anything:

```bash
until [ "$(curl -s localhost:8080/health | grep -o '"ok"')" ]; do sleep 2; done
```

`/health` answers **200 with `{"status": "loading"}`** while the model is still
coming up — that is deliberate, so orchestrators don't kill the container
mid-load. Only `{"status": "ok"}` means it will actually synthesize.

No GPU? Run the whole API surface against the tone-generator engine:

```bash
TTS_ENGINE=fake .venv/bin/python -m uvicorn app.main:app --port 8080
```

Everything in this document works against it. The audio is a beep, not speech.

---

## The browser console

Open **<http://localhost:8080/>**. It is a single static page, served by the
server itself, and it drives everything in this document:

1. **Set the active voice** — upload a clip, type its exact transcript.
2. **Speak** — stream text over the WebSocket and hear it in that voice, with
   live playback, time-to-first-audio, realtime factor and a barge-in button.

   The **Playback** control is the one to understand. This card generates audio
   about 2.6x slower than it plays, so a client has to buffer, and how deep
   decides whether speech is continuous. *smooth* (the default) works out how
   much it needs from the delivery rate and the length of the reply and starts
   then, which plays a turn through without a single gap; *lowest latency*
   starts on the first chunk and stalls when it runs dry. On hardware that keeps
   up, both start immediately. See
   [docs/LATENCY.md](LATENCY.md#playing-it-without-breaking-up) for the
   arithmetic and the measurements.
3. **Clone once** — try a clip through `POST /v1/tts/clone` before committing
   to it.

It is same-origin on purpose: a page served from anywhere else could not open
a `ws://` connection to your server or POST to it without CORS. No build step,
no CDN — [static/index.html](../static/index.html) is the whole thing,
including its MessagePack decoder, so it works on a host with no outbound
network.

---

## Which API do I want?

| | `POST /v1/tts/clone` | `WS /v1/tts/live` |
| --- | --- | --- |
| You have | the whole text up front | text arriving as it is produced (an LLM, a live transcript) |
| Voice | the clip you upload, every time | the **active voice** by default; or an uploaded clip, a stored `reference_id`, or the model default |
| Transport | one HTTP request, streamed response body | msgpack frames, connection held open |
| Interruption | close the connection | `cancel` frame, 0–1 ms, session stays usable |
| Reach for it when | "say this in this voice, give me a file" | you are building a conversational agent |

Both paths run the same segmenter, the same engine and the same encoders, so
identical text + voice + seed produce identical audio either way. Both stream
output: first bytes leave as soon as the first segment decodes.

---

## Set the active voice — `POST /v1/voice`

The server holds **one active voice**. Set it once, and every WebSocket turn
that doesn't name a voice of its own is spoken in it — until you replace it or
clear it. That's the shape a conversational client wants: pick a voice at the
top of a session, then just stream text.

```bash
curl -X POST http://localhost:8080/v1/voice \
  -F reference_audio=@voice.wav \
  -F reference_text="The quick brown fox jumps over the lazy dog."
```

```json
{
  "voice_id": "9f2c41ab7e0d5c83",
  "text": "The quick brown fox jumps over the lazy dog.",
  "filename": "voice.wav",
  "content_type": "audio/wav",
  "created_at": 1758641234.5,
  "duration_s": 10.2,
  "size_bytes": 900044,
  "reference_id": "_active"
}
```

From then on a bare `start` frame is enough:

```python
await ws.send(ormsgpack.packb({"event": "start", "request": {"format": "pcm"}}))
# -> {"event": "started", ..., "voice_id": "9f2c41ab7e0d5c83"}
```

Same rule as everywhere else: `reference_text` must be the clip's
[exact transcript](#the-transcript-must-be-exact), and a blank one is rejected.
Same two body encodings — multipart with a file part, or JSON with base64 in
`reference_audio`.

Setting a voice is pure I/O and returns immediately; it works even while the
model is still loading. The clip's one-off encode is paid by the first turn
that actually speaks in it.

### The rest of the endpoint

| Request | Does |
| --- | --- |
| `GET /v1/voice` | Metadata for the active voice, or `404` if none is set. |
| `GET /v1/voice/audio` | The stored clip itself, for playback or download. |
| `DELETE /v1/voice` | Clears it. Always `200`; the body says `{"cleared": true\|false}`. |

`GET /status` also carries an `active_voice` object (or `null`).

### Which voice a turn actually gets

Explicit beats ambient. In order:

| The `start` frame has | The turn uses | `started.voice_id` |
| --- | --- | --- |
| `references: [{audio, text}]` | that clip | `"inline"` |
| `reference_id: "narrator"` | `references/narrator/` | `"narrator"` |
| `use_active_voice: false` | the model's own default voice | `null` |
| none of the above, a voice is set | **the active voice** | its `voice_id` |
| none of the above, no voice set | the model's own default voice | `null` |

The `started` frame's `voice_id` is how you confirm which one you got without
guessing — worth checking, because the wrong voice is audible rather than an
error.

`use_active_voice: false` opts one session out without disturbing anyone else;
`DELETE /v1/voice` changes it for everybody.

### Things worth knowing

- **A voice change never lands mid-sentence.** A turn resolves its voice at
  `start`, so a session already speaking finishes in the voice it began with.
  The next `start` picks up the new one.
- **It survives a restart.** The clip, its transcript and a small `meta.json`
  are written to `references/_active/`, so the server comes back up in the same
  voice rather than silently reverting to the default.
- **It doubles as a stored voice.** That directory is the same layout
  `reference_id` reads, so `reference_id: "_active"` names it explicitly.
- **`POST /v1/tts/clone` ignores it entirely.** That route clones the clip
  attached to that request, always — a route whose job is "clone this" should
  never quietly clone something else.
- **One voice, server-wide.** Every client shares it. If you need per-client
  voices, send `references` on each `start` frame instead; that always wins.
- **Clips are capped at 32 MiB**, which is two orders of magnitude above a
  sensible ~10 s reference.

---

## Clone a voice — `POST /v1/tts/clone`

Upload a short clip of a voice, tell the server exactly what is said in it,
and get back new speech in that voice.

```
POST /v1/tts/clone
Content-Type: multipart/form-data   (or application/json)
```

| Field | Required | What it is |
| --- | --- | --- |
| `reference_audio` | **yes** | The voice to clone. A file part under multipart; base64 under JSON. |
| `reference_text` | **yes** | The **exact transcript** of `reference_audio`. |
| `text` | **yes** | What you want said, in that voice. |

Everything else is optional and listed [below](#request-fields).

### The transcript must be exact

`reference_text` is not a label, a description, or a name for the voice. It is
the words spoken in the clip, verbatim.

This is in-context cloning: the model is conditioned on the reference audio and
its transcript **as a matched pair**, and infers timbre, accent, pacing and
delivery from how the two line up. Feed it a transcript that doesn't match the
audio and you break that alignment.

The important part: **a wrong transcript is not an error.** Nothing fails. You
get audio back, and it is simply worse — a less convincing voice,
mispronunciations, sometimes words that were never in your text. That silent
failure is why the server rejects an empty `reference_text` outright instead of
quietly cloning without one.

```
clip says:  "The quick brown fox jumps over the lazy dog."

reference_text = "The quick brown fox jumps over the lazy dog."   ✅ exact
reference_text = "the quick brown fox jumps over the lazy dog"    ✅ fine — case
                                                                     and trailing
                                                                     punctuation are
                                                                     not critical
reference_text = "The quick brown fox jumps over the dog."        ❌ words dropped
reference_text = "A man reading a sentence."                      ❌ a description
reference_text = "narrator_voice_v2"                              ❌ a name
reference_text = ""                                               ❌ rejected, 400
```

If you don't have a transcript, produce one with any ASR model
(`whisper`, `faster-whisper`) and read it before you use it — ASR output is a
draft, and the point of this field is that it is *right*.

### Making a good reference clip

| | |
| --- | --- |
| **Length** | ~10 s. This is the single biggest lever on latency — see below. |
| **Content** | Continuous natural speech. One speaker. No music, no overlap. |
| **Quality** | Clean and dry. Reverb, clipping and background noise get cloned too. |
| **Format** | Anything `torchaudio` reads: wav, flac, mp3, ogg, m4a. Resampled and downmixed to mono for you. |
| **Style** | The clone inherits the delivery. A flat read clones flat; an animated read clones animated. |

Clip length matters more than it looks. A reference is re-encoded into the
prompt on every request that uses it: a 10.2 s clip is ~220 prompt tokens, a
31.7 s clip is ~683. You pay that on *every* request, not once. A longer clip
is not a better clone — it is a slower one.

The server logs a warning above 20 s.

### Examples

**curl — the short version**

```bash
curl -X POST http://localhost:8080/v1/tts/clone \
  -F reference_audio=@voice.wav \
  -F reference_text="The quick brown fox jumps over the lazy dog." \
  -F text="Hello. This should sound like the clip I just uploaded." \
  -o cloned.wav
```

**curl — with parameters, and the response headers**

```bash
curl -X POST http://localhost:8080/v1/tts/clone \
  -F reference_audio=@voice.wav \
  -F reference_text="The quick brown fox jumps over the lazy dog." \
  -F text="Slower, steadier, and reproducible." \
  -F format=wav \
  -F temperature=0.7 \
  -F seed=12345 \
  -D headers.txt \
  -o cloned.wav

grep -i '^x-' headers.txt     # X-Seed, X-Sample-Rate, X-Audio-Format, X-Request-Id
```

**Python**

```python
import httpx

with open("voice.wav", "rb") as f:
    response = httpx.post(
        "http://localhost:8080/v1/tts/clone",
        files={"reference_audio": ("voice.wav", f, "audio/wav")},
        data={
            "reference_text": "The quick brown fox jumps over the lazy dog.",
            "text": "Hello from the cloned voice.",
            "format": "wav",
        },
        timeout=300.0,       # generation is slower than realtime on small GPUs
    )
response.raise_for_status()
print("seed:", response.headers["x-seed"])
open("cloned.wav", "wb").write(response.content)
```

**Python, writing the stream to disk as it arrives**

The response body streams, so you can start playing or writing before
generation finishes:

```python
import httpx

with open("voice.wav", "rb") as f, open("cloned.wav", "wb") as out:
    with httpx.stream(
        "POST",
        "http://localhost:8080/v1/tts/clone",
        files={"reference_audio": ("voice.wav", f, "audio/wav")},
        data={"reference_text": "...exact transcript...", "text": "...", "format": "pcm"},
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        for piece in response.iter_bytes():
            out.write(piece)
```

Abandoning the response (closing the connection, breaking out of the loop)
cancels the GPU job. The server does not keep generating for a client that
left.

**JSON, for clients that already speak the WebSocket protocol**

`reference_audio` is base64, the same shape as `references[].audio` on a
`start` frame:

```python
import base64, httpx

payload = {
    "reference_audio": base64.b64encode(open("voice.wav", "rb").read()).decode(),
    "reference_text": "The quick brown fox jumps over the lazy dog.",
    "text": "Hello from the cloned voice.",
    "format": "mp3",
}
r = httpx.post("http://localhost:8080/v1/tts/clone", json=payload, timeout=300.0)
open("cloned.mp3", "wb").write(r.content)
```

**Browser**

```js
const form = new FormData();
form.append("reference_audio", fileInput.files[0]);
form.append("reference_text", "The quick brown fox jumps over the lazy dog.");
form.append("text", "Hello from the cloned voice.");
form.append("format", "mp3");

const res = await fetch("http://localhost:8080/v1/tts/clone", {
  method: "POST",
  body: form,
});
if (!res.ok) throw new Error((await res.json()).error);
new Audio(URL.createObjectURL(await res.blob())).play();
```

### Request fields

Names and bounds are identical to the WebSocket `start` frame, so anything you
tune here transfers there unchanged.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `reference_audio` | file / base64 | — | **Required.** The voice to clone. |
| `reference_text` | str | — | **Required.** Exact transcript of the clip. |
| `text` | str | — | **Required.** What to say. |
| `language` | str | *unset* | Hint: `en`, `zh`, `ja`, … ~50 supported. |
| `format` | `wav`\|`pcm`\|`mp3`\|`flac` | `wav` | See [Audio formats](#audio-formats). |
| `seed` | int | random | Pin it to make a request reproducible. Echoed in `X-Seed`. |
| `normalize` | bool | `true` | Text normalization. Improves stability on numbers and abbreviations. |
| `chunk_length` | int 100–300 | `TTS_CHUNK_LENGTH` (`200`) | Generation batch size in **bytes**, not characters. A segment longer than this is generated as several batches, each conditioned on the previous one's codes. Matters most for non-Latin scripts — Tamil is ~2.76 bytes/char, so 220 characters is ~600 bytes. |
| `max_new_tokens` | int 0–8192 | `0` | Frames per batch. `0` means "as much as the context allows" (~1650 frames, ~76 s). |
| `temperature` | float 0.1–1.0 | `0.8` | Lower ⇒ steadier and flatter. |
| `top_p` | float 0.1–1.0 | `0.8` | |
| `top_k` | int 0–1000 | `30` | |
| `repetition_penalty` | float 0.9–2.0 | `1.1` | Raise if the voice loops or stutters. |

`references` and `reference_id` are **rejected** on this route. It always
clones the clip you upload — that is the whole contract. For a stored voice,
use the WebSocket API ([below](#reusing-a-voice-across-requests)).

Under multipart, blank optional fields (`top_p=""`) are ignored rather than
treated as invalid, so an HTML form can post every field it has.

### Response

`200` with the audio as the body, streamed.

| Header | Meaning |
| --- | --- |
| `Content-Type` | `audio/wav`, `audio/mpeg`, `audio/flac`, or `application/octet-stream` for `pcm`. |
| `X-Audio-Format` | The format actually produced — see the fallback note below. |
| `X-Sample-Rate` | Hz. 44100 for S2-Pro. **`pcm` carries no header, so this is the only place the rate appears.** |
| `X-Seed` | The seed used. Send it back as `seed` to reproduce this exact output. |
| `X-Request-Id` | Matches the server log lines for this request. |
| `X-Warning` | Present only if your requested format was unavailable and the server fell back to `wav`. |

If `mp3` or `flac` isn't supported by the installed libsndfile, the server
**falls back to wav** rather than failing, and says so in `X-Warning` and
`X-Audio-Format`. Read `X-Audio-Format`, not your own request, when naming the
file you write.

### Errors

Errors are JSON: `{"error": "...", "type": "..."}`.

| Status | When |
| --- | --- |
| `400` | A required field is missing or blank, `reference_id`/`references` was sent, or the body could not be parsed. |
| `415` | `Content-Type` is neither `multipart/form-data` nor `application/json`. |
| `422` | A parameter is out of range or unknown (`temperature=9.0`, `speed=2`). |
| `500` | Generation failed. |
| `503` | The model is still loading, or failed to load. Poll `/health`. |

One asymmetry worth knowing: a failure **before** the first audio byte comes
back as a real HTTP status. A failure **after** that can't — the status line is
already sent — so the stream is truncated and the error goes to the server log
under the `X-Request-Id` you were given. Short, truncated output with no error
is worth a look in the logs.

---

## Stream speech — `WS /v1/tts/live`

For text you don't have yet. Connect to `ws://<host>:8080/v1/tts/live`, push
text as it is produced, get audio back as it is decoded.

Frames are MessagePack maps with an `event` discriminator. **JSON text frames
are also accepted**, so you can drive it from a browser console or `websocat`
with no msgpack codec at all.

### The shape of a turn

```
client ──► {"event": "start",  "request": {"format": "wav", ...}}
server ◄── {"event": "started", "session_id", "sample_rate", "format", "seed", "voice_id"}

client ──► {"event": "text", "text": "Hello, "}
client ──► {"event": "text", "text": "this is streaming. "}
server ◄── {"event": "audio", "audio": <bytes>, "time": 480.0}
client ──► {"event": "text", "text": "Second sentence."}
server ◄── {"event": "audio", "audio": <bytes>, "time": 1310.0}

client ──► {"event": "stop"}
server ◄── {"event": "audio", ...}               # the drained remainder
server ◄── {"event": "finish", "reason": "stop", "error": null}
```

`start` must come first. `text`/`flush`/`stop` before it — or after the turn
has finished — is a protocol error and closes the connection. `ping` (→ `pong`)
is legal any time. `cancel` on an idle session is a harmless no-op, not an
error, so a barge-in that races the end of a turn never kills the connection.

`start` also accepts a `debug` flag alongside `request`. It is currently
accepted and ignored: `log` frames are sent whenever there is something to
report, regardless.

### Client → server

| Event | Payload | Meaning |
| --- | --- | --- |
| `start` | `{"request": {...}, "debug": false}` | Open a turn. Fixes the voice and sampling parameters for everything that follows. |
| `text` | `{"text": "a fragment "}` | Append text. Send it as fast as you produce it. |
| `flush` | — | Speak what is buffered *now*, without waiting for a sentence boundary. |
| `stop` | — | No more text. Drain, then `finish`. |
| `cancel` | — | Barge-in. Abandon in-flight audio immediately; the session stays open. |
| `ping` | — | Liveness. |

The `request` object takes the same fields as the clone route, plus:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `references` | list of `{audio, text}` | `[]` | Voice cloning. `audio` is raw bytes over msgpack, base64 over JSON. `text` is the clip's exact transcript — [same rule as above](#the-transcript-must-be-exact). Overrides the active voice. |
| `reference_id` | str | `null` | A stored voice. See [below](#reusing-a-voice-across-requests). Overrides the active voice. |
| `use_active_voice` | bool | `true` | Set `false` to get the model's own default voice without clearing the server-wide one. |
| `text` | str | `""` | Optional initial text; usually you stream instead. |

Send none of the first three and you get [the active voice](#set-the-active-voice--post-v1voice),
which is usually what you want.

`seed` is drawn once per turn when you omit it, so every segment of one
utterance shares a voice. It comes back in the `started` frame.

### Server → client

| Event | Payload |
| --- | --- |
| `started` | `{"session_id", "sample_rate", "format", "seed", "voice_id"}` — `voice_id` says which voice this turn got; `null` means no cloning |
| `audio` | `{"audio": <bytes>, "time": <ms of audio emitted so far>}` |
| `log` | `{"message"}` — a non-fatal warning, e.g. a format fallback or a synthesis error about to become `finish(error)` |
| `finish` | `{"reason": "stop"\|"cancel"\|"error", "error": str\|null}` |
| `pong` | — |

Concatenating every `audio` payload in order yields a playable file; for `wav`
the first one carries the RIFF header.

### Cloning over the WebSocket

```python
import base64, asyncio, ormsgpack, websockets

async def main():
    clip = open("voice.wav", "rb").read()
    async with websockets.connect("ws://localhost:8080/v1/tts/live") as ws:
        await ws.send(ormsgpack.packb({
            "event": "start",
            "request": {
                "format": "wav",
                "references": [{
                    "audio": clip,                       # raw bytes over msgpack
                    "text": "The quick brown fox jumps over the lazy dog.",
                }],
            },
        }))
        for piece in ["Streaming ", "text in, ", "cloned audio out."]:
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

The clip is encoded once and cached by content hash, so resending the same
bytes on later turns costs nothing beyond the upload.

### Barge-in

`cancel` abandons in-flight audio within a single decode step (0–1 ms
measured). Unlike `stop`, it does **not** end the turn — the session stays
`active` under the same voice and seed, so a conversational agent can
interrupt and keep talking without a new `start`:

```python
await ws.send(ormsgpack.packb({"event": "cancel"}))   # ◄── user started speaking
# ... server sends finish(reason="cancel") ...
await ws.send(ormsgpack.packb({"event": "text", "text": "Sorry — go ahead."}))
```

A few already-in-flight `audio` frames may still arrive after you send
`cancel`. Drop everything you receive until the `finish` frame.

### Reference client

```bash
.venv/bin/python examples/client.py --text "Hello there." --out out.wav

.venv/bin/python examples/client.py \
    --text "Say it in my voice." \
    --ref-audio voice.wav --ref-text "The exact transcript of voice.wav." \
    --format mp3 --out clone.mp3

.venv/bin/python examples/client.py --text "A long story we interrupt." \
    --stream-text --cancel-after 0.4 --out out.wav
```

It prints time-to-first-audio and realtime factor per run, and every frame it
sends is exactly what `app/protocol.py` defines — it doubles as living
documentation.

---

## Reusing a voice across requests

For one voice that every WebSocket turn should use, set
[the active voice](#set-the-active-voice--post-v1voice) — that is what it is
for, and it already survives restarts.

For a *named set* of voices that clients pick between, drop them in the repo:

```
references/
└── narrator/
    ├── clip.wav      # the audio
    └── clip.lab      # its exact transcript, plain text
```

Then name it from the WebSocket API:

```json
{"event": "start", "request": {"reference_id": "narrator", "format": "wav"}}
```

The `.lab` file holds the same exact transcript, under the same rule. Several
clips in one directory are all used as prompts, in sorted filename order.

The first request encodes each clip and writes a `clip.tokens.pt` next to it,
so a restarted process — or a second one — skips the encode entirely.

`reference_id` is deliberately **not** accepted on `/v1/tts/clone`: that route
always clones what you upload, so "which voice did I get?" never has two
answers.

The active voice lives in this same layout, at `references/_active/`, so it is
also reachable explicitly as `reference_id: "_active"`.

---

## Audio formats

| Format | Content-Type | Use it when |
| --- | --- | --- |
| `wav` | `audio/wav` | Default. Universally playable. Streamed with "size unknown" placeholders, so the length is filled in by the player. |
| `pcm` | `application/octet-stream` | **Lowest latency.** Headerless signed 16-bit little-endian mono. Feed it straight to an audio device. |
| `mp3` | `audio/mpeg` | Bandwidth matters. No VBR header, so duration may not show before playback ends. |
| `flac` | `audio/flac` | Lossless archival. `total_samples` is 0 (spec-legal "unknown") in a live stream. |

### The wav header says "size unknown"

Because the audio is streamed, the RIFF and `data` size fields are written as
`0xFFFFFFFF` and never patched — there is no seeking back on a socket. Players
that handle live/piped wav (ffplay, vlc, browsers) read to EOF and are fine,
and `soundfile`/libsndfile reads the file correctly. **Python's stdlib `wave`
module does not**: it believes the placeholder and reports a nonsense frame
count.

```python
import wave
w = wave.open("cloned.wav")
w.getnframes() / w.getframerate()   # 48695.77 — the 0xFFFFFFFF placeholder

import soundfile as sf
data, rate = sf.read("cloned.wav")
len(data) / rate                    # 4.32 — correct
```

If you need a fixed-size wav (to seek, or to report a duration up front),
re-encode once you have the whole file: `ffmpeg -i cloned.wav -c copy fixed.wav`.
Or request `pcm` and write your own header — you know the length by then.

Playing raw `pcm` — the rate comes from `X-Sample-Rate`:

```bash
ffplay -f s16le -ar 44100 -ac 1 -i cloned.pcm
sox -t raw -r 44100 -e signed -b 16 -c 1 cloned.pcm cloned.wav   # or convert it
```

`mp3` and `flac` depend on the installed libsndfile. If it can't encode them
the server falls back to `wav` and tells you in `X-Warning` — check
`X-Audio-Format` before naming your output file.

---

## Expressive tags

Inline `[bracket]` instructions work in `text` on both APIs, and the segmenter
is careful never to cut a segment in the middle of one:

```
"That's incredible. [laugh] I did not see that coming. [sighs] Anyway."
```

S2 does **not** use a fixed tag vocabulary. It takes free-form natural-language
instructions placed at the word or phrase where they should take effect, so
these are all valid:

```
[laugh]   [whispers]   [sighs]   [super happy]
[whisper in small voice]   [professional broadcast tone]   [pitch up]
```

Write what you want the voice to do. Tags are a hint, not a guarantee —
delivery also depends on the reference clip's own style and on `temperature`,
and an instruction far from anything in the training distribution may simply
be ignored.

---

## Tuning

### Latency

Time-to-first-audio on a GTX 1650 is **286 ms** for Latin text streamed in the
way a voice agent produces it, and **123 ms** once the prompt is already
prefilled. **[docs/LATENCY.md](LATENCY.md)** has the full breakdown; the short
version is:

```
TTFA = (N−1) × 48 ms   prompt tokens not already cached
     +  97 ms          the last prompt token, which also samples frame 1
     +  19 ms          codec decode of that frame
```

`N` is `prefix_cache.last_prefilled_tokens` in `/status`. Three things drive it
down, in order:

1. **Set the voice once, at startup.** `POST /v1/voice` prefills the reference
   clip's ~240–350 prompt tokens in the background, and every turn afterwards
   reuses them. Setting it per turn throws that away.
2. **Stream `text` frames as your LLM emits them** rather than sending the
   finished sentence. The server prefills the prompt in that gap. Sending it
   whole gives it no gap to work in, and costs roughly 300 ms.
3. **Keep reference clips near 9–10 s.** Prompt length scales with clip length,
   and a longer one costs more on every cache miss.

Throughput is a separate problem: generation runs ~2.5× slower than realtime on
this card, so audio starts quickly and then falls behind. Clients must buffer.
That is the NF4 kernel and the card, not the server.

On the clone route, latency is also proportional to how much text you sent:
you get the first segment quickly, but the last one lands after everything
before it has been generated.

Set generous client timeouts. On a small GPU a 3-sentence utterance can take
tens of seconds end to end; `/status` reports `last_ttfa_ms` from the most
recent request.

### Quality

| Symptom | Try |
| --- | --- |
| Voice doesn't resemble the clip | Check `reference_text` is *exact*. This is the usual cause. |
| Voice drifts between sentences | Pin `seed`. Raise `chunk_length` toward 300 so a sentence is fewer batches. |
| Speech stops mid-word | Check `prefix_cache.truncations` in `/status`. Above zero means a batch ran out of token budget — shorten the reference clip or raise `TTS_MAX_SEQ_LEN`. |
| Long silence, then everything at once | An unclosed `[` used to stall segmentation for a whole turn; fixed, but check your text for stray brackets. |
| A click where a short segment should be | Fixed (a first-token `im_end` was decoded as audio). Report it if you still hear one. |
| Flat, robotic delivery | Raise `temperature` toward 1.0. Use a more animated reference clip. |
| Rambling, unstable, invented words | Lower `temperature` to ~0.6. Raise `repetition_penalty` toward 1.3. |
| Stutters or loops a syllable | Raise `repetition_penalty`. |
| Numbers and abbreviations read wrong | Keep `normalize: true`. Spell out the tricky ones in `text`. |
| Background noise or reverb in the output | It is in your reference clip. The clone inherits the room. |

---

## Operations

### `GET /health`

| Response | Meaning |
| --- | --- |
| `200 {"status": "loading"}` | Model still coming up. **Not an error** — don't restart the container. |
| `200 {"status": "ok"}` | Ready. |
| `503 {"status": "error", "error": "..."}` | The load genuinely failed. |

### `GET /status`

Engine state, device, VRAM, queue depth, request and cancellation counters,
`last_ttfa_ms`, and the full effective configuration.

```bash
curl -s localhost:8080/status | python -m json.tool
```

### Concurrency

There is **one GPU worker**, by design: S2-Pro in NF4 leaves no headroom for
concurrent generations. Requests from many clients are queued and served in
order — concurrency lives in the transport layer, not on the GPU. Up to
`TTS_MAX_SESSIONS` (default 8) WebSocket connections are accepted; a ninth is
closed with code 1013.

A queued request's latency is its own generation time *plus* everything ahead
of it. Watch `queue_depth` in `/status`.

### Configuration

All environment variables, read identically by `run.sh` and the app; `.env` at
the repo root is sourced if present. The full table is in the
[README](../README.md#configuration). The ones you are most likely to touch:

| Variable | Default | Effect |
| --- | --- | --- |
| `TTS_PORT` | `8080` | Bind port. |
| `TTS_MIN_SEGMENT_CHARS` | `24` | Lower ⇒ faster first audio, choppier prosody. |
| `TTS_CHUNK_LENGTH` | `200` | **Bytes** per generated batch. Lower it for non-Latin scripts. |
| `TTS_STREAM_FIRST_FRAMES` | `1` | Frames in the first audio block — the direct TTFA lever, ~90 ms each. |
| `TTS_PREFIX_CACHE` | `true` | Reuse prompt KV across requests. Turning it off costs seconds per turn. |
| `TTS_MAX_SESSIONS` | `8` | Concurrent WebSocket connections. |
| `TTS_LOG_LEVEL` | `INFO` | `DEBUG` to trace segmentation and engine jobs. |
| `TTS_ENGINE` | *(unset)* | `fake` for a GPU-free server. |

`TTS_CHUNK_LENGTH` and `TTS_DEFAULT_FORMAT` are the server-wide defaults for
the per-request `chunk_length` and `format` fields. A request that names either
field keeps its own value; one that does not inherits the server's. (They used
to be read by nothing at all — `TTSParams` carried its own defaults, so after
validation there was no way to tell a configured value from an unset one. See
`with_server_defaults` in `app/protocol.py`.)

---

## Troubleshooting

| What you see | What it is |
| --- | --- |
| `503`, `{"status": "loading"}` | The model is still loading (52–82 s on a GTX 1650). Poll `/health`. |
| `503 engine unavailable: ...` | The load failed. Check the server log; `./run.sh --preflight-only` verifies the environment. |
| `400` mentioning `reference_text` | The transcript is missing or blank. It must be the clip's exact words. |
| `415` | Set `Content-Type` — `curl -F` or `-H 'Content-Type: application/json'`. |
| `422` | A parameter is out of range or misspelled. The body lists the offending field. |
| WebSocket closes with 1013 | At `TTS_MAX_SESSIONS`, or the engine isn't up yet. |
| WebSocket closes with 1002 | Protocol error — usually `text` before `start`. The `finish` frame before the close says which. |
| Output is silent but well-formed | The Turing fp16 codec bug. `TTS_CODEC_PRECISION=auto` (the default) handles it; if you forced `fp16` below sm_80, don't. |
| Output is truncated, no error | A mid-stream failure after the status line was sent. Search the log for your `X-Request-Id`. |
| Clone sounds nothing like the clip | `reference_text` almost certainly doesn't match the audio. Check it word by word. |
| WebSocket speaks in the wrong voice | Read `started.voice_id`. `"inline"`/a name means your own `references`/`reference_id` overrode the active voice; `null` means none was set or `use_active_voice` was false. |
| The active voice came back after a restart | It is meant to — it persists in `references/_active/`. `DELETE /v1/voice` to be rid of it. |
| `GET /` is a 404 | `static/index.html` is missing. The server works fine without it; the page is only the test console. |
| Everything is slow | Expected below Ampere. Check `/status` for `queue_depth` — you may be behind another request. |
| CUDA OOM on a 4 GB card | Another process has the GPU, or `TTS_OFFLOAD_EMBEDDINGS` got turned off. Drop `TTS_MAX_SEQ_LEN` to 1024. |

---

## Testing your integration

The fake engine gives you the entire API surface with no GPU and no weights:

```bash
TTS_ENGINE=fake .venv/bin/python -m uvicorn app.main:app --port 8080
```

Both routes behave identically — same validation, same formats, same headers,
same streaming and cancellation — so you can build and test a client against
it and only need a GPU to hear the result.

```bash
.venv/bin/python -m pytest tests -v        # the server's own suite, no GPU needed
```
