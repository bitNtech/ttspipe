# ttsPIPE

Real-time text-to-speech pipeline for **Tamil, Tanglish and English**: the post-LLM half of a
voice agent (prePIPE is the pre-LLM half). It takes an LLM's streamed text, speaks it with the lowest
possible time-to-first-audio, and supports full-duplex interruption (duck / stop with exact
`heard_text` / resume). A web control center shows every stage live.

The default engine is **Fish Audio S2-Pro** (4B, NF4) on the GPU, served by the Fish server in the
parent repo (`..`), with zero-shot **voice cloning**. Every other engine runs on CPU; the `balanced`
profile (Piper) is the CPU-only setup.

* Binding backend ↔ frontend contract: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* Model verification reports with numbers measured on CPU: [docs/research/](docs/research/README.md)

## Quick start

```bash
./run.sh setup      # .venv (core deps only) + frontend npm install
./run.sh models     # ML deps + model files for the engines the builtin profiles use (incl. the Fish client)
./run.sh dev        # backend :8200 + dashboard :5174 (open http://127.0.0.1:5174, or /?mock=1 without a backend)
./run.sh test       # pytest (core deps only) + tsc -b
./run.sh smoke      # speak every sample script with tts/tone and the default engine, check stop/heard_text
./run.sh doctor     # environment, modules, models, ports
./run.sh stop
```

The server starts and serves `/api/modules` with **only** `backend/requirements.txt` installed. Every
stage has a builtin fallback; a configured ML engine that is missing falls back to the builtin and
`/api/health` reports `degraded`.

The dashboard has four pages:

* **Test** — type or load text, pick any engine and voice, then *Speak* it live through the real-time
  pipeline or *Render* it to a WAV; every run keeps its TTFA / first-synth / RTF / duration and audio.
* **Engines** — every TTS engine with status, model files (download), parameters (apply, or make it the
  pipeline engine), voice preview, profile switching, and the Fish S2-Pro server (start / stop / log).
* **Voice cloning** — record (or upload) a ~10 s clip, give its exact transcript, save it; test the clone
  on Fish S2-Pro and make it the pipeline voice. Clones live in `data/voices/`.
* **Monitor** — the live panel grid: pipeline graph, output timeline, playback state, latency,
  utterances, diagnostics, A/B compare.

## Fish Audio S2-Pro (GPU)

`tts/fish_s2` is a client: the model runs in the Fish server repo one directory up (`TTSPIPE_FISH_DIR`,
its own `.venv` with CUDA torch and the ~4.9 GB `checkpoints/s2-pro-nf4`, provisioned by that repo's
`./run.sh`). When the engine loads and the server is not running, ttsPIPE starts it (about 1.5 min
to load on an RTX 2050) and logs to `data/logs/fish-server.log`. Measured on the RTX 2050 laptop
([docs/research/fish-s2-pro.md](docs/research/fish-s2-pro.md)): first audio ~85–145 ms for English and
~0.45–0.8 s for Tamil from the engine, RTF ~1.0–1.3, so the default profile uses a 400 ms jitter buffer
and long utterances can still underrun. Licence: **Fish Audio Research Licence — non-commercial**; any
commercial use needs a licence from Fish Audio.

## Pipeline

```
LLM deltas → ingest → normalize → segment → chunker → g2p → prosody → tts (+ shadow tts) → post → PCM16 frames
```

| stage | builtin | job |
|---|---|---|
| ingest | `plain` | strip markdown / emoji / code / URLs, SSML-lite (`<break>`, `<emphasis>`, `<lang>`, `<prosody>`), hold partial words |
| normalize | `rules` | numbers, ₹, dates, times, phone numbers, units, abbreviations — worded in Tamil or English by context |
| segment | `script` | `ta` / `en` / `ta-Latn` runs (Unicode ranges + Tanglish lexicon + suffix heuristics) |
| chunker | `punct` | small first chunk (≥ 4 words at a clause end) for fast first audio, then sentences / clauses |
| g2p | `lexicon` | `config/lexicon.yaml` overrides + romanized Tamil → Tamil script |
| prosody | `rules` | pauses, question pitch, emphasis, SSML-lite rate/pitch |
| tts | `tone`, `clips`, `lang_router` | test tones with exact word timings; phrase clips; per-language routing to two engines |
| post | `master`, `passthrough` | DC, de-click fades, silence trim, loudness, limiter |

ML engines, with licences and measured CPU numbers in [docs/research/README.md](docs/research/README.md):

| module | family (`./run.sh models <family>`) | use |
|---|---|---|
| `tts/fish_s2` | `fish` | **default** (GPU, Fish server): Tamil, English, Tanglish in one voice; cloned voices |
| `tts/piper` | `piper` | CPU default (`balanced`): Tamil AI4Bharat Rasa voices (Tanglish in Tamil script works in one voice) + English ljspeech |
| `tts/kokoro` | `kokoro` | English, `high_quality` profile |
| `tts/mms_tam` | `mms` | Tamil, non-commercial licence; A/B shadow in `ab_tts_compare` |

The `balanced` profile (CPU-only) routes Tamil / romanized-Tamil runs to Piper Rasa and English runs
of ≥ 3 words to Piper ljspeech via `tts/lang_router`. Until `./run.sh models` has run, the TTS stage
falls back to `tts/tone` and `/api/health` reports `degraded`.

Licences: Fish S2-Pro weights are non-commercial (Fish Audio Research Licence); the `piper-tts` runtime and eSpeak NG are GPL-3.0; MMS-TTS weights are non-commercial;
the Piper `lessac` English voice is trained on research-only data. See the research reports.

## Clients

`ws://host:8200/ws/tts?session_id=&label=&voice=` — send `{"type":"text","delta":…,"utterance_id":…}`,
`{"type":"end_utterance"}`, `{"type":"speak","text":…}`, `duck` / `stop` / `resume` / `reset` / `ping`;
receive binary `[0x01][PCM16 LE 20 ms frame]` plus JSON events. On `stop` the server sends
`utterance.interrupted {heard_text, unplayed_text}` for the LLM's context. Full protocol: ARCHITECTURE.md §8.

```python
import asyncio, json, websockets
async def main():
    async with websockets.connect("ws://127.0.0.1:8200/ws/tts?label=demo") as ws:
        await ws.send(json.dumps({"type": "speak", "text": "வணக்கம்! Your order ships tomorrow."}))
        async for m in ws:
            if isinstance(m, bytes): pass                     # 20 ms PCM16 frame after the 1-byte kind
            elif json.loads(m)["type"] == "utterance.end": break
asyncio.run(main())
```

## Configuration

`config/pipeline.yaml` holds the defaults; runtime edits from the dashboard or REST persist to
`data/pipeline.override.yaml`. Builtin profiles live in `config/profiles/` (`fish_s2`, `balanced`, `lowest_latency`,
`high_quality`, `minimal`, `ab_tts_compare`, `english_fast`); user profiles and layouts in `data/`.
Environment: `TTSPIPE_HOST`, `TTSPIPE_PORT` (8200), `TTSPIPE_DATA_DIR`, `TTSPIPE_CONFIG`,
`TTSPIPE_MODELS_DIR`, `TTSPIPE_THREADS`, `TTSPIPE_LOG_LEVEL`, `TTSPIPE_FISH_DIR`, `TTSPIPE_FISH_URL`
(`http://127.0.0.1:8080`), `HF_TOKEN` (gated downloads only).

## prePIPE bridge

Set `bridge.enabled: true` and `bridge.prepipe_url` to map prePIPE `barge_in.candidate` / `barge_in` /
`backchannel` to duck / stop / resume, report the agent's speaking state back, and optionally stream the
played audio as prePIPE `kind=2` reference audio.
