# Fish Audio S2-Pro (NF4) — `tts/fish_s2`

| | |
|---|---|
| Model | `scarxity/fish-speech-s2-pro-nf4`: bitsandbytes NF4 quantisation of Fish Audio S2-Pro (4B Dual-AR transformer over a 10-codebook RVQ codec) |
| Runtime | The Fish server in the parent repo (`..`, `app/engine/*`): FastAPI + one GPU worker, custom NF4 GEMV kernels, CUDA graphs, streaming codec. It has its own `.venv` (torch 2.8.0+cu128). ttsPIPE talks to `ws://127.0.0.1:8080/v1/tts/live` (msgpack). |
| Download | ~4.9 GB (`model.pth` 3.1 GB + `codec.pth` 1.9 GB), fetched by that repo's `./run.sh` |
| Licence | **Fish Audio Research Licence**: research and non-commercial use free; any commercial use (incl. a hosted service or internal business operations) needs a separate written licence from Fish Audio (checkpoint `LICENSE.md`) |
| Languages | ~50 incl. Tamil and English; Tanglish (Tamil script with English words) in one voice |
| Cloning | Zero-shot from a reference clip + its exact transcript (~10 s is best) |
| Streaming | Yes: 44.1 kHz PCM16 in blocks of 1–4 codec frames (46.4 ms each) |
| Hardware | GPU only. Measured on the dev laptop: RTX 2050 4 GB (sm_86), Windows 11, 16 GB RAM — **not** the CPU-only target box |

## Measured (RTX 2050, 2026-10-09)

Server start to ready: 79.5 s model load (+ ~15 s active-voice prefix warm-up). VRAM in use while
serving: 3971 of 4096 MiB (the server caps itself; see the parent repo's docs/LATENCY.md).

Engine alone (`tts/fish_s2.synthesize`, one turn per text, warm server, server active voice):

| text | first audio | total | audio | RTF |
|---|---|---|---|---|
| Tamil sentence (5 words) | 660 ms | 5.15 s | 4.55 s | 1.13 |
| English sentence (9 words) | 83 ms | 2.84 s | 2.79 s | 1.02 |
| English, model default voice | 97 ms | 2.61 s | 2.55 s | 1.02 |
| Tanglish (Tamil script + English) | 779 ms | 3.04 s | 2.28 s | 1.34 |

Tamil costs more before the first frame because it is token-dense (about 11 prompt tokens per word),
and all of that prompt is prefilled before sampling starts.

Through the full pipeline (default profile: 5-word first chunk, 300-char chunks, 400 ms jitter buffer):

| run | TTFA | first synth | RTF | notes |
|---|---|---|---|---|
| Live `/ws/tts`, Tanglish, 15 words, cloned voice | 881 ms | 144 ms | 1.23 | underruns at chunk starts: generation is slower than playback |
| Render, Tamil + English, 11 words | 1.08 s | 444 ms | 1.08 | |
| Render, new clone, first use (cold) | 12.7 s | 11.1 s | 4.3 | reference encoded on the server's CPU codec + 220-token prefix prefill |
| Render, same clone after the save-time warm-up | 1.10 s | 461 ms | 1.06 | ttsPIPE now warms a clone when it is saved |

TTFA in the pipeline = first synth + the time to fill the jitter buffer at RTF ≈ 1.

## Verdict

Best voice quality and the only engine here with Tamil + English in one natural voice and voice
cloning, so it is the default engine on a machine with this GPU. Limits:

* **Not real time with margin**: RTF 1.0–1.3 on an RTX 2050 means long utterances underrun (short
  silences) unless the jitter buffer covers the deficit. 400 ms is the compromise; raise
  `playback.jitter_ms` for smoother long replies at the cost of first audio.
* **Voice switches are not free**: the server keeps the KV prefix for one voice; switching voices
  (A/B, clone tests) re-prefills ~220 tokens.
* **Non-commercial licence.**
* No speed or pitch control (the module ignores `voice.speed` / `pitch`; tag `no-speed`).
* Needs the GPU box; the CPU-only target falls back to the builtin tone unless the `balanced` (Piper)
  profile is applied.
