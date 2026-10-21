# Meta MMS-TTS Tamil (`facebook/mms-tts-tam`) — verification report

**Verdict: usable but not the default.** Real-time on CPU (RTF ≈ 0.54 at 4 threads), but **non-commercial
licence**, no streaming, Latin text dropped, and about 5× slower than Piper Rasa. Shipped as
`tts/mms_tam` (family `mms`) for comparison and A/B tests.

| item | verified value |
|---|---|
| repo | <https://huggingface.co/facebook/mms-tts-tam> (not gated) |
| licence | **CC-BY-NC-4.0** (HF `cardData.license`): **no commercial use** |
| download | `model.safetensors` 145 MB + 4 small JSON files (139 MB on disk) |
| languages | Tamil only. `tokenizer_config.json`: `is_uroman: false`, so it takes Tamil script directly. Vocabulary: 58 symbols = Tamil letters/signs, digits and **one Latin letter (`a`)**. |
| voices | 1 (male) |
| native rate | 16 000 Hz |
| streaming | no: VITS synthesizes the whole input; TTFA = full synthesis |
| deps | `torch` (CPU wheel ≈ 200 MB) + `transformers` (`VitsModel`) |
| RAM | RSS 370 MB after load, 595–621 MB peak |
| load | 7 s warm, 13–32 s cold |

## Measured — quiet box, 1 warm-up + 5 runs, median (i5-12450H proxy; target is an 8-core Ryzen, 14 GB, no GPU)

| threads | text | audio s | synth s | RTF | TTFA ms |
|---|---|---|---|---|---|
| 2 | ta | 6.56 | 5.00 | 0.762 | 4 999 |
| 2 | short_ta | 2.05 | 1.51 | 0.740 | 1 515 |
| **4** | ta | 6.56 | 3.56 | **0.543** | 3 560 |
| 4 | tanglish_script | 2.69 | 1.46 | 0.542 | 1 457 |
| 4 | short_ta | 2.05 | 1.45 | 0.707 | **1 447** |
| 8 | ta | 6.56 | 4.81 | 0.734 | 4 813 |
| 8 | short_ta | 2.05 | 1.48 | 0.723 | 1 480 |

(`tanglish_roman`, `en` and `short_en` produce 0.24–0.88 s of audio: everything but the letter `a`
was dropped, see below.)

## Tamil / Tanglish (objective)

Tokenizer round-trip (`tok_*` in `data/research_results/mms-tts-tam-t4.json`):

| input | what the model receives |
|---|---|
| `நாளைக்கு meeting இருக்கு, please time-க்கு வந்துடுங்க.` | `நாளைக்கு  இருக்கு a க்கு வந்துடுங்க` — English words removed |
| `Naalaikku meeting irukku, please time ku vandhudunga.` | `aaa   a   aa` |
| `Hello, how are you?` | `a` |

So with MMS: romanized Tamil **must** be transliterated (ttsPIPE `g2p/lexicon` does this), and every
English word must be routed to an English engine (`tts/lang_router` with `min_run_words: 1`).

Subjective quality: UNVERIFIED (WAVs in `data/research_wavs/mms-tts-tam/`). The MMS paper reports
CER-based intelligibility, not MOS, for Tamil.

## Recommended params

`threads: 4`, `noise_scale: 0.667`, fixed `seed` for deterministic output (needed by the phrase cache).
