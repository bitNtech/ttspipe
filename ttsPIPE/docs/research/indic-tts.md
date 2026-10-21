# AI4Bharat Indic-TTS (FastPitch + HiFi-GAN, Tamil) — verification report

**Verdict: works on CPU but too slow and heavy for the default** (RTF ≈ 0.75–0.84, 2 GB RSS, no
streaming, drops all Latin text). Not shipped as a module; Piper's Rasa voices beat it on every axis
we can measure.

| item | verified value |
|---|---|
| repo | <https://github.com/AI4Bharat/Indic-TTS>, checkpoints: release `v1-checkpoints-release/ta.zip` |
| licence | repo MIT; checkpoints released alongside under the same repo (no separate model licence file found in `ta.zip`). Treat as MIT, verify with AI4Bharat before commercial use. |
| download | `ta.zip` **≈ 3.0 GB** extracted (FastPitch + HiFi-GAN + speaker file) |
| languages | Tamil only (character vocabulary is Tamil script + punctuation) |
| voices | 2: `female`, `male` (`speakers.pth`) |
| native rate | 22 050 Hz |
| streaming | no (whole utterance) |
| deps | Coqui TTS (`coqui-tts` fork for Python 3.12) + torch: ≈ 1.5 GB of packages |
| RAM | RSS 2.06 GB after load, 2.1 GB peak |
| load time | 11–12 s |

## Measured (i5-12450H proxy, CPU only)

| threads | text | audio s | synth s (median) | RTF | TTFA |
|---|---|---|---|---|---|
| 4 | ta | 7.68 | 6.44 | 0.84 | 6.4 s |
| 4 | tanglish_script | 3.15 | 2.45 | 0.78 | 2.4 s |
| 4 | tanglish_roman | 1.64 | 1.06 | 0.65 | 1.1 s (Latin dropped — see below) |
| 4 | en | **error**: `RuntimeError: Given groups=1, weight of size [256, 512, 3], expected input[1, 1, 512] to have 512 channels, but got 1 channels instead` (all characters dropped → empty input) | | | |
| 4 | short_ta | 2.44 | 1.83 | 0.75 | 1.8 s |
| 8 | ta | 7.68 | 6.25 | 0.81 | 6.2 s |
| 8 | short_ta | 2.44 | 1.80 | 0.74 | 1.8 s |

Some of these runs overlapped with other jobs on the box; Piper was re-measured on a quiet box for comparison.

## Tamil / Tanglish

* Tokenizer check (`tok_*` in `data/research_results/indic-tts-fastpitch-t4.json`): Tamil kept, **every
  Latin character dropped** — `"நாளைக்கு meeting இருக்கு, please time-க்கு வந்துடுங்க."` becomes
  `"நாளைக்கு  இருக்கு,   க்கு வந்துடுங்க."`. Tanglish needs transliteration plus routing.
* Subjective quality: UNVERIFIED (WAVs in `data/research_wavs/indic-tts-fastpitch/`).
