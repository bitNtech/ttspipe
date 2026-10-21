# Piper (VITS, ONNX) — Tamil Rasa voices + English voices — verification report

**Verdict: default engine for Tamil and Tanglish, and for English in the `balanced` profile.**
RTF ≈ 0.10–0.14 and a first sentence in ≈ 0.16–0.34 s on 2–4 threads, 150–330 MB RAM, true per-sentence
streaming, and one Tamil voice can say Tanglish written in Tamil script.

| item | verified value |
|---|---|
| runtime | `piper-tts` 1.8.0 (<https://github.com/OHF-voice/piper1-gpl>) — **GPL-3.0-or-later**, bundles eSpeak NG (GPL-3.0). In-process use on our own server is fine; distributing ttsPIPE with it makes the bundle GPL. |
| Tamil voices | `ta_IN-rasa_female-medium`, `ta_IN-rasa_male-medium` — <https://huggingface.co/tinisoft/piper-ta_IN-rasa_female-medium> (and `_male_`). Model card: **CC-BY-4.0**, trained on the AI4Bharat **Rasa** dataset (CC-BY-4.0). Third-party upload (2026-05-18), not an official rhasspy voice. Commercial use: yes, with attribution. |
| English voices | rhasspy `piper-voices` (repo MIT). Per-voice data licence from the MODEL_CARD: `ljspeech` and `kristin` = **public-domain data** (commercial OK); `lessac` = Blizzard-2013 Lessac licence (**research/non-commercial** click-through); `amy` = mimic3 dataset ("see URL", unclear). Default English voice: `en_US-ljspeech-medium`. |
| download | 61–63 MB per voice (`.onnx`) + 5 KB `.onnx.json`; the module's default set (2 Tamil + 1 English) = **186 MB** |
| languages | Tamil voice: eSpeak `ta` phonemes; English words are phonemized with English rules (100 % phoneme-map coverage on every test text, see below). English voices: `en_US`. |
| native rate | 22 050 Hz |
| streaming | **yes, per sentence** (`PiperVoice.synthesize` yields one `AudioChunk` per sentence) |
| word timings | not without patching the model (`include_alignments` needs the `onnx` package); ttsPIPE estimates them |
| deps | `piper-tts` + `onnxruntime` ≈ 60 MB of wheels, no torch |
| RAM | RSS 150–160 MB after loading one voice, 270–335 MB peak |
| load | 4–7 s per voice (cold), < 1 s warm |

## Measured — quiet box, 1 warm-up + 5 runs, median (i5-12450H proxy; target is an 8-core Ryzen, 14 GB, no GPU)

TTFA = time to the first sentence's audio. VITS samples durations stochastically (`noise_w`), so audio
length varies a little between runs.

### `ta_IN-rasa_female-medium`

| threads | text | audio s | synth s | RTF | TTFA ms |
|---|---|---|---|---|---|
| 1 | ta | 4.56 | 0.81 | 0.177 | 548 |
| 1 | short_ta | 1.46 | 0.33 | 0.225 | 309 |
| 2 | ta | 4.83 | 0.55 | 0.114 | 368 |
| 2 | tanglish_script | 2.46 | 0.31 | 0.125 | 324 |
| 2 | tanglish_roman (raw Latin) | 3.24 | 0.37 | 0.115 | 377 |
| 2 | en | 3.52 | 0.43 | 0.121 | 97 |
| 2 | short_ta | 1.86 | 0.24 | 0.129 | 233 |
| 2 | short_en | 1.17 | 0.15 | 0.131 | 165 |
| **4** | ta | 4.62 | 0.47 | **0.102** | 337 |
| 4 | tanglish_script | 2.68 | 0.26 | 0.097 | 273 |
| 4 | tanglish_roman (raw Latin) | 3.18 | 0.27 | 0.086 | 284 |
| 4 | en | 3.59 | 0.59 | 0.164 | 105 |
| 4 | short_ta | 1.66 | 0.18 | 0.107 | **204** |
| 4 | short_en | 1.25 | 0.15 | 0.118 | 141 |
| 8 (earlier run, other load present) | short_ta | 2.00 | 0.48 | 0.243 | 437 |

8 threads is slower than 4 on this 4P+4E CPU (work lands on the efficiency cores). The 8-core Ryzen
target has no E-cores, so re-check 4 vs 8 there; ttsPIPE defaults to **4**.

### `ta_IN-rasa_male-medium` (2 threads)

| text | audio s | synth s | RTF | TTFA ms |
|---|---|---|---|---|
| ta | 3.98 | 0.56 | 0.140 | 409 |
| tanglish_script | 2.32 | 0.34 | 0.145 | 345 |
| short_ta | 1.28 | 0.17 | 0.130 | 163 |
| en | 5.06 | 0.49 | 0.096 | 79 |

### English voices (`en` / `short_en`)

| voice | threads | en RTF | en TTFA ms | short_en TTFA ms |
|---|---|---|---|---|
| en_US-ljspeech-medium | 2 | 0.138 | 86 | 154 |
| en_US-ljspeech-medium | 4 | 0.112 | 65 | 145 |
| en_US-kristin-medium | 2 | 0.145 | 122 | 161 |
| en_US-lessac-medium | 2 | 0.121 | 84 | 153 |

(`en` TTFA is short because its first sentence is just "Hello!".)

## Tamil / Tanglish findings (objective)

* eSpeak `ta` output for `நாளைக்கு meeting இருக்கு, please time-க்கு வந்துடுங்க.` is
  `nˈaːɭaɪkkʉ mˈiːtɪŋ ˈiɹʉkkʉ, plˈiːz tˈaɪmkkʉ vˈʌndʉɖˌʉŋɡʌ.`: English words get English phonemes,
  and every symbol is in the Rasa voice's 157-entry phoneme map (coverage 1.0). Whether the voice
  *renders* rare English phonemes (`θ`, `ɹ`, `ʃ`) well depends on how often they appear in Rasa's
  training audio. That is UNVERIFIED without a listening test, which is why `balanced` routes English
  runs of ≥ 3 words to an English voice.
* Romanized Tamil (`Naalaikku … irukku`) is phonemized as **English** (`nˈɑːleɪkˌuː`), so it would
  sound anglicized. ttsPIPE's `g2p/lexicon` transliterates `ta-Latn` to Tamil script before TTS.
* Durations are plausible: about 0.33–0.4 s per Tamil word, in line with normal speech.
* WAVs: `data/research_wavs/piper/`. Subjective MOS: UNVERIFIED (no published MOS for these voices).

## Recommended params

`threads: 4`, `noise_scale: 0.667`, `noise_w: 0.8` (voice defaults), voices
`ta_IN-rasa_female-medium` (Tamil) + `en_US-ljspeech-medium` (English, commercially clean).
