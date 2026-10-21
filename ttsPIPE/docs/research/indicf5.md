# AI4Bharat IndicF5 — verification report

**Verdict: not usable for ttsPIPE on CPU** (RTF 12–126; first audio after 25–280 s). Kept out of the module set.

| item | verified value |
|---|---|
| repo | <https://huggingface.co/ai4bharat/IndicF5> (code: F5-TTS flow matching + Vocos vocoder) |
| licence | MIT (HF `cardData.license`), repo is **gated** (`gated: auto`, click-through). Commercial use: yes (MIT). |
| download | model.safetensors + config ≈ **1.4 GB** on disk (`data/models/indicf5`, 1.4 G). Fetched from the ungated mirror `rsolanki1822/IndicF5-mirror`, whose LFS sha256 matches upstream (checked via HF API `lfs.sha256`). |
| languages | 11 Indic languages incl. Tamil (model card); English not a training language |
| voices | **zero-shot voice cloning**: every request needs a reference clip + its transcript |
| native rate | 24 000 Hz |
| streaming | no — whole utterance from a flow-matching ODE; TTFA = full synthesis time |
| deps | torch, f5-tts code (repo `model.py`), vocos, safetensors; separate venv (`.venv-research-indic`) |
| RAM | RSS 2.0 GB after load, **3.3 GB peak** |
| load time | 13 s (warm disk), 87 s cold |

## Measured (i5-12450H proxy, CPU only, `CUDA_VISIBLE_DEVICES=""`)

Target deployment box is an 8-core Ryzen with 14 GB RAM and no GPU; these numbers come from the
i5-12450H (4P+4E cores) proxy. Script: `scripts/research/bench_indicf5.py`.

| config | text | audio s | synth s (median) | RTF | TTFA |
|---|---|---|---|---|---|
| NFE 32 (default), 8 threads, Punjabi prompt | short_ta | 2.2 | 280.8 (1 run) | 126.6 | 280 s |
| NFE 8, 4 threads, Tamil prompt | ta | 5.2 | 60.3 | 11.7 | 60 s |
| NFE 8, 4 threads | tanglish_script | 3.0 | 41.0 | 13.8 | 41 s |
| NFE 8, 4 threads | tanglish_roman | 1.4 | 24.5 | 17.5 | 24 s |
| NFE 8, 4 threads | short_ta | 1.9 | 25.8 | 13.4 | 26 s |

Even with 4× fewer flow steps (NFE 8, which also lowers quality) the model is >10× slower than real time.
These runs overlapped with other work on the box at times, which inflates them; the order of magnitude
does not change the verdict.

## Quality notes

* Tamil script: WAVs in `data/research_wavs/indicf5/` (subjective quality UNVERIFIED — no listening test here).
* Tanglish: Latin words go through the model's character vocabulary; behaviour on English words UNVERIFIED.
* Needs a reference speaker clip; the Tamil prompt used here was a Piper Rasa output (synthetic, documented in the script).

## Recommendation

Do not ship as a real-time engine on CPU. It would only fit an offline "render a phrase-cache clip"
job (`tts/clips`), where a 60 s render per prompt is acceptable.
