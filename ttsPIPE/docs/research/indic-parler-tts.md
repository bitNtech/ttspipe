# AI4Bharat Indic Parler-TTS — verification report

**Verdict: not usable in real time on CPU** (RTF 15–17, first streamed audio after 7–9 s, 4.4 GB RAM).
Its strengths (a natural-language voice description, 69 speakers, code-mixed text) cannot pay for that
latency. Not shipped as a module.

| item | verified value |
|---|---|
| repo | <https://huggingface.co/ai4bharat/indic-parler-tts>; runtime `parler-tts` (git main, pins `transformers==4.46.1`) |
| licence | **Apache-2.0** (HF `cardData.license`); repo is **gated** (`gated: auto`). Weights fetched from mirror `RXD03/indic-parler-tts`; LFS sha256 matches upstream. Commercial use: yes. |
| download | **3.6 GB** (`model.safetensors` 3.5 GB + tokenizer files); the description encoder tokenizer (`google/flan-t5-large`) downloads on first use |
| languages | 21 incl. Tamil and English. The prompt tokenizer (90 714 pieces) keeps Latin: `நாளைக்கு meeting இருக்கு` → `▁நாள ைக்கு ▁meeting ▁இருக்கு`, 0 unknown tokens on every test text, so Tanglish in either script reaches the model |
| voices | 69 named speakers selected by a text description (Tamil: Jaya, Kavitha, Ravi, …) |
| native rate | 44 100 Hz (DAC codec at 87 frames/s) |
| streaming | yes: `ParlerTTSStreamer` decodes every `play_steps` frames (0.5 s used here) |
| deps | `parler-tts`, `transformers==4.46.1`, `descript-audio-codec`, `torch`, `torchaudio` (matching versions; a mismatched torchaudio fails with `OSError: [WinError 127]`). Needs its **own venv**: the transformers pin conflicts with other families. `attn_implementation="sdpa"` fails on the T5 encoder (`ValueError: T5EncoderModel does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention yet`) → use `"eager"`. |
| RAM | RSS 3.98 GB after load, **4.36 GB peak** |
| load | 34 s |

## Measured — 4 threads, 1 warm-up + 2 runs, median (i5-12450H proxy; target is an 8-core Ryzen, 14 GB, no GPU)

| text | audio s | synth s | RTF | TTFA (first streamed 0.5 s) |
|---|---|---|---|---|
| ta | 4.83 | 80.8 | 16.7 | 9.2 s |
| tanglish_script | 2.36 | 37.9 | 16.1 | 8.2 s |
| short_ta | 1.63 | 24.2 | 14.9 | 7.5 s |
| short_en | 1.64 | 25.9 | 15.8 | 7.4 s |

Two timed runs instead of five: one run takes 25–80 s, and the verdict holds at any reasonable variance.

## Notes

* Output level varies by speaker description (−15.7 to −32 dBFS RMS). `post/master` would need to normalize hard.
* Subjective quality: UNVERIFIED here. The model card reports MOS for its Indic speakers.
* A GPU or a distilled checkpoint would be needed. Revisit if AI4Bharat ships a "mini" Indic Parler.
