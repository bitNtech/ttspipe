# Kokoro-82M (via `kokoro-onnx`) — verification report

**Verdict: English only, `high_quality` English option, not the default.** It is the most natural
small English model of the set, but on this CPU it runs at RTF ≈ 0.63–0.85 (4 threads) with ~0.8–1.1 s
to the first sentence, versus ~0.1 RTF / 0.15 s for Piper English. No Tamil.

| item | verified value |
|---|---|
| repo | weights <https://huggingface.co/hexgrad/Kokoro-82M>; ONNX runtime package <https://github.com/thewh1teagle/kokoro-onnx> (0.6.1); model files from release `model-files-v1.1` |
| licence | weights **Apache-2.0** (HF card). `kokoro-onnx` MIT; it depends on `phonemizer` (GPL-3.0) and `espeakng-loader` (bundles eSpeak NG, GPL-3.0). Commercial use of the weights: yes. |
| download | `kokoro-v1.0.onnx` 310 MB (fp32), `kokoro-v1.0.int8.onnx` 88 MB, `kokoro-v1.0.fp16.onnx` 169 MB, `voices-v1.0.bin` 27 MB |
| languages | 54 voices: American and British English, plus es, fr, hi, it, ja, pt, zh. **No Tamil** — Tamil text phonemized with eSpeak `ta` loses its `ʉ` vowels on Kokoro's vocabulary (`ˈiɹʉkkʉ` → `ˈiɹkk`): unusable. |
| voices used | `af_heart`, `af_bella`, `af_nicole`, `am_michael`, `am_adam`, `bf_emma`, `bm_george` |
| native rate | 24 000 Hz |
| streaming | the library only splits at 510 phonemes; ttsPIPE gives it one chunk at a time (non-streaming module) |
| deps | `kokoro-onnx`, `onnxruntime`, `phonemizer`, `espeakng-loader` (≈ 80 MB of wheels, no torch) |
| RAM | RSS 400 MB after load (fp32), 625 MB peak; int8 206/435 MB |
| load | 1.6–2.1 s |

## Measured — quiet box, 1 warm-up + 5 runs, median (i5-12450H proxy; target is an 8-core Ryzen, 14 GB, no GPU)

TTFA = first sentence synthesized on its own (app-side sentence split).

| variant | threads | en RTF | en TTFA ms | short_en ("Hello, how are you?") synth ms | RTF |
|---|---|---|---|---|---|
| fp32 | 2 | 1.127 | 1 355 | 1 530 | 1.136 |
| **fp32** | **4** | **0.632** | **821** | **1 140** | 0.846 |
| fp32 | 8 | 0.992 | 1 361 | 1 740 | 1.291 |
| int8 | 4 | 4.477 | 5 954 | 7 312 | 5.556 |

* The int8 file is **5–7× slower** than fp32 here: the dynamically quantized ops are slow on this CPU's
  onnxruntime build. Do not use it for speed.
* 8 threads is slower than 4 (E-cores). On the 8-core Ryzen target, 8 threads may beat 4; re-check there.
* Short inputs have a large fixed cost: about 1.1 s even for a 1.3 s phrase, so first-chunk latency stays near 1 s.

## Recommendation

English engine for `high_quality` (via `tts/lang_router` `en_impl: kokoro`, voice `af_heart`), threads 4,
fp32. `balanced` and `lowest_latency` use Piper English for its ~7× lower first-audio latency.
