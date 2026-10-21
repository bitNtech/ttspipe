# eSpeak NG (Tamil `ta`, English `en-us`) — verification report

**Verdict: not a voice for users** (formant synthesis, robotic), but it is the **phonemizer inside Piper
and Kokoro**, so its licence matters to every espeak-based engine. Not shipped as a TTS module.

| item | verified value |
|---|---|
| repo | <https://github.com/espeak-ng/espeak-ng> (1.52.0) |
| licence | **GPL-3.0-or-later**. Linking it in-process makes the combined program GPL when *distributed*; running it on your own server is not distribution. A separate `espeak-ng.exe` subprocess keeps it at arm's length. |
| download | 13 MB (`espeakng-loader` wheel ships `espeak-ng.dll` + data; MSI 1.52.0 also fetched) |
| languages | `ta` voice exists and reads Tamil script; `en-us`; Latin words inside a `ta` request are switched to English rules |
| streaming | yes (synth callback every ≤ 200 ms buffer) |
| native rate | 22 050 Hz |
| RAM | RSS 37 MB after load, 39 MB peak |
| load | 0.04–0.09 s |

## Measured (i5-12450H proxy, 1 thread, quiet box, 5 runs median)

| text | audio s | synth ms | RTF | first callback |
|---|---|---|---|---|
| ta | 4.87 | 4.5 | 0.0009 | 0.6 ms |
| tanglish_script | 3.05 | 2.9 | 0.0009 | 0.7 ms |
| tanglish_roman | 3.04 | 2.7 | 0.0009 | 0.7 ms |
| en | 3.95 | 3.1 | 0.0008 | 0.5 ms |
| short_ta | 1.81 | 2.0 | 0.0011 | 0.6 ms |
| short_en | 1.11 | 1.3 | 0.0012 | 0.5 ms |

## Relevance to ttsPIPE

* Piper's Tamil Rasa voices use espeak `ta` phonemes. eSpeak phonemizes English words inside Tamil text
  with English rules (`meeting` → `mˈiːtɪŋ`, `please` → `plˈiːz`), and every resulting IPA symbol exists
  in the Rasa voice's phoneme map (coverage 1.0 on all test texts). That is why one Piper Tamil voice can
  say Tanglish written in Tamil script. Romanized Tamil is phonemized as English
  (`Naalaikku` → `nˈɑːleɪkˌuː`), so ttsPIPE transliterates `ta-Latn` to Tamil script in `g2p` first.
* Could serve as an always-available emergency voice; `tts/tone` already covers the builtin role, so it is not shipped.
