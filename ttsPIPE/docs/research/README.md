# ttsPIPE engine research — summary

Every number here was measured on the **i5-12450H proxy box** (4 P-cores + 4 E-cores, 12 threads,
15.7 GB RAM, CPU only with `CUDA_VISIBLE_DEVICES=""`). The deployment target is an 8-core Ryzen with
14 GB RAM and no GPU. That box has no efficiency cores, so the thread-count sweet spot may move from 4
to 8 there; re-run `scripts/research/run_clean_suite.sh` on it before tuning further.

Method: `scripts/research/bench_*.py` (shared harness `bench_common.py`), 1 warm-up + 5 timed runs per
text (2 for Parler), median reported, on a quiet box (no other jobs), unless a report says otherwise.
Raw JSON: `data/research_results/`. WAVs for listening: `data/research_wavs/<engine>/`. Subjective
quality is marked UNVERIFIED throughout: no listening test was run.

Test texts: Tamil sentence, Tanglish in Tamil script, Tanglish romanized, English sentence,
4-word Tamil, 4-word English (see `bench_common.py`).

## Summary table

RTF = synthesis time ÷ audio duration (lower is faster; < 1 is faster than real time). "TTFA short" =
time to the first audio of the 4-word Tamil text (English for English-only engines), engine alone.

| engine | licence | commercial? | download MB | languages | streaming | CPU RTF (median, best threads) | TTFA short ms | RAM peak MB | verdict |
|---|---|---|---|---|---|---|---|---|---|
| [Fish Audio S2-Pro NF4](fish-s2-pro.md) — **GPU** (RTX 2050) | Fish Audio Research (non-commercial) | **no** (needs Fish licence) | 4 990 | ~50 incl. ta, en; Tanglish in one voice; voice cloning | yes (46 ms frames) | 1.0–1.3 (GPU) | 83 (en) / 660 (ta) | 3 970 VRAM | **default on the GPU box** |
| [Piper Rasa ta (female/male)](piper.md) | voices CC-BY-4.0; runtime GPL-3.0 | yes (attribution; GPL if distributed) | 64 / voice | ta, + English words in Tamil text | per sentence | **0.10** (4 thr) | **204** | 300 | **default Tamil + Tanglish** |
| [Piper en (ljspeech/kristin)](piper.md) | MIT model, public-domain data; GPL runtime | yes | 64 / voice | en | per sentence | 0.11 (4 thr) | 145 | 335 | **default English** |
| [Piper en lessac](piper.md) | Blizzard-2013 data licence | **no** (research only) | 64 | en | per sentence | 0.12 (2 thr) | 153 | 303 | available, not default |
| [Kokoro-82M fp32](kokoro.md) | Apache-2.0; GPL phonemizer | yes | 354 | en (no Tamil) | no (per call) | 0.63 (4 thr) | 1 140 (en) | 625 | `high_quality` English |
| [Kokoro-82M int8](kokoro.md) | Apache-2.0 | yes | 142 | en | no | 4.5 (4 thr) | 7 312 (en) | 435 | rejected (slower than fp32 here) |
| [MMS-TTS tam](mms-tts-tam.md) | **CC-BY-NC-4.0** | **no** | 145 | ta only, drops Latin | no | 0.54 (4 thr) | 1 447 | 620 | A/B shadow only |
| [Indic-TTS FastPitch ta](indic-tts.md) | MIT (repo) | yes | 3 000 | ta only, drops Latin; English input crashes | no | 0.75 (4 thr) | 1 826 | 2 100 | rejected (slow, heavy) |
| [Indic Parler-TTS](indic-parler-tts.md) | Apache-2.0 (gated) | yes | 3 600 | 21 incl. ta + en, keeps Latin | yes | 14.9–16.7 (4 thr) | 7 490 | 4 363 | rejected (not real time) |
| [IndicF5](indicf5.md) | MIT (gated) | yes | 1 400 | Indic incl. ta (voice cloning) | no | 11.7 (NFE 8) – 126 (NFE 32) | 25 849 | 3 325 | rejected (not real time) |
| [eSpeak NG](espeak-ng.md) | GPL-3.0 | yes (GPL) | 13 | ta, en | yes | 0.001 | 2 | 39 | not a user voice; phonemizer of Piper/Kokoro |

Not measured, with the reason:

* **Svara / Sarvam / other 2026 Indic open-weight TTS**: not evaluated in this round.
* **IIT Madras Fastspeech2_HS (Tamil)**: files downloaded (`data/models/fs2hs`, 196 MB), but it needs
  the ESPnet-based repo code. Not wired up in this round.
* **AI4Bharat IndicXlit (romanized → Tamil)**: `ai4bharat-transliteration` 1.1.3 pulls `fairseq==0.12.2`,
  which has no CPython 3.12 wheel (source build only). Not installed. ttsPIPE uses its own rule-based
  transliterator in `g2p/lexicon`; IndicXlit is the upgrade path.

## Tamil / Tanglish: what the measurements say

* Only Piper (via eSpeak `ta`) and Parler keep English words inside Tamil text. MMS and FastPitch
  silently drop every Latin character (tokenizer round-trips in their reports).
* Romanized Tamil is read as English by every engine. Transliterating to Tamil script first (ttsPIPE
  `g2p/lexicon`) is required regardless of engine.
* Routing design and voice matching: [lang_router.md](lang_router.md).

## End-to-end TTFA per profile (full pipeline, real-time pacing)

`scripts/research/profile_ttfa.py`: private server, phrase cache **off**, each sample script streamed
as LLM deltas (3 chars every 15 ms), 3 runs per script, 6 scripts → n = 18 per profile. TTFA is
measured from the first text delta to the first 20 ms frame sent.

| profile | TTS | TTFA p50 | TTFA p95 | TTFA from `end_utterance` p50 | first synth p50 | RTF p50 | underruns |
|---|---|---|---|---|---|---|---|
| `minimal` | tone (test) | 80 ms | 100 ms | 52 ms | 53 ms | 0.07 | 0 |
| `lowest_latency` | Piper Rasa only | **171 ms** | 423 ms | 114 ms | 134 ms | 0.13 | 0 |
| `english_fast` | Piper ljspeech | 160 ms | 342 ms | 94 ms | 124 ms | 0.08 | 0 |
| **`balanced`** | router: Piper Rasa + Piper ljspeech | **241 ms** | **384 ms** | 141 ms | 190 ms | 0.10 | 0 |
| `high_quality` | router: Piper Rasa + Kokoro | 1 172 ms | 2 421 ms | 1 045 ms | 1 131 ms | 0.52 | 0 |

Per script (`balanced`, p50): ta 101 ms · Tanglish script 302 ms · Tanglish romanized 263 ms · en 81 ms ·
numbers/dates 381 ms · long paragraph 161 ms. `high_quality` stays at 80–260 ms when the reply starts
in Tamil but takes 2.1–2.4 s when it starts in English (Kokoro's first 6-word chunk). TTFA "from end"
is lower than from the first delta because synthesis starts before the LLM finishes.

With the phrase cache **on** and a repeated prompt, TTFA is 40–60 ms for every profile (cache hit;
`data/research_results/profile_ttfa_cache_hits.json`).

## Recommendation for `balanced` profile

* **Tamil engine:** `tts/piper` voice `ta_IN-rasa_female-medium` (`ta_IN-rasa_male-medium` for a male
  agent). It is fastest (RTF 0.10) and smallest (64 MB, 300 MB RAM), streams per sentence, is
  commercially usable (CC-BY-4.0 voice), and speaks English words inside Tamil text.
* **English engine:** `tts/piper` voice `en_US-ljspeech-medium` (public-domain data). Same runtime and
  rate (22.05 kHz), RTF 0.11.
* **Router:** yes. `tts/lang_router` with `min_run_words: 3`: one- or two-word English islands stay in the
  Tamil voice, longer English stretches switch to the English voice, `ta-Latn` is transliterated and
  sent to the Tamil voice.
* **First chunk:** 4 words at a clause end (`first_chunk_min_words: 4`, max 10, 400 ms max wait).
  Piper synthesizes a 4-word Tamil phrase (~1.7 s of audio) in ~170–200 ms, which is about one jitter
  buffer of slack. Going down to 3 words saves little and costs prosody.
* **Threads:** 4 (8 was slower on the proxy's E-cores).
* **Expected TTFA p50 on CPU:** see the table above (**241 ms p50, 384 ms p95** on the proxy box).

## `lowest_latency`

Piper, single Tamil voice for everything (no voice switch, no second model), 3-word first chunk,
250 ms max wait, 40 ms jitter buffer, look-ahead 1.

## `high_quality`

Piper Rasa for Tamil, Kokoro-82M (`af_heart`) for English runs: the most natural English we measured.
Kokoro costs ~1 s of first audio when an utterance *starts* in English, and RTF 0.63, so
`high_quality` uses a 120 ms jitter buffer and 6-word first chunks.
