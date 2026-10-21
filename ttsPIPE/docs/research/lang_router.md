# Language router (`tts/lang_router`) — design note

## Problem

Tanglish puts Tamil and English in one sentence, written in Tamil script, Latin script ("romanized
Tamil") or both. No open CPU engine we verified handles all three well:

| engine | Tamil script | English words (Latin) | romanized Tamil |
|---|---|---|---|
| Piper Rasa (ta) | yes | phonemized with English rules; every phoneme is in the voice map (rendering quality UNVERIFIED) | phonemized as **English** → anglicized |
| MMS-TTS tam | yes | **dropped** (vocab has no Latin letters except `a`) | **dropped** |
| Indic-TTS FastPitch ta | yes | **dropped**; pure-English input crashes | **dropped** |
| Kokoro / Piper en | no (Kokoro has no Tamil training; Piper en reads Tamil via eSpeak `ta` with an English voice) | yes | anglicized |

## Design (implemented)

1. `segment/script` labels each word `ta` (Tamil script), `en` or `ta-Latn` (lexicon + suffix heuristics).
2. `g2p/lexicon` transliterates `ta-Latn` → Tamil script (rule-based; IndicXlit is the upgrade path).
3. `tts/lang_router` splits a chunk into language runs: `ta-Latn` maps to `ta_latn_to` (default `ta`,
   using the transliteration). Runs shorter than `min_run_words` merge into a neighbour, preferring the
   chunk's dominant language. **Tamil-script runs only merge into Tamil runs** (an English engine cannot
   read Tamil script). Each run goes to `ta_impl`/`ta_voice` or `en_impl`/`en_voice`.
4. Pieces keep their own sample rate. The runtime resamples each piece and `post/master` evens out
   loudness (plus per-engine `ta_gain_db` / `en_gain_db`).

## Why `min_run_words = 3` in `balanced`

* One- or two-word English islands ("meeting", "please time-க்கு") stay inside the Tamil sentence and
  the Tamil Piper voice says them with eSpeak's English phonemes. That keeps one speaker, one prosody
  contour and no voice switch mid-phrase. Switching voices for a single word is the most audible
  artefact of routing.
* Longer English stretches (≥ 3 words, e.g. "Your order has shipped") go to the English voice, which
  was trained on English audio.
* With MMS or FastPitch as the Tamil engine, set `min_run_words: 1`: every English word must go to the
  English engine, otherwise it is silently dropped.

## Voice matching

* Pair by gender: `ta_IN-rasa_female-medium` ↔ `en_US-ljspeech-medium` (female) and
  `ta_IN-rasa_male-medium` ↔ an English male voice (Kokoro `am_michael`, or Piper `en_US-ryan` / `joe`).
* Both Piper voices are 22.05 kHz, peak-normalized per sentence (`normalize_audio`), so their levels
  match within a few dB. `post/master` normalizes to −18 LUFS (approximate) on top of that.
* Speaking rate on the test texts: Rasa female reads the 8-word Tamil sentence in 4.6 s (1.7 words/s,
  but Tamil words are long and agglutinative); ljspeech reads the 11-word English sentence in 4.0 s
  (2.7 words/s). No rate correction is applied; `voice.speed` scales both. A syllable-rate match
  is an open item for the listening test.

## Measured cost

Routing adds no model of its own. A mixed chunk is two engine calls in sequence on the TTS thread:
first audio = the first run's synthesis time. With Piper on both sides that is ≈ 0.1 × the first
run's audio duration (see `piper.md`).
