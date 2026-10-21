# ttsPIPE architecture — binding contract

Status: **binding**. Backend (`backend/`) and frontend (`frontend/`) both follow this document.
`frontend/src/types/contract.ts` mirrors §6–§10 field by field. Any deviation found later is fixed
here first, then in code. Section numbers are stable; add new items at the end of a section.

Contract version: `1.1` (reported by `GET /api/health` as `contract`).

---

## 1. Goals

ttsPIPE is the post-LLM half of a voice agent. prePIPE (caller audio → … → `turn.end`) feeds an LLM;
the LLM's streamed reply enters ttsPIPE as text deltas and leaves as paced PCM audio.

1. **Lowest time-to-first-audio (TTFA)** on a CPU-only box (8-core, 14 GB RAM, no GPU).
   Headline metric: TTFA = first text delta → first audio frame sent. Also tracked from
   `end_utterance` → first audio frame sent. p50/p95 per profile.
2. **Tamil, Tanglish and English.** Tanglish = Tamil and English mixed in one sentence, written in
   Tamil script, Latin script ("romanized Tamil") or both.
3. **Full duplex.** Duck, stop (with an exact `heard_text`) and resume at any time, driven by the
   client or by prePIPE's barge-in events.
4. **Replaceable stages.** Every stage is a plugin chosen at runtime and hot-swapped without dropping
   sessions. The server starts and serves `/api/modules` with no ML packages installed.
5. **Observable.** Every stage call, chunk, state change and frame level is visible on the dashboard
   in real time, and every number on the dashboard comes from an event or endpoint defined here.

Non-goals: voice cloning UI, multi-speaker dialogue, GPU scheduling, SSML beyond §4.1 "SSML-lite".

---

## 2. Layout

```
ttsPIPE/
  run.sh                         setup | models | dev | test | stop | doctor
  config/pipeline.yaml           committed defaults (§11)
  config/profiles/*.yaml         builtin profiles, read-only (§13)
  config/lexicon.yaml            pronunciation overrides + Tanglish word list (§12.4)
  config/scripts.yaml            sample texts served by GET /api/scripts (§9.11)
  data/                          git-ignored runtime data (TTSPIPE_DATA_DIR)
    pipeline.override.yaml       runtime config edits, deep-merged over config/pipeline.yaml
    profiles/*.yaml              user profiles
    layouts/*.json               dashboard layouts
    renders/<session>/<utt>.wav  played audio per utterance (+ .json metadata)
    clips/                       pre-recorded phrase WAVs for tts/clips (+ index.yaml)
    models/                      downloaded model files, one folder per <kind>/<impl>
    ratings.jsonl                A/B ratings (§9.13)
    logs/ttspipe.log             rotating server log
  docs/ARCHITECTURE.md           this file
  docs/research/*.md             model verification reports with measured numbers
  backend/
    requirements.txt             core deps only (fastapi, uvicorn, numpy, pyyaml, psutil, websockets)
    requirements/ml-*.txt        one file per ML family
    app/main.py                  FastAPI app factory, lifespan, static mount
    app/settings.py              TTSPIPE_* environment variables (§2.1)
    app/core/                    audio events base registry config manager session runtime playback
                                 metrics logs textutil profiles (+layouts, renders) graph cache bridge lexicon
    app/modules/<kind>/<impl>.py one implementation per file, @register, auto-discovered
    app/api/rest.py, ws.py
    tests/
  frontend/                      Vite + React 19 + TS; dev server :5174 proxies /api, /metrics, /ws
  scripts/download_models.py, doctor.py, smoke.py, research/
```

### 2.1 Settings (environment)

| var | default | meaning |
|---|---|---|
| `TTSPIPE_HOST` | `127.0.0.1` | bind address |
| `TTSPIPE_PORT` | `8200` | HTTP/WS port |
| `TTSPIPE_DATA_DIR` | `<repo>/data` | runtime data root |
| `TTSPIPE_CONFIG` | `<repo>/config/pipeline.yaml` | defaults file |
| `TTSPIPE_PROFILES_DIR` | `<repo>/config/profiles` | builtin profiles |
| `TTSPIPE_LEXICON` | `<repo>/config/lexicon.yaml` | lexicon |
| `TTSPIPE_LOG_LEVEL` | `INFO` | log level |
| `TTSPIPE_THREADS` | `0` (= physical cores) | intra-op threads handed to ML engines |
| `TTSPIPE_STATIC_DIR` | `<repo>/frontend/dist` | built UI served at `/` if present |
| `TTSPIPE_MODELS_DIR` | `<data>/models` | model files root (shared between data dirs, e.g. by smoke.py) |
| `TTSPIPE_FISH_DIR` | `<repo>/..` if it contains `app/engine/fish.py` | Fish S2-Pro server repo (its own `.venv`, `checkpoints/s2-pro-nf4`) that ttsPIPE starts on demand |
| `TTSPIPE_FISH_URL` | `http://127.0.0.1:8080` | Fish S2-Pro server; `tts/fish_s2` streams from `ws://…/v1/tts/live` |
| `HF_TOKEN` | — | used only for gated Hugging Face downloads |

---

## 3. Audio model

* Internal sample format: `float32` mono, nominal range [-1, 1].
* Output wire format: **PCM16 little-endian mono**. Rate `audio.out_rate` ∈ {16000, 22050, 24000,
  48000}, default **24000**. Rate changes apply to sessions opened after the change.
* **Frame** = `audio.frame_ms` = 20 ms → 320 / 441 / 480 / 960 samples. Every binary WS message
  carries exactly one frame.
* Each engine produces audio at its native rate. The runtime resamples with a **stateful
  windowed-sinc resampler** (`core/audio.py: StreamResampler`): Kaiser-windowed sinc, 16 taps per
  side at the lower rate, so its delay is < 1 ms; it keeps the input tail between calls and is
  frame-exact: after `flush()` the total output equals `round(total_in * out_rate / in_rate)`.
  Nothing in the audio path may buffer more than one frame beyond the jitter buffer (§5.4).
* **Stream clock `t_out_ms`**: per session, `tick_index × frame_ms`, where the pacer advances one tick
  per frame period from session start, whether or not a frame is sent on that tick. A frame sent on
  tick *n* starts at `t_out_ms = n × frame_ms`. Because frames are sent strictly in real time (one per
  tick), `t_out_ms` is the playback position of audio actually sent to the client. Every session event
  carries the `t_out_ms` of the moment it was emitted. Global events carry `null`.
* Client playback: clients play each frame on receipt behind a small scheduling cushion
  (`client_buffer_ms`, recommended 40–80 ms). The server does not send ahead of real time.

---

## 4. Stages

Pipeline order and the methods each kind implements. A **stage** is a slot in the pipeline; a
**module** is an implementation (`<kind>/<impl>`) loaded into a slot.

| # | kind | granularity | method(s) |
|---|---|---|---|
| 1 | `ingest` | LLM token stream | `feed(state, delta) -> list[TextSpan]`, `flush(state) -> list[TextSpan]` |
| 2 | `normalize` | per text span | `normalize(text, *, language) -> NormResult` |
| 3 | `segment` | per normalized span | `split(text) -> list[LangSegment]` |
| 4 | `chunker` | streaming | `push(state, segments, policy, now_ms) -> list[Chunk]`, `flush(state, policy) -> list[Chunk]` |
| 5 | `g2p` | per chunk | `apply(chunk) -> Chunk` |
| 6 | `prosody` | per chunk | `plan(chunk, ctx) -> ProsodyPlan` |
| 7 | `tts` | per chunk, streaming if able | `synthesize(state, chunk, plan, voice) -> Iterator[AudioPiece]` |
| 8 | `post` | per output frame | `process(state, frame) -> Frame \| None` |

### 4.1 Stage semantics

**ingest** receives raw LLM deltas (any split: mid-word, mid-markdown). It must:
* emit text only at word boundaries (hold the trailing partial word until whitespace or `flush`);
* strip markdown (`**`, `*`, `_`, `#` headings, `>` quotes, list bullets, tables, inline/fenced code —
  fenced code blocks are dropped entirely; inline code keeps its text), emoji, URLs (replaced by
  nothing, or by the link text for `[text](url)`), HTML tags other than SSML-lite;
* parse **SSML-lite**: `<break>` (default 300 ms), `<break time="500ms"/>`, `<break time="1s"/>`,
  `<emphasis>…</emphasis>`, `<lang xml:lang="en|ta">…</lang>`, `<prosody rate="0.9" pitch="+2st">…</prosody>`.
  Breaks become `TextSpan(kind="break")`; other tags set span attributes.
* `flush` is called at `end_utterance` and returns the held remainder.

**normalize** rewrites a span into speakable words: cardinals, decimals, ordinals (`1st`, `2வது`),
percent, currency (`₹`, `Rs.`, `INR`, `$`), dates (`DD/MM/YYYY`, `YYYY-MM-DD`, `12 Mar 2024`),
times (`10:30`, `10:30 AM`, `10.30 மணி`), phone numbers (digit-wise, grouped), units (km, kg, g, ml,
l, cm, mm, m, °C, %, GB, MB), abbreviations (`Dr.`, `Mr.`, `etc.`, `e.g.`). `language` is
`"ta"`, `"en"` or `"auto"`; with `auto` each token is worded in the script of its neighbours (Tamil
script context → Tamil words, Latin context → English words). Normalization never reorders words. It
returns a word map (§6.3) so playback position can be mapped back to the original text.

**segment** partitions normalized text into language runs. It **must not change text**: the
concatenation of `segment.text` with single spaces equals the whitespace-normalized input, and run
boundaries fall on word boundaries. Languages: `ta` (Tamil script), `en` (English, Latin script),
`ta-Latn` (romanized Tamil, Latin script). Digits and punctuation attach to the preceding word.

**chunker** decides when to send text to TTS. Policy (§11 `chunker`) comes from config and is passed
in. Rules for `chunker/punct`:
* **First chunk** of an utterance: flush as soon as it has ≥ `first_chunk_min_words` words and the
  last word ends a clause (`, ; : — .` `! ? । …` or newline), or it reaches `first_chunk_max_words`,
  or `first_chunk_max_wait_ms` has passed since the utterance's first text and it has ≥ 1 word.
  Reason `first`.
* **Later chunks**: flush at sentence end (`. ! ? । …` newline) → `punct`; at clause punctuation once
  ≥ `min_chunk_chars` → `punct`; at a word boundary once ≥ `max_chunk_chars` → `max_len`.
* `TextSpan(kind="break")` flushes the pending text (reason `punct`) and sets `pause_after_ms`.
* `flush` at `end_utterance` → `end`.
* If the active TTS module is non-streaming, the runtime passes
  `max_chunk_chars = min(max_chunk_chars, nonstreaming_max_chunk_chars)`.
* The runtime calls `push(state, [], policy, now_ms)` on a 20 ms timer while text is pending, so
  time-based flushes do not need new text.

**g2p** applies `config/lexicon.yaml` overrides (whole-word, case-insensitive for Latin), optionally
transliterates `ta-Latn` runs to Tamil script, and may fill `chunk.phonemes` for engines that accept
phonemes. It writes `chunk.tts_text` and `chunk.segments[*].tts_text`; it never edits `chunk.words`.

**prosody** returns a `ProsodyPlan`. Effective engine speed = `voice.speed × plan.speed`; effective
pitch = `voice.pitch + plan.pitch_st` semitones (engines without pitch control ignore it and set
`tags: ["no_pitch"]`).

**tts** yields `AudioPiece`s for one chunk. Streaming engines yield as audio becomes available;
non-streaming engines yield once. The runtime checks the cancel token between pieces. Optional word
timings (§6.6) enable exact `heard_text`.

**post** processes each 20 ms output frame before duck/fade gain and PCM16 encoding. It may drop a
frame (return `None`) only if `frame.trimmable` is true (leading/trailing silence of a chunk, never a
planned pause or a break).

### 4.2 Module interface (`core/base.py`)

```python
@dataclass
class ModuleInfo:
    id: str                 # "<kind>/<impl>", e.g. "tts/tone"
    kind: str               # one of the 8 kinds or "tts" for routers
    name: str               # human name
    description: str
    model_id: str | None    # upstream repo id, e.g. "facebook/mms-tts-tam"
    size_mb: float          # download size (0 for builtins)
    ram_mb: float           # measured resident size when loaded (estimate, §9.5)
    license: str            # SPDX id or exact licence name; "builtin" for ours
    streaming: bool         # tts: yields incremental audio; others: false
    requires: list[str]     # import names checked with importlib.util.find_spec
    params_schema: dict     # §4.3
    default_params: dict    # derived from params_schema defaults
    tags: list[str]         # e.g. ["builtin"], ["no_pitch"], ["router"], ["experimental"]
    languages: list[str]    # ["ta","en","ta-Latn"] subset; [] = language-agnostic
    voices: list[VoiceInfo] # tts only
    builtin: bool           # true = no ML deps, always available
    native_rate: int | None # tts only
    family: str             # ML dependency family → requirements/ml-<family>.txt ("" = core only)

class BaseModule:
    info: ClassVar[ModuleInfo]
    def __init__(self, params: dict): ...          # cheap; NO heavy imports, NO I/O
    @classmethod
    def availability(cls) -> tuple[bool, str|None] # find_spec on info.requires; never imports
    def model_files(self) -> list[ModelFile]       # files load() needs (may be [])
    def load(self) -> None                         # heavy imports + model load happen HERE ONLY
    def unload(self) -> None
    def new_state(self) -> Any                     # per-session state; None if stateless
    def close_state(self, state) -> None
    def update_params(self, params: dict) -> bool  # True = applied live; False = needs reload
```

Rules:
* Module files must import with only core deps installed. Heavy libraries are imported **inside
  `load()`** (or functions it calls).
* **Threading model (binding):** each stage slot owns one single-thread executor shared by all sessions
  (§5.2). All methods of a loaded module (`load`, stage methods, `new_state`, `close_state`, `unload`)
  run on that one thread, so modules need no locks and need no thread-local storage. **Per-session
  module state lives in the object returned by `new_state()`**, passed back into every stage call.
  Stateless kinds (`normalize`, `segment`, `g2p`, `prosody`) take no state argument.
* Failures raise exceptions. The runtime counts consecutive failures per slot; 3 in a row marks the
  module `error` and the slot falls back to its kind's builtin until the module is reloaded (§5.6).

### 4.3 `params_schema`

A JSON-schema subset, one entry per param:

```json
{ "speed": {"type": "number", "default": 1.0, "min": 0.5, "max": 2.0, "description": "…"},
  "voice": {"type": "string", "default": "f1", "enum": ["f1", "m1"]},
  "trim":  {"type": "boolean", "default": true},
  "steps": {"type": "integer", "default": 16, "min": 4, "max": 64} }
```

`type` ∈ `number | integer | string | boolean | array`. Optional keys: `default`, `min`, `max`, `enum`,
`description`, `unit` (display only, e.g. `"ms"`), `items` (for arrays: `{"type": …}`),
`advanced` (bool; UI hides behind "advanced"). Validation: unknown keys → error; wrong type → error
(int accepted for number); out of `[min,max]` → error; not in `enum` → error. Error text format:
`stages.<kind>.params.<name>: <reason>`; REST returns it as HTTP 422 `{"detail": "<text>"}`.

---

## 5. Session runtime

### 5.1 Objects

* **Manager** (`core/manager.py`): one per process. Owns config, the 8 stage slots (+ shadow slot),
  module loading, hot swap, the event bus, metrics, phrase cache and sessions.
* **Session** (`core/session.py`): one per `/ws/tts` connection (or offline render). Owns per-session
  module states, the utterance list, the playback manager and its pacer thread.
* **Utterance** = one LLM reply, id `utterance_id` (client-supplied or `u<n>`). Contains chunks.
* **Chunk** has `chunk_id` = `"<utterance_id>.<index>"`, index from 0.

### 5.2 Threads and flow

```
client text ─► [intake task] ingest → normalize → segment → chunker → g2p → prosody ─► chunk queue
                                                                                         │
                 [synthesis task]  (look-ahead gate) ─► tts slot executor ─► audio queue │
                 [shadow task]     optional, same chunks ─► shadow executor (not played) │
                 [pacer thread]  every 20 ms: frame ─► post executor-free call ─► gain ─► WS
```

* Each stage slot has **one single-thread executor** (`ThreadPoolExecutor(max_workers=1)`) shared by
  all sessions. Stages therefore never block each other, a slow `g2p` never delays `normalize`, and
  one module instance is only touched by one thread. Exception: `load()` of a new module runs on a
  loader thread *before* the module is published to the slot, so nothing touches it concurrently.
  *Ceiling:* concurrent sessions serialize on the TTS thread. On an 8-core CPU box the engine uses
  all cores itself, so this is the throughput-optimal choice; more sessions need more boxes.
* The **intake task** (asyncio) runs stages 1–6 per text event and pushes chunks.
* The **synthesis task** pulls chunks in order and submits `synthesize` to the TTS executor when the
  look-ahead gate opens: chunk *i* may start when `i ≤ max(playing_index, 0) + lookahead_chunks`
  (default 2). Synthesis of chunk N+1 therefore overlaps playback of chunk N.
* The **pacer** is a dedicated thread per session. It ticks every `frame_ms` against an absolute
  deadline (`deadline += frame_ms`; it catches up without skipping if late). On each tick it takes
  one frame from the playback buffer, calls `post.process` **inline on the pacer thread** (post is
  per-frame and cheap; post modules must not block — §4.1), applies duck/fade gain, encodes PCM16 and
  hands it to the WS writer. It emits `audio.level` every `level_every_ticks` ticks (default 2 → 25 Hz;
  ≈30 Hz requirement met within the 20 ms frame grid) and `playback.progress` on word changes.
* Post state: the post module's `new_state()` is per session and only touched by that session's
  pacer thread. This is the one exception to the "one executor per slot" rule; post modules must be
  re-entrant across sessions (no shared mutable state outside `state`).

### 5.3 Timing record per chunk

Each chunk records (ms on the server monotonic clock, relative to utterance start = first text):
`t_text` (first char received), `t_norm`, `t_flush`, `t_synth_start`, `t_first_audio`, `t_last_audio`,
`t_play_start`, `t_played` — plus `t_out_ms` stream-clock values for `play_start` and `played`.

### 5.4 Jitter buffer, underrun, silence

* The pacer starts playing a chunk when it has ≥ `jitter_ms` (default 60) of that chunk buffered, or
  the chunk's synthesis is complete. For later chunks of the same utterance the same rule applies.
* If the current chunk is exhausted and the next chunk is not ready: the pacer emits `underrun`
  (once per gap) and, if `fill_underrun_silence` is true, sends silence frames for up to
  `max_silence_gap_ms`; after that it stops sending until audio arrives (state stays `speaking`).
* `prosody.pause_after_ms` is inserted as non-trimmable silence frames after a chunk.

### 5.5 Playback state machine (`core/playback.py: PlaybackManager`)

States: `idle`, `buffering`, `synthesizing`, `speaking`, `ducked`, `interrupted`, `done`.

| from | event | to | notes |
|---|---|---|---|
| idle | first text of an utterance | buffering | |
| buffering | first chunk enters synthesis | synthesizing | |
| synthesizing | first frame sent | speaking (or ducked if duck active) | TTFA measured here |
| speaking | `duck` | ducked | gain → `-duck_db` over `duck_ramp_ms`; synthesis continues |
| ducked | `resume` | speaking | gain → 0 dB over `duck_ramp_ms` |
| ducked | `duck_release_ms` passes with no stop/resume | speaking | auto-release; reason `duck_timeout`; 0 = never |
| buffering/synthesizing/speaking/ducked | `stop` | interrupted | fade over `stop_fade_ms`; cancel queued text + synthesis |
| interrupted | (immediately after fade) | idle | |
| speaking/ducked | last frame of last chunk of an ended utterance, no queued utterance | done | |
| done | (immediately) | idle | |

* `duck` while idle/buffering/synthesizing sets the gain target; playback then starts in `ducked`.
* Utterances play back-to-back in order. A new utterance arriving while one is speaking queues
  behind it; the state stays `speaking`.
* `stop` cancels **everything** queued in the session (current and queued utterances).
* After `stop`, further `text` deltas for the interrupted `utterance_id` are dropped until
  `end_utterance` or a different `utterance_id` arrives (the LLM usually keeps streaming briefly).
  Clients SHOULD send `utterance_id`.
* `reset` = `stop` + clear utterance history and session counters.

### 5.6 Hot swap, errors, fallback

* `PUT /api/pipeline/stages/{kind}` validates, then loads the new module **in the background** on a
  loader thread. When loaded, the slot swaps atomically (new calls go to the new module; per-session
  states are recreated lazily on next use). The old module's `unload` is submitted to the old slot
  executor, so it runs only **after its queued jobs drain**. In-flight synthesis completes on the
  old module.
* Param-only changes call `update_params`; if it returns False, the module reloads as above.
* If a configured module is unavailable (missing deps or files) or fails to load, the slot uses its
  kind's **builtin** (§12.1) and reports `status: ready`, `fallback: true`, non-null `error`.
  `/api/health` then reports `degraded`.
* 3 consecutive call failures → module `status: error`; the slot uses the builtin until
  `POST /api/modules/{kind}/{impl}/load` or another PUT. Each failure also emits `error`.
* Disabled stage (`enabled: false`): identity for text stages, passthrough for `post`. `ingest`,
  `chunker` and `tts` cannot be disabled (validation error).

### 5.7 heard_text

When stopped, `heard_text` is the original (post-ingest, pre-normalize) text of every **word that was
at least 50 % played** before the cut, where the cut position = stream position at `stop` +
`stop_fade_ms` (the fade is audible). Word positions come from engine word timings when present
(`AudioPiece.word_timings`), else they are estimated by distributing the chunk's audio duration over
its spoken words in proportion to character count (excluding leading/trailing trimmed silence).
Spoken words map to original words through the normalization word map (§6.3): a run of spoken words
produced from one original token counts as heard when its last spoken word is heard.
`unplayed_text` = the rest of the utterance's received text (including text not yet chunked).

### 5.8 Phrase cache (`core/cache.py`)

Key = sha1 of `(tts impl, effective tts params, voice, chunk.tts_text, plan.speed, plan.pitch_st)`.
Value = complete native-rate PCM + word timings + rate. Only fully synthesized chunks are cached. LRU
bounded by `cache.max_mb` (default 64). On hit, the chunk's audio is available immediately
(`synth.first_audio.latency_ms` ≈ 0, `cached: true`). Hit rate is reported in `metrics.cache`.

### 5.9 prePIPE bridge (`core/bridge.py`, optional)

If `bridge.enabled` and `bridge.prepipe_url` are set, the server subscribes to
`<prepipe_url>/ws/events?session_id=*` and maps (configurable in `bridge.map`):

| prePIPE event | ttsPIPE action |
|---|---|
| `barge_in.candidate` | duck |
| `barge_in` | stop |
| `backchannel` | resume |

Session mapping (`bridge.session_map`): `same_id` (prePIPE `session_id` == ttsPIPE `session_id`) or
`latest` (the most recently started ttsPIPE session). Agent speaking state is reported to
`POST <prepipe_url>/api/sessions/{id}/agent` with `{"speaking": bool, "utterance_id", "t_out_ms"}`
on every transition into/out of `speaking`. If `bridge.send_reference_audio`, played frames are
forwarded to `<prepipe_url><bridge.reference_ws_path>?session_id=<id>` as binary
`[kind:uint8=2][PCM16 LE]` (prePIPE agent reference audio for echo-aware VAD).
*Assumption (unverified, no prePIPE on this box):* prePIPE's event envelope matches §7.1 and its audio
WS accepts kind=2 frames at `/ws/audio`. Bridge status is in `GET /api/state` → `bridge` and the
`bridge.status` event.

---

## 6. Data types (`core/base.py`)

All are dataclasses; JSON forms use the same field names. Times are ms.

### 6.1 TextSpan
```
text: str            # ingested text (markdown stripped), ends at a word boundary
kind: "text" | "break"
break_ms: int        # for kind=break
emphasis: bool
lang_hint: str|None  # from <lang>
rate: float|None     # from <prosody rate>
pitch_st: float|None # from <prosody pitch>
```
### 6.2 Word
```
text: str     # spoken (normalized) word
raw: str      # original token(s) this word came from; "" for the 2nd..nth word of a group
group: int    # index of the original token group within the utterance
lang: str     # filled by the runtime after segmentation
```
### 6.3 NormResult
```
text: str                           # normalized text
changed: bool
edits: list[{before, after}]        # human-readable replacements, in order
words: list[Word]                   # word map; " ".join(w.text) == text (whitespace-normalized)
```
### 6.4 LangSegment
```
text: str
lang: "ta" | "en" | "ta-Latn"
words: list[Word]   # attached by the runtime (segmenters return text+lang only)
tts_text: str       # set by g2p; defaults to text
attrs: dict         # SSML-lite: emphasis, rate, pitch_st; {"break_ms": n} with empty text = a break
```
### 6.5 Chunk
```
chunk_id: str; utterance_id: str; index: int
text: str            # original text of its words (" ".join of raw, non-empty)
norm_text: str       # spoken text before g2p
tts_text: str        # text given to the engine (after g2p)
segments: list[LangSegment]
lang: str            # dominant language by word count
words: list[Word]
reason: "first" | "punct" | "max_len" | "end"
is_first: bool; is_last: bool
pause_after_ms: int
phonemes: str|None
emphasis: bool; rate: float|None; pitch_st: float|None   # from SSML-lite
```
### 6.6 AudioPiece / WordTiming
```
AudioPiece: pcm: np.ndarray[float32] (mono); sample_rate: int;
            word_timings: list[WordTiming]|None; is_last: bool
WordTiming: word: str; start_ms: float; end_ms: float   # relative to the chunk's first sample
```
### 6.7 ProsodyPlan
```
speed: float (1.0); pitch_st: float (0.0); pause_after_ms: int; emphasis: list[int] (word indexes);
style: str ("neutral" | "question" | "exclaim" | "calm")
```
### 6.8 Voice / VoiceInfo
```
Voice:     impl: str; voice_id: str; speed: float (1.0); pitch: float (0.0 semitones)
VoiceInfo: id; name; language (ta|en|multi); gender (female|male|neutral); sample (url|null)
```
### 6.9 Frame
```
pcm: np.ndarray[float32] (exactly frame samples); utterance_id: str|None; chunk_id: str|None
first_of_chunk: bool; last_of_chunk: bool; trimmable: bool; kind: "speech" | "pause" | "silence"
rate: int (output rate); fade_ms: int (= playback.crossfade_ms; post fades in/out at chunk joins)
```
### 6.10 ModelFile
```
name: str; url: str; path: str (relative to data/models); size_mb: float; sha256: str|None
present: bool (computed)
```

---

## 7. Events

### 7.1 Envelope
```json
{"type": "chunk.flushed", "session_id": "s1a2b3", "ts": 1760000000123, "t_out_ms": 1840, "data": {}}
```
* `ts`: server wall clock, integer Unix epoch **milliseconds**.
* `t_out_ms`: session stream clock (§3) at emission; `null` for global events (`metrics`, `log`,
  `stage.status`, `model.download`, `bridge.status`) and `session_id` is `null` for them too.

### 7.2 Types and `data`

| type | data | rate |
|---|---|---|
| `session.started` | `{label, voice: Voice, sample_rate, frame_ms, format: "pcm16le", profile}` | once |
| `session.ended` | `{reason, utterances, duration_ms}` | once |
| `text.delta` | `{utterance_id, delta, chars_total}` | per delta |
| `text.normalized` | `{utterance_id, before, after, changed, edits, latency_ms}` | per span |
| `segment` | `{utterance_id, spans: [{text, lang}], latency_ms}` | per span |
| `chunk.flushed` | `{utterance_id, chunk_id, index, text, norm_text, tts_text, lang, words, reason, segments:[{text,lang}], pause_after_ms, since_text_ms}` | per chunk |
| `synth.start` | `{utterance_id, chunk_id, impl, queue_ms, cached}` | per chunk |
| `synth.first_audio` | `{utterance_id, chunk_id, latency_ms, cached}` (latency = synth start → first piece) | per chunk |
| `synth.done` | `{utterance_id, chunk_id, audio_ms, synth_ms, rtf, cached, word_timings: bool}` | per chunk |
| `audio.level` | `{rms, peak, rms_db, gain_db, wave: [min,max, …]}` (wave = 8 min/max pairs over the last 2 frames) | ~25–30 Hz |
| `playback.state` | `{state, prev, reason, utterance_id}` | per change |
| `playback.chunk` | `{utterance_id, chunk_id, event: "start"\|"end", t_out_ms}` | per chunk |
| `playback.progress` | `{utterance_id, chunk_id, word_index, heard_text}` | per word |
| `underrun` | `{utterance_id, waiting_chunk_id, gap_ms}` | per gap |
| `duck` | `{action: "duck"\|"resume"\|"release", gain_db, source: "client"\|"api"\|"bridge"\|"timeout"}` | per action |
| `utterance.started` | `{utterance_id}` | per utterance |
| `utterance.interrupted` | `{utterance_id, heard_text, unplayed_text, at_ms, source}` (`at_ms` = stream position of the cut) | per stop |
| `utterance.end` | `{utterance_id, text, normalized, chunks: [ChunkSummary], interrupted, profile, latency: {ttfa_ms, ttfa_end_ms, text_to_first_chunk_ms, first_synth_ms, total_synth_ms, audio_ms, rtf}, render_id}` | per utterance |
| `shadow` | `{utterance_id, chunk_id, impl, first_audio_ms, synth_ms, audio_ms, rtf, error, render_id}` | per chunk |
| `stage.status` | `StageStatus` (§9.3) | per change |
| `stage.activity` | `{kind, impl, utterance_id, chunk_id, ms, ok}` | per call |
| `metrics` | `Metrics` (§9.14) | 1 Hz |
| `log` | `{level, logger, msg, session_id}` | per record (≥ INFO) |
| `error` | `{where, message, kind, impl, fatal}` | per error |
| `model.download` | `{kind, impl, file, done_mb, total_mb, status: "running"\|"done"\|"error", error}` | ~2 Hz |
| `bridge.status` | `{enabled, connected, url, error, last_event}` | per change |
| `pong` | `{t, server_ts}` | per ping (WS tts only) |

`ChunkSummary` = `{chunk_id, index, text, norm_text, lang, reason, words, latency: {flush_ms,
synth_queue_ms, first_audio_ms, synth_ms, audio_ms, rtf}, played: "full"|"partial"|"none", cached}`.

**Latency definitions** (per utterance; `t0` = first text received):
* `ttfa_ms` = first frame sent − `t0`. `ttfa_end_ms` = first frame sent − `end_utterance` received
  (negative when audio started before the LLM finished; `null` if not ended).
* `text_to_first_chunk_ms` = first `chunk.flushed` − `t0`.
* `first_synth_ms` = first chunk's synth start → first audio piece.
* `total_synth_ms` = Σ chunk synth time; `rtf` = total_synth_ms / audio_ms.

### 7.3 Bus and back-pressure (`core/events.py`)

Each subscriber has a bounded queue (`events.queue_size`, default 2000). When full, the bus drops the
**oldest non-critical** event. **Critical** types are never dropped: `utterance.*`, `playback.state`,
`chunk.flushed`, `synth.first_audio`, `stage.status`, `error`, `session.*`. High-rate, droppable:
`audio.level`, `playback.progress`, `stage.activity`, `text.delta`, `log`, `metrics`. Drops are
counted in `metrics.events_dropped`.

---

## 8. WebSocket protocol

### 8.1 `/ws/tts?session_id=&label=&voice=`

One session per connection. `session_id` optional (server generates `s` + 6 hex). Reconnecting with
an id already open closes the old connection with code 4409. `voice` = `<impl>:<voice_id>` or
`<voice_id>` (applies to the configured TTS impl); default from config.

Client → server (text frames, JSON):

| message | effect |
|---|---|
| `{"type":"text","delta":"…","utterance_id"?:str}` | append LLM text; opens an utterance if none open |
| `{"type":"end_utterance","utterance_id"?:str}` | no more text for the open utterance |
| `{"type":"speak","text":"…","utterance_id"?:str}` | text + end_utterance in one message |
| `{"type":"duck"}` / `{"type":"stop"}` / `{"type":"resume"}` | §5.5 |
| `{"type":"ping","t":number}` | server replies `pong` with the same `t` |
| `{"type":"reset"}` | §5.5 |
| `{"type":"voice","voice":"impl:id","speed"?:n,"pitch"?:n}` | session voice change, next chunk |

A `text` with an `utterance_id` different from the open one implicitly ends the open one. Unknown
types → `error` event, connection stays open.

Server → client:
* **binary**: `[kind:uint8 = 1][PCM16 LE mono frame]` — one 20 ms frame per message.
* **text**: event envelopes (§7) for this session only, including `session.started` first.

### 8.2 `/ws/events?session_id=<id|*>`

Dashboard monitor. `*` (default) = all sessions. Delivers session events plus all global events
(`metrics`, `log`, `stage.status`, `model.download`, `bridge.status`). Read-only; client messages are
ignored except `{"type":"ping","t"}` → `pong`.

---

## 9. REST API (`/api`)

JSON in/out. Errors: `{"detail": str}` with 400/404/409/422. All list endpoints return arrays.

### 9.1 `GET /api/health`
`{status: "ok"|"degraded"|"error", contract: "1.0", version, uptime_s, reasons: [str],
stages: {kind: status}}`. `degraded` if any slot is in fallback or error; `error` if TTS has no
working module (never with the builtin present).

### 9.2 `GET /api/modules`
`[ModuleEntry]`, `ModuleEntry = ModuleInfo + {available: bool, reason: str|null, loaded: bool,
status: StageStatusValue|null, models_present: bool}`.

### 9.3 `GET /api/pipeline`
```
{ config: Config,                          # effective merged config (§11)
  stages: {kind: StageStatus},             # 8 kinds
  shadow: StageStatus | null,
  graph: {nodes: [GraphNode], edges: [GraphEdge], positions: {node_id: {x,y}}},
  active_profile: str|null, dirty: bool }
StageStatus = { kind, impl, configured_impl, enabled, status: "ready"|"loading"|"unloaded"|"error"|"disabled",
  error: str|null, fallback: bool, busy: bool, params: {…effective}, load_ms, count, errors,
  consecutive_errors, last_ms, p50_ms, p95_ms }
GraphNode = { id, kind: "source"|"sink"|<stage kind>|"shadow", label, impl: str|null }
GraphEdge = { from, to, dashed: bool }
```
Node ids: `text_in, ingest, normalize, segment, chunker, g2p, prosody, tts, shadow_tts, post,
audio_out`. Edges: the chain, plus `prosody → shadow_tts` (dashed) when a shadow is configured.

### 9.4 Stage and runtime config writes
* `PUT /api/pipeline/stages/{kind}` body `{impl?, params?, enabled?}` → 200 `StageStatus` (status may
  be `loading`). 422 on validation error; 404 unknown kind/impl.
* `PUT /api/pipeline/playback` body ⊆ `Config.playback` ∪ `{out_rate}` → 200 `Config.playback`.
* `PUT /api/pipeline/chunker` body ⊆ `Config.chunker` → 200.
* `PUT /api/pipeline/voice` body ⊆ `Config.voice` → 200.
* `PUT /api/pipeline/shadow` body `{impl: str|null, params?, voice_id?}` → 200 (null removes).
* `PUT /api/pipeline/positions` body `{node_id: {x, y}}` → graph editor node positions.
* `PUT /api/pipeline` body `Config` (full) → atomic validate + apply; 422 on any error.
* `POST /api/pipeline/validate` body `Config` (full or partial) →
  `{ok, errors: [str], warnings: [str], estimate: {download_mb, ram_mb}}`.
* `POST /api/pipeline/reset-config` → deletes the override; returns the new `/api/pipeline` body.

All writes persist to `data/pipeline.override.yaml` and mark `dirty` if the result differs from the
active profile.

### 9.5 Modules and models
* `POST /api/modules/{kind}/{impl}/load` → loads it into its slot (makes it the configured impl).
* `POST /api/modules/{kind}/{impl}/unload` → unload; the slot falls back to the builtin.
* `GET /api/models` → `[{id, kind, impl, files: [ModelFile], present, size_mb, downloading, progress}]`.
* `POST /api/models/{kind}/{impl}/download` → 202; progress via `model.download`.
`estimate.ram_mb` sums `ModuleInfo.ram_mb` of configured modules (+ shadow);
`estimate.download_mb` sums sizes of missing model files.

### 9.6 Voices
* `GET /api/voices` → `[{impl, id, name, language, gender, sample}]` across TTS modules.
* `POST /api/voices/{impl}/preview` body `{voice_id?, text?, speed?, pitch?, params?}` → `audio/wav`
  (out_rate, PCM16). Loads the module if needed (into the auxiliary pool, not the slot). `params`
  replaces the engine's params for this preview only.
* Voice lists are dynamic: `TTSModule.list_voices()` (default `info.voices`); zero-shot engines
  (tag `cloning`, i.e. `tts/fish_s2`) add every cloned voice as `clone:<id>`. `/api/modules` reports
  the same list in `voices`.
* Cloned-voice library (`data/voices/<id>/{clip.wav, clip.lab, meta.json}`; `core/clones.py`):
  * `GET /api/clones` → `[CloneVoice]`, `CloneVoice = {id, name, language, notes, created,
    duration_s, sample_rate, peak_db, warning, transcript, voice_id: "clone:<id>", audio_url}`.
  * `POST /api/clones` body `{name, transcript, audio_b64 (WAV), language?, notes?}` → 201
    `CloneVoice`; 422 without a transcript, for a non-WAV body, a silent clip or one outside 2–30 s.
    The clip is stored peak-normalised mono. If the Fish server is up, the new reference is encoded
    there in the background so its first real use does not pay the ~10 s encode.
  * `GET /api/clones/{id}`, `GET /api/clones/{id}/audio` (`audio/wav`), `PUT /api/clones/{id}`
    `{name?, transcript?, language?, notes?}`, `DELETE /api/clones/{id}`. Edits and deletes clear the
    phrase cache. Each change emits `clones.changed` `{id, action}` (global).
* Fish S2-Pro server (`core/fish_server.py`): `GET /api/fish` → `FishStatus = {url, dir, health:
  "ok"|"loading"|"starting"|"error"|"offline", error, managed, pid, started_s_ago, venv_present,
  checkpoint_present, can_start, info: subset of the server's /status, log_tail: [str]}`;
  `POST /api/fish/start` (409 when no server repo/venv is found), `POST /api/fish/stop` (stops the
  child ttsPIPE started, else whatever listens on the port). Log: `data/logs/fish-server.log`.

### 9.7 State and sessions
* `GET /api/state` → `{paused, sessions: int, active_profile, bridge: BridgeStatus, cache: CacheStats,
  uptime_s}`.
* `GET /api/sessions` → `[SessionSummary]` = `{id, label, started, state, voice, utterances,
  last_utterance_id, t_out_ms, connected}`.
* `GET /api/sessions/{id}` → `SessionSummary + {utterances: [UtteranceSummary] (last 50)}`.
  `UtteranceSummary` = `utterance.end` data (+ `status: "open"|"speaking"|"final"|"interrupted"`,
  `heard_text`, `unplayed_text`, `edits`, `segments`).
* `POST /api/sessions/{id}/speak` `{text, utterance_id?}`; `.../duck`, `.../stop`, `.../resume` →
  `{ok: true}`.

### 9.8 Renders
* `GET /api/renders?session_id=` → `[{id, session_id, utterance_id, text, duration_ms, created,
  interrupted, sample_rate, url}]` (`url` = `/api/renders/{id}.wav`).
* `GET /api/renders/{id}.wav` (download), `DELETE /api/renders/{id}`.
Render id = `<session_id>__<utterance_id>`.

### 9.9 Runtime
`POST /api/runtime/pause` → intake and synthesis stop taking new work, pacers stop sending audio
(stream clock keeps running). `POST /api/runtime/resume`. Both return `/api/state`.

### 9.10 Metrics and logs
* `GET /api/metrics` → `Metrics` (§9.14). `GET /metrics` (root) → Prometheus text format.
* `GET /api/logs?level=&limit=200&session_id=` → `[{ts, level, logger, msg, session_id}]`.

### 9.11 Scripts
`GET /api/scripts` → `[{id, title, language: "ta"|"tanglish"|"en"|"mixed", text}]`. At least: Tamil,
Tanglish (script), Tanglish (romanized), English, numbers/dates stress test, long paragraph.

### 9.12 Profiles and layouts
* `GET /api/profiles` → `[{name, builtin, description, updated}]`.
* `GET /api/profiles/{name}` → `{name, builtin, description, config: PartialConfig}`.
* `PUT /api/profiles/{name}` body `{description?, config?: PartialConfig, from_current?: bool}` →
  saves a user profile (409 for builtin names). `from_current: true` snapshots the effective config.
* `DELETE /api/profiles/{name}` (409 for builtin).
* `POST /api/profiles/{name}/apply` → config = defaults ⊕ profile; `active_profile = name`; returns
  the `/api/pipeline` body. Applying hot-swaps changed stages.
* `GET /api/layouts` → `[{name, updated}]`; `GET/PUT/DELETE /api/layouts/{name}` with body
  `Layout` (§14).

### 9.13 Compare (A/B)
* `POST /api/compare` body `{text, a: Variant, b: Variant}`, `Variant = {profile?: str, config?:
  PartialConfig, label?: str}` → renders the text offline (no pacing) through each variant →
  `{a: CompareResult, b: CompareResult}`, `CompareResult = {label, render_id, url, tts_impl,
  voice_id, metrics: {ttfa_ms, first_synth_ms, total_synth_ms, audio_ms, rtf, chunks}, error}`.
  Variant modules other than the slots' are loaded into the auxiliary pool (§9.6).
* `POST /api/compare/ratings` `{render_id, tts_impl, voice_id, score: 1..5, text, rater?}` → `{ok}`.
* `GET /api/compare/ratings` → `{rows: [...], summary: [{tts_impl, voice_id, n, mos, ci95}]}`.
* `POST /api/render` body `{text} + Variant` → one `CompareResult` (renders saved under session
  `test`); `metrics` also carries `wall_ms`. Used by the Test and Voice cloning pages.

### 9.14 Metrics object
```
{ uptime_s, sessions, paused,
  stages: {kind: {impl, count, errors, last_ms, p50_ms, p95_ms, spark: [p95 per 10 s, last 30]}},
  ttfa: {by_profile: {name: {n, p50, p95, last}}, end_by_profile: {…same…}},
  latency: {ttfa: {last,p50,p95}, ttfa_end:{…}, text_to_first_chunk:{…}, first_synth:{…},
            normalize:{…}, chunk_rtf:{…}},
  throughput: {chunks_per_min, audio_s_per_min, chars_per_min},
  cache: {hits, misses, hit_rate, entries, mb},
  underruns, interruptions, utterances, events_dropped,
  system: {cpu_pct, rss_mb, threads, cores} }
```

---

## 10. Status values and UI palette

| status | meaning | colour |
|---|---|---|
| `ready` | loaded and serving | green |
| `loading` | loading in background | purple |
| `unloaded` | known, not loaded | orange |
| `error` | failed; slot uses builtin | red |
| `disabled` | stage disabled | grey |
| busy (flag) | a call is in flight | outline ring |
| fallback (flag) | builtin standing in | warning triangle |

---

## 11. Config schema (`config/pipeline.yaml`)

```yaml
audio:    {out_rate: 24000, frame_ms: 20}
stages:
  ingest:    {enabled: true, impl: plain,     params: {}}
  normalize: {enabled: true, impl: rules,     params: {}}
  segment:   {enabled: true, impl: script,    params: {}}
  chunker:   {enabled: true, impl: punct,     params: {}}
  g2p:       {enabled: true, impl: lexicon,   params: {}}
  prosody:   {enabled: true, impl: rules,     params: {}}
  tts:       {enabled: true, impl: <default>, params: {}}
  post:      {enabled: true, impl: master,    params: {}}
voice:    {impl: <tts impl>, voice_id: <id>, speed: 1.0, pitch: 0.0}
playback: {lookahead_chunks: 2, jitter_ms: 60, duck_db: 12, duck_ramp_ms: 80, duck_release_ms: 2000,
           stop_fade_ms: 60, crossfade_ms: 8, max_silence_gap_ms: 400, fill_underrun_silence: true,
           level_every_ticks: 2}
chunker:  {first_chunk_min_words: 4, first_chunk_max_words: 10, first_chunk_max_wait_ms: 400,
           min_chunk_chars: 40, max_chunk_chars: 220, nonstreaming_max_chunk_chars: 140}
cache:    {enabled: true, max_mb: 64}
renders:  {enabled: true, keep: 200}
events:   {queue_size: 2000}
graph:    {shadows: {tts: null},   # or {impl, params, voice_id}
           positions: {}}
bridge:   {enabled: false, prepipe_url: "", session_map: latest, send_reference_audio: false,
           reference_ws_path: /ws/audio,
           map: {barge_in.candidate: duck, barge_in: stop, backchannel: resume}}
active_profile: balanced
```

Rules:
* Runtime edits persist to `data/pipeline.override.yaml`, deep-merged (dicts merge, everything else
  replaces) over the defaults at startup. If the merged override fails validation, each invalid key
  is dropped with a `log` warning; if still invalid the whole override is ignored.
* `stages.<kind>.impl` is the impl name without the kind prefix.
* `voice.impl` must equal `stages.tts.impl` or a sub-engine of a router (validation warning if not).
* Validation (`POST /api/pipeline/validate`) errors: unknown kind/impl, params schema errors, invalid
  `audio.out_rate`, `frame_ms` ≠ 20 (only 20 supported), negative timers, disabling a required stage.
  Warnings: module unavailable (will fall back), model files missing (download size), shadow equals
  primary, RAM estimate above 75 % of total RAM.

---

## 12. Builtin and ML modules

### 12.1 Builtins (no ML deps, always available; the fallback of their kind)

| id | what |
|---|---|
| `ingest/plain` | markdown/emoji/URL/code stripping, SSML-lite, word-boundary holding |
| `normalize/rules` | Tamil + English wording for numbers, ₹, dates, times, phones, units, abbreviations |
| `segment/script` | Unicode-range runs + Tanglish lexicon & suffix heuristics for `ta-Latn` |
| `chunker/punct` | §4.1 rules |
| `g2p/lexicon` | lexicon overrides + rule-based romanized-Tamil → Tamil-script transliteration |
| `prosody/rules` | punctuation pauses, question/exclaim pitch & style, emphasis, SSML-lite rate/pitch |
| `tts/tone` | deterministic syllable-rate tones, streaming per word, exact word timings; simulated latency/RTF params for testing |
| `tts/clips` | pre-recorded phrase WAVs from `data/clips/` (exact normalized-text match); falls back to `tts/tone` |
| `post/passthrough` | identity |

Also builtin (numpy only) but not a fallback: `post/master` (DC block, de-click fades at chunk joins,
silence trim, approximate short-term loudness normalization to `target_lufs`, peak limiter).

### 12.2 ML modules

The concrete list, licences and measured numbers live in `docs/research/README.md`; this table is the
contract for ids and families (`requirements/ml-<family>.txt`).

| id | family | languages | notes |
|---|---|---|---|
| `tts/lang_router` | — (builtin router) | ta, en, ta-Latn | routes each language run to `ta_impl` / `en_impl`, merges runs shorter than `min_run_words`; loads in the background like an ML module |
| `tts/piper` | `piper` | ta, en, ta-Latn | Piper VITS ONNX; Tamil AI4Bharat Rasa voices + English ljspeech/kristin/lessac; streams per sentence; CPU default (`balanced`) |
| `tts/kokoro` | `kokoro` | en | Kokoro-82M ONNX; English only; non-streaming; `high_quality` English |
| `tts/mms_tam` | `mms` | ta | Meta MMS-TTS Tamil (VITS, transformers); **CC-BY-NC-4.0**; drops Latin text; A/B shadow only |
| `tts/fish_s2` | `fish` | ta, en, ta-Latn, ~50 more | Fish Audio S2-Pro 4B NF4 on the GPU via the Fish server (`TTSPIPE_FISH_URL`), one start→text→stop turn per chunk on a kept-open socket, 44.1 kHz PCM streamed in ~186 ms blocks; voices `server_active`, `model_default`, `clone:<id>`; tags `remote` (loads outside the global load lock; `load()` waits for — and with `autostart` starts — the server), `cloning`, `no-speed`; **Fish Audio Research Licence (non-commercial)**; **default** |

Evaluated and rejected (reports in `docs/research/`): IndicF5 (RTF 12–126), Indic Parler-TTS (RTF ≈ 16,
4.4 GB), AI4Bharat Indic-TTS FastPitch (RTF ≈ 0.8, 2 GB, drops Latin), eSpeak NG (robotic).

### 12.3 Router semantics (`tts/lang_router`)

Params: `ta_impl`, `ta_voice`, `en_impl`, `en_voice`, `ta_latn_to` (`ta` | `en`, default `ta`),
`min_run_words` (default 2), `ta_gain_db` / `en_gain_db` (loudness match). A chunk is split into maximal
runs by language (after mapping `ta-Latn` per `ta_latn_to`); runs shorter than `min_run_words` merge into
a neighbouring run (preferring the chunk's dominant language). Tamil-script runs only ever merge into
Tamil runs (an English engine cannot read Tamil script). `ta-Latn` text routed to the English engine
keeps its Latin spelling; routed to the Tamil engine it uses the g2p transliteration. Engines without
word timings get per-run estimated timings from the router. Runs are synthesized in order, each by
its engine; pieces carry their own `sample_rate`. Sub-engines are loaded by the router's `load()` via
the registry and run on the TTS slot thread. The router is `streaming` if both sub-engines are.

### 12.4 Lexicon (`config/lexicon.yaml`)

```yaml
words:            # whole-word overrides applied by g2p; key match is case-insensitive for Latin
  en: {"SBI": "S B I", "UPI": "U P I"}
  ta: {"IndiGo": "இண்டிகோ"}
tanglish:         # romanized-Tamil words for segment/script (lowercase)
  - enna
  - romba
```

---

## 13. Profiles

A profile is a **partial config** (any subset of §11 except `active_profile`) plus `description`.
Builtins live in `config/profiles/<name>.yaml` and are read-only; user profiles in
`data/profiles/<name>.yaml`. Applying = defaults ⊕ profile (deep merge) → becomes the new override.
`dirty` = effective config ≠ defaults ⊕ active profile.

Builtin profiles: `fish_s2` (default; GPU), `lowest_latency`, `balanced` (CPU-only default), `high_quality`, `minimal` (builtins only),
`ab_tts_compare` (sets a shadow TTS), `english_fast`. Their engine choices follow the measurements in
`docs/research/README.md`.

Merge semantics (all config merges, §11): dicts merge recursively, except that `stages.<kind>.params`
and `graph.shadows.tts` are replaced as a whole, and giving a stage a different `impl` without `params`
resets its params to `{}`.

---

## 14. Frontend contract

* Stack: Vite + React 19 + TypeScript, no UI framework, hand-written CSS with light (default) and
  dark themes (`data-theme` on `<html>`), monospace for numbers.
* `src/types/contract.ts` mirrors §6–§11; every field optional; code defensively.
* Data sources: REST (§9) + `/ws/events` (monitor) + its own `/ws/tts` connection (label
  `dashboard`) for the Speak box, played through WebAudio.
* **Mock mode** `?mock=1`: synthetic event stream and mock API implementing the same contract.
* Backend offline: banner with retry count; reconnect with backoff.
* Layout engine: panel registry (`id → {title, component, minW, minH, defaultSize}`), grid of
  12 columns, drag to move, corner to resize, collapse / maximize / close per panel. `Layout` =
  `{name, cols: 12, row_h: 28, panels: [{id, x, y, w, h, collapsed?, hidden?}], updated?}`; saved via
  `/api/layouts`. Presets ship in the frontend.
* Pages (hash routes): `#/test` (default) test bench — text, engine + voice, Live (through the
  real-time pipeline, switching it to the chosen engine/voice) or Render (offline `POST /api/render`,
  pipeline unchanged), with every run's metrics and audio; `#/engines` — every TTS engine with status,
  model files + download, parameters (apply / use as pipeline engine), voice preview, profile apply and
  the Fish server card (status, start/stop, log); `#/voices` — voice cloning: record (MediaRecorder,
  raw capture) or upload, decoded and converted to trimmed mono 16-bit WAV in the browser, exact
  transcript, save, then per-clone test / use in pipeline / edit / delete; `#/monitor` — the panel
  grid below.
* Panels: Top bar, Pipeline graph (Live/Edit studio), Live output signal timeline, Playback state,
  Language segments, Voice, Duplex, Latency, Utterances, A/B compare, Diagnostics (Metrics, Logs,
  Events, Modules/Models), Config drawer (Stage, Voice, Playback, Chunker).

---

## 15. Testing contract

* `./run.sh test` = `pytest backend/tests` (core units, every module, API, regressions) + `tsc -b`.
* `./run.sh smoke [--url URL]` runs `scripts/smoke.py` (a private server on a free port by default).
* Tests must pass with **core deps only**; ML-module tests skip when their family is not installed.
* `scripts/smoke.py` speaks every `/api/scripts` text end to end with `tts/tone` and with the default
  engine, prints TTFA, and checks stop-mid-utterance `heard_text` against the word timeline.

---

## 16. Change log

* 1.0 — initial contract.
* 1.0.1 — added during implementation: `LangSegment.attrs`, `Frame.rate`/`fade_ms`, `ModuleInfo.family`,
  `PUT /api/pipeline/positions`, `TTSPIPE_MODELS_DIR`, `config/scripts.yaml`; router merge rule and gain
  params; `post` timings are recorded in metrics from the pacer thread (no `stage.activity` per frame).
  Late-joining dashboards hydrate sessions from `GET /api/sessions/{id}`.
* 1.0.2 — ML modules `tts/piper`, `tts/kokoro`, `tts/mms_tam` (§12.2); stage params atomic in merges (§13);
  default config = `balanced` (language router over Piper).
* 1.1 — `tts/fish_s2` (Fish S2-Pro on the GPU) is the default engine (profile `fish_s2`; Piper stays as
  `balanced`); cloned-voice library and `/api/clones`; Fish server control `/api/fish`; `POST /api/render`;
  preview `params`; dynamic `list_voices()`; event `clones.changed`; settings `TTSPIPE_FISH_DIR` /
  `TTSPIPE_FISH_URL`; module tags `remote`, `cloning`, `no-speed`; UI pages Test / Engines / Voice
  cloning / Monitor.
