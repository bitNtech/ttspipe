// Mock backend for ?mock=1: an in-memory implementation of the REST contract plus a synthetic event stream.
import type {
  Config, Envelope, Kind, Layout, ModuleEntry, ParamsSchema, PipelineView, PlaybackStateName, Profile,
  StageStatus, UtteranceLatency,
} from "../types/contract";
import { KINDS } from "../types/contract";

type Emit = (ev: Envelope) => void;
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

const S = (type: string, def: unknown, extra: Record<string, unknown> = {}) => ({ type, default: def, ...extra }) as any;

const BUILTIN: [string, string, ParamsSchema, Partial<ModuleEntry>?][] = [
  ["ingest/plain", "Plain text + markdown strip", { strip_urls: S("boolean", true), strip_emoji: S("boolean", true), default_break_ms: S("integer", 300, { min: 0, max: 5000, unit: "ms" }) }],
  ["normalize/rules", "Rule-based normalizer (ta/en)", { language: S("string", "auto", { enum: ["auto", "ta", "en"] }), en_number_style: S("string", "indian", { enum: ["indian", "western"] }), phone_digits: S("boolean", true) }],
  ["segment/script", "Script + Tanglish lexicon", { detect_romanized: S("boolean", true), heuristics: S("boolean", true) }],
  ["chunker/punct", "Punctuation chunker", { clause_flush: S("boolean", true) }],
  ["g2p/lexicon", "Lexicon + Tanglish transliteration", { apply_lexicon: S("boolean", true), transliterate_romanized: S("boolean", true) }],
  ["prosody/rules", "Rule-based prosody", { sentence_pause_ms: S("integer", 180, { min: 0, max: 2000, unit: "ms" }), question_pitch_st: S("number", 1, { min: -6, max: 6 }) }],
  ["tts/tone", "Tone (test engine)", { syllable_ms: S("integer", 150, { min: 40, max: 600, unit: "ms" }), latency_ms: S("integer", 30, { min: 0, max: 5000, unit: "ms" }), rtf: S("number", 0.05, { min: 0, max: 3 }) },
    { streaming: true, native_rate: 22050, voices: [{ id: "tone-f", name: "Tone F (220 Hz)", language: "multi", gender: "female" }, { id: "tone-m", name: "Tone M (130 Hz)", language: "multi", gender: "male" }] }],
  ["tts/clips", "Phrase clips", { fallback: S("string", "tone", { enum: ["tone", "silence"] }) }],
  ["post/passthrough", "Passthrough", {}],
  ["post/master", "Master (DC, fades, trim, loudness, limiter)", { target_lufs: S("number", -18, { min: -40, max: -6, unit: "LUFS" }), trim: S("boolean", true), limiter_db: S("number", -1, { min: -12, max: 0 }) }],
];
const ML: Partial<ModuleEntry>[] = [
  { id: "tts/mms_tam", name: "MMS-TTS Tamil (VITS)", license: "CC-BY-NC-4.0", size_mb: 145, ram_mb: 450, streaming: false, native_rate: 16000, languages: ["ta"], requires: ["transformers", "torch"], available: false, reason: "missing python package(s): transformers, torch", voices: [{ id: "default", name: "MMS Tamil", language: "ta", gender: "male" }] },
  { id: "tts/piper", name: "Piper (VITS, ONNX)", license: "GPL-3.0 runtime + per-voice", size_mb: 244, ram_mb: 330, streaming: true, native_rate: 22050, languages: ["ta", "en", "ta-Latn"], requires: ["piper"], available: false, reason: "missing python package(s): piper", voices: [{ id: "ta_IN-rasa_female-medium", name: "ta IN rasa female", language: "ta", gender: "female" }, { id: "en_US-ljspeech-medium", name: "en US ljspeech", language: "en", gender: "female" }] },
  { id: "tts/fish_s2", name: "Fish Audio S2-Pro (NF4, GPU)", license: "Fish Audio Research License", size_mb: 4990, ram_mb: 3500, streaming: true, native_rate: 44100, languages: ["ta", "en", "ta-Latn", "multi"], requires: ["msgpack"], tags: ["remote", "gpu", "cloning"], available: true, reason: null, models_present: true, voices: [{ id: "server_active", name: "Server active voice", language: "multi" }, { id: "model_default", name: "S2-Pro default voice", language: "multi" }],
    params_schema: { temperature: { type: "number", default: 0.8, min: 0.1, max: 1 }, top_p: { type: "number", default: 0.8, min: 0.1, max: 1 }, seed: { type: "integer", default: 42, min: -1, max: 2147483647 } }, default_params: { temperature: 0.8, top_p: 0.8, seed: 42 } },
  { id: "tts/kokoro", name: "Kokoro-82M (ONNX)", license: "Apache-2.0", size_mb: 330, ram_mb: 600, streaming: false, native_rate: 24000, languages: ["en"], requires: ["kokoro_onnx"], available: false, reason: "missing python package(s): kokoro_onnx", voices: [{ id: "af_heart", name: "Heart (US F)", language: "en", gender: "female" }, { id: "am_michael", name: "Michael (US M)", language: "en", gender: "male" }] },
  { id: "tts/lang_router", name: "Language router", license: "builtin", streaming: true, languages: ["ta", "en", "ta-Latn"], tags: ["router"], available: true, reason: null },
];

function moduleList(): ModuleEntry[] {
  const out: ModuleEntry[] = BUILTIN.map(([id, name, ps, extra]) => ({
    id, kind: id.split("/")[0] as Kind, name, description: `${name} (mock)`, license: "builtin", size_mb: 0, ram_mb: 0,
    streaming: false, requires: [], params_schema: ps, default_params: Object.fromEntries(Object.entries(ps).map(([k, v]) => [k, v.default])),
    tags: ["builtin"], languages: [], voices: [], builtin: true, available: true, reason: null, models_present: true, ...extra,
  }));
  for (const m of ML) out.push({ kind: m.id!.split("/")[0] as Kind, description: `${m.name} (mock)`, params_schema: {}, default_params: {}, tags: [], builtin: false, models_present: false, ...m });
  return out;
}

const DEFAULT: Config = {
  audio: { out_rate: 24000, frame_ms: 20 },
  stages: { ingest: { enabled: true, impl: "plain", params: {} }, normalize: { enabled: true, impl: "rules", params: {} }, segment: { enabled: true, impl: "script", params: {} }, chunker: { enabled: true, impl: "punct", params: {} }, g2p: { enabled: true, impl: "lexicon", params: {} }, prosody: { enabled: true, impl: "rules", params: {} }, tts: { enabled: true, impl: "tone", params: {} }, post: { enabled: true, impl: "master", params: {} } },
  voice: { impl: "tone", voice_id: "tone-f", speed: 1, pitch: 0 },
  playback: { lookahead_chunks: 2, jitter_ms: 60, duck_db: 12, duck_ramp_ms: 80, duck_release_ms: 2000, stop_fade_ms: 60, crossfade_ms: 8, max_silence_gap_ms: 400, fill_underrun_silence: true, level_every_ticks: 2 },
  chunker: { first_chunk_min_words: 4, first_chunk_max_words: 10, first_chunk_max_wait_ms: 400, min_chunk_chars: 40, max_chunk_chars: 220, nonstreaming_max_chunk_chars: 140 },
  cache: { enabled: true, max_mb: 64 }, renders: { enabled: true, keep: 200 }, events: { queue_size: 2000 },
  graph: { shadows: { tts: null }, positions: {} },
  bridge: { enabled: false, prepipe_url: "", session_map: "latest", send_reference_audio: false, reference_ws_path: "/ws/audio", map: {} },
  active_profile: "balanced",
};
let config: Config = clone(DEFAULT);
const PROFILES: Record<string, Profile> = {
  balanced: { name: "balanced", builtin: true, description: "Default (mock)", config: {} },
  lowest_latency: { name: "lowest_latency", builtin: true, description: "Smallest first chunk", config: { chunker: { first_chunk_min_words: 3 }, playback: { jitter_ms: 40, lookahead_chunks: 1 } } },
  high_quality: { name: "high_quality", builtin: true, description: "Longer chunks", config: { chunker: { first_chunk_min_words: 6 } } },
  minimal: { name: "minimal", builtin: true, description: "Builtins only", config: { stages: { post: { impl: "passthrough" } } } },
  ab_tts_compare: { name: "ab_tts_compare", builtin: true, description: "Shadow engine", config: { graph: { shadows: { tts: { impl: "tone", params: {}, voice_id: "tone-m" } } } } },
  english_fast: { name: "english_fast", builtin: true, description: "English traffic", config: { segment: undefined } as Config },
};

function merge(a: any, b: any): any {
  const out = clone(a ?? {});
  for (const [k, v] of Object.entries(b ?? {})) {
    if (v && typeof v === "object" && !Array.isArray(v) && out[k] && typeof out[k] === "object") out[k] = merge(out[k], v);
    else if (v !== undefined) out[k] = clone(v);
  }
  return out;
}

function stageStatus(kind: Kind): StageStatus {
  const st = config.stages?.[kind] ?? {};
  const mod = moduleList().find((m) => m.id === `${kind}/${st.impl}`);
  const fallback = !!mod && mod.available === false;
  return {
    kind, name: kind, impl: fallback ? (kind === "tts" ? "tone" : st.impl) : st.impl, configured_impl: st.impl,
    enabled: st.enabled !== false, status: st.enabled === false ? "disabled" : "ready", error: fallback ? mod?.reason ?? null : null, fallback,
    busy: false, params: { ...(mod?.default_params ?? {}), ...(st.params ?? {}) }, load_ms: 1, count: engine.counts[kind] ?? 0, errors: 0,
    consecutive_errors: 0, last_ms: engine.lastMs[kind] ?? null, p50_ms: engine.lastMs[kind] ?? null, p95_ms: (engine.lastMs[kind] ?? 0) * 1.6 || null,
  };
}

function pipelineView(): PipelineView {
  const stages = Object.fromEntries(KINDS.map((k) => [k, stageStatus(k)])) as PipelineView["stages"];
  const sh = config.graph?.shadows?.tts;
  const nodes: { id: string; kind: string; label: string; impl: string | null }[] = [{ id: "text_in", kind: "source", label: "TEXT IN", impl: null }, ...KINDS.map((k) => ({ id: k, kind: k, label: k.toUpperCase(), impl: stages![k]!.impl ?? null }))];
  if (sh) nodes.push({ id: "shadow_tts", kind: "shadow", label: "SHADOW TTS", impl: sh.impl ?? null });
  nodes.push({ id: "audio_out", kind: "sink", label: "AUDIO OUT", impl: null });
  const chain = ["text_in", ...KINDS, "audio_out"];
  const edges = chain.slice(1).map((b, i) => ({ from: chain[i], to: b, dashed: false }));
  if (sh) edges.push({ from: "prosody", to: "shadow_tts", dashed: true });
  return { config, stages, shadow: sh ? { ...stageStatus("tts"), name: "shadow", impl: sh.impl } : null, graph: { nodes, edges, positions: config.graph?.positions ?? {} }, active_profile: config.active_profile, dirty: false };
}

function emitStatus(kind: Kind) { engine.emit({ type: "stage.status", session_id: null, ts: Date.now(), t_out_ms: null, data: stageStatus(kind) }); }

const LAYOUT_KEY = "ttspipe.mock.layouts";
const layouts = (): Record<string, Layout> => { try { return JSON.parse(localStorage.getItem(LAYOUT_KEY) ?? "{}"); } catch { return {}; } };
const saveLayouts = (l: Record<string, Layout>) => { try { localStorage.setItem(LAYOUT_KEY, JSON.stringify(l)); } catch { /* private mode */ } };

const SCRIPTS = [
  { id: "ta", title: "Tamil — weather", language: "ta", text: "வணக்கம்! இன்று வானிலை மிகவும் நன்றாக இருக்கிறது. நாளை மாலை லேசான மழை பெய்யக்கூடும்." },
  { id: "tanglish_script", title: "Tanglish — Tamil script", language: "tanglish", text: "நாளைக்கு meeting இருக்கு, please time-க்கு வந்துடுங்க." },
  { id: "tanglish_roman", title: "Tanglish — romanized", language: "tanglish", text: "Naalaikku meeting irukku, please time ku vandhudunga." },
  { id: "en", title: "English — order update", language: "en", text: "Hello! Your order has shipped and should arrive by Thursday afternoon. You can track it any time from the app." },
  { id: "numbers", title: "Numbers, dates & money stress test", language: "mixed", text: "Your bill of ₹1,24,500.75 is due on 15/11/2026 at 10:30 AM. Call +91 98765 43210." },
  { id: "long", title: "Long paragraph (English)", language: "en", text: "Thanks for calling. I've checked your account and I can see the payment you made yesterday evening. It usually takes one to two working days for the bank to confirm a transfer, so the status should update by tomorrow." },
];

function toneWav(seconds: number, freq = 220): Blob {
  const rate = 16000, n = Math.floor(seconds * rate);
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const syl = Math.floor(i / (rate * 0.15));
    const f = freq * Math.pow(2, ((syl * 7) % 5) / 12);
    const env = Math.min(1, (i % (rate * 0.15)) / 200, ((rate * 0.15) - (i % (rate * 0.15))) / 200);
    v.setInt16(44 + i * 2, Math.sin((2 * Math.PI * f * i) / rate) * 8000 * env, true);
  }
  return new Blob([buf], { type: "audio/wav" });
}
const blobUrls: Record<string, string> = {};

// ---- synthetic engine ---------------------------------------------------------------------------

const TAMIL = /[஀-௿]/;
const TANGLISH = new Set("naalaikku irukku vandhudunga enna romba nalla vaanga pannunga sollunga ku la".split(" "));
const langOf = (w: string) => (TAMIL.test(w) ? "ta" : TANGLISH.has(w.toLowerCase().replace(/[^\w]/g, "")) ? "ta-Latn" : "en");

class MockEngine {
  handlers: Emit[] = [];
  sid = "mock1";
  t0 = performance.now();
  state: PlaybackStateName = "idle";
  n = 0;
  counts: Record<string, number> = {};
  lastMs: Record<string, number> = {};
  queue: Promise<void> = Promise.resolve();
  cancel = 0;
  ducked = false;
  heard: string[] = [];
  started = false;
  metrics = { ttfa: [] as number[], utterances: 0, interruptions: 0 };

  tOut() { return Math.round((performance.now() - this.t0) / 20) * 20; }
  emit(ev: Envelope) { this.handlers.forEach((h) => h(ev)); }
  ev(type: string, data: Record<string, unknown>) { this.emit({ type, session_id: this.sid, ts: Date.now(), t_out_ms: this.tOut(), data }); }
  setState(s: PlaybackStateName, reason: string, uid?: string) {
    if (s === this.state) return;
    this.ev("playback.state", { state: s, prev: this.state, reason, utterance_id: uid });
    this.state = s;
  }
  activity(kind: string, ms: number, uid?: string, cid?: string) {
    this.counts[kind] = (this.counts[kind] ?? 0) + 1; this.lastMs[kind] = ms;
    this.ev("stage.activity", { kind, impl: config.stages?.[kind as Kind]?.impl, ms, ok: true, utterance_id: uid, chunk_id: cid });
  }

  start(h: Emit) {
    this.handlers.push(h);
    if (this.started) return;
    this.started = true;
    this.ev("session.started", { label: "mock", sample_rate: 24000, frame_ms: 20, format: "pcm16le", voice: config.voice, profile: "balanced" });
    setInterval(() => this.emit({ type: "metrics", session_id: null, ts: Date.now(), t_out_ms: null, data: this.metricsSnap() }), 1000);
    setInterval(() => { if (this.state === "idle") this.ev("audio.level", { rms: 0, peak: 0, rms_db: -120, gain_db: 0, wave: new Array(16).fill(0) }); }, 400);
    let k = 0;
    const auto = () => { if (this.state === "idle") this.speak(SCRIPTS[k++ % SCRIPTS.length].text, k % 2 === 0); };
    setTimeout(auto, 600);
    setInterval(auto, 14000);
    this.emit({ type: "log", session_id: null, ts: Date.now(), t_out_ms: null, data: { level: "INFO", logger: "mock", msg: "mock backend started" } });
  }

  metricsSnap() {
    const pct = (a: number[], p: number) => { if (!a.length) return null; const s = [...a].sort((x, y) => x - y); return Math.round(s[Math.floor((s.length - 1) * p)]); };
    const stages = Object.fromEntries(KINDS.map((k) => [k, { impl: config.stages?.[k]?.impl, count: this.counts[k] ?? 0, errors: 0, last_ms: this.lastMs[k] ?? null, p50_ms: this.lastMs[k] ?? null, p95_ms: (this.lastMs[k] ?? 0) * 1.5, spark: Array.from({ length: 20 }, (_, i) => (this.lastMs[k] ?? 1) * (1 + 0.3 * Math.sin(i + k.length))) }]));
    const t = { n: this.metrics.ttfa.length, last: this.metrics.ttfa.at(-1) ?? null, p50: pct(this.metrics.ttfa, 0.5), p95: pct(this.metrics.ttfa, 0.95) };
    return {
      uptime_s: (performance.now() - this.t0) / 1000, sessions: 1, paused: false, stages,
      ttfa: { by_profile: { balanced: t }, end_by_profile: { balanced: { ...t, p50: (t.p50 ?? 0) - 120 } } },
      latency: { ttfa: t, ttfa_end: t, text_to_first_chunk: { n: 1, last: 14, p50: 14, p95: 22 }, first_synth: { n: 1, last: 110, p50: 105, p95: 160 }, normalize: { n: 1, last: 0.6, p50: 0.5, p95: 1.1 }, chunk_rtf: { n: 1, last: 0.31, p50: 0.3, p95: 0.42 } },
      throughput: { chunks_per_min: 12, audio_s_per_min: 22, chars_per_min: 600 },
      cache: { hits: 3, misses: 20, hit_rate: 0.13, entries: 20, mb: 4.2, enabled: true },
      underruns: 1, interruptions: this.metrics.interruptions, utterances: this.metrics.utterances, events_dropped: 0,
      system: { cpu_pct: 35 + 20 * Math.random(), rss_mb: 812, threads: 31, cores: 8 },
    };
  }

  speak(text: string, llm = false) {
    const myCancel = this.cancel;
    this.queue = this.queue.then(() => (myCancel === this.cancel ? this.run(text, llm, ++this.n) : undefined));
  }

  async run(text: string, llm: boolean, n: number) {
    const uid = `u${n}`, gen = this.cancel;
    const alive = () => gen === this.cancel;
    const t0 = performance.now();
    this.ev("utterance.started", { utterance_id: uid });
    this.setState("buffering", "text", uid);
    const words = text.split(/\s+/).filter(Boolean);
    if (llm) for (let i = 0; i < text.length && alive(); i += 4) { this.ev("text.delta", { utterance_id: uid, delta: text.slice(i, i + 4), chars_total: i + 4 }); await sleep(25); }
    else this.ev("text.delta", { utterance_id: uid, delta: text, chars_total: text.length });
    if (!alive()) return;
    const norm = text.replace(/₹([\d,]+)/g, (_, d) => `${d.replace(/,/g, "")} rupees`).replace(/10:30 AM/g, "ten thirty a m");
    this.ev("text.normalized", { utterance_id: uid, before: text, after: norm, changed: norm !== text, edits: [], latency_ms: 0.6 });
    this.activity("ingest", 0.2, uid); this.activity("normalize", 0.6, uid);
    const spans: { text: string; lang: string }[] = [];
    for (const w of norm.split(/\s+/)) { const l = langOf(w); if (spans.length && spans.at(-1)!.lang === l) spans.at(-1)!.text += " " + w; else spans.push({ text: w, lang: l }); }
    this.ev("segment", { utterance_id: uid, spans, latency_ms: 0.3 }); this.activity("segment", 0.3, uid);
    // chunks: first 4+ words to a clause end, then sentences
    const chunks: string[] = []; let cur: string[] = [];
    for (const w of norm.split(/\s+/)) { cur.push(w); const first = chunks.length === 0; if ((first && cur.length >= 4 && /[,.!?]$/.test(w)) || (!first && /[.!?]$/.test(w)) || (first && cur.length >= 10)) { chunks.push(cur.join(" ")); cur = []; } }
    if (cur.length) chunks.push(cur.join(" "));
    let ttfa: number | null = null;
    const heardWords: string[] = [];
    const chunkSummaries: unknown[] = [];
    let synthTotal = 0, audioTotal = 0;
    for (let i = 0; i < chunks.length && alive(); i++) {
      const c = chunks[i], cid = `${uid}.${i}`, cw = c.split(" ");
      const lang = langOf(cw[0]);
      this.ev("chunk.flushed", { utterance_id: uid, chunk_id: cid, index: i, text: c, norm_text: c, tts_text: c, lang, words: cw.length, reason: i === 0 ? "first" : i === chunks.length - 1 ? "end" : "punct", segments: [{ text: c, lang }], pause_after_ms: 180, since_text_ms: Math.round(performance.now() - t0) });
      this.activity("chunker", 0.4, uid, cid); this.activity("g2p", 0.3, uid, cid); this.activity("prosody", 0.1, uid, cid);
      this.ev("synth.start", { utterance_id: uid, chunk_id: cid, impl: config.stages?.tts?.impl, queue_ms: 0.4, cached: false });
      if (this.state === "buffering") this.setState("synthesizing", "synth_start", uid);
      const audioMs = cw.length * 330, synthMs = audioMs * 0.3;
      await sleep(110);
      this.ev("synth.first_audio", { utterance_id: uid, chunk_id: cid, latency_ms: 110, cached: false });
      this.activity("tts", synthMs, uid, cid);
      setTimeout(() => this.ev("synth.done", { utterance_id: uid, chunk_id: cid, audio_ms: audioMs, synth_ms: synthMs, rtf: 0.3, cached: false, word_timings: true }), synthMs);
      synthTotal += synthMs; audioTotal += audioMs;
      if (ttfa === null) { ttfa = Math.round(performance.now() - t0); this.setState(this.ducked ? "ducked" : "speaking", "first_audio", uid); }
      this.ev("playback.chunk", { utterance_id: uid, chunk_id: cid, event: "start", t_out_ms: this.tOut() });
      for (let w = 0; w < cw.length && alive(); w++) {
        for (let f = 0; f < 8 && alive(); f++) {
          const a = (this.ducked ? 0.25 : 1) * (0.25 + 0.2 * Math.sin(f + w));
          const wave = Array.from({ length: 16 }, (_, j) => (j % 2 ? 1 : -1) * a * (0.6 + 0.4 * Math.sin(j * 1.3 + f)));
          this.ev("audio.level", { rms: a * 0.6, peak: a, rms_db: 20 * Math.log10(a * 0.6), gain_db: this.ducked ? -12 : 0, wave });
          await sleep(330 / 8);
        }
        if (!alive()) break;
        heardWords.push(cw[w]); this.heard = heardWords;
        this.ev("playback.progress", { utterance_id: uid, chunk_id: cid, word_index: w, heard_text: heardWords.join(" ") });
      }
      this.ev("playback.chunk", { utterance_id: uid, chunk_id: cid, event: "end", t_out_ms: this.tOut() });
      chunkSummaries.push({ chunk_id: cid, index: i, text: c, norm_text: c, lang, reason: i === 0 ? "first" : "punct", words: cw.length, played: alive() ? "full" : "partial", cached: false, latency: { flush_ms: 12, synth_queue_ms: 0.4, first_audio_ms: 110, synth_ms: synthMs, audio_ms: audioMs, rtf: 0.3 } });
      if (alive() && i < chunks.length - 1) await sleep(120);
    }
    const interrupted = !alive();
    const latency: UtteranceLatency = { ttfa_ms: ttfa, ttfa_end_ms: ttfa !== null ? ttfa - 40 : null, text_to_first_chunk_ms: 14, first_synth_ms: 110, total_synth_ms: synthTotal, audio_ms: audioTotal, rtf: 0.3 };
    if (ttfa !== null) this.metrics.ttfa.push(ttfa);
    this.metrics.utterances++;
    if (interrupted) {
      this.metrics.interruptions++;
      const heard = heardWords.join(" ");
      this.ev("utterance.interrupted", { utterance_id: uid, heard_text: heard, unplayed_text: words.slice(heardWords.length).join(" "), at_ms: this.tOut(), source: "client" });
      this.setState("interrupted", "client", uid);
    } else {
      this.setState("done", "end", uid);
    }
    this.setState("idle", interrupted ? "interrupted" : "done", uid);
    const rid = `${this.sid}__${uid}`;
    blobUrls[rid] = URL.createObjectURL(toneWav(Math.max(0.5, audioTotal / 1000)));
    this.ev("utterance.end", { utterance_id: uid, text, normalized: norm, chunks: chunkSummaries, interrupted, profile: "balanced", latency, render_id: rid });
  }

  stop() {
    this.cancel++;
    this.queue = Promise.resolve();
    if (this.state === "buffering" || this.state === "synthesizing") this.setState("idle", "interrupted");
  }
  duck() { this.ducked = true; if (this.state === "speaking") this.setState("ducked", "client"); this.ev("duck", { action: "duck", gain_db: -12, source: "client" }); }
  resume() { if (!this.ducked) return; this.ducked = false; if (this.state === "ducked") this.setState("speaking", "client"); this.ev("duck", { action: "resume", gain_db: 0, source: "client" }); }
}

export const engine = new MockEngine();

const CLONES: { id: string; name: string; transcript: string; language?: string; created: number; duration_s: number; voice_id: string; audio_url: string }[] = [];

export const mockApi = {
  health: async () => ({ status: "degraded", contract: "1.1", version: "mock", uptime_s: 1, reasons: [], stages: {} }),
  modules: async () => moduleList(),
  pipeline: async () => pipelineView(),
  putStage: async (kind: Kind, body: { impl?: string; params?: Record<string, unknown>; enabled?: boolean }) => {
    const st = config.stages![kind]!;
    if (body.impl) { if (body.impl !== st.impl) st.params = {}; st.impl = body.impl; }
    if (body.params) st.params = { ...(st.params ?? {}), ...body.params };
    if (body.enabled !== undefined) {
      if (!body.enabled && ["ingest", "chunker", "tts"].includes(kind)) throw new Error(`stages.${kind}.enabled: ${kind} cannot be disabled`);
      st.enabled = body.enabled;
    }
    emitStatus(kind);
    return stageStatus(kind);
  },
  putPlayback: async (b: Record<string, unknown>) => { const { out_rate, ...rest } = b; if (out_rate) config.audio!.out_rate = out_rate as number; config.playback = { ...config.playback, ...rest }; return config.playback; },
  putChunker: async (b: Record<string, unknown>) => (config.chunker = { ...config.chunker, ...b }),
  putVoice: async (b: Record<string, unknown>) => (config.voice = { ...config.voice, ...b }),
  putShadow: async (b: { impl: string | null; params?: Record<string, unknown>; voice_id?: string | null }) => { config.graph!.shadows!.tts = b.impl ? { impl: b.impl, params: b.params ?? {}, voice_id: b.voice_id } : null; return pipelineView(); },
  putPositions: async (b: Record<string, { x: number; y: number }>) => (config.graph!.positions = b),
  putPipeline: async (cfg: Config) => { config = merge(DEFAULT, cfg); return pipelineView(); },
  validate: async (cfg: Config) => {
    const c = merge(config, cfg);
    const warnings: string[] = [];
    let dl = 0, ram = 0;
    for (const k of KINDS) {
      const m = moduleList().find((x) => x.id === `${k}/${c.stages?.[k]?.impl}`);
      if (m && m.available === false) warnings.push(`stages.${k}: ${m.id} unavailable (${m.reason}); will fall back to builtin`);
      dl += m && !m.models_present ? m.size_mb ?? 0 : 0; ram += m?.ram_mb ?? 0;
    }
    return { ok: true, errors: [], warnings, estimate: { download_mb: dl, ram_mb: ram } };
  },
  resetConfig: async () => { config = clone(DEFAULT); return pipelineView(); },
  loadModule: async (kind: Kind, impl: string) => mockApi.putStage(kind, { impl }),
  unloadModule: async (kind: Kind) => stageStatus(kind),
  models: async () => ML.filter((m) => m.size_mb).map((m) => ({ id: m.id, kind: m.id!.split("/")[0], impl: m.id!.split("/")[1], files: [{ name: "model.bin", size_mb: m.size_mb, present: false }], present: false, size_mb: m.size_mb, downloading: false, progress: null })),
  download: async (kind: string, impl: string) => {
    let done = 0;
    const total = ML.find((m) => m.id === `${kind}/${impl}`)?.size_mb ?? 100;
    const iv = setInterval(() => { done = Math.min(total, done + total / 10); engine.emit({ type: "model.download", session_id: null, ts: Date.now(), t_out_ms: null, data: { kind, impl, file: "model.bin", done_mb: done, total_mb: total, status: done >= total ? "done" : "running" } }); if (done >= total) clearInterval(iv); }, 300);
    return { status: "running" };
  },
  voices: async () => moduleList().filter((m) => m.kind === "tts").flatMap((m) => (m.voices ?? []).map((v) => ({ ...v, impl: m.id!.split("/")[1] }))),
  preview: async () => toneWav(1.2),
  state: async () => ({ paused: false, sessions: 1, active_profile: config.active_profile, bridge: { enabled: false, connected: false, url: null, error: null, last_event: null, events_mapped: 0 }, cache: {}, uptime_s: 1 }),
  sessions: async () => [{ id: engine.sid, label: "mock", state: engine.state, utterances: engine.n, connected: true, t_out_ms: engine.tOut() }],
  session: async () => ({ id: engine.sid, label: "mock", utterances: [] }),
  sessionAction: async (_id: string, a: "duck" | "stop" | "resume") => { engine[a](); return { ok: true }; },
  renders: async () => Object.keys(blobUrls).map((id) => ({ id, url: blobUrls[id], session_id: engine.sid, utterance_id: id.split("__")[1], text: "", duration_ms: 1000, created: Date.now() })),
  deleteRender: async () => ({ ok: true }),
  runtime: async (a: string) => ({ paused: a === "pause" }),
  metrics: async () => engine.metricsSnap(),
  logs: async () => [{ ts: Date.now(), level: "INFO", logger: "mock", msg: "mock logs" }],
  scripts: async () => SCRIPTS,
  profiles: async () => Object.values(PROFILES).map((p) => ({ name: p.name, builtin: p.builtin, description: p.description, updated: Date.now() })),
  profile: async (name: string) => PROFILES[name],
  putProfile: async (name: string, b: { description?: string; config?: Config; from_current?: boolean }) => {
    if (PROFILES[name]?.builtin) throw new Error(`profile '${name}' is builtin (read-only)`);
    PROFILES[name] = { name, builtin: false, description: b.description ?? "", config: b.from_current ? clone(config) : b.config ?? {} };
    return PROFILES[name];
  },
  deleteProfile: async (name: string) => { if (PROFILES[name]?.builtin) throw new Error("builtin"); delete PROFILES[name]; return { ok: true }; },
  applyProfile: async (name: string) => { config = merge(DEFAULT, PROFILES[name]?.config ?? {}); config.active_profile = name; KINDS.forEach(emitStatus); return pipelineView(); },
  layouts: async () => Object.keys(layouts()).map((name) => ({ name, updated: layouts()[name].updated })),
  layout: async (name: string) => { const l = layouts()[name]; if (!l) throw new Error("404"); return l; },
  putLayout: async (name: string, body: Layout) => { const l = layouts(); l[name] = { ...body, name, updated: Date.now() }; saveLayouts(l); return l[name]; },
  deleteLayout: async (name: string) => { const l = layouts(); delete l[name]; saveLayouts(l); return { ok: true }; },
  compare: async (text: string, a: { label?: string; profile?: string }, b: { label?: string; profile?: string }) => {
    await sleep(500);
    const mk = (v: { label?: string; profile?: string }, f: number) => {
      const audio = text.split(/\s+/).length * 330 * f;
      return { label: v.label ?? v.profile ?? "variant", render_id: `cmp_${f}`, url: URL.createObjectURL(toneWav(audio / 1000, 180 * f)), tts_impl: "tone", voice_id: "tone-f", error: null, metrics: { ttfa_ms: 120 * f, first_synth_ms: 100 * f, total_synth_ms: audio * 0.3, audio_ms: audio, rtf: 0.3 * f, chunks: 3 } };
    };
    return { a: mk(a, 1), b: mk(b, 1.3) };
  },
  render: async (text: string, v: { label?: string }) => (await mockApi.compare(text, v, v)).a,
  clones: async () => [...CLONES],
  createClone: async (b: { name: string; transcript: string; audio_b64: string; language?: string }) => {
    const c = { id: `c${CLONES.length + 1}`, name: b.name, transcript: b.transcript, language: b.language, created: Date.now() / 1000, duration_s: 9.5, voice_id: `clone:c${CLONES.length + 1}`, audio_url: URL.createObjectURL(toneWav(2)) };
    CLONES.unshift(c);
    return c;
  },
  updateClone: async (id: string, b: Record<string, string>) => Object.assign(CLONES.find((c) => c.id === id) ?? {}, b),
  deleteClone: async (id: string) => { CLONES.splice(CLONES.findIndex((c) => c.id === id), 1); return { ok: true }; },
  fish: async () => ({ url: "http://127.0.0.1:8080", dir: "(mock)", health: "ok", managed: false, venv_present: true, checkpoint_present: true, can_start: false, info: { device: "cuda", quantization: "nf4", sample_rate: 44100 }, log_tail: ["mock fish server"] }),
  fishAction: async () => mockApi.fish(),
  rate: async () => ({ ok: true }),
  ratings: async () => ({ rows: [], summary: [{ tts_impl: "tone", voice_id: "tone-f", n: 4, mos: 3.5, ci95: 0.6 }] }),
};
