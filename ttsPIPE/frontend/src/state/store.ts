// Central client store: reduces /ws/events envelopes into per-topic state. Mutable + versioned per topic;
// components subscribe to topics with useTopic() and re-render at most once per animation frame.
import { useSyncExternalStore } from "react";
import type {
  AudioLevelData, ChunkFlushedData, Envelope, InterruptedData, LogRecord, Metrics, ModuleEntry, PipelineView,
  PlaybackStateName, ShadowData, StageActivityData, StageStatus, UtteranceSummary, BridgeStatus, Health, SessionDetail,
} from "../types/contract";

export type Topic = "conn" | "sessions" | "utterances" | "timeline" | "metrics" | "logs" | "events" | "pipeline" | "modules" | "level" | "toast";

export interface TLChunk {
  chunk_id: string; utterance_id: string; text: string; lang: string; reason?: string;
  t_flush: number; t_synth_start?: number; t_first_audio?: number; t_synth_done?: number;
  t_play_start?: number; t_play_end?: number; cached?: boolean;
}
export interface TLLevel { t: number; min: number; max: number; rms: number }
export interface TLState { t: number; state: PlaybackStateName }
export interface TLMarker { t: number; kind: "duck" | "resume" | "stop" | "underrun" | "error"; label?: string }

export interface UttView extends UtteranceSummary {
  utterance_id: string;
  t0?: number; // t_out_ms when started
  ts?: number; // wall clock
  live_text: string;
  norm_parts: { before: string; after: string; changed: boolean }[];
  spans: { text: string; lang: string }[];
  chunk_events: ChunkFlushedData[];
  progress_heard?: string;
}

export interface SessionView {
  id: string; label: string; started: number; connected: boolean; sample_rate?: number;
  state: PlaybackStateName; prev?: PlaybackStateName; reason?: string; t_out_ms: number; last_ts: number;
  utts: Map<string, UttView>; order: string[];
  chunks: Map<string, TLChunk>;
  levels: TLLevel[]; states: TLState[]; markers: TLMarker[];
  last_interrupted?: InterruptedData; gain_db: number; ducked: boolean; level?: AudioLevelData;
  shadow: ShadowData[];
}

const MAX_LEVELS = 3000; // ~2 min at 25 Hz
const MAX_EVENTS = 1500;
const MAX_LOGS = 800;

class Store {
  versions: Record<Topic, number> = { conn: 0, sessions: 0, utterances: 0, timeline: 0, metrics: 0, logs: 0, events: 0, pipeline: 0, modules: 0, level: 0, toast: 0 };
  listeners = new Set<() => void>();
  dirty = new Set<Topic>();
  raf = 0;

  // connection
  mode: "live" | "mock" = "live";
  connected = false;
  retries = 0;
  rtt: number | null = null;
  health: Health | null = null;

  sessions = new Map<string, SessionView>();
  selected: string = "auto"; // "auto" = latest
  metrics: Metrics | null = null;
  metricsHist: Metrics[] = [];
  logs: LogRecord[] = [];
  events: Envelope[] = [];
  pipeline: PipelineView | null = null;
  modules: ModuleEntry[] = [];
  activity: Record<string, { t: number; ms: number; ok: boolean; impl?: string }> = {};
  bridge: BridgeStatus | null = null;
  downloads: Record<string, { done_mb?: number; total_mb?: number; status?: string; error?: string | null }> = {};
  toastMsg: string | null = null;
  paused = false;

  touch(...topics: Topic[]) {
    for (const t of topics) this.dirty.add(t);
    if (!this.raf) {
      this.raf = requestAnimationFrame(() => {
        this.raf = 0;
        for (const t of this.dirty) this.versions[t]++;
        this.dirty.clear();
        this.versions = { ...this.versions };
        this.listeners.forEach((l) => l());
      });
    }
  }

  subscribe = (l: () => void) => {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  };

  toast(msg: string) {
    this.toastMsg = msg;
    this.touch("toast");
    setTimeout(() => { if (this.toastMsg === msg) { this.toastMsg = null; this.touch("toast"); } }, 3500);
  }

  current(): SessionView | null {
    if (this.selected !== "auto") return this.sessions.get(this.selected) ?? null;
    let best: SessionView | null = null;
    for (const s of this.sessions.values()) if (!best || s.last_ts > best.last_ts) best = s;
    return best;
  }

  session(id: string, ts: number): SessionView {
    let s = this.sessions.get(id);
    if (!s) {
      s = { id, label: "", started: ts, connected: true, state: "idle", t_out_ms: 0, last_ts: ts, utts: new Map(), order: [],
        chunks: new Map(), levels: [], states: [], markers: [], gain_db: 0, ducked: false, shadow: [] };
      this.sessions.set(id, s);
      this.touch("sessions");
    }
    return s;
  }

  utt(s: SessionView, uid: string, t: number, ts: number): UttView {
    let u = s.utts.get(uid);
    if (!u) {
      u = { utterance_id: uid, t0: t, ts, live_text: "", norm_parts: [], spans: [], chunk_events: [], status: "open" };
      s.utts.set(uid, u);
      s.order.push(uid);
      if (s.order.length > 100) s.utts.delete(s.order.shift()!);
    }
    return u;
  }

  ingest(ev: Envelope) {
    if (!ev || typeof ev !== "object") return;
    const d: any = ev.data ?? {};
    const type = ev.type ?? "";
    if (type !== "audio.level" && type !== "stage.activity") {
      this.events.push(ev);
      if (this.events.length > MAX_EVENTS) this.events.splice(0, this.events.length - MAX_EVENTS);
      this.touch("events");
    }
    switch (type) {
      case "metrics":
        this.metrics = d as Metrics;
        this.metricsHist.push(d);
        if (this.metricsHist.length > 300) this.metricsHist.shift();
        this.paused = !!d.paused;
        this.touch("metrics");
        return;
      case "log":
        this.logs.push({ ts: ev.ts, ...d });
        if (this.logs.length > MAX_LOGS) this.logs.splice(0, this.logs.length - MAX_LOGS);
        this.touch("logs");
        return;
      case "stage.status": {
        const st = d as StageStatus;
        if (this.pipeline?.stages && st.kind) {
          if (st.name === "shadow") this.pipeline.shadow = st;
          else if (!st.name || st.name === st.kind) this.pipeline.stages[st.kind] = st;
          this.pipeline = { ...this.pipeline };
        }
        this.touch("pipeline", "modules");
        return;
      }
      case "stage.activity": {
        const a = d as StageActivityData;
        if (a.kind) this.activity[a.kind] = { t: performance.now(), ms: a.ms ?? 0, ok: a.ok !== false, impl: a.impl };
        this.touch("pipeline");
        return;
      }
      case "bridge.status":
        this.bridge = d;
        this.touch("sessions");
        return;
      case "model.download":
        this.downloads[`${d.kind}/${d.impl}`] = d;
        this.touch("modules");
        return;
    }
    if (!ev.session_id) return;
    const ts = ev.ts ?? Date.now();
    const t = ev.t_out_ms ?? 0;
    const s = this.session(ev.session_id, ts);
    s.last_ts = Math.max(s.last_ts, ts);
    s.t_out_ms = Math.max(s.t_out_ms, t);
    switch (type) {
      case "session.started":
        s.label = d.label ?? ""; s.sample_rate = d.sample_rate; s.connected = true; s.started = ts;
        this.touch("sessions");
        break;
      case "session.ended":
        s.connected = false;
        this.touch("sessions");
        break;
      case "utterance.started":
        this.utt(s, d.utterance_id, t, ts);
        this.touch("utterances");
        break;
      case "text.delta": {
        const u = this.utt(s, d.utterance_id, t, ts);
        u.live_text += d.delta ?? "";
        this.touch("utterances");
        break;
      }
      case "text.normalized": {
        const u = this.utt(s, d.utterance_id, t, ts);
        u.norm_parts.push({ before: d.before ?? "", after: d.after ?? "", changed: !!d.changed });
        this.touch("utterances");
        break;
      }
      case "segment": {
        const u = this.utt(s, d.utterance_id, t, ts);
        for (const sp of d.spans ?? []) u.spans.push({ text: sp.text ?? "", lang: sp.lang ?? "en" });
        this.touch("utterances");
        break;
      }
      case "chunk.flushed": {
        const c = d as ChunkFlushedData;
        const u = this.utt(s, c.utterance_id ?? "?", t, ts);
        u.chunk_events.push(c);
        s.chunks.set(c.chunk_id ?? "?", { chunk_id: c.chunk_id ?? "?", utterance_id: c.utterance_id ?? "?", text: c.text ?? "", lang: c.lang ?? "en", reason: c.reason, t_flush: t });
        this.trimChunks(s);
        this.touch("utterances", "timeline");
        break;
      }
      case "synth.start": {
        const c = s.chunks.get(d.chunk_id);
        if (c) { c.t_synth_start = t; c.cached = d.cached; }
        this.touch("timeline");
        break;
      }
      case "synth.first_audio": {
        const c = s.chunks.get(d.chunk_id);
        if (c) c.t_first_audio = t;
        this.touch("timeline");
        break;
      }
      case "synth.done": {
        const c = s.chunks.get(d.chunk_id);
        if (c) c.t_synth_done = t;
        this.touch("timeline");
        break;
      }
      case "playback.chunk": {
        const c = s.chunks.get(d.chunk_id);
        if (c) { if (d.event === "start") c.t_play_start = d.t_out_ms ?? t; else c.t_play_end = d.t_out_ms ?? t; }
        this.touch("timeline");
        break;
      }
      case "playback.state":
        s.prev = d.prev; s.state = d.state ?? "idle"; s.reason = d.reason;
        s.states.push({ t, state: s.state });
        if (s.states.length > 2000) s.states.shift();
        this.touch("sessions", "timeline");
        break;
      case "audio.level": {
        const lv = d as AudioLevelData;
        const w = lv.wave ?? [];
        let mn = 0, mx = 0;
        for (let i = 0; i < w.length; i += 2) { mn = Math.min(mn, w[i]); mx = Math.max(mx, w[i + 1]); }
        s.levels.push({ t, min: mn, max: mx, rms: lv.rms ?? 0 });
        if (s.levels.length > MAX_LEVELS) s.levels.splice(0, s.levels.length - MAX_LEVELS);
        s.level = lv; s.gain_db = lv.gain_db ?? 0;
        this.touch("level", "timeline");
        break;
      }
      case "playback.progress": {
        const u = s.utts.get(d.utterance_id);
        if (u) u.progress_heard = d.heard_text;
        this.touch("utterances");
        break;
      }
      case "underrun":
        s.markers.push({ t, kind: "underrun", label: d.waiting_chunk_id ?? "" });
        this.touch("timeline");
        break;
      case "duck":
        s.ducked = d.action === "duck";
        s.markers.push({ t, kind: d.action === "duck" ? "duck" : "resume", label: d.source });
        this.touch("timeline", "sessions");
        break;
      case "utterance.interrupted": {
        s.last_interrupted = d;
        s.markers.push({ t, kind: "stop", label: d.source });
        const u = this.utt(s, d.utterance_id, t, ts);
        u.heard_text = d.heard_text; u.unplayed_text = d.unplayed_text; u.status = "interrupted";
        this.touch("timeline", "utterances", "sessions");
        break;
      }
      case "utterance.end": {
        const u = this.utt(s, d.utterance_id, t, ts);
        Object.assign(u, d as UtteranceSummary);
        u.status = d.interrupted ? "interrupted" : "final";
        this.touch("utterances");
        break;
      }
      case "shadow":
        s.shadow.push(d);
        if (s.shadow.length > 50) s.shadow.shift();
        this.touch("utterances");
        break;
      case "error":
        s.markers.push({ t, kind: "error", label: d.message });
        this.touch("timeline");
        break;
    }
    if (s.markers.length > 500) s.markers.shift();
  }

  /** Seed a session that started before this dashboard connected (GET /api/sessions/{id}). */
  hydrate(d: SessionDetail) {
    if (!d?.id) return;
    const s = this.session(d.id, d.started ?? Date.now());
    s.label = d.label ?? s.label;
    s.connected = d.connected ?? true;
    if (d.state) s.state = d.state;
    s.t_out_ms = Math.max(s.t_out_ms, d.t_out_ms ?? 0);
    for (const row of d.utterances ?? []) {
      if (!row.utterance_id || s.utts.has(row.utterance_id)) continue;
      const u = this.utt(s, row.utterance_id, 0, (row.created ?? Date.now() / 1000) * 1000);
      Object.assign(u, row);
      u.live_text = row.text ?? "";
      u.spans = (row.segments ?? []).map((x) => ({ text: x.text ?? "", lang: x.lang ?? "en" }));
      if (row.normalized && row.text && row.normalized !== row.text) u.norm_parts = [{ before: row.text, after: row.normalized, changed: true }];
    }
    this.touch("sessions", "utterances");
  }

  trimChunks(s: SessionView) {
    if (s.chunks.size > 400) {
      const first = s.chunks.keys().next().value;
      if (first !== undefined) s.chunks.delete(first);
    }
  }
}

export const store = new Store();

export function useTopic(...topics: Topic[]): number {
  return useSyncExternalStore(store.subscribe, () => topics.reduce((a, t) => a * 31 + store.versions[t], 7));
}
