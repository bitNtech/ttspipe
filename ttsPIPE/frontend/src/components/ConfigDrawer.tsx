import { useEffect, useState } from "react";
import { api, errMsg } from "../api/client";
import { refreshAll } from "../api/live";
import { store, useTopic } from "../state/store";
import type { Kind, ParamsSchema } from "../types/contract";
import { KINDS } from "../types/contract";
import { SchemaForm, StatusDot } from "./common";

// Mirrors config/pipeline.yaml defaults and backend config.PLAYBACK_KEYS / CHUNKER_KEYS ranges (§11).
const PLAYBACK_SCHEMA: ParamsSchema = {
  lookahead_chunks: { type: "integer", default: 2, min: 0, max: 8, description: "Chunks synthesized ahead of the one playing" },
  jitter_ms: { type: "integer", default: 60, min: 0, max: 1000, unit: "ms", description: "Audio buffered before a chunk starts" },
  duck_db: { type: "number", default: 12, min: 0, max: 60, unit: "dB" },
  duck_ramp_ms: { type: "integer", default: 80, min: 0, max: 2000, unit: "ms" },
  duck_release_ms: { type: "integer", default: 2000, min: 0, max: 60000, unit: "ms", description: "Auto-resume after a duck (0 = never)" },
  stop_fade_ms: { type: "integer", default: 60, min: 0, max: 1000, unit: "ms" },
  crossfade_ms: { type: "integer", default: 8, min: 0, max: 100, unit: "ms", description: "De-click fade at chunk joins" },
  max_silence_gap_ms: { type: "integer", default: 400, min: 0, max: 10000, unit: "ms" },
  fill_underrun_silence: { type: "boolean", default: true },
  level_every_ticks: { type: "integer", default: 2, min: 1, max: 50, advanced: true },
};
const CHUNKER_SCHEMA: ParamsSchema = {
  first_chunk_min_words: { type: "integer", default: 4, min: 1, max: 50, description: "First chunk flushes at a clause end once it has this many words" },
  first_chunk_max_words: { type: "integer", default: 10, min: 1, max: 100 },
  first_chunk_max_wait_ms: { type: "integer", default: 400, min: 0, max: 10000, unit: "ms" },
  min_chunk_chars: { type: "integer", default: 40, min: 1, max: 2000 },
  max_chunk_chars: { type: "integer", default: 220, min: 10, max: 5000 },
  nonstreaming_max_chunk_chars: { type: "integer", default: 140, min: 10, max: 5000, description: "Cap used when the TTS engine cannot stream" },
};
const defaultsOf = (s: ParamsSchema) => Object.fromEntries(Object.entries(s).map(([k, v]) => [k, v.default]));

type Tab = "stage" | "voice" | "playback" | "chunker";

export function ConfigDrawer({ initialKind, onClose }: { initialKind?: Kind; onClose: () => void }) {
  useTopic("pipeline", "modules");
  const [tab, setTab] = useState<Tab>("stage");
  const [kind, setKind] = useState<Kind>(initialKind ?? "tts");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const cfg = store.pipeline?.config ?? {};
  const st = store.pipeline?.stages?.[kind];
  const stCfg = cfg.stages?.[kind] ?? {};
  const [impl, setImpl] = useState(stCfg.impl ?? "");
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [pb, setPb] = useState<Record<string, unknown>>({});
  const [ch, setCh] = useState<Record<string, unknown>>({});
  const [rate, setRate] = useState(cfg.audio?.out_rate ?? 24000);
  useEffect(() => { if (initialKind) { setKind(initialKind); setTab("stage"); } }, [initialKind]);
  useEffect(() => { setImpl(stCfg.impl ?? ""); setParams({ ...(st?.params ?? {}) }); }, [kind, stCfg.impl, st?.params]);
  useEffect(() => { setPb({ ...defaultsOf(PLAYBACK_SCHEMA), ...(cfg.playback ?? {}) }); setCh({ ...defaultsOf(CHUNKER_SCHEMA), ...(cfg.chunker ?? {}) }); setRate(cfg.audio?.out_rate ?? 24000); }, [cfg.playback, cfg.chunker, cfg.audio?.out_rate]);
  const mod = store.modules.find((m) => m.id === `${kind}/${impl}`);
  const run = async (fn: () => Promise<unknown>, ok: string) => {
    try { await fn(); setMsg({ ok: true, text: ok }); refreshAll(); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  };
  return (
    <>
      <div className="drawer-back" onClick={onClose} />
      <aside className="drawer">
        <div className="drawer-head"><b>Config</b><span className="muted small">{store.pipeline?.active_profile ?? ""}{store.pipeline?.dirty ? " ● modified" : ""}</span><button className="icon-btn right" onClick={onClose}>✕</button></div>
        <div className="tabs">{(["stage", "voice", "playback", "chunker"] as Tab[]).map((t) => <div key={t} className={`tab${tab === t ? " on" : ""}`} onClick={() => { setTab(t); setMsg(null); }}>{t[0].toUpperCase() + t.slice(1)}</div>)}</div>
        <div className="drawer-body col">
          {msg && <div className={`notice small ${msg.ok ? "ok" : "err"}`}>{msg.text}</div>}
          {tab === "stage" && (
            <>
              <div className="row wrap">
                <select value={kind} onChange={(e) => setKind(e.target.value as Kind)}>{KINDS.map((k) => <option key={k}>{k}</option>)}</select>
                <StatusDot st={st} /><span className="mono small">{st?.impl}</span>
                {st?.load_ms != null && <span className="muted small">loaded in {Math.round(st.load_ms)} ms</span>}
              </div>
              {st?.error && <div className="notice err small">{st.error}</div>}
              <div className="form">
                <label>impl</label>
                <select value={impl} onChange={(e) => { setImpl(e.target.value); setParams({}); }}>
                  {store.modules.filter((m) => m.kind === kind).map((m) => <option key={m.id} value={m.id!.split("/")[1]}>{m.id!.split("/")[1]}{m.available === false ? " (unavailable)" : ""}</option>)}
                </select>
                {!["ingest", "chunker", "tts"].includes(kind) && <>
                  <label>enabled</label>
                  <input type="checkbox" checked={stCfg.enabled !== false} onChange={(e) => run(() => api.putStage(kind, { enabled: e.target.checked }), "Saved.")} />
                </>}
              </div>
              {mod?.available === false && <div className="notice warn small">{mod.reason} — the builtin will stand in.</div>}
              <div className="muted small">{mod?.description}</div>
              <div className="section-title">Parameters</div>
              <SchemaForm schema={mod?.params_schema ?? {}} values={{ ...(mod?.default_params ?? {}), ...params }} onChange={(n, v) => setParams((p) => ({ ...p, [n]: v }))} />
              <div className="row">
                <button className="btn primary sm" onClick={() => run(() => api.putStage(kind, { impl, params }), "Applied (hot swap in background).")}>Apply</button>
                <button className="btn sm" onClick={() => run(() => api.putStage(kind, { impl, params: {} }), "Reset to defaults.")}>Reset to defaults</button>
              </div>
            </>
          )}
          {tab === "voice" && <VoiceForm run={run} />}
          {tab === "playback" && (
            <>
              <div className="form">
                <label>out_rate</label>
                <select value={rate} onChange={(e) => setRate(+e.target.value)}>{[16000, 22050, 24000, 48000].map((r) => <option key={r} value={r}>{r} Hz</option>)}</select>
                <div className="desc">Applies to sessions opened after the change.</div>
              </div>
              <SchemaForm schema={PLAYBACK_SCHEMA} values={pb} onChange={(n, v) => setPb((p) => ({ ...p, [n]: v }))} />
              <div className="row">
                <button className="btn primary sm" onClick={() => run(() => api.putPlayback({ ...pb, out_rate: rate }), "Playback saved.")}>Apply</button>
                <button className="btn sm" onClick={() => run(() => api.putPlayback({ ...defaultsOf(PLAYBACK_SCHEMA), out_rate: 24000 }), "Playback reset.")}>Reset to defaults</button>
              </div>
            </>
          )}
          {tab === "chunker" && (
            <>
              <SchemaForm schema={CHUNKER_SCHEMA} values={ch} onChange={(n, v) => setCh((p) => ({ ...p, [n]: v }))} />
              <div className="row">
                <button className="btn primary sm" onClick={() => run(() => api.putChunker(ch), "Chunker saved.")}>Apply</button>
                <button className="btn sm" onClick={() => run(() => api.putChunker(defaultsOf(CHUNKER_SCHEMA)), "Chunker reset.")}>Reset to defaults</button>
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  );
}

function VoiceForm({ run }: { run: (fn: () => Promise<unknown>, ok: string) => void }) {
  const cfg = store.pipeline?.config ?? {};
  const impl = store.pipeline?.stages?.tts?.impl ?? cfg.stages?.tts?.impl ?? "";
  const mod = store.modules.find((m) => m.id === `tts/${impl}`);
  const [v, setV] = useState<Record<string, unknown>>({ voice_id: cfg.voice?.voice_id ?? "", speed: cfg.voice?.speed ?? 1, pitch: cfg.voice?.pitch ?? 0 });
  const schema: ParamsSchema = {
    voice_id: { type: "string", default: mod?.voices?.[0]?.id ?? "", enum: (mod?.voices ?? []).map((x) => x.id ?? "") },
    speed: { type: "number", default: 1, min: 0.5, max: 2 },
    pitch: { type: "number", default: 0, min: -12, max: 12, unit: "st", description: "Ignored by engines without pitch control" },
  };
  if (!schema.voice_id.enum?.length) delete schema.voice_id.enum;
  return (
    <>
      <div className="small">Engine <span className="mono">{impl}</span></div>
      <SchemaForm schema={schema} values={v} onChange={(n, x) => setV((p) => ({ ...p, [n]: x }))} />
      <div className="row">
        <button className="btn primary sm" onClick={() => run(() => api.putVoice({ impl, ...v }), "Voice saved.")}>Apply</button>
        <button className="btn sm" onClick={() => run(() => api.putVoice({ impl, voice_id: String(schema.voice_id.default), speed: 1, pitch: 0 }), "Voice reset.")}>Reset to defaults</button>
      </div>
    </>
  );
}
