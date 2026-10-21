// TTS engine setup: every engine with its status, model files, parameters and voices; switch the pipeline's
// engine, download models, preview voices, and run the Fish S2-Pro GPU server.
import { useEffect, useMemo, useState } from "react";
import { api, errMsg } from "../api/client";
import { refreshAll } from "../api/live";
import { SchemaForm, StatusDot, fmt, fmtMs } from "../components/common";
import { store, useTopic } from "../state/store";
import type { ModelEntry, ModuleEntry, ProfileEntry } from "../types/contract";
import { applyPipelineVoice, FISH, FishServerCard, implOf, pipelineTts, sampleText, ttsModules, useFish } from "./shared";

let lastSelected = "";

export function EnginesPage({ onNavigate }: { onNavigate: (page: string) => void }) {
  useTopic("pipeline", "modules", "metrics");
  const pipeImpl = pipelineTts();
  const [sel, setSel] = useState(lastSelected);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [profiles, setProfiles] = useState<ProfileEntry[]>([]);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  useEffect(() => { if (!sel && pipeImpl) setSel(pipeImpl); lastSelected = sel; }, [sel, pipeImpl]);
  const loadModels = () => api.models().then(setModels).catch(() => undefined);
  useEffect(() => { loadModels(); api.profiles().then(setProfiles).catch(() => undefined); }, [store.connected]);
  const downloading = Object.values(store.downloads).some((d) => d.status === "running");
  useEffect(() => { if (!downloading) loadModels(); }, [downloading]);

  const stage = store.pipeline?.stages?.tts;
  const cfgVoice = store.pipeline?.config?.voice;
  const mods = ttsModules();
  const mod = mods.find((m) => implOf(m) === sel);
  const tm = store.metrics?.stages?.tts;
  const run = async (fn: () => Promise<unknown>, ok: string) => {
    try { await fn(); setMsg({ ok: true, text: ok }); await refreshAll(); loadModels(); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  };

  return (
    <div className="page">
      <section className="card">
        <h3>Pipeline engine</h3>
        <div className="row wrap" style={{ gap: 12 }}>
          <span className="row"><StatusDot st={stage} /><b className="mono">{stage?.impl ?? "—"}</b>{stage?.configured_impl && stage.configured_impl !== stage.impl && <span className="badge err">configured {stage.configured_impl}</span>}</span>
          <span className="small">status <b>{stage?.status ?? "—"}</b></span>
          <span className="small">voice <b className="mono">{cfgVoice?.voice_id || "default"}</b> · {fmt(cfgVoice?.speed ?? 1, 2)}×</span>
          {stage?.load_ms != null && <span className="small muted">loaded in {fmtMs(stage.load_ms)}</span>}
          {tm?.p50_ms != null && <span className="small muted">synth p50 {fmtMs(tm.p50_ms)} · p95 {fmtMs(tm.p95_ms)}</span>}
          <span className="right row">
            <span className="small muted">Profile</span>
            <select value="" onChange={(e) => e.target.value && run(() => api.applyProfile(e.target.value), `Profile ${e.target.value} applied.`)}>
              <option value="">{store.pipeline?.active_profile ?? "custom"}{store.pipeline?.dirty ? " (modified)" : ""} — apply…</option>
              {profiles.map((p) => <option key={p.name} value={p.name} title={p.description}>{p.name}{p.builtin ? "" : " (user)"}</option>)}
            </select>
            <button className="btn sm" onClick={() => onNavigate("test")}>Test it →</button>
          </span>
        </div>
        {stage?.status === "loading" && <div className="notice small" style={{ marginTop: 8 }}>Loading {stage.configured_impl}… sessions keep the previous engine until it is ready.</div>}
        {stage?.error && <div className="notice err small" style={{ marginTop: 8 }}>{stage.error}</div>}
        {msg && <div className={`notice small ${msg.ok ? "ok" : "err"}`} style={{ marginTop: 8 }}>{msg.text}</div>}
      </section>

      <div className="engines">
        <div className="col" style={{ gap: 6 }}>
          {mods.map((m) => <EngineTile key={m.id} m={m} active={implOf(m) === pipeImpl} selected={implOf(m) === sel} model={models.find((x) => x.id === m.id)} onClick={() => setSel(implOf(m))} />)}
        </div>
        {mod ? <EngineDetail key={mod.id} m={mod} active={implOf(mod) === pipeImpl} model={models.find((x) => x.id === mod.id)} run={run} onNavigate={onNavigate} /> : <section className="card muted">Select an engine.</section>}
      </div>
    </div>
  );
}

function EngineTile({ m, active, selected, model, onClick }: { m: ModuleEntry; active: boolean; selected: boolean; model?: ModelEntry; onClick: () => void }) {
  const st = active ? store.pipeline?.stages?.tts : null;
  return (
    <div className={`tile${selected ? " sel" : ""}${m.available === false ? " unavail" : ""}`} onClick={onClick}>
      <div className="row">
        {active ? <StatusDot st={st} /> : <span className={`dot ${m.available === false ? "error" : "unloaded"}`} />}
        <b>{m.name}</b>
        {active && <span className="badge accent right">pipeline</span>}
      </div>
      <div className="row wrap small" style={{ gap: 4, marginTop: 3 }}>
        <span className="mono muted">{implOf(m)}</span>
        {(m.languages ?? []).map((l) => <span key={l} className="badge">{l}</span>)}
        {m.tags?.includes("gpu") && <span className="badge accent">GPU</span>}
        {m.tags?.includes("cloning") && <span className="badge accent">voice cloning</span>}
        {m.builtin && <span className="badge">builtin</span>}
        {m.available === false && <span className="badge err">unavailable</span>}
        {model && !model.present && m.available !== false && <span className="badge warn">models missing</span>}
      </div>
    </div>
  );
}

function EngineDetail({ m, active, model, run, onNavigate }: {
  m: ModuleEntry; active: boolean; model?: ModelEntry; run: (fn: () => Promise<unknown>, ok: string) => Promise<void>; onNavigate: (p: string) => void;
}) {
  const impl = implOf(m);
  const stage = store.pipeline?.stages?.tts;
  const cfgVoice = store.pipeline?.config?.voice;
  const base = useMemo(() => ({ ...(m.default_params ?? {}), ...(active ? stage?.params ?? {} : {}) }), [m, active, stage?.params]);
  const [params, setParams] = useState<Record<string, unknown>>(base);
  useEffect(() => setParams(base), [base]);
  const voices = m.voices ?? [];
  const [voice, setVoice] = useState(active && cfgVoice?.voice_id ? cfgVoice.voice_id : voices[0]?.id ?? "");
  const [speed, setSpeed] = useState(active ? cfgVoice?.speed ?? 1 : 1);
  const [text, setText] = useState(sampleText(m.languages?.includes("ta") ? (m.languages.includes("en") ? undefined : "ta") : "en"));
  const [audio, setAudio] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewErr, setPreviewErr] = useState<string | null>(null);
  const [fish, refreshFish] = useFish(impl === FISH ? 2000 : 60000);
  const dl = store.downloads[m.id ?? ""];
  const dirty = JSON.stringify(params) !== JSON.stringify(base);

  const preview = async () => {
    setPreviewing(true); setPreviewErr(null);
    try {
      const blob = await api.preview(impl, { voice_id: voice, speed, text, ...(dirty || !active ? { params } : {}) });
      setAudio(URL.createObjectURL(blob));
    } catch (e) { setPreviewErr(errMsg(e)); }
    setPreviewing(false);
  };

  return (
    <section className="card">
      <h3>{m.name} <span className="mono muted small">tts/{impl}</span></h3>
      <div className="muted">{m.description}</div>
      <div className="kv small" style={{ marginTop: 8 }}>
        <span className="k">Licence</span><span className="v">{m.license}</span><span />
        {m.model_id && <><span className="k">Model</span><span className="v">{m.model_id}</span><span /></>}
        <span className="k">Languages</span><span className="v">{(m.languages ?? []).join(", ") || "any"}</span><span />
        <span className="k">Output</span><span className="v">{m.native_rate ? `${m.native_rate} Hz` : "—"} · {m.streaming ? "streams audio" : "whole chunk per call"}</span><span />
        <span className="k">Size / RAM</span><span className="v">{m.size_mb ? `${m.size_mb} MB` : "—"} / {m.ram_mb ? `${m.ram_mb} MB` : "—"}</span><span />
      </div>

      {m.available === false && (
        <div className="notice warn small" style={{ marginTop: 8 }}>
          {m.reason}. Install with <span className="mono">./run.sh models {m.family || impl}</span> and restart the backend.
        </div>
      )}

      {impl === FISH && (
        <>
          <div className="section-title">Fish S2-Pro GPU server</div>
          <FishServerCard st={fish} refresh={refreshFish} />
        </>
      )}

      {model && (
        <>
          <div className="section-title">Model files</div>
          <table className="tbl">
            <tbody>
              {(model.files ?? []).map((f) => (
                <tr key={f.path}><td>{f.present ? <span style={{ color: "var(--ok)" }}>✓</span> : <span style={{ color: "var(--warn)" }}>missing</span>}</td><td className="mono small">{f.name}</td><td className="n">{f.size_mb ? `${fmt(f.size_mb, 0)} MB` : ""}</td></tr>
              ))}
            </tbody>
          </table>
          {!model.present && (
            <div className="row" style={{ marginTop: 6 }}>
              <button className="btn sm" disabled={dl?.status === "running"} onClick={() => run(() => api.download("tts", impl), "Download started.")}>Download {fmt(model.size_mb, 0)} MB</button>
              {dl && <span className="small mono">{dl.status} {fmt(dl.done_mb, 0)}/{fmt(dl.total_mb, 0)} MB {dl.error ?? ""}</span>}
            </div>
          )}
        </>
      )}

      <div className="section-title">Parameters {active ? "(live pipeline)" : ""}</div>
      <SchemaForm schema={m.params_schema ?? {}} values={params} onChange={(n, v) => setParams((p) => ({ ...p, [n]: v }))} />
      <div className="row wrap" style={{ marginTop: 8 }}>
        {active ? (
          <>
            <button className="btn primary sm" disabled={!dirty} onClick={() => run(() => api.putStage("tts", { impl, params }), "Parameters applied (hot swap in the background).")}>Apply</button>
            <button className="btn sm" onClick={() => run(() => api.loadModule("tts", impl), `Reloading ${impl}.`)}>Reload engine</button>
          </>
        ) : (
          <button className="btn primary sm" disabled={m.available === false} onClick={() => run(async () => { await api.putStage("tts", { impl, params }); if (voice) await api.putVoice({ impl, voice_id: voice, speed }); }, `Pipeline switched to ${impl}; loading in the background.`)}>Use as pipeline engine</button>
        )}
        <button className="btn sm" onClick={() => setParams({ ...(m.default_params ?? {}) })}>Defaults</button>
        {dirty && <span className="badge warn">unsaved changes</span>}
      </div>

      <div className="section-title">Voice &amp; preview</div>
      <div className="form">
        <label>Voice</label>
        <div className="row">
          <select className="grow" value={voice} onChange={(e) => setVoice(e.target.value)}>
            {voices.map((v) => <option key={v.id} value={v.id}>{v.name ?? v.id}{v.language ? ` · ${v.language}` : ""}{v.gender && v.gender !== "neutral" ? ` · ${v.gender}` : ""}</option>)}
            {!voices.length && <option value="">(engine default)</option>}
          </select>
          {m.tags?.includes("cloning") && <button className="btn sm" onClick={() => onNavigate("voices")}>Clone a voice…</button>}
        </div>
        <label>Speed</label>
        <div className="ctl"><input type="range" min={0.5} max={2} step={0.05} value={speed} onChange={(e) => setSpeed(+e.target.value)} /><span className="mono">{speed.toFixed(2)}×</span></div>
        <label>Text</label>
        <input type="text" value={text} onChange={(e) => setText(e.target.value)} />
      </div>
      <div className="row wrap" style={{ marginTop: 8 }}>
        <button className="btn sm" disabled={previewing || m.available === false} onClick={preview}>{previewing ? "Rendering…" : "▶ Preview"}</button>
        <button className="btn sm" disabled={m.available === false} onClick={() => run(() => applyPipelineVoice(impl, voice, { speed }), `Pipeline voice set to ${voice}.`)}>{active ? "Set as pipeline voice" : "Use engine + voice in pipeline"}</button>
        {active && cfgVoice?.voice_id === voice && <span className="badge ok">current pipeline voice</span>}
      </div>
      {impl === FISH && fish?.health !== "ok" && <div className="notice small" style={{ marginTop: 6 }}>Previews need the Fish server running (it starts automatically when the engine loads, if autostart is on).</div>}
      {previewErr && <div className="notice err small" style={{ marginTop: 6 }}>{previewErr}</div>}
      {audio && <audio controls autoPlay src={audio} style={{ marginTop: 6 }} />}
    </section>
  );
}
