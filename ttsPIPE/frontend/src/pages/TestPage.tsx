// TTS test bench: type or pick text, choose engine + voice, then speak it live through the real-time pipeline or
// render it to a WAV; every run is kept with its latency metrics and audio.
import { useEffect, useMemo, useState } from "react";
import { api, errMsg, MOCK } from "../api/client";
import { speaker } from "../api/live";
import { SchemaForm, StatusDot } from "../components/common";
import { store, useTopic, type UttView } from "../state/store";
import type { CompareResult, Script } from "../types/contract";
import { applyPipelineVoice, download, implOf, MetricChips, pipelineTts, ttsModule, ttsModules } from "./shared";

interface Run {
  id: string; mode: "live" | "render"; text: string; impl: string; voice: string; t: number;
  status: "running" | "done" | "error"; result?: CompareResult; error?: string;
}
let RUNS: Run[] = []; // survives page switches

const findUtt = (uid: string): UttView | undefined => {
  for (const s of store.sessions.values()) { const u = s.utts.get(uid); if (u) return u; }
  return undefined;
};

export function TestPage({ onNavigate }: { onNavigate: (page: string) => void }) {
  useTopic("pipeline", "modules", "utterances", "conn");
  const [text, setText] = useState("வணக்கம்! Your order has shipped and will arrive by Thursday. நன்றி!");
  const [scripts, setScripts] = useState<Script[]>([]);
  const [mode, setMode] = useState<"live" | "render">("live");
  const [llm, setLlm] = useState(true);
  const [impl, setImpl] = useState("");
  const [voice, setVoice] = useState("");
  const [speed, setSpeed] = useState(1);
  const [pitch, setPitch] = useState(0);
  const [params, setParams] = useState<Record<string, unknown>>({});
  const [showParams, setShowParams] = useState(false);
  const [runs, setRuns] = useState<Run[]>(RUNS);
  const [busy, setBusy] = useState(false);
  const save = (r: Run[]) => { RUNS = r.slice(0, 40); setRuns(RUNS); };
  const patch = (id: string, p: Partial<Run>) => save(RUNS.map((r) => (r.id === id ? { ...r, ...p } : r)));

  useEffect(() => { api.scripts().then(setScripts).catch(() => undefined); }, [store.connected]);
  // default to the pipeline's engine and voice once they are known
  const pipeImpl = pipelineTts();
  const pipeVoice = store.pipeline?.config?.voice;
  useEffect(() => {
    if (!impl && pipeImpl) {
      setImpl(pipeImpl);
      setVoice(pipeVoice?.voice_id ?? "");
      setSpeed(pipeVoice?.speed ?? 1);
      setPitch(pipeVoice?.pitch ?? 0);
    }
  }, [pipeImpl, impl, pipeVoice?.voice_id, pipeVoice?.speed, pipeVoice?.pitch]);
  const mod = ttsModule(impl);
  const voices = mod?.voices ?? [];
  useEffect(() => { if (voices.length && !voices.some((v) => v.id === voice)) setVoice(voices[0].id ?? ""); }, [impl, voices.length]); // eslint-disable-line react-hooks/exhaustive-deps
  const baseParams = useMemo(() => ({ ...(mod?.default_params ?? {}), ...(impl === pipeImpl ? store.pipeline?.stages?.tts?.params ?? {} : {}) }), [mod, impl, pipeImpl]);
  useEffect(() => setParams(baseParams), [baseParams]);
  const stage = store.pipeline?.stages?.tts;

  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const pipelineDiffers = impl !== pipeImpl || voice !== (pipeVoice?.voice_id ?? "") || speed !== (pipeVoice?.speed ?? 1) || pitch !== (pipeVoice?.pitch ?? 0);

  const run = async () => {
    if (!text.trim() || !impl) return;
    const id = `r${Date.now().toString(36)}`;
    const r: Run = { id, mode, text, impl, voice, t: Date.now(), status: "running" };
    save([r, ...RUNS]);
    setBusy(true);
    try {
      if (mode === "render") {
        const res = await api.render(text, {
          label: `${impl} · ${voice}`,
          config: { stages: { tts: { impl, params } }, voice: { impl, voice_id: voice, speed, pitch } },
        });
        patch(id, { status: res.error ? "error" : "done", result: res, error: res.error ?? undefined });
      } else {
        if (pipelineDiffers) {
          await applyPipelineVoice(impl, voice, { speed, pitch });
          store.toast(`Pipeline now uses ${impl} · ${voice}`);
        }
        await speaker.voice(`${impl}:${voice}`, speed, pitch);
        await speaker.speak(text, llm, id);
        patch(id, { status: "done" });
      }
    } catch (e) {
      patch(id, { status: "error", error: errMsg(e) });
    }
    setBusy(false);
  };

  return (
    <div className="page">
      <div className="page-grid">
        <div className="col" style={{ gap: 12 }}>
          <section className="card">
            <h3>Text <span className="muted small">{words} words · {text.length} chars</span></h3>
            <textarea className="bigtext" rows={6} value={text} onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); }} placeholder="Type Tamil, English or Tanglish…" />
            <div className="row wrap">
              <select value="" onChange={(e) => { const s = scripts.find((x) => x.id === e.target.value); if (s?.text) setText(s.text); }}>
                <option value="">Load a sample script…</option>
                {scripts.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
              </select>
              <button className="btn sm" onClick={() => setText("")}>Clear</button>
              <span className="muted small right">Ctrl+Enter to run</span>
            </div>
          </section>

          <section className="card">
            <h3>Engine &amp; voice</h3>
            <div className="form">
              <label>Engine</label>
              <div className="row">
                <select className="grow" value={impl} onChange={(e) => setImpl(e.target.value)}>
                  {ttsModules().map((m) => (
                    <option key={m.id} value={implOf(m)} disabled={m.available === false}>
                      {m.name} ({implOf(m)}){implOf(m) === pipeImpl ? " · pipeline" : ""}{m.available === false ? " · unavailable" : ""}
                    </option>
                  ))}
                </select>
                {impl === pipeImpl && <StatusDot st={stage} />}
              </div>
              {mod?.description && <div className="desc">{mod.description}</div>}
              <label>Voice</label>
              <div className="row">
                <select className="grow" value={voice} onChange={(e) => setVoice(e.target.value)}>
                  {voices.map((v) => <option key={v.id} value={v.id}>{v.name ?? v.id}{v.language ? ` · ${v.language}` : ""}</option>)}
                  {!voices.some((v) => v.id === voice) && <option value={voice}>{voice || "(default)"}</option>}
                </select>
                {mod?.tags?.includes("cloning") && <button className="btn sm" onClick={() => onNavigate("voices")}>Clone a voice…</button>}
              </div>
              <label>Speed</label>
              <div className="ctl"><input type="range" min={0.5} max={2} step={0.05} value={speed} onChange={(e) => setSpeed(+e.target.value)} /><span className="mono">{speed.toFixed(2)}×</span></div>
              <label>Pitch</label>
              <div className="ctl"><input type="range" min={-12} max={12} step={0.5} value={pitch} onChange={(e) => setPitch(+e.target.value)} /><span className="mono">{pitch.toFixed(1)} st</span></div>
              {impl === "fish_s2" && <div className="desc">Fish S2-Pro has no speed/pitch control; it ignores both.</div>}
            </div>
            {impl === pipeImpl && stage?.status === "loading" && <div className="notice small" style={{ marginTop: 8 }}>The pipeline is loading {impl}; live runs start once it is ready.</div>}
            {impl === pipeImpl && stage?.fallback && <div className="notice err small" style={{ marginTop: 8 }}>{impl} failed to load ({stage.error}); the builtin {stage.impl} is speaking instead. <button className="btn sm" onClick={() => onNavigate("engines")}>Open Engines</button></div>}
            {mod?.available === false && <div className="notice warn small" style={{ marginTop: 8 }}>{mod.reason}</div>}
            <div style={{ marginTop: 8 }}>
              <button className="btn sm" onClick={() => setShowParams(!showParams)}>{showParams ? "Hide" : "Show"} engine parameters</button>
              {showParams && (
                <div style={{ marginTop: 6 }}>
                  <div className="muted small" style={{ marginBottom: 4 }}>Used by <b>Render</b> runs only. To change the live pipeline, use the Engines page.</div>
                  <SchemaForm schema={mod?.params_schema ?? {}} values={params} onChange={(n, v) => setParams((p) => ({ ...p, [n]: v }))} />
                </div>
              )}
            </div>
          </section>

          <section className="card">
            <h3>Run</h3>
            <div className="seg">
              <button className={mode === "live" ? "on" : ""} onClick={() => setMode("live")}>Live (real-time pipeline)</button>
              <button className={mode === "render" ? "on" : ""} onClick={() => setMode("render")}>Render to WAV</button>
            </div>
            <div className="muted small" style={{ margin: "6px 0" }}>
              {mode === "live"
                ? "Streams through the real pipeline and plays here, paced in real time, with full metrics on the Monitor page. Runs with a different engine or voice switch the pipeline to it first."
                : "Renders offline with the engine, voice and parameters above (the live pipeline is not changed) and keeps a WAV you can replay or download."}
            </div>
            <div className="row wrap">
              <button className="btn primary" disabled={busy || !text.trim() || !impl || mod?.available === false} onClick={run}>{mode === "live" ? "▶ Speak" : "Render"}</button>
              {mode === "live" && <>
                <label className="check small"><input type="checkbox" checked={llm} onChange={(e) => setLlm(e.target.checked)} />Stream as LLM tokens</label>
                <button className="btn danger" onClick={() => speaker.control("stop")}>Stop</button>
                <button className="btn" onClick={() => speaker.control("duck")}>Duck</button>
                <button className="btn" onClick={() => speaker.control("resume")}>Resume</button>
              </>}
              {mode === "live" && pipelineDiffers && impl && <span className="badge warn">will switch pipeline to {impl} · {voice || "default"}</span>}
            </div>
          </section>
        </div>

        <section className="card">
          <h3>Runs <span className="muted small">newest first</span>{runs.length > 0 && <button className="btn sm right" onClick={() => save([])}>Clear</button>}</h3>
          {!runs.length && <div className="muted small">Nothing yet. Pick an engine and press Speak or Render.</div>}
          {runs.map((r) => <RunRow key={r.id} r={r} onNavigate={onNavigate} />)}
        </section>
      </div>
    </div>
  );
}

function RunRow({ r, onNavigate }: { r: Run; onNavigate: (page: string) => void }) {
  const u = r.mode === "live" ? findUtt(r.id) : undefined;
  const lat = u?.latency;
  const liveMetrics = lat ? { ttfa_ms: lat.ttfa_ms, first_synth_ms: lat.first_synth_ms, audio_ms: lat.audio_ms, rtf: lat.rtf, chunks: u?.chunks?.length } : null;
  const status = r.status === "running" ? "running" : r.status === "error" ? "error" : r.mode === "live" ? (u?.status ?? (MOCK ? "sent" : "queued")) : "done";
  const url = r.result?.url ?? (u?.render_id ? `/api/renders/${u.render_id}.wav` : null);
  return (
    <div className="run">
      <div className="row wrap">
        <span className={`badge ${r.mode === "live" ? "accent" : ""}`}>{r.mode}</span>
        <span className="mono small">{r.impl} · {r.voice || "default"}</span>
        <span className={`badge ${status === "error" || status === "interrupted" ? "err" : status === "final" || status === "done" ? "ok" : ""}`}>{status}</span>
        <span className="muted small right">{new Date(r.t).toLocaleTimeString()}</span>
      </div>
      <div className="run-text">{r.text}</div>
      {r.error && <div className="notice err small">{r.error}</div>}
      {r.result?.tts_impl && r.result.tts_impl !== r.impl && <div className="notice warn small">{r.impl} was unavailable; rendered with {r.result.tts_impl}.</div>}
      <MetricChips m={r.result?.metrics ?? liveMetrics} />
      {u?.interrupted && <div className="small">Heard: <b>{u.heard_text || "—"}</b> <span className="muted">· unplayed: {u.unplayed_text || "—"}</span></div>}
      {url && (
        <div className="row">
          <audio controls src={url} preload="none" className="grow" />
          <button className="btn sm" onClick={() => download(url, `ttspipe-${r.impl}-${r.id}.wav`)}>Download</button>
        </div>
      )}
      {r.mode === "live" && r.status === "done" && <button className="btn sm" onClick={() => onNavigate("monitor")}>Details on Monitor</button>}
      {r.mode === "live" && lat?.ttfa_ms == null && r.status === "done" && u?.status !== "final" && <div className="muted small">Metrics appear when the utterance finishes.</div>}
    </div>
  );
}
