import { useEffect, useRef, useState } from "react";
import { api, errMsg, MOCK } from "../api/client";
import { speaker } from "../api/live";
import { store, useTopic } from "../state/store";
import type { LayoutPanel, Script, VoiceEntry } from "../types/contract";

export const PAGES: [string, string][] = [["test", "Test"], ["engines", "Engines"], ["voices", "Voice cloning"], ["monitor", "Monitor"]];

interface Props {
  page: string;
  onPage: (page: string) => void;
  panels: LayoutPanel[];
  titles: Record<string, string>;
  editing: boolean;
  preset: string;
  presets: string[];
  onPreset: (name: string) => void;
  onTogglePanel: (id: string) => void;
  onEdit: () => void;
  onReset: () => void;
  onConfig: () => void;
}

function useOutside(ref: React.RefObject<HTMLElement | null>, close: () => void) {
  useEffect(() => {
    const h = (e: MouseEvent) => { if (ref.current && !ref.current.contains(e.target as Node)) close(); };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, [ref, close]);
}

export function TopBar(p: Props) {
  useTopic("conn", "metrics", "sessions", "level", "pipeline");
  const [text, setText] = useState("வணக்கம்! Your order has shipped and will arrive by Thursday.");
  const [llm, setLlm] = useState(true);
  const [scripts, setScripts] = useState<Script[]>([]);
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const [panelsOpen, setPanelsOpen] = useState(false);
  const [theme, setTheme] = useState(() => { try { return localStorage.getItem("ttspipe.theme") ?? "light"; } catch { return "light"; } });
  const [vol, setVol] = useState(0.9);
  const menuRef = useRef<HTMLDivElement>(null);
  useOutside(menuRef, () => setPanelsOpen(false));
  useEffect(() => { api.scripts().then(setScripts).catch(() => undefined); api.voices().then(setVoices).catch(() => undefined); }, [store.connected]);
  useEffect(() => { document.documentElement.dataset.theme = theme; try { localStorage.setItem("ttspipe.theme", theme); } catch { /* ignore */ } }, [theme]);

  const m = store.metrics;
  const cur = store.current();
  const lv = cur?.level;
  const ttsImpl = store.pipeline?.stages?.tts?.impl ?? "";
  const voiceId = store.pipeline?.config?.voice?.voice_id ?? "";
  const speak = () => text.trim() && speaker.speak(text, llm).catch((e) => store.toast(errMsg(e)));
  const sessions = [...store.sessions.values()].sort((a, b) => b.last_ts - a.last_ts);
  const peakPct = Math.min(100, Math.max(0, ((lv?.rms_db ?? -120) + 60) * (100 / 60)));
  return (
    <div className="topbar">
      <div className="logo"><svg width="20" height="20" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="var(--accent)" /><path d="M6 16h3l2-6 3 12 3-9 2 5 2-2h5" stroke="#fff" strokeWidth="2.4" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>tts<b>PIPE</b></div>
      <span className="chip" title="Event stream connection">
        <span className={`dot ${store.connected ? (MOCK ? "loading" : "ready") : "error"}`} />{MOCK ? "mock" : store.connected ? "live" : "offline"}
        {store.rtt != null && !MOCK && <span className="mono">{store.rtt}ms</span>}
      </span>
      <nav className="nav">
        {PAGES.map(([id, label]) => <a key={id} href={`#/${id}`} className={p.page === id ? "on" : ""} onClick={(e) => { e.preventDefault(); p.onPage(id); }}>{label}</a>)}
      </nav>
      <span className="chip mono" title="Server CPU / RSS">CPU {m?.system?.cpu_pct != null ? Math.round(m.system.cpu_pct) : "—"}% · {m?.system?.rss_mb != null ? Math.round(m.system.rss_mb) : "—"}MB</span>
      {p.page === "monitor" && <>
      <select value={store.selected} onChange={(e) => { store.selected = e.target.value; store.touch("sessions", "utterances", "timeline", "level"); }} title="Session shown in panels" style={{ maxWidth: 220 }}>
        <option value="auto">Latest (auto){cur ? ` — ${cur.label || cur.id}` : ""}</option>
        {sessions.map((s) => <option key={s.id} value={s.id}>{s.label ? `${s.label} · ` : ""}{s.id}{s.connected ? "" : " (closed)"}</option>)}
      </select>
      <div className="speakbox">
        <input type="text" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && speak()} placeholder="Text to speak…" />
        <button className="btn primary" onClick={speak}>Speak</button>
        <label className="check small" title="Replay the text as LLM token deltas"><input type="checkbox" checked={llm} onChange={(e) => setLlm(e.target.checked)} />Stream as LLM</label>
      </div>
      <select value="" onChange={(e) => { const s = scripts.find((x) => x.id === e.target.value); if (s?.text) setText(s.text); }} title="Sample scripts">
        <option value="">Sample script…</option>
        {scripts.map((s) => <option key={s.id} value={s.id}>{s.title}</option>)}
      </select>
      <select value={voiceId} title="Voice" onChange={(e) => { api.putVoice({ impl: ttsImpl, voice_id: e.target.value }).catch((er) => store.toast(errMsg(er))); speaker.voice(`${ttsImpl}:${e.target.value}`); }}>
        {voices.filter((v) => v.impl === ttsImpl).map((v) => <option key={v.id} value={v.id}>{v.name ?? v.id}</option>)}
        {!voices.some((v) => v.impl === ttsImpl && v.id === voiceId) && <option value={voiceId}>{voiceId || "voice"}</option>}
      </select>
      </>}
      {p.page !== "monitor" && <span className="grow" />}
      <button className="btn danger" onClick={() => speaker.control("stop")}>Stop</button>
      <button className={`btn${cur?.ducked ? " on" : ""}`} onClick={() => speaker.control("duck")}>Duck</button>
      <button className="btn" onClick={() => speaker.control("resume")}>Resume</button>
      <span className="row" title={`Output level ${lv?.rms_db ?? -120} dBFS; slider = local playback volume`}>
        <span className="meter"><i style={{ width: `${peakPct}%` }} /></span>
        <input type="range" min={0} max={1} step={0.05} value={vol} style={{ width: 60 }} onChange={(e) => { setVol(+e.target.value); speaker.setVolume(+e.target.value); }} />
      </span>
      <span className="sep" />
      {p.page === "monitor" && <>
      <select value={p.preset} onChange={(e) => p.onPreset(e.target.value)} title="Layout preset">
        {p.presets.map((x) => <option key={x} value={x}>{x[0].toUpperCase() + x.slice(1)}</option>)}
      </select>
      <div className="dropdown" ref={menuRef}>
        <button className="btn" onClick={() => setPanelsOpen(!panelsOpen)}>Panels ▾</button>
        {panelsOpen && (
          <div className="menu" style={{ right: 0 }}>
            {p.panels.map((x) => (
              <div key={x.id} className="item" onClick={() => p.onTogglePanel(x.id)}>
                <input type="checkbox" readOnly checked={!x.hidden} />{p.titles[x.id] ?? x.id}
              </div>
            ))}
          </div>
        )}
      </div>
      <button className={`btn${p.editing ? " on" : ""}`} onClick={p.onEdit} title="Drag panel headers, resize from the corner">Edit</button>
      <button className="btn" onClick={p.onReset} title="Reset layout to the preset">Reset</button>
      <button className={`btn${store.paused ? " on" : ""}`} onClick={() => api.runtime(store.paused ? "resume" : "pause").then((s) => { store.paused = !!s.paused; store.touch("metrics"); }).catch((e) => store.toast(errMsg(e)))}>{store.paused ? "Unpause" : "Pause"}</button>
      </>}
      <button className="btn" onClick={p.onConfig} title="Pipeline stages, playback and chunker settings">⚙ Pipeline config</button>
      <button className="btn" title="Theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? "☀" : "☾"}</button>
    </div>
  );
}
