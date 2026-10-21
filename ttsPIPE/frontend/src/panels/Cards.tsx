import { useEffect, useRef, useState } from "react";
import { api, errMsg } from "../api/client";
import { speaker } from "../api/live";
import { fmt, fmtMs, LangChip, langClass } from "../components/common";
import { store, useTopic } from "../state/store";
import type { PlaybackStateName, VoiceEntry } from "../types/contract";

// ---- Playback state ----------------------------------------------------------------------------

const SM_NODES: Record<PlaybackStateName, { x: number; y: number }> = {
  idle: { x: 4, y: 6 }, buffering: { x: 82, y: 6 }, synthesizing: { x: 160, y: 6 },
  speaking: { x: 160, y: 62 }, ducked: { x: 160, y: 118 }, done: { x: 82, y: 62 }, interrupted: { x: 4, y: 62 },
};
const SM_EDGES: [PlaybackStateName, PlaybackStateName][] = [
  ["idle", "buffering"], ["buffering", "synthesizing"], ["synthesizing", "speaking"], ["speaking", "ducked"], ["ducked", "speaking"],
  ["speaking", "done"], ["done", "idle"], ["speaking", "interrupted"], ["interrupted", "idle"],
];
const NW = 74, NH = 22;

export function PlaybackCard() {
  useTopic("sessions", "level");
  const s = store.current();
  const state = s?.state ?? "idle";
  const c = (n: PlaybackStateName) => ({ x: SM_NODES[n].x + NW / 2, y: SM_NODES[n].y + NH / 2 });
  return (
    <div className="col" style={{ height: "100%" }}>
      <div className={`big-state st-${state}`}>{state}</div>
      <div className="small muted">
        {s ? <>prev <b>{s.prev ?? "—"}</b> · {s.reason ?? "—"}<br />t_out <span className="mono">{(s.t_out_ms / 1000).toFixed(2)}s</span> · gain <span className="mono">{fmt(s.gain_db, 1)} dB</span></> : "no session"}
      </div>
      <svg viewBox="0 0 238 146" style={{ width: "100%", flex: 1, minHeight: 0 }} preserveAspectRatio="xMidYMin meet">
        <defs><marker id="smarr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--text-3)" /></marker></defs>
        {SM_EDGES.map(([a, b], i) => {
          const A = c(a), B = c(b);
          const back = SM_EDGES.some(([x, y]) => x === b && y === a);
          const off = back ? (a < b ? 5 : -5) : 0;
          const dx = B.x - A.x, dy = B.y - A.y, len = Math.hypot(dx, dy) || 1;
          const sx = A.x + (dx / len) * (Math.abs(dx) > Math.abs(dy) ? NW / 2 : NH / 2), sy2 = A.y + (dy / len) * (Math.abs(dx) > Math.abs(dy) ? NW / 2 : NH / 2);
          const ex = B.x - (dx / len) * (Math.abs(dx) > Math.abs(dy) ? NW / 2 : NH / 2), ey = B.y - (dy / len) * (Math.abs(dx) > Math.abs(dy) ? NW / 2 : NH / 2);
          const hot = s?.prev === a && state === b;
          return <line key={i} x1={sx + (dy ? off : 0)} y1={sy2 + (dx ? off : 0)} x2={ex + (dy ? off : 0)} y2={ey + (dx ? off : 0)} stroke={hot ? "var(--accent)" : "var(--border-strong)"} strokeWidth={hot ? 2 : 1.2} markerEnd="url(#smarr)" />;
        })}
        {(Object.keys(SM_NODES) as PlaybackStateName[]).map((n) => {
          const p = SM_NODES[n];
          const on = n === state;
          return (
            <g key={n} transform={`translate(${p.x},${p.y})`}>
              <rect width={NW} height={NH} rx={5} fill={on ? "var(--accent)" : "var(--panel)"} stroke={on ? "var(--accent)" : "var(--border-strong)"} />
              <text x={NW / 2} y={15} textAnchor="middle" fontSize={11} fill={on ? "#fff" : "var(--text)"}>{n}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ---- Language segments -------------------------------------------------------------------------

export function SegmentsCard() {
  useTopic("utterances");
  const s = store.current();
  const u = s ? s.utts.get(s.order[s.order.length - 1] ?? "") : undefined;
  const spans = u?.spans?.length ? u.spans : (u?.segments ?? []).map((x) => ({ text: x.text ?? "", lang: x.lang ?? "en" }));
  if (!spans.length) return <div className="muted small">Segments of the current utterance appear here.</div>;
  const tot: Record<string, number> = {};
  for (const sp of spans) tot[sp.lang] = (tot[sp.lang] ?? 0) + sp.text.replace(/\s/g, "").length;
  const all = Object.values(tot).reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="col">
      <div className="row small muted"><span className="mono">#{u?.utterance_id}</span><span className="right">{spans.length} runs</span></div>
      <div className="segbar">
        {Object.entries(tot).map(([l, n]) => <div key={l} style={{ width: `${(100 * n) / all}%`, background: `var(--${langClass(l)})` }} title={`${l} ${Math.round((100 * n) / all)}%`} />)}
      </div>
      <div className="row wrap small">{Object.entries(tot).map(([l, n]) => <span key={l} className="row" style={{ gap: 3 }}><LangChip lang={l} /><span className="mono">{Math.round((100 * n) / all)}%</span></span>)}</div>
      <div className="segchips">{spans.map((sp, i) => <span key={i} className={`segchip ${langClass(sp.lang)}`} title={sp.lang}>{sp.text}</span>)}</div>
    </div>
  );
}

// ---- Voice -------------------------------------------------------------------------------------

export function VoiceCard() {
  useTopic("pipeline");
  const cfg = store.pipeline?.config;
  const ttsImpl = store.pipeline?.stages?.tts?.impl ?? cfg?.stages?.tts?.impl ?? "";
  const [voices, setVoices] = useState<VoiceEntry[]>([]);
  const [speed, setSpeed] = useState(cfg?.voice?.speed ?? 1);
  const [pitch, setPitch] = useState(cfg?.voice?.pitch ?? 0);
  const [vid, setVid] = useState(cfg?.voice?.voice_id ?? "");
  const [msg, setMsg] = useState<string | null>(null);
  const timer = useRef<number>(0);
  useEffect(() => { api.voices().then(setVoices).catch(() => undefined); }, []);
  useEffect(() => { setSpeed(cfg?.voice?.speed ?? 1); setPitch(cfg?.voice?.pitch ?? 0); setVid(cfg?.voice?.voice_id ?? ""); }, [cfg?.voice?.speed, cfg?.voice?.pitch, cfg?.voice?.voice_id]);
  const mine = voices.filter((v) => v.impl === ttsImpl);
  const push = (patch: Record<string, unknown>) => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => api.putVoice({ impl: ttsImpl, ...patch }).then(() => setMsg(null)).catch((e) => setMsg(errMsg(e))), 250);
  };
  const preview = async () => {
    try {
      const blob = await api.preview(ttsImpl, { voice_id: vid, speed, pitch });
      new Audio(URL.createObjectURL(blob)).play();
    } catch (e) { setMsg(errMsg(e)); }
  };
  return (
    <div className="form">
      <label>engine</label><span className="mono">{ttsImpl || "—"}</span>
      <label>voice</label>
      <select value={vid} onChange={(e) => { setVid(e.target.value); push({ voice_id: e.target.value }); speaker.voice(`${ttsImpl}:${e.target.value}`); }}>
        {mine.length === 0 && <option value={vid}>{vid || "(default)"}</option>}
        {mine.map((v) => <option key={v.id} value={v.id}>{v.name ?? v.id} · {v.language}{v.gender ? ` · ${v.gender}` : ""}</option>)}
      </select>
      <label>speed</label>
      <div className="ctl"><input type="range" min={0.5} max={2} step={0.05} value={speed} onChange={(e) => { setSpeed(+e.target.value); push({ speed: +e.target.value }); }} /><span className="mono" style={{ width: 36 }}>{speed.toFixed(2)}×</span></div>
      <label>pitch</label>
      <div className="ctl"><input type="range" min={-6} max={6} step={0.5} value={pitch} onChange={(e) => { setPitch(+e.target.value); push({ pitch: +e.target.value }); }} /><span className="mono" style={{ width: 36 }}>{pitch > 0 ? "+" : ""}{pitch.toFixed(1)}st</span></div>
      <span /><div className="row"><button className="btn sm" onClick={preview}>▶ Preview</button>{msg && <span className="small" style={{ color: "var(--err)" }}>{msg}</span>}</div>
    </div>
  );
}

// ---- Duplex ------------------------------------------------------------------------------------

export function DuplexCard() {
  useTopic("sessions");
  const s = store.current();
  const b = store.bridge;
  const li = s?.last_interrupted;
  return (
    <div className="col">
      <div className="row wrap">
        <button className="btn sm" title="prePIPE barge_in.candidate → duck" onClick={() => speaker.control("duck")}>Barge-in candidate</button>
        <button className="btn sm danger" title="prePIPE barge_in → stop" onClick={() => speaker.control("stop")}>Barge-in</button>
        <button className="btn sm" title="prePIPE backchannel → resume" onClick={() => speaker.control("resume")}>Backchannel</button>
      </div>
      <div className="small row">
        <span className={`dot ${b?.connected ? "ready" : b?.enabled ? "error" : "disabled"}`} />
        prePIPE bridge: {b?.enabled ? (b.connected ? `connected ${b.url ?? ""}` : `disconnected${b.error ? ` (${b.error})` : ""}`) : "off"}
        {b?.last_event && <span className="muted"> · last {b.last_event}</span>}
      </div>
      {li ? (
        <div className="small">
          <div><span className="badge ok">heard</span> {li.heard_text || <span className="muted">(nothing)</span>}</div>
          <div style={{ marginTop: 2 }}><span className="badge err">unplayed</span> <span className="muted">{li.unplayed_text || "—"}</span></div>
          <div className="muted mono">#{li.utterance_id} cut at {fmt((li.at_ms ?? 0) / 1000, 2)}s · {li.source}</div>
        </div>
      ) : <div className="muted small">Stop mid-utterance to see heard_text vs unplayed_text.</div>}
    </div>
  );
}

// ---- Latency -----------------------------------------------------------------------------------

export function LatencyCard() {
  useTopic("metrics");
  const m = store.metrics;
  const L = m?.latency ?? {};
  const rows: [string, string, (v: number | null | undefined) => string][] = [
    ["TTFA", "ttfa", fmtMs], ["TTFA from end", "ttfa_end", fmtMs], ["text→1st chunk", "text_to_first_chunk", fmtMs],
    ["first synth", "first_synth", fmtMs], ["normalize", "normalize", (v) => fmt(v, 1, "ms")], ["chunk RTF", "chunk_rtf", (v) => fmt(v, 2)],
  ];
  return (
    <div className="col">
      <div className="kv small">
        <span className="muted" /><span className="muted" style={{ textAlign: "right" }}>last</span><span className="muted" style={{ textAlign: "right" }}>p95</span>
        {rows.map(([label, key, f]) => {
          const st = (L as Record<string, { last?: number | null; p95?: number | null }>)[key];
          return [<span key={key + "k"} className="k">{label}</span>, <span key={key + "v"} className="v">{f(st?.last)}</span>, <span key={key + "p"} className="v">{f(st?.p95)}</span>];
        })}
        <span className="k">underruns</span><span className="v">{m?.underruns ?? "—"}</span><span />
        <span className="k">cache hit</span><span className="v">{m?.cache?.hit_rate != null ? `${Math.round(m.cache.hit_rate * 100)}%` : "—"}</span><span />
      </div>
      {m?.ttfa?.by_profile && Object.keys(m.ttfa.by_profile).length > 0 && (
        <table className="tbl small">
          <thead><tr><th>profile</th><th>n</th><th>p50</th><th>p95</th></tr></thead>
          <tbody>{Object.entries(m.ttfa.by_profile).map(([p, s]) => <tr key={p}><td className="ellipsis" title={p}>{p}</td><td className="n">{s.n}</td><td className="n">{fmtMs(s.p50)}</td><td className="n">{fmtMs(s.p95)}</td></tr>)}</tbody>
        </table>
      )}
    </div>
  );
}
