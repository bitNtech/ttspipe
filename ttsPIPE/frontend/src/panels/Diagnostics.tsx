import { useState } from "react";
import { api, errMsg } from "../api/client";
import { refreshAll } from "../api/live";
import { fmt, fmtMs, Sparkline, StatusDot } from "../components/common";
import { store, useTopic } from "../state/store";
import type { Envelope, Kind } from "../types/contract";
import { KINDS } from "../types/contract";

type Tab = "metrics" | "logs" | "events" | "modules";

export function DiagnosticsPanel() {
  const [tab, setTab] = useState<Tab>("metrics");
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="tabs">
        {(["metrics", "logs", "events", "modules"] as Tab[]).map((t) => (
          <div key={t} className={`tab${tab === t ? " on" : ""}`} onClick={() => setTab(t)}>{t === "modules" ? "Modules / Models" : t[0].toUpperCase() + t.slice(1)}</div>
        ))}
      </div>
      <div style={{ flex: 1, minHeight: 0, overflow: "auto", padding: 6 }}>
        {tab === "metrics" && <MetricsTab />}
        {tab === "logs" && <LogsTab />}
        {tab === "events" && <EventsTab />}
        {tab === "modules" && <ModulesTab />}
      </div>
    </div>
  );
}

function MetricsTab() {
  useTopic("metrics");
  const m = store.metrics;
  if (!m) return <div className="muted small">Waiting for metrics…</div>;
  const st = m.stages ?? {};
  const kinds = [...KINDS.filter((k) => st[k]), ...Object.keys(st).filter((k) => !(KINDS as string[]).includes(k))];
  const max = Math.max(1, ...kinds.map((k) => st[k]?.p95_ms ?? 0));
  return (
    <div className="col">
      <table className="tbl">
        <thead><tr><th>stage</th><th>impl</th><th style={{ width: "30%" }}>p50 / p95</th><th>p50</th><th>p95</th><th>p95 trend</th><th>calls</th><th>err</th></tr></thead>
        <tbody>
          {kinds.map((k) => {
            const s = st[k]!;
            return (
              <tr key={k}>
                <td>{k}</td><td className="mono small">{s.impl}</td>
                <td>
                  <div className="bar" style={{ width: `${(100 * (s.p50_ms ?? 0)) / max}%` }} />
                  <div className="bar p95" style={{ width: `${(100 * (s.p95_ms ?? 0)) / max}%`, marginTop: 1 }} />
                </td>
                <td className="n">{fmt(s.p50_ms, 1)}</td><td className="n">{fmt(s.p95_ms, 1)}</td>
                <td><Sparkline data={s.spark ?? []} w={70} /></td>
                <td className="n">{s.count}</td><td className="n" style={{ color: s.errors ? "var(--err)" : undefined }}>{s.errors}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="kv small" style={{ gridTemplateColumns: "auto 1fr auto 1fr" }}>
        <span className="k">chunks/min</span><span className="v">{fmt(m.throughput?.chunks_per_min)}</span>
        <span className="k">audio s/min</span><span className="v">{fmt(m.throughput?.audio_s_per_min, 1)}</span>
        <span className="k">chars/min</span><span className="v">{fmt(m.throughput?.chars_per_min)}</span>
        <span className="k">utterances</span><span className="v">{m.utterances ?? 0}</span>
        <span className="k">interruptions</span><span className="v">{m.interruptions ?? 0}</span>
        <span className="k">underruns</span><span className="v">{m.underruns ?? 0}</span>
        <span className="k">CPU</span><span className="v">{fmt(m.system?.cpu_pct)}%</span>
        <span className="k">RSS</span><span className="v">{fmt(m.system?.rss_mb)} MB</span>
        <span className="k">threads</span><span className="v">{m.system?.threads ?? "—"}</span>
        <span className="k">events dropped</span><span className="v">{m.events_dropped ?? 0}</span>
        <span className="k">cache</span><span className="v">{m.cache?.entries ?? 0} · {fmt(m.cache?.mb, 1)}MB</span>
        <span className="k">TTFA p50/p95</span><span className="v">{fmtMs(m.latency?.ttfa?.p50)} / {fmtMs(m.latency?.ttfa?.p95)}</span>
      </div>
      <div className="row small muted">TTFA p95 history <Sparkline data={store.metricsHist.map((x) => x.latency?.ttfa?.p95 ?? null)} w={160} /></div>
    </div>
  );
}

function LogsTab() {
  useTopic("logs");
  const [level, setLevel] = useState("INFO");
  const [q, setQ] = useState("");
  const order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"];
  const rows = store.logs.filter((l) => order.indexOf(l.level ?? "INFO") >= order.indexOf(level) && (!q || (l.msg ?? "").toLowerCase().includes(q.toLowerCase()))).slice(-400);
  return (
    <div className="col" style={{ height: "100%" }}>
      <div className="row">
        <select value={level} onChange={(e) => setLevel(e.target.value)}>{order.map((o) => <option key={o}>{o}</option>)}</select>
        <input type="text" placeholder="filter" value={q} onChange={(e) => setQ(e.target.value)} className="grow" />
        <span className="muted small">{rows.length}</span>
      </div>
      <div className="loglines">
        {rows.slice().reverse().map((l, i) => (
          <div key={i} className={`l lvl-${l.level}`}>{l.ts ? new Date(l.ts).toLocaleTimeString([], { hour12: false }) : ""} {l.level?.padEnd(7)} {l.logger}: {l.msg}{l.session_id ? ` [${l.session_id}]` : ""}</div>
        ))}
      </div>
    </div>
  );
}

function EventsTab() {
  useTopic("events");
  const [q, setQ] = useState("");
  const [hideLevels, setHideLevels] = useState(true);
  const [frozen, setFrozen] = useState<Envelope[] | null>(null);
  const src = frozen ?? store.events;
  const rows = src.filter((e) => (!hideLevels || !["audio.level", "metrics", "playback.progress"].includes(e.type ?? "")) && (!q || (e.type ?? "").includes(q) || JSON.stringify(e.data).includes(q))).slice(-300);
  return (
    <div className="col" style={{ height: "100%" }}>
      <div className="row">
        <input type="text" placeholder="type or text filter" value={q} onChange={(e) => setQ(e.target.value)} className="grow" />
        <label className="check small"><input type="checkbox" checked={hideLevels} onChange={(e) => setHideLevels(e.target.checked)} />hide high-rate</label>
        <button className="btn sm" onClick={() => setFrozen(frozen ? null : [...store.events])}>{frozen ? "Resume" : "Freeze"}</button>
      </div>
      <div className="loglines">
        {rows.slice().reverse().map((e, i) => (
          <div key={i} className="l"><b>{e.type}</b> <span className="muted">{e.session_id ?? "global"} t={e.t_out_ms ?? "—"}</span> {JSON.stringify(e.data)?.slice(0, 300)}</div>
        ))}
      </div>
    </div>
  );
}

function ModulesTab() {
  useTopic("modules", "pipeline");
  const [msg, setMsg] = useState<string | null>(null);
  const act = async (fn: () => Promise<unknown>) => { try { await fn(); setMsg(null); setTimeout(refreshAll, 300); } catch (e) { setMsg(errMsg(e)); } };
  return (
    <div className="col">
      {msg && <div className="notice err small">{msg}</div>}
      <table className="tbl">
        <thead><tr><th /><th>module</th><th>status / reason</th><th>licence</th><th>size</th><th>RAM</th><th /></tr></thead>
        <tbody>
          {store.modules.map((m) => {
            const kind = m.kind as Kind;
            const slot = store.pipeline?.stages?.[kind];
            const active = slot?.impl === m.id?.split("/")[1];
            const dl = store.downloads[m.id ?? ""];
            return (
              <tr key={m.id}>
                <td>{active ? <StatusDot st={slot} /> : <span className={`dot ${m.available === false ? "error" : "unloaded"}`} />}</td>
                <td><div className="mono">{m.id}</div><div className="muted small">{m.name}{m.streaming ? " · streaming" : ""}{m.languages?.length ? ` · ${m.languages.join("/")}` : ""}</div></td>
                <td className="small">{m.available === false ? <span style={{ color: "var(--err)" }}>{m.reason}</span> : active ? (slot?.error ?? slot?.status) : "available"}
                  {!m.models_present && m.available !== false && <div style={{ color: "var(--warn)" }}>model files missing</div>}
                  {dl && <div className="mono">{dl.status} {fmt(dl.done_mb, 0)}/{fmt(dl.total_mb, 0)} MB {dl.error ?? ""}</div>}
                </td>
                <td className="small mono">{m.license}</td>
                <td className="n">{m.size_mb ? `${m.size_mb}MB` : "—"}</td>
                <td className="n">{m.ram_mb ? `${m.ram_mb}MB` : "—"}</td>
                <td className="row" style={{ gap: 3 }}>
                  {!active && <button className="btn sm" disabled={m.available === false} onClick={() => act(() => api.loadModule(kind, m.id!.split("/")[1]))}>Load</button>}
                  {active && !m.builtin && <button className="btn sm" onClick={() => act(() => api.unloadModule(kind, m.id!.split("/")[1]))}>Unload</button>}
                  {!m.models_present && <button className="btn sm" onClick={() => act(() => api.download(kind, m.id!.split("/")[1]))}>Download</button>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
