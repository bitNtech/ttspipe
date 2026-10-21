import { useEffect, useState } from "react";
import { api, errMsg } from "../api/client";
import { fmt, fmtMs } from "../components/common";
import { store, useTopic } from "../state/store";
import type { CompareResult, CompareVariant, ProfileEntry, RatingSummary } from "../types/contract";

function variantOf(v: string): CompareVariant {
  const [kind, name] = v.split(":");
  if (kind === "profile") return { profile: name, label: name };
  return { label: name, config: { stages: { tts: { impl: name, params: {} } }, voice: { impl: name } } };
}

export function ComparePanel() {
  useTopic("modules");
  const [text, setText] = useState("Hello! உங்கள் order நாளை வந்துவிடும்.");
  const [a, setA] = useState("profile:balanced");
  const [b, setB] = useState("profile:minimal");
  const [res, setRes] = useState<{ a?: CompareResult; b?: CompareResult } | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [profiles, setProfiles] = useState<ProfileEntry[]>([]);
  const [mos, setMos] = useState<RatingSummary[]>([]);
  const [rated, setRated] = useState<Record<string, number>>({});
  const loadMos = () => api.ratings().then((r) => setMos(r.summary ?? [])).catch(() => undefined);
  useEffect(() => { api.profiles().then(setProfiles).catch(() => undefined); loadMos(); }, []);
  const engines = store.modules.filter((m) => m.kind === "tts");
  const opts = [...profiles.map((p) => `profile:${p.name}`), ...engines.map((m) => `engine:${m.id!.split("/")[1]}`)];

  const run = async () => {
    setBusy(true); setErr(null); setRes(null); setRated({});
    try { setRes(await api.compare(text, variantOf(a), variantOf(b))); } catch (e) { setErr(errMsg(e)); } finally { setBusy(false); }
  };
  const rate = async (r: CompareResult, score: number) => {
    try {
      await api.rate({ render_id: r.render_id, tts_impl: r.tts_impl, voice_id: r.voice_id, score, text });
      setRated((x) => ({ ...x, [r.render_id ?? ""]: score }));
      loadMos();
    } catch (e) { setErr(errMsg(e)); }
  };
  const Side = ({ r, label }: { r?: CompareResult; label: string }) => (
    <div className="side col">
      <div className="row"><b>{label}</b><span className="mono small muted ellipsis">{r?.label} · {r?.tts_impl}{r?.voice_id ? `:${r.voice_id}` : ""}</span></div>
      {r?.error ? <div className="notice err small">{r.error}</div> : r?.url ? (
        <>
          <audio controls src={r.url} />
          <div className="kv small">
            <span className="k">TTFA</span><span className="v">{fmtMs(r.metrics?.ttfa_ms)}</span><span />
            <span className="k">first synth</span><span className="v">{fmtMs(r.metrics?.first_synth_ms)}</span><span />
            <span className="k">RTF</span><span className="v">{fmt(r.metrics?.rtf, 2)}</span><span />
            <span className="k">audio</span><span className="v">{fmtMs(r.metrics?.audio_ms)}</span><span />
          </div>
          <div className="stars row">{[1, 2, 3, 4, 5].map((n) => <button key={n} className={rated[r.render_id ?? ""] === n ? "on" : ""} onClick={() => rate(r, n)}>{n}</button>)}</div>
        </>
      ) : <div className="muted small">—</div>}
    </div>
  );
  return (
    <div className="col">
      <textarea rows={2} value={text} onChange={(e) => setText(e.target.value)} />
      <div className="row wrap">
        <span className="small muted">A</span>
        <select value={a} onChange={(e) => setA(e.target.value)}>{opts.map((o) => <option key={o} value={o}>{o}</option>)}</select>
        <span className="small muted">B</span>
        <select value={b} onChange={(e) => setB(e.target.value)}>{opts.map((o) => <option key={o} value={o}>{o}</option>)}</select>
        <button className="btn sm primary" disabled={busy || !text.trim()} onClick={run}>{busy ? "Rendering…" : "Compare"}</button>
      </div>
      {err && <div className="notice err small">{err}</div>}
      {res && <div className="ab"><Side r={res.a} label="A" /><Side r={res.b} label="B" /></div>}
      {mos.length > 0 && (
        <table className="tbl small">
          <thead><tr><th>engine</th><th>voice</th><th>n</th><th>MOS</th><th>±95%</th></tr></thead>
          <tbody>{mos.map((m, i) => <tr key={i}><td className="mono">{m.tts_impl}</td><td>{m.voice_id ?? "—"}</td><td className="n">{m.n}</td><td className="n">{fmt(m.mos, 2)}</td><td className="n">{fmt(m.ci95, 2)}</td></tr>)}</tbody>
        </table>
      )}
    </div>
  );
}
