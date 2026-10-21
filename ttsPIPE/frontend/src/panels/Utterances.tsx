import { useEffect, useRef, useState } from "react";
import { MOCK } from "../api/client";
import { fmt, fmtMs, LangChip, WordDiff } from "../components/common";
import { store, useTopic, type UttView } from "../state/store";

export const uttOpts = { autoScroll: true };

export function UtterancesControls() {
  const [on, setOn] = useState(uttOpts.autoScroll);
  return <label className="check small"><input type="checkbox" checked={on} onChange={(e) => { uttOpts.autoScroll = e.target.checked; setOn(e.target.checked); }} />auto-scroll</label>;
}

const renderUrl = (rid?: string | null) => (rid ? (MOCK ? null : `/api/renders/${rid}.wav`) : null);

function Heard({ text, heard }: { text: string; heard: string }) {
  const words = text.split(/\s+/).filter(Boolean);
  const n = heard.split(/\s+/).filter(Boolean).length;
  return (
    <>
      {words.slice(0, n).join(" ")}
      <span className="heard-mark" title="playback was cut here">heard until here</span>
      <span className="unheard">{words.slice(n).join(" ")}</span>
    </>
  );
}

function Row({ u }: { u: UttView }) {
  const [open, setOpen] = useState(false);
  const raw = u.text ?? u.live_text;
  const langs = [...new Set((u.spans.length ? u.spans : u.segments ?? []).map((s) => s.lang ?? "en"))];
  const status = u.status ?? "open";
  const badge = status === "interrupted" ? "err" : status === "final" ? "ok" : "accent";
  const url = renderUrl(u.render_id);
  const lat = u.latency ?? {};
  const changed = u.norm_parts.some((p) => p.changed);
  const time = u.ts ? new Date(u.ts).toLocaleTimeString([], { hour12: false }) : "";
  return (
    <div className="utt">
      <div className="head">
        <span className="mono small muted">{time}</span>
        <span className="mono small">#{u.utterance_id}</span>
        {langs.map((l) => <LangChip key={l} lang={l} />)}
        <span className={`badge ${badge}`}>{status}</span>
        {lat.ttfa_ms != null && <span className="badge" title="time to first audio">TTFA {fmtMs(lat.ttfa_ms)}</span>}
        {lat.rtf != null && <span className="badge" title="real-time factor">RTF {fmt(lat.rtf, 2)}</span>}
        <span className="right row">
          {url && <button className="icon-btn" title="Replay" onClick={() => new Audio(url).play()}>▶</button>}
          {url && <a className="icon-btn" title="Download WAV" href={url} download>⬇</a>}
          <button className="icon-btn" title="Chunks" onClick={() => setOpen(!open)}>{open ? "▴" : "▾"} {u.chunks?.length ?? u.chunk_events.length}</button>
        </span>
      </div>
      <div className="text">
        {status === "interrupted" && u.heard_text != null ? <Heard text={raw} heard={u.heard_text} /> : (
          status !== "final" && u.progress_heard ? <Heard text={raw} heard={u.progress_heard} /> : raw
        )}
      </div>
      {changed && (
        <div className="text small" title="raw → normalized">
          <span className="muted">norm: </span>
          {u.norm_parts.map((p, i) => <span key={i}>{p.changed ? <WordDiff before={p.before} after={p.after} /> : p.after + " "}</span>)}
        </div>
      )}
      {open && (
        <table className="chunks">
          <thead><tr><th>#</th><th>lang</th><th>reason</th><th>text</th><th>flush</th><th>1st audio</th><th>synth</th><th>rtf</th><th>play</th></tr></thead>
          <tbody>
            {(u.chunks?.length ? u.chunks : u.chunk_events.map((c) => ({ ...c, latency: undefined, played: undefined }))).map((c) => (
              <tr key={c.chunk_id}>
                <td className="n">{c.index}</td><td><LangChip lang={c.lang} /></td><td>{c.reason}</td>
                <td className="ellipsis" style={{ maxWidth: 220 }} title={c.text}>{c.text}</td>
                <td className="n">{fmtMs(c.latency?.flush_ms)}</td><td className="n">{fmtMs(c.latency?.first_audio_ms)}{(c as { cached?: boolean }).cached ? " (c)" : ""}</td>
                <td className="n">{fmtMs(c.latency?.synth_ms)}</td><td className="n">{fmt(c.latency?.rtf, 2)}</td><td>{c.played ?? "…"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export function UtterancesPanel() {
  useTopic("utterances", "sessions");
  const ref = useRef<HTMLDivElement>(null);
  const s = store.current();
  const rows = s ? s.order.map((id) => s.utts.get(id)!).filter(Boolean) : [];
  useEffect(() => { if (uttOpts.autoScroll && ref.current) ref.current.scrollTop = ref.current.scrollHeight; });
  if (!rows.length) return <div className="muted small">No utterances yet. Type in the Speak box or pick a sample script.</div>;
  return <div ref={ref} style={{ height: "100%", overflow: "auto" }}>{rows.map((u) => <Row key={u.utterance_id} u={u} />)}</div>;
}
