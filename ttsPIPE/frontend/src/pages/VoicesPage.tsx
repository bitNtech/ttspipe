// Voice cloning: record or upload a ~10 s reference clip with its exact transcript, save it to the voice library,
// test it on Fish S2-Pro, and make it the pipeline's voice.
import { useEffect, useRef, useState } from "react";
import { api, errMsg } from "../api/client";
import { blobToBase64, Recorder, toWav, type Clip } from "../audio/wav";
import { useTopic, store } from "../state/store";
import type { CloneVoice, CompareResult } from "../types/contract";
import { applyPipelineVoice, download, FISH, FishPill, FishServerCard, MetricChips, pipelineTts, sampleText, useFish } from "./shared";

const PROMPTS: { lang: string; label: string; text: string }[] = [
  { lang: "ta", label: "Tamil", text: "இன்று காலை நான் சந்தைக்குச் சென்று காய்கறிகளும் பழங்களும் வாங்கினேன். வானிலை மிகவும் இனிமையாக இருந்தது, அதனால் நடந்தே வீட்டுக்குத் திரும்பினேன்." },
  { lang: "en", label: "English", text: "This morning I walked to the market and bought fresh vegetables and fruit. The weather was lovely, so I took the long way home along the river." },
  { lang: "multi", label: "Tanglish", text: "இன்னைக்கு office-ல ஒரு important meeting இருந்துச்சு. Presentation நல்லா போச்சு, அதனால everyone ரொம்ப happy-ஆ இருந்தாங்க." },
];

export function VoicesPage({ onNavigate }: { onNavigate: (page: string) => void }) {
  useTopic("pipeline", "modules");
  const [fish, refreshFish] = useFish(3000);
  const [clones, setClones] = useState<CloneVoice[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const load = () => api.clones().then(setClones).catch((e) => setErr(errMsg(e)));
  useEffect(() => { load(); }, [store.connected]);
  const fishMod = store.modules.find((m) => m.id === `tts/${FISH}`);

  return (
    <div className="page">
      <section className="card">
        <div className="row wrap">
          <h3 style={{ margin: 0 }}>Voice cloning</h3>
          <FishPill st={fish} />
          <span className="muted small">Cloned voices are spoken by Fish Audio S2-Pro on the GPU (zero-shot: no training, the clip is the prompt).</span>
        </div>
        {fishMod?.available === false && <div className="notice warn small" style={{ marginTop: 8 }}>{fishMod.reason}. Install with <span className="mono">./run.sh models fish</span> and restart the backend.</div>}
        {fish?.health !== "ok" && <div style={{ marginTop: 8 }}><FishServerCard st={fish} refresh={refreshFish} compact /></div>}
        {err && <div className="notice err small" style={{ marginTop: 8 }}>{err}</div>}
      </section>
      <div className="page-grid">
        <NewVoice onSaved={(c) => { setClones((xs) => [c, ...xs]); api.modules().then((m) => { store.modules = m; store.touch("modules"); }); }} />
        <section className="card">
          <h3>Voice library <span className="muted small">{clones.length} voice{clones.length === 1 ? "" : "s"} · data/voices</span></h3>
          {!clones.length && <div className="muted small">No cloned voices yet. Record or upload a clip on the left.</div>}
          {clones.map((c) => <CloneRow key={c.id} c={c} onChange={load} onNavigate={onNavigate} />)}
        </section>
      </div>
    </div>
  );
}

function NewVoice({ onSaved }: { onSaved: (c: CloneVoice) => void }) {
  const [clip, setClip] = useState<(Clip & { url: string }) | null>(null);
  const [rec, setRec] = useState<Recorder | null>(null);
  const [level, setLevel] = useState(0);
  const [secs, setSecs] = useState(0);
  const [prompt, setPrompt] = useState(0);
  const [transcript, setTranscript] = useState("");
  const [name, setName] = useState("");
  const [lang, setLang] = useState("ta");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!rec) return;
    const iv = window.setInterval(() => {
      const s = (performance.now() - rec.started) / 1000;
      setSecs(s);
      if (s >= 30) stop();
    }, 100);
    return () => window.clearInterval(iv);
  }, [rec]); // eslint-disable-line react-hooks/exhaustive-deps

  const take = async (blob: Blob) => {
    try {
      const c = await toWav(blob);
      setClip({ ...c, url: URL.createObjectURL(c.wav) });
      setMsg(null);
    } catch (e) {
      setMsg({ ok: false, text: `Could not decode that audio: ${errMsg(e)}` });
    }
  };
  const start = async () => {
    const r = new Recorder();
    try { await r.start(setLevel); setRec(r); setClip(null); setMsg(null); } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
  };
  const stop = async () => {
    if (!rec) return;
    const blob = await rec.stop();
    setRec(null); setLevel(0);
    await take(blob);
  };
  const save = async () => {
    if (!clip) return;
    setBusy(true);
    try {
      const c = await api.createClone({ name, transcript, language: lang, audio_b64: await blobToBase64(clip.wav) });
      setMsg({ ok: true, text: `Saved "${c.name}". Test it in the library →` });
      onSaved(c);
      setClip(null); setTranscript(""); setName("");
    } catch (e) { setMsg({ ok: false, text: errMsg(e) }); }
    setBusy(false);
  };

  const d = clip?.seconds ?? 0;
  const quality = !clip ? null : d < 3 ? ["err", "Too short: record at least 3 s (about 10 s is best)."] : d > 20 ? ["warn", "Long clip: every second is replayed on each request and slows first audio; 8–12 s is best."] : clip.peakDb < -24 ? ["warn", `Quiet recording (peak ${clip.peakDb.toFixed(0)} dBFS): move closer to the microphone.`] : ["ok", `${d.toFixed(1)} s — good length.`];
  const canSave = !!clip && d >= 2 && d <= 30 && transcript.trim() && name.trim() && !busy;

  return (
    <section className="card">
      <h3>New cloned voice</h3>
      <ol className="steps">
        <li>
          <b>Reference clip</b> — one speaker, quiet room, natural pace, about 10 seconds.
          <div className="prompt">
            <div className="row wrap">
              <span className="small muted">Read this aloud:</span>
              {PROMPTS.map((p, i) => <button key={p.label} className={`btn sm${prompt === i ? " on" : ""}`} onClick={() => { setPrompt(i); setLang(p.lang); }}>{p.label}</button>)}
            </div>
            <div className="prompt-text">{PROMPTS[prompt].text}</div>
          </div>
          <div className="row wrap" style={{ marginTop: 6 }}>
            {!rec ? <button className="btn primary" onClick={start}>● Record</button> : <button className="btn danger" onClick={stop}>■ Stop ({secs.toFixed(1)} s)</button>}
            {rec && <span className="meter"><i style={{ width: `${Math.round(level * 100)}%` }} /></span>}
            <span className="muted small">or</span>
            <button className="btn" disabled={!!rec} onClick={() => fileRef.current?.click()}>Upload audio file…</button>
            <input ref={fileRef} type="file" accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.webm" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) { take(f); if (!name) setName(f.name.replace(/\.[^.]+$/, "")); } e.target.value = ""; }} />
          </div>
          {clip && (
            <div className="col" style={{ marginTop: 6 }}>
              <audio controls src={clip.url} />
              {quality && <div className={`notice small ${quality[0]}`}>{quality[1]}</div>}
              <div className="muted small">Converted to mono 16-bit WAV at {clip.rate} Hz with leading/trailing silence trimmed.</div>
            </div>
          )}
        </li>
        <li>
          <b>Exact transcript</b> — the words spoken in the clip, verbatim. A wrong transcript silently degrades the voice.
          <textarea rows={3} style={{ width: "100%", marginTop: 4 }} value={transcript} onChange={(e) => setTranscript(e.target.value)} placeholder="What is said in the clip…" />
          <button className="btn sm" onClick={() => setTranscript(PROMPTS[prompt].text)}>Use the prompt I read</button>
        </li>
        <li>
          <b>Name it</b>
          <div className="form" style={{ marginTop: 4 }}>
            <label>Name</label>
            <input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Priya (warm, Tamil)" />
            <label>Language</label>
            <select value={lang} onChange={(e) => setLang(e.target.value)}>
              <option value="ta">Tamil</option><option value="en">English</option><option value="multi">Tamil + English</option>
            </select>
          </div>
        </li>
      </ol>
      <div className="row">
        <button className="btn primary" disabled={!canSave} onClick={save}>{busy ? "Saving…" : "Save voice"}</button>
        {!canSave && !busy && <span className="muted small">{!clip ? "Record or upload a clip" : !transcript.trim() ? "Add the transcript" : !name.trim() ? "Give it a name" : "Clip must be 2–30 s"}</span>}
      </div>
      {msg && <div className={`notice small ${msg.ok ? "ok" : "err"}`} style={{ marginTop: 6 }}>{msg.text}</div>}
    </section>
  );
}

function CloneRow({ c, onChange, onNavigate }: { c: CloneVoice; onChange: () => void; onNavigate: (p: string) => void }) {
  const [edit, setEdit] = useState(false);
  const [name, setName] = useState(c.name ?? "");
  const [transcript, setTranscript] = useState(c.transcript ?? "");
  const [text, setText] = useState(sampleText(c.language));
  const [res, setRes] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const cfgVoice = store.pipeline?.config?.voice;
  const inUse = pipelineTts() === FISH && cfgVoice?.voice_id === c.voice_id;

  const test = async () => {
    setBusy(true); setMsg(null); setRes(null);
    try {
      const r = await api.render(text, { label: c.name, config: { stages: { tts: { impl: FISH } }, voice: { impl: FISH, voice_id: c.voice_id, speed: 1, pitch: 0 } } });
      if (r.error) setMsg(r.error);
      else if (r.tts_impl !== FISH) setMsg(`Fish S2-Pro was unavailable; this was rendered by ${r.tts_impl}. Start the Fish server and retry.`);
      setRes(r);
    } catch (e) { setMsg(errMsg(e)); }
    setBusy(false);
  };
  const act = async (fn: () => Promise<unknown>) => { try { await fn(); setMsg(null); onChange(); } catch (e) { setMsg(errMsg(e)); } };

  return (
    <div className="run">
      <div className="row wrap">
        <b>{c.name}</b>
        <span className="badge">{c.language || "multi"}</span>
        <span className="badge">{c.duration_s?.toFixed(1)} s</span>
        {inUse && <span className="badge ok">pipeline voice</span>}
        <span className="muted small right">{c.created ? new Date(c.created * 1000).toLocaleDateString() : ""}</span>
      </div>
      {c.warning && <div className="notice warn small">{c.warning}</div>}
      <div className="row"><span className="small muted">Reference</span><audio controls preload="none" src={c.audio_url} className="grow" /></div>
      {!edit ? (
        <div className="run-text small">“{c.transcript}”</div>
      ) : (
        <div className="form">
          <label>Name</label><input type="text" value={name} onChange={(e) => setName(e.target.value)} />
          <label>Transcript</label><textarea rows={3} value={transcript} onChange={(e) => setTranscript(e.target.value)} />
        </div>
      )}
      <div className="row">
        <input type="text" className="grow" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === "Enter" && test()} />
        <button className="btn primary sm" disabled={busy || !text.trim()} onClick={test}>{busy ? "Rendering…" : "▶ Test"}</button>
      </div>
      {res?.url && (
        <div className="col">
          <div className="row"><audio controls autoPlay src={res.url} className="grow" /><button className="btn sm" onClick={() => download(res.url!, `${c.id}.wav`)}>Download</button></div>
          <MetricChips m={res.metrics} />
        </div>
      )}
      {msg && <div className="notice err small">{msg}</div>}
      <div className="row wrap">
        <button className="btn sm" disabled={inUse} onClick={() => act(async () => { await applyPipelineVoice(FISH, c.voice_id); store.toast(`Pipeline now speaks as ${c.name} (Fish S2-Pro)`); })}>Use in pipeline</button>
        <button className="btn sm" onClick={() => onNavigate("test")}>Open test bench</button>
        {!edit ? <button className="btn sm" onClick={() => setEdit(true)}>Edit</button> : <>
          <button className="btn sm primary" onClick={() => act(async () => { await api.updateClone(c.id!, { name, transcript }); setEdit(false); })}>Save</button>
          <button className="btn sm" onClick={() => { setEdit(false); setName(c.name ?? ""); setTranscript(c.transcript ?? ""); }}>Cancel</button>
        </>}
        <button className="btn sm danger right" onClick={() => window.confirm(`Delete the cloned voice "${c.name}"?`) && act(() => api.deleteClone(c.id!))}>Delete</button>
      </div>
    </div>
  );
}
