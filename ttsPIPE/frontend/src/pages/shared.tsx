// Pieces shared by the Test, Engines and Voices pages.
import { useEffect, useState } from "react";
import { api, errMsg } from "../api/client";
import { refreshAll } from "../api/live";
import { fmt, fmtMs } from "../components/common";
import { store } from "../state/store";
import type { CompareResult, FishStatus, ModuleEntry } from "../types/contract";

export const FISH = "fish_s2";
export const implOf = (m?: ModuleEntry | null) => m?.id?.split("/")[1] ?? "";
export const ttsModules = () => store.modules.filter((m) => m.kind === "tts");
export const ttsModule = (impl: string) => store.modules.find((m) => m.id === `tts/${impl}`);
export const pipelineTts = () => store.pipeline?.stages?.tts?.impl ?? store.pipeline?.config?.stages?.tts?.impl ?? "";

/** Make impl the pipeline's TTS engine (keeping its current params if it already is), then set the voice. */
export async function applyPipelineVoice(impl: string, voiceId?: string, extra?: { speed?: number; pitch?: number }) {
  const cfgImpl = store.pipeline?.config?.stages?.tts?.impl;
  if (cfgImpl !== impl) await api.putStage("tts", { impl });
  if (voiceId !== undefined) await api.putVoice({ impl, voice_id: voiceId, ...extra });
  await refreshAll();
}

export function MetricChips({ m }: { m?: CompareResult["metrics"] | null }) {
  if (!m) return null;
  return (
    <span className="row wrap" style={{ gap: 4 }}>
      <span className="badge" title="Time to first audio">TTFA {fmtMs(m.ttfa_ms)}</span>
      <span className="badge" title="First synthesis call, text in → first audio out">first synth {fmtMs(m.first_synth_ms)}</span>
      <span className={`badge${(m.rtf ?? 0) > 1 ? " warn" : ""}`} title="Real-time factor: synthesis time ÷ audio duration (< 1 is faster than real time)">RTF {fmt(m.rtf, 2)}</span>
      <span className="badge" title="Audio duration">audio {fmtMs(m.audio_ms)}</span>
      {m.chunks != null && <span className="badge">{m.chunks} chunk{m.chunks === 1 ? "" : "s"}</span>}
      {m.wall_ms != null && <span className="badge" title="Wall time of the whole render">wall {fmtMs(m.wall_ms)}</span>}
    </span>
  );
}

const HEALTH_LABEL: Record<string, string> = {
  ok: "ready", loading: "loading model", starting: "starting", error: "load failed", offline: "not running",
};
const HEALTH_DOT: Record<string, string> = { ok: "ready", loading: "loading", starting: "loading", error: "error", offline: "unloaded" };

/** Polls /api/fish every `ms` while mounted. */
export function useFish(ms = 2000): [FishStatus | null, () => void] {
  const [st, setSt] = useState<FishStatus | null>(null);
  const [n, setN] = useState(0);
  useEffect(() => {
    let live = true;
    const tick = () => api.fish().then((s) => live && setSt(s)).catch(() => live && setSt(null));
    tick();
    const iv = window.setInterval(tick, ms);
    return () => { live = false; window.clearInterval(iv); };
  }, [ms, n]);
  return [st, () => setN((x) => x + 1)];
}

export function FishPill({ st }: { st: FishStatus | null }) {
  const h = st?.health ?? "offline";
  return <span className="chip"><span className={`dot ${HEALTH_DOT[h]}`} />Fish S2-Pro: {HEALTH_LABEL[h] ?? h}</span>;
}

export function FishServerCard({ st, refresh, compact = false }: { st: FishStatus | null; refresh: () => void; compact?: boolean }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const act = async (a: "start" | "stop") => {
    setBusy(true); setErr(null);
    try { await api.fishAction(a); } catch (e) { setErr(errMsg(e)); }
    setBusy(false); refresh();
  };
  const h = st?.health ?? "offline";
  const info = st?.info ?? {};
  const vram = info.vram as Record<string, number> | undefined;
  return (
    <div className="col">
      <div className="row wrap">
        <FishPill st={st} />
        {st?.managed && <span className="badge">started by ttsPIPE · pid {st.pid}</span>}
        <span className="right row">
          {(h === "offline" || h === "error") && <button className="btn primary sm" disabled={busy || !st?.venv_present} onClick={() => act("start")}>Start server</button>}
          {h !== "offline" && <button className="btn danger sm" disabled={busy} onClick={() => act("stop")}>Stop server</button>}
        </span>
      </div>
      {(h === "loading" || h === "starting") && <div className="notice small">Loading the 4B model onto the GPU takes about 1–2 minutes{st?.started_s_ago != null ? ` (${Math.round(st.started_s_ago)} s so far)` : ""}. The engine becomes ready by itself.</div>}
      {st?.error && <div className="notice err small">{st.error}</div>}
      {err && <div className="notice err small">{err}</div>}
      {st && !st.venv_present && <div className="notice warn small">No Fish server found{st.dir ? ` in ${st.dir}` : ""}. Provision it with that repo's <span className="mono">./run.sh</span>, or set <span className="mono">TTSPIPE_FISH_DIR</span>.</div>}
      {st && st.venv_present && !st.checkpoint_present && <div className="notice warn small">The S2-Pro checkpoint (checkpoints/s2-pro-nf4) is missing; the Fish repo's <span className="mono">./run.sh</span> downloads it (~4.9 GB).</div>}
      {!compact && (
        <div className="kv small">
          <span className="k">URL</span><span className="v">{st?.url ?? "—"}</span><span />
          <span className="k">Repo</span><span className="v ellipsis" title={st?.dir ?? ""}>{st?.dir ?? "—"}</span><span />
          {info.device && <><span className="k">Device</span><span className="v">{String(info.device)} · {String(info.quantization ?? "")} {String(info.precision ?? "")}</span><span /></>}
          {info.sample_rate && <><span className="k">Sample rate</span><span className="v">{String(info.sample_rate)} Hz</span><span /></>}
          {info.load_seconds != null && <><span className="k">Model load</span><span className="v">{fmt(Number(info.load_seconds), 1, " s")}</span><span /></>}
          {vram && Object.keys(vram).length > 0 && <><span className="k">VRAM</span><span className="v">{Object.entries(vram).slice(0, 3).map(([k, v]) => `${k} ${fmt(Number(v), 0)}`).join(" · ")}</span><span /></>}
          {info.total_requests != null && <><span className="k">Requests</span><span className="v">{String(info.total_requests)} ({String(info.total_failed ?? 0)} failed) · {fmt(Number(info.total_audio_seconds ?? 0), 0, " s audio")}</span><span /></>}
          {info.last_ttfa_ms != null && <><span className="k">Last TTFA (server)</span><span className="v">{fmtMs(Number(info.last_ttfa_ms))}</span><span /></>}
          {info.active_voice && <><span className="k">Server active voice</span><span className="v ellipsis">{String((info.active_voice as Record<string, unknown>).filename ?? (info.active_voice as Record<string, unknown>).voice_id ?? "set")}</span><span /></>}
        </div>
      )}
      {!compact && (
        <div>
          <button className="btn sm" onClick={() => setShowLog(!showLog)}>{showLog ? "Hide" : "Show"} server log</button>
          {showLog && <pre className="logbox">{(st?.log_tail ?? []).join("\n") || "(empty)"}</pre>}
        </div>
      )}
    </div>
  );
}

export function sampleText(lang?: string) {
  if (lang === "ta") return "வணக்கம்! உங்கள் ஆர்டர் அனுப்பப்பட்டது, வியாழக்கிழமைக்குள் வந்து சேரும்.";
  if (lang === "en") return "Hello! Your order has shipped and will arrive by Thursday.";
  return "வணக்கம்! Your order has shipped, வியாழக்கிழமைக்குள் வந்து சேரும்.";
}

export const download = (url: string, name: string) => {
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
};
