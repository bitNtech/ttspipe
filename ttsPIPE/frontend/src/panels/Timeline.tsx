import { useEffect, useRef, useState } from "react";
import { store } from "../state/store";
import type { PlaybackStateName } from "../types/contract";

const LANES = [
  { id: "wave", label: "output", h: 0.32 },
  { id: "chunks", label: "chunks", h: 0.2 },
  { id: "synth", label: "synth", h: 0.22 },
  { id: "state", label: "state", h: 0.1 },
  { id: "marks", label: "events", h: 0.16 },
];
const STATE_VAR: Record<PlaybackStateName, string> = {
  idle: "--panel-2", buffering: "--warn", synthesizing: "--loading", speaking: "--ok", ducked: "--accent", interrupted: "--err", done: "--text-3",
};
const MARK_VAR: Record<string, string> = { duck: "--accent", resume: "--ok", stop: "--err", underrun: "--warn", error: "--err" };

export const timelineWindow = { sec: 20 };

export function TimelineControls() {
  const [sec, setSec] = useState(timelineWindow.sec);
  return (
    <select value={sec} onChange={(e) => { timelineWindow.sec = Number(e.target.value); setSec(timelineWindow.sec); }} title="Visible window">
      {[5, 10, 20, 60, 120].map((s) => <option key={s} value={s}>{s}s</option>)}
    </select>
  );
}

export function TimelinePanel() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    let raf = 0;
    let colors: Record<string, string> = {};
    let theme = "";
    const css = (v: string) => colors[v] ?? (colors[v] = getComputedStyle(document.documentElement).getPropertyValue(v).trim() || "#888");
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const cv = ref.current;
      if (!cv) return;
      const th = document.documentElement.dataset.theme ?? "";
      if (th !== theme) { theme = th; colors = {}; }
      const dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth, H = cv.clientHeight;
      if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
      const g = cv.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, W, H);
      const s = store.current();
      const LBL = 52;
      const plotW = W - LBL - 6;
      const win = timelineWindow.sec * 1000;
      let tNow = 0;
      if (s) tNow = s.t_out_ms + (s.connected && store.connected ? Math.min(Date.now() - s.last_ts, 60000) : 0);
      const t1 = Math.max(tNow, win);
      const t0 = Math.max(0, t1 - win); // never show time before t=0
      const X = (t: number) => LBL + ((t - t0) / (t1 - t0)) * plotW;
      let y = 2;
      const lanes: Record<string, { y: number; h: number }> = {};
      for (const l of LANES) { const h = (H - 18) * l.h; lanes[l.id] = { y, h }; y += h; }
      g.font = "10px " + (getComputedStyle(cv).fontFamily || "sans-serif");
      // lane labels + separators
      for (const l of LANES) {
        const { y: ly, h } = lanes[l.id];
        g.fillStyle = css("--text-3");
        g.fillText(l.label, 4, ly + h / 2 + 3);
        g.strokeStyle = css("--border");
        g.beginPath(); g.moveTo(LBL, ly + h); g.lineTo(W, ly + h); g.stroke();
      }
      // time axis
      g.fillStyle = css("--text-3");
      const step = win <= 10000 ? 1000 : win <= 30000 ? 2000 : win <= 60000 ? 5000 : 10000;
      for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
        const x = X(t);
        g.strokeStyle = css("--border"); g.globalAlpha = 0.5;
        g.beginPath(); g.moveTo(x, 2); g.lineTo(x, H - 16); g.stroke(); g.globalAlpha = 1;
        g.fillText(`${(t / 1000).toFixed(0)}s`, x - 6, H - 4);
      }
      if (!s) { g.fillStyle = css("--text-3"); g.fillText("no session yet — press Speak", LBL + 10, H / 2); return; }
      // waveform
      const wv = lanes.wave, mid = wv.y + wv.h / 2;
      g.fillStyle = css("--accent");
      for (const lv of s.levels) {
        if (lv.t < t0 - 100) continue;
        const x = X(lv.t);
        const top = mid - Math.min(1, lv.max) * (wv.h / 2 - 2), bot = mid - Math.max(-1, lv.min) * (wv.h / 2 - 2);
        g.fillRect(x, top, Math.max(1, plotW / (win / 40)), Math.max(1, bot - top));
      }
      // chunks (aligned to when their audio played)
      const ch = lanes.chunks, sy = lanes.synth;
      let row = 0;
      for (const c of s.chunks.values()) {
        const langVar = c.lang === "ta" ? "--ta" : c.lang === "en" ? "--en" : "--tl";
        if (c.t_play_start !== undefined) {
          const xa = X(c.t_play_start), xb = X(c.t_play_end ?? tNow);
          if (xb >= LBL && xa <= W) {
            g.fillStyle = css(langVar); g.globalAlpha = 0.22;
            g.fillRect(xa, ch.y + 3, Math.max(2, xb - xa), ch.h - 6);
            g.globalAlpha = 1; g.strokeStyle = css(langVar);
            g.strokeRect(xa + 0.5, ch.y + 3.5, Math.max(2, xb - xa) - 1, ch.h - 7);
            g.save(); g.beginPath(); g.rect(xa, ch.y, Math.max(0, xb - xa - 2), ch.h); g.clip();
            g.fillStyle = css("--text"); g.fillText(c.text, xa + 3, ch.y + ch.h / 2 + 3); g.restore();
          }
        }
        // synth bars: start → first audio (thin) → done (solid)
        if (c.t_synth_start !== undefined) {
          const rh = (sy.h - 6) / 2, ry = sy.y + 3 + (row % 2) * rh;
          row++;
          const xs = X(c.t_synth_start), xf = X(c.t_first_audio ?? tNow), xd = X(c.t_synth_done ?? tNow);
          if (xd >= LBL && xs <= W) {
            g.fillStyle = css(c.cached ? "--ok" : "--loading"); g.globalAlpha = 0.35;
            g.fillRect(xs, ry + rh * 0.3, Math.max(1, xf - xs), rh * 0.4);
            g.globalAlpha = 0.85;
            g.fillRect(xf, ry + 1, Math.max(1, xd - xf), rh - 2);
            g.globalAlpha = 1;
          }
        }
      }
      // playback state lane
      const st = lanes.state;
      for (let i = 0; i < s.states.length; i++) {
        const a = s.states[i], b = s.states[i + 1];
        const xa = X(a.t), xb = X(b ? b.t : tNow);
        if (xb < LBL || xa > W) continue;
        g.fillStyle = css(STATE_VAR[a.state] ?? "--panel-2");
        g.fillRect(Math.max(LBL, xa), st.y + 3, Math.max(1, xb - Math.max(LBL, xa)), st.h - 6);
      }
      // event markers
      const mk = lanes.marks;
      for (const m of s.markers) {
        if (m.t < t0) continue;
        const x = X(m.t);
        g.strokeStyle = css(MARK_VAR[m.kind] ?? "--text"); g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(x, 2); g.lineTo(x, mk.y + mk.h - 2); g.stroke(); g.lineWidth = 1;
        g.fillStyle = css(MARK_VAR[m.kind] ?? "--text");
        g.fillText(m.kind, x + 2, mk.y + mk.h / 2 + 3);
      }
      // playback cursor
      g.strokeStyle = css("--err"); g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(X(tNow), 2); g.lineTo(X(tNow), H - 16); g.stroke(); g.setLineDash([]);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);
  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div style={{ flex: 1, minHeight: 0 }}><canvas ref={ref} className="fill" /></div>
      <div className="legend">
        <span><i style={{ background: "var(--accent)" }} />waveform</span>
        <span><i style={{ background: "var(--ta)" }} />ta</span><span><i style={{ background: "var(--en)" }} />en</span><span><i style={{ background: "var(--tl)" }} />ta-Latn</span>
        <span><i style={{ background: "var(--loading)", opacity: 0.4 }} />synth→first audio</span><span><i style={{ background: "var(--loading)" }} />→done</span><span><i style={{ background: "var(--ok)" }} />cached / speaking</span>
        <span><i style={{ background: "var(--warn)" }} />buffering / underrun</span><span><i style={{ background: "var(--err)" }} />stop · cursor</span>
      </div>
    </div>
  );
}
