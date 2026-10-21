import { useEffect, useMemo, useRef, useState } from "react";
import { api, errMsg } from "../api/client";
import { refreshAll } from "../api/live";
import { fmtMs, SchemaForm, StatusDot } from "../components/common";
import { store, useTopic } from "../state/store";
import type { Config, Kind, ModuleEntry, ProfileEntry, StageStatus, ValidateResult } from "../types/contract";
import { KINDS } from "../types/contract";

const W = 124, H = 56;
const DEFAULT_POS: Record<string, { x: number; y: number }> = {
  text_in: { x: 16, y: 14 }, ingest: { x: 166, y: 14 }, normalize: { x: 316, y: 14 }, segment: { x: 466, y: 14 }, chunker: { x: 616, y: 14 },
  g2p: { x: 616, y: 104 }, prosody: { x: 466, y: 104 }, tts: { x: 316, y: 104 }, post: { x: 166, y: 104 }, audio_out: { x: 16, y: 104 },
  shadow_tts: { x: 316, y: 190 },
};
const CHAIN = ["text_in", ...KINDS, "audio_out"];
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

function edgePath(a: { x: number; y: number }, b: { x: number; y: number }) {
  const ac = { x: a.x + W / 2, y: a.y + H / 2 }, bc = { x: b.x + W / 2, y: b.y + H / 2 };
  if (Math.abs(ac.y - bc.y) < 5) {
    const dir = bc.x > ac.x ? 1 : -1;
    return `M${ac.x + (dir * W) / 2},${ac.y} L${bc.x - (dir * W) / 2},${bc.y}`;
  }
  if (Math.abs(ac.x - bc.x) < 5) {
    const dir = bc.y > ac.y ? 1 : -1;
    return `M${ac.x},${ac.y + (dir * H) / 2} L${bc.x},${bc.y - (dir * H) / 2}`;
  }
  return `M${ac.x},${ac.y + H / 2} C${ac.x},${bc.y} ${bc.x - W / 2 - 30},${bc.y} ${bc.x - W / 2},${bc.y}`;
}

interface GraphProps {
  stages: Partial<Record<Kind, StageStatus>>;
  shadow?: { impl?: string | null; st?: StageStatus | null } | null;
  positions: Record<string, { x: number; y: number }>;
  selected?: string | null;
  onSelect?: (id: string) => void;
  onMove?: (id: string, p: { x: number; y: number }) => void;
  live: boolean;
}

function GraphSvg({ stages, shadow, positions, selected, onSelect, onMove, live }: GraphProps) {
  useTopic("pipeline", "level");
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<{ id: string; dx: number; dy: number } | null>(null);
  const [, tick] = useState(0);
  useEffect(() => { const iv = setInterval(() => tick((x) => x + 1), 250); return () => clearInterval(iv); }, []);
  const pos = (id: string) => positions[id] ?? DEFAULT_POS[id] ?? { x: 0, y: 0 };
  const now = performance.now();
  const active = (k: string) => { const a = store.activity[k]; return !!a && now - a.t < 500; };
  const sess = store.current();
  const speaking = sess?.state === "speaking" || sess?.state === "ducked";
  const flowInto = (id: string) => (id === "audio_out" ? speaking : id === "shadow_tts" ? active("shadow") : active(id));

  const toSvg = (e: React.PointerEvent) => {
    const svg = svgRef.current!;
    const pt = svg.createSVGPoint();
    pt.x = e.clientX; pt.y = e.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM()!.inverse());
    return { x: p.x, y: p.y };
  };

  const nodes = [...CHAIN, ...(shadow ? ["shadow_tts"] : [])];
  const edges: [string, string, boolean][] = CHAIN.slice(1).map((b, i) => [CHAIN[i], b, false]);
  if (shadow) edges.push(["prosody", "shadow_tts", true]);
  return (
    <svg ref={svgRef} className="graph-svg" viewBox="0 0 756 252" preserveAspectRatio="xMidYMid meet"
      onPointerMove={(e) => { if (drag && onMove) { const p = toSvg(e); onMove(drag.id, { x: Math.round(p.x - drag.dx), y: Math.round(p.y - drag.dy) }); } }}
      onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="var(--border-strong)" />
        </marker>
      </defs>
      {edges.map(([a, b, dashed]) => (
        <path key={`${a}-${b}`} d={edgePath(pos(a), pos(b))} className={`gedge${dashed ? " dashed" : ""}${live && flowInto(b) ? " flow" : ""}`} markerEnd="url(#arr)" />
      ))}
      {nodes.map((id) => {
        const p = pos(id);
        const endpoint = id === "text_in" || id === "audio_out";
        const st = id === "shadow_tts" ? shadow?.st ?? null : (stages as Record<string, StageStatus>)[id];
        const status = st?.enabled === false ? "disabled" : st?.status ?? "unloaded";
        const label = id === "text_in" ? "TEXT IN" : id === "audio_out" ? "AUDIO OUT" : id === "shadow_tts" ? "SHADOW TTS" : id.toUpperCase();
        const impl = id === "shadow_tts" ? shadow?.impl : st?.impl;
        const color = { ready: "var(--ok)", loading: "var(--loading)", unloaded: "var(--unloaded)", error: "var(--err)", disabled: "var(--disabled)" }[status] ?? "var(--disabled)";
        const act = store.activity[id === "shadow_tts" ? "shadow" : id];
        const busy = st?.busy || (act && now - act.t < 300);
        return (
          <g key={id} className={`gnode${selected === id ? " sel" : ""}${endpoint ? " endpoint" : ""}`} transform={`translate(${p.x},${p.y})`}
            style={{ cursor: onMove ? "grab" : onSelect ? "pointer" : "default" }}
            onPointerDown={(e) => { if (onMove) { const q = toSvg(e); setDrag({ id, dx: q.x - p.x, dy: q.y - p.y }); } }}
            onClick={() => onSelect?.(id)}>
            <rect width={W} height={H} rx={7} strokeDasharray={id === "shadow_tts" ? "5 3" : undefined} style={busy ? { stroke: "var(--accent)", strokeWidth: 2 } : undefined} />
            <text x={10} y={17} fontWeight={600}>{label}</text>
            {!endpoint && <circle cx={W - 12} cy={13} r={5} fill={color}><title>{status}{st?.error ? ` — ${st.error}` : ""}</title></circle>}
            {st?.fallback && <text x={W - 30} y={17} fill="var(--warn)" fontSize={11}>▲<title>fallback: {st.error}</title></text>}
            {!endpoint && <text className="impl" x={10} y={33}>{impl ?? "—"}</text>}
            {!endpoint && st && <text className="lat" x={10} y={48}>{fmtMs(st.last_ms ?? act?.ms ?? null)} · p95 {fmtMs(st.p95_ms)}</text>}
            {id === "text_in" && <text className="lat" x={10} y={36}>LLM deltas</text>}
            {id === "audio_out" && <text className="lat" x={10} y={36}>{store.pipeline?.config?.audio?.out_rate ?? 24000} Hz PCM16</text>}
          </g>
        );
      })}
    </svg>
  );
}

export function PipelineGraphPanel({ edit, onOpenStage }: { edit: boolean; onOpenStage: (k: Kind) => void }) {
  useTopic("pipeline");
  const pipe = store.pipeline;
  if (!pipe) return <div className="muted small">Loading pipeline…</div>;
  if (edit) return <Studio />;
  const sh = pipe.config?.graph?.shadows?.tts;
  return (
    <GraphSvg stages={pipe.stages ?? {}} shadow={sh ? { impl: sh.impl, st: pipe.shadow } : null}
      positions={(pipe.graph?.positions ?? {}) as Record<string, { x: number; y: number }>} live
      onSelect={(id) => (KINDS as string[]).includes(id) && onOpenStage(id as Kind)} />
  );
}

// ---- Edit mode studio ---------------------------------------------------------------------------

function Studio() {
  useTopic("pipeline", "modules");
  const pipe = store.pipeline!;
  const [draft, setDraft] = useState<Config>(() => clone(pipe.config ?? {}));
  const [sel, setSel] = useState<string>("tts");
  const [result, setResult] = useState<ValidateResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [profiles, setProfiles] = useState<ProfileEntry[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const baseline = useMemo(() => JSON.stringify(pipe.config), [pipe.config]);
  const dirty = JSON.stringify(draft) !== baseline;
  const loadProfiles = () => api.profiles().then(setProfiles).catch(() => undefined);
  useEffect(() => { loadProfiles(); }, []);

  const mods = store.modules;
  const selKind: Kind | null = sel === "shadow_tts" ? "tts" : (KINDS as string[]).includes(sel) ? (sel as Kind) : null;
  const isShadow = sel === "shadow_tts";
  const stageCfg = selKind ? (isShadow ? draft.graph?.shadows?.tts ?? null : draft.stages?.[selKind]) : null;
  const mod = selKind && stageCfg ? mods.find((m) => m.id === `${selKind}/${stageCfg.impl}`) : undefined;

  const update = (fn: (c: Config) => void) => { const c = clone(draft); fn(c); setDraft(c); setResult(null); };
  const setImpl = (impl: string) => update((c) => {
    if (!selKind) return;
    if (isShadow) { c.graph = c.graph ?? {}; c.graph.shadows = { tts: impl ? { impl, params: {}, voice_id: null } : null }; return; }
    c.stages![selKind] = { ...c.stages![selKind], impl, params: {} };
    if (selKind === "tts") c.voice = { ...c.voice, impl, voice_id: mods.find((m) => m.id === `tts/${impl}`)?.voices?.[0]?.id ?? "" };
  });
  const setParam = (name: string, v: unknown) => update((c) => {
    if (!selKind) return;
    const target = isShadow ? c.graph?.shadows?.tts : c.stages![selKind];
    if (target) target.params = { ...(target.params ?? {}), [name]: v };
  });

  const validate = async () => {
    setBusy(true);
    try { const r = await api.validate(draft); setResult(r); return r; } catch (e) { setMsg(errMsg(e)); return null; } finally { setBusy(false); }
  };
  const apply = async () => {
    const r = await validate();
    if (!r?.ok) return;
    if (r.warnings?.length && !confirm(`Apply with warnings?\n\n${r.warnings.join("\n")}`)) return;
    setBusy(true);
    try { store.pipeline = await api.putPipeline(draft); store.touch("pipeline"); setMsg("Applied."); refreshAll(); }
    catch (e) { setMsg(errMsg(e)); } finally { setBusy(false); }
  };
  const applyProfile = async (name: string) => {
    try { const p = await api.applyProfile(name); store.pipeline = p; store.touch("pipeline"); setDraft(clone(p.config ?? {})); setMsg(`Profile ${name} applied.`); }
    catch (e) { setMsg(errMsg(e)); }
  };
  const saveAs = async () => {
    const name = prompt("Save current draft as profile name:");
    if (!name) return;
    try { await api.putProfile(name, { description: "saved from studio", config: draft }); await loadProfiles(); setMsg(`Saved ${name}.`); } catch (e) { setMsg(errMsg(e)); }
  };
  const exportProfile = () => {
    const blob = new Blob([JSON.stringify({ description: "exported", config: draft }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `ttspipe-profile-${pipe.active_profile ?? "custom"}.json`;
    a.click();
  };
  const importProfile = async (f: File) => {
    try {
      const j = JSON.parse(await f.text());
      const name = prompt("Import as profile name:", f.name.replace(/\.json$/, "").replace(/[^\w.-]/g, "_"));
      if (!name) return;
      await api.putProfile(name, { description: j.description ?? "imported", config: j.config ?? j });
      await loadProfiles();
      setMsg(`Imported ${name}.`);
    } catch (e) { setMsg(errMsg(e)); }
  };

  const positions = (draft.graph?.positions ?? pipe.graph?.positions ?? {}) as Record<string, { x: number; y: number }>;
  const sh = draft.graph?.shadows?.tts;
  return (
    <div className="col" style={{ height: "100%" }}>
      <div className="profiles-bar">
        <span className="muted small">Profile</span>
        <select value={pipe.active_profile ?? ""} onChange={(e) => applyProfile(e.target.value)} style={{ minWidth: 140 }}>
          <option value="" disabled>(none)</option>
          {profiles.map((p) => <option key={p.name} value={p.name}>{p.name}{p.builtin ? "" : " (user)"}</option>)}
        </select>
        {(dirty || pipe.dirty) && <span className="badge warn" title="Draft or live config differs from the profile">● modified</span>}
        <button className="btn sm" onClick={saveAs}>Save as…</button>
        <button className="btn sm" onClick={exportProfile}>Export</button>
        <label className="btn sm">Import<input type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importProfile(e.target.files[0])} /></label>
        <span className="right" />
        <button className="btn sm" disabled={!dirty} onClick={() => { setDraft(clone(pipe.config ?? {})); setResult(null); }}>Revert</button>
        <button className="btn sm" disabled={busy} onClick={validate}>Validate</button>
        <button className="btn sm primary" disabled={busy || !dirty} onClick={apply}>Apply</button>
      </div>
      {msg && <div className="notice small" onClick={() => setMsg(null)}>{msg}</div>}
      {result && (
        <div className={`notice small ${result.ok ? (result.warnings?.length ? "warn" : "ok") : "err"}`}>
          {result.ok ? "Valid." : "Invalid."} Estimate: download {result.estimate?.download_mb ?? 0} MB · RAM {result.estimate?.ram_mb ?? 0} MB
          {[...(result.errors ?? []), ...(result.warnings ?? [])].map((x, i) => <div key={i}>• {x}</div>)}
        </div>
      )}
      <div className="studio grow" style={{ minHeight: 0 }}>
        <div className="palette">
          {KINDS.map((k) => (
            <div key={k}>
              <div className="section-title">{k}</div>
              {mods.filter((m) => m.kind === k).map((m) => (
                <div key={m.id} className={`mod${m.available === false ? " unavail" : ""}`} title={`${m.description ?? ""}\n${m.available === false ? m.reason : ""}`}
                  onClick={() => { setSel(k); update((c) => { c.stages![k] = { ...c.stages![k], impl: m.id!.split("/")[1], params: {} }; if (k === "tts") c.voice = { ...c.voice, impl: m.id!.split("/")[1], voice_id: m.voices?.[0]?.id ?? "" }; }); }}>
                  <div className="row"><span className="mono small grow ellipsis">{m.id!.split("/")[1]}</span>{m.available === false && <span className="tri">▲</span>}</div>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div style={{ border: "1px solid var(--border)", borderRadius: 5, minHeight: 0 }}>
          <GraphSvg stages={Object.fromEntries(KINDS.map((k) => [k, { ...pipe.stages?.[k], impl: draft.stages?.[k]?.impl, enabled: draft.stages?.[k]?.enabled }])) as Partial<Record<Kind, StageStatus>>}
            shadow={sh ? { impl: sh.impl, st: pipe.shadow } : null} positions={positions} selected={sel} live={false}
            onSelect={setSel}
            onMove={(id, p) => update((c) => { c.graph = c.graph ?? {}; c.graph.positions = { ...(c.graph.positions ?? {}), [id]: p }; })} />
        </div>
        <div className="inspector">
          {!selKind ? <div className="muted small">Select a stage node.</div> : (
            <Inspector kind={selKind} isShadow={isShadow} cfg={stageCfg ?? null} mod={mod} mods={mods.filter((m) => m.kind === selKind)}
              live={isShadow ? pipe.shadow ?? null : pipe.stages?.[selKind] ?? null}
              setImpl={setImpl} setParam={setParam}
              setEnabled={(v) => update((c) => { c.stages![selKind] = { ...c.stages![selKind], enabled: v }; })}
              resetParams={() => update((c) => { const t = isShadow ? c.graph?.shadows?.tts : c.stages![selKind]; if (t) t.params = {}; })} />
          )}
          {selKind === "tts" && !isShadow && (
            <div style={{ marginTop: 10 }}>
              <div className="section-title">Shadow TTS (A/B)</div>
              <select value={sh?.impl ?? ""} onChange={(e) => update((c) => { c.graph = c.graph ?? {}; c.graph.shadows = { tts: e.target.value ? { impl: e.target.value, params: {}, voice_id: null } : null }; })}>
                <option value="">none</option>
                {mods.filter((m) => m.kind === "tts").map((m) => <option key={m.id} value={m.id!.split("/")[1]}>{m.id}</option>)}
              </select>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Inspector({ kind, isShadow, cfg, mod, mods, live, setImpl, setParam, setEnabled, resetParams }: {
  kind: Kind; isShadow: boolean; cfg: { impl?: string; params?: Record<string, unknown>; enabled?: boolean } | null; mod?: ModuleEntry; mods: ModuleEntry[];
  live: StageStatus | null; setImpl: (i: string) => void; setParam: (n: string, v: unknown) => void; setEnabled: (v: boolean) => void; resetParams: () => void;
}) {
  return (
    <div className="col">
      <div className="row"><b>{isShadow ? "SHADOW TTS" : kind.toUpperCase()}</b><StatusDot st={live} /><span className="muted small right">{live?.impl}</span></div>
      <select value={cfg?.impl ?? ""} onChange={(e) => setImpl(e.target.value)}>
        {isShadow && <option value="">none</option>}
        {mods.map((m) => <option key={m.id} value={m.id!.split("/")[1]}>{m.id}{m.available === false ? " (unavailable)" : ""}</option>)}
      </select>
      {!isShadow && !["ingest", "chunker", "tts"].includes(kind) && (
        <label className="check"><input type="checkbox" checked={cfg?.enabled !== false} onChange={(e) => setEnabled(e.target.checked)} /> enabled</label>
      )}
      {mod && (
        <div className="small">
          <div className="muted">{mod.description}</div>
          <div>licence <span className="mono">{mod.license}</span> · {mod.size_mb ? `${mod.size_mb} MB` : "no download"}{mod.ram_mb ? ` · ~${mod.ram_mb} MB RAM` : ""}{mod.streaming ? " · streaming" : ""}</div>
          {mod.languages?.length ? <div>languages: {mod.languages.join(", ")}</div> : null}
          {mod.available === false && <div className="notice warn" style={{ marginTop: 4 }}>{mod.reason}</div>}
        </div>
      )}
      {live?.error && <div className="notice err small">{live.error}</div>}
      <div className="row"><div className="section-title grow">Parameters</div><button className="btn sm" onClick={resetParams}>Reset to defaults</button></div>
      {mod ? <SchemaForm schema={mod.params_schema ?? {}} values={{ ...(mod.default_params ?? {}), ...(cfg?.params ?? {}) }} onChange={setParam} /> : <div className="muted small">Unknown module.</div>}
    </div>
  );
}
