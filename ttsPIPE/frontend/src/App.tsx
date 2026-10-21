import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, MOCK } from "./api/client";
import { refreshAll, startMonitor } from "./api/live";
import { ConfigDrawer } from "./components/ConfigDrawer";
import { PAGES, TopBar } from "./components/TopBar";
import { EnginesPage } from "./pages/EnginesPage";
import { TestPage } from "./pages/TestPage";
import { VoicesPage } from "./pages/VoicesPage";
import { Grid, resolve, type PanelDef } from "./layout/grid";
import { PRESETS } from "./layout/presets";
import { DuplexCard, LatencyCard, PlaybackCard, SegmentsCard, VoiceCard } from "./panels/Cards";
import { ComparePanel } from "./panels/Compare";
import { DiagnosticsPanel } from "./panels/Diagnostics";
import { PipelineGraphPanel } from "./panels/PipelineGraph";
import { TimelineControls, TimelinePanel } from "./panels/Timeline";
import { UtterancesControls, UtterancesPanel } from "./panels/Utterances";
import { store, useTopic } from "./state/store";
import type { Kind, LayoutPanel } from "./types/contract";

const LAYOUT_NAME = "last";
const pageFromHash = () => { const h = location.hash.replace(/^#\/?/, ""); return PAGES.some(([id]) => id === h) ? h : "test"; };
const clone = <T,>(x: T): T => JSON.parse(JSON.stringify(x));

export default function App() {
  useTopic("conn", "toast");
  const [page, setPageState] = useState(pageFromHash);
  const setPage = useCallback((p: string) => { setPageState(p); if (location.hash !== `#/${p}`) history.pushState(null, "", `#/${p}`); }, []);
  useEffect(() => { const h = () => setPageState(pageFromHash()); window.addEventListener("popstate", h); window.addEventListener("hashchange", h); return () => { window.removeEventListener("popstate", h); window.removeEventListener("hashchange", h); }; }, []);
  const [preset, setPreset] = useState("operator");
  const [panels, setPanels] = useState<LayoutPanel[]>(() => clone(PRESETS.operator.panels!));
  const [editing, setEditing] = useState(false);
  const [graphEdit, setGraphEdit] = useState(false);
  const [drawer, setDrawer] = useState<{ kind?: Kind } | null>(null);
  const loaded = useRef(false);
  const saveTimer = useRef<number>(0);

  useEffect(() => {
    startMonitor();
    refreshAll();
    api.layout(LAYOUT_NAME).then((l) => {
      if (l?.panels?.length) {
        // keep panels added in newer versions
        const known = new Set(l.panels.map((p) => p.id));
        setPanels(resolve([...l.panels, ...PRESETS.operator.panels!.filter((p) => !known.has(p.id))]));
        if (l.name && PRESETS[l.name]) setPreset(l.name);
      }
    }).catch(() => undefined).finally(() => { loaded.current = true; });
    const iv = window.setInterval(() => { if (!store.connected && !MOCK) refreshAll(); }, 5000);
    return () => window.clearInterval(iv);
  }, []);

  const onChange = useCallback((p: LayoutPanel[]) => {
    setPanels(p);
    if (!loaded.current) return;
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(() => api.putLayout(LAYOUT_NAME, { cols: 12, row_h: 28, panels: p }).catch(() => undefined), 800);
  }, []);

  const defs: Record<string, PanelDef> = useMemo(() => ({
    graph: {
      id: "graph", title: "Pipeline", minW: 4, minH: 5, render: () => <PipelineGraphPanel edit={graphEdit} onOpenStage={(k) => setDrawer({ kind: k })} />,
      extra: () => (
        <div className="row">
          <button className={`btn sm${!graphEdit ? " on" : ""}`} onClick={() => setGraphEdit(false)}>Live</button>
          <button className={`btn sm${graphEdit ? " on" : ""}`} onClick={() => setGraphEdit(true)}>Edit</button>
        </div>
      ),
    },
    timeline: { id: "timeline", title: "Live output signal", minW: 4, minH: 5, flush: true, render: () => <TimelinePanel />, extra: () => <TimelineControls /> },
    playback: { id: "playback", title: "Playback state", minW: 2, minH: 5, render: () => <PlaybackCard /> },
    segments: { id: "segments", title: "Language segments", minW: 2, minH: 3, render: () => <SegmentsCard /> },
    voice: { id: "voice", title: "Voice", minW: 2, minH: 4, render: () => <VoiceCard /> },
    duplex: { id: "duplex", title: "Duplex", minW: 2, minH: 4, render: () => <DuplexCard /> },
    latency: { id: "latency", title: "Latency", minW: 2, minH: 5, render: () => <LatencyCard /> },
    utterances: { id: "utterances", title: "Utterances", minW: 3, minH: 4, render: () => <UtterancesPanel />, extra: () => <UtterancesControls /> },
    compare: { id: "compare", title: "A/B compare", minW: 3, minH: 4, render: () => <ComparePanel /> },
    diagnostics: { id: "diagnostics", title: "Diagnostics", minW: 3, minH: 5, flush: true, render: () => <DiagnosticsPanel /> },
  }), [graphEdit]);

  // when the studio opens, give the graph room
  useEffect(() => {
    if (!graphEdit) return;
    setPanels((ps) => ps.map((p) => (p.id === "graph" && p.h < 12 ? { ...p, h: 12 } : p)));
  }, [graphEdit]);

  const titles = Object.fromEntries(Object.values(defs).map((d) => [d.id, d.title]));
  return (
    <div className="app">
      <TopBar page={page} onPage={setPage} panels={panels} titles={titles} editing={editing} preset={preset} presets={Object.keys(PRESETS)}
        onPreset={(n) => { setPreset(n); onChange(clone(PRESETS[n].panels!)); }}
        onTogglePanel={(id) => onChange(resolve(panels.map((p) => (p.id === id ? { ...p, hidden: !p.hidden } : p)), id))}
        onEdit={() => setEditing(!editing)} onReset={() => onChange(clone(PRESETS[preset].panels!))} onConfig={() => setDrawer({})} />
      {MOCK && <div className="banner mock">Mock mode: synthetic event stream and API — no backend needed. Remove <span className="mono">?mock=1</span> for live data.</div>}
      {!MOCK && !store.connected && <div className="banner">Backend offline — retrying (attempt {store.retries}). Start it with <span className="mono">./run.sh dev</span>.</div>}
      {page === "test" && <TestPage onNavigate={setPage} />}
      {page === "engines" && <EnginesPage onNavigate={setPage} />}
      {page === "voices" && <VoicesPage onNavigate={setPage} />}
      {page === "monitor" && <div className="grid-wrap"><Grid panels={panels} defs={defs} editing={editing} onChange={onChange} /></div>}
      {drawer && <ConfigDrawer initialKind={drawer.kind} onClose={() => setDrawer(null)} />}
      {store.toastMsg && <div className="toast">{store.toastMsg}</div>}
    </div>
  );
}
