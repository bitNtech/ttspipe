// Draggable / resizable 12-column grid with vertical compaction, collapse, maximize and close.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { LayoutPanel } from "../types/contract";

export interface PanelDef {
  id: string;
  title: string;
  render: () => ReactNode;
  minW?: number;
  minH?: number;
  flush?: boolean;
  extra?: () => ReactNode; // header controls
}

export const COLS = 12;
export const ROW_H = 28;
export const GAP = 6;

const overlap = (a: LayoutPanel, b: LayoutPanel) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

export function effective(p: LayoutPanel): LayoutPanel {
  return p.collapsed ? { ...p, h: 1 } : p;
}

/** Push panels that collide with `moved` downwards, then compact everything upwards. */
export function resolve(panels: LayoutPanel[], movedId?: string): LayoutPanel[] {
  const vis = panels.filter((p) => !p.hidden).map((p) => ({ ...p }));
  const hidden = panels.filter((p) => p.hidden);
  const moved = vis.find((p) => p.id === movedId);
  const order = vis.filter((p) => p !== moved).sort((a, b) => a.y - b.y || a.x - b.x);
  const placed: LayoutPanel[] = moved ? [moved] : [];
  for (const p of order) {
    // compact up
    let y = 0;
    while (true) {
      const cand = { ...effective(p), y };
      const hit = placed.find((q) => overlap(cand, effective(q)));
      if (!hit) break;
      y = effective(hit).y + effective(hit).h;
    }
    p.y = y;
    placed.push(p);
  }
  if (moved) {
    // the moved panel also compacts upwards if there is room
    const others = placed.filter((p) => p !== moved);
    let y = moved.y;
    while (y > 0 && !others.some((q) => overlap({ ...effective(moved), y: y - 1 }, effective(q)))) y--;
    moved.y = y;
  }
  return [...placed, ...hidden];
}

interface GridProps {
  panels: LayoutPanel[];
  defs: Record<string, PanelDef>;
  editing: boolean;
  onChange: (p: LayoutPanel[]) => void;
}

export function Grid({ panels, defs, editing, onChange }: GridProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(1200);
  const [drag, setDrag] = useState<{ id: string; mode: "move" | "resize"; sx: number; sy: number; orig: LayoutPanel; cur: LayoutPanel } | null>(null);
  const [maxed, setMaxed] = useState<string | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    setWidth(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const colW = (width - GAP * (COLS - 1)) / COLS;
  const px = (p: LayoutPanel) => ({
    left: p.x * (colW + GAP), top: p.y * (ROW_H + GAP),
    width: p.w * colW + (p.w - 1) * GAP, height: p.h * ROW_H + (p.h - 1) * GAP,
  });

  useEffect(() => {
    if (!drag) return;
    const move = (e: PointerEvent) => {
      const dx = Math.round((e.clientX - drag.sx) / (colW + GAP));
      const dy = Math.round((e.clientY - drag.sy) / (ROW_H + GAP));
      const def = defs[drag.id];
      const o = drag.orig;
      const cur = drag.mode === "move"
        ? { ...o, x: Math.max(0, Math.min(COLS - o.w, o.x + dx)), y: Math.max(0, o.y + dy) }
        : { ...o, w: Math.max(def?.minW ?? 2, Math.min(COLS - o.x, o.w + dx)), h: Math.max(def?.minH ?? 3, o.h + dy) };
      setDrag({ ...drag, cur });
    };
    const up = () => {
      onChange(resolve(panels.map((p) => (p.id === drag.id ? drag.cur : p)), drag.id));
      setDrag(null);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
    return () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); };
  }, [drag, colW, panels, defs, onChange]);

  const visible = panels.filter((p) => !p.hidden && defs[p.id]);
  const bottom = Math.max(0, ...visible.map((p) => effective(p).y + effective(p).h));
  const set = (id: string, patch: Partial<LayoutPanel>) => onChange(resolve(panels.map((p) => (p.id === id ? { ...p, ...patch } : p)), id));

  return (
    <div ref={ref} className={`grid${editing ? " editing" : ""}`} style={{ height: bottom * (ROW_H + GAP) }}>
      {drag && <div className="ghost" style={px(drag.cur)} />}
      {visible.map((p) => {
        const def = defs[p.id];
        const e = effective(p);
        const isDrag = drag?.id === p.id;
        const box = isDrag && drag.mode === "move" ? { ...px(e), left: px(drag.cur).left, top: px(drag.cur).top }
          : isDrag ? px(drag.cur) : px(e);
        const isMax = maxed === p.id;
        return (
          <section key={p.id} className={`panel${isDrag ? " dragging" : ""}${isMax ? " max" : ""}`} style={isMax ? undefined : box}>
            <header className="panel-head"
              onPointerDown={(ev) => {
                if (!editing || isMax || (ev.target as HTMLElement).closest("button,select,input")) return;
                setDrag({ id: p.id, mode: "move", sx: ev.clientX, sy: ev.clientY, orig: p, cur: p });
              }}>
              <span className="title">{def.title}</span>
              <div className="row grow" style={{ justifyContent: "flex-end", minWidth: 0 }}>{!p.collapsed && def.extra?.()}</div>
              <button className="icon-btn" title={p.collapsed ? "Expand" : "Collapse"} onClick={() => set(p.id, { collapsed: !p.collapsed })}>{p.collapsed ? "▸" : "▾"}</button>
              <button className="icon-btn" title={isMax ? "Restore" : "Maximize"} onClick={() => setMaxed(isMax ? null : p.id)}>{isMax ? "❐" : "□"}</button>
              <button className="icon-btn" title="Close (re-open from Panels)" onClick={() => { setMaxed(null); set(p.id, { hidden: true }); }}>✕</button>
            </header>
            {!p.collapsed && <div className={`panel-body${def.flush ? " flush" : ""}`}>{def.render()}</div>}
            {editing && !p.collapsed && !isMax && (
              <div className="resize" onPointerDown={(ev) => { ev.stopPropagation(); setDrag({ id: p.id, mode: "resize", sx: ev.clientX, sy: ev.clientY, orig: p, cur: p }); }} />
            )}
          </section>
        );
      })}
    </div>
  );
}
