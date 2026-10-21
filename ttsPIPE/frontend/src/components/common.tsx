import { useState, type ReactNode } from "react";
import type { ParamSchema, ParamsSchema, StageStatus } from "../types/contract";

export const fmt = (v: number | null | undefined, digits = 0, unit = "") =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${v.toFixed(digits)}${unit}`;
export const fmtMs = (v: number | null | undefined) =>
  v === null || v === undefined ? "—" : v >= 1000 ? `${(v / 1000).toFixed(2)}s` : v < 10 ? `${v.toFixed(1)}ms` : `${Math.round(v)}ms`;

export function StatusDot({ st }: { st?: StageStatus | null }) {
  const s = st?.enabled === false ? "disabled" : st?.status ?? "unloaded";
  return (
    <span className="row" style={{ gap: 3 }}>
      <span className={`dot ${s}${st?.busy ? " busy" : ""}`} title={`${s}${st?.error ? ` — ${st.error}` : ""}`} />
      {st?.fallback && <span className="tri" title={`fallback: ${st?.error ?? ""}`}>▲</span>}
    </span>
  );
}

export const langClass = (l?: string) => (l === "ta" ? "ta" : l === "en" ? "en" : "tl");
export const langLabel = (l?: string) => (l === "ta-Latn" ? "ta-Latn" : l ?? "?");
export function LangChip({ lang }: { lang?: string }) {
  return <span className={`lang ${langClass(lang)}`}>{langLabel(lang)}</span>;
}

export function Sparkline({ data, w = 80, h = 18, color = "var(--accent)" }: { data: (number | null | undefined)[]; w?: number; h?: number; color?: string }) {
  const vals = data.filter((v): v is number => typeof v === "number");
  if (vals.length < 2) return <svg width={w} height={h} />;
  const max = Math.max(...vals) || 1;
  const step = w / (data.length - 1);
  const pts = data.map((v, i) => `${(i * step).toFixed(1)},${(h - 1 - ((v ?? 0) / max) * (h - 2)).toFixed(1)}`).join(" ");
  return <svg width={w} height={h}><polyline points={pts} fill="none" stroke={color} strokeWidth={1.3} /></svg>;
}

/** Word-level diff of raw → normalized: removed words struck red, inserted words green. */
export function WordDiff({ before, after }: { before: string; after: string }) {
  const a = before.split(/\s+/).filter(Boolean), b = after.split(/\s+/).filter(Boolean);
  const n = a.length, m = b.length;
  const L: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = a[i] === b[j] ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
  const out: ReactNode[] = [];
  let i = 0, j = 0, k = 0;
  let del: string[] = [], ins: string[] = [];
  const flush = () => {
    if (del.length) out.push(<del key={k++}>{del.join(" ")}</del>, " ");
    if (ins.length) out.push(<ins key={k++}>{ins.join(" ")}</ins>, " ");
    del = []; ins = [];
  };
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === b[j]) { flush(); out.push(a[i] + " "); i++; j++; }
    else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) ins.push(b[j++]);
    else del.push(a[i++]);
  }
  flush();
  return <>{out}</>;
}

function Field({ name, s, value, onChange }: { name: string; s: ParamSchema; value: unknown; onChange: (v: unknown) => void }) {
  const t = s.type;
  if (t === "boolean") return <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />;
  if (s.enum) {
    return (
      <select value={String(value ?? "")} onChange={(e) => {
        const raw = e.target.value;
        const match = s.enum!.find((x) => String(x) === raw);
        onChange(match ?? raw);
      }}>
        {s.enum.map((x) => <option key={String(x)} value={String(x)}>{String(x)}</option>)}
      </select>
    );
  }
  if (t === "number" || t === "integer") {
    const num = typeof value === "number" ? value : Number(s.default ?? 0);
    const step = t === "integer" ? 1 : Math.abs((s.max ?? 1) - (s.min ?? 0)) > 20 ? 0.5 : 0.01;
    const conv = (v: string) => (t === "integer" ? Math.round(Number(v)) : Number(v));
    return (
      <div className="ctl">
        {s.min !== undefined && s.max !== undefined && (
          <input type="range" min={s.min} max={s.max} step={step} value={num} onChange={(e) => onChange(conv(e.target.value))} aria-label={name} />
        )}
        <input type="number" min={s.min} max={s.max} step={step} value={num} onChange={(e) => onChange(conv(e.target.value))} />
        {s.unit && <span className="muted small">{s.unit}</span>}
      </div>
    );
  }
  if (t === "array") {
    return <input type="text" value={Array.isArray(value) ? value.join(", ") : ""} onChange={(e) => onChange(e.target.value.split(",").map((x) => x.trim()).filter(Boolean))} />;
  }
  return <input type="text" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} />;
}

/** Form generated from a params_schema (§4.3). */
export function SchemaForm({ schema, values, onChange, showAdvanced = false }: {
  schema: ParamsSchema; values: Record<string, unknown>; onChange: (name: string, v: unknown) => void; showAdvanced?: boolean;
}) {
  const [adv, setAdv] = useState(showAdvanced);
  const entries = Object.entries(schema ?? {});
  if (!entries.length) return <div className="muted small">No parameters.</div>;
  const hasAdv = entries.some(([, s]) => s.advanced);
  return (
    <div className="form">
      {entries.filter(([, s]) => adv || !s.advanced).map(([name, s]) => (
        <Row key={name} name={name} s={s}>
          <Field name={name} s={s} value={values?.[name] ?? s.default} onChange={(v) => onChange(name, v)} />
        </Row>
      ))}
      {hasAdv && <button className="btn sm" style={{ gridColumn: "1 / -1", justifySelf: "start" }} onClick={() => setAdv(!adv)}>{adv ? "Hide advanced" : "Show advanced"}</button>}
    </div>
  );
}

function Row({ name, s, children }: { name: string; s: ParamSchema; children: ReactNode }) {
  return (
    <>
      <label title={s.description}>{name}</label>
      {children}
      {s.description && <div className="desc">{s.description}</div>}
    </>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="muted small" style={{ padding: 8, textAlign: "center" }}>{children}</div>;
}
