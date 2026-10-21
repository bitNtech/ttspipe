import type { Layout, LayoutPanel } from "../types/contract";

const P = (id: string, x: number, y: number, w: number, h: number): LayoutPanel => ({ id, x, y, w, h });

// Operator: everything visible at 1600×1000 without scrolling (26 rows × 34 px + top bar).
export const PRESETS: Record<string, Layout> = {
  operator: {
    name: "operator", cols: 12, row_h: 28, panels: [
      P("graph", 0, 0, 7, 8), P("playback", 7, 0, 2, 8), P("segments", 9, 0, 3, 8),
      P("timeline", 0, 8, 7, 8), P("latency", 7, 8, 2, 8), P("voice", 9, 8, 3, 8),
      P("utterances", 0, 16, 5, 10), P("diagnostics", 5, 16, 4, 10), P("duplex", 9, 16, 3, 5), P("compare", 9, 21, 3, 5),
    ],
  },
  engineering: {
    name: "engineering", cols: 12, row_h: 28, panels: [
      P("graph", 0, 0, 8, 10), P("latency", 8, 0, 2, 10), P("playback", 10, 0, 2, 10),
      P("timeline", 0, 10, 12, 7), P("diagnostics", 0, 17, 7, 12), P("utterances", 7, 17, 5, 12),
      P("segments", 0, 29, 4, 6), P("voice", 4, 29, 4, 6), P("duplex", 8, 29, 4, 6), P("compare", 0, 35, 12, 7),
    ],
  },
  compact: {
    name: "compact", cols: 12, row_h: 28, panels: [
      P("timeline", 0, 0, 8, 8), P("playback", 8, 0, 4, 8), P("utterances", 0, 8, 8, 12), P("latency", 8, 8, 4, 6),
      P("duplex", 8, 14, 4, 6), { ...P("graph", 0, 20, 12, 8), hidden: true }, { ...P("segments", 0, 28, 4, 6), hidden: true },
      { ...P("voice", 4, 28, 4, 6), hidden: true }, { ...P("diagnostics", 0, 34, 12, 10), hidden: true }, { ...P("compare", 0, 44, 12, 7), hidden: true },
    ],
  },
};
