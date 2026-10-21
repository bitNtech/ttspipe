// REST client (ARCHITECTURE.md §9). In mock mode (?mock=1) every call is served by mock.ts.
import type {
  CompareResult, CompareVariant, Config, Health, Layout, LayoutEntry, LogRecord, Metrics, ModelEntry, ModuleEntry,
  PipelineView, PlaybackConfig, ChunkerConfig, Profile, ProfileEntry, RatingSummary, RenderEntry, RuntimeState,
  Script, SessionDetail, SessionSummary, StageStatus, ValidateResult, Voice, VoiceEntry, CloneVoice, FishStatus,
} from "../types/contract";
import { mockApi } from "./mock";

export const MOCK = new URLSearchParams(location.search).get("mock") === "1";

export class ApiError extends Error {
  status: number;
  constructor(status: number, msg: string) { super(msg); this.status = status; }
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method, headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail); } catch { /* not json */ }
    throw new ApiError(r.status, msg);
  }
  const ct = r.headers.get("content-type") ?? "";
  return (ct.includes("json") ? r.json() : r.blob()) as Promise<T>;
}

const realApi = {
  health: () => req<Health>("GET", "/api/health"),
  modules: () => req<ModuleEntry[]>("GET", "/api/modules"),
  pipeline: () => req<PipelineView>("GET", "/api/pipeline"),
  putStage: (kind: string, body: { impl?: string; params?: Record<string, unknown>; enabled?: boolean }) => req<StageStatus>("PUT", `/api/pipeline/stages/${kind}`, body),
  putPlayback: (body: PlaybackConfig & { out_rate?: number }) => req<PlaybackConfig>("PUT", "/api/pipeline/playback", body),
  putChunker: (body: ChunkerConfig) => req<ChunkerConfig>("PUT", "/api/pipeline/chunker", body),
  putVoice: (body: Voice) => req<Voice>("PUT", "/api/pipeline/voice", body),
  putShadow: (body: { impl: string | null; params?: Record<string, unknown>; voice_id?: string | null }) => req<PipelineView>("PUT", "/api/pipeline/shadow", body),
  putPositions: (body: Record<string, { x: number; y: number }>) => req<unknown>("PUT", "/api/pipeline/positions", body),
  putPipeline: (cfg: Config) => req<PipelineView>("PUT", "/api/pipeline", cfg),
  validate: (cfg: Config) => req<ValidateResult>("POST", "/api/pipeline/validate", cfg),
  resetConfig: () => req<PipelineView>("POST", "/api/pipeline/reset-config"),
  loadModule: (kind: string, impl: string) => req<StageStatus>("POST", `/api/modules/${kind}/${impl}/load`),
  unloadModule: (kind: string, impl: string) => req<StageStatus>("POST", `/api/modules/${kind}/${impl}/unload`),
  models: () => req<ModelEntry[]>("GET", "/api/models"),
  download: (kind: string, impl: string) => req<unknown>("POST", `/api/models/${kind}/${impl}/download`),
  voices: () => req<VoiceEntry[]>("GET", "/api/voices"),
  preview: (impl: string, body: { voice_id?: string; text?: string; speed?: number; pitch?: number; params?: Record<string, unknown> }) => req<Blob>("POST", `/api/voices/${impl}/preview`, body),
  state: () => req<RuntimeState>("GET", "/api/state"),
  sessions: () => req<SessionSummary[]>("GET", "/api/sessions"),
  session: (id: string) => req<SessionDetail>("GET", `/api/sessions/${id}`),
  sessionAction: (id: string, action: "duck" | "stop" | "resume") => req<unknown>("POST", `/api/sessions/${id}/${action}`),
  renders: (sid?: string) => req<RenderEntry[]>("GET", `/api/renders${sid ? `?session_id=${encodeURIComponent(sid)}` : ""}`),
  deleteRender: (id: string) => req<unknown>("DELETE", `/api/renders/${id}`),
  runtime: (action: "pause" | "resume") => req<RuntimeState>("POST", `/api/runtime/${action}`),
  metrics: () => req<Metrics>("GET", "/api/metrics"),
  logs: (level = "DEBUG", limit = 300) => req<LogRecord[]>("GET", `/api/logs?level=${level}&limit=${limit}`),
  scripts: () => req<Script[]>("GET", "/api/scripts"),
  profiles: () => req<ProfileEntry[]>("GET", "/api/profiles"),
  profile: (name: string) => req<Profile>("GET", `/api/profiles/${name}`),
  putProfile: (name: string, body: { description?: string; config?: Config; from_current?: boolean }) => req<Profile>("PUT", `/api/profiles/${name}`, body),
  deleteProfile: (name: string) => req<unknown>("DELETE", `/api/profiles/${name}`),
  applyProfile: (name: string) => req<PipelineView>("POST", `/api/profiles/${name}/apply`),
  layouts: () => req<LayoutEntry[]>("GET", "/api/layouts"),
  layout: (name: string) => req<Layout>("GET", `/api/layouts/${name}`),
  putLayout: (name: string, body: Layout) => req<Layout>("PUT", `/api/layouts/${name}`, body),
  deleteLayout: (name: string) => req<unknown>("DELETE", `/api/layouts/${name}`),
  compare: (text: string, a: CompareVariant, b: CompareVariant) => req<{ a?: CompareResult; b?: CompareResult }>("POST", "/api/compare", { text, a, b }),
  rate: (body: { render_id?: string | null; tts_impl?: string; voice_id?: string | null; score: number; text?: string }) => req<unknown>("POST", "/api/compare/ratings", body),
  ratings: () => req<{ rows?: unknown[]; summary?: RatingSummary[] }>("GET", "/api/compare/ratings"),
  render: (text: string, v: CompareVariant) => req<CompareResult>("POST", "/api/render", { text, ...v }),
  clones: () => req<CloneVoice[]>("GET", "/api/clones"),
  createClone: (body: { name: string; transcript: string; audio_b64: string; language?: string; notes?: string }) => req<CloneVoice>("POST", "/api/clones", body),
  updateClone: (id: string, body: { name?: string; transcript?: string; language?: string; notes?: string }) => req<CloneVoice>("PUT", `/api/clones/${id}`, body),
  deleteClone: (id: string) => req<unknown>("DELETE", `/api/clones/${id}`),
  fish: () => req<FishStatus>("GET", "/api/fish"),
  fishAction: (action: "start" | "stop") => req<FishStatus>("POST", `/api/fish/${action}`),
};

export type Api = typeof realApi;
export const api: Api = MOCK ? (mockApi as unknown as Api) : realApi;

export function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
