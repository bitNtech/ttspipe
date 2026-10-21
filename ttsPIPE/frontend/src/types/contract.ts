// Mirrors docs/ARCHITECTURE.md §4.3, §6–§11 and §14. Every field is optional: code defensively.

export const CONTRACT_VERSION = "1.1";

export type Kind = "ingest" | "normalize" | "segment" | "chunker" | "g2p" | "prosody" | "tts" | "post";
export const KINDS: Kind[] = ["ingest", "normalize", "segment", "chunker", "g2p", "prosody", "tts", "post"];
export type Lang = "ta" | "en" | "ta-Latn" | string;
export type StatusValue = "ready" | "loading" | "unloaded" | "error" | "disabled";
export type PlaybackStateName = "idle" | "buffering" | "synthesizing" | "speaking" | "ducked" | "interrupted" | "done";
export const PLAYBACK_STATES: PlaybackStateName[] = ["idle", "buffering", "synthesizing", "speaking", "ducked", "interrupted", "done"];

// §4.3
export interface ParamSchema {
  type?: "number" | "integer" | "string" | "boolean" | "array";
  default?: unknown;
  min?: number;
  max?: number;
  enum?: unknown[];
  description?: string;
  unit?: string;
  items?: { type?: string };
  advanced?: boolean;
  nullable?: boolean;
}
export type ParamsSchema = Record<string, ParamSchema>;

// §6.8
export interface Voice { impl?: string; voice_id?: string; speed?: number; pitch?: number }
export interface VoiceInfo { id?: string; name?: string; language?: string; gender?: string; sample?: string | null }
export interface VoiceEntry extends VoiceInfo { impl?: string }

// §6.10
export interface ModelFile { name?: string; url?: string; path?: string; size_mb?: number; sha256?: string | null; present?: boolean }

// §4.2 + §9.2
export interface ModuleInfo {
  id?: string; kind?: Kind; name?: string; description?: string; model_id?: string | null;
  size_mb?: number; ram_mb?: number; license?: string; streaming?: boolean; requires?: string[];
  params_schema?: ParamsSchema; default_params?: Record<string, unknown>; tags?: string[];
  languages?: string[]; voices?: VoiceInfo[]; builtin?: boolean; native_rate?: number | null; family?: string;
}
export interface ModuleEntry extends ModuleInfo {
  available?: boolean; reason?: string | null; loaded?: boolean; status?: StatusValue | null; models_present?: boolean;
}

// §9.3
export interface StageStatus {
  kind?: Kind; name?: string; impl?: string | null; configured_impl?: string | null; enabled?: boolean;
  status?: StatusValue; error?: string | null; fallback?: boolean; busy?: boolean;
  params?: Record<string, unknown>; load_ms?: number | null; count?: number; errors?: number;
  consecutive_errors?: number; last_ms?: number | null; p50_ms?: number | null; p95_ms?: number | null;
}
export interface GraphNode { id?: string; kind?: string; label?: string; impl?: string | null }
export interface GraphEdge { from?: string; to?: string; dashed?: boolean }
export interface Graph { nodes?: GraphNode[]; edges?: GraphEdge[]; positions?: Record<string, { x?: number; y?: number }> }

// §11
export interface StageConfig { enabled?: boolean; impl?: string; params?: Record<string, unknown> }
export interface PlaybackConfig {
  lookahead_chunks?: number; jitter_ms?: number; duck_db?: number; duck_ramp_ms?: number; duck_release_ms?: number;
  stop_fade_ms?: number; crossfade_ms?: number; max_silence_gap_ms?: number; fill_underrun_silence?: boolean;
  level_every_ticks?: number;
}
export interface ChunkerConfig {
  first_chunk_min_words?: number; first_chunk_max_words?: number; first_chunk_max_wait_ms?: number;
  min_chunk_chars?: number; max_chunk_chars?: number; nonstreaming_max_chunk_chars?: number;
}
export interface ShadowConfig { impl?: string; params?: Record<string, unknown>; voice_id?: string | null }
export interface BridgeConfig {
  enabled?: boolean; prepipe_url?: string; session_map?: "same_id" | "latest"; send_reference_audio?: boolean;
  reference_ws_path?: string; map?: Record<string, string>;
}
export interface Config {
  audio?: { out_rate?: number; frame_ms?: number };
  stages?: Partial<Record<Kind, StageConfig>>;
  voice?: Voice;
  playback?: PlaybackConfig;
  chunker?: ChunkerConfig;
  cache?: { enabled?: boolean; max_mb?: number };
  renders?: { enabled?: boolean; keep?: number };
  events?: { queue_size?: number };
  graph?: { shadows?: { tts?: ShadowConfig | null }; positions?: Record<string, { x?: number; y?: number }> };
  bridge?: BridgeConfig;
  active_profile?: string | null;
}
export type PartialConfig = Config;

export interface PipelineView {
  config?: Config; stages?: Partial<Record<Kind, StageStatus>>; shadow?: StageStatus | null; graph?: Graph;
  active_profile?: string | null; dirty?: boolean;
}

// §9.1
export interface Health {
  status?: "ok" | "degraded" | "error"; contract?: string; version?: string; uptime_s?: number;
  reasons?: string[]; stages?: Partial<Record<Kind, StatusValue>>;
}

export interface ValidateResult { ok?: boolean; errors?: string[]; warnings?: string[]; estimate?: { download_mb?: number; ram_mb?: number } }

// §9.5
export interface ModelEntry {
  id?: string; kind?: Kind; impl?: string; files?: ModelFile[]; present?: boolean; size_mb?: number;
  downloading?: boolean; progress?: { status?: string; done_mb?: number; total_mb?: number; error?: string | null } | null;
}

// §9.7
export interface BridgeStatus { enabled?: boolean; connected?: boolean; url?: string | null; error?: string | null; last_event?: string | null; events_mapped?: number }
export interface CacheStats { hits?: number; misses?: number; hit_rate?: number | null; entries?: number; mb?: number; enabled?: boolean }
export interface RuntimeState { paused?: boolean; sessions?: number; active_profile?: string | null; bridge?: BridgeStatus | null; cache?: CacheStats; uptime_s?: number }
export interface SessionSummary {
  id?: string; label?: string; started?: number; state?: PlaybackStateName; voice?: Voice; utterances?: number;
  last_utterance_id?: string | null; t_out_ms?: number; connected?: boolean;
}
export interface ChunkLatency { flush_ms?: number | null; synth_queue_ms?: number | null; first_audio_ms?: number | null; synth_ms?: number | null; audio_ms?: number | null; rtf?: number | null }
export interface ChunkSummary {
  chunk_id?: string; index?: number; text?: string; norm_text?: string; lang?: Lang; reason?: string; words?: number;
  latency?: ChunkLatency; played?: "full" | "partial" | "none"; cached?: boolean;
}
export interface UtteranceLatency {
  ttfa_ms?: number | null; ttfa_end_ms?: number | null; text_to_first_chunk_ms?: number | null; first_synth_ms?: number | null;
  total_synth_ms?: number | null; audio_ms?: number | null; rtf?: number | null;
}
export interface UtteranceSummary {
  utterance_id?: string; text?: string; normalized?: string; chunks?: ChunkSummary[]; interrupted?: boolean;
  profile?: string | null; latency?: UtteranceLatency; render_id?: string | null;
  status?: "open" | "speaking" | "final" | "interrupted"; heard_text?: string | null; unplayed_text?: string | null;
  edits?: { before?: string; after?: string }[]; segments?: { text?: string; lang?: Lang }[]; created?: number;
}
export interface SessionDetail extends Omit<SessionSummary, "utterances"> { utterances?: UtteranceSummary[] }

// §9.8
export interface RenderEntry {
  id?: string; session_id?: string; utterance_id?: string; text?: string; duration_ms?: number; created?: number;
  interrupted?: boolean; sample_rate?: number; url?: string; shadow?: boolean; heard_text?: string | null;
}
// §9.10
export interface LogRecord { ts?: number; level?: string; logger?: string; msg?: string; session_id?: string | null }
// §9.11
export interface Script { id?: string; title?: string; language?: string; text?: string }
// §9.12
export interface ProfileEntry { name?: string; builtin?: boolean; description?: string; updated?: number }
export interface Profile { name?: string; builtin?: boolean; description?: string; config?: PartialConfig }
export interface LayoutPanel { id: string; x: number; y: number; w: number; h: number; collapsed?: boolean; hidden?: boolean }
export interface Layout { name?: string; cols?: number; row_h?: number; panels?: LayoutPanel[]; updated?: number }
export interface LayoutEntry { name?: string; updated?: number }
// §9.13
export interface CompareVariant { profile?: string; config?: PartialConfig; label?: string }
export interface CompareResult {
  label?: string; render_id?: string | null; url?: string | null; tts_impl?: string; voice_id?: string | null;
  metrics?: { ttfa_ms?: number | null; first_synth_ms?: number | null; total_synth_ms?: number | null; audio_ms?: number | null; rtf?: number | null; chunks?: number; wall_ms?: number };
  error?: string | null;
}
export interface RatingSummary { tts_impl?: string; voice_id?: string | null; n?: number; mos?: number; ci95?: number | null }

// §9.14
export interface LatStat { n?: number; last?: number | null; p50?: number | null; p95?: number | null }
export interface StageMetric { impl?: string | null; count?: number; errors?: number; last_ms?: number | null; p50_ms?: number | null; p95_ms?: number | null; spark?: (number | null)[] }
export interface Metrics {
  uptime_s?: number; sessions?: number; paused?: boolean;
  stages?: Record<string, StageMetric>;
  ttfa?: { by_profile?: Record<string, LatStat>; end_by_profile?: Record<string, LatStat> };
  latency?: Partial<Record<"ttfa" | "ttfa_end" | "text_to_first_chunk" | "first_synth" | "normalize" | "chunk_rtf", LatStat>>;
  throughput?: { chunks_per_min?: number; audio_s_per_min?: number; chars_per_min?: number };
  cache?: CacheStats; underruns?: number; interruptions?: number; utterances?: number; events_dropped?: number;
  system?: { cpu_pct?: number; rss_mb?: number; threads?: number; cores?: number };
}

// §7.1 + §7.2
export type EventType =
  | "session.started" | "session.ended" | "text.delta" | "text.normalized" | "segment" | "chunk.flushed"
  | "synth.start" | "synth.first_audio" | "synth.done" | "audio.level" | "playback.state" | "playback.chunk"
  | "playback.progress" | "underrun" | "duck" | "utterance.started" | "utterance.interrupted" | "utterance.end"
  | "shadow" | "stage.status" | "stage.activity" | "metrics" | "log" | "error" | "model.download" | "bridge.status" | "pong";

export interface Envelope<D = Record<string, any>> {
  type?: EventType | string; session_id?: string | null; ts?: number; t_out_ms?: number | null; data?: D;
}

export interface ChunkFlushedData {
  utterance_id?: string; chunk_id?: string; index?: number; text?: string; norm_text?: string; tts_text?: string;
  lang?: Lang; words?: number; reason?: "first" | "punct" | "max_len" | "end"; segments?: { text?: string; lang?: Lang }[];
  pause_after_ms?: number; since_text_ms?: number;
}
export interface AudioLevelData { rms?: number; peak?: number; rms_db?: number; gain_db?: number; wave?: number[] }
export interface PlaybackStateData { state?: PlaybackStateName; prev?: PlaybackStateName; reason?: string; utterance_id?: string | null }
export interface InterruptedData { utterance_id?: string; heard_text?: string; unplayed_text?: string; at_ms?: number; source?: string }
export interface StageActivityData { kind?: string; impl?: string; utterance_id?: string | null; chunk_id?: string | null; ms?: number; ok?: boolean }
export interface ShadowData { utterance_id?: string; chunk_id?: string; impl?: string; first_audio_ms?: number | null; synth_ms?: number | null; audio_ms?: number | null; rtf?: number | null; error?: string | null }

// Cloned-voice library (data/voices) and the Fish S2-Pro GPU server (tts/fish_s2)
export interface CloneVoice {
  id?: string; name?: string; language?: string; notes?: string; created?: number; duration_s?: number; sample_rate?: number;
  peak_db?: number; warning?: string | null; transcript?: string; voice_id?: string; audio_url?: string;
}
export type FishHealth = "ok" | "loading" | "starting" | "error" | "offline";
export interface FishStatus {
  url?: string; dir?: string | null; health?: FishHealth; error?: string | null; managed?: boolean; pid?: number | null;
  started_s_ago?: number | null; venv_present?: boolean; checkpoint_present?: boolean; can_start?: boolean;
  info?: Record<string, any> | null; log_tail?: string[];
}
