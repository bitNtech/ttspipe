// Live connections: /ws/events monitor stream and the dashboard's own /ws/tts speaking socket (§8, §14).
import { store } from "../state/store";
import type { Envelope } from "../types/contract";
import { MOCK, api } from "./client";
import { engine } from "./mock";

const wsUrl = (path: string) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}${path}`;

// ---- monitor stream ----------------------------------------------------------------------------

let started = false;

export function startMonitor() {
  if (started) return; // StrictMode runs effects twice in dev
  started = true;
  if (MOCK) {
    store.mode = "mock";
    store.connected = true;
    store.touch("conn");
    engine.start((ev) => store.ingest(ev));
    return;
  }
  let ws: WebSocket | null = null;
  let ping: number | undefined;
  const connect = () => {
    ws = new WebSocket(wsUrl("/ws/events?session_id=*"));
    ws.onopen = () => {
      store.connected = true;
      store.retries = 0;
      store.touch("conn");
      refreshAll();
      ping = window.setInterval(() => ws?.readyState === 1 && ws.send(JSON.stringify({ type: "ping", t: performance.now() })), 3000);
    };
    ws.onmessage = (m) => {
      let ev: Envelope;
      try { ev = JSON.parse(m.data); } catch { return; }
      if (ev.type === "pong") {
        store.rtt = Math.round(performance.now() - Number(ev.data?.t ?? 0));
        store.touch("conn");
        return;
      }
      store.ingest(ev);
    };
    ws.onclose = () => {
      window.clearInterval(ping);
      store.connected = false;
      store.retries++;
      store.touch("conn");
      setTimeout(connect, Math.min(10000, 500 * 2 ** Math.min(store.retries, 5)));
    };
    ws.onerror = () => ws?.close();
  };
  connect();
}

export async function refreshAll() {
  try {
    const [pipe, mods, health] = await Promise.all([api.pipeline(), api.modules(), api.health()]);
    store.pipeline = pipe;
    store.modules = mods;
    store.health = health;
    store.touch("pipeline", "modules", "conn");
    const sessions = await api.sessions();
    for (const s of (Array.isArray(sessions) ? sessions : []).slice(-5)) {
      if (s.id && !store.sessions.get(s.id)?.utts.size) store.hydrate(await api.session(s.id));
    }
    const st = await api.state();
    store.bridge = st.bridge ?? null;
    store.paused = !!st.paused;
    const logs = await api.logs("INFO", 300);
    if (Array.isArray(logs) && !store.logs.length) { store.logs = logs; store.touch("logs"); }
  } catch { /* backend offline: banner shows retries */ }
}

// ---- dashboard speaking socket + WebAudio playback ----------------------------------------------

class Speaker {
  ws: WebSocket | null = null;
  ctx: AudioContext | null = null;
  gain: GainNode | null = null;
  rate = 24000;
  playAt = 0;
  volume = 0.9;
  sessionId: string | null = null;
  streaming = 0;

  async ensure(): Promise<void> {
    if (MOCK) return;
    if (this.ws && this.ws.readyState <= 1) {
      if (this.ws.readyState === 1) return;
      await new Promise((r) => this.ws!.addEventListener("open", r, { once: true }));
      return;
    }
    this.ws = new WebSocket(wsUrl("/ws/tts?label=dashboard"));
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (m) => {
      if (typeof m.data === "string") {
        try {
          const ev = JSON.parse(m.data);
          if (ev.type === "session.started") {
            this.sessionId = ev.session_id;
            this.rate = ev.data?.sample_rate ?? 24000;
          }
        } catch { /* ignore */ }
        return;
      }
      this.play(new Int16Array(m.data.slice(1))); // 1-byte type header: Int16Array(buf, 1) throws (offset must be even)
    };
    this.ws.onclose = () => { this.ws = null; };
    await new Promise((r, j) => { this.ws!.onopen = r; this.ws!.onerror = j; });
  }

  play(pcm: Int16Array) {
    if (!this.ctx) {
      this.ctx = new AudioContext({ sampleRate: this.rate });
      this.gain = this.ctx.createGain();
      this.gain.gain.value = this.volume;
      this.gain.connect(this.ctx.destination);
    }
    const buf = this.ctx.createBuffer(1, pcm.length, this.rate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.gain!);
    const now = this.ctx.currentTime;
    if (this.playAt < now + 0.01) this.playAt = now + 0.06; // 60 ms client cushion after a gap
    src.start(this.playAt);
    this.playAt += buf.duration;
  }

  setVolume(v: number) {
    this.volume = v;
    if (this.gain) this.gain.gain.value = v;
  }

  async send(msg: Record<string, unknown>) {
    await this.ensure();
    this.ws?.send(JSON.stringify(msg));
  }

  /** Speaks text on the dashboard session; uid lets callers find the utterance in the store. */
  async speak(text: string, streamAsLlm: boolean, uid = `d${Date.now().toString(36)}`) {
    if (this.ctx?.state === "suspended") await this.ctx.resume();
    if (MOCK) { engine.speak(text, streamAsLlm); return; }
    if (!streamAsLlm) { await this.send({ type: "speak", text, utterance_id: uid }); return; }
    const gen = ++this.streaming;
    // replay as LLM token deltas: 1-4 chars every 20-60 ms
    for (let i = 0; i < text.length && gen === this.streaming;) {
      const n = 1 + Math.floor(Math.random() * 4);
      await this.send({ type: "text", delta: text.slice(i, i + n), utterance_id: uid });
      i += n;
      await new Promise((r) => setTimeout(r, 20 + Math.random() * 40));
    }
    await this.send({ type: "end_utterance", utterance_id: uid });
  }

  async control(type: "duck" | "stop" | "resume") {
    if (type === "stop") this.streaming++;
    if (MOCK) { engine[type](); return; }
    const target = store.current();
    if (target && target.id !== this.sessionId && target.connected) {
      await api.sessionAction(target.id, type);
      return;
    }
    await this.send({ type });
  }

  async voice(voice: string, speed?: number, pitch?: number) {
    if (!MOCK) await this.send({ type: "voice", voice, speed, pitch });
  }
}

export const speaker = new Speaker();
