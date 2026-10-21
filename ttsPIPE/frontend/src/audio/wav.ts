// Browser-side audio helpers for voice cloning: decode any recording/upload, trim silence, encode mono 16-bit WAV.

export interface Clip { wav: Blob; seconds: number; peakDb: number; rate: number }

const RATE = 44100;

export async function toWav(src: Blob, trim = true): Promise<Clip> {
  const ctx = new OfflineAudioContext(1, 1, RATE); // decodeAudioData resamples to the context rate
  const audio = await ctx.decodeAudioData(await src.arrayBuffer());
  const n = audio.length, chs = audio.numberOfChannels;
  let mono = new Float32Array(n);
  for (let c = 0; c < chs; c++) {
    const d = audio.getChannelData(c);
    for (let i = 0; i < n; i++) mono[i] += d[i] / chs;
  }
  if (trim) {
    // drop leading/trailing silence below -45 dBFS, keeping 120 ms of padding
    const thr = 10 ** (-45 / 20), pad = Math.round(0.12 * RATE);
    let a = 0, b = n - 1;
    while (a < n && Math.abs(mono[a]) < thr) a++;
    while (b > a && Math.abs(mono[b]) < thr) b--;
    if (b > a) mono = mono.slice(Math.max(0, a - pad), Math.min(n, b + pad));
  }
  let peak = 0;
  for (let i = 0; i < mono.length; i++) peak = Math.max(peak, Math.abs(mono[i]));
  return { wav: encodeWav(mono, RATE), seconds: mono.length / RATE, peakDb: 20 * Math.log10(Math.max(peak, 1e-9)), rate: RATE };
}

export function encodeWav(x: Float32Array, rate: number): Blob {
  const buf = new ArrayBuffer(44 + x.length * 2), v = new DataView(buf);
  const w = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([buf], { type: "audio/wav" });
}

export function blobToBase64(b: Blob): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result).split(",", 2)[1] ?? "");
    r.onerror = () => rej(r.error);
    r.readAsDataURL(b);
  });
}

/** Microphone recorder with a live level (0..1). Raw capture: no echo cancellation / noise suppression / AGC. */
export class Recorder {
  private rec: MediaRecorder | null = null;
  private stream: MediaStream | null = null;
  private ctx: AudioContext | null = null;
  private parts: Blob[] = [];
  level = 0;
  started = 0;

  async start(onLevel: (lv: number) => void): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("This browser cannot record audio (needs https or localhost).");
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 } });
    this.ctx = new AudioContext();
    const an = this.ctx.createAnalyser();
    an.fftSize = 1024;
    this.ctx.createMediaStreamSource(this.stream).connect(an);
    const data = new Float32Array(an.fftSize);
    const tick = () => {
      if (!this.rec) return;
      an.getFloatTimeDomainData(data);
      let s = 0;
      for (const v of data) s += v * v;
      onLevel(Math.min(1, Math.sqrt(s / data.length) * 4));
      requestAnimationFrame(tick);
    };
    this.parts = [];
    this.rec = new MediaRecorder(this.stream);
    this.rec.ondataavailable = (e) => e.data.size && this.parts.push(e.data);
    this.rec.start(250);
    this.started = performance.now();
    tick();
  }

  stop(): Promise<Blob> {
    return new Promise((res) => {
      const rec = this.rec;
      if (!rec) return res(new Blob());
      rec.onstop = () => {
        res(new Blob(this.parts, { type: rec.mimeType }));
        this.stream?.getTracks().forEach((t) => t.stop());
        this.ctx?.close();
        this.rec = null;
      };
      rec.stop();
    });
  }

  get recording() { return this.rec !== null; }
}
