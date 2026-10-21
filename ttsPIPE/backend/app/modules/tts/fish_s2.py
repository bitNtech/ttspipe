"""tts/fish_s2: Fish Audio S2-Pro (4B Dual-AR, NF4) on the GPU, via the Fish server in the parent repo.

The server (app/core/fish_server.py) owns the GPU and streams 16-bit PCM over its /v1/tts/live WebSocket. Each
ttsPIPE chunk is one start → text → stop turn on a kept-open socket; audio blocks are yielded as they arrive.
Voices: the server's active voice, the model's own default voice, or any clone in data/voices ("clone:<id>"),
sent as a reference clip + transcript (the server caches the encoded prompt and its KV prefix by content).
"""
from __future__ import annotations

import numpy as np

from app.core import clones
from app.core.base import AudioPiece, Chunk, ModuleInfo, ProsodyPlan, TTSModule, Voice, VoiceInfo, Word, schema
from app.core.fish_server import SERVER, _get
from app.core.registry import register

STATIC_VOICES = [
    VoiceInfo("server_active", "Server active voice (set on the Fish server)", "multi"),
    VoiceInfo("model_default", "S2-Pro default voice (no cloning)", "multi"),
]


@register
class FishS2TTS(TTSModule):
    info = ModuleInfo(
        id="tts/fish_s2", kind="tts", name="Fish Audio S2-Pro (NF4, GPU)",
        description="4B-parameter multilingual model (Tamil, English and ~50 more; Tanglish in one voice) with zero-shot "
                    "voice cloning from a ~10 s clip. Runs on the GPU in the Fish server (../, own .venv), which ttsPIPE "
                    "starts on demand. About real time on an RTX 2050 (RTF ~1), so give it a larger jitter buffer.",
        model_id="scarxity/fish-speech-s2-pro-nf4", size_mb=4990, ram_mb=3500,
        license="Fish Audio Research License (non-commercial without a Fish Audio licence)", streaming=True,
        native_rate=44100, requires=["msgpack", "websockets"], family="fish", languages=["ta", "en", "ta-Latn", "multi"],
        tags=["remote", "gpu", "cloning", "no-speed"], voices=STATIC_VOICES,
        params_schema=schema(
            autostart=("boolean", True, {"description": "Start the Fish server when this engine loads and it is not running"}),
            load_timeout_s=("integer", 300, {"min": 10, "max": 1800, "unit": "s", "description": "How long load waits for the model to come up"}),
            temperature=("number", 0.8, {"min": 0.1, "max": 1.0, "description": "Sampling temperature (lower = steadier, flatter)"}),
            top_p=("number", 0.8, {"min": 0.1, "max": 1.0}),
            top_k=("integer", 30, {"min": 1, "max": 1000, "advanced": True}),
            repetition_penalty=("number", 1.1, {"min": 0.9, "max": 2.0, "advanced": True}),
            seed=("integer", 42, {"min": -1, "description": "Fixed seed keeps the default voice consistent across chunks (-1 = random)"}),
            style=("string", "", {"description": "Delivery hint put before every turn, e.g. [warm, friendly, conversational] (S2 takes free-form [instructions])"}),
            language=("string", "", {"description": "Language hint, e.g. ta or en (empty = the model detects it)"}),
            normalize=("boolean", True, {"advanced": True, "description": "Server-side text normalisation"}),
            chunk_length=("integer", 200, {"min": 100, "max": 300, "advanced": True}),
            timeout_s=("number", 60.0, {"min": 5, "max": 600, "unit": "s", "advanced": True, "description": "Max wait for the next audio block"}),
        ),
    )

    server = SERVER  # tts/indic_mio swaps in its own

    def __init__(self, params=None):
        super().__init__(params)
        self.ws = None

    @classmethod
    def list_voices(cls) -> list[VoiceInfo]:
        return STATIC_VOICES + [VoiceInfo(c["voice_id"], f"{c['name']} (clone)", c.get("language") or "multi")
                                for c in clones.store().list()]

    def cache_voice(self, voice_id: str) -> str:
        """Phrase-cache identity: the server's active voice can change under the same "server_active" id, and a
        stale hit would replay audio in the old voice (heard as several voices in one script)."""
        if voice_id != "server_active":
            return voice_id
        _, v = _get(f"{self.server.url}/v1/voice", 1.0)
        return f"server_active:{(v or {}).get('voice_id')}"

    def load(self) -> None:
        self.server.ensure_ready(bool(self.params["autostart"]), float(self.params["load_timeout_s"]))
        self.loaded = True

    def unload(self) -> None:
        self._close()
        self.loaded = False

    def _close(self) -> None:
        if self.ws is not None:
            try:
                self.ws.close()
            except Exception:
                pass
            self.ws = None

    def _conn(self):
        from websockets.sync.client import connect

        if self.ws is None:
            self.ws = connect(self.server.ws_url(), max_size=None, open_timeout=10, compression=None)
        return self.ws

    def request(self, voice_id: str) -> dict:
        p = self.params
        req = {"format": "pcm", "normalize": bool(p["normalize"]), "chunk_length": int(p["chunk_length"]),
               "temperature": float(p["temperature"]), "top_p": float(p["top_p"]), "top_k": int(p["top_k"]),
               "repetition_penalty": float(p["repetition_penalty"])}
        if int(p["seed"]) >= 0:
            req["seed"] = int(p["seed"])
        if p["language"]:
            req["language"] = p["language"]
        cid = clones.clone_id(voice_id)
        if cid:
            audio, text = clones.store().clip(cid)  # read per chunk (page-cached), so transcript edits apply at once
            req["references"] = [{"audio": audio, "text": text}]
        elif voice_id == "model_default":
            req["use_active_voice"] = False
        return req

    def warm(self, voice_id: str) -> None:
        """Have the server encode a new clone's reference now (a one-off ~10 s on its CPU codec), not on first use."""
        c = Chunk("warm.0", "warm", 0, [Word("Hello.", "Hello.")], [], tts_text="Hello.")
        for _ in self.synthesize(None, c, ProsodyPlan(), Voice(impl=self.info.impl, voice_id=voice_id)):
            pass
        self._close()

    def synthesize(self, state, chunk, plan, voice):
        import msgpack

        text = chunk.tts_text.strip()
        rate = 44100
        if not text:
            yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=rate, is_last=True)
            return
        req = self.request(voice.voice_id)
        ok = False
        try:
            try:
                ws = self._conn()
                ws.send(msgpack.packb({"event": "start", "request": req}))
            except Exception:  # stale kept-open socket (server restarted): reconnect once
                self._close()
                ws = self._conn()
                ws.send(msgpack.packb({"event": "start", "request": req}))
            ws.send(msgpack.packb({"event": "text", "text": f"{self.params.get('style', '')} {text}".strip()}))
            ws.send(msgpack.packb({"event": "stop"}))
            while True:
                msg = msgpack.unpackb(ws.recv(timeout=float(self.params["timeout_s"])))
                ev = msg.get("event")
                if ev == "started":
                    rate = int(msg.get("sample_rate") or rate)
                elif ev == "audio":
                    pcm = np.frombuffer(msg["audio"], "<i2").astype(np.float32) / 32768.0
                    if len(pcm):
                        yield AudioPiece(pcm=pcm, sample_rate=rate)
                elif ev == "finish":
                    if msg.get("reason") == "error":
                        raise RuntimeError(f"Fish server: {msg.get('error')}")
                    break
            ok = True
        finally:
            if not ok:  # cancelled mid-stream (generator closed) or failed: dropping the socket cancels the GPU job
                self._close()
        yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=rate, is_last=True)
