"""Cloned-voice library, tts/fish_s2 against a fake Fish S2-Pro socket, and the render / clones / fish routes."""
import base64
import threading

import msgpack
import numpy as np
import pytest

from app.core import clones
from app.core.audio import wav_bytes
from app.core.base import Chunk, LangSegment, ProsodyPlan, Voice, Word


def speech(seconds=4.0, rate=16000):
    t = np.arange(int(seconds * rate)) / rate
    return wav_bytes((0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), rate)


def chunk(text):
    ws = [Word(w, w, i, "en") for i, w in enumerate(text.split())]
    return Chunk("u.0", "u", 0, ws, [LangSegment(text, "en", ws)])


# ---- clone store / API --------------------------------------------------------------------------


def test_clone_store_validates_and_round_trips(tmp_path):
    st = clones.CloneStore(tmp_path)
    with pytest.raises(ValueError, match="transcript"):
        st.create("A", "", speech())
    with pytest.raises(ValueError, match="s long"):
        st.create("A", "hi", speech(0.5))
    with pytest.raises(ValueError, match="WAV"):
        st.create("A", "hi", b"not audio")
    c = st.create("Priya Warm", "hello there", speech(), "ta")
    assert c["voice_id"] == f"clone:{c['id']}" and c["id"].startswith("priya-warm-")
    audio, text = st.clip(c["id"])
    assert text == "hello there" and audio[:4] == b"RIFF"
    assert st.update(c["id"], {"transcript": "hello again"})["transcript"] == "hello again"
    assert [x["id"] for x in st.list()] == [c["id"]]
    st.delete(c["id"])
    assert st.list() == []
    with pytest.raises(KeyError):
        st.get("../etc")


def test_clones_api(client):
    b64 = base64.b64encode(speech()).decode()
    assert client.post("/api/clones", json={"name": "x", "transcript": "", "audio_b64": b64}).status_code == 422
    r = client.post("/api/clones", json={"name": "Asha", "transcript": "vanakkam", "audio_b64": b64, "language": "ta"})
    assert r.status_code == 201, r.text
    cid = r.json()["id"]
    assert client.get("/api/clones").json()[0]["id"] == cid
    assert client.get(f"/api/clones/{cid}/audio").content[:4] == b"RIFF"
    assert client.put(f"/api/clones/{cid}", json={"name": "Asha 2"}).json()["name"] == "Asha 2"
    # zero-shot engines list the clone as a voice
    fish = next(m for m in client.get("/api/modules").json() if m["id"] == "tts/fish_s2")
    assert f"clone:{cid}" in [v["id"] for v in fish["voices"]]
    assert f"clone:{cid}" in [v["id"] for v in client.get("/api/voices").json() if v["impl"] == "fish_s2"]
    assert client.delete(f"/api/clones/{cid}").status_code == 200
    assert client.get(f"/api/clones/{cid}").status_code == 404


def test_render_endpoint_returns_wav_and_metrics(client):
    r = client.post("/api/render", json={"text": "Hello there, testing one two.",
                                         "config": {"voice": {"impl": "tone", "voice_id": "tone-m"}}}).json()
    assert r["error"] is None and r["tts_impl"] == "tone" and r["voice_id"] == "tone-m"
    assert r["metrics"]["audio_ms"] > 0 and r["metrics"]["ttfa_ms"] is not None
    assert client.get(r["url"]).content[:4] == b"RIFF"
    assert client.post("/api/render", json={"text": " "}).status_code == 422


def test_fish_status_offline(client, monkeypatch):
    from app import settings

    monkeypatch.setattr(settings.current, "fish_url", "http://127.0.0.1:9")  # nothing listens on the discard port
    s = client.get("/api/fish").json()
    assert s["health"] == "offline" and s["url"] == "http://127.0.0.1:9" and isinstance(s["log_tail"], list)


# ---- tts/fish_s2 against a fake server ----------------------------------------------------------


@pytest.fixture
def fake_fish(monkeypatch):
    """A /v1/tts/live stand-in: started → audio blocks → finish(stop) per turn; records each start request."""
    from websockets.sync.server import serve

    from app.core import fish_server

    seen = []

    def handler(ws):
        req = None
        for raw in ws:
            msg = msgpack.unpackb(raw)
            ev = msg["event"]
            if ev == "start":
                req = msg["request"]
                seen.append(req)
                ws.send(msgpack.packb({"event": "started", "session_id": "s", "sample_rate": 44100, "format": "pcm",
                                       "seed": 1, "voice_id": None}))
            elif ev == "text" and msg["text"] == "fail":
                ws.send(msgpack.packb({"event": "finish", "reason": "error", "error": "boom"}))
            elif ev == "stop":
                for _ in range(3):
                    ws.send(msgpack.packb({"event": "audio", "audio": (np.ones(2048, "<i2") * 3000).tobytes(), "time": 0}))
                ws.send(msgpack.packb({"event": "finish", "reason": "stop", "error": None}))

    srv = serve(handler, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.socket.getsockname()[1]
    monkeypatch.setattr(fish_server.SERVER, "ws_url", lambda: f"ws://127.0.0.1:{port}/v1/tts/live")
    yield seen
    srv.shutdown()


def test_fish_module_streams_and_sends_voice(fake_fish, tmp_path, monkeypatch):
    from app.modules.tts.fish_s2 import FishS2TTS

    st = clones.CloneStore(tmp_path)
    c = st.create("Ref", "the exact words", speech())
    monkeypatch.setattr(clones, "store", lambda: st)
    mod = FishS2TTS({"seed": 7, "language": "ta"})
    pieces = list(mod.synthesize(None, chunk("Hello there."), ProsodyPlan(), Voice("fish_s2", c["voice_id"])))
    assert pieces[-1].is_last and all(p.sample_rate == 44100 for p in pieces)
    pcm = np.concatenate([p.pcm for p in pieces])
    assert len(pcm) == 3 * 2048 and abs(pcm[0] - 3000 / 32768) < 1e-6
    req = fake_fish[-1]
    assert req["format"] == "pcm" and req["seed"] == 7 and req["language"] == "ta"
    assert req["references"][0]["text"] == "the exact words" and req["references"][0]["audio"][:4] == b"RIFF"
    # the socket is reused across turns; model_default opts out of the server's active voice
    ws = mod.ws
    list(mod.synthesize(None, chunk("Again."), ProsodyPlan(), Voice("fish_s2", "model_default")))
    assert mod.ws is ws and fake_fish[-1]["use_active_voice"] is False and "references" not in fake_fish[-1]
    list(mod.synthesize(None, chunk("Third."), ProsodyPlan(), Voice("fish_s2", "server_active")))
    assert "use_active_voice" not in fake_fish[-1]


def test_fish_module_error_and_cancel_drop_the_socket(fake_fish):
    from app.modules.tts.fish_s2 import FishS2TTS

    mod = FishS2TTS({})
    with pytest.raises(RuntimeError, match="boom"):
        list(mod.synthesize(None, chunk("fail"), ProsodyPlan(), Voice("fish_s2", "server_active")))
    assert mod.ws is None
    gen = mod.synthesize(None, chunk("Hello."), ProsodyPlan(), Voice("fish_s2", "server_active"))
    next(gen)
    gen.close()  # barge-in mid-stream: the server cancels the GPU job when the socket drops
    assert mod.ws is None
    assert len(list(mod.synthesize(None, chunk("Hello."), ProsodyPlan(), Voice("fish_s2", "server_active")))) == 4
