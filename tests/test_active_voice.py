"""The active voice: set it over HTTP, and every WebSocket turn inherits it.

The fake engine has no codec, so it cannot prove a clone *sounds* right --
that lives in tests/test_integration_gpu.py. What is provable here is the
wiring that decides which voice a turn gets, which is where the bugs that
matter would be: a `start` frame silently keeping the default voice, or
silently overriding an explicit one.

`VoiceStore` is exercised against a temp directory; the routes and sessions
against a `TTS_ENGINE=fake` app whose store is redirected there too, so no
test ever writes into the repo's own `references/`.
"""

from __future__ import annotations

import io
import os
import struct
import time
import wave

os.environ.setdefault("TTS_ENGINE", "fake")

import ormsgpack
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.voice import ACTIVE_VOICE_ID, VoiceStore

TRANSCRIPT = "The quick brown fox jumps over the lazy dog."
OTHER_TRANSCRIPT = "A completely different sentence, in a different voice."


def _wav(seconds: float = 1.0, sample_rate: int = 24000, tone: int = 1000) -> bytes:
    frames = int(seconds * sample_rate)
    samples = b"".join(struct.pack("<h", (i % (2 * tone)) - tone) for i in range(frames))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples)
    return buf.getvalue()


CLIP = _wav()
OTHER_CLIP = _wav(seconds=0.5, tone=700)


def _wait_ready(client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = client.get("/health").json()
        if last.get("status") == "ok":
            return
        time.sleep(0.02)
    raise TimeoutError(f"engine never became ready, last /health={last!r}")


@pytest.fixture()
def client(tmp_path):
    """A live app whose voice store is redirected into a temp directory."""
    with TestClient(app) as c:
        _wait_ready(c)
        original = app.state.voices
        app.state.voices = VoiceStore(tmp_path / "active")
        try:
            yield c
        finally:
            app.state.voices = original


def _set_voice(client, clip=CLIP, transcript=TRANSCRIPT, filename="voice.wav"):
    return client.post(
        "/v1/voice",
        data={"reference_text": transcript},
        files={"reference_audio": (filename, clip, "audio/wav")},
    )


def _started(ws, request=None):
    ws.send_bytes(ormsgpack.packb({"event": "start", "request": request or {"format": "pcm"}}))
    return ormsgpack.unpackb(ws.receive_bytes())


def _drain(ws, limit=500):
    for _ in range(limit):
        frame = ormsgpack.unpackb(ws.receive_bytes())
        if frame["event"] == "finish":
            return frame
    raise AssertionError("never saw a finish frame")


# -- VoiceStore on its own --


def test_store_round_trips_a_voice(tmp_path):
    store = VoiceStore(tmp_path / "active")
    assert store.get() is None

    voice = store.set(CLIP, TRANSCRIPT, filename="clip.wav", content_type="audio/wav")
    assert voice.text == TRANSCRIPT
    assert voice.audio == CLIP
    assert voice.duration_s == pytest.approx(1.0, abs=0.05)
    assert store.get() == voice


def test_store_survives_a_restart(tmp_path):
    VoiceStore(tmp_path / "active").set(CLIP, TRANSCRIPT, filename="clip.wav")

    reloaded = VoiceStore(tmp_path / "active").get()
    assert reloaded is not None
    assert reloaded.audio == CLIP
    assert reloaded.text == TRANSCRIPT


def test_store_writes_the_reference_id_layout(tmp_path):
    """The clip doubles as reference_id="_active", which needs a .lab sibling."""
    directory = tmp_path / "active"
    VoiceStore(directory).set(CLIP, TRANSCRIPT, filename="clip.wav")

    assert (directory / "clip.wav").read_bytes() == CLIP
    assert (directory / "clip.lab").read_text(encoding="utf-8") == TRANSCRIPT
    assert (directory / "meta.json").is_file()


def test_store_replacing_a_voice_leaves_no_trace_of_the_old_one(tmp_path):
    directory = tmp_path / "active"
    store = VoiceStore(directory)
    store.set(CLIP, TRANSCRIPT, filename="first.ogg")
    store.set(OTHER_CLIP, OTHER_TRANSCRIPT, filename="second.wav")

    # On disk the clip is always `clip<ext>` (the .lab convention pairs by
    # stem); `voice.filename` remembers what was uploaded. So the old file is
    # gone by name as well as by content.
    assert not (directory / "clip.ogg").exists()
    assert (directory / "clip.wav").read_bytes() == OTHER_CLIP
    assert store.get().text == OTHER_TRANSCRIPT
    assert store.get().filename == "second.wav"


def test_store_clear_is_idempotent(tmp_path):
    store = VoiceStore(tmp_path / "active")
    store.set(CLIP, TRANSCRIPT)

    assert store.clear() is True
    assert store.clear() is False
    assert store.get() is None
    assert not (tmp_path / "active").exists()


def test_store_labels_the_clip_by_extension_not_by_client_guess(tmp_path):
    """`curl -F` calls a .wav `application/octet-stream`; a browser needs better."""
    store = VoiceStore(tmp_path / "active")

    voice = store.set(CLIP, TRANSCRIPT, filename="v.wav", content_type="application/octet-stream")
    assert voice.content_type == "audio/wav"

    # An unrecognised extension falls back to whatever the client claimed.
    other = store.set(CLIP, TRANSCRIPT, filename="v.weird", content_type="audio/basic")
    assert other.content_type == "audio/basic"


def test_store_rejects_an_empty_transcript(tmp_path):
    store = VoiceStore(tmp_path / "active")
    with pytest.raises(ValueError):
        store.set(CLIP, "   ")
    with pytest.raises(ValueError):
        store.set(b"", TRANSCRIPT)


def test_store_ignores_an_unreadable_directory(tmp_path):
    directory = tmp_path / "active"
    directory.mkdir()
    (directory / "meta.json").write_text("{ this is not json", encoding="utf-8")

    assert VoiceStore(directory).get() is None  # boots clean rather than raising


# -- the /v1/voice routes --


def test_set_get_and_clear_over_http(client):
    assert client.get("/v1/voice").status_code == 404

    created = _set_voice(client)
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["text"] == TRANSCRIPT
    assert body["reference_id"] == ACTIVE_VOICE_ID
    assert body["size_bytes"] == len(CLIP)
    assert "audio" not in body  # metadata only; raw bytes are not JSON-safe

    fetched = client.get("/v1/voice")
    assert fetched.status_code == 200
    assert fetched.json()["voice_id"] == body["voice_id"]

    audio = client.get("/v1/voice/audio")
    assert audio.status_code == 200
    assert audio.content == CLIP
    assert audio.headers["x-voice-id"] == body["voice_id"]
    assert audio.headers["content-type"].startswith("audio/wav")

    assert client.delete("/v1/voice").json() == {"cleared": True}
    assert client.delete("/v1/voice").json() == {"cleared": False}
    assert client.get("/v1/voice").status_code == 404
    assert client.get("/v1/voice/audio").status_code == 404


def test_set_over_json_with_base64(client):
    import base64

    response = client.post(
        "/v1/voice",
        json={
            "reference_audio": base64.b64encode(CLIP).decode("ascii"),
            "reference_text": TRANSCRIPT,
            "filename": "from-json.wav",
        },
    )
    assert response.status_code == 200, response.text
    assert client.get("/v1/voice/audio").content == CLIP


def test_set_requires_the_exact_transcript(client):
    response = client.post(
        "/v1/voice", files={"reference_audio": ("voice.wav", CLIP, "audio/wav")}
    )
    assert response.status_code == 400
    assert "verbatim" in response.json()["error"]


def test_set_requires_audio(client):
    response = client.post("/v1/voice", data={"reference_text": TRANSCRIPT})
    assert response.status_code == 400
    assert "reference_audio" in response.json()["error"]


def test_set_rejects_an_oversized_clip(client):
    from app.http.common import MAX_CLIP_BYTES

    response = client.post(
        "/v1/voice",
        data={"reference_text": TRANSCRIPT},
        files={"reference_audio": ("huge.wav", b"\0" * (MAX_CLIP_BYTES + 1), "audio/wav")},
    )
    assert response.status_code == 400
    assert "limit" in response.json()["error"]


def test_status_reports_the_active_voice(client):
    assert client.get("/status").json()["active_voice"] is None

    voice_id = _set_voice(client).json()["voice_id"]
    assert client.get("/status").json()["active_voice"]["voice_id"] == voice_id


# -- what the WebSocket actually does with it --


def test_start_without_a_voice_reports_none(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws)["voice_id"] is None
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)


def test_start_inherits_the_active_voice(client):
    voice_id = _set_voice(client).json()["voice_id"]

    with client.websocket_connect("/v1/tts/live") as ws:
        started = _started(ws)
        assert started["voice_id"] == voice_id
        ws.send_bytes(ormsgpack.packb({"event": "text", "text": "Speak in the set voice."}))
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        assert _drain(ws)["reason"] == "stop"


def test_the_voice_persists_across_turns_and_connections(client):
    voice_id = _set_voice(client).json()["voice_id"]

    for _ in range(2):
        with client.websocket_connect("/v1/tts/live") as ws:
            for _turn in range(2):
                assert _started(ws)["voice_id"] == voice_id
                ws.send_bytes(ormsgpack.packb({"event": "stop"}))
                _drain(ws)


def test_changing_the_voice_changes_what_the_next_turn_gets(client):
    first = _set_voice(client).json()["voice_id"]
    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws)["voice_id"] == first
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)

    second = _set_voice(client, OTHER_CLIP, OTHER_TRANSCRIPT).json()["voice_id"]
    assert second != first
    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws)["voice_id"] == second
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)


def test_clearing_the_voice_returns_turns_to_the_default(client):
    _set_voice(client)
    client.delete("/v1/voice")

    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws)["voice_id"] is None
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)


def test_an_explicit_reference_beats_the_active_voice(client):
    import base64

    _set_voice(client)
    request = {
        "format": "pcm",
        "references": [
            {"audio": base64.b64encode(OTHER_CLIP).decode("ascii"), "text": OTHER_TRANSCRIPT}
        ],
    }
    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws, request)["voice_id"] == "inline"
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)


def test_an_explicit_reference_id_beats_the_active_voice(client):
    _set_voice(client)
    with client.websocket_connect("/v1/tts/live") as ws:
        started = _started(ws, {"format": "pcm", "reference_id": "narrator"})
        assert started["voice_id"] == "narrator"
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)


def test_use_active_voice_false_opts_out_without_clearing(client):
    voice_id = _set_voice(client).json()["voice_id"]

    with client.websocket_connect("/v1/tts/live") as ws:
        assert _started(ws, {"format": "pcm", "use_active_voice": False})["voice_id"] is None
        ws.send_bytes(ormsgpack.packb({"event": "stop"}))
        _drain(ws)

    # ...and the server-wide voice is untouched for everyone else.
    assert client.get("/v1/voice").json()["voice_id"] == voice_id


def test_clone_route_ignores_the_active_voice(client):
    """`/v1/tts/clone` clones its upload, full stop -- it never inherits."""
    _set_voice(client)
    response = client.post(
        "/v1/tts/clone",
        data={"reference_text": OTHER_TRANSCRIPT, "text": "One shot.", "format": "pcm"},
        files={"reference_audio": ("other.wav", OTHER_CLIP, "audio/wav")},
    )
    assert response.status_code == 200, response.text

    rejected = client.post(
        "/v1/tts/clone",
        data={"reference_text": TRANSCRIPT, "text": "One shot.", "use_active_voice": "false"},
        files={"reference_audio": ("voice.wav", CLIP, "audio/wav")},
    )
    assert rejected.status_code == 400
    assert "use_active_voice" in rejected.json()["error"]


# -- the test page --


def test_the_console_page_is_served(client):
    for path in ("/", "/ui"):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.headers["content-type"].startswith("text/html")
        assert "TTS Console" in response.text
