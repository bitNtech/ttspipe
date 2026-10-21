"""End-to-end WebSocket flow against `TTS_ENGINE=fake` -- no GPU, no model.

Exercises the full stack together (app.main, app.ws.router, app.session,
app.segmenter, app.audio, app.engine.fake) the way a real client would, over
FastAPI's TestClient WebSocket support.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TTS_ENGINE", "fake")

import ormsgpack
import pytest
from starlette.testclient import TestClient

from app.main import app


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _unpack(raw: bytes) -> dict:
    return ormsgpack.unpackb(raw)


def _wait_ready(client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        r = client.get("/health")
        last = r.json()
        if last.get("status") == "ok":
            return
        time.sleep(0.02)
    raise TimeoutError(f"engine never became ready, last /health={last!r}")


@pytest.fixture()
def client():
    with TestClient(app) as c:
        _wait_ready(c)
        yield c


def _drain_until(ws, event_name: str, limit: int = 500) -> tuple[list[dict], dict]:
    """Read frames until one with `event == event_name` arrives; return (others, that_one)."""
    others = []
    for _ in range(limit):
        frame = _unpack(ws.receive_bytes())
        if frame["event"] == event_name:
            return others, frame
        others.append(frame)
    raise AssertionError(f"never saw event {event_name!r} within {limit} frames")


# -- HTTP surface --


def test_health_reports_ok_once_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_reports_engine_and_server_state(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["backend"] == "fake"
    assert body["ready"] is True
    assert body["sample_rate"] > 0
    assert isinstance(body["uptime_seconds"], float)
    assert body["active_sessions"] == 0
    assert "settings" in body and isinstance(body["settings"], dict)
    assert body["settings"]["max_sessions"] > 0


# -- full flow: start -> streamed text -> audio frames -> stop -> finish --


def test_full_flow_start_text_audio_stop(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "wav"}}))
        started = _unpack(ws.receive_bytes())
        assert started["event"] == "started"
        assert started["format"] == "wav"
        assert started["sample_rate"] > 0
        assert isinstance(started["seed"], int)

        # feed text in several fragments, as a streaming client would
        for piece in ["Hello there, ", "this is a streaming test. ", "Here is a second sentence."]:
            ws.send_bytes(_pack({"event": "text", "text": piece}))
        ws.send_bytes(_pack({"event": "stop"}))

        others, finish = _drain_until(ws, "finish")
        audio_frames = [f for f in others if f["event"] == "audio"]

        assert len(audio_frames) >= 2
        assert finish["reason"] == "stop"
        assert finish["error"] is None

        times = [f["time"] for f in audio_frames]
        assert times == sorted(times)
        assert all(t > 0 for t in times)

        # the very first audio payload carries the WAV header
        assert audio_frames[0]["audio"][:4] == b"RIFF"
        for f in audio_frames:
            assert isinstance(f["audio"], (bytes, bytearray))
            assert len(f["audio"]) > 0


def test_pcm_format_has_no_header_and_even_byte_length(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        started = _unpack(ws.receive_bytes())
        assert started["format"] == "pcm"

        ws.send_bytes(_pack({"event": "text", "text": "Short pcm test."}))
        ws.send_bytes(_pack({"event": "stop"}))

        others, finish = _drain_until(ws, "finish")
        audio_frames = [f for f in others if f["event"] == "audio"]
        assert audio_frames
        for f in audio_frames:
            assert len(f["audio"]) % 2 == 0  # whole int16 samples, no header bytes
        assert finish["reason"] == "stop"


# -- cancel / barge-in --


def test_cancel_stops_audio_promptly_and_session_stays_usable(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        _unpack(ws.receive_bytes())  # started

        long_text = "word " * 300  # plenty long enough to still be streaming
        ws.send_bytes(_pack({"event": "text", "text": long_text}))

        first = _unpack(ws.receive_bytes())
        assert first["event"] == "audio"

        ws.send_bytes(_pack({"event": "cancel"}))

        # audio must stop *promptly*: only a small, bounded number of
        # already-in-flight frames may still arrive before the finish(cancel)
        stray_audio_frames, cancel_finish = _drain_until(ws, "finish", limit=20)
        assert all(f["event"] == "audio" for f in stray_audio_frames)
        assert len(stray_audio_frames) <= 5, "cancel did not stop audio promptly"
        assert cancel_finish["reason"] == "cancel"

        # the session is immediately usable again, with no new 'start'
        ws.send_bytes(_pack({"event": "text", "text": "Still here after the interruption."}))
        ws.send_bytes(_pack({"event": "stop"}))

        others, finish = _drain_until(ws, "finish")
        assert any(f["event"] == "audio" for f in others)
        assert finish["reason"] == "stop"


def test_multiple_turns_on_one_connection(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        for i in range(2):
            ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
            started = _unpack(ws.receive_bytes())
            assert started["event"] == "started"

            ws.send_bytes(_pack({"event": "text", "text": f"Turn number {i}."}))
            ws.send_bytes(_pack({"event": "stop"}))

            others, finish = _drain_until(ws, "finish")
            assert any(f["event"] == "audio" for f in others)
            assert finish["reason"] == "stop"


# -- protocol errors / event ordering --


def test_text_before_start_is_protocol_error_and_closes(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "text", "text": "too early"}))
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "finish"
        assert frame["reason"] == "error"
        assert frame["error"]

        with pytest.raises(Exception):
            ws.receive_bytes()  # server closed the connection


def test_second_start_while_active_is_protocol_error(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {}}))
        _unpack(ws.receive_bytes())  # started

        ws.send_bytes(_pack({"event": "start", "request": {}}))
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "finish"
        assert frame["reason"] == "error"


def test_malformed_frame_is_protocol_error(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(b"\xff\xfe not a real frame")
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "finish"
        assert frame["reason"] == "error"


# -- ping/pong --


def test_ping_pong(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "ping"}))
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "pong"


def test_ping_before_start_is_allowed(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "ping"}))
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "pong"
        # session should still be idle and accept a normal start afterwards
        ws.send_bytes(_pack({"event": "start", "request": {}}))
        started = _unpack(ws.receive_bytes())
        assert started["event"] == "started"
