"""The production surface: output rate conversion, mu-law, context tags,
readiness, metrics, draining, API keys, idle timeouts and engine timeouts.

All against `TTS_ENGINE=fake`, like the rest of the transport tests.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TTS_ENGINE", "fake")

import numpy as np
import ormsgpack
import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import config
from app.audio import (
    SUPPORTED_SAMPLE_RATES,
    PCMEncoder,
    ResamplingEncoder,
    create_encoder,
    float_to_mulaw,
    float_to_pcm16,
)
from app.engine.base import AudioChunk, CancelToken
from app.main import app, create_app
from app.metrics import METRICS
from app.protocol import (
    ProtocolError,
    audio_frame,
    decode_client_frame,
    finish_frame,
)


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _unpack(raw: bytes) -> dict:
    return ormsgpack.unpackb(raw)


def _wait_ready(client: TestClient, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.get("/ready").status_code == 200:
            return
        time.sleep(0.02)
    raise TimeoutError("engine never became ready")


@pytest.fixture()
def client():
    with TestClient(app) as c:
        _wait_ready(c)
        yield c


@pytest.fixture()
def app_with_env(monkeypatch):
    """Build a fresh app under extra environment, restoring settings after."""

    def build(**env: str):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        config.reset_settings_cache()
        return create_app()

    yield build
    config.reset_settings_cache()


def _turn(ws, request: dict, text: str, **start) -> tuple[dict, bytes, list[dict]]:
    ws.send_bytes(_pack({"event": "start", "request": request, **start}))
    ws.send_bytes(_pack({"event": "text", "text": text}))
    ws.send_bytes(_pack({"event": "stop"}))
    started = None
    audio = b""
    frames = []
    for _ in range(2000):
        frame = _unpack(ws.receive_bytes())
        frames.append(frame)
        if frame["event"] == "started":
            started = frame
        elif frame["event"] == "audio":
            audio += frame["audio"]
        elif frame["event"] == "finish":
            return started, audio, frames
    raise AssertionError("turn never finished")


# -- audio --


def test_mulaw_matches_g711_reference():
    audioop = pytest.importorskip("audioop")
    x = np.linspace(-1.0, 1.0, 100_001, dtype=np.float32)
    assert float_to_mulaw(x) == audioop.lin2ulaw(float_to_pcm16(x), 2)


def test_resampling_is_seamless_across_chunks():
    """Chunked streaming conversion must equal one-shot conversion: no clicks."""
    rate_in, rate_out = 44100, 16000
    t = np.arange(rate_in) / rate_in
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def run(pieces):
        enc = ResamplingEncoder(PCMEncoder(), rate_in, rate_out)
        out = b"".join(enc.encode(AudioChunk(p, rate_in)) for p in pieces) + enc.finalize()
        return np.frombuffer(out, "<i2").astype(np.int32)

    whole = run([tone])
    chunked = run(np.array_split(tone, 37))
    assert len(whole) == len(chunked) == pytest.approx(rate_out, abs=2)
    assert np.max(np.abs(whole - chunked)) <= 1  # rounding only


def test_create_encoder_rejects_unsupported_output_rate():
    from app.audio import UnsupportedFormatError

    with pytest.raises(UnsupportedFormatError):
        create_encoder("pcm", 44100, out_rate=11025)


# -- protocol --


def test_sample_rate_is_validated():
    with pytest.raises(ProtocolError):
        decode_client_frame(_pack({"event": "start", "request": {"sample_rate": 11025}}))
    for rate in SUPPORTED_SAMPLE_RATES:
        decode_client_frame(_pack({"event": "start", "request": {"sample_rate": rate}}))


def test_untagged_frames_are_unchanged():
    assert "context_id" not in _unpack(audio_frame(b"x", 1.0))
    assert "context_id" not in _unpack(finish_frame("stop"))
    assert _unpack(finish_frame("stop", context_id="c1"))["context_id"] == "c1"


def test_cancel_token_child():
    parent = CancelToken()
    child = parent.child()
    child.cancel()
    assert child.cancelled and not parent.cancelled
    other = parent.child()
    parent.cancel()
    assert other.cancelled


# -- WebSocket --


@pytest.mark.parametrize(("fmt", "rate", "width"), [("pcm", 16000, 2), ("pcm", 24000, 2), ("mulaw", 8000, 1)])
def test_ws_output_rate_and_format(client, fmt, rate, width):
    with client.websocket_connect("/v1/tts/live") as ws:
        started, audio, frames = _turn(
            ws, {"format": fmt, "sample_rate": rate}, "Resample this sentence, please."
        )
    assert started["sample_rate"] == rate
    assert started["format"] == fmt
    duration_ms = max(f["time"] for f in frames if f["event"] == "audio")
    samples = len(audio) / width
    assert samples == pytest.approx(duration_ms * rate / 1000.0, rel=0.01)


def test_ws_context_id_is_echoed(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        _started, _audio, frames = _turn(
            ws, {"format": "pcm"}, "Tag every frame of this turn.", context_id="turn-42"
        )
    tagged = [f for f in frames if f["event"] in ("started", "audio", "finish")]
    assert tagged and all(f.get("context_id") == "turn-42" for f in tagged)


def test_server_default_sample_rate(app_with_env):
    custom = app_with_env(TTS_DEFAULT_SAMPLE_RATE="16000")
    with TestClient(custom) as c:
        _wait_ready(c)
        with c.websocket_connect("/v1/tts/live") as ws:
            started, _audio, _frames = _turn(ws, {"format": "pcm"}, "Default rate applies.")
            assert started["sample_rate"] == 16000
            started, _audio, _frames = _turn(
                ws, {"format": "pcm", "sample_rate": 24000}, "Explicit rate wins."
            )
            assert started["sample_rate"] == 24000


# -- operational endpoints --


def test_ready_and_metrics(client):
    assert client.get("/ready").status_code == 200
    before = METRICS.turns.value(reason="stop")
    with client.websocket_connect("/v1/tts/live") as ws:
        _turn(ws, {"format": "pcm"}, "Count this turn in the metrics.")
    body = client.get("/metrics").text
    assert METRICS.turns.value(reason="stop") == before + 1
    for name in (
        "tts_first_audio_seconds_bucket",
        "tts_sessions_total",
        'tts_turns_total{reason="stop"}',
        "tts_ready 1.0",
        "tts_engine_queue_depth",
    ):
        assert name in body
    assert METRICS.first_audio.count > 0


def test_not_ready_while_the_active_voice_warms(client):
    """A turn routed in before the voice prefix is warm queues behind it."""
    client.app.state.warming = True
    try:
        assert client.get("/ready").json() == {"ready": False, "reason": "warming"}
        assert client.get("/health").json()["status"] == "ok"  # alive, just not ready
    finally:
        client.app.state.warming = False
    assert client.get("/ready").status_code == 200


def test_drain_refuses_new_sessions(app_with_env):
    custom = app_with_env()
    with TestClient(custom) as c:
        _wait_ready(c)
        with c.websocket_connect("/v1/tts/live") as open_ws:
            assert c.post("/admin/drain").json()["active_sessions"] == 1
            assert c.get("/ready").json() == {"ready": False, "reason": "draining"}
            assert c.get("/health").status_code == 200  # still alive, just not taking work
            with pytest.raises(WebSocketDisconnect) as exc:
                with c.websocket_connect("/v1/tts/live") as ws:
                    ws.receive_bytes()
            assert exc.value.code == 1012
            # The session that was already open keeps working.
            started, audio, _ = _turn(open_ws, {"format": "pcm"}, "Still serving you.")
            assert started and audio


def test_health_reports_a_dead_worker(client):
    engine = client.app.state.engine
    type(engine).worker_dead = property(lambda self: True)
    try:
        assert client.get("/health").status_code == 503
        assert client.get("/ready").status_code == 503
    finally:
        del type(engine).worker_dead


# -- auth --


def test_api_keys(app_with_env):
    secured = app_with_env(TTS_API_KEYS="k-one, k-two")
    with TestClient(secured) as c:
        _wait_ready(c)
        assert c.get("/ready").status_code == 200  # probes stay open
        assert c.get("/status").status_code == 401
        assert c.get("/metrics").status_code == 401
        assert c.get("/status", headers={"Authorization": "Bearer k-two"}).status_code == 200
        assert c.get("/status", headers={"X-API-Key": "k-one"}).status_code == 200
        assert c.get("/status", headers={"X-API-Key": "nope"}).status_code == 401
        status = c.get("/status?api_key=k-one").json()
        assert status["settings"]["api_keys_configured"] == 2
        assert "k-one" not in str(status)

        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/v1/tts/live") as ws:
                ws.receive_bytes()
        with c.websocket_connect("/v1/tts/live", headers={"Authorization": "Bearer k-one"}) as ws:
            started, audio, _ = _turn(ws, {"format": "pcm"}, "Authorised speech.")
            assert started and audio


# -- timeouts --


def test_idle_session_is_closed(app_with_env):
    custom = app_with_env(TTS_SESSION_IDLE_TIMEOUT_S="0.3")
    with TestClient(custom) as c:
        _wait_ready(c)
        with c.websocket_connect("/v1/tts/live") as ws:
            # Activity resets the clock...
            for _ in range(3):
                ws.send_bytes(_pack({"event": "ping"}))
                assert _unpack(ws.receive_bytes())["event"] == "pong"
                time.sleep(0.15)
            # ...and silence ends the session.
            with pytest.raises(WebSocketDisconnect):
                ws.receive_bytes()
        assert c.get("/status").json()["active_sessions"] == 0


def test_busy_session_is_not_idle_closed(app_with_env):
    """A long reply keeps the inbound side quiet; that is not idleness."""
    custom = app_with_env(TTS_SESSION_IDLE_TIMEOUT_S="0.3")
    with TestClient(custom) as c:
        _wait_ready(c)
        with c.websocket_connect("/v1/tts/live") as ws:
            text = " ".join(["This is a fairly long sentence to keep the engine busy."] * 6)
            started, audio, frames = _turn(ws, {"format": "pcm"}, text)
            assert frames[-1]["reason"] == "stop"
