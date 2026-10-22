"""`POST /v1/tts`, `POST /v1/audio/speech`, `GET /v1/voices`, `GET /v1/info`,
against `TTS_ENGINE=fake`."""

from __future__ import annotations

import os
import time

os.environ.setdefault("TTS_ENGINE", "fake")

import pytest
from starlette.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        deadline = time.monotonic() + 5
        while c.get("/health").json().get("status") != "ok":
            assert time.monotonic() < deadline, "engine never became ready"
            time.sleep(0.02)
        yield c


def test_tts_streams_audio_in_the_requested_shape(client):
    r = client.post("/v1/tts", json={"text": "Hello there. Second sentence.", "format": "pcm",
                                     "sample_rate": 16000, "seed": 7})
    assert r.status_code == 200, r.text
    assert r.headers["x-sample-rate"] == "16000"
    assert r.headers["x-audio-format"] == "pcm"
    assert r.headers["x-seed"] == "7"
    assert r.headers["x-voice-id"]
    assert len(r.content) > 1000 and len(r.content) % 2 == 0


def test_tts_accepts_a_form_body_and_defaults_to_wav(client):
    r = client.post("/v1/tts", data={"text": "Form encoded."})
    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF"


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 400),
        ({"text": "x" * 9000}, 413),
        ({"text": "hi", "temperature": 7}, 422),
        ({"text": "hi", "nonsense": 1}, 422),
        ({"text": "hi", "reference_id": "../../etc"}, 422),
        ({"text": "hi", "reference_id": "no-such-voice"}, 404),
    ],
)
def test_tts_rejects_bad_requests(client, body, status):
    assert client.post("/v1/tts", json=body).status_code == status


def test_openai_speech_pcm_is_24khz(client):
    r = client.post("/v1/audio/speech", json={"model": "tts-1", "input": "Hi.", "voice": "alloy",
                                              "response_format": "pcm"})
    assert r.status_code == 200, r.text
    assert r.headers["x-sample-rate"] == "24000"
    assert r.headers["content-type"] == "application/octet-stream"


def test_openai_speech_defaults_to_mp3_and_flags_speed(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "nova", "speed": 1.5})
    assert r.status_code == 200, r.text
    assert r.headers["x-audio-format"] in ("mp3", "wav")  # wav only if libsndfile lacks mp3
    assert "speed" in r.headers.get("x-warning", "")


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"input": "", "voice": "alloy"}, 400),
        ({"input": "Hi.", "response_format": "opus"}, 400),
        ({"input": "Hi.", "voice": "somebody-else"}, 404),
    ],
)
def test_openai_speech_rejects_bad_requests(client, body, status):
    assert client.post("/v1/audio/speech", json=body).status_code == status


def test_openai_speech_model_voice_skips_cloning(client):
    r = client.post("/v1/audio/speech", json={"input": "Hi.", "voice": "model", "response_format": "wav"})
    assert r.status_code == 200
    assert r.headers["x-voice-id"] == "model"


def test_voices_and_info(client):
    voices = client.get("/v1/voices").json()
    assert voices["default"] in ("active", "model")
    assert isinstance(voices["voices"], list)

    info = client.get("/v1/info").json()
    assert info["ready"] is True
    assert info["native_sample_rate"] == 24000
    assert "mulaw" in info["formats"] and 16000 in info["sample_rates"]
    assert info["routes"]["stream"] == "/v1/tts/live"
