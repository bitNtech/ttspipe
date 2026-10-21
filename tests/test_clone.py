"""`POST /v1/tts/clone` against `TTS_ENGINE=fake` -- no GPU, no model weights.

The fake engine ignores `references` (it has no codec to encode them with),
so what these tests can prove is everything up to the engine seam: both body
encodings, the validation that protects the "transcript must be exact"
contract, format negotiation, streamed response framing, and the headers a
caller needs to interpret the bytes. Cloning *quality* is a GPU concern and
lives in tests/test_integration_gpu.py.
"""

from __future__ import annotations

import base64
import io
import os
import struct
import time
import wave

os.environ.setdefault("TTS_ENGINE", "fake")

import pytest
from starlette.testclient import TestClient

from app.main import app

REFERENCE_TEXT = "This is the exact transcript of the reference clip."


def _reference_wav(seconds: float = 1.0, sample_rate: int = 24000) -> bytes:
    """A real, well-formed mono 16-bit WAV, so this is not fake-engine-only."""
    frames = int(seconds * sample_rate)
    samples = b"".join(struct.pack("<h", (i % 2000) - 1000) for i in range(frames))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples)
    return buf.getvalue()


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
def client():
    with TestClient(app) as c:
        _wait_ready(c)
        yield c


@pytest.fixture(scope="module")
def reference_audio() -> bytes:
    return _reference_wav()


def _post_multipart(client: TestClient, reference_audio: bytes, **fields):
    data = {"text": "Say this in the cloned voice.", "reference_text": REFERENCE_TEXT}
    data.update({k: str(v) for k, v in fields.items()})
    return client.post(
        "/v1/tts/clone",
        data=data,
        files={"reference_audio": ("ref.wav", reference_audio, "audio/wav")},
    )


def _post_json(client: TestClient, payload: dict):
    return client.post("/v1/tts/clone", json=payload)


# -- happy paths, both encodings --


def test_multipart_upload_returns_streamed_wav(client, reference_audio):
    r = _post_multipart(client, reference_audio)

    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["x-audio-format"] == "wav"
    assert int(r.headers["x-sample-rate"]) > 0
    assert int(r.headers["x-seed"]) >= 0
    assert r.headers["x-request-id"]

    body = r.content
    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
    assert len(body) > 44  # a header and then actual audio


def test_json_body_with_base64_audio(client, reference_audio):
    r = _post_json(
        client,
        {
            "text": "Say this in the cloned voice.",
            "reference_text": REFERENCE_TEXT,
            "reference_audio": base64.b64encode(reference_audio).decode("ascii"),
        },
    )

    assert r.status_code == 200, r.text
    assert r.content[:4] == b"RIFF"


def test_pcm_format_is_headerless_int16(client, reference_audio):
    r = _post_multipart(client, reference_audio, format="pcm")

    assert r.status_code == 200, r.text
    assert r.headers["x-audio-format"] == "pcm"
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert r.content[:4] != b"RIFF"
    assert len(r.content) % 2 == 0  # whole samples, no stray header bytes


def test_sampling_parameters_are_accepted_and_bounded(client, reference_audio):
    ok = _post_multipart(client, reference_audio, temperature=0.6, top_p=0.9, top_k=20)
    assert ok.status_code == 200, ok.text

    out_of_range = _post_multipart(client, reference_audio, temperature=9.0)
    assert out_of_range.status_code == 422
    assert out_of_range.json()["type"] == "ValidationError"


def test_explicit_seed_is_honoured_and_reproducible(client, reference_audio):
    first = _post_multipart(client, reference_audio, format="pcm", seed=12345)
    second = _post_multipart(client, reference_audio, format="pcm", seed=12345)

    assert first.headers["x-seed"] == "12345"
    assert second.headers["x-seed"] == "12345"
    assert first.content == second.content  # the fake engine is deterministic


def test_blank_optional_form_fields_are_ignored(client, reference_audio):
    # An HTML form posts every field it has, including the untouched ones.
    r = _post_multipart(client, reference_audio, top_p="", seed="", language="")
    assert r.status_code == 200, r.text


# -- the contract that matters: the transcript must be there, and exact --


def test_missing_reference_text_is_rejected_with_an_explanation(client, reference_audio):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "Say this."},
        files={"reference_audio": ("ref.wav", reference_audio, "audio/wav")},
    )
    assert r.status_code == 400
    message = r.json()["error"]
    assert "reference_text" in message
    assert "verbatim" in message or "exact" in message


def test_whitespace_only_reference_text_is_rejected(client, reference_audio):
    r = _post_multipart(client, reference_audio, reference_text="   ")
    assert r.status_code == 400
    assert "reference_text" in r.json()["error"]


def test_missing_reference_audio_is_rejected(client):
    r = client.post(
        "/v1/tts/clone",
        data={"text": "Say this.", "reference_text": REFERENCE_TEXT},
    )
    assert r.status_code == 400
    assert "reference_audio" in r.json()["error"]


def test_missing_text_is_rejected(client, reference_audio):
    r = client.post(
        "/v1/tts/clone",
        data={"reference_text": REFERENCE_TEXT},
        files={"reference_audio": ("ref.wav", reference_audio, "audio/wav")},
    )
    assert r.status_code == 400
    assert "'text' is required" in r.json()["error"]


# -- rejected inputs --


def test_reference_id_is_not_accepted_here(client, reference_audio):
    r = _post_multipart(client, reference_audio, reference_id="some-voice")
    assert r.status_code == 400
    assert "reference_id" in r.json()["error"]


def test_unknown_field_is_a_validation_error(client, reference_audio):
    r = _post_multipart(client, reference_audio, speed=2.0)
    assert r.status_code == 422
    assert r.json()["type"] == "ValidationError"


def test_unsupported_content_type_is_415(client):
    r = client.post(
        "/v1/tts/clone",
        content=b"text=hello",
        headers={"content-type": "text/plain"},
    )
    assert r.status_code == 415
    assert r.json()["type"] == "UnsupportedMediaType"


def test_malformed_json_body_is_400(client):
    r = client.post(
        "/v1/tts/clone",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 400


def test_json_array_body_is_400(client):
    r = client.post("/v1/tts/clone", json=["not", "an", "object"])
    assert r.status_code == 400


# -- the one-shot synthesis primitive the route is built on --


async def _ready_engine():
    """A loaded `FakeEngine`. A plain helper, not a fixture: pytest-asyncio's
    async fixtures need their own decorator and this file has no other need
    for one."""
    from app.engine.fake import FakeEngine

    engine = FakeEngine()
    await engine.load()
    return engine


def _one_shot(engine, text: str, **params):
    from app.config import Settings
    from app.protocol import TTSParams
    from app.synthesis import OneShotSynthesis

    return OneShotSynthesis(
        engine=engine, settings=Settings(), params=TTSParams(**params), text=text
    )


@pytest.mark.asyncio
async def test_abandoning_the_stream_cancels_the_engine_job():
    """A client that hangs up must not leave the GPU generating for nobody."""
    engine = await _ready_engine()
    synthesis = _one_shot(
        engine,
        "A long utterance. With several sentences. That nobody waits for.",
        format="pcm",
    )
    body = synthesis.stream()

    assert await anext(body)  # the first payload arrives
    assert not synthesis._cancel.cancelled

    await body.aclose()  # what Starlette does when the client disconnects
    assert synthesis._cancel.cancelled


@pytest.mark.asyncio
async def test_empty_text_still_yields_a_well_formed_container():
    engine = await _ready_engine()
    synthesis = _one_shot(engine, "   ", format="wav")
    body = b"".join([piece async for piece in synthesis.stream()])

    assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"


@pytest.mark.asyncio
async def test_stream_is_single_use():
    engine = await _ready_engine()
    synthesis = _one_shot(engine, "Once is enough.", format="pcm")
    _ = [piece async for piece in synthesis.stream()]

    with pytest.raises(RuntimeError, match="single-use"):
        await anext(synthesis.stream())
