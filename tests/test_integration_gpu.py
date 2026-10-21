"""Integration tests against a live server with the real model loaded.

These are the only tests that touch CUDA, the vendored fork and the actual
checkpoint, so they are opt-in: they run when a server is already up and ready,
and skip otherwise. Start one with `./run.sh` and re-run pytest.

Point them elsewhere with TTS_E2E_URL, e.g.
    TTS_E2E_URL=http://gpu-box:8080 pytest tests/test_integration_gpu.py -v

They deliberately assert on *audible* output. The failure mode that motivated
them was a decode path that returned NaN, which the wire format happily carried
as well-formed, perfectly silent audio -- every structural check passed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

import pytest

BASE_URL = os.environ.get("TTS_E2E_URL", "http://localhost:8080").rstrip("/")
WS_URL = BASE_URL.replace("http://", "ws://").replace("https://", "wss://") + "/v1/tts/live"

pytestmark = pytest.mark.integration


def _get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def _server_ready() -> bool:
    try:
        return _get("/health").get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return False


requires_server = pytest.mark.skipif(
    not _server_ready(),
    reason=f"no ready TTS server at {BASE_URL} (start one with ./run.sh)",
)


@requires_server
def test_health_and_status_report_a_loaded_engine():
    assert _get("/health")["status"] == "ok"

    status = _get("/status")
    assert status["ready"] is True
    assert status["loading"] is False
    assert status["last_error"] is None
    assert status["sample_rate"] > 0
    assert status["backend"]
    # A real GPU deployment must report real memory, not an empty dict.
    if status["device"].startswith("cuda"):
        assert status["vram"]["total_mib"] > 0
        assert status["vram"]["used_mib"] > 0
        assert status["quantization"]
    assert "settings" in status and "uptime_seconds" in status


@requires_server
def test_synthesis_produces_audible_audio():
    """The regression that matters: structurally valid but silent output."""
    np = pytest.importorskip("numpy")
    ormsgpack = pytest.importorskip("ormsgpack")
    ws_mod = pytest.importorskip("websockets.sync.client")

    with ws_mod.connect(WS_URL, max_size=None, open_timeout=30) as ws:
        ws.send(ormsgpack.packb({"event": "start", "request": {"format": "pcm"}}))
        ws.send(ormsgpack.packb({"event": "text", "text": "Testing one, two, three."}))
        ws.send(ormsgpack.packb({"event": "stop"}))

        pcm = bytearray()
        started = None
        reason = None
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            msg = ormsgpack.unpackb(ws.recv(timeout=600))
            if msg["event"] == "started":
                started = msg
            elif msg["event"] == "audio":
                pcm += msg["audio"]
            elif msg["event"] == "finish":
                reason = msg["reason"]
                break

    assert reason == "stop", f"unexpected finish: {reason}"
    assert started and started["sample_rate"] > 0
    assert len(pcm) > 0, "no audio was returned"

    samples = np.frombuffer(bytes(pcm), dtype="<i2").astype("float32") / 32768.0
    duration = len(samples) / started["sample_rate"]
    peak = float(np.abs(samples).max())
    rms = float(np.sqrt((samples**2).mean()))

    assert duration > 0.3, f"only {duration:.2f}s of audio"
    assert np.isfinite(samples).all(), "decoder emitted non-finite samples"
    assert peak > 0.01, f"audio is effectively silent (peak {peak:.5f})"
    assert rms > 0.001, f"audio has no energy (rms {rms:.6f})"


@requires_server
def test_barge_in_is_acknowledged_promptly():
    ormsgpack = pytest.importorskip("ormsgpack")
    ws_mod = pytest.importorskip("websockets.sync.client")

    long_text = (
        "This is a deliberately long utterance. It has several sentences. "
        "It should still be generating when the cancel arrives. One more for luck."
    )
    with ws_mod.connect(WS_URL, max_size=None, open_timeout=30) as ws:
        ws.send(ormsgpack.packb({"event": "start", "request": {"format": "pcm"}}))
        ws.send(ormsgpack.packb({"event": "text", "text": long_text}))
        ws.send(ormsgpack.packb({"event": "stop"}))

        time.sleep(5.0)
        sent_at = time.perf_counter()
        ws.send(ormsgpack.packb({"event": "cancel"}))

        reason = None
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            msg = ormsgpack.unpackb(ws.recv(timeout=120))
            if msg["event"] == "finish":
                reason = msg["reason"]
                break
        latency_ms = (time.perf_counter() - sent_at) * 1000.0

    assert reason == "cancel", f"cancel after stop was not honoured (got {reason!r})"
    # Generation of one segment takes tens of seconds on modest hardware; an
    # acknowledgement inside two seconds proves it did not wait for that.
    assert latency_ms < 2000, f"barge-in took {latency_ms:.0f}ms"


@requires_server
def test_server_recovers_after_a_client_vanishes():
    """A dropped client must not leave the GPU generating for nobody."""
    ormsgpack = pytest.importorskip("ormsgpack")
    ws_mod = pytest.importorskip("websockets.sync.client")

    ws = ws_mod.connect(WS_URL, max_size=None, open_timeout=30)
    ws.send(ormsgpack.packb({"event": "start", "request": {"format": "pcm"}}))
    ws.send(ormsgpack.packb({"event": "text", "text": "Vanishing mid sentence, goodbye."}))
    ws.send(ormsgpack.packb({"event": "stop"}))
    time.sleep(4.0)
    ws.close()

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        s = _get("/status")
        if s["active_sessions"] == 0 and s["active_jobs"] == 0:
            break
        time.sleep(2.0)
    else:
        pytest.fail(f"server still busy 120s after disconnect: {_get('/status')}")

    assert _get("/health")["status"] == "ok"
    assert _get("/status")["total_failed"] == 0


def _clone_route_available() -> bool:
    """A deliberately invalid POST: 404 means the server predates the route."""
    request = urllib.request.Request(
        f"{BASE_URL}/v1/tts/clone", data=b"", method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(request, timeout=5.0)
        return True
    except urllib.error.HTTPError as exc:
        return exc.code != 404
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


requires_clone_route = pytest.mark.skipif(
    _server_ready() and not _clone_route_available(),
    reason=f"the server at {BASE_URL} predates POST /v1/tts/clone; restart it",
)


@requires_server
@requires_clone_route
def test_clone_route_speaks_new_text_in_the_reference_voice():
    """End-to-end voice cloning, with a reference clip the server itself made.

    Synthesizing the reference first is what makes this self-contained *and*
    honest: the transcript handed to `/v1/tts/clone` is known to be exactly
    what the clip says, which is the contract the route enforces and the one
    thing a checked-in fixture clip could not guarantee stayed true.
    """
    np = pytest.importorskip("numpy")
    httpx = pytest.importorskip("httpx")
    ormsgpack = pytest.importorskip("ormsgpack")
    ws_mod = pytest.importorskip("websockets.sync.client")

    import io
    import wave

    reference_text = "The quick brown fox jumps over the lazy dog."

    # 1. Produce a reference clip, and know its transcript exactly.
    pcm = bytearray()
    sample_rate = None
    with ws_mod.connect(WS_URL, max_size=None, open_timeout=30) as ws:
        ws.send(ormsgpack.packb({"event": "start", "request": {"format": "pcm"}}))
        ws.send(ormsgpack.packb({"event": "text", "text": reference_text}))
        ws.send(ormsgpack.packb({"event": "stop"}))
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            msg = ormsgpack.unpackb(ws.recv(timeout=600))
            if msg["event"] == "started":
                sample_rate = msg["sample_rate"]
            elif msg["event"] == "audio":
                pcm += msg["audio"]
            elif msg["event"] == "finish":
                assert msg["reason"] == "stop", msg
                break

    assert sample_rate and len(pcm) > 0, "could not build a reference clip"

    clip = io.BytesIO()
    with wave.open(clip, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(bytes(pcm))
    reference_wav = clip.getvalue()

    # 2. Clone it.
    response = httpx.post(
        f"{BASE_URL}/v1/tts/clone",
        files={"reference_audio": ("reference.wav", reference_wav, "audio/wav")},
        data={
            "reference_text": reference_text,
            "text": "Cloned speech, in the voice of the reference clip.",
            "format": "pcm",
            "seed": "4242",
        },
        timeout=600.0,
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-audio-format"] == "pcm"
    assert response.headers["x-seed"] == "4242"
    assert int(response.headers["x-sample-rate"]) == sample_rate

    # 3. The same audibility checks as the streaming path: structurally valid
    #    but silent output is the failure mode that motivates this whole file.
    samples = np.frombuffer(response.content, dtype="<i2").astype("float32") / 32768.0
    duration = len(samples) / sample_rate
    assert duration > 0.3, f"only {duration:.2f}s of cloned audio"
    assert np.isfinite(samples).all(), "decoder emitted non-finite samples"
    assert float(np.abs(samples).max()) > 0.01, "cloned audio is effectively silent"
    assert float(np.sqrt((samples**2).mean())) > 0.001, "cloned audio has no energy"


@requires_server
@requires_clone_route
def test_clone_route_rejects_a_missing_transcript():
    """The contract the route exists to enforce, checked against a live server."""
    httpx = pytest.importorskip("httpx")

    response = httpx.post(
        f"{BASE_URL}/v1/tts/clone",
        files={"reference_audio": ("reference.wav", b"RIFF....WAVE", "audio/wav")},
        data={"text": "This should never be synthesized."},
        timeout=30.0,
    )
    assert response.status_code == 400
    assert "reference_text" in response.json()["error"]


@requires_server
@requires_clone_route
def test_active_voice_is_used_by_the_websocket():
    """The end-to-end promise: set a voice over HTTP, the socket speaks in it.

    Like the clone test, the reference is synthesized here so its transcript is
    known to be exactly right. The server's existing active voice, if any, is
    restored afterwards -- these tests run against a live deployment and must
    not leave it speaking in a fixture voice.
    """
    np = pytest.importorskip("numpy")
    httpx = pytest.importorskip("httpx")
    ormsgpack = pytest.importorskip("ormsgpack")
    ws_mod = pytest.importorskip("websockets.sync.client")

    import io
    import wave

    reference_text = "The quick brown fox jumps over the lazy dog."

    def speak(request, text="Speaking with whichever voice is in force."):
        pcm = bytearray()
        started = None
        with ws_mod.connect(WS_URL, max_size=None, open_timeout=30) as ws:
            ws.send(ormsgpack.packb({"event": "start", "request": request}))
            ws.send(ormsgpack.packb({"event": "text", "text": text}))
            ws.send(ormsgpack.packb({"event": "stop"}))
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                msg = ormsgpack.unpackb(ws.recv(timeout=600))
                if msg["event"] == "started":
                    started = msg
                elif msg["event"] == "audio":
                    pcm += msg["audio"]
                elif msg["event"] == "finish":
                    assert msg["reason"] == "stop", msg
                    break
        assert started is not None
        return started, bytes(pcm)

    # Grab the clip's *bytes* now, not in the finally block. Fetching them
    # after the delete below is too late - /v1/voice/audio 404s once the voice
    # is gone, so the restore silently never happened and this test ate
    # whatever voice the server was serving with.
    previous = httpx.get(f"{BASE_URL}/v1/voice", timeout=10.0)
    previous_voice = previous.json() if previous.status_code == 200 else None
    previous_audio = None
    if previous_voice is not None:
        saved = httpx.get(f"{BASE_URL}/v1/voice/audio", timeout=30.0)
        previous_audio = saved.content if saved.status_code == 200 else None

    try:
        # 1. Build a reference clip whose transcript we know exactly.
        started, pcm = speak({"format": "pcm"}, reference_text)
        sample_rate = started["sample_rate"]
        assert len(pcm) > 0

        clip = io.BytesIO()
        with wave.open(clip, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)

        # 2. Make it the active voice.
        response = httpx.post(
            f"{BASE_URL}/v1/voice",
            files={"reference_audio": ("reference.wav", clip.getvalue(), "audio/wav")},
            data={"reference_text": reference_text},
            timeout=60.0,
        )
        assert response.status_code == 200, response.text
        voice_id = response.json()["voice_id"]
        assert httpx.get(f"{BASE_URL}/v1/voice", timeout=10.0).json()["voice_id"] == voice_id

        # 3. A bare `start` frame now inherits it, and still makes real sound.
        started, pcm = speak({"format": "pcm"})
        assert started["voice_id"] == voice_id, "the socket ignored the active voice"

        samples = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768.0
        assert len(samples) / sample_rate > 0.3
        assert np.isfinite(samples).all()
        assert float(np.abs(samples).max()) > 0.01, "cloned audio is effectively silent"

        # 4. Opting out is per-session and leaves the server-wide voice alone.
        started, _ = speak({"format": "pcm", "use_active_voice": False})
        assert started["voice_id"] is None
        assert httpx.get(f"{BASE_URL}/v1/voice", timeout=10.0).json()["voice_id"] == voice_id

        # 5. Clearing it returns the socket to the default voice.
        assert httpx.delete(f"{BASE_URL}/v1/voice", timeout=10.0).json() == {"cleared": True}
        started, _ = speak({"format": "pcm"})
        assert started["voice_id"] is None

    finally:
        httpx.delete(f"{BASE_URL}/v1/voice", timeout=10.0)
        if previous_voice is not None and previous_audio is not None:
            httpx.post(
                f"{BASE_URL}/v1/voice",
                files={
                    "reference_audio": (
                        previous_voice["filename"] or "restored.wav",
                        previous_audio,
                    )
                },
                data={"reference_text": previous_voice["text"]},
                timeout=60.0,
            )


@requires_server
def test_the_console_page_is_served_by_a_live_server():
    request = urllib.request.Request(f"{BASE_URL}/", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            body = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            pytest.skip("static/index.html is not installed on this server")
        raise

    assert content_type.startswith("text/html")
    assert "TTS Console" in body
    assert "/v1/tts/live" in body and "/v1/voice" in body
