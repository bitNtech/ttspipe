"""Turn-lifecycle regressions: encoder finalisation and pipelined turns.

Both behaviours here were broken once and are invisible to a wav-only test,
because `WavEncoder.finalize()` legitimately returns nothing. Only the
soundfile-backed formats carry a tail, so that is what these assert.
"""

from __future__ import annotations

import os
import shutil
import subprocess

os.environ.setdefault("TTS_ENGINE", "fake")

import ormsgpack
import pytest
from starlette.testclient import TestClient

import app.session as session_mod
from app.audio import StreamEncoder
from app.main import app

TAIL = b"__TAIL_SENTINEL__"


def _pack(payload: dict) -> bytes:
    return ormsgpack.packb(payload)


def _unpack(raw: bytes) -> dict:
    return ormsgpack.unpackb(raw)


class _SentinelEncoder(StreamEncoder):
    """Emits a recognisable tail so we can prove finalize() reached the wire."""

    def __init__(self) -> None:
        self.finalized = 0

    def header(self) -> bytes | None:
        return b"__HEAD__"

    def encode(self, chunk) -> bytes:
        return b"\x00\x00" * len(chunk.samples)

    def finalize(self) -> bytes:
        self.finalized += 1
        return TAIL if self.finalized == 1 else b""


@pytest.fixture()
def client():
    with TestClient(app) as c:
        for _ in range(500):
            if c.get("/health").json().get("status") == "ok":
                break
        yield c


def _collect(ws) -> tuple[bytes, list[dict]]:
    """Read to the first `finish`, returning concatenated audio and all frames."""
    audio = bytearray()
    frames = []
    for _ in range(500):
        frame = _unpack(ws.receive_bytes())
        frames.append(frame)
        if frame["event"] == "audio":
            audio += frame["audio"]
        if frame["event"] == "finish":
            return bytes(audio), frames
    raise AssertionError("never saw a finish frame")


def test_stop_flushes_the_encoder_tail(client, monkeypatch):
    encoders: list[_SentinelEncoder] = []

    def fake_resolve_format(fmt, sample_rate):
        enc = _SentinelEncoder()
        encoders.append(enc)
        return "wav", enc, None

    monkeypatch.setattr(session_mod, "resolve_format", fake_resolve_format)

    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "wav"}}))
        ws.send_bytes(_pack({"event": "text", "text": "Finalize me please."}))
        ws.send_bytes(_pack({"event": "stop"}))
        audio, frames = _collect(ws)

    assert audio.startswith(b"__HEAD__")
    assert audio.endswith(TAIL), "encoder tail never reached the client"
    assert frames[-1]["event"] == "finish" and frames[-1]["reason"] == "stop"
    # The tail must precede the finish frame, not trail after it.
    assert encoders and encoders[0].finalized >= 1


@pytest.mark.parametrize("fmt", ["mp3", "flac"])
def test_encoder_tail_is_not_empty(fmt):
    """The tail these formats flush at close() is what the bug used to drop.

    It is not a rounding error: for a short mp3 the tail outweighs everything
    streamed before it, so losing it loses most of the audio.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    from app.audio import UnsupportedFormatError, create_encoder
    from app.engine.base import AudioChunk

    try:
        enc = create_encoder(fmt, 24000)
    except UnsupportedFormatError:
        pytest.skip(f"{fmt} unavailable in this build")

    enc.header()
    for _ in range(5):
        enc.encode(AudioChunk(samples=np.zeros(4800, dtype=np.float32), sample_rate=24000))
    assert len(enc.finalize()) > 0, f"{fmt} flushes nothing at close; test is not proving anything"


@pytest.mark.parametrize("fmt", ["mp3", "flac"])
def test_streamed_format_decodes_to_the_reported_duration(client, fmt, tmp_path):
    """What the client concatenates off the wire must decode to what we promised."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")

    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": fmt}}))
        ws.send_bytes(
            _pack({"event": "text", "text": "A complete stream, all the way to the end."})
        )
        ws.send_bytes(_pack({"event": "stop"}))
        audio, frames = _collect(ws)

    started = next(f for f in frames if f["event"] == "started")
    if started["format"] != fmt:
        pytest.skip(f"server degraded {fmt} to {started['format']}")

    reported_ms = max(f["time"] for f in frames if f["event"] == "audio")
    path = tmp_path / f"out.{fmt}"
    path.write_bytes(audio)
    # ffprobe reports N/A for a stream whose header carries no length, so
    # decode it to raw PCM and count the samples that actually come out.
    rate = started["sample_rate"]
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path),
         "-f", "s16le", "-ac", "1", "-ar", str(rate), "-"],
        capture_output=True, timeout=120,
    )
    assert decoded.returncode == 0, decoded.stderr.decode()[:400]
    decoded_ms = (len(decoded.stdout) / 2) / rate * 1000.0
    # Truncation is the bug being guarded against, so the floor is exact.
    # The ceiling is loose on purpose: a streamed mp3 carries no Xing header,
    # so the decoder cannot trim the codec's own priming padding (~90ms).
    assert decoded_ms >= reported_ms - 1.0, (
        f"{fmt}: decoded only {decoded_ms:.0f}ms off the wire but the server "
        f"reported {reported_ms:.0f}ms -- the stream is truncated"
    )
    assert decoded_ms <= reported_ms + 150.0, (
        f"{fmt}: decoded {decoded_ms:.0f}ms against a reported {reported_ms:.0f}ms; "
        "that is more than codec padding explains"
    )


def test_start_may_be_pipelined_behind_stop(client):
    """A client that sends `start` without waiting for `finish` is not an error."""
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        ws.send_bytes(_pack({"event": "text", "text": "First turn."}))
        ws.send_bytes(_pack({"event": "stop"}))
        # No waiting: the second turn goes out immediately behind the first.
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        ws.send_bytes(_pack({"event": "text", "text": "Second turn."}))
        ws.send_bytes(_pack({"event": "stop"}))

        finishes = []
        starteds = []
        for _ in range(500):
            frame = _unpack(ws.receive_bytes())
            if frame["event"] == "started":
                starteds.append(frame)
            if frame["event"] == "finish":
                finishes.append(frame)
                if len(finishes) == 2:
                    break

    assert [f["reason"] for f in finishes] == ["stop", "stop"]
    assert len(starteds) == 2, "the pipelined second turn never started"
    assert starteds[0]["session_id"] == starteds[1]["session_id"]


def test_turn_counters_reset_between_turns(client):
    """`time` restarts at zero each turn rather than accumulating forever."""
    with client.websocket_connect("/v1/tts/live") as ws:
        times_per_turn = []
        for _ in range(2):
            ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
            ws.send_bytes(_pack({"event": "text", "text": "Short line."}))
            ws.send_bytes(_pack({"event": "stop"}))
            _, frames = _collect(ws)
            times_per_turn.append([f["time"] for f in frames if f["event"] == "audio"])

    assert times_per_turn[0] and times_per_turn[1]
    assert times_per_turn[1][0] < times_per_turn[0][-1], (
        "second turn's clock continued from the first instead of resetting"
    )


def test_cancel_after_stop_is_not_a_protocol_error(client):
    """Barge-in while the server is still speaking a finished sentence.

    The client has sent `stop` and audio is still arriving; interrupting then
    is the most common barge-in of all, and used to close the socket with 1002.
    """
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        ws.send_bytes(_pack({"event": "text", "text": "One. Two. Three. Four. Five."}))
        ws.send_bytes(_pack({"event": "stop"}))
        ws.send_bytes(_pack({"event": "cancel"}))

        reasons = []
        for _ in range(500):
            frame = _unpack(ws.receive_bytes())
            if frame["event"] == "finish":
                reasons.append(frame["reason"])
                if "cancel" in reasons or "stop" in reasons:
                    break
        assert "error" not in reasons, f"cancel after stop was rejected: {reasons}"
        assert reasons, "no finish frame arrived"

        # And the session must still be usable for a fresh turn afterwards.
        ws.send_bytes(_pack({"event": "start", "request": {"format": "pcm"}}))
        ws.send_bytes(_pack({"event": "text", "text": "Still alive."}))
        ws.send_bytes(_pack({"event": "stop"}))
        _, frames = _collect(ws)
        assert frames[-1]["event"] == "finish"
        assert frames[-1]["reason"] in ("stop", "cancel")


def test_cancel_while_idle_is_ignored(client):
    """A redundant barge-in is not a protocol violation."""
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "cancel"}))
        ws.send_bytes(_pack({"event": "ping"}))
        frame = _unpack(ws.receive_bytes())
        assert frame["event"] == "pong", f"idle cancel broke the session: {frame}"


# --------------------------------------------------------------------------
# Barge-in must leave a stream the client can actually decode
# --------------------------------------------------------------------------


def test_cancel_starts_a_fresh_stream_for_whatever_is_said_next(client):
    """A barge-in ends the audio the client was playing, and the client throws
    it away. Carrying the same encoder and clock into the next thing said hands
    it a container whose header it has already discarded, and timestamps that
    still count the interrupted speech - both sound like silence or static
    right after an interruption."""
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "wav"}}))
        while _unpack(ws.receive_bytes())["event"] != "started":
            pass

        ws.send_bytes(_pack({"event": "text", "text": "A first sentence to interrupt. "}))
        ws.send_bytes(_pack({"event": "cancel"}))

        reasons = []
        for _ in range(40):
            frame = _unpack(ws.receive_bytes())
            if frame["event"] == "finish":
                reasons.append(frame["reason"])
                break
        assert "cancel" in reasons

        ws.send_bytes(_pack({"event": "text", "text": "And now something else entirely. "}))
        ws.send_bytes(_pack({"event": "stop"}))

        times, saw_audio = [], False
        for _ in range(200):
            frame = _unpack(ws.receive_bytes())
            if frame["event"] == "audio":
                saw_audio = True
                times.append(frame["time"])
            elif frame["event"] == "finish":
                break

        assert saw_audio, "no audio after a barge-in"
        # The clock restarted rather than continuing from the cancelled speech.
        assert times[0] < 5000.0, f"post-cancel timestamps kept counting: {times[:3]}"


def test_cancel_keeps_the_session_usable(client):
    with client.websocket_connect("/v1/tts/live") as ws:
        ws.send_bytes(_pack({"event": "start", "request": {"format": "wav"}}))
        while _unpack(ws.receive_bytes())["event"] != "started":
            pass
        ws.send_bytes(_pack({"event": "cancel"}))
        ws.send_bytes(_pack({"event": "ping"}))
        for _ in range(20):
            frame = _unpack(ws.receive_bytes())
            if frame["event"] == "pong":
                return
        raise AssertionError("session stopped responding after a cancel")
