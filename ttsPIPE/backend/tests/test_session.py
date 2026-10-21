"""Session runtime over /ws/tts: synthesis, playback state machine, duplex, heard_text, hot swap, fallback."""
import json
import time

import numpy as np
import pytest

from app.core import registry
from app.core.base import ModuleInfo, TTSModule, schema

LONG = ("One two three four five, six seven eight nine ten. Eleven twelve thirteen fourteen fifteen sixteen. "
        "Seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three.")


def pump(ws, until=None, timeout=15.0, stop_after=None):
    """Read messages until an event of type `until` (or predicate) arrives. Returns (events, frames)."""
    evs, frames = [], 0
    t_end = time.time() + timeout
    while time.time() < t_end:
        m = ws.receive()
        if m.get("bytes") is not None:
            assert m["bytes"][0] == 1
            frames += 1
            continue
        if m.get("text") is None:
            continue
        ev = json.loads(m["text"])
        evs.append(ev)
        if until and (ev["type"] == until if isinstance(until, str) else until(ev)):
            return evs, frames
    raise AssertionError(f"timeout waiting for {until}; got {[e['type'] for e in evs][-20:]}")


def types(evs):
    return [e["type"] for e in evs]


def test_speak_end_to_end(client):
    with client.websocket_connect("/ws/tts?label=t") as ws:
        evs, _ = pump(ws, "session.started")
        assert evs[0]["data"]["sample_rate"] == 24000 and evs[0]["data"]["format"] == "pcm16le"
        ws.send_text(json.dumps({"type": "speak", "text": "Hello there my friend, ₹500 is due today."}))
        evs, frames = pump(ws, "utterance.end")
        t = types(evs)
        for needed in ("utterance.started", "text.normalized", "segment", "chunk.flushed", "synth.start",
                       "synth.first_audio", "synth.done", "playback.state", "audio.level", "playback.progress"):
            assert needed in t, needed
        states = [e["data"]["state"] for e in evs if e["type"] == "playback.state"]
        assert states == ["buffering", "synthesizing", "speaking", "done", "idle"]
        end = evs[-1]["data"]
        assert end["latency"]["ttfa_ms"] > 0 and end["latency"]["text_to_first_chunk_ms"] is not None
        assert end["normalized"].count("five hundred rupees") == 1
        assert all(e["t_out_ms"] is not None and e["session_id"] for e in evs)
        assert frames > 10
        audio_ms = end["latency"]["audio_ms"]
        assert frames * 20 >= audio_ms * 0.8  # post trims silence, but most audio is sent


def test_llm_deltas_first_chunk_small(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        text = "Sure, I can help you with that today. Your ticket number is ready and the team will call soon."
        for i in range(0, len(text), 4):
            ws.send_text(json.dumps({"type": "text", "delta": text[i:i + 4], "utterance_id": "r1"}))
        ws.send_text(json.dumps({"type": "end_utterance", "utterance_id": "r1"}))
        evs, _ = pump(ws, "utterance.end")
        chunks = [e["data"] for e in evs if e["type"] == "chunk.flushed"]
        assert chunks[0]["reason"] == "first" and chunks[0]["words"] <= 10
        assert len(chunks) >= 2
        assert evs[-1]["data"]["utterance_id"] == "r1"


def test_stop_mid_utterance_heard_text(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": LONG, "utterance_id": "x"}))
        evs, _ = pump(ws, lambda e: e["type"] == "playback.progress" and e["data"]["word_index"] >= 3)
        last_progress = evs[-1]["data"]["heard_text"]
        ws.send_text(json.dumps({"type": "stop"}))
        evs, _ = pump(ws, "utterance.end")
        intr = [e["data"] for e in evs if e["type"] == "utterance.interrupted"][0]
        heard, unplayed = intr["heard_text"], intr["unplayed_text"]
        words = LONG.split()
        assert heard and heard.split() == words[:len(heard.split())]  # a prefix, whole words
        assert len(heard.split()) < len(words)
        assert (heard + " " + unplayed).split() == words
        assert heard.startswith(last_progress)
        states = [e["data"]["state"] for e in evs if e["type"] == "playback.state"]
        assert states[-2:] == ["interrupted", "idle"]
        assert evs[-1]["data"]["interrupted"] is True


def test_deltas_after_stop_are_dropped_until_end(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "text", "delta": LONG, "utterance_id": "a"}))
        pump(ws, "playback.chunk")
        ws.send_text(json.dumps({"type": "stop"}))
        pump(ws, "utterance.end")
        ws.send_text(json.dumps({"type": "text", "delta": "more words that should vanish now ok. ", "utterance_id": "a"}))
        ws.send_text(json.dumps({"type": "end_utterance", "utterance_id": "a"}))
        ws.send_text(json.dumps({"type": "speak", "text": "Fresh reply here.", "utterance_id": "b"}))
        evs, _ = pump(ws, "utterance.end")
        flushed = [e["data"] for e in evs if e["type"] == "chunk.flushed"]
        assert all(c["utterance_id"] == "b" for c in flushed)
        assert evs[-1]["data"]["utterance_id"] == "b"


def test_deltas_without_ids_after_stop_are_dropped(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "text", "delta": LONG}))
        pump(ws, "playback.chunk")
        ws.send_text(json.dumps({"type": "stop"}))
        pump(ws, "utterance.end")
        ws.send_text(json.dumps({"type": "text", "delta": "trailing tokens of the old reply. "}))
        ws.send_text(json.dumps({"type": "end_utterance"}))
        ws.send_text(json.dumps({"type": "text", "delta": "Brand new reply now."}))
        ws.send_text(json.dumps({"type": "end_utterance"}))
        evs, _ = pump(ws, "utterance.end")
        flushed = " ".join(e["data"]["text"] for e in evs if e["type"] == "chunk.flushed")
        assert "trailing" not in flushed and "Brand new reply" in flushed


def test_stop_before_audio_gives_empty_heard(client):
    client.manager.patch(["stages", "tts", "params"], {"latency_ms": 800})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "Nothing will be heard of this sentence at all."}))
        pump(ws, "synth.start")
        ws.send_text(json.dumps({"type": "stop"}))
        evs, _ = pump(ws, "utterance.end")
        intr = [e["data"] for e in evs if e["type"] == "utterance.interrupted"][0]
        assert intr["heard_text"] == "" and intr["unplayed_text"].startswith("Nothing")


def test_duck_and_resume(client):
    client.manager.patch(["playback"], {"duck_release_ms": 0})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": LONG}))
        pump(ws, lambda e: e["type"] == "playback.state" and e["data"]["state"] == "speaking")
        ws.send_text(json.dumps({"type": "duck"}))
        evs, _ = pump(ws, lambda e: e["type"] == "audio.level" and e["data"]["gain_db"] <= -11.9)
        assert any(e["type"] == "duck" and e["data"]["action"] == "duck" for e in evs)
        assert any(e["type"] == "playback.state" and e["data"]["state"] == "ducked" for e in evs)
        ws.send_text(json.dumps({"type": "resume"}))
        evs, _ = pump(ws, lambda e: e["type"] == "playback.state" and e["data"]["state"] == "speaking")
        ws.send_text(json.dumps({"type": "stop"}))
        pump(ws, "utterance.end")


def test_duck_auto_release(client):
    client.manager.patch(["playback"], {"duck_release_ms": 200})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": LONG}))
        pump(ws, lambda e: e["type"] == "playback.state" and e["data"]["state"] == "speaking")
        ws.send_text(json.dumps({"type": "duck"}))
        evs, _ = pump(ws, lambda e: e["type"] == "duck" and e["data"]["action"] == "release")
        assert evs[-1]["data"]["source"] == "timeout"
        ws.send_text(json.dumps({"type": "stop"}))
        pump(ws, "utterance.end")


def test_lookahead_overlaps_synthesis_with_playback(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": LONG}))
        evs, _ = pump(ws, "utterance.end", timeout=30)
        idx = {(e["type"], (e["data"].get("chunk_id"), e["data"].get("event"))): i for i, e in enumerate(evs)}
        # chunk 1 synthesis starts before chunk 0 finishes playing
        assert idx[("synth.start", ("u1.1", None))] < idx[("playback.chunk", ("u1.0", "end"))]


def test_lookahead_zero_serializes(client):
    client.manager.patch(["playback"], {"lookahead_chunks": 0})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": LONG}))
        evs, _ = pump(ws, "utterance.end", timeout=30)
        idx = {(e["type"], (e["data"].get("chunk_id"), e["data"].get("event"))): i for i, e in enumerate(evs)}
        assert idx[("synth.start", ("u1.1", None))] > idx[("playback.chunk", ("u1.0", "end"))]


def test_underrun_when_engine_slower_than_realtime(client):
    client.manager.patch(["stages", "tts", "params"], {"rtf": 1.6, "word_pieces": True})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "one two three four five six seven eight"}))
        evs, _ = pump(ws, "utterance.end", timeout=30)
        assert "underrun" in types(evs)


def test_phrase_cache_hit(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        for _ in range(2):
            ws.send_text(json.dumps({"type": "speak", "text": "Thank you for calling."}))
            evs, _ = pump(ws, "utterance.end")
        starts = [e["data"]["cached"] for e in evs if e["type"] == "synth.start"]
        assert starts == [True]
        assert client.get("/api/metrics").json()["cache"]["hit_rate"] > 0


def test_hot_swap_params_mid_session(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "First reply here."}))
        _, f1 = pump(ws, "utterance.end")
        r = client.put("/api/pipeline/stages/tts", json={"params": {"syllable_ms": 120}})
        assert r.status_code == 200 and r.json()["params"]["syllable_ms"] == 120
        ws.send_text(json.dumps({"type": "speak", "text": "Second reply here."}))
        _, f2 = pump(ws, "utterance.end")
        assert f2 > f1 * 1.4  # slower syllables → longer audio from the swapped params


@pytest.fixture
def fake_modules():
    class Missing(TTSModule):
        info = ModuleInfo(id="tts/fake_missing", kind="tts", name="needs a missing package",
                          requires=["definitely_not_installed_pkg"], native_rate=16000)

    class Broken(TTSModule):
        info = ModuleInfo(id="tts/fake_broken", kind="tts", name="always raises", native_rate=16000,
                          params_schema=schema(x=("integer", 1)))

        def synthesize(self, state, chunk, plan, voice):
            raise RuntimeError("boom")
            yield

    registry.register(Missing)
    registry.register(Broken)
    yield
    registry._REGISTRY.pop("tts/fake_missing", None)
    registry._REGISTRY.pop("tts/fake_broken", None)


def test_unavailable_module_falls_back_to_builtin(client, fake_modules):
    r = client.put("/api/pipeline/stages/tts", json={"impl": "fake_missing"})
    assert r.status_code == 200
    client.manager.wait_ready(10)
    st = client.get("/api/pipeline").json()["stages"]["tts"]
    assert st["status"] == "ready" and st["fallback"] and st["impl"] == "tone" and st["error"]
    assert "definitely_not_installed_pkg" in st["error"]
    h = client.get("/api/health").json()
    assert h["status"] == "degraded" and h["reasons"]
    v = client.post("/api/pipeline/validate", json={}).json()
    assert v["ok"] and any("unavailable" in w for w in v["warnings"])


def test_three_errors_mark_module_error_and_fall_back(client, fake_modules):
    client.put("/api/pipeline/stages/tts", json={"impl": "fake_broken"})
    client.manager.wait_ready(10)
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "One two three four. Five. Six. Seven. Eight nine ten."}))
        evs, frames = pump(ws, "utterance.end", timeout=30)
    errs = [e for e in evs if e["type"] == "error"]
    assert len(errs) >= 3
    st = client.get("/api/pipeline").json()["stages"]["tts"]
    assert st["status"] == "error" and st["fallback"] and st["impl"] == "tone"
    assert frames > 0  # the builtin took over for later chunks
    r = client.post("/api/modules/tts/fake_broken/load")
    assert r.status_code == 200


def test_monitor_stream_sees_session_and_global_events(client):
    with client.websocket_connect("/ws/events?session_id=*") as mon:
        with client.websocket_connect("/ws/tts?session_id=mon1") as ws:
            ws.send_text(json.dumps({"type": "speak", "text": "Monitor me."}))
            pump(ws, "utterance.end")
        evs, _ = pump(mon, lambda e: e["type"] == "metrics", timeout=5)
        t = types(evs)
        assert "utterance.end" in t and "metrics" in t
        assert all(e["session_id"] in ("mon1", None) for e in evs)


def test_ping_pong_and_bad_messages(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text("not json")
        pump(ws, "error")
        ws.send_text(json.dumps({"type": "nope"}))
        pump(ws, "error")
        ws.send_text(json.dumps({"type": "ping", "t": 123}))
        evs, _ = pump(ws, "pong")
        assert evs[-1]["data"]["t"] == 123


def test_markdown_reply_and_ssml_break(client):
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": '**Done!** See https://x.y <break time="400ms"/> Bye now.'}))
        evs, _ = pump(ws, "utterance.end")
        texts = " ".join(e["data"]["text"] for e in evs if e["type"] == "chunk.flushed")
        assert "**" not in texts and "https" not in texts
        pauses = [e["data"]["pause_after_ms"] for e in evs if e["type"] == "chunk.flushed"]
        assert 400 in pauses


def test_offline_render_matches_realtime_content(client):
    import asyncio
    m = client.manager
    r = client.post("/api/voices/tone/preview", json={"voice_id": "tone-m", "text": "Preview this."})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and len(r.content) > 1000
    from app.core.audio import read_wav
    x, rate = read_wav(r.content)
    assert rate == 24000 and np.abs(x).max() > 0.01
