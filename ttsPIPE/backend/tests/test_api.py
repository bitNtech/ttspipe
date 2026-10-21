"""REST API (§9) and regressions."""
import json
import time

from tests.test_session import pump


def test_health_and_modules_without_ml_deps(client):
    h = client.get("/api/health").json()
    assert h["status"] in ("ok", "degraded") and h["contract"] == "1.1"
    mods = client.get("/api/modules").json()
    ids = {m["id"] for m in mods}
    for builtin in ("ingest/plain", "normalize/rules", "segment/script", "chunker/punct", "g2p/lexicon",
                    "prosody/rules", "tts/tone", "tts/clips", "post/passthrough", "post/master"):
        assert builtin in ids
    for m in mods:
        for key in ("available", "reason", "params_schema", "default_params", "license", "streaming", "languages"):
            assert key in m, key


def test_pipeline_view_shape(client):
    p = client.get("/api/pipeline").json()
    assert set(p["stages"]) == {"ingest", "normalize", "segment", "chunker", "g2p", "prosody", "tts", "post"}
    st = p["stages"]["tts"]
    for key in ("impl", "configured_impl", "status", "error", "fallback", "busy", "params", "p95_ms"):
        assert key in st
    assert st["params"]["syllable_ms"] == 60  # effective params
    ids = [n["id"] for n in p["graph"]["nodes"]]
    assert ids[0] == "text_in" and ids[-1] == "audio_out" and "shadow_tts" not in ids
    assert "dirty" in p and "active_profile" in p


def test_put_stage_validation(client):
    assert client.put("/api/pipeline/stages/bogus", json={}).status_code == 404
    assert client.put("/api/pipeline/stages/tts", json={"impl": "nope"}).status_code == 404
    r = client.put("/api/pipeline/stages/tts", json={"params": {"rtf": 99}})
    assert r.status_code == 422 and "stages.tts.params.rtf" in r.json()["detail"]
    r = client.put("/api/pipeline/stages/tts", json={"enabled": False})
    assert r.status_code == 422 and "cannot be disabled" in r.json()["detail"]
    r = client.put("/api/pipeline/stages/normalize", json={"enabled": False})
    assert r.status_code == 200 and r.json()["status"] == "disabled"


def test_disabled_normalize_is_identity(client):
    client.put("/api/pipeline/stages/normalize", json={"enabled": False})
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "Pay ₹500 now."}))
        evs, _ = pump(ws, "utterance.end")
    assert evs[-1]["data"]["normalized"] == "Pay ₹500 now."


def test_playback_chunker_voice_put(client):
    r = client.put("/api/pipeline/playback", json={"jitter_ms": 80, "out_rate": 16000})
    assert r.status_code == 200 and r.json()["jitter_ms"] == 80
    assert client.get("/api/pipeline").json()["config"]["audio"]["out_rate"] == 16000
    assert client.put("/api/pipeline/playback", json={"out_rate": 12345}).status_code == 422
    assert client.put("/api/pipeline/playback", json={"bogus": 1}).status_code == 422
    assert client.put("/api/pipeline/chunker", json={"first_chunk_min_words": 0}).status_code == 422
    r = client.put("/api/pipeline/chunker", json={"first_chunk_min_words": 3})
    assert r.json()["first_chunk_min_words"] == 3
    r = client.put("/api/pipeline/voice", json={"voice_id": "tone-m", "speed": 1.2})
    assert r.json()["voice_id"] == "tone-m"
    assert client.put("/api/pipeline/voice", json={"speed": 9}).status_code == 422
    # persisted override
    ov = (client.manager.s.data_dir / "pipeline.override.yaml").read_text()
    assert "jitter_ms: 80" in ov and "tone-m" in ov


def test_new_session_uses_new_out_rate(client):
    client.put("/api/pipeline/playback", json={"out_rate": 16000})
    with client.websocket_connect("/ws/tts") as ws:
        evs, _ = pump(ws, "session.started")
        assert evs[-1]["data"]["sample_rate"] == 16000
        ws.send_text(json.dumps({"type": "speak", "text": "Rate check."}))
        while True:
            m = ws.receive()
            if m.get("bytes"):
                assert len(m["bytes"]) == 1 + 320 * 2  # one 20 ms frame at 16 kHz
                break


def test_validate_and_reset(client):
    r = client.post("/api/pipeline/validate", json={"stages": {"tts": {"params": {"rtf": -1}}}}).json()
    assert not r["ok"] and r["errors"]
    r = client.post("/api/pipeline/validate", json={"playback": {"jitter_ms": 100}}).json()
    assert r["ok"] and "estimate" in r and "ram_mb" in r["estimate"]
    client.put("/api/pipeline/playback", json={"jitter_ms": 90})
    p = client.post("/api/pipeline/reset-config").json()
    assert p["config"]["playback"]["jitter_ms"] == 60
    assert not (client.manager.s.data_dir / "pipeline.override.yaml").exists()


def test_put_whole_pipeline_atomic(client):
    cfg = client.get("/api/pipeline").json()["config"]
    cfg["playback"]["jitter_ms"] = 70
    cfg["stages"]["tts"]["params"] = {"rtf": 50}
    assert client.put("/api/pipeline", json=cfg).status_code == 422
    assert client.get("/api/pipeline").json()["config"]["playback"]["jitter_ms"] == 60  # nothing applied
    cfg["stages"]["tts"]["params"] = {}
    assert client.put("/api/pipeline", json=cfg).status_code == 200
    assert client.get("/api/pipeline").json()["config"]["playback"]["jitter_ms"] == 70


def test_profiles_crud_and_apply(client):
    names = {p["name"]: p for p in client.get("/api/profiles").json()}
    for b in ("lowest_latency", "balanced", "high_quality", "minimal", "ab_tts_compare", "english_fast"):
        assert names[b]["builtin"]
    assert client.put("/api/profiles/balanced", json={"config": {}}).status_code == 409
    assert client.delete("/api/profiles/minimal").status_code == 409
    r = client.put("/api/profiles/mine", json={"description": "x", "config": {"playback": {"jitter_ms": 100}}})
    assert r.status_code == 200
    assert client.put("/api/profiles/bad", json={"config": {"playback": {"jitter_ms": -1}}}).status_code == 422
    assert client.put("/api/profiles/bad name!", json={}).status_code in (404, 422)
    p = client.post("/api/profiles/mine/apply").json()
    assert p["active_profile"] == "mine" and p["config"]["playback"]["jitter_ms"] == 100 and not p["dirty"]
    client.put("/api/pipeline/playback", json={"jitter_ms": 110})
    assert client.get("/api/pipeline").json()["dirty"]
    r = client.put("/api/profiles/snap", json={"from_current": True}).json()
    assert r["config"]["playback"]["jitter_ms"] == 110
    p = client.post("/api/profiles/ab_tts_compare/apply").json()
    assert "shadow_tts" in [n["id"] for n in p["graph"]["nodes"]]
    assert p["shadow"]["status"] in ("ready", "loading")
    assert client.delete("/api/profiles/mine").status_code == 200
    assert client.get("/api/profiles/mine").status_code == 404


def test_shadow_engine_emits_shadow_events(client):
    client.post("/api/profiles/ab_tts_compare/apply")
    client.manager.wait_ready(10)
    with client.websocket_connect("/ws/tts") as ws:
        pump(ws, "session.started")
        ws.send_text(json.dumps({"type": "speak", "text": "Shadow test sentence."}))
        evs, _ = pump(ws, "utterance.end")
        if "shadow" not in [e["type"] for e in evs]:
            evs, _ = pump(ws, "shadow")
    sh = [e["data"] for e in evs if e["type"] == "shadow"]
    assert sh and sh[0]["impl"] == "tone" and sh[0]["synth_ms"] is not None


def test_layouts_crud(client):
    assert client.put("/api/layouts/a", json={"cols": 12}).status_code == 422
    lay = {"cols": 12, "row_h": 28, "panels": [{"id": "graph", "x": 0, "y": 0, "w": 6, "h": 8}]}
    assert client.put("/api/layouts/a", json=lay).json()["name"] == "a"
    assert [l["name"] for l in client.get("/api/layouts").json()] == ["a"]
    assert client.get("/api/layouts/a").json()["panels"][0]["id"] == "graph"
    assert client.delete("/api/layouts/a").status_code == 200
    assert client.get("/api/layouts/a").status_code == 404


def test_sessions_renders_and_actions(client):
    with client.websocket_connect("/ws/tts?session_id=abc&label=L") as ws:
        pump(ws, "session.started")
        r = client.post("/api/sessions/abc/speak", json={"text": "Render me please."})
        assert r.status_code == 200
        pump(ws, "utterance.end")
        s = client.get("/api/sessions").json()
        assert s[0]["id"] == "abc" and s[0]["label"] == "L"
        d = client.get("/api/sessions/abc").json()
        u = d["utterances"][-1]
        assert u["status"] == "final" and u["chunks"] and u["latency"]["ttfa_ms"]
        for a in ("duck", "resume", "stop"):
            assert client.post(f"/api/sessions/abc/{a}").json()["ok"]
        assert client.post("/api/sessions/abc/fly").status_code == 404
    assert client.post("/api/sessions/zzz/stop").status_code == 404
    for _ in range(50):
        rs = client.get("/api/renders?session_id=abc").json()
        if rs:
            break
        time.sleep(0.05)
    assert rs and rs[0]["url"].endswith(".wav")
    wav = client.get(rs[0]["url"])
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    assert client.delete(f"/api/renders/{rs[0]['id']}").status_code == 200


def test_reconnect_same_session_id_closes_old(client):
    with client.websocket_connect("/ws/tts?session_id=dup") as ws1:
        pump(ws1, "session.started")
        with client.websocket_connect("/ws/tts?session_id=dup") as ws2:
            pump(ws2, "session.started")
            m = ws1.receive()
            assert m.get("code") == 4409 or m["type"] == "websocket.close"


def test_runtime_pause_resume(client):
    assert client.post("/api/runtime/pause").json()["paused"] is True
    assert client.get("/api/state").json()["paused"] is True
    assert client.post("/api/runtime/resume").json()["paused"] is False
    assert client.post("/api/runtime/explode").status_code == 404


def test_metrics_logs_scripts_voices_models(client):
    m = client.get("/api/metrics").json()
    for k in ("stages", "ttfa", "latency", "throughput", "cache", "system", "underruns", "events_dropped"):
        assert k in m
    prom = client.get("/metrics").text
    assert "ttspipe_uptime_seconds" in prom
    assert isinstance(client.get("/api/logs?level=INFO&limit=5").json(), list)
    scripts = client.get("/api/scripts").json()
    langs = {s["language"] for s in scripts}
    assert {"ta", "tanglish", "en", "mixed"} <= langs and any(s["id"] == "long" for s in scripts)
    voices = client.get("/api/voices").json()
    assert any(v["impl"] == "tone" and v["id"] == "tone-f" for v in voices)
    assert isinstance(client.get("/api/models").json(), list)
    assert client.post("/api/models/tts/nope/download").status_code == 404


def test_compare_and_ratings(client):
    body = {"text": "Compare these two.", "a": {"label": "fast", "config": {"stages": {"tts": {"params": {"syllable_ms": 60}}}}},
            "b": {"profile": "minimal"}}
    r = client.post("/api/compare", json=body).json()
    for side in ("a", "b"):
        assert r[side]["error"] is None, r[side]
        assert r[side]["metrics"]["audio_ms"] > 0 and r[side]["url"].endswith(".wav")
        assert client.get(r[side]["url"]).status_code == 200
    assert r["a"]["metrics"]["audio_ms"] < r["b"]["metrics"]["audio_ms"]
    assert client.post("/api/compare", json={"text": ""}).status_code == 422
    assert client.post("/api/compare/ratings", json={"score": 7}).status_code == 422
    for s in (4, 5, 3):
        client.post("/api/compare/ratings", json={"render_id": r["a"]["render_id"], "tts_impl": "tone",
                                                  "voice_id": "tone-f", "score": s, "text": "x"})
    summ = client.get("/api/compare/ratings").json()["summary"]
    assert summ[0]["n"] == 3 and summ[0]["mos"] == 4.0


def test_module_load_unload(client):
    r = client.post("/api/modules/post/passthrough/load").json()
    assert r["impl"] == "passthrough"
    r = client.post("/api/modules/post/passthrough/unload").json()
    assert r["impl"] == "passthrough"  # builtin of its kind stays
    assert client.post("/api/modules/post/nope/load").status_code == 404
