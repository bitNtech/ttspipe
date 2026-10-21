"""REST API (ARCHITECTURE.md §9)."""
from __future__ import annotations

import base64
import copy
import json
import logging
import threading
import time
from pathlib import Path

import yaml
from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.core import config as C
from app.core import clones, fish_server, logs, registry
from app.core.audio import wav_bytes
from app.core.base import KINDS, Voice
from app.core.manager import Manager
from app.settings import REPO

log = logging.getLogger("ttspipe.api")
router = APIRouter(prefix="/api")


def M(req: Request) -> Manager:
    return req.app.state.manager


def _apply(m: Manager, new: dict) -> None:
    errors, _ = m.set_config(new)
    if errors:
        raise HTTPException(422, "; ".join(errors))


def _session(m: Manager, sid: str):
    s = m.sessions.get(sid)
    if s is None:
        raise HTTPException(404, f"no session {sid}")
    return s


# ---- health / modules / pipeline ---------------------------------------------------------------


@router.get("/health")
def health(req: Request):
    return M(req).health()


@router.get("/modules")
def modules(req: Request):
    return M(req).modules()


@router.get("/pipeline")
def pipeline(req: Request):
    return M(req).pipeline_view()


@router.put("/pipeline/stages/{kind}")
def put_stage(kind: str, req: Request, body: dict = Body(...)):
    m = M(req)
    if kind not in KINDS:
        raise HTTPException(404, f"unknown stage kind {kind}")
    new = copy.deepcopy(m.config)
    st = new["stages"][kind]
    if "impl" in body:
        try:
            registry.get(f"{kind}/{body['impl']}")
        except KeyError:
            raise HTTPException(404, f"unknown module {kind}/{body['impl']}")
        if body["impl"] != st["impl"]:
            st["params"] = {}
        st["impl"] = body["impl"]
    if "params" in body:
        if not isinstance(body["params"], dict):
            raise HTTPException(422, f"stages.{kind}.params: expected object")
        st["params"] = {**(st.get("params") or {}), **body["params"]} if "impl" not in body else dict(body["params"])
    if "enabled" in body:
        st["enabled"] = bool(body["enabled"])
    if kind == "tts" and "impl" in body:
        new.setdefault("voice", {})["impl"] = body["impl"]
    _apply(m, new)
    return m.slots[kind].status_dict()


def _put_section(m: Manager, section: str, body: dict, keys: dict) -> dict:
    unknown = [k for k in body if k not in keys]
    if unknown:
        raise HTTPException(422, f"{section}.{unknown[0]}: unknown key")
    new = copy.deepcopy(m.config)
    new.setdefault(section, {}).update(body)
    _apply(m, new)
    return m.config[section]


@router.put("/pipeline/playback")
def put_playback(req: Request, body: dict = Body(...)):
    m = M(req)
    body = dict(body)
    out_rate = body.pop("out_rate", None)
    if out_rate is not None:
        new = copy.deepcopy(m.config)
        new["audio"]["out_rate"] = out_rate
        _apply(m, new)
    return _put_section(m, "playback", body, C.PLAYBACK_KEYS) if body else m.config["playback"]


@router.put("/pipeline/chunker")
def put_chunker(req: Request, body: dict = Body(...)):
    return _put_section(M(req), "chunker", body, C.CHUNKER_KEYS)


@router.put("/pipeline/voice")
def put_voice(req: Request, body: dict = Body(...)):
    m = M(req)
    out = _put_section(m, "voice", body, C.VOICE_KEYS)
    for s in m.sessions.values():
        if s.record:
            s.set_voice(f"{out.get('impl')}:{out.get('voice_id')}" if out.get("voice_id") else None,
                        out.get("speed"), out.get("pitch"))
    return out


@router.put("/pipeline/shadow")
def put_shadow(req: Request, body: dict = Body(...)):
    m = M(req)
    new = copy.deepcopy(m.config)
    shadows = new.setdefault("graph", {}).setdefault("shadows", {})
    if not body.get("impl"):
        shadows["tts"] = None
    else:
        shadows["tts"] = {"impl": body["impl"], "params": body.get("params") or {}, "voice_id": body.get("voice_id")}
    _apply(m, new)
    return m.pipeline_view()


@router.put("/pipeline/positions")
def put_positions(req: Request, body: dict = Body(...)):
    m = M(req)
    new = copy.deepcopy(m.config)
    new.setdefault("graph", {})["positions"] = body
    _apply(m, new)
    return new["graph"]["positions"]


@router.put("/pipeline")
def put_pipeline(req: Request, body: dict = Body(...)):
    m = M(req)
    new = C.deep_merge(m.defaults, body)
    _apply(m, new)
    return m.pipeline_view()


@router.post("/pipeline/validate")
def validate(req: Request, body: dict = Body(default={})):
    return M(req).validate(body)


@router.post("/pipeline/reset-config")
def reset_config(req: Request):
    m = M(req)
    m.reset_config()
    return m.pipeline_view()


# ---- modules / models / voices -----------------------------------------------------------------


@router.post("/modules/{kind}/{impl}/load")
def load_module(kind: str, impl: str, req: Request):
    m = M(req)
    if kind not in KINDS:
        raise HTTPException(404, "unknown kind")
    try:
        registry.get(f"{kind}/{impl}")
    except KeyError:
        raise HTTPException(404, f"unknown module {kind}/{impl}")
    st = m.config["stages"][kind]
    if st["impl"] == impl:
        m.slots[kind].configured_impl = None  # force a reload (e.g. after an error)
        m.slots[kind].configure(impl, st.get("params"), st.get("enabled", True))
        return m.slots[kind].status_dict()
    return put_stage(kind, req, {"impl": impl})


@router.post("/modules/{kind}/{impl}/unload")
def unload_module(kind: str, impl: str, req: Request):
    m = M(req)
    slot = m.slots.get(kind)
    if slot is None:
        raise HTTPException(404, "unknown kind")
    if slot.impl == impl:
        slot.unload_to_builtin(f"{impl} unloaded by request")
        if registry.BUILTIN_FALLBACK[kind] != impl:
            slot.status = "unloaded"
            slot._emit_status()
    return slot.status_dict()


@router.get("/models")
def models(req: Request):
    return M(req).models()


@router.post("/models/{kind}/{impl}/download", status_code=202)
def download(kind: str, impl: str, req: Request):
    try:
        return M(req).download(kind, impl)
    except KeyError:
        raise HTTPException(404, f"unknown module {kind}/{impl}")


@router.get("/voices")
def voices(req: Request):
    out = []
    for cls in registry.all_modules():
        if cls.info.kind != "tts":
            continue
        for v in cls.list_voices():
            out.append({"impl": cls.info.impl, "id": v.id, "name": v.name, "language": v.language,
                        "gender": v.gender, "sample": v.sample})
    return out


@router.post("/voices/{impl}/preview")
async def preview(impl: str, req: Request, body: dict = Body(default={})):
    m = M(req)
    try:
        cls = registry.get(f"tts/{impl}")
    except KeyError:
        raise HTTPException(404, f"unknown tts module {impl}")
    text = body.get("text") or ("வணக்கம், இது ஒரு குரல் மாதிரி." if "ta" in cls.info.languages and "en" not in cls.info.languages
                                 else "Hello, this is a voice preview.")
    cfg = copy.deepcopy(m.config)
    if cfg["stages"]["tts"]["impl"] != impl or body.get("params") is not None:
        cfg["stages"]["tts"] = {"enabled": True, "impl": impl, "params": body.get("params") or {}}
    voice = Voice(impl=impl, voice_id=body.get("voice_id") or "", speed=float(body.get("speed", 1.0)),
                  pitch=float(body.get("pitch", 0.0)))
    try:
        r = await m.render_offline(text, cfg, voice, label="preview")
    except (ValueError, TimeoutError) as e:
        raise HTTPException(422, str(e))
    return Response(wav_bytes(r["pcm"], r["rate"]), media_type="audio/wav")


# ---- state / sessions --------------------------------------------------------------------------


@router.get("/state")
def state(req: Request):
    return M(req).state()


@router.get("/sessions")
def sessions(req: Request):
    return [s.summary() for s in M(req).sessions.values() if s.record]


@router.get("/sessions/{sid}")
def session(sid: str, req: Request):
    s = _session(M(req), sid)
    return {**s.summary(), "utterances": s.utterance_rows()}


@router.post("/sessions/{sid}/speak")
def session_speak(sid: str, req: Request, body: dict = Body(...)):
    s = _session(M(req), sid)
    if not isinstance(body.get("text"), str) or not body["text"].strip():
        raise HTTPException(422, "text: required")
    return {"ok": True, "utterance_id": s.speak(body["text"], body.get("utterance_id"))}


@router.post("/sessions/{sid}/{action}")
def session_action(sid: str, action: str, req: Request):
    if action not in ("duck", "stop", "resume"):
        raise HTTPException(404, f"unknown action {action}")
    getattr(_session(M(req), sid), action)("api")
    return {"ok": True}


# ---- renders -----------------------------------------------------------------------------------


@router.get("/renders")
def renders(req: Request, session_id: str | None = None):
    return M(req).renders.list(session_id)


@router.get("/renders/{rid}.wav")
def render_wav(rid: str, req: Request):
    p = M(req).renders.path(rid)
    if p is None:
        raise HTTPException(404, "no such render")
    return FileResponse(p, media_type="audio/wav", filename=f"{rid}.wav")


@router.delete("/renders/{rid}")
def render_delete(rid: str, req: Request):
    if not M(req).renders.delete(rid):
        raise HTTPException(404, "no such render")
    return {"ok": True}


# ---- runtime / metrics / logs / scripts --------------------------------------------------------


@router.post("/runtime/{action}")
def runtime(action: str, req: Request):
    m = M(req)
    if action not in ("pause", "resume"):
        raise HTTPException(404, "unknown action")
    m.paused = action == "pause"
    m.bus.emit("log", {"level": "INFO", "logger": "ttspipe.runtime", "msg": f"runtime {action}d", "session_id": None})
    return m.state()


@router.get("/metrics")
def metrics(req: Request):
    return M(req).metrics_snapshot()


@router.get("/logs")
def get_logs(level: str = "DEBUG", limit: int = 200, session_id: str | None = None):
    return logs.ring.query(level, min(max(limit, 1), 2000), session_id)


@router.get("/scripts")
def scripts():
    p = REPO / "config" / "scripts.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else []


# ---- profiles / layouts ------------------------------------------------------------------------


@router.get("/profiles")
def profiles(req: Request):
    return M(req).profiles.list()


@router.get("/profiles/{name}")
def profile(name: str, req: Request):
    p = M(req).profiles.get(name)
    if p is None:
        raise HTTPException(404, f"no profile {name}")
    return p


@router.put("/profiles/{name}")
def put_profile(name: str, req: Request, body: dict = Body(default={})):
    m = M(req)
    if body.get("from_current"):
        cfg = C.diff(m.defaults, m.config)
    else:
        cfg = body.get("config") or {}
    cfg.pop("active_profile", None)
    errors, _ = C.validate(C.deep_merge(m.defaults, cfg))
    if errors:
        raise HTTPException(422, "; ".join(errors))
    try:
        return m.profiles.save(name, body.get("description", ""), cfg)
    except PermissionError as e:
        raise HTTPException(409, str(e))
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.delete("/profiles/{name}")
def delete_profile(name: str, req: Request):
    try:
        if not M(req).profiles.delete(name):
            raise HTTPException(404, f"no profile {name}")
    except PermissionError as e:
        raise HTTPException(409, str(e))
    return {"ok": True}


@router.post("/profiles/{name}/apply")
def apply_profile(name: str, req: Request):
    m = M(req)
    try:
        errors, _ = m.apply_profile(name)
    except KeyError:
        raise HTTPException(404, f"no profile {name}")
    if errors:
        raise HTTPException(422, "; ".join(errors))
    return m.pipeline_view()


@router.get("/layouts")
def layouts(req: Request):
    return M(req).layouts.list()


@router.get("/layouts/{name}")
def layout(name: str, req: Request):
    try:
        lay = M(req).layouts.get(name)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if lay is None:
        raise HTTPException(404, f"no layout {name}")
    return lay


@router.put("/layouts/{name}")
def put_layout(name: str, req: Request, body: dict = Body(...)):
    if not isinstance(body.get("panels"), list):
        raise HTTPException(422, "panels: expected array")
    try:
        return M(req).layouts.save(name, body)
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.delete("/layouts/{name}")
def delete_layout(name: str, req: Request):
    try:
        if not M(req).layouts.delete(name):
            raise HTTPException(404, f"no layout {name}")
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}


# ---- compare / ratings -------------------------------------------------------------------------


def _variant_cfg(m: Manager, v: dict) -> dict:
    cfg = copy.deepcopy(m.config)
    if v.get("profile"):
        prof = m.profiles.get(v["profile"])
        if prof is None:
            raise HTTPException(404, f"no profile {v['profile']}")
        cfg = C.deep_merge(m.defaults, prof["config"])
    if v.get("config"):
        cfg = C.deep_merge(cfg, v["config"])
    if cfg["stages"]["tts"]["impl"] != (cfg.get("voice") or {}).get("impl"):
        cfg.setdefault("voice", {})["impl"] = cfg["stages"]["tts"]["impl"]
    return cfg


async def _render_variant(m: Manager, text: str, v: dict, side: str, folder: str) -> dict:
    """Offline render of text through cfg (current pipeline + profile/config overrides), saved as a WAV render."""
    cfg = _variant_cfg(m, v)
    label = v.get("label") or v.get("profile") or cfg["stages"]["tts"]["impl"]
    try:
        vc = cfg.get("voice") or {}
        voice = Voice(impl=cfg["stages"]["tts"]["impl"], voice_id=vc.get("voice_id", ""),
                      speed=float(vc.get("speed", 1.0)), pitch=float(vc.get("pitch", 0.0)))
        r = await m.render_offline(text, cfg, voice, label=f"{folder}-{side}")
        rid = m.renders.save(folder, f"{int(time.time()*1000)}_{side}", r["pcm"], r["rate"],
                             {"text": text, "label": label, "tts_impl": r["tts_impl"], "voice_id": r["voice_id"]},
                             sync=True)
        lat = r["latency"]
        return {"label": label, "render_id": rid, "url": f"/api/renders/{rid}.wav", "tts_impl": r["tts_impl"],
                "voice_id": r["voice_id"], "error": None,
                "metrics": {"ttfa_ms": lat.get("ttfa_ms"), "first_synth_ms": lat.get("first_synth_ms"),
                            "total_synth_ms": lat.get("total_synth_ms"), "audio_ms": lat.get("audio_ms"),
                            "rtf": lat.get("rtf"), "chunks": r["chunks"], "wall_ms": r["wall_ms"]}}
    except HTTPException:
        raise
    except Exception as e:
        return {"label": label, "render_id": None, "url": None, "tts_impl": cfg["stages"]["tts"]["impl"],
                "voice_id": None, "metrics": {}, "error": f"{type(e).__name__}: {e}"}


def _text(body: dict) -> str:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(422, "text: required")
    return text


@router.post("/compare")
async def compare(req: Request, body: dict = Body(...)):
    m = M(req)
    text = _text(body)
    return {side: await _render_variant(m, text, body.get(side) or {}, side, "compare") for side in ("a", "b")}


@router.post("/render")
async def render(req: Request, body: dict = Body(...)):
    """Test render: {text, profile?, config?: partial pipeline config (e.g. stages.tts + voice)} → WAV + metrics."""
    return await _render_variant(M(req), _text(body), body, "t", "test")


# ---- cloned voices (data/voices) ----------------------------------------------------------------


@router.get("/clones")
def clones_list():
    return clones.store().list()


@router.post("/clones", status_code=201)
def clones_create(req: Request, body: dict = Body(...)):
    """{name, transcript, audio_b64 (WAV), language?, notes?}"""
    try:
        wav = base64.b64decode(body.get("audio_b64") or "", validate=True)
    except ValueError:
        raise HTTPException(422, "audio_b64: not valid base64")
    try:
        c = clones.store().create(body.get("name", ""), body.get("transcript", ""), wav, body.get("language", ""),
                                  body.get("notes", ""))
    except ValueError as e:
        raise HTTPException(422, str(e))
    M(req).bus.emit("clones.changed", {"id": c["id"], "action": "created"})
    if fish_server.SERVER.health()[0] == "ok":
        threading.Thread(target=_warm_clone, args=(c["voice_id"],), daemon=True, name="fish-warm").start()
    return c


def _warm_clone(voice_id: str) -> None:
    from app.modules.tts.fish_s2 import FishS2TTS

    try:
        FishS2TTS({}).warm(voice_id)
    except Exception as e:  # only costs the first real utterance the encode it would have paid anyway
        log.warning("warming %s on the Fish server failed: %s", voice_id, e)


def _clone_or_404(fn, cid: str):
    try:
        return fn(cid)
    except KeyError:
        raise HTTPException(404, f"no cloned voice {cid}")


@router.get("/clones/{cid}")
def clones_get(cid: str):
    return _clone_or_404(clones.store().get, cid)


@router.get("/clones/{cid}/audio")
def clones_audio(cid: str):
    audio, _ = _clone_or_404(clones.store().clip, cid)
    return Response(audio, media_type="audio/wav")


@router.put("/clones/{cid}")
def clones_update(cid: str, req: Request, body: dict = Body(...)):
    try:
        c = _clone_or_404(lambda x: clones.store().update(x, body), cid)
    except ValueError as e:
        raise HTTPException(422, str(e))
    M(req).cache.clear()  # cached phrases were spoken with the old transcript
    M(req).bus.emit("clones.changed", {"id": cid, "action": "updated"})
    return c


@router.delete("/clones/{cid}")
def clones_delete(cid: str, req: Request):
    _clone_or_404(clones.store().delete, cid)
    M(req).cache.clear()
    M(req).bus.emit("clones.changed", {"id": cid, "action": "deleted"})
    return {"deleted": cid}


# ---- Fish S2-Pro GPU server -------------------------------------------------------------------


@router.get("/fish")
def fish_status():
    return fish_server.SERVER.status()


@router.post("/fish/{action}")
def fish_action(action: str):
    try:
        if action == "start":
            return fish_server.SERVER.start()
        if action == "stop":
            return fish_server.SERVER.stop()
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    raise HTTPException(404, "action must be start | stop")


def _ratings_path(m: Manager) -> Path:
    return m.s.data_dir / "ratings.jsonl"


@router.post("/compare/ratings")
def rate(req: Request, body: dict = Body(...)):
    score = body.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise HTTPException(422, "score: integer 1..5 required")
    row = {k: body.get(k) for k in ("render_id", "tts_impl", "voice_id", "score", "text", "rater")}
    row["ts"] = int(time.time() * 1000)
    p = _ratings_path(M(req))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"ok": True}


@router.get("/compare/ratings")
def ratings(req: Request):
    p = _ratings_path(M(req))
    rows = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    groups: dict[tuple, list[int]] = {}
    for r in rows:
        groups.setdefault((r.get("tts_impl"), r.get("voice_id")), []).append(r["score"])
    summary = []
    for (impl, vid), scores in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(scores)
        mos = sum(scores) / n
        sd = (sum((s - mos) ** 2 for s in scores) / (n - 1)) ** 0.5 if n > 1 else 0.0
        summary.append({"tts_impl": impl, "voice_id": vid, "n": n, "mos": round(mos, 2),
                        "ci95": round(1.96 * sd / n ** 0.5, 2) if n > 1 else None})
    return {"rows": rows[-500:], "summary": summary}


root_router = APIRouter()


@root_router.get("/metrics", response_class=PlainTextResponse)
def prometheus(req: Request):
    m = M(req)
    return m.metrics.prometheus(m.metrics_snapshot())
