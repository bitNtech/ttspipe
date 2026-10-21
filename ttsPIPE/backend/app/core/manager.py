"""Manager: config, stage slots, sessions, models, compare/preview (§5.1)."""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import os
import threading
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
import psutil

from app import settings as settings_mod
from app.settings import CONTRACT, VERSION, Settings

from . import config as C
from . import graph, logs, registry
from .base import KINDS, Voice
from .bridge import Bridge
from .cache import PhraseCache
from .fillers import FillerBank
from .events import Bus
from .metrics import Metrics
from .profiles import LayoutStore, ProfileStore, RenderStore
from .runtime import StageSlot
from .session import Session, new_session_id

log = logging.getLogger("ttspipe.manager")
AUX_MAX = 4


class Manager:
    def __init__(self, s: Settings):
        self.s = s
        settings_mod.current = s
        self.bus = Bus()
        self.metrics = Metrics()
        self.override_path = s.data_dir / "pipeline.override.yaml"
        registry.discover()
        self.defaults, self.config, self.config_warnings = C.load_effective(s.config, self.override_path)
        self.bus.queue_size = int((self.config.get("events") or {}).get("queue_size", 2000))
        cache = self.config.get("cache") or {}
        self.cache = PhraseCache(cache.get("max_mb", 64), cache.get("enabled", True))
        self.fillers = FillerBank()
        rend = self.config.get("renders") or {}
        self.renders = RenderStore(s.data_dir / "renders", rend.get("keep", 200), rend.get("enabled", True))
        self.profiles = ProfileStore(s.profiles_dir, s.data_dir / "profiles")
        self.layouts = LayoutStore(s.data_dir / "layouts")
        self.slots: dict[str, StageSlot] = {k: StageSlot(k, self.bus, self.metrics, s.models_dir) for k in KINDS}
        self.shadow: StageSlot | None = None
        self.sessions: dict[str, Session] = {}
        self.aux: dict[str, StageSlot] = {}
        self.downloads: dict[str, dict] = {}
        self.paused = False
        self.started = time.time()
        self.bridge: Bridge | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []

    # ---- lifecycle ----------------------------------------------------------------------------

    async def start(self, background: bool = True) -> None:
        self.loop = asyncio.get_running_loop()
        logs.ring.bus = self.bus
        os.environ.setdefault("OMP_NUM_THREADS", str(self.s.threads))
        for kind in KINDS:
            st = self.config["stages"][kind]
            self.slots[kind].configure(st["impl"], st.get("params"), st.get("enabled", True), background=background)
        self._configure_shadow(background)
        self.bridge = Bridge(self)
        await self.bridge.restart()
        self._tasks.append(asyncio.create_task(self._metrics_loop(), name="metrics"))
        for w in self.config_warnings:
            self.bus.emit("log", {"level": "WARNING", "logger": "ttspipe.config", "msg": w, "session_id": None})

    async def shutdown(self) -> None:
        for t in self._tasks:
            t.cancel()
        for sess in list(self.sessions.values()):
            await sess.close("shutdown")
        if self.bridge:
            await self.bridge.stop()
        for slot in [*self.slots.values(), *( [self.shadow] if self.shadow else []), *self.aux.values()]:
            slot.shutdown()

    async def _metrics_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                self.bus.emit("metrics", self.metrics_snapshot())
            except Exception:
                log.exception("metrics snapshot failed")

    def metrics_snapshot(self) -> dict:
        return self.metrics.snapshot(sessions=len(self.sessions), paused=self.paused, cache=self.cache.stats(),
                                     events_dropped=self.bus.dropped)

    def wait_ready(self, timeout: float = 120) -> bool:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end:
            slots = [*self.slots.values(), *([self.shadow] if self.shadow else [])]
            if all(s.status != "loading" for s in slots):
                return True
            time.sleep(0.05)
        return False

    # ---- config -------------------------------------------------------------------------------

    def _configure_shadow(self, background: bool = True) -> None:
        sh = ((self.config.get("graph") or {}).get("shadows") or {}).get("tts")
        if not sh:
            if self.shadow:
                self.shadow.shutdown()
                self.shadow = None
            return
        if self.shadow is None:
            self.shadow = StageSlot("tts", self.bus, self.metrics, self.s.models_dir, name="shadow")
        self.shadow.configure(sh["impl"], sh.get("params"), True, background=background)

    def dirty(self) -> bool:
        name = self.config.get("active_profile")
        prof = self.profiles.get(name) if name else None
        expected = C.deep_merge(self.defaults, prof["config"]) if prof else self.defaults
        strip = lambda d: {k: v for k, v in d.items() if k != "active_profile"}
        return strip(expected) != strip(self.config)

    def set_config(self, new: dict, *, persist: bool = True) -> tuple[list[str], list[str]]:
        errors, warnings = C.validate(new)
        if errors:
            return errors, warnings
        old, self.config = self.config, copy.deepcopy(new)
        for kind in KINDS:
            if old["stages"].get(kind) != new["stages"][kind]:
                st = new["stages"][kind]
                self.slots[kind].configure(st["impl"], st.get("params"), st.get("enabled", True))
        if (old.get("graph") or {}).get("shadows") != (new.get("graph") or {}).get("shadows"):
            self._configure_shadow()
        if old.get("playback") != new.get("playback"):
            for sess in self.sessions.values():
                if sess.playback and sess._cfg is None:
                    sess.playback.update_cfg(new["playback"])
        cache = new.get("cache") or {}
        self.cache.configure(cache.get("max_mb", 64), cache.get("enabled", True))
        rend = new.get("renders") or {}
        self.renders.keep, self.renders.enabled = rend.get("keep", 200), rend.get("enabled", True)
        if old.get("bridge") != new.get("bridge") and self.bridge and self.loop:
            if _in_loop(self.loop):
                asyncio.ensure_future(self.bridge.restart())
            else:
                asyncio.run_coroutine_threadsafe(self.bridge.restart(), self.loop)
        if persist:
            C.save_yaml(self.override_path, C.diff(self.defaults, self.config))
        return [], warnings

    def patch(self, path: list[str], value) -> tuple[list[str], list[str]]:
        new = copy.deepcopy(self.config)
        node = new
        for p in path[:-1]:
            node = node.setdefault(p, {})
        if isinstance(value, dict) and isinstance(node.get(path[-1]), dict):
            node[path[-1]] = C.deep_merge(node[path[-1]], value)
        else:
            node[path[-1]] = value
        return self.set_config(new)

    def reset_config(self) -> None:
        self.override_path.unlink(missing_ok=True)
        self.set_config(copy.deepcopy(self.defaults), persist=False)

    def apply_profile(self, name: str) -> tuple[list[str], list[str]]:
        prof = self.profiles.get(name)
        if prof is None:
            raise KeyError(name)
        new = C.deep_merge(self.defaults, prof["config"])
        new["active_profile"] = name
        return self.set_config(new)

    def estimate(self, cfg: dict) -> dict:
        ram, dl = 0.0, 0.0
        ids = [f"{k}/{cfg['stages'][k]['impl']}" for k in KINDS if k in cfg.get("stages", {})]
        sh = ((cfg.get("graph") or {}).get("shadows") or {}).get("tts")
        if sh:
            ids.append(f"tts/{sh['impl']}")
        for mid in ids:
            try:
                cls = registry.get(mid)
            except KeyError:
                continue
            ram += cls.info.ram_mb
            dl += sum(f.size_mb for f in self._model_files(cls) if not f.present)
        return {"download_mb": round(dl, 1), "ram_mb": round(ram, 1)}

    def validate(self, partial: dict) -> dict:
        cfg = C.deep_merge(self.config, partial or {})
        errors, warnings = C.validate(cfg)
        est = self.estimate(cfg) if not errors else {"download_mb": 0, "ram_mb": 0}
        total_mb = psutil.virtual_memory().total / 1e6
        if est["ram_mb"] > 0.75 * total_mb:
            warnings.append(f"estimated RAM {est['ram_mb']:.0f} MB exceeds 75% of {total_mb:.0f} MB")
        if est["download_mb"]:
            warnings.append(f"model files missing: {est['download_mb']:.0f} MB to download")
        return {"ok": not errors, "errors": errors, "warnings": warnings, "estimate": est}

    # ---- views --------------------------------------------------------------------------------

    def stage_statuses(self) -> dict:
        return {k: s.status_dict() for k, s in self.slots.items()}

    def pipeline_view(self) -> dict:
        stages = self.stage_statuses()
        return {"config": self.config, "stages": stages, "shadow": self.shadow.status_dict() if self.shadow else None,
                "graph": graph.build(self.config, stages), "active_profile": self.config.get("active_profile"),
                "dirty": self.dirty()}

    def health(self) -> dict:
        reasons = []
        for k, s in self.slots.items():
            if s.fallback or s.status == "error":
                reasons.append(f"{k}: {s.configured_impl} → {s.impl} ({s.error})")
        if self.shadow and (self.shadow.fallback or self.shadow.status == "error"):
            reasons.append(f"shadow: {self.shadow.error}")
        tts_ok = self.slots["tts"].module is not None
        return {"status": "error" if not tts_ok else ("degraded" if reasons else "ok"), "contract": CONTRACT,
                "version": VERSION, "uptime_s": round(time.time() - self.started, 1), "reasons": reasons,
                "stages": {k: s.status_dict()["status"] for k, s in self.slots.items()}}

    def state(self) -> dict:
        return {"paused": self.paused, "sessions": len(self.sessions), "active_profile": self.config.get("active_profile"),
                "bridge": self.bridge.status() if self.bridge else None, "cache": self.cache.stats(),
                "uptime_s": round(time.time() - self.started, 1)}

    def _model_files(self, cls, params: dict | None = None):
        try:
            inst = cls(params or {})
        except Exception:
            return []
        inst.models_dir = self.s.models_dir / cls.info.kind / cls.info.impl
        files = inst.model_files()
        for f in files:
            f.present = (self.s.models_dir / f.path).exists()
        return files

    def modules(self) -> list[dict]:
        out = []
        for cls in registry.all_modules():
            info = cls.info
            ok, reason = cls.availability()
            slot = self.slots.get(info.kind)
            loaded = slot is not None and slot.impl == info.impl and slot.module is not None
            if self.shadow and self.shadow.impl == info.impl and info.kind == "tts":
                loaded = True
            files = self._model_files(cls)
            d = info.to_dict()
            if info.kind == "tts":
                d["voices"] = [dict(v.__dict__) for v in cls.list_voices()]
            d.update({"available": ok, "reason": reason, "loaded": loaded,
                      "status": slot.status_dict()["status"] if loaded else ("unloaded" if ok else "error"),
                      "models_present": all(f.present for f in files)})
            out.append(d)
        return out

    def models(self) -> list[dict]:
        out = []
        for cls in registry.all_modules():
            files = self._model_files(cls)
            if not files:
                continue
            mid = cls.info.id
            dl = self.downloads.get(mid, {})
            out.append({"id": mid, "kind": cls.info.kind, "impl": cls.info.impl,
                        "files": [f.__dict__ for f in files], "present": all(f.present for f in files),
                        "size_mb": round(sum(f.size_mb for f in files), 1), "downloading": dl.get("status") == "running",
                        "progress": dl or None})
        return out

    def download(self, kind: str, impl: str) -> dict:
        cls = registry.get(f"{kind}/{impl}")
        mid = cls.info.id
        if self.downloads.get(mid, {}).get("status") == "running":
            return self.downloads[mid]
        files = [f for f in self._model_files(cls) if not f.present]
        state = {"status": "running", "done_mb": 0.0, "total_mb": round(sum(f.size_mb for f in files), 1), "error": None}
        self.downloads[mid] = state

        def emit(file: str):
            self.bus.emit("model.download", {"kind": kind, "impl": impl, "file": file, **state})

        def run():
            try:
                for f in files:
                    download_file(f.url, self.s.models_dir / f.path, state, lambda: emit(f.name))
                state["status"] = "done"
            except Exception as e:
                state.update(status="error", error=f"{type(e).__name__}: {e}")
            emit("")

        threading.Thread(target=run, daemon=True, name=f"download-{mid}").start()
        return state

    # ---- sessions -----------------------------------------------------------------------------

    async def open_session(self, sid: str | None = None, label: str = "", voice: str | None = None,
                           before_start=None) -> Session:
        sid = sid or new_session_id()
        if sid in self.sessions:
            await self.sessions[sid].close("replaced")
            self.sessions.pop(sid, None)
        sess = Session(self, sid, label)
        if voice:
            sess.set_voice(voice)
        self.sessions[sid] = sess
        if before_start:
            before_start(sess)  # e.g. subscribe so session.started is not missed
        await sess.start()
        return sess

    async def close_session(self, sid: str, reason: str = "closed") -> None:
        sess = self.sessions.pop(sid, None)
        if sess:
            await sess.close(reason)

    # ---- auxiliary pool: preview / compare ----------------------------------------------------

    async def aux_slot(self, kind: str, impl: str, params: dict | None) -> StageSlot:
        main = self.slots[kind]
        if main.impl == impl and not main.fallback and (params is None or main.module.params ==
                                                       {**main.module.info.default_params, **params}):
            return main
        key = f"{kind}/{impl}:{json.dumps(params or {}, sort_keys=True)}"
        slot = self.aux.get(key)
        if slot is None:
            slot = StageSlot(kind, self.bus, self.metrics, self.s.models_dir, name=f"aux-{kind}")
            slot.global_emit = False
            await asyncio.to_thread(slot.configure, impl, params or {}, True, background=False)
            self.aux[key] = slot
            while len(self.aux) > AUX_MAX:
                old_key = next(iter(self.aux))
                self.aux.pop(old_key).shutdown()
        return slot

    async def render_offline(self, text: str, cfg: dict, voice: Voice | None = None, label: str = "offline") -> dict:
        """Run text through a full pipeline built from cfg without real-time pacing (compare / preview)."""
        errors, _ = C.validate(cfg)
        if errors:
            raise ValueError("; ".join(errors))
        slots = {}
        for k in KINDS:
            st = cfg["stages"][k]
            slots[k] = await self.aux_slot(k, st["impl"], st.get("params") or {})
        sid = "x" + uuid.uuid4().hex[:6]
        sess = Session(self, sid, label, voice, realtime=False, slots=slots, shadow=False, cfg=cfg,
                       sink=lambda b: None, record=False)
        await sess.start()
        t0 = time.perf_counter()
        uid = sess.speak(text)
        ok = await sess.wait_idle(timeout=600)
        u = sess.utts.get(uid)
        await sess.close("offline")
        if not ok or u is None:
            raise TimeoutError("offline render timed out")
        lat = (u.summary or {}).get("latency", {})
        pcm = getattr(u, "render_pcm", np.zeros(0, np.float32))
        return {"pcm": pcm, "rate": sess.rate, "latency": lat, "chunks": len(u.chunks),
                "wall_ms": round((time.perf_counter() - t0) * 1000, 1), "tts_impl": slots["tts"].impl,
                "voice_id": sess._voice_for(slots["tts"].module).voice_id, "session_id": sid, "utterance_id": uid,
                "summary": u.summary}


def _in_loop(loop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def download_file(url: str, dest: Path, state: dict, progress) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ttsPIPE"}
    if os.environ.get("HF_TOKEN") and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=headers)
    last = 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            state["done_mb"] = round(state["done_mb"] + len(b) / 1e6, 1)
            if time.monotonic() - last > 0.5:
                last = time.monotonic()
                progress()
    tmp.replace(dest)
