"""Profiles (partial configs, §13), dashboard layouts (§14) and rendered utterances (§9.8)."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import numpy as np

from .audio import wav_bytes
from .config import load_yaml, save_yaml

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def check_name(name: str) -> None:
    if not NAME_RE.match(name or ""):
        raise ValueError("name must match [A-Za-z0-9_.-]{1,64}")


class ProfileStore:
    def __init__(self, builtin_dir: Path, user_dir: Path):
        self.builtin_dir, self.user_dir = builtin_dir, user_dir

    def _builtin(self) -> dict[str, Path]:
        return {p.stem: p for p in sorted(self.builtin_dir.glob("*.yaml"))} if self.builtin_dir.exists() else {}

    def _user(self) -> dict[str, Path]:
        return {p.stem: p for p in sorted(self.user_dir.glob("*.yaml"))} if self.user_dir.exists() else {}

    def is_builtin(self, name: str) -> bool:
        return name in self._builtin()

    def list(self) -> list[dict]:
        out = []
        for builtin, items in ((True, self._builtin()), (False, self._user())):
            for name, p in items.items():
                d = load_yaml(p)
                out.append({"name": name, "builtin": builtin, "description": d.get("description", ""),
                            "updated": int(p.stat().st_mtime * 1000)})
        return out

    def get(self, name: str) -> dict | None:
        p = self._builtin().get(name) or self._user().get(name)
        if p is None:
            return None
        d = load_yaml(p)
        return {"name": name, "builtin": self.is_builtin(name), "description": d.get("description", ""),
                "config": d.get("config") or {}}

    def save(self, name: str, description: str, config: dict) -> dict:
        check_name(name)
        if self.is_builtin(name):
            raise PermissionError(f"profile {name!r} is builtin (read-only)")
        config = {k: v for k, v in config.items() if k != "active_profile"}
        save_yaml(self.user_dir / f"{name}.yaml", {"description": description, "config": config})
        return self.get(name)

    def delete(self, name: str) -> bool:
        if self.is_builtin(name):
            raise PermissionError(f"profile {name!r} is builtin (read-only)")
        p = self._user().get(name)
        if p is None:
            return False
        p.unlink()
        return True


class LayoutStore:
    def __init__(self, d: Path):
        self.d = d

    def list(self) -> list[dict]:
        if not self.d.exists():
            return []
        return [{"name": p.stem, "updated": int(p.stat().st_mtime * 1000)} for p in sorted(self.d.glob("*.json"))]

    def get(self, name: str) -> dict | None:
        check_name(name)
        p = self.d / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save(self, name: str, layout: dict) -> dict:
        check_name(name)
        self.d.mkdir(parents=True, exist_ok=True)
        layout = {**layout, "name": name, "updated": int(time.time() * 1000)}
        (self.d / f"{name}.json").write_text(json.dumps(layout, indent=1), encoding="utf-8")
        return layout

    def delete(self, name: str) -> bool:
        check_name(name)
        p = self.d / f"{name}.json"
        if p.exists():
            p.unlink()
            return True
        return False


class RenderStore:
    """data/renders/<session>/<utterance>.wav + .json; render id = <session>__<utterance>."""

    def __init__(self, d: Path, keep: int = 200, enabled: bool = True):
        self.d, self.keep, self.enabled = d, keep, enabled
        self.lock = threading.Lock()

    @staticmethod
    def _safe(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "_", s)

    def path(self, render_id: str) -> Path | None:
        sid, _, uid = render_id.partition("__")
        if not sid or not uid:
            return None
        p = self.d / self._safe(sid) / f"{self._safe(uid)}.wav"
        return p if p.exists() else None

    def save(self, sid: str, uid: str, pcm: np.ndarray, rate: int, meta: dict, *, sync: bool = False) -> str | None:
        if not (self.enabled or sync) or not len(pcm):
            return None
        rid = f"{sid}__{uid}"
        if sync:
            self._write(rid, sid, uid, pcm, rate, meta)
        else:
            threading.Thread(target=self._write, args=(rid, sid, uid, pcm, rate, meta), daemon=True).start()
        return rid

    def _write(self, rid: str, sid: str, uid: str, pcm: np.ndarray, rate: int, meta: dict) -> None:
        folder = self.d / self._safe(sid)
        with self.lock:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{self._safe(uid)}.wav").write_bytes(wav_bytes(pcm, rate))
            (folder / f"{self._safe(uid)}.json").write_text(json.dumps({
                **meta, "id": rid, "session_id": sid, "utterance_id": uid, "sample_rate": rate,
                "duration_ms": round(len(pcm) * 1000 / rate), "created": int(time.time() * 1000)},
                ensure_ascii=False), encoding="utf-8")
            self._prune()

    def _prune(self) -> None:
        metas = sorted(self.d.glob("*/*.json"), key=lambda p: p.stat().st_mtime)
        for m in metas[:max(0, len(metas) - self.keep)]:
            m.unlink(missing_ok=True)
            m.with_suffix(".wav").unlink(missing_ok=True)

    def list(self, session_id: str | None = None) -> list[dict]:
        if not self.d.exists():
            return []
        out = []
        for m in sorted(self.d.glob("*/*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                d = json.loads(m.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if session_id and d.get("session_id") != session_id:
                continue
            d["url"] = f"/api/renders/{d['id']}.wav"
            out.append({k: d.get(k) for k in ("id", "session_id", "utterance_id", "text", "duration_ms", "created",
                                              "interrupted", "sample_rate", "url", "shadow", "heard_text")})
        return out

    def delete(self, render_id: str) -> bool:
        p = self.path(render_id)
        if p is None:
            return False
        p.unlink(missing_ok=True)
        p.with_suffix(".json").unlink(missing_ok=True)
        return True
