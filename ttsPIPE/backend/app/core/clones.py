"""Cloned-voice library: data/voices/<id>/{clip.wav, clip.lab, meta.json}.

A clone is a short reference clip plus its exact transcript. Zero-shot engines (tts/fish_s2) condition on the
pair; a transcript that does not match the audio gives a quietly worse voice, so it is required. Voice ids are
"clone:<id>". The clip is stored as normalised mono 16-bit WAV (the UI converts any upload or recording).
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from .audio import read_wav, wav_bytes

PREFIX = "clone:"
MIN_S, MAX_S, WARN_S = 2.0, 30.0, 20.0
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def voice_id(cid: str) -> str:
    return PREFIX + cid


def clone_id(vid: str | None) -> str | None:
    return vid[len(PREFIX):] if vid and vid.startswith(PREFIX) else None


_LOCK = threading.Lock()


def store() -> "CloneStore":
    from app import settings

    return CloneStore(settings.current.data_dir / "voices")


class CloneStore:
    def __init__(self, root: Path):
        self.root = root
        self.lock = _LOCK

    def _dir(self, cid: str) -> Path:
        if not _ID.match(cid or ""):
            raise KeyError(cid)
        return self.root / cid

    def list(self) -> list[dict]:
        out = []
        if self.root.exists():
            for d in sorted(self.root.iterdir()):
                try:
                    out.append(self.get(d.name))
                except (KeyError, OSError, ValueError):
                    continue
        return sorted(out, key=lambda m: m.get("created", 0), reverse=True)

    def get(self, cid: str) -> dict:
        d = self._dir(cid)
        if not (d / "meta.json").exists():
            raise KeyError(cid)
        meta = json.loads((d / "meta.json").read_text("utf-8"))
        meta["transcript"] = (d / "clip.lab").read_text("utf-8")
        meta["voice_id"] = voice_id(cid)
        meta["audio_url"] = f"/api/clones/{cid}/audio"
        return meta

    def clip(self, cid: str) -> tuple[bytes, str]:
        d = self._dir(cid)
        if not (d / "clip.wav").exists():
            raise KeyError(cid)
        return (d / "clip.wav").read_bytes(), (d / "clip.lab").read_text("utf-8")

    def create(self, name: str, transcript: str, wav: bytes, language: str = "", notes: str = "") -> dict:
        name, transcript = (name or "").strip(), (transcript or "").strip()
        if not name:
            raise ValueError("name: required")
        if not transcript:
            raise ValueError("transcript: required — the exact words spoken in the clip (a wrong transcript "
                             "silently degrades the cloned voice)")
        try:
            pcm, rate = read_wav(wav)
        except Exception as e:
            raise ValueError(f"audio: not a readable WAV file ({e})") from e
        dur = len(pcm) / rate if rate else 0.0
        if dur < MIN_S or dur > MAX_S:
            raise ValueError(f"audio: {dur:.1f} s long; use a {MIN_S:.0f}–{MAX_S:.0f} s clip (about 10 s is best)")
        peak = float(np.max(np.abs(pcm))) if len(pcm) else 0.0
        if peak < 0.01:
            raise ValueError("audio: the clip is silent")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "voice"
        cid = f"{slug}-{uuid.uuid4().hex[:6]}"
        meta = {"id": cid, "name": name, "language": language or "", "notes": notes or "", "created": time.time(),
                "duration_s": round(dur, 2), "sample_rate": rate, "peak_db": round(20 * np.log10(max(peak, 1e-9)), 1),
                "warning": f"{dur:.0f} s is long: every second is replayed as prompt tokens on each request; ~10 s is best"
                if dur > WARN_S else None}
        with self.lock:
            d = self._dir(cid)
            d.mkdir(parents=True, exist_ok=False)
            (d / "clip.wav").write_bytes(wav_bytes(pcm / max(peak, 1e-9) * 0.9, rate))  # peak-normalised mono
            (d / "clip.lab").write_text(transcript, "utf-8")
            (d / "meta.json").write_text(json.dumps(meta, indent=1), "utf-8")
        return self.get(cid)

    def update(self, cid: str, body: dict) -> dict:
        with self.lock:
            d = self._dir(cid)
            meta = json.loads((d / "meta.json").read_text("utf-8"))
            for k in ("name", "language", "notes"):
                if k in body:
                    meta[k] = str(body[k] or "").strip()
            if not meta["name"]:
                raise ValueError("name: required")
            if "transcript" in body:
                t = str(body["transcript"] or "").strip()
                if not t:
                    raise ValueError("transcript: required")
                (d / "clip.lab").write_text(t, "utf-8")
            (d / "meta.json").write_text(json.dumps(meta, indent=1), "utf-8")
        return self.get(cid)

    def delete(self, cid: str) -> None:
        d = self._dir(cid)
        if not d.exists():
            raise KeyError(cid)
        shutil.rmtree(d)
