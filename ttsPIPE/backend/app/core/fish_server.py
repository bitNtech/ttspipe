"""The Fish Audio S2-Pro (NF4) GPU server, run as a child process (tts/fish_s2 talks to it over its WebSocket).

The model, its CUDA torch and the vendored fish-speech fork live in that repo's own .venv (settings.fish_dir,
the parent of ttsPIPE by default); ttsPIPE never imports torch for it. This module starts / stops that server
and reports its health, so the UI and tts/fish_s2's load() can bring it up on demand.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import psutil

from app import settings

log = logging.getLogger("ttspipe.fish")


def _get(url: str, timeout: float) -> tuple[int, dict | None]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:  # /health answers 503 with a JSON body when the model failed to load
        try:
            return e.code, json.loads(e.read() or b"null")
        except Exception:
            return e.code, None
    except Exception:
        return 0, None


class FishServer:
    def __init__(self, url_attr: str = "fish_url", env: dict | None = None, label: str = "Fish S2-Pro",
                 log_name: str = "fish-server.log") -> None:
        self.url_attr, self.extra_env, self.label, self.log_name = url_attr, env or {}, label, log_name
        self.proc: subprocess.Popen | None = None
        self.started_at: float | None = None
        self.lock = threading.Lock()

    # ---- paths ---------------------------------------------------------------------------------

    @property
    def url(self) -> str:
        return getattr(settings.current, self.url_attr).rstrip("/")

    @property
    def dir(self) -> Path | None:
        return settings.current.fish_dir

    @property
    def python(self) -> Path | None:
        if self.dir is None:
            return None
        for p in (self.dir / ".venv" / "Scripts" / "python.exe", self.dir / ".venv" / "bin" / "python"):
            if p.exists():
                return p
        return None

    @property
    def log_path(self) -> Path:
        return settings.current.data_dir / "logs" / self.log_name

    def ws_url(self) -> str:
        u = urlparse(self.url)
        return f"{'wss' if u.scheme == 'https' else 'ws'}://{u.netloc}/v1/tts/live"

    # ---- state ---------------------------------------------------------------------------------

    def health(self, timeout: float = 1.0) -> tuple[str, str | None]:
        """ok | loading | error | offline, plus the load error if any."""
        code, body = _get(f"{self.url}/health", timeout)
        if code == 0 or not isinstance(body, dict):
            return "offline", None
        return str(body.get("status", "offline")), body.get("error")

    def managed(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def status(self) -> dict:
        health, error = self.health()
        if health == "offline" and self.managed():
            health = "starting"  # process up, HTTP not listening yet
        if health == "offline" and self.proc is not None and self.proc.poll() is not None:
            error = f"server process exited with code {self.proc.returncode}; see the log below"
        info = None
        if health in ("ok", "loading"):
            _, st = _get(f"{self.url}/status", 2.0)
            if isinstance(st, dict):
                keys = ("ready", "loading", "device", "backend", "model_repo", "sample_rate", "precision", "quantization",
                        "load_seconds", "last_error", "queue_depth", "active_jobs", "total_requests", "total_failed",
                        "total_audio_seconds", "last_ttfa_ms", "vram", "uptime_seconds", "active_voice")
                info = {k: st.get(k) for k in keys}
        ckpt = self.dir / "checkpoints" / "s2-pro-nf4" if self.dir else None
        return {
            "url": self.url, "dir": str(self.dir) if self.dir else None, "health": health, "error": error,
            "managed": self.managed(), "pid": self.proc.pid if self.managed() else None,
            "started_s_ago": round(time.time() - self.started_at, 1) if self.started_at and self.managed() else None,
            "venv_present": self.python is not None,
            "checkpoint_present": bool(ckpt and (ckpt / "model.pth").exists() and (ckpt / "codec.pth").exists()),
            "can_start": self.python is not None and health in ("offline", "error"),
            "info": info, "log_tail": self.log_tail(40),
        }

    def log_tail(self, n: int = 40) -> list[str]:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 16384))
                lines = f.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
            return [ln for ln in lines if ln.strip()][-n:]
        except OSError:
            return []

    # ---- control -------------------------------------------------------------------------------

    def start(self) -> dict:
        with self.lock:
            health, _ = self.health()
            if health in ("ok", "loading") or self.managed():
                return self.status()
            if self.dir is None or self.python is None:
                raise RuntimeError("Fish S2-Pro server not found: set TTSPIPE_FISH_DIR to the repo that contains "
                                   "app/engine/fish.py and its .venv (provision it with that repo's ./run.sh)")
            port = urlparse(self.url).port or 8080
            env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                   "PYTHONPATH": str(self.dir / "vendor" / "fish-speech"), "HF_HOME": str(self.dir / ".hf-home"),
                   "PYTHONUTF8": "1", "TTS_PORT": str(port), **self.extra_env}
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            logf = open(self.log_path, "ab")
            logf.write(f"\n==== ttsPIPE starting {self.label} server {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n".encode())
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
            self.proc = subprocess.Popen(
                [str(self.python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                cwd=self.dir, env=env, stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                creationflags=flags)
            logf.close()
            self.started_at = time.time()
            log.info("started %s server pid %s on port %s", self.label, self.proc.pid, port)
            return self.status()

    def stop(self) -> dict:
        with self.lock:
            pids = []
            if self.managed():
                pids.append(self.proc.pid)
            else:  # started outside ttsPIPE: stop whatever listens on the port
                port = urlparse(self.url).port or 8080
                try:
                    pids += [c.pid for c in psutil.net_connections("tcp")
                             if c.laddr and c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid]
                except psutil.Error:
                    pass
            for pid in set(pids):
                try:
                    p = psutil.Process(pid)
                    procs = p.children(recursive=True) + [p]
                    for q in procs:
                        q.kill()
                    psutil.wait_procs(procs, timeout=10)
                except psutil.Error:
                    pass
            self.proc = None
            self.started_at = None
        return self.status()

    def ensure_ready(self, autostart: bool, timeout_s: float) -> None:
        """Block until /health says ok (starting the server first if allowed). Raises with a readable reason."""
        health, error = self.health()
        if health == "ok":
            return
        if health == "error":
            raise RuntimeError(f"{self.label} server failed to load its model: {error}")
        if health == "offline":
            if not autostart:
                raise RuntimeError(f"{self.label} server is not running at {self.url} (start it from the Engines page)")
            self.start()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            health, error = self.health()
            if health == "ok":
                return
            if health == "error":
                raise RuntimeError(f"{self.label} server failed to load its model: {error}")
            if self.proc is not None and self.proc.poll() is not None:
                tail = " | ".join(self.log_tail(3))
                raise RuntimeError(f"{self.label} server exited with code {self.proc.returncode}: {tail}")
            time.sleep(1.0)
        raise TimeoutError(f"{self.label} server not ready after {timeout_s:.0f} s (still {health})")


SERVER = FishServer()
# Indic-Mio: the same parent-repo server, started with TTS_ENGINE=mio on its own port. It shares the one GPU
# with Fish S2-Pro, which will not fit beside it in 4 GB - stop one before loading the other.
MIO_SERVER = FishServer("mio_url", {"TTS_ENGINE": "mio"}, "Indic-Mio", "mio-server.log")
