"""prePIPE bridge (§5.9): barge-in events → duck/stop/resume; agent speaking state + reference audio back."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.request
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .manager import Manager

log = logging.getLogger("ttspipe.bridge")


def ws_url(http_url: str, path: str) -> str:
    u = http_url.rstrip("/")
    u = "ws://" + u[7:] if u.startswith("http://") else "wss://" + u[8:] if u.startswith("https://") else u
    return u + path


class Bridge:
    def __init__(self, manager: "Manager"):
        self.m = manager
        self.task: asyncio.Task | None = None
        self.connected = False
        self.error: str | None = None
        self.last_event: str | None = None
        self.mapped = 0
        self.ref_queues: dict[str, asyncio.Queue] = {}
        manager.bus.taps.append(self._tap)

    @property
    def cfg(self) -> dict:
        return self.m.config.get("bridge") or {}

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.get("enabled") and self.cfg.get("prepipe_url"))

    def status(self) -> dict:
        return {"enabled": self.enabled, "connected": self.connected, "url": self.cfg.get("prepipe_url") or None,
                "error": self.error, "last_event": self.last_event, "events_mapped": self.mapped}

    def _emit_status(self) -> None:
        self.m.bus.emit("bridge.status", self.status())

    async def restart(self) -> None:
        if self.task:
            self.task.cancel()
            self.task = None
        self.connected, self.error = False, None
        if self.enabled:
            self.task = asyncio.create_task(self._run(), name="bridge")
        self._emit_status()

    async def stop(self) -> None:
        if self.task:
            self.task.cancel()

    async def _run(self) -> None:
        import websockets

        url = ws_url(self.cfg["prepipe_url"], "/ws/events?session_id=*")
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, open_timeout=5) as ws:
                    self.connected, self.error, backoff = True, None, 1.0
                    self._emit_status()
                    async for msg in ws:
                        if isinstance(msg, bytes):
                            continue
                        self._handle(json.loads(msg))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"
            if self.connected:
                self.connected = False
            self._emit_status()
            await asyncio.sleep(backoff)
            backoff = min(30.0, backoff * 2)

    def _target(self, prepipe_sid: str | None):
        sessions = [s for s in self.m.sessions.values() if s.record and not s.closed]
        if self.cfg.get("session_map", "latest") == "same_id":
            return self.m.sessions.get(prepipe_sid or "")
        return max(sessions, key=lambda s: s.started, default=None)

    def _handle(self, ev: dict) -> None:
        action = (self.cfg.get("map") or {}).get(ev.get("type"))
        if action not in ("duck", "stop", "resume"):
            return
        s = self._target(ev.get("session_id"))
        self.last_event = ev.get("type")
        if s is None:
            return
        self.mapped += 1
        getattr(s, action)("bridge")
        self._emit_status()

    # ---- outbound -----------------------------------------------------------------------------

    def _tap(self, ev: dict) -> None:
        if ev.get("type") != "playback.state" or not self.enabled:
            return
        d = ev["data"]
        speaking = d["state"] in ("speaking", "ducked")
        was = d.get("prev") in ("speaking", "ducked")
        if speaking == was:
            return
        url = f"{self.cfg['prepipe_url'].rstrip('/')}/api/sessions/{ev['session_id']}/agent"
        body = json.dumps({"speaking": speaking, "utterance_id": d.get("utterance_id"),
                           "t_out_ms": ev.get("t_out_ms")}).encode()

        def post():
            try:
                req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=2).read()
            except Exception as e:
                log.debug("agent state post failed: %s", e)

        threading.Thread(target=post, daemon=True).start()

    def ref_sink(self, sid: str) -> Callable[[bytes], None] | None:
        if not (self.enabled and self.cfg.get("send_reference_audio")):
            return None
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=500)
        self.ref_queues[sid] = q
        asyncio.create_task(self._ref_sender(sid, q))

        def sink(data: bytes) -> None:
            def put():
                if not q.full():
                    q.put_nowait(data)
            loop.call_soon_threadsafe(put)

        return sink

    async def _ref_sender(self, sid: str, q: asyncio.Queue) -> None:
        import websockets

        url = ws_url(self.cfg["prepipe_url"], f"{self.cfg.get('reference_ws_path', '/ws/audio')}?session_id={sid}")
        try:
            async with websockets.connect(url, open_timeout=5) as ws:
                while sid in self.m.sessions:
                    await ws.send(await q.get())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.info("reference audio link for %s closed: %s", sid, e)
        finally:
            self.ref_queues.pop(sid, None)
