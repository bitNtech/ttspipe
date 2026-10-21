"""Event bus with per-subscriber back-pressure (ARCHITECTURE.md §7.3).

emit() is thread-safe: the pacer thread, stage executors and asyncio tasks all publish.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from typing import Any, Callable

CRITICAL_PREFIXES = ("utterance.", "session.")
CRITICAL = {"playback.state", "chunk.flushed", "synth.first_audio", "stage.status", "error"}
GLOBAL_TYPES = {"metrics", "log", "stage.status", "model.download", "bridge.status", "clones.changed"}


def is_critical(t: str) -> bool:
    return t in CRITICAL or t.startswith(CRITICAL_PREFIXES)


def now_ms() -> int:
    return int(time.time() * 1000)


def envelope(type_: str, data: dict, session_id: str | None = None, t_out_ms: int | None = None) -> dict:
    return {"type": type_, "session_id": session_id, "ts": now_ms(), "t_out_ms": t_out_ms, "data": data}


class Subscriber:
    def __init__(self, loop: asyncio.AbstractEventLoop, filt: Callable[[dict], bool], maxlen: int, bus: "Bus"):
        self.loop, self.filt, self.maxlen, self.bus = loop, filt, maxlen, bus
        self.q: deque[dict] = deque()
        self.lock = threading.Lock()
        self.ev = asyncio.Event()
        self.closed = False

    def put(self, ev: dict) -> None:
        with self.lock:
            self.q.append(ev)
            if len(self.q) > self.maxlen:
                # drop the oldest non-critical event; never drop critical ones
                for i, old in enumerate(self.q):
                    if not is_critical(old["type"]):
                        del self.q[i]
                        self.bus.dropped += 1
                        break
        try:
            self.loop.call_soon_threadsafe(self.ev.set)
        except RuntimeError:  # loop closed
            self.closed = True

    async def get_batch(self, max_items: int = 200) -> list[dict]:
        while True:
            with self.lock:
                if self.q:
                    n = min(max_items, len(self.q))
                    return [self.q.popleft() for _ in range(n)]
                self.ev.clear()
            await self.ev.wait()


class Bus:
    def __init__(self, queue_size: int = 2000):
        self.queue_size = queue_size
        self.subs: list[Subscriber] = []
        self.lock = threading.Lock()
        self.dropped = 0
        self.taps: list[Callable[[dict], None]] = []  # synchronous observers (metrics, session history)

    def subscribe(self, loop, filt: Callable[[dict], bool]) -> Subscriber:
        s = Subscriber(loop, filt, self.queue_size, self)
        with self.lock:
            self.subs.append(s)
        return s

    def unsubscribe(self, s: Subscriber) -> None:
        with self.lock:
            if s in self.subs:
                self.subs.remove(s)

    def emit(self, type_: str, data: dict | None = None, session_id: str | None = None,
             t_out_ms: int | None = None) -> dict:
        ev = envelope(type_, data or {}, session_id, t_out_ms)
        self.publish(ev)
        return ev

    def publish(self, ev: dict) -> None:
        for tap in self.taps:
            try:
                tap(ev)
            except Exception:
                pass
        with self.lock:
            subs = list(self.subs)
        for s in subs:
            if not s.closed and s.filt(ev):
                s.put(ev)


def dumps(ev: Any) -> str:
    return json.dumps(ev, ensure_ascii=False, separators=(",", ":"), default=_default)


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    return str(o)
