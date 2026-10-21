"""Phrase cache: (engine, params, voice, text, prosody) → complete native-rate audio (§5.8)."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .base import WordTiming


@dataclass
class CacheEntry:
    pcm: np.ndarray
    rate: int
    timings: list[WordTiming] | None


def cache_key(impl: str, params: dict, voice_id: str, speed: float, pitch: float, text: str) -> str:
    raw = json.dumps([impl, params, voice_id, round(speed, 3), round(pitch, 2), text], sort_keys=True,
                     ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class PhraseCache:
    def __init__(self, max_mb: float = 64, enabled: bool = True):
        self.max_bytes = int(max_mb * 1e6)
        self.enabled = enabled
        self.items: OrderedDict[str, CacheEntry] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        if not self.enabled:
            return None
        with self.lock:
            e = self.items.get(key)
            if e is None:
                self.misses += 1
                return None
            self.items.move_to_end(key)
            self.hits += 1
            return e

    def put(self, key: str, entry: CacheEntry) -> None:
        if not self.enabled or entry.pcm.nbytes > self.max_bytes:
            return
        with self.lock:
            if key in self.items:
                return
            self.items[key] = entry
            self.bytes += entry.pcm.nbytes
            while self.bytes > self.max_bytes and self.items:
                _, old = self.items.popitem(last=False)
                self.bytes -= old.pcm.nbytes

    def configure(self, max_mb: float, enabled: bool) -> None:
        with self.lock:
            self.max_bytes, self.enabled = int(max_mb * 1e6), enabled
            while self.bytes > self.max_bytes and self.items:
                _, old = self.items.popitem(last=False)
                self.bytes -= old.pcm.nbytes

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.bytes = 0

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hits / total, 3) if total else None,
                "entries": len(self.items), "mb": round(self.bytes / 1e6, 2), "enabled": self.enabled}
