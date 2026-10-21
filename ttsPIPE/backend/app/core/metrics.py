"""Rolling latency stats, counters and the Metrics object (ARCHITECTURE.md §9.14)."""
from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque

import psutil


def pct(vals, p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 2)


class Stat:
    def __init__(self, window: int = 500, bin_s: float = 10.0, bins: int = 30):
        self.vals: deque[float] = deque(maxlen=window)
        self.last: float | None = None
        self.count = 0
        self.bin_s, self.cur_bin, self.cur_vals = bin_s, int(time.time() // bin_s), []
        self.spark: deque[float] = deque(maxlen=bins)

    def add(self, v: float) -> None:
        v = float(v)
        self.vals.append(v)
        self.last = round(v, 2)
        self.count += 1
        b = int(time.time() // self.bin_s)
        if b != self.cur_bin:
            if self.cur_vals:
                self.spark.append(pct(self.cur_vals, 0.95))
            self.cur_bin, self.cur_vals = b, []
        self.cur_vals.append(v)

    def summary(self) -> dict:
        v = list(self.vals)
        return {"n": self.count, "last": self.last, "p50": pct(v, 0.5), "p95": pct(v, 0.95)}

    def sparkline(self) -> list[float]:
        out = list(self.spark)
        if self.cur_vals:
            out.append(pct(self.cur_vals, 0.95))
        return out[-30:]


class Metrics:
    LAT_KEYS = ("ttfa", "ttfa_end", "text_to_first_chunk", "first_synth", "normalize", "chunk_rtf")

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.stage: dict[str, Stat] = defaultdict(Stat)
        self.stage_counts: dict[str, dict] = defaultdict(lambda: {"count": 0, "errors": 0, "impl": None})
        self.lat: dict[str, Stat] = {k: Stat() for k in self.LAT_KEYS}
        self.ttfa_by_profile: dict[str, Stat] = defaultdict(Stat)
        self.ttfa_end_by_profile: dict[str, Stat] = defaultdict(Stat)
        self.counters = defaultdict(int)
        self.recent: deque[tuple[float, str, float]] = deque(maxlen=5000)  # (ts, what, amount)
        self.proc = psutil.Process(os.getpid())
        self.proc.cpu_percent(None)

    def stage_call(self, kind: str, impl: str, ms: float, ok: bool) -> None:
        with self.lock:
            c = self.stage_counts[kind]
            c["impl"] = impl
            c["count"] += 1
            if not ok:
                c["errors"] += 1
            else:
                self.stage[kind].add(ms)

    def add(self, key: str, v: float | None) -> None:
        if v is None:
            return
        with self.lock:
            self.lat[key].add(v)

    def ttfa(self, profile: str, ttfa_ms: float | None, ttfa_end_ms: float | None) -> None:
        with self.lock:
            if ttfa_ms is not None:
                self.lat["ttfa"].add(ttfa_ms)
                self.ttfa_by_profile[profile or "-"].add(ttfa_ms)
            if ttfa_end_ms is not None:
                self.lat["ttfa_end"].add(ttfa_end_ms)
                self.ttfa_end_by_profile[profile or "-"].add(ttfa_end_ms)

    def inc(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counters[name] += n

    def flow(self, what: str, amount: float) -> None:
        with self.lock:
            self.recent.append((time.time(), what, amount))

    def stage_summary(self, kind: str) -> dict:
        with self.lock:
            s = self.stage.get(kind)
            c = self.stage_counts.get(kind, {"count": 0, "errors": 0, "impl": None})
            sm = s.summary() if s else {"last": None, "p50": None, "p95": None}
            return {"impl": c["impl"], "count": c["count"], "errors": c["errors"], "last_ms": sm["last"],
                    "p50_ms": sm["p50"], "p95_ms": sm["p95"], "spark": s.sparkline() if s else []}

    def snapshot(self, *, sessions: int, paused: bool, cache: dict, events_dropped: int) -> dict:
        now = time.time()
        with self.lock:
            per_min = defaultdict(float)
            for ts, what, amt in self.recent:
                if now - ts <= 60:
                    per_min[what] += amt
            kinds = list(self.stage_counts)
        stages = {k: self.stage_summary(k) for k in kinds}
        with self.lock:
            mem = self.proc.memory_info().rss / 1e6
            return {
                "uptime_s": round(now - self.started, 1),
                "sessions": sessions,
                "paused": paused,
                "stages": stages,
                "ttfa": {"by_profile": {k: v.summary() for k, v in self.ttfa_by_profile.items()},
                         "end_by_profile": {k: v.summary() for k, v in self.ttfa_end_by_profile.items()}},
                "latency": {k: v.summary() for k, v in self.lat.items()},
                "throughput": {"chunks_per_min": per_min["chunks"], "audio_s_per_min": round(per_min["audio_s"], 1),
                               "chars_per_min": per_min["chars"]},
                "cache": cache,
                "underruns": self.counters["underruns"],
                "interruptions": self.counters["interruptions"],
                "utterances": self.counters["utterances"],
                "events_dropped": events_dropped,
                "system": {"cpu_pct": self.proc.cpu_percent(None), "rss_mb": round(mem, 1),
                           "threads": self.proc.num_threads(), "cores": psutil.cpu_count(logical=False)},
            }

    def prometheus(self, snap: dict) -> str:
        lines = []

        def g(name, val, labels="", help_=None):
            if val is None:
                return
            if help_:
                lines.append(f"# HELP ttspipe_{name} {help_}")
            lines.append(f"ttspipe_{name}{labels} {val}")

        g("uptime_seconds", snap["uptime_s"])
        g("sessions", snap["sessions"])
        for k, s in snap["stages"].items():
            lab = f'{{kind="{k}",impl="{s["impl"]}"}}'
            g("stage_calls_total", s["count"], lab)
            g("stage_errors_total", s["errors"], lab)
            g("stage_p95_ms", s["p95_ms"], lab)
        for prof, s in snap["ttfa"]["by_profile"].items():
            g("ttfa_p50_ms", s["p50"], f'{{profile="{prof}"}}')
            g("ttfa_p95_ms", s["p95"], f'{{profile="{prof}"}}')
        g("cache_hit_rate", snap["cache"].get("hit_rate"))
        g("underruns_total", snap["underruns"])
        g("interruptions_total", snap["interruptions"])
        g("utterances_total", snap["utterances"])
        g("events_dropped_total", snap["events_dropped"])
        g("rss_mb", snap["system"]["rss_mb"])
        g("cpu_percent", snap["system"]["cpu_pct"])
        return "\n".join(lines) + "\n"
