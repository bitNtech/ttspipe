"""Prometheus metrics, in the text exposition format, with no dependency.

`/status` answers "what is this process doing right now" for a human. This is
the other half: counters and latency histograms a scraper can turn into rates,
percentiles and alerts. The figure that matters most to a voice agent is
`tts_first_audio_seconds` - from the moment a turn's first segment was ready
to synthesize to the moment its first audio was queued for the socket. That is
the server's whole share of the agent's response latency, queueing included,
which the engine's own `last_ttfa_ms` does not see.

Hand-rolled because the format is a few lines of text and the server should
not grow a dependency for it. Updates happen on the event loop only, so there
is no locking; the engine's own counters are read at scrape time instead of
being mirrored here.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:  # pragma: no cover - typing only
    from app.engine.base import EngineStatus

#: Latency buckets in seconds, dense where a voice agent cares (50-500 ms).
LATENCY_BUCKETS = (0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0)


def _labels(labels: dict[str, str]) -> str:
    if not labels:
        return ""
    inner = ",".join(
        f'{k}="{str(v).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in sorted(labels.items())
    )
    return "{" + inner + "}"


def _num(value: float) -> str:
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    return repr(float(value))


class Counter:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._values: dict[tuple, float] = {}

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = tuple(sorted(labels.items()))
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(tuple(sorted(labels.items())), 0.0)

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} counter"
        for key, value in sorted(self._values.items()):
            yield f"{self.name}{_labels(dict(key))} {_num(value)}"


class Gauge:
    def __init__(self, name: str, help_: str) -> None:
        self.name, self.help = name, help_
        self._value = 0.0

    def set(self, value: float) -> None:
        self._value = float(value)

    def inc(self, amount: float = 1.0) -> None:
        self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        self._value -= amount

    @property
    def value(self) -> float:
        return self._value

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} gauge"
        yield f"{self.name} {_num(self._value)}"


class Histogram:
    def __init__(self, name: str, help_: str, buckets: tuple[float, ...] = LATENCY_BUCKETS):
        self.name, self.help = name, help_
        self.buckets = tuple(sorted(buckets)) + (math.inf,)
        self._counts = [0] * len(self.buckets)
        self._sum = 0.0
        self._count = 0

    def observe(self, value: float) -> None:
        self._sum += value
        self._count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self._counts[i] += 1
                break

    @property
    def count(self) -> int:
        return self._count

    def render(self) -> Iterable[str]:
        yield f"# HELP {self.name} {self.help}"
        yield f"# TYPE {self.name} histogram"
        cumulative = 0
        for bound, n in zip(self.buckets, self._counts):
            cumulative += n
            yield f'{self.name}_bucket{{le="{_num(bound)}"}} {cumulative}'
        yield f"{self.name}_sum {_num(self._sum)}"
        yield f"{self.name}_count {self._count}"


class Metrics:
    """Every series this server exports. One instance per process."""

    def __init__(self) -> None:
        self.sessions_active = Gauge("tts_sessions_active", "Open WebSocket sessions.")
        self.sessions_total = Counter("tts_sessions_total", "WebSocket sessions accepted.")
        self.sessions_rejected = Counter(
            "tts_sessions_rejected_total", "WebSocket connections refused, by reason."
        )
        self.turns = Counter(
            "tts_turns_total", "Finish frames sent, by reason (a barged-in turn counts cancel, then stop)."
        )
        self.segments = Counter("tts_segments_total", "Text segments sent to the engine.")
        self.segment_failures = Counter(
            "tts_segment_failures_total", "Segments that failed in the engine."
        )
        self.first_audio = Histogram(
            "tts_first_audio_seconds",
            "Turn's first segment ready -> its first audio queued for the client.",
        )
        self.segment_first_audio = Histogram(
            "tts_segment_first_audio_seconds",
            "Engine call -> first audio chunk, per segment (includes queueing).",
        )
        self.audio_seconds = Counter("tts_audio_seconds_total", "Seconds of audio delivered.")
        self.http_requests = Counter("tts_http_synthesis_total", "One-shot HTTP syntheses, by status.")
        self.auth_failures = Counter("tts_auth_failures_total", "Requests refused for a bad API key.")
        self.idle_timeouts = Counter(
            "tts_session_idle_timeouts_total", "Sessions closed for sending nothing."
        )

    def _own(self):
        return (
            self.sessions_active,
            self.sessions_total,
            self.sessions_rejected,
            self.turns,
            self.segments,
            self.segment_failures,
            self.first_audio,
            self.segment_first_audio,
            self.audio_seconds,
            self.http_requests,
            self.auth_failures,
            self.idle_timeouts,
        )

    def render(
        self, status: "EngineStatus | None", *, ready: bool, draining: bool, uptime_s: float
    ) -> str:
        lines: list[str] = []
        for metric in self._own():
            lines.extend(metric.render())

        def gauge(name: str, help_: str, value: float | None) -> None:
            if value is None:
                return
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {_num(value)}")

        def counter(name: str, help_: str, value: float | None) -> None:
            if value is None:
                return
            lines.append(f"# HELP {name} {help_}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {_num(value)}")

        gauge("tts_ready", "1 when the engine is loaded and serving.", float(ready))
        gauge("tts_draining", "1 once a drain has been requested.", float(draining))
        gauge("tts_uptime_seconds", "Seconds since the process started.", uptime_s)
        if status is not None:
            gauge("tts_engine_queue_depth", "Jobs waiting for the engine worker.", status.queue_depth)
            gauge("tts_engine_active_jobs", "Segments the engine is serving now.", status.active_jobs)
            counter("tts_engine_requests_total", "Segments submitted to the engine.", status.total_requests)
            counter("tts_engine_failed_total", "Segments the engine failed.", status.total_failed)
            counter("tts_engine_cancelled_total", "Segments cancelled mid-synthesis.", status.total_cancelled)
            if status.last_ttfa_ms is not None:
                gauge(
                    "tts_engine_last_ttfa_seconds",
                    "Engine-side time to first audio of the latest segment.",
                    status.last_ttfa_ms / 1000.0,
                )
            cache = status.prefix_cache or {}
            if "hits" in cache:
                counter("tts_prefix_cache_hits_total", "Prompt prefix cache hits.", cache.get("hits"))
                counter("tts_prefix_cache_misses_total", "Prompt prefix cache misses.", cache.get("misses"))
            for key, value in (status.vram or {}).items():
                if isinstance(value, (int, float)):
                    gauge(f"tts_vram_{key}", f"VRAM figure '{key}' from /status.", value)
        return "\n".join(lines) + "\n"


#: The process-wide registry. Tests may read it; nothing should replace it.
METRICS = Metrics()
