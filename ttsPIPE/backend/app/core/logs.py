"""Logging: ring buffer for /api/logs, `log` events on the bus, rotating file."""
from __future__ import annotations

import logging
import logging.handlers
from collections import deque
from pathlib import Path

LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


class RingHandler(logging.Handler):
    def __init__(self, maxlen: int = 2000):
        super().__init__()
        self.buf: deque[dict] = deque(maxlen=maxlen)
        self.bus = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            item = {"ts": int(record.created * 1000), "level": record.levelname, "logger": record.name,
                    "msg": record.getMessage(), "session_id": getattr(record, "session_id", None)}
        except Exception:
            return
        self.buf.append(item)
        if self.bus is not None and record.levelno >= logging.INFO:
            self.bus.emit("log", {k: item[k] for k in ("level", "logger", "msg", "session_id")})

    def query(self, level: str = "DEBUG", limit: int = 200, session_id: str | None = None) -> list[dict]:
        lv = LEVELS.get(level.upper(), 10)
        out = [r for r in self.buf if LEVELS.get(r["level"], 0) >= lv
               and (session_id is None or r["session_id"] == session_id)]
        return out[-limit:]


ring = RingHandler()


def setup(level: str = "INFO", log_dir: Path | None = None) -> None:
    root = logging.getLogger("ttspipe")
    root.setLevel(LEVELS.get(level.upper(), 20))
    root.propagate = False
    if ring not in root.handlers:
        root.addHandler(ring)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(sh)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.handlers.RotatingFileHandler(log_dir / "ttspipe.log", maxBytes=5_000_000,
                                                      backupCount=3, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            root.addHandler(fh)
