"""config/lexicon.yaml loader (reloads when the file changes)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

from app import settings


@lru_cache(maxsize=4)
def _load(path: str, mtime: float) -> dict:
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    words = d.get("words") or {}
    return {
        "words": {lang: {str(k).lower() if str(k).isascii() else str(k): str(v) for k, v in (m or {}).items()}
                  for lang, m in words.items()},
        "tanglish": {str(w).lower() for w in (d.get("tanglish") or [])},
    }


def get(path: Path | None = None) -> dict:
    p = Path(path or settings.current.lexicon)
    try:
        return _load(str(p), os.path.getmtime(p))
    except OSError:
        return {"words": {}, "tanglish": set()}
