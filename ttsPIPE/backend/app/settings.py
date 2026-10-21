"""TTSPIPE_* environment settings (ARCHITECTURE.md §2.1)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parents[2]
VERSION = "0.1.0"
CONTRACT = "1.1"


@dataclass
class Settings:
    host: str
    port: int
    data_dir: Path
    config: Path
    profiles_dir: Path
    lexicon: Path
    log_level: str
    threads: int
    static_dir: Path
    models_override: Path | None = None
    fish_dir: Path | None = None  # Fish S2-Pro server repo (the parent of ttsPIPE when it contains one)
    fish_url: str = "http://127.0.0.1:8080"
    mio_url: str = "http://127.0.0.1:8081"

    @classmethod
    def from_env(cls, **over) -> "Settings":
        e = os.environ.get
        threads = int(e("TTSPIPE_THREADS", "0")) or (psutil.cpu_count(logical=False) or 4)
        s = cls(
            host=e("TTSPIPE_HOST", "127.0.0.1"),
            port=int(e("TTSPIPE_PORT", "8200")),
            data_dir=Path(e("TTSPIPE_DATA_DIR", str(REPO / "data"))),
            config=Path(e("TTSPIPE_CONFIG", str(REPO / "config" / "pipeline.yaml"))),
            profiles_dir=Path(e("TTSPIPE_PROFILES_DIR", str(REPO / "config" / "profiles"))),
            lexicon=Path(e("TTSPIPE_LEXICON", str(REPO / "config" / "lexicon.yaml"))),
            log_level=e("TTSPIPE_LOG_LEVEL", "INFO"),
            threads=threads,
            static_dir=Path(e("TTSPIPE_STATIC_DIR", str(REPO / "frontend" / "dist"))),
            models_override=Path(e("TTSPIPE_MODELS_DIR")) if e("TTSPIPE_MODELS_DIR") else None,
            fish_dir=Path(e("TTSPIPE_FISH_DIR")) if e("TTSPIPE_FISH_DIR") else (
                REPO.parent if (REPO.parent / "app" / "engine" / "fish.py").exists() else None),
            fish_url=e("TTSPIPE_FISH_URL", "http://127.0.0.1:8080"),
            mio_url=e("TTSPIPE_MIO_URL", "http://127.0.0.1:8081"),
        )
        for k, v in over.items():
            setattr(s, k, v)
        return s

    @property
    def models_dir(self) -> Path:
        return self.models_override or self.data_dir / "models"


# Process-wide settings used by modules that need paths (lexicon, clips, models). Set by main.create_app.
current: Settings = Settings.from_env()
