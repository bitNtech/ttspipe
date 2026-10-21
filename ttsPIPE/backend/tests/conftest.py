import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("PYTHONUTF8", "1")

from app.settings import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    """Repo defaults with the builtin tone engine, so tests run with core deps only and stay fast."""
    import yaml

    from app.core.config import load_yaml

    base = Settings.from_env()
    cfg = load_yaml(base.config)
    cfg["stages"]["tts"] = {"enabled": True, "impl": "tone", "params": {}}
    cfg["voice"] = {"impl": "tone", "voice_id": "tone-f", "speed": 1.0, "pitch": 0.0}
    cfg["playback"].update(jitter_ms=60, lookahead_chunks=2)  # the repo default is tuned for the GPU engine
    cfg["chunker"].update(first_chunk_min_words=4, max_chunk_chars=220)
    path = tmp_path / "pipeline.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return Settings.from_env(data_dir=tmp_path / "data", static_dir=tmp_path / "nostatic", log_level="WARNING",
                             config=path, fish_url="http://127.0.0.1:9")  # never reach a real Fish server


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings, background_load=False)
    with TestClient(app) as c:
        m = app.state.manager
        # fast tone voice so realtime tests finish quickly
        m.patch(["stages", "tts", "params"], {"syllable_ms": 60, "word_gap_ms": 20, "latency_ms": 5})
        m.wait_ready(10)
        c.manager = m
        yield c
