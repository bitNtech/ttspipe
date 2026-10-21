"""ML TTS modules: run only when their family is installed and model files are present (§15)."""
import numpy as np
import pytest

from app.core import registry
from app.core.base import Chunk, LangSegment, ProsodyPlan, Voice, Word
from app.settings import Settings

ML = [c for c in registry.all_modules() if c.info.kind == "tts" and c.info.family]
TEXT = {"ta": "வணக்கம், எப்படி இருக்கீங்க?", "en": "Hello, how are you today?"}


def chunk(text, lang):
    ws = [Word(w, w, i, lang) for i, w in enumerate(text.split())]
    return Chunk("u.0", "u", 0, ws, [LangSegment(text, lang, ws)])


def ready(cls):
    ok, reason = cls.availability()
    if not ok:
        pytest.skip(reason)
    if "remote" in cls.info.tags:
        from app import settings
        from app.core.fish_server import SERVER

        settings.current = Settings.from_env()  # API tests point it at a dead port

        if SERVER.health()[0] != "ok":
            pytest.skip(f"{cls.info.id}: server not running at {SERVER.url}")
    inst = cls({})
    inst.models_dir = Settings.from_env().models_dir / cls.info.kind / cls.info.impl
    missing = [f.name for f in inst.model_files() if not (Settings.from_env().models_dir / f.path).exists()]
    if missing:
        pytest.skip(f"model files missing: {missing}")
    return inst


@pytest.fixture(scope="module")
def loaded():
    cache = {}

    def get(cls):
        if cls.info.id not in cache:
            inst = ready(cls)
            inst.load()
            cache[cls.info.id] = inst
        return cache[cls.info.id]

    yield get
    for m in cache.values():
        m.unload()


@pytest.mark.parametrize("cls", ML, ids=[c.info.id for c in ML])
def test_ml_tts_speaks_its_languages(cls, loaded):
    mod = loaded(cls)
    for lang in [l for l in ("ta", "en") if l in cls.info.languages]:
        voice = next((v.id for v in cls.info.voices if v.language in (lang, "multi")), cls.info.voices[0].id)
        pieces = list(mod.synthesize(mod.new_state(), chunk(TEXT[lang], lang), ProsodyPlan(), Voice(cls.info.impl, voice)))
        assert pieces and pieces[-1].is_last
        pcm = np.concatenate([p.pcm for p in pieces])
        rate = pieces[0].sample_rate
        dur = len(pcm) / rate
        assert rate == cls.info.native_rate
        assert 0.5 < dur < 6.0, f"{lang}: {dur:.2f}s"
        assert np.abs(pcm).max() > 0.05 and np.isfinite(pcm).all()


@pytest.mark.parametrize("cls", ML, ids=[c.info.id for c in ML])
def test_ml_tts_speed(cls, loaded):
    if "no-speed" in cls.info.tags:
        pytest.skip("engine has no speed control")
    mod = loaded(cls)
    lang = cls.info.languages[0]
    voice = next((v.id for v in cls.info.voices if v.language in (lang, "multi")), cls.info.voices[0].id)
    n = lambda s: sum(len(p.pcm) for p in mod.synthesize(mod.new_state(), chunk(TEXT["ta" if lang == "ta" else "en"], lang),
                                                       ProsodyPlan(), Voice(cls.info.impl, voice, speed=s)))
    assert n(1.5) < n(1.0) * 0.85
