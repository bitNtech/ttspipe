"""Core units: resampler, number wording, config merge/validate, event bus back-pressure, metrics."""
import asyncio

import numpy as np
import pytest

from app.core import config as C
from app.core import textutil as T
from app.core.audio import StreamResampler, from_pcm16, read_wav, resample_all, to_pcm16, wav_bytes
from app.core.events import Bus
from app.core.metrics import Stat, pct


@pytest.mark.parametrize("a,b", [(22050, 24000), (16000, 24000), (44100, 24000), (24000, 16000),
                                 (24000, 48000), (24000, 22050), (24000, 24000)])
def test_resampler_frame_exact_streaming(a, b):
    x = np.random.default_rng(0).standard_normal(a).astype(np.float32) * 0.1
    r = StreamResampler(a, b)
    out = [r.process(x[i:i + 211]) for i in range(0, len(x), 211)] + [r.flush()]
    y = np.concatenate(out)
    assert len(y) == round(len(x) * b / a)
    np.testing.assert_allclose(y, resample_all(x, a, b), atol=1e-6)


def test_resampler_low_delay_and_accurate():
    r = StreamResampler(22050, 24000)
    assert r.half * 1000 / 22050 < 1.0  # < 1 ms of look-ahead
    t = np.arange(22050) / 22050
    y = resample_all(np.sin(2 * np.pi * 440 * t).astype(np.float32), 22050, 24000)
    ideal = np.sin(2 * np.pi * 440 * np.arange(len(y)) / 24000)
    assert np.abs(y[100:-100] - ideal[100:-100]).max() < 1e-3
    # first output appears after a single small input block
    r2 = StreamResampler(22050, 24000)
    assert len(r2.process(np.zeros(441, np.float32))) > 400


def test_pcm16_and_wav_roundtrip():
    x = np.linspace(-1, 1, 1000, dtype=np.float32)
    assert np.abs(from_pcm16(to_pcm16(x)) - x).max() < 1e-4
    y, rate = read_wav(wav_bytes(x, 16000))
    assert rate == 16000 and len(y) == 1000


def test_english_numbers():
    assert T.num_en(0) == "zero"
    assert T.num_en(21) == "twenty-one"
    assert T.num_en(150000) == "one lakh fifty thousand"
    assert T.num_en(1_000_000, indian=False) == "one million"
    assert T.num_en(105) == "one hundred and five"
    assert T.ord_en(21) == "twenty-first" and T.ord_en(12) == "twelfth" and T.ord_en(40) == "fortieth"
    assert T.year_en(2024) == "twenty twenty-four" and T.year_en(2005) == "two thousand five"
    assert T.year_en(1905) == "nineteen oh five"


def test_tamil_numbers():
    assert T.num_ta(5) == "ஐந்து"
    assert T.num_ta(12) == "பன்னிரண்டு"
    assert T.num_ta(25) == "இருபத்து ஐந்து"
    assert T.num_ta(100) == "நூறு"
    assert T.num_ta(125) == "நூற்று இருபத்து ஐந்து"
    assert T.num_ta(1000) == "ஆயிரம்"
    assert T.num_ta(2024) == "இரண்டாயிரத்து இருபத்து நான்கு"
    assert T.num_ta(150000) == "ஒரு லட்சத்து ஐம்பது ஆயிரம்"
    assert T.num_ta(10**7) == "ஒரு கோடி"
    assert T.ord_ta(2) == "இரண்டாவது" and T.ord_ta(1) == "முதலாவது"
    assert T.date_ord_ta(12) == "பன்னிரண்டாம்"


def test_syllables():
    assert T.syllables("hello") == 2
    assert T.syllables("வணக்கம்") == 3  # வ ண க்க ம் → வ, ண, க (க் and ம் are dead)
    assert T.syllables("...") == 0


def test_deep_merge_and_diff_roundtrip():
    base = {"a": {"b": 1, "c": 2}, "d": [1], "e": 1}
    cur = {"a": {"b": 1, "c": 3}, "d": [2], "e": 1}
    d = C.diff(base, cur)
    assert d == {"a": {"c": 3}, "d": [2]}
    assert C.deep_merge(base, d) == cur


def test_validate_params_errors():
    schema = {"speed": {"type": "number", "min": 0.5, "max": 2}, "mode": {"type": "string", "enum": ["a", "b"]},
              "n": {"type": "integer"}}
    errs = C.validate_params(schema, {"speed": 3, "mode": "z", "n": 1.5, "bogus": 1}, "stages.tts.params")
    assert "stages.tts.params.speed: 3 > max 2" in errs
    assert any("not in" in e for e in errs)
    assert "stages.tts.params.n: expected integer" in errs
    assert "stages.tts.params.bogus: unknown parameter" in errs


def test_load_effective_drops_invalid_override(tmp_path, settings):
    import yaml
    ov = tmp_path / "ov.yaml"
    ov.write_text(yaml.safe_dump({"playback": {"jitter_ms": -5, "duck_db": 6}, "audio": {"out_rate": 12345}}))
    defaults, eff, warns = C.load_effective(settings.config, ov)
    assert eff["playback"]["duck_db"] == 6  # valid key kept
    assert eff["playback"]["jitter_ms"] == defaults["playback"]["jitter_ms"]
    assert eff["audio"]["out_rate"] == defaults["audio"]["out_rate"]
    assert len(warns) >= 2


def test_validate_rejects_disabling_tts(settings):
    defaults, _, _ = C.load_effective(settings.config, settings.data_dir / "none.yaml")
    cfg = C.deep_merge(defaults, {"stages": {"tts": {"enabled": False}}})
    errors, _ = C.validate(cfg)
    assert any("cannot be disabled" in e for e in errors)


def test_bus_drops_oldest_noncritical_only():
    async def run():
        bus = Bus(queue_size=10)
        sub = bus.subscribe(asyncio.get_running_loop(), lambda ev: True)
        for i in range(5):
            bus.emit("utterance.end", {"i": i})
        for i in range(50):
            bus.emit("audio.level", {"i": i})
        types = [e["type"] for e in await sub.get_batch(100)]
        assert types.count("utterance.end") == 5  # critical never dropped
        assert len(types) == 10
        assert bus.dropped == 45
    asyncio.run(run())


def test_bus_critical_overflow_still_kept():
    async def run():
        bus = Bus(queue_size=3)
        sub = bus.subscribe(asyncio.get_running_loop(), lambda ev: True)
        for i in range(6):
            bus.emit("chunk.flushed", {"i": i})
        assert len(await sub.get_batch(100)) == 6
    asyncio.run(run())


def test_stats():
    s = Stat()
    for v in range(1, 101):
        s.add(v)
    sm = s.summary()
    assert sm["p50"] == 50.5 and sm["p95"] == pytest.approx(95.05) and sm["n"] == 100
    assert pct([], 0.5) is None
