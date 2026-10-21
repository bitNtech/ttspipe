"""Every registered module: interface contract; every builtin: behaviour."""
import importlib
import sys

import numpy as np
import pytest

from app.core import registry
from app.core.base import (KINDS, Chunk, ChunkPolicy, Frame, LangSegment, ProsodyPlan, Voice, Word)
from app.core.config import validate_params
from app.modules.chunker.punct import PunctChunker
from app.modules.g2p.lexicon import LexiconG2P, translit_word
from app.modules.ingest.plain import PlainIngest
from app.modules.normalize.rules import RulesNormalize
from app.modules.post.master import MasterPost
from app.modules.prosody.rules import RulesProsody
from app.modules.segment.script import ScriptSegment
from app.modules.tts.clips import ClipsTTS
from app.modules.tts.tone import ToneTTS

ALL = registry.all_modules()
HEAVY = {"torch", "transformers", "onnxruntime", "kokoro_onnx", "piper", "sherpa_onnx", "parler_tts", "f5_tts"}


@pytest.mark.parametrize("cls", ALL, ids=[c.info.id for c in ALL])
def test_module_contract(cls):
    info = cls.info
    assert info.kind in KINDS
    assert info.id == f"{info.kind}/{info.impl}"
    assert info.name and isinstance(info.params_schema, dict)
    for name, s in info.params_schema.items():
        assert s["type"] in ("number", "integer", "string", "boolean", "array"), name
        assert "default" in s, name
    # defaults validate against their own schema
    assert validate_params(info.params_schema, info.default_params, "p") == [] or \
        all(info.params_schema[k].get("nullable") for k in info.default_params if info.default_params[k] is None)
    ok, reason = cls.availability()
    assert isinstance(ok, bool) and (ok or reason)
    if info.builtin:
        assert ok and info.requires == []
    if info.kind == "tts":
        assert info.native_rate or "router" in info.tags or info.id == "tts/clips"
    inst = cls({})  # constructor must be cheap and import-free
    assert isinstance(inst.model_files(), list)


def test_module_files_do_not_import_heavy_libs():
    before = set(sys.modules)
    for cls in ALL:
        importlib.reload(sys.modules[cls.__module__])
    newly = {m.split(".")[0] for m in set(sys.modules) - before}
    assert not (newly & HEAVY), newly & HEAVY


def test_builtin_fallback_for_each_kind_exists():
    for kind, impl in registry.BUILTIN_FALLBACK.items():
        assert registry.get(f"{kind}/{impl}").info.builtin


# ---- ingest -----------------------------------------------------------------------------------


def feed_all(ing, deltas):
    st = ing.new_state()
    spans = []
    for d in deltas:
        spans += ing.feed(st, d)
    spans += ing.flush(st)
    return spans


def test_ingest_holds_partial_words():
    ing = PlainIngest()
    st = ing.new_state()
    assert ing.feed(st, "Hel") == []
    out = ing.feed(st, "lo wor")
    assert "".join(s.text for s in out).strip() == "Hello"
    out = ing.flush(st)
    assert "wor" in "".join(s.text for s in out)


def test_ingest_strips_markdown_emoji_urls_code():
    ing = PlainIngest()
    text = "**Sure!** Visit [our site](https://x.com/a) or https://y.org now 🚚\n```python\nprint(1)\n```\n- item `one`\n"
    deltas = [text[i:i + 3] for i in range(0, len(text), 3)]
    out = " ".join(" ".join(s.text for s in feed_all(ing, deltas)).split())
    assert "**" not in out and "http" not in out and "🚚" not in out and "print" not in out
    assert "Sure!" in out and "our site" in out and "item one" in out


def test_ingest_ssml_lite():
    ing = PlainIngest()
    spans = feed_all(ing, ['Hi <break time="500ms"/> there <emphasis>now</emphasis> <prosody rate="0.8">slow</prosody>'])
    kinds = [s.kind for s in spans]
    assert "break" in kinds and [s for s in spans if s.kind == "break"][0].break_ms == 500
    assert any(s.emphasis and "now" in s.text for s in spans)
    assert any(s.rate == 0.8 and "slow" in s.text for s in spans)


def test_ingest_keeps_rupee_and_degree():
    out = " ".join(s.text for s in feed_all(PlainIngest(), ["Pay ₹500 at 30°C "]))
    assert "₹500" in out and "30°C" in out


# ---- normalize --------------------------------------------------------------------------------


@pytest.mark.parametrize("src,expect", [
    ("Pay ₹1,50,000 now", "Pay one lakh fifty thousand rupees now"),
    ("on 12/03/2024 ok", "on twelfth March, twenty twenty-four ok"),
    ("at 10:30 AM.", "at ten thirty a m."),
    ("call +91 98765 43210", "call plus nine one, nine eight seven six five, four three two one zero"),
    ("25% off", "twenty-five percent off"),
    ("Dr. Rao", "Doctor Rao"),
    ("the 21st", "the twenty-first"),
    ("costs $5.25", "costs five dollars twenty-five cents"),
    ("₹500 கொடுங்க", "ஐந்நூறு ரூபாய் கொடுங்க"),
    ("நாளை 10:30 மணிக்கு", "நாளை பத்து முப்பது மணிக்கு"),
    ("2வது இடம்", "இரண்டாவது இடம்"),
    ("4G phone", "four G phone"),
    ("5 km away", "five kilometres away"),
])
def test_normalize_rules(src, expect):
    assert RulesNormalize().normalize(src).text == expect


def test_normalize_word_map_groups():
    r = RulesNormalize().normalize("Pay ₹500 today")
    assert [w.text for w in r.words] == ["Pay", "five", "hundred", "rupees", "today"]
    assert [w.raw for w in r.words] == ["Pay", "₹500", "", "", "today"]
    assert r.words[1].group == r.words[3].group != r.words[0].group
    assert r.changed and r.edits == [{"before": "₹500", "after": "five hundred rupees"}]


def test_normalize_unchanged_text():
    r = RulesNormalize().normalize("hello world")
    assert not r.changed and r.edits == []


# ---- segment / g2p ------------------------------------------------------------------------------


def test_segment_scripts_and_tanglish():
    seg = ScriptSegment()
    s = seg.split("நாளைக்கு meeting இருக்கு, please வாங்க")
    assert [x.lang for x in s] == ["ta", "en", "ta", "en", "ta"]
    s = seg.split("Naalaikku meeting irukku, please vandhudunga.")
    assert [x.lang for x in s][0] == "ta-Latn" and "en" in [x.lang for x in s]
    assert " ".join(x.text for x in s) == "Naalaikku meeting irukku, please vandhudunga."
    assert [x.lang for x in seg.split("Hello! Your order has shipped.")] == ["en"]


def test_segment_never_changes_text():
    seg = ScriptSegment()
    for t in ["a  b", "₹ 500 கொடு", "12 , 13", "ok"]:
        assert " ".join(x.text for x in seg.split(t)) == " ".join(t.split())


def test_transliteration():
    assert translit_word("enna") == "என்ன"
    assert translit_word("romba") == "ரொம்ப"
    assert translit_word("vaanga") == "வாங்க"
    assert translit_word("irukku") == "இருக்கு"


def test_g2p_lexicon_and_translit():
    words = [Word("SBI", "SBI", 0, "en"), Word("romba", "romba", 1, "ta-Latn")]
    c = Chunk("u.0", "u", 0, words, [LangSegment("SBI", "en", [words[0]]), LangSegment("romba", "ta-Latn", [words[1]])])
    out = LexiconG2P().apply(c)
    assert out.tts_text == "S B I ரொம்ப"
    assert out.norm_text == "SBI romba"  # words untouched


# ---- chunker / prosody --------------------------------------------------------------------------


def words_of(text, lang="en"):
    return [Word(w, w, i, lang) for i, w in enumerate(text.split())]


def seg_of(text):
    ws = words_of(text)
    return [LangSegment(text, "en", ws)]


def test_chunker_first_chunk_small_then_sentences():
    ch = PunctChunker()
    st = ch.new_state()
    p = ChunkPolicy(first_chunk_min_words=4, min_chunk_chars=40)
    out = ch.push(st, seg_of("Hello there my good friend, how are you doing today? I am fine."), p, 0)
    assert [c.reason for c in out] == ["first", "punct", "punct"]
    assert out[0].norm_text == "Hello there my good friend,"
    assert out[0].is_first and not out[1].is_first
    assert ch.flush(st, p) == []


def test_chunker_first_chunk_timeout_and_max_words():
    ch = PunctChunker()
    p = ChunkPolicy(first_chunk_min_words=4, first_chunk_max_words=6, first_chunk_max_wait_ms=300)
    st = ch.new_state()
    assert ch.push(st, seg_of("Well um"), p, 10) == []
    out = ch.push(st, [], p, 350)
    assert len(out) == 1 and out[0].reason == "first"
    st = ch.new_state()
    out = ch.push(st, seg_of("one two three four five six seven"), p, 0)
    assert out[0].reason == "first" and len(out[0].words) == 6


def test_chunker_max_len_and_end_and_break():
    ch = PunctChunker()
    p = ChunkPolicy(first_chunk_min_words=1, max_chunk_chars=30)
    st = ch.new_state()
    out = ch.push(st, seg_of("Hi, " + "word " * 20), p, 0)
    assert out[0].reason == "first" and any(c.reason == "max_len" for c in out[1:])
    rest = ch.flush(st, p)
    assert rest and rest[-1].reason == "end" and rest[-1].is_last
    st = ch.new_state()
    st.first_done = True
    out = ch.push(st, seg_of("wait") + [LangSegment("", "en", [], attrs={"break_ms": 700})], p, 0)
    assert out[-1].pause_after_ms == 700


def test_prosody_rules():
    pr = RulesProsody()
    c = Chunk("u.0", "u", 0, words_of("Are you there?"), [], is_first=True)
    plan = pr.plan(c, {})
    assert plan.style == "question" and plan.pitch_st > 0 and plan.pause_after_ms > 0
    c = Chunk("u.1", "u", 1, words_of("This is IMPORTANT."), [], is_last=True)
    plan = pr.plan(c, {})
    assert plan.pause_after_ms == 0 and 2 in plan.emphasis


# ---- tts ----------------------------------------------------------------------------------------


def test_tone_streams_words_with_exact_timings():
    tts = ToneTTS({"latency_ms": 0, "rtf": 0})
    c = Chunk("u.0", "u", 0, words_of("hello big world"), [])
    pieces = list(tts.synthesize(None, c, ProsodyPlan(), Voice("tone", "tone-f")))
    assert len(pieces) == 3 and pieces[-1].is_last
    tim = [t for p in pieces for t in p.word_timings]
    assert [t.word for t in tim] == ["hello", "big", "world"]
    total = sum(len(p.pcm) for p in pieces) * 1000 / pieces[0].sample_rate
    assert tim[-1].end_ms == pytest.approx(total, abs=1)
    assert all(a.end_ms <= b.start_ms for a, b in zip(tim, tim[1:]))
    # deterministic
    again = list(tts.synthesize(None, c, ProsodyPlan(), Voice("tone", "tone-f")))
    assert np.array_equal(pieces[0].pcm, again[0].pcm)


def test_tone_speed_changes_duration():
    c = Chunk("u.0", "u", 0, words_of("hello world"), [])
    tts = ToneTTS({"latency_ms": 0, "rtf": 0})
    n1 = sum(len(p.pcm) for p in tts.synthesize(None, c, ProsodyPlan(), Voice("tone", "tone-f", speed=1.0)))
    n2 = sum(len(p.pcm) for p in tts.synthesize(None, c, ProsodyPlan(), Voice("tone", "tone-f", speed=2.0)))
    assert n2 < n1 * 0.6


def test_clips_exact_match_and_fallback(tmp_path):
    from app.core.audio import wav_bytes
    (tmp_path / "hi.wav").write_bytes(wav_bytes(np.ones(1600, np.float32) * 0.1, 16000))
    (tmp_path / "hi.txt").write_text("Hello there!", encoding="utf-8")
    tts = ClipsTTS({"clips_dir": str(tmp_path)})
    tts.load()
    c = Chunk("u.0", "u", 0, words_of("hello there"), [])
    p = list(tts.synthesize(None, c, ProsodyPlan(), Voice("clips", "tone-f")))
    assert len(p) == 1 and p[0].sample_rate == 16000 and len(p[0].pcm) == 1600
    c2 = Chunk("u.1", "u", 1, words_of("something else"), [])
    p2 = list(tts.synthesize(None, c2, ProsodyPlan(), Voice("clips", "tone-f")))
    assert sum(len(x.pcm) for x in p2) > 0


# ---- post ---------------------------------------------------------------------------------------


def frame(x, **kw):
    return Frame(pcm=np.asarray(x, np.float32), rate=24000, **kw)


def test_master_trims_leading_silence_only_when_trimmable():
    m = MasterPost()
    st = m.new_state()
    assert m.process(st, frame(np.zeros(480), first_of_chunk=True, trimmable=True)) is None
    assert m.process(st, frame(np.zeros(480), kind="pause")) is not None
    loud = 0.3 * np.sin(np.arange(480) / 3)
    assert m.process(st, frame(loud, trimmable=True)) is not None


def test_master_limiter_and_loudness():
    m = MasterPost({"target_lufs": -18})
    st = m.new_state()
    peak = 0
    for i in range(100):
        out = m.process(st, frame(0.99 * np.sin(np.arange(480) / 2 + i), trimmable=True))
        peak = max(peak, np.abs(out.pcm).max())
    assert peak <= 10 ** (-1 / 20) + 1e-6
    quiet_st = m.new_state()
    outs = [m.process(quiet_st, frame(0.01 * np.sin(np.arange(480) / 2 + i), trimmable=True)) for i in range(100)]
    assert np.sqrt(np.mean(outs[-1].pcm ** 2)) > 0.01 * 2  # gain was raised


def test_master_fade_in_at_chunk_start():
    m = MasterPost({"loudness": False, "dc_block": False})
    st = m.new_state()
    out = m.process(st, frame(np.ones(480) * 0.5, first_of_chunk=True, trimmable=True, fade_ms=8))
    assert out.pcm[0] == 0 and out.pcm[-1] == pytest.approx(0.5, abs=1e-3)


# ---- router -------------------------------------------------------------------------------------


def router_chunk(spec):
    words, segs, g = [], [], 0
    for text, lang in spec:
        ws = []
        for w in text.split():
            ws.append(Word(w, w, g, lang))
            g += 1
        words += ws
        segs.append(LangSegment(text, lang, ws))
    return Chunk("u.0", "u", 0, words, segs)


def test_router_runs_merge_short_runs():
    from app.modules.tts.lang_router import LangRouterTTS
    r = LangRouterTTS({"min_run_words": 2})
    c = router_chunk([("நாளைக்கு", "ta"), ("meeting", "en"), ("இருக்கு, வாங்க", "ta"), ("please come on time", "en")])
    runs = r.runs(c)
    assert [l for l, _ in runs] == ["ta", "en"]  # the 1-word English run merged into Tamil
    assert " ".join(s.text for s in runs[0][1]) == "நாளைக்கு meeting இருக்கு, வாங்க"
    r2 = LangRouterTTS({"ta_latn_to": "en", "min_run_words": 1})
    assert [l for l, _ in r2.runs(router_chunk([("enna", "ta-Latn"), ("hello", "en")]))] == ["en"]


def test_router_synthesizes_with_offset_timings():
    from app.modules.tts.lang_router import LangRouterTTS
    r = LangRouterTTS({"min_run_words": 1})
    r.load()
    st = r.new_state()
    c = router_chunk([("வணக்கம் நண்பா", "ta"), ("hello there", "en")])
    pieces = list(r.synthesize(st, c, ProsodyPlan(), Voice("lang_router", "")))
    tims = [t for p in pieces for t in (p.word_timings or [])]
    assert [t.word for t in tims] == ["வணக்கம்", "நண்பா", "hello", "there"]
    assert all(a.end_ms <= b.start_ms + 0.1 for a, b in zip(tims, tims[1:]))
    assert pieces[-1].is_last and not any(p.is_last for p in pieces[:-1])
    total = sum(len(p.pcm) / p.sample_rate * 1000 for p in pieces)
    assert tims[-1].end_ms == pytest.approx(total, abs=1)
