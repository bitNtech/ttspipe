"""tts/lang_router: send each language run of a chunk to a Tamil or an English engine (§12.3)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from app.core import registry
from app.core.base import AudioPiece, Chunk, LangSegment, ModuleInfo, TTSModule, Voice, WordTiming, schema
from app.core.registry import register


@register
class LangRouterTTS(TTSModule):
    info = ModuleInfo(
        id="tts/lang_router", kind="tts", name="Language router",
        description="Splits each chunk into language runs and synthesizes Tamil runs with ta_impl and English runs "
                    "with en_impl. Runs shorter than min_run_words merge into their neighbour. Sub-engines load with the router.",
        builtin=True, tags=["builtin", "router"], streaming=True, languages=["ta", "en", "ta-Latn"],
        params_schema=schema(
            ta_impl=("string", "tone", {"description": "TTS module for Tamil runs"}),
            ta_voice=("string", "tone-m"),
            en_impl=("string", "tone", {"description": "TTS module for English runs"}),
            en_voice=("string", "tone-f"),
            ta_latn_to=("string", "ta", {"enum": ["ta", "en"], "description": "Engine for romanized Tamil"}),
            min_run_words=("integer", 2, {"min": 1, "max": 20, "description": "Shorter runs merge into a neighbour"}),
            ta_gain_db=("number", 0.0, {"min": -12, "max": 12, "unit": "dB", "description": "Loudness match"}),
            en_gain_db=("number", 0.0, {"min": -12, "max": 12, "unit": "dB"}),
        ),
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.subs: dict[str, TTSModule] = {}

    def model_files(self):
        out = []
        for lang in ("ta", "en"):
            try:
                cls = registry.get(f"tts/{self.params[f'{lang}_impl']}")
            except KeyError:
                continue
            inst = cls({})
            if self.models_dir is not None:
                inst.models_dir = self.models_dir.parent / cls.info.impl
            out += inst.model_files()
        return out

    def load(self) -> None:
        self.subs = {}
        built: dict[str, TTSModule] = {}
        for lang in ("ta", "en"):
            impl = self.params[f"{lang}_impl"]
            if impl == "lang_router":
                raise ValueError("router cannot route to itself")
            if impl not in built:
                cls = registry.get(f"tts/{impl}")
                ok, reason = cls.availability()
                if not ok:
                    raise RuntimeError(f"{lang}_impl {impl}: {reason}")
                mod = cls({})
                mod.models_dir = (self.models_dir.parent / impl) if self.models_dir is not None else None
                mod.load()
                built[impl] = mod
            self.subs[lang] = built[impl]
        self.loaded = True

    def unload(self) -> None:
        for m in {id(m): m for m in self.subs.values()}.values():
            m.unload()
        self.subs = {}
        self.loaded = False

    def update_params(self, params: dict) -> bool:
        return False  # sub-engines may change: reload

    def new_state(self):
        return {lang: m.new_state() for lang, m in self.subs.items()}

    def close_state(self, state) -> None:
        for lang, m in self.subs.items():
            m.close_state(state.get(lang))

    def runs(self, chunk: Chunk) -> list[tuple[str, list[LangSegment]]]:
        """Maximal language runs after ta-Latn mapping and short-run merging."""
        to = self.params["ta_latn_to"]
        runs: list[tuple[str, list[LangSegment]]] = []
        for seg in chunk.segments:
            lang = to if seg.lang == "ta-Latn" else ("ta" if seg.lang == "ta" else "en")
            if runs and runs[-1][0] == lang:
                runs[-1][1].append(seg)
            else:
                runs.append((lang, [seg]))
        dominant = "ta" if sum(len(s.words) or len(s.text.split()) for l, ss in runs if l == "ta" for s in ss) >= \
            sum(len(s.words) or len(s.text.split()) for l, ss in runs if l == "en" for s in ss) else "en"
        n = lambda ss: sum(len(s.text.split()) for s in ss)
        i = 0
        while i < len(runs) and len(runs) > 1:
            lang, ss = runs[i]
            cands = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
            if lang == "ta":  # Tamil script can only go to the Tamil engine
                cands = [j for j in cands if runs[j][0] == "ta"]
            if n(ss) >= self.params["min_run_words"] or not cands:
                i += 1
                continue
            j = next((j for j in cands if runs[j][0] == dominant), cands[0])
            tgt_lang, tgt = runs[j]
            runs[min(i, j)] = (tgt_lang, (tgt + ss) if j < i else (ss + tgt))
            del runs[max(i, j)]
            i = 0
        # adjacent runs that now share a language join up
        out: list[tuple[str, list[LangSegment]]] = []
        for lang, ss in runs:
            if out and out[-1][0] == lang:
                out[-1][1].extend(ss)
            else:
                out.append((lang, list(ss)))
        return out

    def synthesize(self, state, chunk, plan, voice):
        offset_ms = 0.0
        runs = self.runs(chunk)
        for ri, (lang, segs) in enumerate(runs):
            eng = self.subs[lang]
            # ta-Latn sent to the English engine keeps its Latin spelling, not the transliteration
            text = " ".join((s.text if (s.lang == "ta-Latn" and lang == "en") else s.tts_text) for s in segs)
            words = [w for s in segs for w in s.words]
            sub = replace(chunk, words=words, segments=segs, tts_text=text, norm_text=" ".join(s.text for s in segs),
                          lang=lang, is_last=chunk.is_last and ri == len(runs) - 1)
            v = Voice(impl=eng.info.impl, voice_id=self.params[f"{lang}_voice"], speed=voice.speed, pitch=voice.pitch)
            gain = 10 ** (self.params[f"{lang}_gain_db"] / 20)
            run_ms = 0.0
            for piece in eng.synthesize(state.get(lang) if isinstance(state, dict) else None, sub, plan, v):
                pcm = piece.pcm if gain == 1.0 else (np.asarray(piece.pcm, np.float32) * gain)
                dur = len(piece.pcm) * 1000.0 / piece.sample_rate
                if piece.word_timings:
                    tims = [WordTiming(t.word, t.start_ms + offset_ms, t.end_ms + offset_ms) for t in piece.word_timings]
                elif piece.is_last and run_ms == 0.0:
                    # whole run in one piece without timings: estimate so mixed engines still give timings
                    ws = text.split() or ["_"]
                    tot = sum(len(w) for w in ws) or 1
                    tims, t = [], offset_ms
                    for w in ws:
                        d = dur * len(w) / tot
                        tims.append(WordTiming(w, t, t + d))
                        t += d
                else:
                    tims = None
                run_ms += dur
                yield AudioPiece(pcm=pcm, sample_rate=piece.sample_rate, word_timings=tims,
                                 is_last=piece.is_last and ri == len(runs) - 1)
            offset_ms += run_ms
