"""tts/tone: deterministic syllable-rate tones with exact word timings — exercises the pipeline end to end."""
from __future__ import annotations

import math
import time
import zlib

import numpy as np

from app.core.base import AudioPiece, ModuleInfo, TTSModule, VoiceInfo, WordTiming, schema
from app.core.registry import register
from app.core.textutil import syllables

VOICES = {"tone-f": 220.0, "tone-m": 130.0, "tone-hi": 330.0}


@register
class ToneTTS(TTSModule):
    info = ModuleInfo(
        id="tts/tone", kind="tts", name="Tone (test engine)",
        description="Deterministic tones, one per syllable, streamed word by word with exact word timings. "
                    "latency_ms / rtf simulate engine compute so look-ahead, underruns and interruption can be tested.",
        builtin=True, tags=["builtin", "test"], streaming=True, native_rate=22050,
        languages=["ta", "en", "ta-Latn"],
        voices=[VoiceInfo("tone-f", "Tone F (220 Hz)", "multi", "female"),
                VoiceInfo("tone-m", "Tone M (130 Hz)", "multi", "male"),
                VoiceInfo("tone-hi", "Tone High (330 Hz)", "multi", "neutral")],
        params_schema=schema(
            sample_rate=("integer", 22050, {"enum": [16000, 22050, 24000, 44100, 48000]}),
            syllable_ms=("integer", 150, {"min": 40, "max": 600, "unit": "ms"}),
            word_gap_ms=("integer", 60, {"min": 0, "max": 500, "unit": "ms"}),
            latency_ms=("integer", 30, {"min": 0, "max": 5000, "unit": "ms", "description": "Simulated time to first audio"}),
            rtf=("number", 0.05, {"min": 0.0, "max": 3.0, "description": "Simulated compute per second of audio"}),
            leading_silence_ms=("integer", 0, {"min": 0, "max": 2000, "unit": "ms", "description": "Silence before speech (tests trimming)"}),
            amplitude=("number", 0.3, {"min": 0.01, "max": 1.0}),
            word_pieces=("boolean", True, {"description": "Stream one piece per word (false = whole chunk at once)"}),
        ),
    )

    def _tone(self, n: int, freq: float, rate: int, amp: float) -> np.ndarray:
        t = np.arange(n) / rate
        x = np.sin(2 * math.pi * freq * t) + 0.3 * np.sin(4 * math.pi * freq * t)
        env = np.ones(n)
        r = min(n // 2, int(0.008 * rate))
        if r:
            ramp = np.linspace(0, 1, r)
            env[:r], env[-r:] = ramp, ramp[::-1]
        return (amp / 1.3 * x * env).astype(np.float32)

    def synthesize(self, state, chunk, plan, voice):
        p = self.params
        rate = int(p["sample_rate"])
        speed = max(0.25, voice.speed * plan.speed)
        pitch = voice.pitch + plan.pitch_st
        base = VOICES.get(voice.voice_id, 220.0)
        words = chunk.tts_text.split()
        syl_n = int(p["syllable_ms"] / speed * rate / 1000)
        gap_n = int(p["word_gap_ms"] / speed * rate / 1000)
        t_ms = 0.0
        pending: list[np.ndarray] = []
        timings: list[WordTiming] = []
        lead = int(p["leading_silence_ms"] * rate / 1000)
        if lead:
            pending.append(np.zeros(lead, np.float32))
            t_ms += lead * 1000 / rate
        emph = set(plan.emphasis)
        first_piece = True
        for wi, w in enumerate(words):
            parts = []
            n_syl = max(1, syllables(w))
            for s in range(n_syl):
                h = zlib.crc32(f"{w}|{s}".encode()) % 7
                st = pitch + h - 3
                if plan.style == "question" and wi == len(words) - 1 and s == n_syl - 1:
                    st += 4
                parts.append(self._tone(syl_n, base * 2 ** (st / 12), rate, p["amplitude"] * (1.4 if wi in emph else 1.0)))
            word_pcm = np.concatenate(parts)
            timings.append(WordTiming(w, round(t_ms, 1), round(t_ms + len(word_pcm) * 1000 / rate, 1)))
            pending += [word_pcm, np.zeros(gap_n if wi < len(words) - 1 else 0, np.float32)]
            t_ms += (len(word_pcm) + (gap_n if wi < len(words) - 1 else 0)) * 1000 / rate
            last = wi == len(words) - 1
            if p["word_pieces"] or last:
                pcm = np.concatenate(pending)
                self._simulate(pcm, rate, first=first_piece)
                first_piece = False
                yield AudioPiece(pcm=pcm, sample_rate=rate, word_timings=timings, is_last=last)
                pending, timings = [], []
        if not words:
            yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=rate, word_timings=[], is_last=True)

    def _simulate(self, pcm: np.ndarray, rate: int, first: bool) -> None:
        delay = (self.params["latency_ms"] / 1000 if first else 0) + self.params["rtf"] * len(pcm) / rate
        if delay > 0:
            time.sleep(delay)
