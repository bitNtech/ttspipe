"""post/master: DC removal, de-click fades at chunk joins, silence trim, loudness normalization, limiter."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from app.core.audio import db
from app.core.base import Frame, ModuleInfo, PostModule, schema
from app.core.registry import register


@dataclass
class MasterState:
    dc: float = 0.0
    leading: bool = True
    need_fade_in: bool = False
    silent_ms: float = 0.0
    window: deque = field(default_factory=lambda: deque(maxlen=20))  # 400 ms of speech mean-squares
    gain_db: float = 0.0
    lim: float = 1.0


@register
class MasterPost(PostModule):
    info = ModuleInfo(
        id="post/master", kind="post", name="Master (DC, fades, trim, loudness, limiter)",
        description="Per-frame mastering. Loudness is an approximate short-term LUFS (400 ms window, "
                    "speech-gated, no K-weighting); gain adapts over ~200 ms.",
        builtin=True, tags=["builtin"],
        params_schema=schema(
            dc_block=("boolean", True),
            trim=("boolean", True, {"description": "Drop leading silence and long internal silence of engine audio"}),
            trim_db=("number", -50.0, {"min": -90, "max": -20, "unit": "dBFS"}),
            max_internal_silence_ms=("integer", 300, {"min": 40, "max": 5000, "unit": "ms"}),
            loudness=("boolean", True),
            target_lufs=("number", -18.0, {"min": -40, "max": -6, "unit": "LUFS"}),
            max_gain_db=("number", 12.0, {"min": 0, "max": 30, "unit": "dB"}),
            limiter_db=("number", -1.0, {"min": -12, "max": 0, "unit": "dBFS"}),
        ),
    )

    def new_state(self):
        return MasterState()

    def process(self, state: MasterState, frame: Frame) -> Frame | None:
        p = self.params
        x = frame.pcm.astype(np.float32, copy=True)
        n = len(x)
        frame_ms = n * 1000.0 / frame.rate if n else 20.0
        if frame.first_of_chunk:
            state.leading, state.need_fade_in, state.silent_ms = True, True, 0.0
        if p["dc_block"] and n:
            state.dc = 0.98 * state.dc + 0.02 * float(x.mean())
            x -= state.dc
        level = db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))) if n else -120.0
        if frame.kind == "speech" and p["trim"] and frame.trimmable:
            if level < p["trim_db"]:
                state.silent_ms += frame_ms
                if state.leading or state.silent_ms > p["max_internal_silence_ms"]:
                    return None
            else:
                state.leading, state.silent_ms = False, 0.0
        fade = int(frame.fade_ms * frame.rate / 1000)
        if fade and n and frame.kind == "speech":
            f = min(fade, n)
            if state.need_fade_in:
                x[:f] *= np.linspace(0.0, 1.0, f, dtype=np.float32)
                state.need_fade_in = False
            if frame.last_of_chunk:
                x[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
        if p["loudness"] and n:
            if frame.kind == "speech" and level > -50:
                state.window.append(float(np.mean(x.astype(np.float64) ** 2)))
            if state.window:
                loud = 10 * np.log10(max(np.mean(state.window), 1e-12))
                target = float(np.clip(p["target_lufs"] - loud, -p["max_gain_db"], p["max_gain_db"]))
                prev = state.gain_db
                state.gain_db += (target - state.gain_db) * 0.1
                g = 10 ** (np.linspace(prev, state.gain_db, n) / 20)
                x *= g.astype(np.float32)
        if n:
            ceil = 10 ** (p["limiter_db"] / 20)
            peak = float(np.max(np.abs(x)))
            want = min(1.0, ceil / peak) if peak > 0 else 1.0
            new = want if want < state.lim else min(1.0, state.lim + 0.05)  # instant attack, slow release
            x *= np.linspace(state.lim, new, n, dtype=np.float32)
            state.lim = new
            np.clip(x, -ceil, ceil, out=x)
        frame.pcm = x
        return frame
