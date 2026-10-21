"""tts/clips: pre-recorded phrase WAVs from data/clips/ (exact normalized-text match), else tts/tone."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml

from app import settings
from app.core.audio import read_wav
from app.core.base import AudioPiece, ModuleInfo, TTSModule, schema
from app.core.registry import register

from .tone import ToneTTS


def clip_key(text: str) -> str:
    return " ".join(re.sub(r"[^\w஀-௿\s]", " ", text.lower()).split())


@register
class ClipsTTS(TTSModule):
    info = ModuleInfo(
        id="tts/clips", kind="tts", name="Phrase clips",
        description="Plays pre-recorded WAVs for exact phrase matches. data/clips/index.yaml: [{text, file}] "
                    "or <name>.wav + <name>.txt pairs. Misses fall back to the tone engine.",
        builtin=True, tags=["builtin"], streaming=False, languages=["ta", "en", "ta-Latn"],
        voices=ToneTTS.info.voices,
        params_schema=schema(
            clips_dir=("string", "", {"description": "Folder of clips (default <data>/clips)"}),
            fallback=("string", "tone", {"enum": ["tone", "silence"]}),
        ),
    )

    def __init__(self, params=None):
        super().__init__(params)
        self.index: dict[str, Path] = {}
        self.tone = ToneTTS({"word_pieces": False})

    def _dir(self) -> Path:
        return Path(self.params["clips_dir"]) if self.params["clips_dir"] else settings.current.data_dir / "clips"

    def load(self) -> None:
        d = self._dir()
        self.index = {}
        if d.exists():
            idx = d / "index.yaml"
            if idx.exists():
                for row in yaml.safe_load(idx.read_text(encoding="utf-8")) or []:
                    self.index[clip_key(row["text"])] = d / row["file"]
            for wav in d.glob("*.wav"):
                txt = wav.with_suffix(".txt")
                if txt.exists():
                    self.index[clip_key(txt.read_text(encoding="utf-8"))] = wav
        self.loaded = True

    def synthesize(self, state, chunk, plan, voice):
        path = self.index.get(clip_key(chunk.norm_text)) or self.index.get(clip_key(chunk.text))
        if path and path.exists():
            pcm, rate = read_wav(path)
            yield AudioPiece(pcm=pcm, sample_rate=rate, word_timings=None, is_last=True)
            return
        if self.params["fallback"] == "tone":
            yield from self.tone.synthesize(state, chunk, plan, voice)
        else:
            yield AudioPiece(pcm=np.zeros(0, np.float32), sample_rate=24000, is_last=True)
