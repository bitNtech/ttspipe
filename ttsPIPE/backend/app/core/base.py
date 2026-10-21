"""Data types and the module interface (ARCHITECTURE.md §4.2, §6)."""
from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Iterator

import numpy as np

KINDS = ("ingest", "normalize", "segment", "chunker", "g2p", "prosody", "tts", "post")
LANGS = ("ta", "en", "ta-Latn")


@dataclass
class VoiceInfo:
    id: str
    name: str
    language: str = "multi"  # ta | en | multi
    gender: str = "neutral"
    sample: str | None = None


@dataclass
class ModelFile:
    name: str
    url: str
    path: str  # relative to data/models
    size_mb: float = 0.0
    sha256: str | None = None
    present: bool = False


@dataclass
class ModuleInfo:
    id: str
    kind: str
    name: str
    description: str = ""
    model_id: str | None = None
    size_mb: float = 0.0
    ram_mb: float = 0.0
    license: str = "builtin"
    streaming: bool = False
    requires: list[str] = field(default_factory=list)
    params_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    voices: list[VoiceInfo] = field(default_factory=list)
    builtin: bool = False
    native_rate: int | None = None
    family: str = ""  # ML dependency family → backend/requirements/ml-<family>.txt ("" = core only)

    @property
    def impl(self) -> str:
        return self.id.split("/", 1)[1]

    @property
    def default_params(self) -> dict:
        return {k: v.get("default") for k, v in self.params_schema.items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["default_params"] = self.default_params
        return d


# ---- pipeline data -------------------------------------------------------------------------


@dataclass
class TextSpan:
    text: str
    kind: str = "text"  # text | break
    break_ms: int = 0
    emphasis: bool = False
    lang_hint: str | None = None
    rate: float | None = None
    pitch_st: float | None = None


@dataclass
class Word:
    text: str
    raw: str = ""
    group: int = 0
    lang: str = ""


@dataclass
class NormResult:
    text: str
    changed: bool = False
    edits: list[dict] = field(default_factory=list)
    words: list[Word] = field(default_factory=list)


@dataclass
class LangSegment:
    text: str
    lang: str
    words: list[Word] = field(default_factory=list)
    tts_text: str = ""
    attrs: dict = field(default_factory=dict)  # SSML-lite: emphasis, rate, pitch_st, break_ms

    def __post_init__(self):
        if not self.tts_text:
            self.tts_text = self.text


@dataclass
class Chunk:
    chunk_id: str
    utterance_id: str
    index: int
    words: list[Word]
    segments: list[LangSegment]
    reason: str = "punct"  # first | punct | max_len | end
    is_first: bool = False
    is_last: bool = False
    pause_after_ms: int = 0
    phonemes: str | None = None
    emphasis: bool = False
    rate: float | None = None
    pitch_st: float | None = None
    norm_text: str = ""
    tts_text: str = ""
    lang: str = ""

    def __post_init__(self):
        if not self.norm_text:
            self.norm_text = " ".join(w.text for w in self.words)
        if not self.tts_text:
            self.tts_text = self.norm_text
        if not self.lang:
            self.lang = dominant_lang(self.words)

    @property
    def text(self) -> str:
        return " ".join(w.raw for w in self.words if w.raw)


def dominant_lang(words: list[Word]) -> str:
    counts: dict[str, int] = {}
    for w in words:
        if w.lang:
            counts[w.lang] = counts.get(w.lang, 0) + 1
    return max(counts, key=counts.get) if counts else "en"


@dataclass
class WordTiming:
    word: str
    start_ms: float
    end_ms: float


@dataclass
class AudioPiece:
    pcm: np.ndarray
    sample_rate: int
    word_timings: list[WordTiming] | None = None
    is_last: bool = False


@dataclass
class ProsodyPlan:
    speed: float = 1.0
    pitch_st: float = 0.0
    pause_after_ms: int = 0
    emphasis: list[int] = field(default_factory=list)
    style: str = "neutral"


@dataclass
class Voice:
    impl: str
    voice_id: str = ""
    speed: float = 1.0
    pitch: float = 0.0


@dataclass
class Frame:
    pcm: np.ndarray
    utterance_id: str | None = None
    chunk_id: str | None = None
    first_of_chunk: bool = False
    last_of_chunk: bool = False
    trimmable: bool = False
    kind: str = "speech"  # speech | pause | silence
    rate: int = 24000
    fade_ms: int = 0  # de-click fade the post stage applies at chunk joins (playback.crossfade_ms)


@dataclass
class ChunkPolicy:
    first_chunk_min_words: int = 4
    first_chunk_max_words: int = 10
    first_chunk_max_wait_ms: int = 400
    min_chunk_chars: int = 40
    max_chunk_chars: int = 220
    nonstreaming_max_chunk_chars: int = 140


# ---- module interface ----------------------------------------------------------------------


class BaseModule:
    info: ClassVar[ModuleInfo]

    def __init__(self, params: dict | None = None):
        self.params = {**self.info.default_params, **(params or {})}
        self.loaded = False
        self.models_dir = None  # set by the manager: Path to data/models/<kind>/<impl>

    @classmethod
    def availability(cls) -> tuple[bool, str | None]:
        missing = [r for r in cls.info.requires if importlib.util.find_spec(r.split(".")[0]) is None]
        if missing:
            return False, f"missing python package(s): {', '.join(missing)}"
        return True, None

    def model_files(self) -> list[ModelFile]:
        return []

    def load(self) -> None:
        self.loaded = True

    def unload(self) -> None:
        self.loaded = False

    def new_state(self) -> Any:
        return None

    def close_state(self, state: Any) -> None:
        pass

    def update_params(self, params: dict) -> bool:
        self.params = {**self.info.default_params, **params}
        return True


# Typed stage bases: document the method each kind implements.


class IngestModule(BaseModule):
    def feed(self, state, delta: str) -> list[TextSpan]:
        raise NotImplementedError

    def flush(self, state) -> list[TextSpan]:
        return []


class NormalizeModule(BaseModule):
    def normalize(self, text: str, *, language: str = "auto") -> NormResult:
        raise NotImplementedError


class SegmentModule(BaseModule):
    def split(self, text: str) -> list[LangSegment]:
        raise NotImplementedError


class ChunkerModule(BaseModule):
    def push(self, state, segments: list[LangSegment], policy: ChunkPolicy, now_ms: float) -> list[Chunk]:
        raise NotImplementedError

    def flush(self, state, policy: ChunkPolicy) -> list[Chunk]:
        raise NotImplementedError


class G2PModule(BaseModule):
    def apply(self, chunk: Chunk) -> Chunk:
        raise NotImplementedError


class ProsodyModule(BaseModule):
    def plan(self, chunk: Chunk, ctx: dict) -> ProsodyPlan:
        raise NotImplementedError


class TTSModule(BaseModule):
    @classmethod
    def list_voices(cls) -> list[VoiceInfo]:
        """Voices selectable right now (zero-shot engines add the cloned-voice library)."""
        return cls.info.voices

    def synthesize(self, state, chunk: Chunk, plan: ProsodyPlan, voice: Voice) -> Iterator[AudioPiece]:
        raise NotImplementedError


class PostModule(BaseModule):
    def process(self, state, frame: Frame) -> Frame | None:
        return frame


def schema(**params) -> dict:
    """Small helper: schema(speed=("number", 1.0, {"min":0.5,"max":2})) -> params_schema dict."""
    out = {}
    for name, spec in params.items():
        typ, default, *rest = spec
        extra = rest[0] if rest else {}
        out[name] = {"type": typ, "default": default, **extra}
    return out
