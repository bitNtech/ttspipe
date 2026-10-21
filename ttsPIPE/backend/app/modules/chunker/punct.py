"""chunker/punct: small first chunk for fast first audio, then sentence/clause chunks (§4.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from app.core.base import Chunk, ChunkerModule, ChunkPolicy, LangSegment, ModuleInfo, Word, schema
from app.core.registry import register

SENT_END = set(".!?।…")
CLAUSE_END = set(",;:—–")
CLOSERS = "\"'”’)]»"


def end_char(word: str) -> str:
    w = word.rstrip(CLOSERS)
    return w[-1] if w else ""


def segments_of(words: list[Word]) -> list[LangSegment]:
    segs: list[LangSegment] = []
    for w in words:
        if segs and segs[-1].lang == w.lang:
            segs[-1].words.append(w)
            segs[-1].text += " " + w.text
        else:
            segs.append(LangSegment(text=w.text, lang=w.lang or "en", words=[w]))
    for s in segs:
        s.tts_text = s.text
    return segs


@dataclass
class ChunkerState:
    utterance_id: str = "u0"
    words: list[Word] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    index: int = 0
    first_done: bool = False


@register
class PunctChunker(ChunkerModule):
    info = ModuleInfo(
        id="chunker/punct", kind="chunker", name="Punctuation chunker",
        description="First chunk flushes at ≥ N words + clause end (or max words / max wait); later chunks at sentence or clause ends.",
        builtin=True, tags=["builtin"],
        params_schema=schema(
            clause_flush=("boolean", True, {"description": "Allow flushing at clause punctuation after min_chunk_chars"}),
        ),
    )

    def new_state(self):
        return ChunkerState()

    def _chars(self, words: list[Word]) -> int:
        return sum(len(w.text) + 1 for w in words)

    def _make(self, st: ChunkerState, reason: str, pause_ms: int = 0) -> Chunk:
        words, st.words = st.words, []
        c = Chunk(chunk_id=f"{st.utterance_id}.{st.index}", utterance_id=st.utterance_id, index=st.index,
                  words=words, segments=segments_of(words), reason=reason, is_first=st.index == 0,
                  pause_after_ms=pause_ms, emphasis=bool(st.attrs.get("emphasis")),
                  rate=st.attrs.get("rate"), pitch_st=st.attrs.get("pitch_st"))
        st.index += 1
        st.first_done = True
        st.attrs = {}
        return c

    def _check(self, st: ChunkerState, p: ChunkPolicy) -> str | None:
        if not st.words:
            return None
        last = end_char(st.words[-1].text)
        n = len(st.words)
        if not st.first_done:
            if n >= p.first_chunk_min_words and (last in SENT_END or last in CLAUSE_END):
                return "first"
            if n >= p.first_chunk_max_words:
                return "first"
            return None
        if last in SENT_END:
            return "punct"
        chars = self._chars(st.words)
        if self.params["clause_flush"] and last in CLAUSE_END and chars >= p.min_chunk_chars:
            return "punct"
        if chars >= p.max_chunk_chars:
            return "max_len"
        return None

    def push(self, state: ChunkerState, segments: list[LangSegment], policy: ChunkPolicy, now_ms: float) -> list[Chunk]:
        out: list[Chunk] = []
        for seg in segments:
            brk = int(seg.attrs.get("break_ms", 0)) if seg.attrs else 0
            if brk and not seg.words:
                # an SSML break: flush what we have with the pause, or emit a pause-only chunk
                out.append(self._make(state, "punct", brk))
                continue
            if seg.attrs and not state.words:
                state.attrs = {k: v for k, v in seg.attrs.items() if k != "break_ms"}
            for w in seg.words:
                state.words.append(w)
                reason = self._check(state, policy)
                if reason:
                    out.append(self._make(state, reason))
        if state.words and not state.first_done and now_ms >= policy.first_chunk_max_wait_ms:
            out.append(self._make(state, "first"))
        return out

    def flush(self, state: ChunkerState, policy: ChunkPolicy) -> list[Chunk]:
        if not state.words:
            return []
        c = self._make(state, "end")
        c.is_last = True
        return [c]
