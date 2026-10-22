"""Cut a stream of arbitrary text fragments into speakable segments.

Text arrives from the client one `text` event at a time, at whatever
granularity it likes -- a word, a sentence, a half-sentence. Handing the
whole buffered utterance to the engine only once the client says `stop`
would make time-to-first-audio equal to the length of the whole thing being
typed or generated upstream, which defeats the point of a streaming API. So
this module decides, incrementally and synchronously, the earliest point at
which "what's buffered so far" is worth sending to the engine as its own
segment.

Two competing goals: cut as soon as a sentence plausibly ends (so audio
starts quickly), but never cut somewhere that would butcher the utterance --
mid-word, mid-abbreviation ("Dr."), mid-decimal ("3.5"), or mid-tag
(`[laugh]`). `max_segment_chars` is a backstop for the case where no
sentence end ever shows up (a client streaming an unpunctuated wall of
text): we hard-cut at the last safe word/clause boundary instead of letting
a segment grow without bound.

Pure and synchronous by design -- no I/O, no asyncio -- so every rule above
is unit-testable without a running server.
"""

from __future__ import annotations

SENTENCE_END = ".!?。!?;:"
_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "approx",
    "fig", "no", "inc", "ltd", "co", "corp", "gen", "rev", "capt", "col",
    "cmdr", "lt", "sgt", "maj", "ave", "blvd", "etc", "jan", "feb", "mar",
    "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mt",
}
# How far back the hard-cut boundary search looks for whitespace before
# giving up and treating the language as space-free (CJK and similar).
_WORD_BOUNDARY_WINDOW = 48

# How much text may pile up after an unclosed `[` before we stop believing it
# is the start of a tag.
#
# Holding the cut back at an open bracket is right for `[laugh]`, which closes
# within a few characters. It is catastrophic for a bracket that never closes -
# "the array [0", a stray bracket, a tag an upstream LLM began and abandoned -
# because `_find_cut` scans and hard-cuts against that frozen limit, so no
# boundary past it can ever be found and *nothing* is emitted until `flush`.
# Measured before this bound: 950 characters buffered, zero segments. The whole
# turn goes silent. No real tag is anywhere near this long.
_OPEN_BRACKET_HOLD = 64


def _bracket_info(text: str) -> tuple[list[tuple[int, int]], int | None]:
    """Complete `[...]` spans, plus the start index of a still-open `[` if any."""
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for idx, ch in enumerate(text):
        if ch == "[":
            stack.append(idx)
        elif ch == "]" and stack:
            spans.append((stack.pop(), idx))
    open_pos = stack[0] if stack else None
    return spans, open_pos


def _adjust_cut(pos: int, spans: list[tuple[int, int]], open_pos: int | None) -> int:
    """Pull a candidate cut point back so it never lands inside a `[...]` tag."""
    if open_pos is not None and pos > open_pos:
        pos = open_pos
    for start, end in spans:
        if start < pos <= end:
            pos = start
            break
    return pos


def _is_valid_boundary(buf: str, i: int) -> bool:
    """Guard against abbreviations ("Dr.") and decimals ("3.5") at index i."""
    ch = buf[i]
    if ch == ".":
        prev = buf[i - 1] if i > 0 else ""
        nxt = buf[i + 1] if i + 1 < len(buf) else ""
        if prev.isdigit() and (nxt.isdigit() or nxt == ""):
            return False  # a decimal, or ambiguous ("3." at the edge of the buffer)
        j = i
        while j > 0 and buf[j - 1].isalpha():
            j -= 1
        token = buf[j:i].lower()
        if token and (token in _ABBREVIATIONS or len(token) == 1):
            return False  # "Dr." / "J." (initials)
    elif ch == ":":
        prev = buf[i - 1] if i > 0 else ""
        nxt = buf[i + 1] if i + 1 < len(buf) else ""
        if prev.isdigit() and nxt.isdigit():
            return False  # a ratio or a clock time, e.g. "3:5", "12:30"
    return True


#: Characters that legitimately follow a sentence's final punctuation and
#: belong to the sentence that is ending, not the one that starts next.
_TRAILING = "\"'\u201d\u2019)]\u00bb\u203a"


def _end_of_terminator(buf: str, i: int, limit: int) -> int:
    """Where the sentence really ends, given a terminator at `i`.

    Cutting at `i + 1` splits anything written with more than one closing
    character, and the leftovers open the *next* segment as text the model then
    has to pronounce: "..." becomes a cut after the first dot and a segment
    beginning ".."; "?!" leaves a bare "!"; `He said "Hello." Then...` cuts
    before the quotation mark and starts the next segment with a stray quote.
    Absorbing the whole run, plus any closing quote or bracket that follows it,
    keeps the punctuation with the sentence it punctuates.
    """
    end = i + 1
    while end < limit and buf[end] in SENTENCE_END:
        end += 1
    while end < limit and buf[end] in _TRAILING:
        end += 1
    return end


def _find_word_boundary(buf: str, target: int, limit: int) -> int:
    """Last whitespace-adjacent cut at or before `target`, else `target` itself."""
    hi = min(target, limit, len(buf))
    j = hi
    scanned = 0
    while j > 0 and not buf[j - 1].isspace():
        j -= 1
        scanned += 1
        if scanned > _WORD_BOUNDARY_WINDOW:
            return hi  # no boundary nearby -- treat as a space-free language
    if j > 0:
        return j
    return hi


class Segmenter:
    """Incremental, stateful text-to-segments cutter. One instance per session."""

    def __init__(
        self,
        min_segment_chars: int,
        max_segment_chars: int,
        first_min_chars: int | None = None,
    ):
        # A zero minimum lets `_find_cut` legitimately return 0 (a buffer
        # opening with `[`, or a `.` inside a bracket span), and `push` then
        # loops forever on a buffer it never shortens - synchronously, on the
        # event loop, so the whole connection goes deaf rather than just this
        # turn. One character is the smallest honest minimum.
        if min_segment_chars < 1 or max_segment_chars <= 0:
            raise ValueError("segment char bounds must be positive (min at least 1)")
        self.min_segment_chars = min_segment_chars
        self.max_segment_chars = max(max_segment_chars, min_segment_chars + 1)
        # The first segment of a reply sets its time to first audio, and a
        # reply that opens "Sure." would otherwise wait for the whole next
        # sentence to reach `min_segment_chars`. So until one segment has been
        # cut, any finished sentence of at least this many characters is.
        self.first_min_chars = max(1, first_min_chars) if first_min_chars else min_segment_chars
        self._first = True
        self._buf = ""

    @property
    def pending(self) -> str:
        """Text buffered so far but not yet cut into a segment."""
        return self._buf

    def push(self, text: str) -> list[str]:
        """Feed a fragment; return zero or more segments now ready to speak."""
        if not text:
            return []
        self._buf += text
        segments: list[str] = []
        while True:
            cut = self._find_cut(self._buf)
            if cut is None:
                break
            if cut <= 0:  # belt and braces: never loop on a buffer we don't shorten
                break
            segment, rest = self._buf[:cut].strip(), self._buf[cut:]
            self._buf = rest
            if segment:
                segments.append(segment)
                self._first = False
        return segments

    def flush(self) -> list[str]:
        """Drain whatever is buffered right now, as a single segment."""
        text = self._buf.strip()
        self._buf = ""
        return [text] if text else []

    def close(self) -> list[str]:
        """No more text is coming; same as `flush()`."""
        return self.flush()

    def restart(self) -> None:
        """Drop whatever is buffered and treat what comes next as a new reply
        (after a barge-in), first-segment rule included."""
        self._buf = ""
        self._first = True

    def _find_cut(self, buf: str) -> int | None:
        n = len(buf)
        spans, open_pos = _bracket_info(buf)
        if open_pos is not None and n - open_pos > _OPEN_BRACKET_HOLD:
            open_pos = None  # not a tag after all; stop holding the whole turn
        limit = open_pos if open_pos is not None else n

        i = 0
        while i < limit:
            ch = buf[i]
            if ch in SENTENCE_END or ch == "\n":
                if ch == "\n" or _is_valid_boundary(buf, i):
                    cut = _adjust_cut(_end_of_terminator(buf, i, limit), spans, open_pos)
                    if cut >= (self.first_min_chars if self._first else self.min_segment_chars):
                        return cut
            i += 1

        if limit >= self.max_segment_chars:
            raw = _find_word_boundary(buf, self.max_segment_chars, limit)
            cut = _adjust_cut(raw, spans, open_pos)
            if cut > 0:
                return cut

        return None
