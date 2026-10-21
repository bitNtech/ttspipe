"""ingest/plain: LLM token stream → clean speakable TextSpans (§4.1)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from app.core.base import IngestModule, ModuleInfo, TextSpan, schema
from app.core.registry import register

TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z]+)([^>]*?)(/?)\s*>")
ATTR_RE = re.compile(r'([\w:]+)\s*=\s*"([^"]*)"')
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.I)
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
SSML = {"break", "emphasis", "lang", "prosody", "speak", "s", "p"}
KEEP_SYMBOLS = set("°₹$%&@+=#/")


def _is_emoji(c: str) -> bool:
    o = ord(c)
    if c in KEEP_SYMBOLS:
        return False
    if 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or o in (0xFE0F, 0x200D, 0x20E3) or 0x1F3FB <= o <= 0x1F3FF:
        return True
    return unicodedata.category(c) == "So" and c != "°"


def _parse_ms(v: str) -> int:
    v = v.strip().lower()
    try:
        if v.endswith("ms"):
            return int(float(v[:-2]))
        if v.endswith("s"):
            return int(float(v[:-1]) * 1000)
        return int(float(v))
    except ValueError:
        return 300


@dataclass
class IngestState:
    buf: str = ""
    in_fence: bool = False
    at_line_start: bool = True
    stack: list = field(default_factory=list)  # [(tag, attrs)]


@register
class PlainIngest(IngestModule):
    info = ModuleInfo(
        id="ingest/plain", kind="ingest", name="Plain text + markdown strip",
        description="Strips markdown, emoji, code and URLs; holds partial words; parses SSML-lite tags.",
        builtin=True, tags=["builtin"],
        params_schema=schema(
            drop_code_blocks=("boolean", True, {"description": "Drop fenced code blocks entirely"}),
            strip_urls=("boolean", True),
            strip_emoji=("boolean", True),
            line_end_as_sentence=("boolean", True, {"description": "Add '.' to lines (headings, bullets) without end punctuation"}),
            default_break_ms=("integer", 300, {"min": 0, "max": 5000, "unit": "ms"}),
            max_hold_chars=("integer", 400, {"min": 50, "max": 5000, "description": "Release held text after this many chars"}),
        ),
    )

    def new_state(self):
        return IngestState()

    # ---- streaming boundary -------------------------------------------------------------------

    def _safe_cut(self, buf: str) -> int:
        ws = max(buf.rfind(" "), buf.rfind("\n"), buf.rfind("\t"))
        cut = ws + 1 if ws >= 0 else 0
        head = buf[:cut]
        # unclosed link / tag → stop before it
        lb = head.rfind("[")
        if lb >= 0 and not re.search(r"\]\([^)]*\)", head[lb:]):
            cut = min(cut, lb)
        lt = head.rfind("<")
        if lt >= 0 and ">" not in head[lt:]:
            cut = min(cut, lt)
        if cut == 0 and len(buf) > self.params["max_hold_chars"]:
            cut = len(buf)
        return cut

    def feed(self, state: IngestState, delta: str) -> list[TextSpan]:
        state.buf += delta
        cut = self._safe_cut(state.buf)
        if cut <= 0:
            return []
        text, state.buf = state.buf[:cut], state.buf[cut:]
        return self._process(state, text)

    def flush(self, state: IngestState) -> list[TextSpan]:
        text, state.buf = state.buf, ""
        spans = self._process(state, text + "\n") if text.strip() else []
        state.stack.clear()
        state.in_fence = False
        return spans

    # ---- cleaning -----------------------------------------------------------------------------

    def _clean_line(self, line: str) -> str:
        p = self.params
        s = line
        s = IMG_RE.sub("", s)
        s = LINK_RE.sub(r"\1", s)
        if p["strip_urls"]:
            s = URL_RE.sub("", s)
        s = s.replace("`", "")
        s = re.sub(r"\*{1,3}|(?<!\w)_{1,3}|_{1,3}(?!\w)|~~", "", s)
        if p["strip_emoji"]:
            s = "".join(c for c in s if not _is_emoji(c))
        return s

    def _line_prefix(self, line: str) -> str:
        line = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        line = re.sub(r"^\s*>\s?", "", line)
        line = re.sub(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+", "", line)
        if line.lstrip().startswith("|"):
            if re.fullmatch(r"\s*\|?[\s:|-]+\|?\s*", line):
                return ""
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            line = ", ".join(c for c in cells if c)
        return line

    def _process(self, state: IngestState, text: str) -> list[TextSpan]:
        out: list[TextSpan] = []
        lines = text.split("\n")
        for li, line in enumerate(lines):
            ends_line = li < len(lines) - 1
            if state.at_line_start and re.match(r"^\s*(```|~~~)", line):
                state.in_fence = not state.in_fence
                state.at_line_start = ends_line
                continue
            if state.in_fence and self.params["drop_code_blocks"]:
                if ends_line or line:
                    state.at_line_start = ends_line
                continue
            if state.at_line_start:
                line = self._line_prefix(line)
            self._emit_tagged(state, self._strip_html(line), out)
            if ends_line:
                state.at_line_start = True
                self._end_line(out)
            elif line:
                state.at_line_start = False
        return [s for s in out if s.kind == "break" or s.text.strip()]

    def _strip_html(self, line: str) -> str:
        def repl(m):
            return m.group(0) if m.group(2).lower() in SSML else ""
        return TAG_RE.sub(repl, line)

    def _end_line(self, out: list[TextSpan]) -> None:
        if not self.params["line_end_as_sentence"]:
            return
        for s in reversed(out):
            if s.kind != "text":
                return
            t = s.text.rstrip()
            if not t:
                continue
            if t[-1] not in ".!?।…:;,":
                s.text = t + ". "
            else:
                s.text = t + " "
            return

    def _attrs(self, state: IngestState) -> dict:
        a = {"emphasis": False, "lang_hint": None, "rate": None, "pitch_st": None}
        for tag, attrs in state.stack:
            if tag == "emphasis":
                a["emphasis"] = True
            elif tag == "lang":
                v = attrs.get("xml:lang") or attrs.get("lang")
                a["lang_hint"] = v.split("-")[0] if v else None
            elif tag == "prosody":
                if "rate" in attrs:
                    try:
                        r = attrs["rate"].strip()
                        a["rate"] = float(r[:-1]) / 100 if r.endswith("%") else float(r)
                    except ValueError:
                        pass
                if "pitch" in attrs:
                    m = re.match(r"([+-]?\d+(?:\.\d+)?)\s*st", attrs["pitch"])
                    if m:
                        a["pitch_st"] = float(m.group(1))
        return a

    def _emit_tagged(self, state: IngestState, line: str, out: list[TextSpan]) -> None:
        pos = 0
        for m in TAG_RE.finditer(line):
            self._emit_text(state, line[pos:m.start()], out)
            pos = m.end()
            closing, tag, rest, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
            attrs = dict(ATTR_RE.findall(rest))
            if tag == "break":
                ms = _parse_ms(attrs["time"]) if "time" in attrs else self.params["default_break_ms"]
                out.append(TextSpan(text="", kind="break", break_ms=ms))
            elif tag in ("emphasis", "lang", "prosody"):
                if closing:
                    for i in range(len(state.stack) - 1, -1, -1):
                        if state.stack[i][0] == tag:
                            del state.stack[i]
                            break
                elif not selfclose:
                    state.stack.append((tag, attrs))
        self._emit_text(state, line[pos:], out)

    def _emit_text(self, state: IngestState, text: str, out: list[TextSpan]) -> None:
        text = self._clean_line(text)
        if not text:
            return
        a = self._attrs(state)
        prev = out[-1] if out else None
        if prev and prev.kind == "text" and (prev.emphasis, prev.lang_hint, prev.rate, prev.pitch_st) == \
                (a["emphasis"], a["lang_hint"], a["rate"], a["pitch_st"]):
            prev.text += text
        else:
            out.append(TextSpan(text=text, **a))
