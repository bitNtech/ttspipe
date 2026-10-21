"""g2p/lexicon: lexicon overrides + rule-based romanized Tamil → Tamil script (§4.1)."""
from __future__ import annotations

import re

from app.core import lexicon
from app.core.base import Chunk, G2PModule, ModuleInfo, schema
from app.core.registry import register

VOWELS = {  # token: (independent, sign)
    "aa": ("ஆ", "ா"), "ii": ("ஈ", "ீ"), "ee": ("ஈ", "ீ"), "uu": ("ஊ", "ூ"), "oo": ("ஊ", "ூ"),
    "ai": ("ஐ", "ை"), "au": ("ஔ", "ௌ"), "ae": ("ஏ", "ே"), "oa": ("ஓ", "ோ"),
    "a": ("அ", ""), "i": ("இ", "ி"), "u": ("உ", "ு"), "e": ("எ", "ெ"), "o": ("ஒ", "ொ"),
}
CONS = {
    "zh": "ழ", "dh": "த", "th": "த", "ch": "ச", "sh": "ஷ", "nj": "ஞ", "gn": "ஞ", "ng": "ங", "ph": "ப",
    "bh": "ப", "kh": "க", "gh": "க", "jh": "ஜ", "rr": "ற", "tr": "ற",
    "k": "க", "g": "க", "c": "ச", "s": "ச", "t": "ட", "d": "ட", "n": "ன", "p": "ப", "b": "ப", "m": "ம",
    "y": "ய", "r": "ர", "l": "ல", "v": "வ", "w": "வ", "j": "ஜ", "h": "ஹ", "f": "ஃப", "z": "ஜ", "x": "க்ஸ",
    "q": "க",
}
PULLI = "்"
TOKENS = sorted(list(VOWELS) + list(CONS), key=len, reverse=True)


def _tokenize(word: str) -> list[str]:
    out, i = [], 0
    while i < len(word):
        for t in TOKENS:
            if word.startswith(t, i):
                out.append(t)
                i += len(t)
                break
        else:
            out.append(word[i])
            i += 1
    return out


def translit_word(word: str) -> str:
    """Romanized Tamil word → Tamil script. ponytail: rule-based, ~ casual-spelling quality; IndicXlit is the upgrade."""
    toks = _tokenize(word.lower())
    out: list[str] = []
    n = len(toks)
    i = 0
    while i < n:
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < n else None
        if t in VOWELS:
            ind, sign = VOWELS[t]
            out.append(ind)
            i += 1
            continue
        if t not in CONS:
            out.append(t)
            i += 1
            continue
        base = CONS[t]
        if t in ("rr", "tr"):
            out.append("ற" + PULLI)
            base = "ற"
        if t == "n":
            if i == 0:
                base = "ந"
            elif nxt in ("dh", "th"):
                base = "ந"
            elif nxt == "d":
                base = "ண"
        if t == "ch" and i > 0 and toks[i - 1] in VOWELS and nxt in VOWELS:
            out.append("ச" + PULLI)  # intervocalic ch → ச்ச
        if t == "ng" and nxt in VOWELS:
            out.append("ங" + PULLI)
            base = "க"
        if nxt in VOWELS:
            sign = VOWELS[nxt][1]
            if i + 2 == n and nxt == "o":
                sign = "ோ"  # final o is long: po → போ
            elif i + 2 == n and nxt == "e":
                sign = "ே"  # final e is long: inge → இங்கே
            out.append(base + sign)
            i += 2
        else:
            out.append(base + PULLI)
            i += 1
    return "".join(out)


AFFIX_RE = re.compile(r"^([^\w஀-௿]*)(.*?)([^\w஀-௿]*)$")


@register
class LexiconG2P(G2PModule):
    info = ModuleInfo(
        id="g2p/lexicon", kind="g2p", name="Lexicon + Tanglish transliteration",
        description="Applies config/lexicon.yaml word overrides and transliterates romanized Tamil (ta-Latn) runs to Tamil script.",
        builtin=True, tags=["builtin"], languages=["ta", "en", "ta-Latn"],
        params_schema=schema(
            apply_lexicon=("boolean", True),
            transliterate_romanized=("boolean", True, {"description": "ta-Latn runs → Tamil script for Tamil engines"}),
        ),
    )

    def _word(self, w: str, lang: str, lex: dict) -> str:
        m = AFFIX_RE.match(w)
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        if not core:
            return w
        if self.params["apply_lexicon"]:
            table = lex["words"].get("ta" if lang in ("ta", "ta-Latn") else "en", {})
            key = core.lower() if core.isascii() else core
            if key in table:
                return lead + table[key] + trail
        if lang == "ta-Latn" and self.params["transliterate_romanized"] and core.isascii():
            parts = core.split("-")
            tl = lex["tanglish"]
            conv = [translit_word(p) if (p.lower() in tl or len(parts) == 1) else p for p in parts]
            return lead + "-".join(conv) + trail
        return w

    def apply(self, chunk: Chunk) -> Chunk:
        lex = lexicon.get()
        for seg in chunk.segments:
            seg.tts_text = " ".join(self._word(w, seg.lang, lex) for w in seg.text.split())
        chunk.tts_text = " ".join(s.tts_text for s in chunk.segments)
        return chunk
