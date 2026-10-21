"""segment/script: language runs by Unicode script + Tanglish word list and suffix heuristics (§4.1)."""
from __future__ import annotations

import re

from app.core import lexicon
from app.core import textutil as T
from app.core.base import LangSegment, ModuleInfo, SegmentModule, schema
from app.core.registry import register

# Very common English words: never romanized Tamil even if they look like it.
EN_COMMON = set("""a about after again all also am an and any are as at back be because been before being best
both but by call can come could day did do does done down each even every first for from get give go good
got great had has have he her here him his how i if in into is it its just know last let like little long
look made make man many may me more most much must my new no not now of off oh ok okay old on one only or
other our out over people please put really right said same say see she should show so some still such
take tell than thank thanks that the their them then there these they thing think this those time to today
tomorrow too two up us use very want was way we well were what when where which while who why will with
work would yes yet you your order shipped arrive thursday afternoon meeting late hello hi bye sorry sure
call phone number account payment bank card office team plan ready done fine nice cool please check data
manager boss sir madam mam bro dude anna""".split()) - {"anna"}

TA_SUFFIX = re.compile(r"(nga|ngal|unga|anga|kku|ukku|la|le|oda|ode|dhu|thu|chu|ttu|ppu|kum|rom|ren|raen|"
                       r"ten|tiya|chi|duchu|dichu|laam|lam|naa|nnu|nu|vaa|maa|aa|um|ae|ey)$")
TA_BIGRAM = re.compile(r"(zh|dh|th|kk|pp|tt|nn|ll|aa|ee|oo|ng)")


def looks_tamil(w: str) -> bool:
    return bool(TA_SUFFIX.search(w) and TA_BIGRAM.search(w) and len(w) >= 4)


@register
class ScriptSegment(SegmentModule):
    info = ModuleInfo(
        id="segment/script", kind="segment", name="Script + Tanglish lexicon",
        description="Unicode-range language runs; romanized Tamil detected via lexicon.yaml tanglish list and suffix heuristics.",
        builtin=True, tags=["builtin"], languages=["ta", "en", "ta-Latn"],
        params_schema=schema(
            detect_romanized=("boolean", True, {"description": "Detect romanized Tamil (ta-Latn) in Latin text"}),
            heuristics=("boolean", True, {"description": "Use suffix heuristics for words not in the lexicon"}),
        ),
    )

    def _classify(self, word: str, tl: set[str]) -> tuple[str | None, bool]:
        """(lang, strong). lang None = neutral (digits / punctuation)."""
        s = T.script_of(word)
        if s == "none":
            return None, False
        if s == "ta":
            return "ta", True
        if not self.params["detect_romanized"]:
            return "en", True
        w = T.core_word(word.replace("-", " ")).replace(" ", "-") if "-" in word else T.core_word(word)
        if w in tl:
            return "ta-Latn", True
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            if parts and (parts[-1] in tl or any(p in tl for p in parts)):
                return "ta-Latn", True
        if w in EN_COMMON:
            return "en", True
        if self.params["heuristics"] and looks_tamil(w):
            return "ta-Latn", False
        return "en", False

    def split(self, text: str) -> list[LangSegment]:
        words = text.split()
        if not words:
            return []
        tl = lexicon.get()["tanglish"]
        cls = [self._classify(w, tl) for w in words]
        langs = [c[0] for c in cls]
        # weak English between romanized-Tamil neighbours becomes romanized Tamil when it looks Tamil
        for i, (lang, strong) in enumerate(cls):
            if lang == "en" and not strong:
                prev = next((langs[j] for j in range(i - 1, -1, -1) if langs[j]), None)
                nxt = next((langs[j] for j in range(i + 1, len(langs)) if langs[j]), None)
                if prev == "ta-Latn" and nxt in ("ta-Latn", None) and looks_tamil(T.core_word(words[i])):
                    langs[i] = "ta-Latn"
        # neutral tokens attach to the previous word (or the next one at the start)
        first = next((l for l in langs if l), "en")
        cur = first
        for i, l in enumerate(langs):
            if l is None:
                langs[i] = cur
            else:
                cur = l
        segs: list[LangSegment] = []
        for w, l in zip(words, langs):
            if segs and segs[-1].lang == l:
                segs[-1].text += " " + w
            else:
                segs.append(LangSegment(text=w, lang=l))
        for s in segs:
            s.tts_text = s.text
        return segs
