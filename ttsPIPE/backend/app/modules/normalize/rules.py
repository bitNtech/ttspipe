"""normalize/rules: numbers, currency, dates, times, phones, units, abbreviations in Tamil + English (§4.1)."""
from __future__ import annotations

import re

from app.core import textutil as T
from app.core.base import ModuleInfo, NormalizeModule, NormResult, Word, schema
from app.core.registry import register

LEAD = "([\"'“‘<"
TRAIL = ",;:!?)]\"'”’>.…"

UNITS = {
    "km": ("kilometres", "கிலோமீட்டர்"), "kms": ("kilometres", "கிலோமீட்டர்"),
    "kg": ("kilograms", "கிலோ"), "kgs": ("kilograms", "கிலோ"), "g": ("grams", "கிராம்"),
    "gm": ("grams", "கிராம்"), "gms": ("grams", "கிராம்"), "mg": ("milligrams", "மில்லிகிராம்"),
    "ml": ("millilitres", "மில்லி"), "l": ("litres", "லிட்டர்"), "ltr": ("litres", "லிட்டர்"),
    "cm": ("centimetres", "சென்டிமீட்டர்"), "mm": ("millimetres", "மில்லிமீட்டர்"),
    "m": ("metres", "மீட்டர்"), "°c": ("degrees Celsius", "டிகிரி செல்சியஸ்"),
    "°f": ("degrees Fahrenheit", "டிகிரி ஃபாரன்ஹீட்"), "gb": ("G B", "ஜி பி"), "mb": ("M B", "எம் பி"),
    "kb": ("K B", "கே பி"), "tb": ("T B", "டி பி"), "hr": ("hour", "மணி நேரம்"), "hrs": ("hours", "மணி நேரம்"),
    "min": ("minutes", "நிமிடம்"), "mins": ("minutes", "நிமிடம்"), "sec": ("seconds", "வினாடி"),
    "secs": ("seconds", "வினாடி"), "kmph": ("kilometres per hour", "கிலோமீட்டர் வேகம்"),
    "km/h": ("kilometres per hour", "கிலோமீட்டர் வேகம்"),
}
UNIT_RE = "|".join(sorted((re.escape(u) for u in UNITS), key=len, reverse=True))
NUM = r"-?\d[\d,]*(?:\.\d+)?"

ABBR_EN = {
    "dr.": "Doctor", "mr.": "Mister", "mrs.": "Missus", "ms.": "Miss", "prof.": "Professor",
    "jr.": "Junior", "sr.": "Senior", "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is",
    "vs": "versus", "vs.": "versus", "approx.": "approximately", "govt.": "government", "dept.": "department",
    "&": "and", "@": "at", "+": "plus", "=": "equals", "w/": "with", "a.m.": "a m", "p.m.": "p m",
}
ABBR_TA = {"&": "மற்றும்", "+": "கூட்டல்", "=": "சமம்", "@": "அட்"}

CURRENCY = {"₹": ("rupee", "rupees", "paisa", "paise", "ரூபாய்", "பைசா"),
            "rs": ("rupee", "rupees", "paisa", "paise", "ரூபாய்", "பைசா"),
            "inr": ("rupee", "rupees", "paisa", "paise", "ரூபாய்", "பைசா"),
            "$": ("dollar", "dollars", "cent", "cents", "டாலர்", "சென்ட்"),
            "usd": ("dollar", "dollars", "cent", "cents", "டாலர்", "சென்ட்")}
MULT = {"k": 1000, "l": 10**5, "lakh": 10**5, "lakhs": 10**5, "lac": 10**5, "cr": 10**7, "crore": 10**7,
        "crores": 10**7, "m": 10**6, "mn": 10**6}
YEAR_CUES = {"in", "since", "year", "by", "from", "till", "until", "of", "during", "before", "after"}


def _month(word: str) -> int | None:
    """'Mar' / 'march' / 'Sept' → 2/2/8; anything else (e.g. 'decimal') → None."""
    w = word.lower()
    for i, name in enumerate(T.MONTHS_EN):
        if len(w) >= 3 and (name.lower() == w or (len(w) <= 4 and name.lower().startswith(w))):
            return i
    return None


def _split_affixes(tok: str) -> tuple[str, str, str]:
    i, j = 0, len(tok)
    while i < j and tok[i] in LEAD:
        i += 1
    while j > i and tok[j - 1] in TRAIL:
        j -= 1
    return tok[:i], tok[i:j], tok[j:]


@register
class RulesNormalize(NormalizeModule):
    info = ModuleInfo(
        id="normalize/rules", kind="normalize", name="Rule-based normalizer (ta/en)",
        description="Numbers, ₹, dates, times, phone numbers, units and abbreviations worded in Tamil or English.",
        builtin=True, tags=["builtin"], languages=["ta", "en", "ta-Latn"],
        params_schema=schema(
            language=("string", "auto", {"enum": ["auto", "ta", "en"], "description": "Wording language for numbers"}),
            default_language=("string", "en", {"enum": ["ta", "en"], "description": "Used when context has no letters"}),
            en_number_style=("string", "indian", {"enum": ["indian", "western"], "description": "lakh/crore vs million"}),
            phone_digits=("boolean", True, {"description": "Read 10-13 digit numbers digit by digit"}),
            expand_abbreviations=("boolean", True),
        ),
    )

    # ---- language choice ----------------------------------------------------------------------

    def _lang_at(self, tokens: list[str], i: int, k: int, forced: str) -> str:
        if forced in ("ta", "en"):
            return forced
        if self.params["language"] in ("ta", "en"):
            return self.params["language"]
        order = [i + k] if i + k < len(tokens) else []
        order += list(range(i - 1, -1, -1)) + list(range(i + k + 1, len(tokens)))
        for j in order:
            s = T.script_of(tokens[j])
            if s == "ta":
                return "ta"
            if s == "latin" and not re.fullmatch(r"(?i)(rs\.?|inr|usd|am|pm|a\.m\.|p\.m\.|" + UNIT_RE + r")[.,]?", tokens[j]):
                return "en"
        return self.params["default_language"]

    # ---- number wording -----------------------------------------------------------------------

    def _int(self, s: str) -> tuple[int, bool]:
        """parse '1,50,000' → (150000, indian_grouping)."""
        neg = s.startswith("-")
        s = s.lstrip("-")
        indian = bool(re.fullmatch(r"\d{1,2}(,\d{2})+,\d{3}", s))
        n = int(s.replace(",", "") or 0)
        return (-n if neg else n), indian

    def _number(self, s: str, lang: str, prev: str = "") -> str:
        ip, _, dp = s.partition(".")
        raw_int = ip.lstrip("-").replace(",", "")
        if len(raw_int) > 1 and raw_int.startswith("0") and not dp:
            return T.digits_ta(raw_int) if lang == "ta" else T.digits_en(raw_int)
        n, indian = self._int(ip)
        if len(raw_int) > 13:
            return T.digits_ta(raw_int) if lang == "ta" else T.digits_en(raw_int)
        if lang == "ta":
            w = T.num_ta(n)
            if dp:
                w += " புள்ளி " + T.digits_ta(dp)
            return w
        if not dp and "," not in ip and 1100 <= n <= 2099 and prev.lower().strip(LEAD + TRAIL) in YEAR_CUES:
            return T.year_en(n)
        w = T.num_en(n, indian=indian or self.params["en_number_style"] == "indian")
        if dp:
            w += " point " + T.digits_en(dp)
        return w

    # ---- matchers (return spoken text or None) ------------------------------------------------

    def _phone(self, core: str, lang: str) -> str | None:
        if not self.params["phone_digits"]:
            return None
        if not re.fullmatch(r"\+?\d[\d\s-]{8,18}\d", core):
            return None
        digits = re.sub(r"\D", "", core)
        if not (10 <= len(digits) <= 13):
            return None
        if not (core.startswith("+") or core.startswith("0") or digits[0] in "6789" or " " in core or "-" in core):
            return None
        groups = [g for g in re.split(r"[\s-]+", core) if g]
        say = T.digits_ta if lang == "ta" else T.digits_en
        out = []
        for g in groups:
            words = ("பிளஸ் " if lang == "ta" else "plus ") if g.startswith("+") else ""
            out.append((words + say(g)).strip() + ",")
        return " ".join(out).rstrip(",")

    def _date(self, core: str, lang: str) -> str | None:
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", core)
        if m:
            y, mo, d = int(m[1]), int(m[2]), int(m[3])
        else:
            m = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{2}|\d{4})", core)
            if m:
                d, mo, y = int(m[1]), int(m[2]), int(m[3])
                if y < 100:
                    y += 2000
            else:
                m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?(?:\s+(\d{4}))?", core)
                if m and _month(m[2]) is not None:
                    d, mo, y = int(m[1]), _month(m[2]) + 1, int(m[3]) if m[3] else None
                else:
                    m = re.fullmatch(r"([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?(?:\s+(\d{4}))?", core)
                    if m and _month(m[1]) is not None:
                        d, mo, y = int(m[2]), _month(m[1]) + 1, int(m[3]) if m[3] else None
                    else:
                        m = re.fullmatch(r"(\d{1,2})\s+([஀-௿]+),?(?:\s+(\d{4}))?", core)
                        if m and m[2] in T.MONTH_LOOKUP:
                            d, mo, y = int(m[1]), T.MONTH_LOOKUP[m[2]] + 1, int(m[3]) if m[3] else None
                        else:
                            return None
        if not (1 <= mo <= 12 and 1 <= d <= 31):
            return None
        if lang == "ta":
            s = f"{T.MONTHS_TA[mo - 1]} {T.date_ord_ta(d)} தேதி"
            return s + (f", {T.num_ta(y)}" if y else "")
        s = f"{T.MONTHS_EN[mo - 1]} {T.ord_en(d)}" if core[:1].isalpha() else f"{T.ord_en(d)} {T.MONTHS_EN[mo - 1]}"
        return s + (f", {T.year_en(y)}" if y else "")

    def _time(self, core: str, lang: str, next_tok: str) -> str | None:
        m = re.fullmatch(r"(\d{1,2}):(\d{2})(?:\s?(am|pm|a\.m\.|p\.m\.))?", core, re.I) or \
            re.fullmatch(r"(\d{1,2})\.(\d{2})\s?(am|pm|a\.m\.|p\.m\.)", core, re.I) or \
            re.fullmatch(r"(\d{1,2})()\s?(am|pm|a\.m\.|p\.m\.)", core, re.I)
        if not m:
            return None
        h, mi = int(m[1]), int(m[2] or 0)
        ampm = (m[3] or "").lower().replace(".", "")
        if h > 23 or mi > 59:
            return None
        if lang == "ta":
            pre = ""
            if ampm == "am":
                pre = "காலை "
            elif ampm == "pm":
                pre = "மதியம் " if h in (12, 1, 2, 3) else "மாலை " if h in (4, 5, 6) else "இரவு "
            if next_tok.startswith("மணி"):
                return pre + T.num_ta(h) + (f" {T.num_ta(mi)}" if mi else "")
            return pre + T.num_ta(h) + " மணி" + (f" {T.num_ta(mi)} நிமிடம்" if mi else "")
        s = T.num_en(h)
        if mi == 0:
            s += "" if ampm else " o'clock"
        elif mi < 10:
            s += " oh " + T.num_en(mi)
        else:
            s += " " + T.num_en(mi)
        if ampm:
            s += " " + ("a m" if ampm == "am" else "p m")
        return s

    def _currency(self, core: str, lang: str) -> str | None:
        m = re.fullmatch(r"(₹|rs\.?|inr|\$|usd)\s?(\d[\d,]*(?:\.\d+)?)\s?(k|l|lakhs?|lac|cr|crores?|mn|m)?", core, re.I)
        if not m:
            m2 = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)\s?(₹|/-)", core)
            if not m2:
                return None
            sym, amt, mult = "₹", m2[1], None
        else:
            sym, amt, mult = m[1].lower().rstrip("."), m[2], (m[3] or "").lower() or None
        one, many, sub1, subn, ta_unit, ta_sub = CURRENCY[sym]
        ip, _, dp = amt.partition(".")
        n, _ = self._int(ip)
        if mult:
            # ₹2.5 crore → "two point five crore rupees"
            num = self._number(amt, lang)
            mult_word = {"k": ("thousand", "ஆயிரம்"), "l": ("lakh", "லட்சம்"), "lakh": ("lakh", "லட்சம்"),
                         "lakhs": ("lakh", "லட்சம்"), "lac": ("lakh", "லட்சம்"), "cr": ("crore", "கோடி"),
                         "crore": ("crore", "கோடி"), "crores": ("crore", "கோடி"), "m": ("million", "மில்லியன்"),
                         "mn": ("million", "மில்லியன்")}[mult]
            return f"{num} {mult_word[1]} {ta_unit}" if lang == "ta" else f"{num} {mult_word[0]} {many}"
        paise = int((dp + "0")[:2]) if dp else 0
        if lang == "ta":
            s = ("ஒரு" if n == 1 else T.num_ta(n)) + " " + ta_unit
            if paise:
                s += f" {T.num_ta(paise)} {ta_sub}"
            return s
        s = f"{T.num_en(n, indian=self.params['en_number_style'] == 'indian')} {one if n == 1 else many}"
        if paise:
            s += f" {T.num_en(paise)} {sub1 if paise == 1 else subn}"
        return s

    def _percent_unit(self, core: str, lang: str) -> str | None:
        m = re.fullmatch(rf"({NUM})\s?%", core)
        if m:
            return f"{self._number(m[1], lang)} {'சதவீதம்' if lang == 'ta' else 'percent'}"
        m = re.fullmatch(rf"({NUM})\s?({UNIT_RE})", core, re.I)
        if m and not (len(m[2]) == 1 and m[2].isupper()):  # 4G / 5M are not grams / metres
            unit = UNITS[m[2].lower()]
            return f"{self._number(m[1], lang)} {unit[1] if lang == 'ta' else unit[0]}"
        return None

    def _ordinal(self, core: str, lang: str) -> str | None:
        m = re.fullmatch(r"(\d+)(st|nd|rd|th)", core, re.I)
        if m:
            n = int(m[1])
            return T.ord_ta(n) if lang == "ta" else T.ord_en(n)
        m = re.fullmatch(r"(\d+)\s?-?(வது|ஆவது)", core)
        if m:
            return T.ord_ta(int(m[1]))
        m = re.fullmatch(r"(\d+)\s?-?(ஆம்|ம்)", core)
        if m:
            return T.date_ord_ta(int(m[1]))
        return None

    def _num_suffix(self, core: str, lang: str, prev: str) -> str | None:
        m = re.fullmatch(rf"({NUM})", core)
        if m:
            return self._number(m[1], lang, prev)
        # 10மணிக்கு / 2024ல் — ponytail: suffix glued to the last number word, no sandhi rules
        m = re.fullmatch(r"(\d[\d,]*)-?([஀-௿]+)", core)
        if m:
            words = self._number(m[1], "ta").split(" ")
            return " ".join(words[:-1] + [words[-1] + m[2]]) if m[2] not in ("மணி", "பேர்") else " ".join(words + [m[2]])
        # 4G, COVID-19, A1
        if re.search(r"\d", core) and re.search(r"[A-Za-z஀-௿]", core):
            parts = re.findall(r"\d+|[^\d-]+", core)
            out = []
            for p in parts:
                if p.isdigit():
                    out.append(self._number(p, lang) if len(p) <= 4 else (T.digits_ta(p) if lang == "ta" else T.digits_en(p)))
                else:
                    out.append(p)
            return " ".join(out)
        return None

    def _abbr(self, core: str, lang: str, next_tok: str) -> str | None:
        if not self.params["expand_abbreviations"]:
            return None
        low = core.lower()
        if low in ("no.", "no") and re.match(r"\d", next_tok or ""):
            return "எண்" if lang == "ta" else "number"
        if lang == "ta" and low in ABBR_TA:
            return ABBR_TA[low]
        return ABBR_EN.get(low)

    # ---- main -----------------------------------------------------------------------------------

    def normalize(self, text: str, *, language: str = "auto") -> NormResult:
        text = T.to_ascii_digits(text)
        tokens = text.split()
        words: list[Word] = []
        edits: list[dict] = []
        i, g = 0, 0
        while i < len(tokens):
            matched = False
            for k in (4, 3, 2, 1):
                if i + k > len(tokens):
                    continue
                joined = " ".join(tokens[i:i + k])
                lead, core, trail = _split_affixes(joined)
                if not core:
                    continue
                lang = self._lang_at(tokens, i, k, language)
                prev = tokens[i - 1] if i > 0 else ""
                nxt = tokens[i + k] if i + k < len(tokens) else ""
                spoken = None
                # try with the trailing '.' kept first (abbreviations, a.m.)
                cands = [(core + trail[:1], trail[1:])] if trail.startswith(".") else []
                cands.append((core, trail))
                for c, tr in cands:
                    spoken = (self._phone(c, lang) or self._date(c, lang) or self._time(c, lang, nxt)
                              or self._currency(c, lang) or self._percent_unit(c, lang))
                    if spoken is None and k == 1:
                        spoken = self._ordinal(c, lang) or self._abbr(c, lang, nxt) or self._num_suffix(c, lang, prev)
                    if spoken is not None:
                        if c.endswith(".") and not tr and (not nxt or (c.lower() in ("etc.", "a.m.", "p.m.") and nxt[:1].isupper())):
                            tr = "."  # the abbreviation's period also ended the sentence
                        trail = tr
                        core = c
                        break
                if spoken is None:
                    continue
                parts = spoken.split()
                parts[0] = lead + parts[0]
                parts[-1] = parts[-1] + trail
                for wi, p in enumerate(parts):
                    words.append(Word(text=p, raw=joined if wi == 0 else "", group=g))
                edits.append({"before": core, "after": spoken})
                g += 1
                i += k
                matched = True
                break
            if matched:
                continue
            tok = tokens[i]
            if not T.strip_punct(tok) and words and not re.search(r"[\d₹$%&@+=]", tok):
                words[-1].text += tok  # lone punctuation joins the previous word
                words[-1].raw = (words[-1].raw + " " + tok) if words[-1].raw else words[-1].raw
            else:
                words.append(Word(text=tok, raw=tok, group=g))
                g += 1
            i += 1
        out = " ".join(w.text for w in words)
        return NormResult(text=out, changed=out != T.normalize_ws(text), edits=edits, words=words)
