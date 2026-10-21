"""Text helpers shared by builtin modules: scripts, words, Tamil/English number wording."""
from __future__ import annotations

import re
import unicodedata

TAMIL_RE = re.compile(r"[஀-௿]")
LATIN_RE = re.compile(r"[A-Za-z]")
TAMIL_DIGITS = {chr(0x0BE6 + i): str(i) for i in range(10)}
PULLI = "்"
VOWEL_SIGNS = set("ாிீுூெேைொோௌௗ")


def has_tamil(s: str) -> bool:
    return bool(TAMIL_RE.search(s))


def has_latin(s: str) -> bool:
    return bool(LATIN_RE.search(s))


def script_of(word: str) -> str:
    """'ta' | 'latin' | 'none' (digits/punct only)."""
    t, l = len(TAMIL_RE.findall(word)), len(LATIN_RE.findall(word))
    if t == 0 and l == 0:
        return "none"
    return "ta" if t >= l else "latin"


def strip_punct(word: str) -> str:
    return "".join(c for c in word if not unicodedata.category(c).startswith(("P", "S")))


def core_word(word: str) -> str:
    return strip_punct(word).lower()


def tamil_syllables(word: str) -> int:
    n = 0
    chars = list(word)
    for i, c in enumerate(chars):
        if not TAMIL_RE.match(c) or c in VOWEL_SIGNS or c == PULLI:
            continue
        if i + 1 < len(chars) and chars[i + 1] == PULLI:
            continue
        n += 1
    return n


def latin_syllables(word: str) -> int:
    return len(re.findall(r"[aeiouy]+", word.lower()))


def syllables(word: str) -> int:
    w = strip_punct(word)
    if has_tamil(w):
        n = tamil_syllables(w)
    else:
        n = latin_syllables(w)
        if n == 0 and w:
            n = max(1, len(w) // 2)  # acronyms / consonant clusters
    return max(1, n) if w else 0


# ---- English numbers ------------------------------------------------------------------------

_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _en_below_1000(n: int) -> list[str]:
    out = []
    if n >= 100:
        out += [_EN_ONES[n // 100], "hundred"]
        n %= 100
        if n:
            out.append("and")
    if n >= 20:
        out.append(_EN_TENS[n // 10] + ("-" + _EN_ONES[n % 10] if n % 10 else ""))
    elif n or not out:
        out.append(_EN_ONES[n])
    return out


def num_en(n: int, indian: bool = True) -> str:
    if n < 0:
        return "minus " + num_en(-n, indian)
    if n < 1000:
        return " ".join(_en_below_1000(n))
    parts = []
    if indian:
        for size, name in ((10**7, "crore"), (10**5, "lakh"), (1000, "thousand")):
            if n >= size:
                q, n = divmod(n, size)
                parts += [num_en(q, indian), name]
    else:
        for size, name in ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand")):
            if n >= size:
                q, n = divmod(n, size)
                parts += [num_en(q, indian), name]
    if n:
        if n < 100:
            parts.append("and")
        parts += _en_below_1000(n)
    return " ".join(parts)


_EN_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
           "nine": "ninth", "twelve": "twelfth"}


def ord_en(n: int) -> str:
    words = num_en(n).split(" ")
    last = words[-1]
    head, _, tail = last.rpartition("-")
    if tail in _EN_ORD:
        tail = _EN_ORD[tail]
    elif tail.endswith("y"):
        tail = tail[:-1] + "ieth"
    else:
        tail += "th"
    words[-1] = (head + "-" if head else "") + tail
    return " ".join(words)


def year_en(n: int) -> str:
    if 2000 <= n <= 2009:
        return "two thousand" + (f" {_EN_ONES[n - 2000]}" if n > 2000 else "")
    if 1100 <= n <= 2099 and n % 100:
        hi, lo = divmod(n, 100)
        return f"{num_en(hi)} {('oh ' + num_en(lo)) if lo < 10 else num_en(lo)}"
    if 1100 <= n <= 2099:
        return f"{num_en(n // 100)} hundred"
    return num_en(n, indian=False)


def digits_en(s: str) -> str:
    return " ".join(_EN_ONES[int(c)] for c in s if c.isdigit())


# ---- Tamil numbers --------------------------------------------------------------------------

_TA_ONES = ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
_TA_TEENS = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதிமூன்று", "பதினான்கு", "பதினைந்து", "பதினாறு",
             "பதினேழு", "பதினெட்டு", "பத்தொன்பது"]
_TA_TENS = ["", "", "இருபது", "முப்பது", "நாற்பது", "ஐம்பது", "அறுபது", "எழுபது", "எண்பது", "தொண்ணூறு"]
_TA_TENS_C = ["", "", "இருபத்து", "முப்பத்து", "நாற்பத்து", "ஐம்பத்து", "அறுபத்து", "எழுபத்து",
              "எண்பத்து", "தொண்ணூற்று"]
_TA_HUND = ["", "நூறு", "இருநூறு", "முந்நூறு", "நானூறு", "ஐந்நூறு", "அறுநூறு", "எழுநூறு", "எண்ணூறு",
            "தொள்ளாயிரம்"]
_TA_HUND_C = ["", "நூற்று", "இருநூற்று", "முந்நூற்று", "நானூற்று", "ஐந்நூற்று", "அறுநூற்று",
              "எழுநூற்று", "எண்ணூற்று", "தொள்ளாயிரத்து"]
_TA_THOU = {1: "ஆயிரம்", 2: "இரண்டாயிரம்", 3: "மூவாயிரம்", 4: "நான்காயிரம்", 5: "ஐயாயிரம்",
            6: "ஆறாயிரம்", 7: "ஏழாயிரம்", 8: "எட்டாயிரம்", 9: "ஒன்பதாயிரம்", 10: "பத்தாயிரம்"}


def _ta_below_100(n: int) -> str:
    if n < 10:
        return _TA_ONES[n]
    if n < 20:
        return _TA_TEENS[n - 10]
    t, o = divmod(n, 10)
    return _TA_TENS[t] if o == 0 else f"{_TA_TENS_C[t]} {_TA_ONES[o]}"


def _ta_below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _ta_below_100(r)
    if r == 0:
        return _TA_HUND[h]
    return f"{_TA_HUND_C[h]} {_ta_below_100(r)}"


def _combining(word: str) -> str:
    """ஆயிரம் → ஆயிரத்து, லட்சம் → லட்சத்து, கோடி → கோடியே."""
    if word.endswith("ம்"):
        return word[:-2] + "த்து"
    if word.endswith("கோடி"):
        return word + "யே"
    return word


def num_ta(n: int) -> str:
    if n < 0:
        return "மைனஸ் " + num_ta(-n)
    if n < 1000:
        return _ta_below_1000(n)
    parts: list[str] = []
    for size, name in ((10**7, "கோடி"), (10**5, "லட்சம்")):
        if n >= size:
            q, n = divmod(n, size)
            parts.append(("ஒரு" if q == 1 else num_ta(q)) + " " + name)
    if n >= 1000:
        q, n = divmod(n, 1000)
        parts.append(_TA_THOU.get(q) or f"{num_ta(q)} ஆயிரம்")
    if n:
        parts.append(_ta_below_1000(n))
    # all but the last group take the combining form
    words = " ".join(parts[:-1]).split(" ") if len(parts) > 1 else []
    if words:
        words[-1] = _combining(words[-1])
    return " ".join(words + [parts[-1]]).strip()


def _ta_suffix(word: str, suffix_after_drop: str) -> str:
    if word.endswith("ு") or word.endswith(PULLI):  # ு or ்
        return word[:-1] + suffix_after_drop
    return word + suffix_after_drop


def ord_ta(n: int) -> str:
    """1 → முதலாவது, 2 → இரண்டாவது."""
    if n == 1:
        return "முதலாவது"
    w = num_ta(n)
    return _ta_suffix(w, "ாவது")


def date_ord_ta(n: int) -> str:
    """Day-of-month form: 12 → பன்னிரண்டாம்."""
    if n == 1:
        return "ஒன்றாம்"
    return _ta_suffix(num_ta(n), "ாம்")


def digits_ta(s: str) -> str:
    return " ".join(_TA_ONES[int(c)] for c in s if c.isdigit())


def to_ascii_digits(s: str) -> str:
    return "".join(TAMIL_DIGITS.get(c, c) for c in s)


MONTHS_EN = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
             "October", "November", "December"]
MONTHS_TA = ["ஜனவரி", "பிப்ரவரி", "மார்ச்", "ஏப்ரல்", "மே", "ஜூன்", "ஜூலை", "ஆகஸ்ட்", "செப்டம்பர்",
             "அக்டோபர்", "நவம்பர்", "டிசம்பர்"]
MONTH_LOOKUP = {m[:3].lower(): i for i, m in enumerate(MONTHS_EN)}
MONTH_LOOKUP.update({"sept": 8})
MONTH_LOOKUP.update({m: i for i, m in enumerate(MONTHS_TA)})


def words_of(text: str) -> list[str]:
    return text.split()


def normalize_ws(text: str) -> str:
    return " ".join(text.split())
