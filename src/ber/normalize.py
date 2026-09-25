"""Text normalisation shared by blocking and matching.

Every function here is country-agnostic: it never branches on the country
label, so unseen countries (France in test) go through the same path.
"""

import html
import re
import unicodedata

from indic_transliteration import sanscript

# Unicode script name (first word of unicodedata.name) -> sanscript scheme.
_INDIC = {
    "DEVANAGARI": sanscript.DEVANAGARI,
    "BENGALI": sanscript.BENGALI,
    "GUJARATI": sanscript.GUJARATI,
    "GURMUKHI": sanscript.GURMUKHI,
    "KANNADA": sanscript.KANNADA,
    "MALAYALAM": sanscript.MALAYALAM,
    "ORIYA": sanscript.ORIYA,
    "TAMIL": sanscript.TAMIL,
    "TELUGU": sanscript.TELUGU,
}

_NON_LATIN = re.compile(r"[^\x00-\u024F\s]")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_SPACES = re.compile(r"\s+")
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s"})
_HAS_ALPHA = re.compile(r"[a-z]")
_HAS_DIGIT = re.compile(r"[0-9@$]")

# Light phonetic folding so romanised Indic text and English spellings meet
# (shree/sri, infra/imphra). Applied identically to every record.
_PHONETIC = [
    (re.compile(r"([kgcjtdpb])h"), r"\1"),  # aspirates: kh gh ch jh th dh ph bh
    (re.compile(r"sh"), "s"),
    (re.compile(r"p(?=[^aeiou ]|$)"), "f"),  # iMphrA -> imfra after aspirate drop
    (re.compile(r"w"), "v"),
    (re.compile(r"m(?=[^aeiouym ])"), "n"),  # anusvara before consonant
    (re.compile(r"([a-z])\1+"), r"\1"),  # collapse doubled letters
]


def _script(ch):
    return unicodedata.name(ch, "UNKNOWN").split(" ", 1)[0]


def transliterate(text):
    """Romanise any Indic-script runs in ``text`` (IAST), leave the rest."""
    if not _NON_LATIN.search(text):
        return text
    out, run, run_script = [], [], None
    for ch in text:
        sc = _script(ch) if ord(ch) > 0x24F else None
        if sc in _INDIC:
            if run and sc != run_script:
                out.append(sanscript.transliterate("".join(run), _INDIC[run_script], sanscript.IAST))
                run = []
            run.append(ch)
            run_script = sc
        else:
            if run:
                out.append(sanscript.transliterate("".join(run), _INDIC[run_script], sanscript.IAST))
                run = []
            out.append(ch)
    if run:
        out.append(sanscript.transliterate("".join(run), _INDIC[run_script], sanscript.IAST))
    return "".join(out)


def basic(text):
    """Transliterate, strip accents, lowercase, keep [a-z0-9 ] only."""
    text = transliterate(html.unescape(text))
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ").replace("’", "").replace("'", "")
    text = _NON_ALNUM.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def fix_ocr(text):
    """Map look-alike digits to letters inside tokens that also contain letters (5upreme -> supreme)."""
    return " ".join(
        t.translate(_OCR) if _HAS_ALPHA.search(t) and _HAS_DIGIT.search(t) else t
        for t in text.split(" ")
    )


_DOUBLES = re.compile(r"([a-z])\1+")


_LONG_VOWELS = [(re.compile(r"ee|ii"), "i"), (re.compile(r"oo|uu"), "u")]


def phonetic(text):
    for pat, rep in _LONG_VOWELS:
        text = pat.sub(rep, text)
    text = _DOUBLES.sub(r"\1", text)
    for pat, rep in _PHONETIC:
        text = pat.sub(rep, text)
    return text


def norm_name(text, use_phonetic=True):
    text = fix_ocr(basic(text))
    return phonetic(text) if use_phonetic else text


# Street-type words and their abbreviations collapse to one canonical token, in
# the languages seen in the data (English, Indian English, French). Applied to
# every record the same way; "saint"/"street"/"st" all become "st" because the
# abbreviation is ambiguous between them.
_ADDR_CANON = {}
for canon, variants in {
    "st": ["st", "street", "str", "saint", "sainte", "ste"],
    "rd": ["rd", "road"],
    "av": ["av", "ave", "avenue", "aven"],
    "bd": ["bd", "blvd", "boulevard", "boul", "bld"],
    "dr": ["dr", "drive"],
    "ln": ["ln", "lane"],
    "ct": ["ct", "court"],
    "hwy": ["hwy", "highway"],
    "apt": ["apt", "apartment", "appartement", "appt"],
    "fl": ["fl", "flr", "floor", "etage"],
    "nr": ["nr", "near"],
    "opp": ["opp", "opposite"],
    "r": ["r", "rue"],
    "allee": ["all", "allee", "al"],
    "imp": ["imp", "impasse"],
    "pl": ["pl", "place"],
    "ch": ["ch", "chemin", "che"],
    "rte": ["rte", "route"],
    "qu": ["qu", "quai"],
    "sq": ["sq", "square"],
    "crs": ["crs", "cours"],
    "fg": ["fg", "fbg", "faubourg"],
}.items():
    for v in variants:
        _ADDR_CANON[v] = canon
_NUMBER_WORDS = {"no", "n", "num", "numero", "nos"}  # "No. 14", "N° 14", "Nº 14"


def canon_address(text):
    """Canonicalise street-type words and drop number markers in a ``basic`` string."""
    toks = text.split(" ")
    out = []
    for i, t in enumerate(toks):
        if t in _NUMBER_WORDS and i + 1 < len(toks) and toks[i + 1][:1].isdigit():
            continue
        out.append(_ADDR_CANON.get(t, t))
    return " ".join(out)


def norm_addr(text, use_phonetic=True):
    text = canon_address(basic(text))
    return phonetic(text) if use_phonetic else text


# Consonant skeleton: drop vowels (and h/y), fold voiced/unvoiced and nasal
# pairs. Romanised Indic spellings and English spellings of the same word
# usually share it: "hotel power" / "hotala pavara" -> "htl pvr".
_SKEL_MAP = str.maketrans({"b": "p", "d": "t", "g": "k", "q": "k", "c": "k", "f": "p",
                           "w": "v", "z": "s", "x": "k", "m": "n", "j": "k"})
_SKEL_DROP = re.compile(r"[aeiouyh]")


def skeleton(text):
    """Consonant skeleton of an already ``basic``-normalised string."""
    out = []
    for tok in text.split(" "):
        if tok.isdigit():
            out.append(tok)
            continue
        sk = _DOUBLES.sub(r"\1", _SKEL_DROP.sub("", tok.translate(_SKEL_MAP)))
        if sk:
            out.append(sk)
    return " ".join(out)


def name_skeleton(text):
    return skeleton(fix_ocr(basic(text)))
