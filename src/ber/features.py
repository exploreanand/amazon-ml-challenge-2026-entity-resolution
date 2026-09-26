"""Pairwise features for the matching model.

Input is the candidate table (one row per S1 x S2/S3 pair) carrying the
blocking scores; this module adds string-similarity features computed on
the normalised fields, plus "competition" features that compare a pair
with the other candidates of the same S1 and the same S2/S3 record.
"""

import os
import re
from collections import Counter
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .normalize import basic, fix_ocr, norm_name

# Generic words that say little about identity; removed for the "core" name.
# Spelled the way norm_name outputs them (scop -> scof at the end of a name).
# Societe, Compagnie/Cie, Etablissements/Ets and Association are NOT here: in the
# data they are type words that true variants keep and decoys swap ("X Compagnie SAS").
LEGAL = {
    "pvt", "fvt", "private", "frivate", "ltd", "limited", "lnt", "lt", "prvt", "pr", "l",
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc", "llp", "lf",
    "plc", "gmbh", "sa", "sas", "sarl", "eurl", "sci", "and", "the", "of", "mr", "dba", "fka",
    "india", "services", "group",
    # French legal forms and linking words (v8)
    "sasu", "ei", "selarl", "snc", "scop", "scof", "gie",
    "de", "du", "des", "la", "le", "les", "et",
}
# the v7 feature set (string_features(version="v7")) predates these
_V8_LEGAL = {"sasu", "ei", "selarl", "snc", "scop", "scof", "gie", "de", "du", "des", "la", "le", "les", "et"}
_NUM = re.compile(r"\d+")


def core_tokens(norm_name, legal=LEGAL):
    return [t for t in norm_name.split() if t not in legal]


def without_country(texts, countries, spell):
    """Drop each record's own country name from its text ("marina ecole france" -> "marina ecole").

    The word comes from the record's country label spelled by ``spell``, so an
    unseen country gets the treatment "india" gets from LEGAL.
    """
    out = np.asarray(texts, dtype=object).copy()
    countries = np.asarray(countries)
    for c in pd.unique(countries):
        phrase = spell(c)
        if not phrase:
            continue
        pat = re.compile(rf"(?<!\S){re.escape(phrase)}(?!\S)")
        m = countries == c
        out[m] = [" ".join(pat.sub(" ", t).split()) for t in out[m]]
    return out


def name_token_share(cores, min_count=2):
    """log10 share of records whose core name has each token; tokens seen fewer than min_count times get the default."""
    df = Counter(chain.from_iterable(set(c.split()) for c in cores))
    n = max(len(cores), 1)
    return {t: float(np.log10(k / n)) for t, k in df.items() if k >= min_count}, float(np.log10(1 / n))


def _numbers(text):
    return set(n.lstrip("0") or "0" for n in _NUM.findall(text))


def _first_number(text):
    m = _NUM.search(text)
    return (m.group().lstrip("0") or "0") if m else None


# the phonetic folding in normalize can turn p->f, w->v etc.; fold both sides back
_LETTER_FOLD = str.maketrans({"f": "p", "v": "w", "c": "k", "q": "k", "z": "s", "j": "k"})


def _acronym(tokens):
    return "".join(t[0] for t in tokens if t)


def _acronym_match(n1_tokens, n2_compact):
    """1 if one name is the initials of the other's core words (TP ~ Training Parents)."""
    if not n1_tokens or len(n2_compact) < 2 or len(n1_tokens) < 2:
        return 0.0
    return float(_acronym(n1_tokens).translate(_LETTER_FOLD) == n2_compact.translate(_LETTER_FOLD))


def _street(text):
    """(first number, the next two alphabetic tokens) of an address, e.g. ('115', 'r francisco')."""
    toks = text.split()
    for i, t in enumerate(toks):
        if t[:1].isdigit():
            words = [w for w in toks[i + 1:i + 4] if w.isalpha()][:2]
            return (t.lstrip("0") or "0"), " ".join(words)
    return None, ""


def _jacc(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / max(len(a | b), 1)


# Address tokens that are frequent in the partition (regions, departments, states,
# big cities) are set by string_features; the "rare" address features ignore them,
# so e.g. "Loire-Atlantique" vs "Pays de la Loire" does not count against a pair.
# _SHARE / _SHARE_MIN: log10 share of the partition's records using each core-name token.
_FREQ = frozenset()
_SHARE, _SHARE_MIN = {}, 0.0


def _init_freq(freq, share=({}, 0.0)):
    global _FREQ, _SHARE, _SHARE_MIN
    _FREQ = freq
    _SHARE, _SHARE_MIN = share


def _rare_feats(a1, a2):
    if not (a1 and a2):
        return (np.nan, np.nan, np.nan)
    r1 = {t for t in a1.split() if t not in _FREQ}
    r2 = {t for t in a2.split() if t not in _FREQ}
    if not r1 or not r2:
        return (np.nan, np.nan, float(len(r2 - r1)))
    return (_jacc(r1, r2), fuzz.token_set_ratio(" ".join(sorted(r1)), " ".join(sorted(r2))), float(len(r2 - r1)))


def _swap_feats(c1, c2):
    """How the core names differ: exactly one word swapped (Club -> Comite, or a typo), and how
    common the differing words are in the partition (type words are common, typos rare)."""
    d1, d2 = set(c1) - set(c2), set(c2) - set(c1)
    if not (d1 or d2):
        return (0.0, 0.0, np.nan, np.nan, np.nan)
    shares = [_SHARE.get(t, _SHARE_MIN) for t in d1 | d2]
    one = len(d1) == 1 and len(d2) == 1
    sim = JaroWinkler.similarity(next(iter(d1)), next(iter(d2))) if one else np.nan
    return (float(len(d1) + len(d2)), float(one), sim, min(shares), max(shares))


def _pair_feats(args):
    n1, n2, a1, a2, k1, k2, b1, b2, s1c, s2c = args
    c1, c2 = s1c.split(), s2c.split()
    num1, num2 = _numbers(a1), _numbers(a2)
    t1, t2 = set(a1.split()), set(a2.split())
    return (
        fuzz.ratio(n1, n2), fuzz.token_set_ratio(n1, n2), fuzz.token_sort_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2), JaroWinkler.similarity(n1, n2),
        fuzz.token_set_ratio(s1c, s2c) if (c1 and c2) else np.nan,
        fuzz.ratio(s1c, s2c) if (c1 and c2) else np.nan,
        _jacc(set(c1), set(c2)),
        fuzz.token_set_ratio(k1, k2), fuzz.ratio(k1, k2),
        fuzz.token_set_ratio(a1, a2) if (a1 and a2) else np.nan,
        fuzz.partial_ratio(a1, a2) if (a1 and a2) else np.nan,
        _jacc(t1, t2) if (a1 and a2) else np.nan,
        _jacc(num1, num2) if (num1 and num2) else np.nan,
        float(len(num1 & num2)),
        float(not a2), float(len(c2)), float(len(n2)), float(len(a2)),
        # decoys are near-copies whose name letters or house numbers differ slightly
        float(Levenshtein.distance(n1, n2)),
        float(Levenshtein.distance(s1c, s2c)) if (c1 and c2) else np.nan,
        float(len(num2 - num1)) if a2 else np.nan,
        float(len(num1 - num2)) if a2 else np.nan,
        _first_num_eq(a1, a2),
        float(sum(len(t) for t in c1 if t not in set(c2))),
        float(sum(len(t) for t in c2 if t not in set(c1))),
        # acronyms and trade names at the same street address
        _acronym_feat(b1, b2),
        *_street_feats(a1, a2),
        *_rare_feats(a1, a2),
        *_swap_feats(c1, c2),
    )


_ACRO_STOP = {"and", "the", "of", "de", "des", "du", "la", "le", "les", "et", "pvt", "ltd", "private",
               "limited", "inc", "llc", "llp", "corp", "co", "sa", "sas", "sarl", "eurl", "sasu", "sci"}


def _acronym_feat(b1, b2):
    """Acronym match on plain (non-phonetic) names; b1/b2 are fix_ocr(basic(name))."""
    t1 = list(dict.fromkeys(t for t in b1.split() if t not in _ACRO_STOP))  # dedupe, keep order
    t2 = list(dict.fromkeys(t for t in b2.split() if t not in _ACRO_STOP))
    return max(_acronym_match(t1, "".join(t2)), _acronym_match(t2, "".join(t1)))


def _street_feats(a1, a2):
    (num1, st1), (num2, st2) = _street(a1), _street(a2)
    if num1 is None or num2 is None:
        return (np.nan, np.nan, np.nan)
    same_num = float(num1 == num2)
    st_ratio = fuzz.ratio(st1, st2) if (st1 and st2) else np.nan
    return (same_num, st_ratio, same_num * (st_ratio if st_ratio == st_ratio else 0.0))


def _first_num_eq(a1, a2):
    f1, f2 = _first_number(a1), _first_number(a2)
    if f1 is None or f2 is None:
        return np.nan
    return float(f1 == f2)


PAIR_COLS = [
    "n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw",
    "core_tset", "core_ratio", "core_jacc",
    "sk_tset", "sk_ratio",
    "a_tset", "a_partial", "a_jacc", "num_jacc", "num_common",
    "cand_addr_empty", "cand_core_len", "cand_name_len", "cand_addr_len",
    "n_lev", "core_lev", "num_only_cand", "num_only_s1", "first_num_eq",
    "core_unmatched_s1", "core_unmatched_cand",
    "acronym", "street_num_eq", "street_name_ratio", "street_match",
    "rare_a_jacc", "rare_a_tset", "rare_only_cand",
    "core_symdiff", "swap_one", "swap_sim", "diff_share_min", "diff_share_max",
]


def frequent_address_tokens(naddr, share=0.01):
    """Non-numeric address tokens present in more than ``share`` of the partition's records."""
    df = pd.Series([t for a in naddr for t in set(a.split())]).value_counts()
    return frozenset(t for t in df.index[df.values > share * len(naddr)] if not t[:1].isdigit())


def string_features(pairs, rec, procs=None, chunksize=5000, version="v8"):
    """pairs: DataFrame[s1, cand]; rec: records indexed by entity_id with country, nname, naddr, nskel, bname.

    Core names (LEGAL words and the record's own country name removed) feed the
    core_* and swap features; the acronym feature reads bname without the country.
    version="v7" gives the v7 feature set instead: v7 LEGAL, country word kept,
    no swap features (the features behind the v7 / v7.2 submissions).
    """
    freq = frequent_address_tokens(rec.naddr.values)
    if version == "v7":
        core = [" ".join(core_tokens(n, LEGAL - _V8_LEGAL)) for n in rec.nname.values]
        plain, share, cols = rec.bname.values, ({}, 0.0), PAIR_COLS[:-5]
    else:
        core = [" ".join(core_tokens(n)) for n in without_country(rec.nname.values, rec.country.values, norm_name)]
        plain = without_country(rec.bname.values, rec.country.values, lambda c: fix_ocr(basic(c)))
        share, cols = name_token_share(core), PAIR_COLS
    rec = rec[["nname", "naddr", "nskel"]].assign(core=core, bname=plain)
    r1 = rec.loc[pairs.s1.values]
    r2 = rec.loc[pairs.cand.values]
    args = zip(r1.nname.values, r2.nname.values, r1.naddr.values, r2.naddr.values,
               r1.nskel.values, r2.nskel.values, r1.bname.values, r2.bname.values,
               r1.core.values, r2.core.values)
    with Pool(procs or os.cpu_count(), initializer=_init_freq, initargs=(freq, share)) as p:
        out = p.map(_pair_feats, args, chunksize=chunksize)
    return pd.DataFrame(np.asarray(out, dtype=np.float32)[:, :len(cols)], columns=cols, index=pairs.index)


def competition_features(df, score_cols):
    """For each blocking score, how this pair compares with rival pairs.

    Each S2/S3 record matches at most one S1, so a pair whose score is well
    below the same record's best S1 is probably wrong, and vice versa.
    """
    out = {}
    for c in score_cols:
        s = df[c].fillna(0)
        best_for_cand = s.groupby(df.cand).transform("max")
        best_for_s1 = s.groupby(df.s1).transform("max")
        out[f"{c}_gap_cand"] = s - best_for_cand
        out[f"{c}_gap_s1"] = s - best_for_s1
        out[f"{c}_rank_in_s1"] = s.groupby(df.s1).rank(ascending=False, method="min")
        out[f"{c}_rank_in_cand"] = s.groupby(df.cand).rank(ascending=False, method="min")
    out["n_cands_s1"] = df.groupby("s1").cand.transform("size")
    out["n_s1_per_cand"] = df.groupby("cand").s1.transform("size")
    return pd.DataFrame(out, index=df.index).astype(np.float32)
