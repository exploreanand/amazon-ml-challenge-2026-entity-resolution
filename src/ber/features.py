"""Pairwise features for the matching model.

Input is the candidate table (one row per S1 x S2/S3 pair) carrying the
blocking scores; this module adds string-similarity features computed on
the normalised fields, plus "competition" features that compare a pair
with the other candidates of the same S1 and the same S2/S3 record.
"""

import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

# Generic words that say little about identity; removed for the "core" name.
LEGAL = {
    "pvt", "fvt", "private", "frivate", "ltd", "limited", "lnt", "lt", "prvt", "pr", "l",
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "lc", "llp", "lf",
    "plc", "gmbh", "sa", "sas", "sarl", "eurl", "sci", "and", "the", "of", "mr", "dba", "fka",
    "india", "services", "group",
}
_NUM = re.compile(r"\d+")


def core_tokens(norm_name):
    return [t for t in norm_name.split() if t not in LEGAL]


def _numbers(text):
    return set(n.lstrip("0") or "0" for n in _NUM.findall(text))


def _first_number(text):
    m = _NUM.search(text)
    return (m.group().lstrip("0") or "0") if m else None


def _jacc(a, b):
    if not a and not b:
        return np.nan
    return len(a & b) / max(len(a | b), 1)


def _pair_feats(args):
    n1, n2, a1, a2, k1, k2 = args
    c1, c2 = core_tokens(n1), core_tokens(n2)
    s1c, s2c = " ".join(c1), " ".join(c2)
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
    )


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
]


def string_features(pairs, rec, procs=None, chunksize=5000):
    """pairs: DataFrame[s1, cand]; rec: records indexed by entity_id with nname, naddr, nskel."""
    r1 = rec.loc[pairs.s1.values]
    r2 = rec.loc[pairs.cand.values]
    args = zip(r1.nname.values, r2.nname.values, r1.naddr.values, r2.naddr.values,
               r1.nskel.values, r2.nskel.values)
    with Pool(procs or os.cpu_count()) as p:
        out = p.map(_pair_feats, args, chunksize=chunksize)
    return pd.DataFrame(np.asarray(out, dtype=np.float32), columns=PAIR_COLS, index=pairs.index)


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
