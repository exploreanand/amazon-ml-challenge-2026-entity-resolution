"""Matching model: pair features -> XGBoost probability -> final match lists.

Decision rule (tuned on the validation split for macro F0.5):
  1. each S2/S3 record keeps only its highest-probability S1 (every S2/S3
     record matches at most one S1 in the training data);
  2. a pair is a match if its probability is at least ``threshold``.
An S1 with no pair above the threshold gets an empty list (singleton).
"""

import numpy as np
import pandas as pd

from .features import competition_features, string_features

SCORE_COLS = ["e5", "nameaddr", "skall", "naddr"]
RANK_COLS = [f"{c}_rank" for c in SCORE_COLS]
EXTRA_COLS = ["bridge_sim", "bridge_nb_score"]  # present only for v5+ candidate tables


def build_features(cands, s1, s23, version="v8"):
    """cands: DataFrame[s1, cand, <pass>, <pass>_rank]; s1/s23: prepared records; version: see string_features."""
    rec = pd.concat([s1, s23], ignore_index=True).set_index("entity_id")
    extra = [c for c in EXTRA_COLS if c in cands]
    X = cands[["s1", "cand"] + SCORE_COLS + RANK_COLS + extra].copy()
    X["cand_is_s3"] = X.cand.str.startswith("S3-").astype(np.float32)
    X["n_passes"] = X[RANK_COLS].notna().sum(axis=1).astype(np.float32)
    X = X.join(string_features(X[["s1", "cand"]], rec, version=version))
    X = X.join(competition_features(X, SCORE_COLS + ["n_tset", "a_tset"]))
    return X


def feature_names(X):
    return [c for c in X.columns if c not in ("s1", "cand", "label")]


def add_labels(X, truth):
    t = truth.assign(label=1)
    return X.merge(t, on=["s1", "cand"], how="left").fillna({"label": 0})


def decide(X, prob, threshold):
    """Apply the decision rule; returns DataFrame[s1, cand] of predicted matches."""
    d = X[["s1", "cand"]].assign(p=prob)
    d = d[d.p >= threshold]
    d = d.sort_values("p", ascending=False).drop_duplicates("cand")
    return d[["s1", "cand"]]


def decide_expected_f(X, prob, beta=0.5, shift=0.0, max_k=15, max_rest=20, chunk=200_000):
    """Per-S1 decisions that maximise expected F-beta (plug-in rule, Ye et al., ICML 2012).

    Each S2/S3 record first goes to its highest-probability S1, as in ``decide``.
    An S1 may then predict the top-k of the records it received, for the k with the
    highest expected F-beta under independent pair probabilities (shifted by ``shift``
    on the logit scale; > 0 is stricter). Its other pairs count only as possible
    matches it cannot predict. k = 0 (a singleton) is worth the chance that none of
    its pairs is a match. Returns DataFrame[s1, cand] like ``decide``.
    """
    p = np.clip(np.asarray(prob, dtype=np.float64), 1e-6, 1 - 1e-6)
    d = X[["s1", "cand"]].assign(p=1 / (1 + np.exp(-(np.log(p / (1 - p)) - shift))))
    d = d.sort_values("p", ascending=False, kind="stable")
    own = ~d.cand.duplicated()
    d["code"], s1s = pd.factorize(d.s1)
    d["rank"] = -1
    d.loc[own, "rank"] = d[own].groupby("code").cumcount().values
    sel = own & (d["rank"] < max_k)
    rest = d[~sel].copy()
    rest["rank"] = rest.groupby("code").cumcount().values
    rest = rest[rest["rank"] < max_rest]
    n, b2, width = len(s1s), beta * beta, max_k + max_rest + 1
    Q, R = np.zeros((n, max_k)), np.zeros((n, max_rest))
    Q[d.code[sel].values, d["rank"][sel].values] = d.p[sel].values
    R[rest.code.values, rest["rank"].values] = rest.p.values
    a, b = np.arange(max_k + 1)[:, None], np.arange(width)[None, :]
    best_k = np.zeros(n, dtype=np.int64)
    for s in range(0, n, chunk):
        q, r = Q[s:s + chunk], R[s:s + chunk]
        m = len(q)
        pa = np.zeros((max_k + 1, m, max_k + 1))  # pa[k]: distribution of true matches among the top k
        pa[0, :, 0] = 1
        for k in range(1, max_k + 1):
            pa[k] = pa[k - 1] * (1 - q[:, k - 1:k])
            pa[k, :, 1:] += pa[k - 1, :, :-1] * q[:, k - 1:k]
        pb = np.zeros((m, width))  # distribution of true matches outside the top k (k = max_k first)
        pb[:, 0] = 1
        for j in range(max_rest):
            pb[:, 1:] = pb[:, 1:] * (1 - r[:, j:j + 1]) + pb[:, :-1] * r[:, j:j + 1]
            pb[:, 0] *= 1 - r[:, j]
        ef = np.zeros((m, max_k + 1))
        for k in range(max_k, 0, -1):
            w = (1 + b2) * a / (b2 * (a + b) + k)
            ef[:, k] = ((pa[k] @ w) * pb).sum(1)
            pb[:, 1:] = pb[:, 1:] * (1 - q[:, k - 1:k]) + pb[:, :-1] * q[:, k - 1:k]
            pb[:, 0] *= 1 - q[:, k - 1]
        ef[:, 0] = pb[:, 0]
        best_k[s:s + chunk] = ef.argmax(1)
    keep = sel & (d["rank"] < best_k[d.code.values])
    return d.loc[keep, ["s1", "cand"]]


def to_output(pairs, s1_ids, col):
    """Predicted/candidate pairs -> submission frame with one row per S1 (empty = none)."""
    lists = pairs.groupby("s1").cand.agg(lambda c: ",".join(sorted(set(c))))
    out = pd.DataFrame({"source1_entity_id": pd.Index(s1_ids)})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    return out
