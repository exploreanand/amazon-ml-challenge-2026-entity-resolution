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


def build_features(cands, s1, s23):
    """cands: DataFrame[s1, cand, <pass>, <pass>_rank]; s1/s23: prepared records."""
    rec = pd.concat([s1, s23], ignore_index=True).set_index("entity_id")
    X = cands[["s1", "cand"] + SCORE_COLS + RANK_COLS].copy()
    X["cand_is_s3"] = X.cand.str.startswith("S3-").astype(np.float32)
    X["n_passes"] = X[RANK_COLS].notna().sum(axis=1).astype(np.float32)
    X = X.join(string_features(X[["s1", "cand"]], rec))
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


def to_output(pairs, s1_ids, col):
    """Predicted/candidate pairs -> submission frame with one row per S1 (empty = none)."""
    lists = pairs.groupby("s1").cand.agg(lambda c: ",".join(sorted(set(c))))
    out = pd.DataFrame({"source1_entity_id": pd.Index(s1_ids)})
    out[col] = out.source1_entity_id.map(lists).fillna("")
    return out
