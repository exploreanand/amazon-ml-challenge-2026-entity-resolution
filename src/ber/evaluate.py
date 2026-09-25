"""Blocking and matching metrics on a held-out split of the training data."""

import numpy as np
import pandas as pd


def true_pairs(gt):
    """Ground-truth file -> DataFrame[s1, cand] of true pairs (singletons dropped)."""
    ids = gt.matched_entity_ids.str.split(",")
    pairs = gt.assign(cand=ids).explode("cand")
    pairs = pairs[pairs.cand.notna() & (pairs.cand != "")]
    return pairs.rename(columns={"source1_entity_id": "s1"})[["s1", "cand"]].reset_index(drop=True)


def blocking_report(cands, truth, s1_ids):
    """Recall and size of a candidate set, restricted to the S1 ids in ``s1_ids``.

    cands, truth: DataFrame[s1, cand]. Returns a dict of summary numbers.
    """
    s1_ids = pd.Index(pd.unique(s1_ids))
    c = cands[cands.s1.isin(s1_ids)].drop_duplicates(["s1", "cand"])
    t = truth[truth.s1.isin(s1_ids)]
    hit = t.merge(c, on=["s1", "cand"], how="left", indicator=True)["_merge"].eq("both")
    per_s1 = t.assign(hit=hit.values).groupby("s1").hit.all()
    n_s1 = len(s1_ids)
    singles = s1_ids.difference(t.s1.unique())
    c_single = c[c.s1.isin(singles)]
    return {
        "pair_recall": hit.mean(),
        "s1_all_found": per_s1.mean(),
        "cands_per_s1": len(c) / n_s1,
        "cands_per_singleton": len(c_single) / max(len(singles), 1),
        "singletons_with_no_cands": 1 - c_single.s1.nunique() / max(len(singles), 1),
    }


def f05_macro(pred, truth, s1_ids):
    """Official metric: per-S1 F0.5, averaged over ``s1_ids`` (singletons included)."""
    s1_ids = pd.Index(pd.unique(s1_ids))
    p = pred[pred.s1.isin(s1_ids)].drop_duplicates(["s1", "cand"])
    t = truth[truth.s1.isin(s1_ids)]
    tp = p.merge(t, on=["s1", "cand"]).groupby("s1").size()
    n_p = p.groupby("s1").size().reindex(s1_ids, fill_value=0)
    n_t = t.groupby("s1").size().reindex(s1_ids, fill_value=0)
    tp = tp.reindex(s1_ids, fill_value=0)
    prec = np.where(n_p > 0, tp / n_p.clip(lower=1), 0.0)
    rec = np.where(n_t > 0, tp / n_t.clip(lower=1), 0.0)
    f = np.where(prec + rec > 0, 1.25 * prec * rec / (0.25 * prec + rec + 1e-12), 0.0)
    f = np.where((n_t == 0) & (n_p == 0), 1.0, f)
    return float(f.mean())
