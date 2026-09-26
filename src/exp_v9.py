"""v9: a second cross-encoder (bge-reranker-v2-m3, French-style augmentation) in the v7.2 blend.

    python src/exp_v9.py --data dataset --ce2 work/ce_bge_fr/model/best --out output_v9

Nothing else is retrained. v7's XGBoost probabilities and e5-large scores come
from output_v7/val_band.parquet (uncertain validation pairs), output_v72s/
val_high.parquet (validation pairs XGBoost is sure about) and output_v72/
test_probs.parquet; the new model scores the same pairs. The logistic blend
gets both cross-encoders, and the protected veto and the country-word rule work
as in v7.2 (exp_v81.py). Validation F0.5 (two-fold cross-fit, as in exp_v5) is
reported next to v7.2's, and on France the share of XGBoost-sure pairs each
cross-encoder rejects is reported per name-variant kind.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ber.cross_encoder import score
from ber.evaluate import f05_macro, true_pairs
from ber.match import decide, to_output
from exp_v5 import GRID, HI, LO, THRESHOLD, log, logit, texts
from exp_v81 import country_added, halves, kinds, names
from run_pipeline import is_val, load


def z2(p, ce1, ce2):
    lp, l1, l2 = logit(np.asarray(p, dtype=np.float64)), logit(np.asarray(ce1)), logit(np.asarray(ce2))
    return np.c_[lp, l1, l2, lp * l1, lp * l2]


def ce2_scores(pairs, txt, model, path):
    """Scores of the second cross-encoder for pairs[s1, cand], computed once and kept at ``path``."""
    if not os.path.exists(path):
        s = score(txt.loc[pairs.s1].tolist(), txt.loc[pairs.cand].tolist(), model)
        pairs[["s1", "cand"]].assign(ce2=s).to_parquet(path)
    return pairs[["s1", "cand"]].merge(pd.read_parquet(path), on=["s1", "cand"], how="left").ce2.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--v7", default="output_v7")
    ap.add_argument("--v72s", default="output_v72s", help="exp_v81 rerun of v7.2 that saved val_final / val_high")
    ap.add_argument("--v72", default="output_v72")
    ap.add_argument("--ce2", required=True)
    ap.add_argument("--out", default="output_v9")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {"ce2": args.ce2}

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1v = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1v = s1v[is_val(s1v.entity_id.values)]
    vb = pd.read_parquet(f"{args.v7}/val_band.parquet", columns=["s1", "cand", "label", "p", "ce"])
    vh = pd.read_parquet(f"{args.v72s}/val_high.parquet", columns=["s1", "cand", "label", "p", "ce"])
    V = pd.read_parquet(f"{args.v72s}/val_final.parquet", columns=["s1", "cand", "label", "base", "veto_protected_country"])
    txt = texts(args.data, "train")
    vb["ce2"] = ce2_scores(vb, txt, args.ce2, f"{args.out}/val_band_ce2.parquet")
    vh["ce2"] = ce2_scores(vh, txt, args.ce2, f"{args.out}/val_high_ce2.parquet")
    report["val_band_auc"] = {"e5_large": roc_auc_score(vb.label, vb.ce), "bge": roc_auc_score(vb.label, vb.ce2)}
    log(f"uncertain validation pairs, AUC: {report['val_band_auc']}")

    hb, hh = halves(vb.s1), halves(vh.s1)
    pb, ph = np.zeros(len(vb)), np.zeros(len(vh))
    for a in (True, False):  # the blender fitted on one half of the S1s scores the other half
        m = LogisticRegression(max_iter=1000).fit(z2(vb.p[hb == a], vb.ce[hb == a], vb.ce2[hb == a]), vb.label[hb == a])
        pb[hb != a] = m.predict_proba(z2(vb.p[hb != a], vb.ce[hb != a], vb.ce2[hb != a]))[:, 1]
        ph[hh != a] = m.predict_proba(z2(np.minimum(vh.p[hh != a], HI), vh.ce[hh != a], vh.ce2[hh != a]))[:, 1]
    hk = kinds(vh, names(args.data, "train"))
    ph = np.where(hk == "other", ph, vh.p.values)  # protected veto
    vv = V.merge(vb[["s1", "cand"]].assign(bl=pb), on=["s1", "cand"], how="left")
    vv = vv.merge(vh[["s1", "cand"]].assign(hp=ph), on=["s1", "cand"], how="left")
    pf = vv.hp.fillna(vv.bl).fillna(vv.base).values
    maybe = pf >= 0.5
    ca = np.zeros(len(vv), dtype=bool)
    ca[maybe] = country_added(vv[maybe], names(args.data, "train"), vv.s1[maybe].map(s1v.set_index("entity_id").country).values)
    pf = np.where(ca, 0.0, pf)
    vv[["s1", "cand", "label"]].assign(v9=pf).to_parquet(f"{args.out}/val_final.parquet")
    for name, p in (("v7.2", vv.veto_protected_country.values), ("v9", pf)):
        report[f"val_{name}"] = {t: f05_macro(decide(vv, p, t), truth, s1v.entity_id) for t in GRID}
        pred = decide(vv, p, THRESHOLD)
        report[f"val_{name}_by_country"] = {c: f05_macro(pred, truth, g.entity_id) for c, g in s1v.groupby("country")}
        log(f"val {name}: {report[f'val_{name}']} | at {THRESHOLD} by country {report[f'val_{name}_by_country']}")
    del vv, V, txt

    blender = LogisticRegression(max_iter=1000).fit(z2(vb.p, vb.ce, vb.ce2), vb.label)
    T = pd.read_parquet(f"{args.v72}/test_probs.parquet", columns=["s1", "cand", "p", "ce"])
    has = T.ce.notna().values
    ttxt = texts(args.data, "test")
    T["ce2"] = np.nan
    T.loc[has, "ce2"] = ce2_scores(T[has], ttxt, args.ce2, f"{args.out}/test_ce2.parquet")
    band, high = ((T.p > LO) & (T.p < HI)).values, (T.p >= HI).values
    T["p_final"] = T.p.astype(np.float64)
    T.loc[band, "p_final"] = blender.predict_proba(z2(T.p[band], T.ce[band], T.ce2[band]))[:, 1]
    tn = names(args.data, "test")
    tk = kinds(T[high], tn)
    veto = blender.predict_proba(z2(np.minimum(T.p[high], HI), T.ce[high], T.ce2[high]))[:, 1]
    T.loc[high, "p_final"] = np.where(tk == "other", veto, T.p[high].values)
    t1 = load(f"{args.data}/test/test_source1.tsv")
    country = T.s1.map(t1.set_index("entity_id").country)
    maybe = (T.p_final >= 0.5).values
    ca = np.zeros(len(T), dtype=bool)
    ca[maybe] = country_added(T[maybe], tn, country[maybe].values)
    T.loc[ca, "p_final"] = 0.0
    T.to_parquet(f"{args.out}/test_probs.parquet")

    fr = (country[high] == "France").values
    report["france_sure_pairs_rejected"] = {  # share with cross-encoder < 0.2, per name-variant kind
        k: {"pairs": int((fr & (tk == k)).sum()),
            "e5_large": float((T.ce[high].values[fr & (tk == k)] < 0.2).mean()),
            "bge": float((T.ce2[high].values[fr & (tk == k)] < 0.2).mean())} for k in pd.unique(tk)}
    log(f"France, XGBoost-sure pairs rejected by each cross-encoder: {report['france_sure_pairs_rejected']}")

    to_output(T[["s1", "cand"]], t1.entity_id, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    out = to_output(decide(T, T.p_final.values, THRESHOLD), t1.entity_id, "matched_entity_ids")
    out.to_csv(f"{args.out}/matching_results.tsv", sep="\t", index=False)
    prev = pd.read_csv(f"{args.v72}/matching_results.tsv", sep="\t", dtype=str, keep_default_na=False)
    changed = out.matched_entity_ids.values != prev.set_index("source1_entity_id").matched_entity_ids.reindex(out.source1_entity_id).values
    report["test_rows_changed_vs_v72"] = pd.Series(changed).groupby(
        t1.set_index("entity_id").country.reindex(out.source1_entity_id).values).mean().to_dict()
    log(f"test rows changed vs v7.2: {report['test_rows_changed_vs_v72']}")
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
