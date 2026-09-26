"""v5: XGBoost on bridged candidate sets + cross-encoder blend; validation report and submission.

    python src/exp_v5.py --data dataset --feat work/v5 --ce-model work/ce_base_3ep/model/best --out output_v5

--ce-cache DIR reuses the cross-encoder scores of an earlier run with the same
--ce-model (DIR/val_band.parquet, DIR/test_probs.parquet); only pairs that are
new to the uncertain band get scored.

  1. XGBoost, v3 recipe (18%-orphan training table, 2613 trees), training split only.
  2. Cross-encoder (given checkpoint) scores every val/test pair with XGBoost p in (LO, HI).
  3. Logistic blend fitted on validation; for the report it is cross-fitted on
     two halves of the validation S1s so the numbers are not optimistic.
  4. Submission at THRESHOLD, plus the validation table in val_report.json.
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from ber.augment import orphanize
from ber.cross_encoder import record_text, score
from ber.evaluate import blocking_report, f05_macro, true_pairs
from ber.match import add_labels, decide, feature_names, to_output
from run_pipeline import PARAMS, dmatrix, is_val, load

TREES, LO, HI, THRESHOLD = 2613, 0.005, 0.995, 0.85
GRID = (0.75, 0.8, 0.85, 0.9)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def z(p, ce):
    return np.c_[logit(p), logit(ce), logit(p) * logit(ce)]


def load_split(feat, split):
    return pd.concat([pd.read_parquet(f"{feat}/{f}") for f in sorted(os.listdir(feat))
                      if f.startswith(f"{split}_") and f.endswith("_X.parquet")], ignore_index=True)


def texts(data, split):
    r = pd.concat([load(f"{data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)], ignore_index=True)
    return pd.Series(record_text(r.business_name, r.business_address), index=r.entity_id.values)


def ce_scores(pairs, txt, model, cache=None):
    """Cross-encoder scores for pairs[s1, cand]; pairs found in cache[s1, cand, ce] are not rescored."""
    d = pairs[["s1", "cand"]].reset_index(drop=True)
    d = d.merge(cache, on=["s1", "cand"], how="left") if cache is not None else d.assign(ce=np.nan)
    miss = d.ce.isna().values
    log(f"cross-encoder: {len(d) - miss.sum():,} pairs cached, scoring {miss.sum():,}")
    if miss.any():
        d.loc[miss, "ce"] = score(txt.loc[d.s1[miss]].tolist(), txt.loc[d.cand[miss]].tolist(), model)
    return d.ce.values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="work/v5")
    ap.add_argument("--ce-model", required=True)
    ap.add_argument("--out", default="output_v5")
    ap.add_argument("--ce-cache", help="earlier output dir scored with the same --ce-model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    report = {}
    vcache = tcache = None
    if args.ce_cache:
        vcache = pd.read_parquet(f"{args.ce_cache}/val_band.parquet", columns=["s1", "cand", "ce"])
        tcache = pd.read_parquet(f"{args.ce_cache}/test_probs.parquet", columns=["s1", "cand", "ce"]).dropna()

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1_all = load(f"{args.data}/train/train_source1.tsv")[["entity_id", "country"]]
    s1_all = s1_all[is_val(s1_all.entity_id.values)]
    val_ids = s1_all.entity_id
    full = add_labels(load_split(args.feat, "train"), truth)
    feats = feature_names(full)
    report["blocking_val"] = blocking_report(full[["s1", "cand"]], truth, val_ids)
    log(f"blocking on val: {report['blocking_val']}")
    orph, _ = orphanize(full, 0.18)
    bst = xgb.train(PARAMS, dmatrix(orph[~is_val(orph.s1.values)], feats), TREES)
    bst.save_model(f"{args.out}/xgb_trainsplit.json")
    del orph
    val = full[is_val(full.s1.values)].reset_index(drop=True)
    del full
    val["p"] = bst.predict(xgb.DMatrix(val[feats].astype(np.float32)))
    report["val_xgb"] = {t: f05_macro(decide(val, val.p.values, t), truth, val_ids) for t in GRID}
    log(f"val xgb only: {report['val_xgb']}")

    txt = texts(args.data, "train")
    vb = val[(val.p > LO) & (val.p < HI)][["s1", "cand", "label", "p"]].reset_index(drop=True)
    vb["ce"] = ce_scores(vb, txt, args.ce_model, vcache)
    vb.to_parquet(f"{args.out}/val_band.parquet")
    half = (pd.util.hash_pandas_object(vb.s1, index=False).values % 2) == 0
    pb = np.zeros(len(vb))
    for a, b in ((half, ~half), (~half, half)):
        pb[b] = LogisticRegression(max_iter=1000).fit(z(vb.p[a], vb.ce[a]), vb.label[a]).predict_proba(z(vb.p[b], vb.ce[b]))[:, 1]
    vv = val[["s1", "cand", "p"]].merge(vb[["s1", "cand"]].assign(bl=pb), on=["s1", "cand"], how="left")
    pf = vv.bl.fillna(vv.p).values
    report["val_xgb_ce"] = {t: f05_macro(decide(vv, pf, t), truth, val_ids) for t in GRID}
    log(f"val xgb + cross-encoder: {report['val_xgb_ce']}")
    pred = decide(vv, pf, THRESHOLD)
    report["val_xgb_ce_by_country"] = {c: f05_macro(pred, truth, g.entity_id) for c, g in s1_all.groupby("country")}
    log(f"val xgb + cross-encoder at {THRESHOLD} by country: {report['val_xgb_ce_by_country']}")
    blender = LogisticRegression(max_iter=1000).fit(z(vb.p, vb.ce), vb.label)
    pickle.dump(blender, open(f"{args.out}/blender.pkl", "wb"))
    del val, vv, txt

    T = load_split(args.feat, "test")
    out = T[["s1", "cand"]].copy()
    out["p"] = bst.predict(xgb.DMatrix(T[feats].astype(np.float32)))
    del T
    band = (out.p > LO) & (out.p < HI)
    ttxt = texts(args.data, "test")
    out["ce"] = np.nan
    out.loc[band, "ce"] = ce_scores(out[band], ttxt, args.ce_model, tcache)
    out["p_final"] = out.p.astype(np.float64)
    out.loc[band, "p_final"] = blender.predict_proba(z(out.p[band], out.ce[band]))[:, 1]
    out.to_parquet(f"{args.out}/test_probs.parquet")
    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    to_output(out[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(out, out.p_final.values, THRESHOLD), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    report.update(threshold=THRESHOLD, ce_model=args.ce_model, ce_cache=args.ce_cache,
                  test_pairs=len(out), test_band=int(band.sum()))
    json.dump(report, open(f"{args.out}/val_report.json", "w"), indent=2, default=float)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
