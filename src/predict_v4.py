"""v4 submission: v3-recipe XGBoost + cross-encoder on uncertain pairs, logistic blend.

    python src/predict_v4.py --data dataset --feat work/v3 --ce work/ce --out output_v4

Needs from earlier steps:
  --feat/{train,test}_*_X.parquet       v3 pair features (rebuild_features.py)
  --ce/val_band.parquet, test_band.parquet, model/   cross-encoder scores + model (ce_run.py)

Steps
  1. XGBoost with the v3 recipe (18%-orphan training table, 2613 trees) on the
     training split only, so its validation probabilities are honest.
  2. On validation pairs the cross-encoder scored, fit a logistic blend of
     [logit p_xgb, logit p_ce, their product] -> match probability.
  3. Test: p_xgb for every pair; pairs the cross-encoder scored get the blend.
  4. Decision rule as before (each S2/S3 keeps its best S1, then threshold).
     Threshold 0.85: stricter than the validation optimum (0.80) because the
     leaderboard rewarded stricter thresholds for v3.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from ber.augment import orphanize
from ber.evaluate import true_pairs
from ber.match import add_labels, decide, feature_names, to_output
from run_pipeline import PARAMS, dmatrix, is_val, load

V3_TREES = 2613
THRESHOLD = 0.85


def logit(x):
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x))


def blend_inputs(p, ce):
    return np.c_[logit(p), logit(ce), logit(p) * logit(ce)]


def load_split(feat, split):
    return pd.concat([pd.read_parquet(f"{feat}/{f}") for f in sorted(os.listdir(feat))
                      if f.startswith(f"{split}_") and f.endswith("_X.parquet")], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--ce", default="work/ce")
    ap.add_argument("--out", default="output_v4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    full = add_labels(load_split(args.feat, "train"), truth)
    feats = feature_names(full)
    orph, _ = orphanize(full, 0.18)
    bst = xgb.train(PARAMS, dmatrix(orph[~is_val(orph.s1.values)], feats), V3_TREES)
    bst.save_model(f"{args.out}/xgb_trainsplit.json")
    del orph

    val = full[is_val(full.s1.values)][["s1", "cand"] + feats].reset_index(drop=True)
    del full
    val["p"] = bst.predict(xgb.DMatrix(val[feats].astype(np.float32)))
    vb = pd.read_parquet(f"{args.ce}/val_band.parquet")[["s1", "cand", "label", "ce"]]
    vb = vb.merge(val[["s1", "cand", "p"]], on=["s1", "cand"])
    blender = LogisticRegression(C=1.0, max_iter=1000).fit(blend_inputs(vb.p, vb.ce), vb.label)
    pickle.dump(blender, open(f"{args.out}/blender.pkl", "wb"))

    T = load_split(args.feat, "test")
    out = T[["s1", "cand"]].copy()
    out["p"] = bst.predict(xgb.DMatrix(T[feats].astype(np.float32)))
    del T
    out = out.merge(pd.read_parquet(f"{args.ce}/test_band.parquet")[["s1", "cand", "ce"]],
                    on=["s1", "cand"], how="left")
    has = out.ce.notna().values
    out["p_final"] = out.p.astype(np.float64)
    out.loc[has, "p_final"] = blender.predict_proba(blend_inputs(out.p[has], out.ce[has]))[:, 1]

    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    to_output(out[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(out, out.p_final.values, THRESHOLD), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    json.dump({"threshold": THRESHOLD, "xgb_trees": V3_TREES, "blend_coef": blender.coef_.tolist(),
               "blend_intercept": blender.intercept_.tolist()}, open(f"{args.out}/model_meta.json", "w"), indent=2)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
