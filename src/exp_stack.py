"""v2 experiment: does a stacked second-stage model beat v1 on validation?

    python src/exp_stack.py --data dataset --work work --exp work/v2

Reuses the v1 feature tables in --work; writes everything new to --exp.
"""

import argparse
import html
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ber.evaluate import f05_macro, true_pairs
from ber.match import add_labels, decide, feature_names
from ber.pipeline import encode_e5
from ber.stack import group_features, oof_predict, sibling_similarity
from run_pipeline import PARAMS, dmatrix, is_val, load

STAGE1_TREES = 2742  # early-stopping optimum found for v1 features


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def embeddings(data, split, exp):
    path = f"{exp}/{split}_e5.npy"
    recs = pd.concat([load(f"{data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)], ignore_index=True)
    index_map = pd.Series(np.arange(len(recs)), index=recs.entity_id.values)
    if os.path.exists(path):
        return index_map, np.load(path)
    texts = ("query: " + recs.business_name.map(html.unescape) + ", "
             + recs.business_address.map(html.unescape))
    E = encode_e5(texts)
    np.save(path, E)
    return index_map, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--exp", default="work/v2")
    args = ap.parse_args()
    os.makedirs(args.exp, exist_ok=True)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    X = pd.concat([pd.read_parquet(f"{args.work}/{f}") for f in sorted(os.listdir(args.work))
                   if f.startswith("train_") and f.endswith("_X.parquet")], ignore_index=True)
    X = add_labels(X, truth)
    X["is_val"] = is_val(X.s1.values)
    feats1 = feature_names(X.drop(columns=["is_val"]))
    log(f"loaded {len(X):,} train pairs")

    oof_path = f"{args.exp}/train_oof.npy"
    if os.path.exists(oof_path):
        oof = np.load(oof_path)
    else:
        oof = oof_predict(X, feats1, PARAMS, STAGE1_TREES, dmatrix)
        np.save(oof_path, oof)
    log("stage-1 out-of-fold done")

    G = group_features(X.s1.values, X.cand.values, oof)
    log("group features done")
    S = sibling_similarity(X.s1.values, X.cand.values, oof, embeddings(args.data, "train", args.exp))
    log("sibling features done")
    X = pd.concat([X.reset_index(drop=True), G.reset_index(drop=True), S], axis=1)

    s1_all = load(f"{args.data}/train/train_source1.tsv").entity_id
    val_ids = s1_all[is_val(s1_all.values)]
    trn, val = X[~X.is_val], X[X.is_val].reset_index(drop=True)
    results = {}
    for name, feats in {"stage1_only_oof": feats1 + ["p1"],
                        "stage2_group": feats1 + list(G.columns),
                        "stage2_group_sib": feats1 + list(G.columns) + list(S.columns)}.items():
        dtr, dva = dmatrix(trn, feats), None
        dva = dmatrix(val, feats, ref=dtr)
        bst = xgb.train(PARAMS, dtr, 6000, evals=[(dva, "val")], early_stopping_rounds=100,
                        verbose_eval=False)
        p = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        scores = {t: f05_macro(decide(val, p, t), truth, val_ids)
                  for t in np.round(np.arange(0.5, 0.91, 0.05), 2)}
        best = max(scores, key=scores.get)
        results[name] = {"trees": bst.best_iteration + 1, "threshold": float(best), "val_f05": scores[best]}
        log(f"{name}: {results[name]}")
        np.save(f"{args.exp}/val_p_{name}.npy", p)
    json.dump(results, open(f"{args.exp}/results.json", "w"), indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
