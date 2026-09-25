"""End-to-end pipeline: data -> blocking -> features -> model -> output files.

Usage (GPU needed for blocking):
    python src/run_pipeline.py --data dataset --work work --out output

Steps
  1. For every country label found in each split (never a fixed list), build
     candidates with ber.pipeline.candidates_for_partition.
  2. Build pair features (ber.match.build_features).
  3. Train XGBoost (GPU) on the training split, holding out 20% of train S1 ids
     (hash bucket) to pick the number of trees and the decision threshold.
  4. Score test candidates and write output/matching_results.tsv and
     output/candidate_pairs.tsv (the exact set the model scored).
"""

import argparse
import csv
import json
import os
import time

import xgboost as xgb
import numpy as np
import pandas as pd

from ber.evaluate import f05_macro, true_pairs
from ber.match import add_labels, build_features, decide, feature_names, to_output
from ber.pipeline import candidates_for_partition, prepare

SPLIT_KEY = "ber2026split0001"
PARAMS = dict(objective="binary:logistic", eval_metric="logloss", device="cuda", tree_method="hist",
              max_depth=9, eta=0.1, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, seed=0)


def dmatrix(X, feats, ref=None):
    label = X.label if "label" in X else None
    return xgb.QuantileDMatrix(X[feats].astype(np.float32), label, max_bin=256, ref=ref)


def load(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE)


def is_val(ids):
    h = pd.util.hash_pandas_object(pd.Series(ids), index=False, hash_key=SPLIT_KEY).values
    return (h % 5) == 0


def build_split(split, data_dir, work):
    s = {k: load(f"{data_dir}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)}
    for country in s[1].country.value_counts().index:
        tag = f"{split}_{country}"
        if os.path.exists(f"{work}/{tag}_X.parquet"):
            continue
        t = time.time()
        p1 = prepare(s[1][s[1].country == country])
        p23 = prepare(pd.concat([s[k][s[k].country == country] for k in (2, 3)]))
        c = candidates_for_partition(p1, p23)
        c.insert(0, "cand", p23.entity_id.values[c.i23.values])
        c.insert(0, "s1", p1.entity_id.values[c.i1.values])
        X = build_features(c, p1, p23)
        X.to_parquet(f"{work}/{tag}_X.parquet")
        print(f"{tag}: {len(X):,} candidate pairs in {time.time() - t:.0f}s", flush=True)
    return s[1].entity_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    build_split("train", args.data, args.work)
    test_ids = build_split("test", args.data, args.work)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    X = pd.concat([pd.read_parquet(f"{args.work}/{f}") for f in sorted(os.listdir(args.work))
                   if f.startswith("train_") and f.endswith("_X.parquet")], ignore_index=True)
    X = add_labels(X, truth)
    X["is_val"] = is_val(X.s1.values)
    feats = feature_names(X.drop(columns=["is_val"]))
    trn, val = X[~X.is_val], X[X.is_val]
    dtr = dmatrix(trn, feats)
    dva = dmatrix(val, feats, ref=dtr)
    model = xgb.train(PARAMS, dtr, 3000, evals=[(dva, "val")], early_stopping_rounds=50,
                      verbose_eval=False)
    trees = model.best_iteration + 1
    p_val = model.predict(dva, iteration_range=(0, trees))
    all_s1 = load(f"{args.data}/train/train_source1.tsv").entity_id
    val_ids = all_s1[is_val(all_s1.values)]
    grid = np.round(np.arange(0.40, 0.951, 0.025), 3)
    scores = {float(t): f05_macro(decide(val, p_val, t), truth, val_ids) for t in grid}
    threshold = max(scores, key=scores.get)
    print(f"validation F0.5 {scores[threshold]:.4f} at threshold {threshold}", flush=True)

    # refit on all training S1 with the chosen number of trees
    final = xgb.train(PARAMS, dmatrix(X, feats), trees)
    final.save_model(f"{args.work}/model.json")
    json.dump({"threshold": threshold, "val_f05": scores[threshold], "trees": trees},
              open(f"{args.work}/model_meta.json", "w"), indent=2)

    T = pd.concat([pd.read_parquet(f"{args.work}/{f}") for f in sorted(os.listdir(args.work))
                   if f.startswith("test_") and f.endswith("_X.parquet")], ignore_index=True)
    p = final.predict(xgb.DMatrix(T[feats].astype(np.float32)))
    to_output(T[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(T, p, threshold), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
