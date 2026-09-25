"""v2 submission: stage-1 (v1 model) -> group + sibling features -> stage-2 model.

    python src/predict_stack.py --data dataset --work work --exp work/v2 --out output_v2

Needs from earlier steps:
  --work/{train,test}_*_X.parquet   pair features (run_pipeline.py)
  --work/model.json                 stage-1 model trained on all of train (run_pipeline.py)
  --exp/train_oof.npy, train_e5.npy stage-1 out-of-fold probs + train embeddings (exp_stack.py)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ber.evaluate import true_pairs
from ber.match import add_labels, decide, feature_names, to_output
from ber.stack import group_features, sibling_similarity
from exp_stack import embeddings
from run_pipeline import PARAMS, dmatrix, load

STAGE2_TREES = 890     # 1.1 x the validation early-stopping optimum (809), for training on all data
THRESHOLD = 0.70       # best on validation


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_split(work, split):
    return pd.concat([pd.read_parquet(f"{work}/{f}") for f in sorted(os.listdir(work))
                      if f.startswith(f"{split}_") and f.endswith("_X.parquet")], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--exp", default="work/v2")
    ap.add_argument("--out", default="output_v2")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # stage-2 training table (train, out-of-fold stage-1 probs)
    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    X = add_labels(load_split(args.work, "train"), truth)
    feats1 = feature_names(X)
    oof = np.load(f"{args.exp}/train_oof.npy")
    G = group_features(X.s1.values, X.cand.values, oof)
    S = sibling_similarity(X.s1.values, X.cand.values, oof, embeddings(args.data, "train", args.exp))
    X = pd.concat([X.reset_index(drop=True), G.reset_index(drop=True), S], axis=1)
    feats2 = feats1 + list(G.columns) + list(S.columns)
    model2 = xgb.train(PARAMS, dmatrix(X, feats2), STAGE2_TREES)
    model2.save_model(f"{args.exp}/model_stage2.json")
    log("stage-2 model trained")
    del X, G, S

    # test: stage-1 probs from the all-train v1 model, then stage-2 features
    T = load_split(args.work, "test")
    model1 = xgb.Booster()
    model1.load_model(f"{args.work}/model.json")
    p1 = model1.predict(xgb.DMatrix(T[feats1].astype(np.float32)))
    G = group_features(T.s1.values, T.cand.values, p1)
    S = sibling_similarity(T.s1.values, T.cand.values, p1, embeddings(args.data, "test", args.exp))
    T = pd.concat([T.reset_index(drop=True), G.reset_index(drop=True), S], axis=1)
    p2 = model2.predict(xgb.DMatrix(T[feats2].astype(np.float32)))
    log("test scored")

    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    to_output(T[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(T, p2, THRESHOLD), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    json.dump({"stage2_trees": STAGE2_TREES, "threshold": THRESHOLD, "val_f05": 0.98191},
              open(f"{args.out}/model_meta.json", "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
