"""v3: new features + orphan-aware training/validation, then the test submission.

    python src/exp_v3.py --data dataset --feat work/v3 --out output_v3

Model selection uses the orphan-simulated validation (18% of S1 removed),
which tracks the leaderboard better than plain validation.
Compares: A = trained on the full table, B = trained on the orphan table.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from ber.augment import orphanize
from ber.evaluate import f05_macro, true_pairs
from ber.match import add_labels, decide, feature_names, to_output
from run_pipeline import PARAMS, dmatrix, is_val, load

ORPHAN_FRAC = 0.18
GRID = np.round(np.arange(0.60, 0.951, 0.025), 3)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load_split(feat, split):
    return pd.concat([pd.read_parquet(f"{feat}/{f}") for f in sorted(os.listdir(feat))
                      if f.startswith(f"{split}_") and f.endswith("_X.parquet")], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--out", default="output_v3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    s1_all = load(f"{args.data}/train/train_source1.tsv").entity_id
    val_ids = s1_all[is_val(s1_all.values)]
    full = add_labels(load_split(args.feat, "train"), truth)
    feats = feature_names(full)
    orph, keep = orphanize(full, ORPHAN_FRAC)
    val_ids_orph = pd.Index([i for i in val_ids if i in keep])
    log(f"train pairs: full {len(full):,}, orphan table {len(orph):,}; {len(feats)} features")

    evals = {"val_plain": (full[is_val(full.s1.values)].reset_index(drop=True), val_ids),
             "val_orphan": (orph[is_val(orph.s1.values)].reset_index(drop=True), val_ids_orph)}
    results, models = {}, {}
    for name, table in (("A_full", full), ("B_orphan", orph)):
        trn = table[~is_val(table.s1.values)]
        es = evals["val_orphan"][0]
        dtr = dmatrix(trn, feats)
        bst = xgb.train(PARAMS, dtr, 6000, evals=[(dmatrix(es, feats, ref=dtr), "val")],
                        early_stopping_rounds=100, verbose_eval=False)
        trees = bst.best_iteration + 1
        res = {"trees": trees}
        for ev, (vt, ids) in evals.items():
            p = bst.predict(xgb.DMatrix(vt[feats].astype(np.float32)), iteration_range=(0, trees))
            sc = {float(t): f05_macro(decide(vt, p, t), truth, ids) for t in GRID}
            best = max(sc, key=sc.get)
            res[ev] = {"best_threshold": best, "best_f05": sc[best], "curve": sc}
        results[name] = res
        models[name] = trees
        log(f"{name}: trees {trees} | plain {res['val_plain']['best_f05']:.5f}@{res['val_plain']['best_threshold']}"
            f" | orphan {res['val_orphan']['best_f05']:.5f}@{res['val_orphan']['best_threshold']}")
    json.dump(results, open(f"{args.out}/val_results.json", "w"), indent=2)

    choice = max(results, key=lambda k: results[k]["val_orphan"]["best_f05"])
    threshold = results[choice]["val_orphan"]["best_threshold"]
    table = full if choice == "A_full" else orph
    final = xgb.train(PARAMS, dmatrix(table, feats), int(models[choice] * 1.1))
    final.save_model(f"{args.out}/model.json")
    log(f"final model {choice}, threshold {threshold}")

    T = load_split(args.feat, "test")
    p = final.predict(xgb.DMatrix(T[feats].astype(np.float32)))
    test_ids = load(f"{args.data}/test/test_source1.tsv").entity_id
    to_output(T[["s1", "cand"]], test_ids, "candidate_entity_ids").to_csv(
        f"{args.out}/candidate_pairs.tsv", sep="\t", index=False)
    to_output(decide(T, p, threshold), test_ids, "matched_entity_ids").to_csv(
        f"{args.out}/matching_results.tsv", sep="\t", index=False)
    json.dump({"model": choice, "threshold": threshold, "trees": int(models[choice] * 1.1),
               "val_orphan_f05": results[choice]["val_orphan"]["best_f05"],
               "val_plain_f05": results[choice]["val_plain"]["best_f05"]},
              open(f"{args.out}/model_meta.json", "w"), indent=2)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
