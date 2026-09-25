"""v4 step 1: train the cross-encoder on hard training pairs, score uncertain val/test pairs.

    python src/ce_run.py --data dataset --work work --feat work/v3 --probs runs/v3_lb0.97173/test_probs.parquet --out work/ce

Training pairs come only from training-split S1 ids (never validation):
all pairs whose out-of-fold stage-1 probability is in (LO, HI), plus a random
sample of the confident ones. Scored pairs: val and test pairs in the same band.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from ber.cross_encoder import record_text, score, train
from ber.evaluate import true_pairs
from ber.match import add_labels
from run_pipeline import is_val, load

LO, HI = 0.005, 0.995
N_EASY = 1_000_000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def texts_for(data, split):
    recs = pd.concat([load(f"{data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)], ignore_index=True)
    return pd.Series(record_text(recs.business_name, recs.business_address), index=recs.entity_id.values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out", default="work/ce")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    X = pd.concat([pd.read_parquet(f"{args.feat}/train_{c}_X.parquet", columns=["s1", "cand"])
                   for c in ("India", "US")], ignore_index=True)  # same order as the saved OOF
    X = add_labels(X, truth)
    X["p"] = np.load(f"{args.work}/v2/train_oof.npy")
    X["val"] = is_val(X.s1.values)
    txt = texts_for(args.data, "train")

    tr = X[~X.val]
    hard = tr[(tr.p > LO) & (tr.p < HI)]
    easy = tr[(tr.p <= LO) | (tr.p >= HI)].sample(N_EASY, random_state=0)
    tr = pd.concat([hard, easy]).sample(frac=1, random_state=0)
    log(f"training pairs {len(tr):,} (hard {len(hard):,}, easy {len(easy):,}), positives {tr.label.mean():.1%}")
    model_dir = f"{args.out}/model"
    if not os.path.exists(f"{model_dir}/config.json"):
        train(txt.loc[tr.s1].tolist(), txt.loc[tr.cand].tolist(), tr.label.values.astype(np.float32),
              model_dir, log=log)
    log("trained")

    va = X[X.val & (X.p > LO) & (X.p < HI)]
    va = va.assign(ce=score(txt.loc[va.s1].tolist(), txt.loc[va.cand].tolist(), model_dir))
    va[["s1", "cand", "label", "p", "ce"]].to_parquet(f"{args.out}/val_band.parquet")
    log(f"scored {len(va):,} val band pairs")

    T = pd.read_parquet(args.probs)
    tb = T[(T.p > LO) & (T.p < HI)]
    ttxt = texts_for(args.data, "test")
    tb = tb.assign(ce=score(ttxt.loc[tb.s1].tolist(), ttxt.loc[tb.cand].tolist(), model_dir))
    tb[["s1", "cand", "p", "ce"]].to_parquet(f"{args.out}/test_band.parquet")
    log(f"scored {len(tb):,} test band pairs; DONE")


if __name__ == "__main__":
    main()
