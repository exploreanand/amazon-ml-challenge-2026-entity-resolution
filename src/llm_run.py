"""v8 step: fine-tune a Qwen3 pair classifier (LoRA) and score v7's uncertain pairs.

    python src/llm_run.py --data dataset --work work --feat work/v3 --v7 output_v7 --out work/qwen17 --base Qwen/Qwen3-1.7B

Training pairs: training-split S1 ids only (never validation), the hard band
used for the cross-encoders (out-of-fold stage-1 p in (0.005, 0.995)),
subsampled to N_HARD, plus N_EASY confident pairs.
Scored pairs: exactly the pairs v7's cross-encoder scored (v7 val_band.parquet
and the test pairs with a cross-encoder score in v7 test_probs.parquet), so
the new score can be blended into v7 directly.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from ber.evaluate import true_pairs
from ber.llm_classifier import score, train
from ber.match import add_labels
from ce_run import texts_for
from run_pipeline import is_val, load

LO, HI = 0.005, 0.995
N_HARD, N_EASY = 800_000, 200_000


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--v7", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
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
    hard = tr[(tr.p > LO) & (tr.p < HI)].sample(N_HARD, random_state=0)
    easy = tr[(tr.p <= LO) | (tr.p >= HI)].sample(N_EASY, random_state=0)
    tr = pd.concat([hard, easy]).sample(frac=1, random_state=0)
    log(f"training pairs {len(tr):,}, positives {tr.label.mean():.1%}")

    vb = pd.read_parquet(f"{args.v7}/val_band.parquet")[["s1", "cand", "label"]]
    ev = vb.sample(20_000, random_state=1)
    if not os.path.exists(f"{args.out}/best/adapter_config.json"):
        train(txt.loc[tr.s1].tolist(), txt.loc[tr.cand].tolist(), tr.label.values.astype(np.float32),
              args.out, args.base, epochs=args.epochs, batch=args.batch, log=log,
              eval_data=(txt.loc[ev.s1].tolist(), txt.loc[ev.cand].tolist(), ev.label.values))
    log("trained")

    vb = vb.assign(llm=score(txt.loc[vb.s1].tolist(), txt.loc[vb.cand].tolist(), args.base, f"{args.out}/best"))
    vb.to_parquet(f"{args.out}/val_band.parquet")
    log(f"scored {len(vb):,} val band pairs")
    T = pd.read_parquet(f"{args.v7}/test_probs.parquet", columns=["s1", "cand", "ce"])
    tb = T[T.ce.notna()][["s1", "cand"]]
    ttxt = texts_for(args.data, "test")
    tb = tb.assign(llm=score(ttxt.loc[tb.s1].tolist(), ttxt.loc[tb.cand].tolist(), args.base, f"{args.out}/best"))
    tb.to_parquet(f"{args.out}/test_band.parquet")
    log(f"scored {len(tb):,} test band pairs; DONE")


if __name__ == "__main__":
    main()
