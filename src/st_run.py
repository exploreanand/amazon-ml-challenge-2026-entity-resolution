"""v11 experiment (B): self-training of the French-aware bge cross-encoder on confident France test pairs.

    python src/st_run.py --data dataset --work work --feat work/v3 --init work/ce_bge_fr/model/best \
        --pred runs/v10b_lb0.98464/output/matching_results.tsv --out work/ce_bge_st

France has no labels. Pseudo-labels come from test pairs every model agrees on in
the v10b run (no ground truth is used):
  positive  a France pair v10b predicts, with XGBoost p >= 0.9 and both cross-encoders >= 0.9
  negative  a France pair v10b rejects among the cross-encoder-scored ones, both cross-encoders <= 0.05
Up to --n-pseudo of each are mixed with as many original hard training pairs (the
ce_run selection, 30% rewritten French-style), and --init is fine-tuned for one
epoch at a low learning rate. The usual US/India validation sample is monitored so
that forgetting shows up in the log.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from ber.augment_fr import frenchify
from ber.cross_encoder import train
from ber.evaluate import true_pairs
from ber.match import add_labels
from ce_run import HI, LO, texts_for
from run_pipeline import is_val, load


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--init", required=True, help="cross-encoder checkpoint to continue from")
    ap.add_argument("--v72", default="output_v72", help="test XGBoost p and e5-large scores")
    ap.add_argument("--v9", default="output_v9", help="bge test scores")
    ap.add_argument("--v10b", default="output_v10b", help="French e5-large test scores")
    ap.add_argument("--pred", required=True, help="the v10b matching_results.tsv")
    ap.add_argument("--n-pseudo", type=int, default=300_000)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--eval-n", type=int, default=50_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(0)

    T = pd.read_parquet(f"{args.v72}/test_probs.parquet", columns=["s1", "cand", "p", "ce"])
    T = T[T.ce.notna()]
    T = T.merge(pd.read_parquet(f"{args.v9}/test_ce2.parquet"), on=["s1", "cand"])
    T = T.merge(pd.read_parquet(f"{args.v10b}/test_ce3.parquet").rename(columns={"ce2": "ce3"}), on=["s1", "cand"])
    t1 = load(f"{args.data}/test/test_source1.tsv").set_index("entity_id")
    T = T[T.s1.map(t1.country).eq("France").values]
    pred = load(args.pred)
    pred = pred[pred.matched_entity_ids != ""].assign(cand=lambda f: f.matched_entity_ids.str.split(",")).explode("cand")
    predicted = set(zip(pred.source1_entity_id, pred.cand))
    isp = np.array([k in predicted for k in zip(T.s1, T.cand)])
    pos = T[isp & (T.p >= 0.9) & (T.ce2 >= 0.9) & (T.ce3 >= 0.9)]
    neg = T[~isp & (T.ce2 <= 0.05) & (T.ce3 <= 0.05)]
    log(f"France pseudo-labels available: {len(pos):,} positive, {len(neg):,} negative")
    pos = pos.sample(min(args.n_pseudo, len(pos)), random_state=0)
    neg = neg.sample(min(args.n_pseudo, len(neg)), random_state=0)
    ttxt = texts_for(args.data, "test")
    pa = ttxt.loc[pos.s1].tolist() + ttxt.loc[neg.s1].tolist()
    pb = ttxt.loc[pos.cand].tolist() + ttxt.loc[neg.cand].tolist()
    py = [1.0] * len(pos) + [0.0] * len(neg)
    del ttxt

    truth = true_pairs(load(f"{args.data}/train/train_ground_truth.tsv"))
    X = pd.concat([pd.read_parquet(f"{args.feat}/{f}", columns=["s1", "cand"]) for f in sorted(os.listdir(args.feat))
                   if f.startswith("train_") and f.endswith("_X.parquet")], ignore_index=True)  # exp_stack's OOF order
    X = add_labels(X, truth)
    X["p"] = np.load(f"{args.work}/v2/train_oof.npy")
    X["val"] = is_val(X.s1.values)
    txt = texts_for(args.data, "train")
    hard = X[~X.val & (X.p > LO) & (X.p < HI)]
    orig = hard.sample(min(len(pa), len(hard)), random_state=1)
    oa, ob = txt.loc[orig.s1].tolist(), txt.loc[orig.cand].tolist()
    for i in np.flatnonzero(rng.random(len(oa)) < 0.3):
        oa[i], ob[i] = frenchify(oa[i], ob[i], rng)
    a, b = pa + oa, pb + ob
    y = np.array(py + orig.label.astype(float).tolist(), dtype=np.float32)
    perm = rng.permutation(len(a))
    a, b, y = [a[i] for i in perm], [b[i] for i in perm], y[perm]
    log(f"training pairs: {len(a):,} ({len(pa):,} France pseudo-labelled, {len(oa):,} original)")
    ev = X[X.val & (X.p > LO) & (X.p < HI)].sample(args.eval_n, random_state=1)
    eval_data = (txt.loc[ev.s1].tolist(), txt.loc[ev.cand].tolist(), ev.label.values)
    train(a, b, y, f"{args.out}/model", epochs=1, lr=args.lr, log=log, base=args.init, eval_data=eval_data)
    log("DONE")


if __name__ == "__main__":
    main()
