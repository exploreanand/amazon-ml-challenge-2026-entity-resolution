"""Score candidate pairs with a trained cross-encoder and cache the scores for exp_v9 / exp_v10b.

    python src/score_pairs.py --data dataset --model work/ce_qwen_full/model/best --split test \
        --pairs output_v72/test_probs.parquet --notna ce --out output_v11a/test_ce4.parquet

Writes DataFrame[s1, cand, ce2] (the cache layout exp_v9.ce2_scores reads), so a
second machine can score a model while the main pipeline runs elsewhere.
--notna keeps only the rows where that column is set (the pairs scored by the
other cross-encoders).
"""

import argparse
import os
import time

import pandas as pd

from ber.cross_encoder import score
from exp_v5 import texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=("train", "test"), required=True)
    ap.add_argument("--pairs", required=True, help="parquet with s1, cand")
    ap.add_argument("--notna", help="keep rows where this column is not null")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["s1", "cand"] + ([args.notna] if args.notna else [])
    P = pd.read_parquet(args.pairs, columns=cols)
    if args.notna:
        P = P[P[args.notna].notna()]
    txt = texts(args.data, args.split)
    t = time.time()
    s = score(txt.loc[P.s1].tolist(), txt.loc[P.cand].tolist(), args.model)
    P[["s1", "cand"]].assign(ce2=s).to_parquet(args.out)
    print(time.strftime("%H:%M:%S"), f"scored {len(P):,} pairs in {time.time() - t:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
