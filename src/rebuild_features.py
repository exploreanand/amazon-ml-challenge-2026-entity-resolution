"""Rebuild pair features for existing candidate sets (no new blocking).

    python src/rebuild_features.py --data dataset --work work --out work/v3

Reads the blocking scores/ranks from --work/<split>_<country>_X.parquet,
re-normalises the records with the current ber.normalize and writes new
feature tables to --out with the same file names.
"""

import argparse
import os
import time

import pandas as pd

from ber.match import RANK_COLS, SCORE_COLS, build_features
from ber.pipeline import prepare
from run_pipeline import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/v3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tags = sorted(f[:-len("_X.parquet")] for f in os.listdir(args.work) if f.endswith("_X.parquet"))
    for split in ("train", "test"):
        s = None
        for tag in [t for t in tags if t.startswith(split + "_")]:
            if os.path.exists(f"{args.out}/{tag}_X.parquet"):
                continue
            if s is None:
                s = {k: load(f"{args.data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)}
            country = tag[len(split) + 1:]
            t = time.time()
            c = pd.read_parquet(f"{args.work}/{tag}_X.parquet", columns=["s1", "cand"] + SCORE_COLS + RANK_COLS)
            p1 = prepare(s[1][s[1].country == country])
            p23 = prepare(pd.concat([s[k][s[k].country == country] for k in (2, 3)]))
            X = build_features(c, p1, p23)
            X.to_parquet(f"{args.out}/{tag}_X.parquet")
            print(f"{tag}: {len(X):,} pairs, {X.shape[1]} columns in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
