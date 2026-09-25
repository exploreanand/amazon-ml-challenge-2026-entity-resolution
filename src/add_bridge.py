"""v5: add "sibling bridging" candidates to existing candidate sets, then rebuild features.

    python src/add_bridge.py --data dataset --feat work/v3 --out work/v5

Every S2/S3 record belongs to at most one S1, and the records of one cluster
are noisy copies of the same business. A record that blocking could not
place (no address, or only weak candidates) often has a near-identical name
twin in the other source that *was* found. Bridging lets such a record
inherit its twin's best S1 candidates:

  queries    S2/S3 records with an empty address, or whose best name+address
             similarity to any candidate S1 is < WEAK
  neighbours top-K other S2/S3 records of the same country by char 3-gram
             TF-IDF of the normalised name, cosine >= MIN_SIM
  inherit    each neighbour's top-INHERIT S1s (by e5 + nameaddr score)

New pairs get the same four blocking scores as all other pairs (same
vectorisers, same embedding model) plus two bridge features. On India
validation this raised blocking recall from 96.78% to 97.51% for +11%
candidates.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd

from ber.blocking import dense_topk_search, fit_tfidf, topk_search  # noqa: F401  (topk_search used)
from ber.match import RANK_COLS, SCORE_COLS, build_features
from ber.pipeline import PASSES, _rowwise_dot_dense, _rowwise_dot_sparse, encode_e5, prepare
from run_pipeline import load

WEAK, K, MIN_SIM, INHERIT = 0.5, 4, 0.8, 2


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def bridge_pairs(c, p23):
    """New (s1, cand, bridge_sim, bridge_nb_score) pairs not already in ``c``."""
    best_na = c.groupby("cand").nameaddr.max()
    weak = (p23.naddr == "").values | (p23.entity_id.map(best_na).fillna(0).values < WEAK)
    vec = fit_tfidf(p23.nname, ngram_range=(3, 3))
    Xn = vec.transform(p23.nname)
    q = np.where(weak)[0]
    idx, val = topk_search(Xn, Xn[q], K)
    nb = pd.DataFrame({"cand": np.repeat(p23.entity_id.values[q], K),
                       "nb": p23.entity_id.values[idx.ravel()], "bridge_sim": val.ravel().astype(np.float32)})
    nb = nb[(nb.cand != nb.nb) & (nb.bridge_sim >= MIN_SIM)]
    sc = c.assign(bridge_nb_score=(c.e5 + c.nameaddr).astype(np.float32)).sort_values("bridge_nb_score", ascending=False)
    top = sc.groupby("cand").head(INHERIT)[["cand", "s1", "bridge_nb_score"]].rename(columns={"cand": "nb"})
    new = nb.merge(top, on="nb")
    new = (new.sort_values(["bridge_sim", "bridge_nb_score"], ascending=False)
              .drop_duplicates(["s1", "cand"])[["s1", "cand", "bridge_sim", "bridge_nb_score"]])
    new = new.merge(c[["s1", "cand"]], on=["s1", "cand"], how="left", indicator=True)
    return new[new._merge == "left_only"].drop(columns="_merge").reset_index(drop=True), int(weak.sum())


def pass_scores(new, p1, p23):
    """The four blocking similarities for the new pairs, computed as in ber.pipeline."""
    i1 = pd.Series(np.arange(len(p1)), index=p1.entity_id.values).loc[new.s1].values
    i23 = pd.Series(np.arange(len(p23)), index=p23.entity_id.values).loc[new.cand].values
    u1, inv1 = np.unique(i1, return_inverse=True)
    u23, inv23 = np.unique(i23, return_inverse=True)
    e1 = encode_e5(p1.e5text.values[u1])
    e23 = encode_e5(p23.e5text.values[u23])
    out = {"e5": _rowwise_dot_dense(e1, e23, inv1, inv23)}
    for name, (field, ngram, _) in PASSES.items():
        vec = fit_tfidf(pd.concat([p1[field], p23[field]]), ngram_range=ngram)
        out[name] = _rowwise_dot_sparse(vec.transform(p1[field].values[u1]).tocsr(),
                                        vec.transform(p23[field].values[u23]).tocsr(), inv1, inv23)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset")
    ap.add_argument("--feat", default="work/v3")
    ap.add_argument("--out", default="work/v5")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tags = sorted(f[:-len("_X.parquet")] for f in os.listdir(args.feat) if f.endswith("_X.parquet"))
    for split in ("train", "test"):
        s = None
        for tag in [t for t in tags if t.startswith(split + "_")]:
            if os.path.exists(f"{args.out}/{tag}_X.parquet"):
                continue
            if s is None:
                s = {k: load(f"{args.data}/{split}/{split}_source{k}.tsv") for k in (1, 2, 3)}
            country = tag[len(split) + 1:]
            t = time.time()
            c = pd.read_parquet(f"{args.feat}/{tag}_X.parquet", columns=["s1", "cand"] + SCORE_COLS + RANK_COLS)
            p1 = prepare(s[1][s[1].country == country])
            p23 = prepare(pd.concat([s[k][s[k].country == country] for k in (2, 3)]))
            new, n_q = bridge_pairs(c, p23)
            log(f"{tag}: {n_q:,} weak records -> {len(new):,} bridge pairs ({len(new) / len(c):.1%} more)")
            new = pd.concat([new, pass_scores(new, p1, p23)], axis=1)
            both = pd.concat([c, new], ignore_index=True)
            X = build_features(both, p1, p23)
            X.to_parquet(f"{args.out}/{tag}_X.parquet")
            log(f"{tag}: {len(X):,} pairs, {X.shape[1]} columns in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
