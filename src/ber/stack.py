"""Second-stage (stacked) matcher.

Stage 1 scores every candidate pair on its own. Stage 2 adds cluster
context, computed from out-of-fold stage-1 probabilities:
  - where the pair ranks among its S1's candidates and how strong the S1's
    candidate set is overall;
  - how the pair compares with the same S2/S3 record's rival S1s;
  - sibling similarity: whether the candidate resembles the S1's *other*
    likely members (e5 cosine), since true members are noisy copies of one
    business while decoys resemble only the S1 record.
"""

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

N_FOLDS = 5


def fold_of(ids, n_folds=N_FOLDS, key="ber2026fold00001"):
    h = pd.util.hash_pandas_object(pd.Series(ids), index=False, hash_key=key).values
    return (h % n_folds).astype(np.int8)


def oof_predict(X, feats, params, trees, dmatrix):
    """Out-of-fold stage-1 probabilities, folds split by S1 id."""
    folds = fold_of(X.s1.values)
    oof = np.zeros(len(X), dtype=np.float32)
    for f in range(N_FOLDS):
        tr, te = folds != f, folds == f
        bst = xgb.train(params, dmatrix(X[tr], feats), trees)
        oof[te] = bst.predict(xgb.DMatrix(X.loc[te, feats].astype(np.float32)))
    return oof


def group_features(s1, cand, p):
    """Cluster-context features from stage-1 probabilities (vectorised)."""
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": p})
    g1 = d.groupby("s1").p
    out = pd.DataFrame(index=d.index)
    out["p1"] = d.p
    out["p1_rank_s1"] = g1.rank(ascending=False, method="first")
    out["p1_sum_s1"] = g1.transform("sum")
    out["p1_max_s1"] = g1.transform("max")
    out["p1_n50_s1"] = (d.p > 0.5).groupby(d.s1).transform("sum")
    out["p1_gap_s1"] = d.p - out.p1_max_s1
    # best and second-best S1 probability for the same S2/S3 record
    o = d.sort_values(["cand", "p"], ascending=[True, False])
    r = o.groupby("cand").cumcount()
    best_c = o.p[r == 0].groupby(o.cand[r == 0]).first()
    second_c = o.p[r == 1].groupby(o.cand[r == 1]).first()
    b = d.cand.map(best_c).values
    sc = d.cand.map(second_c).fillna(0).values
    out["p1_gap_cand"] = d.p.values - b
    out["p1_rival_cand"] = np.where(d.p.values >= b, sc, b)
    out["p1_nrival_cand"] = d.groupby("cand").p.transform("size").values - 1
    return out.astype(np.float32)


@torch.no_grad()
def sibling_similarity(s1, cand, p, emb_of, max_sib=24, device="cuda"):
    """For each pair: p-weighted mean and max e5 cosine between the candidate
    and its S1's other candidates (siblings), plus cosine to the best sibling.

    emb_of: dict-like entity_id -> row index into a (n, d) fp16 array E, given
    as (index_map: pd.Series, E: np.ndarray).
    """
    index_map, E = emb_of
    d = pd.DataFrame({"s1": s1, "cand": cand, "p": p})
    d["row"] = np.arange(len(d))
    d = d.sort_values(["s1", "p"], ascending=[True, False])
    d["pos"] = d.groupby("s1").cumcount()
    d = d[d.pos < max_sib]
    n_s1 = d.s1.nunique()
    s1_code = pd.factorize(d.s1)[0]
    emb_idx = index_map.reindex(d.cand.values).values
    dim = E.shape[1]
    Et = torch.from_numpy(E).to(device)
    out_mean = np.full(len(s1), np.nan, dtype=np.float32)
    out_max = np.full(len(s1), np.nan, dtype=np.float32)
    out_top = np.full(len(s1), np.nan, dtype=np.float32)
    chunk = 200_000
    for c0 in range(0, n_s1, chunk):
        m = (s1_code >= c0) & (s1_code < c0 + chunk)
        sc = torch.from_numpy(s1_code[m] - c0).to(device)
        pos = torch.from_numpy(d.pos.values[m]).to(device)
        n = int(min(chunk, n_s1 - c0))
        V = torch.zeros((n, max_sib, dim), dtype=torch.float16, device=device)
        P = torch.zeros((n, max_sib), dtype=torch.float32, device=device)
        V[sc, pos] = Et[torch.from_numpy(emb_idx[m]).to(device)]
        P[sc, pos] = torch.from_numpy(d.p.values[m].astype(np.float32)).to(device)
        S = torch.bmm(V, V.transpose(1, 2)).float()  # (n, k, k) cosine
        eye = torch.eye(max_sib, device=device, dtype=torch.bool)
        W = P.unsqueeze(1).expand(-1, max_sib, -1).masked_fill(eye, 0)  # weight of sibling j for pair i
        mean = (S * W).sum(2) / W.sum(2).clamp(min=1e-6)
        mx = S.masked_fill(eye | (W <= 0.5), -1).max(2).values
        top_j = W.argmax(2, keepdim=True)
        top = S.gather(2, top_j).squeeze(2)
        rows = d.row.values[m]
        out_mean[rows] = mean[sc, pos].cpu().numpy()
        out_max[rows] = mx[sc, pos].cpu().numpy()
        out_top[rows] = top[sc, pos].cpu().numpy()
        del V, P, S, W
    out_max[out_max < -0.5] = np.nan  # no strong sibling
    return pd.DataFrame({"sib_mean": out_mean, "sib_max_strong": out_max, "sib_top": out_top})
