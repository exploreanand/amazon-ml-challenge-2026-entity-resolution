"""Candidate generation for one country partition.

Blocking = union of four *reverse* searches: every S2/S3 record queries all
S1 records of its country and keeps its top-k S1s. Each S2/S3 record matches
at most one S1 (true in train), so a small k gives high recall, and each S1's
candidate list grows with its real cluster size.

  pass       representation                                  k
  e5         multilingual-e5-small embedding of name+address 2
  nameaddr   char 3-gram TF-IDF of normalised name+address   2
  skall      char 2-3-gram TF-IDF of consonant skeletons     1
  naddr      char 3-gram TF-IDF of normalised address        1

Every pass score is then recomputed for *every* candidate pair, so the
matcher sees all four similarities regardless of which pass found the pair.
"""

import html
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from .blocking import dense_topk_search, fit_tfidf, topk_search
from .normalize import basic, name_skeleton, norm_addr, norm_name, skeleton

PASSES = {  # name -> (field, ngram_range, k)
    "nameaddr": ("nameaddr", (3, 3), 2),
    "skall": ("skall", (2, 3), 1),
    "naddr": ("naddr", (3, 3), 1),
}
E5_K = 2
E5_MODEL = "intfloat/multilingual-e5-small"  # MIT licence, 118M params


def _skel_addr(a):
    return skeleton(basic(a))


def _pmap(fn, values, procs=None):
    with Pool(procs or os.cpu_count()) as p:
        return p.map(fn, values, chunksize=20_000)


def prepare(df):
    """Add every normalised field used by blocking and matching."""
    df = df.reset_index(drop=True).copy()
    df["nname"] = _pmap(norm_name, df.business_name.tolist())
    df["naddr"] = _pmap(norm_addr, df.business_address.tolist())
    df["nskel"] = _pmap(name_skeleton, df.business_name.tolist())
    df["askel"] = _pmap(_skel_addr, df.business_address.tolist())
    df["nameaddr"] = df.nname + " " + df.naddr
    df["skall"] = df.nskel + " " + df.askel
    df["e5text"] = ("query: " + df.business_name.map(html.unescape) + ", "
                    + df.business_address.map(html.unescape))
    return df


def encode_e5(texts, model=None):
    from sentence_transformers import SentenceTransformer
    model = model or SentenceTransformer(E5_MODEL, device="cuda")
    model.half()
    model.max_seq_length = 64
    return model.encode(list(texts), batch_size=2048, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float16)


def _rowwise_dot_sparse(A, B, ia, ib, chunk=2_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a, b = A[ia[s:s + chunk]], B[ib[s:s + chunk]]
        out[s:s + chunk] = np.asarray(a.multiply(b).sum(axis=1)).ravel()
    return out


def _rowwise_dot_dense(A, B, ia, ib, chunk=4_000_000):
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        a = torch.from_numpy(A[ia[s:s + chunk]]).cuda().float()
        b = torch.from_numpy(B[ib[s:s + chunk]]).cuda().float()
        out[s:s + chunk] = (a * b).sum(1).cpu().numpy()
    return out


def candidates_for_partition(p1, p23, log=print):
    """p1: prepared S1 rows of one country; p23: prepared S2+S3 rows of the same country.

    Returns DataFrame[i1, i23, <pass>_rank, <pass>] where i1/i23 index p1/p23,
    <pass>_rank is the rank of the S1 in the S2/S3 record's list (NaN if that
    pass did not retrieve the pair) and <pass> is the pass similarity.
    """
    found, mats = [], {}
    t = time.time()
    e1, e23 = encode_e5(p1.e5text), encode_e5(p23.e5text)
    mats["e5"] = (e1, e23)
    idx, _ = dense_topk_search(e1, e23, E5_K)
    found.append(("e5", idx))
    log(f"  e5 done {time.time() - t:.0f}s")
    for name, (field, ngram, k) in PASSES.items():
        t = time.time()
        vec = fit_tfidf(pd.concat([p1[field], p23[field]]), ngram_range=ngram)
        X1, X23 = vec.transform(p1[field]), vec.transform(p23[field])
        mats[name] = (X1.tocsr(), X23.tocsr())
        idx, _ = topk_search(X1, X23, k)
        found.append((name, idx))
        torch.cuda.empty_cache()
        log(f"  {name} done {time.time() - t:.0f}s")

    parts = []
    for name, idx in found:
        k = idx.shape[1]
        parts.append(pd.DataFrame({
            "i1": idx.ravel().astype(np.int32),
            "i23": np.repeat(np.arange(len(p23), dtype=np.int32), k),
            f"{name}_rank": np.tile(np.arange(k, dtype=np.float32), len(p23)),
        }))
    cands = parts[0]
    for part in parts[1:]:
        cands = cands.merge(part, on=["i1", "i23"], how="outer")
    ia, ib = cands.i1.values, cands.i23.values
    cands["e5"] = _rowwise_dot_dense(*mats["e5"], ia, ib)
    for name in PASSES:
        cands[name] = _rowwise_dot_sparse(*mats[name], ia, ib)
    return cands.reset_index(drop=True)
