"""Candidate generation: character n-gram TF-IDF + exact top-k on the GPU.

All searches run inside one country partition (train shows 100% of true
pairs share the country label). Country values are taken from the data,
never from a fixed list.
"""

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def fit_tfidf(corpus, ngram_range=(3, 3), min_df=2, max_df=1.0):
    """max_df < 1 drops the most common n-grams (e.g. "pvt", "ltd"): they carry
    little signal but dominate the sparse matmul cost."""
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range, min_df=min_df, max_df=max_df,
        sublinear_tf=True, dtype=np.float32,
    )
    vec.fit(corpus)
    return vec


def _to_gpu_csr(X, device):
    X = X.tocsr().astype(np.float32)
    return torch.sparse_csr_tensor(
        torch.from_numpy(X.indptr.astype(np.int64)),
        torch.from_numpy(X.indices.astype(np.int64)),
        torch.from_numpy(X.data),
        size=X.shape, device=device,
    )


@torch.no_grad()
def topk_search(index_X, query_X, k, max_score_cells=1_500_000_000, device="cuda"):
    """Exact cosine top-k of every query row against every index row.

    Rows are L2-normalised TF-IDF, so the dot product is the cosine.
    Returns (idx, score), both shaped (n_queries, k); idx points into index rows.
    """
    k = min(k, index_X.shape[0])
    # the score block (and its transpose) must fit in GPU memory: ~6 GB each at fp32
    batch = int(max(64, min(2048, max_score_cells // index_X.shape[0])))
    index = _to_gpu_csr(index_X, device)
    query_X = query_X.tocsr().astype(np.float32)
    n = query_X.shape[0]
    out_idx = np.empty((n, k), dtype=np.int32)
    out_val = np.empty((n, k), dtype=np.float16)
    for start in range(0, n, batch):
        q = query_X[start:start + batch]
        qd = torch.zeros((q.shape[0], q.shape[1]), dtype=torch.float32, device=device)
        rows = torch.from_numpy(np.repeat(np.arange(q.shape[0]), np.diff(q.indptr))).to(device)
        cols = torch.from_numpy(q.indices.astype(np.int64)).to(device)
        qd[rows, cols] = torch.from_numpy(q.data).to(device)
        # (n_index, b) -> (b, n_index): top-k along a contiguous dim is ~2.5x faster
        scores = torch.sparse.mm(index, qd.T.contiguous()).T.contiguous()
        val, idx = torch.topk(scores, k, dim=1)
        out_idx[start:start + q.shape[0]] = idx.cpu().numpy()
        out_val[start:start + q.shape[0]] = val.cpu().numpy()
        del qd, scores, val, idx
    return out_idx, out_val


@torch.no_grad()
def dense_topk_search(index_E, query_E, k, max_score_cells=2_000_000_000, device="cuda"):
    """Exact inner-product top-k for L2-normalised dense embeddings (fp16 matmul)."""
    k = min(k, index_E.shape[0])
    batch = int(max(64, min(8192, max_score_cells // index_E.shape[0])))
    index = torch.from_numpy(np.ascontiguousarray(index_E)).to(device, torch.float16)
    n = query_E.shape[0]
    out_idx = np.empty((n, k), dtype=np.int32)
    out_val = np.empty((n, k), dtype=np.float16)
    for start in range(0, n, batch):
        q = torch.from_numpy(np.ascontiguousarray(query_E[start:start + batch])).to(device, torch.float16)
        val, idx = torch.topk(q @ index.T, k, dim=1)
        out_idx[start:start + len(q)] = idx.cpu().numpy()
        out_val[start:start + len(q)] = val.cpu().numpy()
    return out_idx, out_val
