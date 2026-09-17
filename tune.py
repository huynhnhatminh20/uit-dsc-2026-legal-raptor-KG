"""Grid-search the fusion hyper-parameters without re-running retrieval.

Each dev question is scored ONCE at the segment level; document-level scores are
cached for every pooling depth, then the fusion grid is evaluated in memory.
A 300-question sweep over ~100 configurations takes a couple of minutes instead
of hours.

Tune against candidate recall@100 first (that is the ceiling the cross-encoder
inherits), then against recall@5.
"""
from __future__ import annotations

import itertools
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm.std import tqdm

import legalir_retriever as L


def cache_dev_scores(retriever, dev: Dict, pools: Sequence[int] = (1, 2, 3)) -> List[Dict]:
    """Precompute per-question document scores for each pooling depth."""
    cached = []
    n_docs = len(retriever.doc_ids)
    for _, item in tqdm(list(dev.items()), desc="Caching dev scores"):
        bm, dense = retriever._segment_scores(item["question"])
        entry = {"gold": set(map(str, item["answer"])), "bm": {}, "dense": {}}
        for p in pools:
            entry["bm"][p] = L._topk_pool(bm, retriever.seg_doc_index, n_docs, pool=p)
            entry["dense"][p] = L._topk_pool(dense, retriever.seg_doc_index, n_docs, pool=p)
        cached.append(entry)
    return cached


def _rrf(scores: np.ndarray, weight: float, k: float) -> np.ndarray:
    out = np.zeros_like(scores, dtype=np.float32)
    nz = np.flatnonzero(scores > 0)
    if nz.size == 0:
        return out
    order = nz[np.argsort(scores[nz])[::-1]]
    out[order] = weight / (k + np.arange(1, len(order) + 1, dtype=np.float32))
    return out


def grid_search(retriever, cached: List[Dict],
                w_dense: Sequence[float] = (0.8, 1.0, 1.2, 1.4, 1.6, 2.0),
                rrf_k: Sequence[float] = (10.0, 20.0, 40.0, 60.0),
                pools: Sequence[int] = (1, 2, 3),
                depths: Sequence[int] = (5, 20, 100),
                objective: int = 100, top: int = 12) -> List[Tuple]:
    doc_ids = retriever.doc_ids
    results = []
    for p, k, wd in tqdm(list(itertools.product(pools, rrf_k, w_dense)), desc="Grid"):
        hits = {d: 0 for d in depths}
        for entry in cached:
            fused = _rrf(entry["bm"][p], 1.0, k) + _rrf(entry["dense"][p], wd, k)
            m = max(depths)
            top_idx = np.argpartition(fused, -m)[-m:]
            top_idx = top_idx[np.argsort(fused[top_idx])[::-1]]
            ranked = doc_ids[top_idx]
            for d in depths:
                if entry["gold"] & set(ranked[:d].tolist()):
                    hits[d] += 1
        n = max(len(cached), 1)
        results.append(({"pool": p, "rrf_k": k, "w_dense": wd},
                        {f"recall@{d}": round(hits[d] / n, 4) for d in depths}))
    results.sort(key=lambda r: (r[1][f"recall@{objective}"], r[1]["recall@5"]), reverse=True)
    for cfg, sc in results[:top]:
        print(cfg, sc)
    return results


def apply_config(cfg: Dict) -> None:
    """Push a winning configuration into the live module."""
    L.DOC_POOL_SEGMENTS = int(cfg["pool"])
    L.RRF_K = float(cfg["rrf_k"])
    L.W_DENSE = float(cfg["w_dense"])
    print("Applied:", cfg)
