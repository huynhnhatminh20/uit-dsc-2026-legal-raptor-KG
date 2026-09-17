"""Member A - lossless legal segmentation, exact dense index, stage diagnostics."""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
from tqdm.std import tqdm

from legalir_retriever import LegalIRRetriever, DOC_CANDIDATES

_RETRIEVER: Optional[LegalIRRetriever] = None


def _context_dir() -> str:
    return os.environ.get("LEGALIR_CONTEXT_DIR", "data_legalir")


def get_retriever(force_rebuild: bool = False) -> LegalIRRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or force_rebuild:
        cache_dir = os.environ.get("LEGALIR_CACHE_DIR", "/kaggle/working/legalir_doc_index")
        _RETRIEVER = LegalIRRetriever(_context_dir(), cache_dir=cache_dir).build_or_load(
            force_rebuild=force_rebuild)
    return _RETRIEVER


def build_raptor(legal=None, emb=None):
    """Compatibility shim. The task scores document IDs, so no summary tree is built."""
    r = get_retriever()
    return {"nodes": [], "levels": [{"level": 0}],
            "num_segments": len(r.segments), "num_documents": int(len(r.doc_ids))}


def build_vector_store(emb=None):
    r = get_retriever()
    return {"index": r.index, "num_nodes": len(r.segments),
            "index_type": type(r.index).__name__}


def get_raptor_nodes() -> List[Dict]:
    r = get_retriever()
    return [{"id": str(i), "text": x["text"], "level": 0, "metadata": {"doc_id": x["doc_id"]}}
            for i, x in enumerate(r.segments)]


def get_vector_store():
    return build_vector_store()


def get_node_doc_map() -> Dict[str, str]:
    r = get_retriever()
    return {str(i): x["doc_id"] for i, x in enumerate(r.segments)}


# --------------------------------------------------------------------------
# Diagnostics: this is the number that actually caps your final score.
# --------------------------------------------------------------------------
def candidate_recall(dev: Dict, depths=(5, 10, 20, 50, 100, 200), n_docs: int = None,
                     limit: int = None) -> Dict[int, float]:
    """Recall@N of the RETRIEVAL stage, before the cross-encoder.

    The cross-encoder can only reorder what retrieval hands it, so final
    recall@5 is bounded by candidate_recall at the candidate depth.  Run this
    first whenever the score moves; it tells you whether to tune retrieval or
    the reranker.
    """
    r = get_retriever()
    n_docs = n_docs or max(depths)
    items = list(dev.items())[: limit or len(dev)]
    hits = {d: 0 for d in depths}
    for _, item in tqdm(items, desc="Candidate recall"):
        gold = set(map(str, item["answer"]))
        ranked = [c["id"] for c in r.candidate_documents(item["question"], n_docs=n_docs)]
        for d in depths:
            if gold & set(ranked[:d]):
                hits[d] += 1
    return {d: round(h / max(len(items), 1), 4) for d, h in hits.items()}
