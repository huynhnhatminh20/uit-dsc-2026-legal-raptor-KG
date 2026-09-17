"""Member A - lossless legal segmentation and exact dense document retrieval."""
from __future__ import annotations
import os
from typing import Dict, List, Optional
from legalir_retriever import LegalIRRetriever

_RETRIEVER: Optional[LegalIRRetriever] = None

def _context_dir() -> str:
    return os.environ.get("LEGALIR_CONTEXT_DIR", "data_legalir")

def get_retriever(force_rebuild: bool = False) -> LegalIRRetriever:
    global _RETRIEVER
    if _RETRIEVER is None or force_rebuild:
        cache_dir = os.environ.get("LEGALIR_CACHE_DIR", "/kaggle/working/legalir_doc_index")
        _RETRIEVER = LegalIRRetriever(_context_dir(), cache_dir=cache_dir).build_or_load(force_rebuild=force_rebuild)
    return _RETRIEVER

def build_raptor(legal=None, emb=None):
    """Compatibility API.  The task scores document IDs, so no lossy RAPTOR summaries are built."""
    r = get_retriever()
    # Do not materialize a second Python copy of every segment merely to mimic
    # the old RAPTOR tree; that copy can consume several GB on Kaggle.
    return {"nodes": [], "levels": [{"level": 0}], "num_segments": len(r.segments)}

def build_vector_store(emb=None):
    r = get_retriever()
    return {"index": r.index, "num_nodes": len(r.segments), "index_type": "IVF-IP"}

def get_raptor_nodes() -> List[Dict]:
    r = get_retriever()
    return [{"id": str(i), "text": x["text"], "level": 0,
             "metadata": {"doc_id": x["doc_id"]}} for i, x in enumerate(r.segments)]

def get_vector_store():
    return build_vector_store()

def get_node_doc_map() -> Dict[str, str]:
    r = get_retriever()
    return {str(i): x["doc_id"] for i, x in enumerate(r.segments)}
