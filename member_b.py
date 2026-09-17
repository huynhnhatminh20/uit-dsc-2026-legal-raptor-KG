"""Member B - document-level BM25+dense fusion; no graph-induced candidate loss."""
from __future__ import annotations
from typing import Dict, List
from member_a import get_retriever

def build_bm25():
    r = get_retriever()
    return r.bm25

def build_graph(legal=None):
    # Kept only for notebook compatibility.  A graph is not useful when the
    # evaluation label is a document ID and it formerly displaced good chunks.
    return {"status": "disabled", "reason": "document-level fusion is used"}

def get_graph():
    return build_graph()

def hybrid_retrieve(query: str, top_k: int = 120) -> List[Dict]:
    """Return unique document candidates, already aggregated before reranking."""
    return get_retriever().candidate_documents(query, n_docs=min(max(top_k, 1), 200))
