"""Member B - document-level BM25 + dense fusion with top-k segment pooling."""
from __future__ import annotations

from typing import Dict, List

from member_a import get_retriever
from legalir_retriever import DOC_CANDIDATES, RERANK_SEGMENTS_PER_DOC


def build_bm25():
    return get_retriever().bm25


def build_graph(legal=None):
    """Disabled on purpose.

    A KG over chunks does not help when the label is a document ID: it merges
    evidence across unrelated documents and it previously evicted correct
    chunks from the candidate list.  Document-level rank fusion replaces it.
    """
    return {"status": "disabled", "reason": "document-level fusion is used"}


def get_graph():
    return build_graph()


def hybrid_retrieve(query: str, top_k: int = DOC_CANDIDATES,
                    segments_per_doc: int = RERANK_SEGMENTS_PER_DOC) -> List[Dict]:
    """Unique document candidates, each carrying its best passages.

    `top_k` counts DOCUMENTS, not chunks.  Every returned dict has `texts`,
    the list of passages the cross-encoder should score for that document.
    """
    r = get_retriever()
    n = int(max(1, min(top_k, len(r.doc_ids))))
    return r.candidate_documents(query, n_docs=n, segments_per_doc=segments_per_doc)
