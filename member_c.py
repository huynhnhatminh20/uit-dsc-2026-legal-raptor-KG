"""Member C - batched reranking, BTC-safe submission and evaluation."""
from __future__ import annotations
import json
from typing import Dict, List
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
_RERANKER = None

class EmbeddingWrapper:
    def load(self, force_reload: bool = False):
        self.model = SentenceTransformer("BAAI/bge-m3")
        return self
    def encode(self, *args, **kwargs): return self.model.encode(*args, **kwargs)

class LegalModelWrapper:
    def load(self, force_reload: bool = False): return self

class LegalReranker:
    def load(self, force_reload: bool = False):
        global _RERANKER
        if _RERANKER is None or force_reload:
            _RERANKER = CrossEncoder(RERANKER_MODEL, max_length=1024, trust_remote_code=True)
        self.reranker = _RERANKER
        return self
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 4) -> List[str]:
        if not 1 <= top_k <= 5: raise ValueError("BTC only accepts 1..5 IDs")
        if not candidates: return []
        if not hasattr(self, "reranker"): self.load()
        # B has already made candidates unique by document; keep that invariant.
        unique, seen = [], set()
        for c in candidates:
            doc_id = str(c.get("id", ""))
            if doc_id and doc_id not in seen:
                seen.add(doc_id); unique.append(c)
        scores = self.reranker.predict([(query, c.get("text", "")) for c in unique], batch_size=48)
        return [unique[int(i)]["id"] for i in np.argsort(scores)[::-1][:top_k]]

def get_reranker(force_reload: bool = False): return LegalReranker().load(force_reload)
def get_embedder(force_reload: bool = False): return EmbeddingWrapper().load(force_reload)
def get_llm(force_reload: bool = False): return LegalModelWrapper().load(force_reload)

def evaluate_recall_precision(ground_truth: Dict, predictions: Dict, k: int = 5) -> Dict:
    recalls, precisions, violated = [], [], 0
    for qid, item in ground_truth.items():
        gt = set(map(str, (item.get("answer") or []) if isinstance(item, dict) else item))
        raw = predictions.get(str(qid), predictions.get(qid, {}))
        pred = raw.get("answer", []) if isinstance(raw, dict) else raw
        pred = list(map(str, pred or []))
        if len(pred) > k:
            recalls.append(0.0); precisions.append(0.0); violated += 1; continue
        if not gt:
            continue
        hit = len(gt.intersection(pred))
        recalls.append(hit / len(gt)); precisions.append(hit / len(pred) if pred else 0.0)
    return {"recall@5": float(np.mean(recalls)) if recalls else 0.0,
            "precision@5": float(np.mean(precisions)) if precisions else 0.0,
            "recall": float(np.mean(recalls)) if recalls else 0.0,
            "precision": float(np.mean(precisions)) if precisions else 0.0,
            "total_queries": len(recalls), "violated": violated}

def save_submission(submission: Dict, filename: str = "submission.json"):
    with open(filename, "w", encoding="utf-8") as f: json.dump(submission, f, ensure_ascii=False, indent=2)
