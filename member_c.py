"""Member C - multi-passage reranking, BTC-safe submission and evaluation."""
from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm.std import tqdm

RERANKER_MODEL = os.environ.get("LEGALIR_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_BATCH = int(os.environ.get("LEGALIR_RERANK_BATCH", "64"))
RERANK_MAX_LEN = int(os.environ.get("LEGALIR_RERANK_MAXLEN", "1024"))
MAX_IDS = 5  # hard limit from the task description

_RERANKERS: Dict[str, object] = {}


# --------------------------------------------------------------------------
# Thin model wrappers (kept for notebook compatibility)
# --------------------------------------------------------------------------
class EmbeddingWrapper:
    def load(self, force_reload: bool = False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(os.environ.get("LEGALIR_EMBED_MODEL", "BAAI/bge-m3"))
        return self

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)


class LegalModelWrapper:
    def load(self, force_reload: bool = False):
        return self


class LegalReranker:
    """Cross-encoder reranker that scores SEVERAL passages per document.

    The old version scored one window per document and took that as the
    document's score.  When the answer lived in a later window the document was
    judged on irrelevant text and fell out of the top 5 even though retrieval
    had found it.  Here each document contributes max over its best passages.
    """

    def __init__(self, model_name: str = RERANKER_MODEL, weight: float = 1.0):
        self.model_name = model_name
        self.weight = weight
        self.reranker = None

    def load(self, force_reload: bool = False):
        from sentence_transformers import CrossEncoder
        if force_reload or self.model_name not in _RERANKERS:
            _RERANKERS[self.model_name] = CrossEncoder(
                self.model_name, max_length=RERANK_MAX_LEN, trust_remote_code=True)
        self.reranker = _RERANKERS[self.model_name]
        return self

    # -- scoring ---------------------------------------------------------
    def _predict(self, pairs: Sequence[Sequence[str]]) -> np.ndarray:
        """Length-sorted prediction: cuts padding waste substantially."""
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        if self.reranker is None:
            self.load()
        order = np.argsort([len(a) + len(b) for a, b in pairs])
        sorted_pairs = [tuple(pairs[int(i)]) for i in order]
        scores = np.asarray(
            self.reranker.predict(sorted_pairs, batch_size=RERANK_BATCH,
                                  show_progress_bar=False), dtype=np.float32)
        out = np.empty_like(scores)
        out[order] = scores
        return out

    def score_documents(self, query: str, candidates: List[Dict]) -> np.ndarray:
        pairs, owner = [], []
        for i, c in enumerate(candidates):
            texts = c.get("texts") or [c.get("text", "")]
            for t in texts:
                if t:
                    pairs.append((query, t))
                    owner.append(i)
        scores = self._predict(pairs)
        best = np.full(len(candidates), -1e9, dtype=np.float32)
        if len(scores):
            np.maximum.at(best, np.asarray(owner, dtype=np.int64), scores)
        return best

    def rerank(self, query: str, candidates: List[Dict], top_k: int = MAX_IDS,
               prior_weight: float = 0.0) -> List[str]:
        if not 1 <= top_k <= MAX_IDS:
            raise ValueError("BTC chỉ chấp nhận 1..5 document_id")
        if not candidates:
            return []
        unique, seen = [], set()
        for c in candidates:
            doc_id = str(c.get("id", ""))
            if doc_id and doc_id not in seen:
                seen.add(doc_id)
                unique.append(c)
        best = self.score_documents(query, unique)
        if prior_weight:
            ranks = np.asarray([c.get("retrieval_rank", i + 1) for i, c in enumerate(unique)],
                               dtype=np.float32)
            best = best + prior_weight * (1.0 / (20.0 + ranks))
        order = np.argsort(best)[::-1][:top_k]
        return [unique[int(i)]["id"] for i in order]


class RerankerEnsemble:
    """Average the z-scored outputs of several cross-encoders.

    Two dissimilar rerankers typically add 1-3 points of recall@5 over the best
    single one.  Use it once a single model has plateaued.
    """

    def __init__(self, models: Iterable[LegalReranker]):
        self.models = list(models)

    def load(self, force_reload: bool = False):
        for m in self.models:
            m.load(force_reload)
        return self

    def rerank(self, query: str, candidates: List[Dict], top_k: int = MAX_IDS,
               prior_weight: float = 0.0) -> List[str]:
        if not candidates:
            return []
        total = np.zeros(len(candidates), dtype=np.float32)
        for m in self.models:
            s = m.score_documents(query, candidates)
            s = (s - s.mean()) / (s.std() + 1e-6)
            total += m.weight * s
        if prior_weight:
            ranks = np.asarray([c.get("retrieval_rank", i + 1) for i, c in enumerate(candidates)],
                               dtype=np.float32)
            total = total + prior_weight * (1.0 / (20.0 + ranks))
        order = np.argsort(total)[::-1][:top_k]
        return [candidates[int(i)]["id"] for i in order]


def get_reranker(force_reload: bool = False, model_name: str = RERANKER_MODEL) -> LegalReranker:
    return LegalReranker(model_name).load(force_reload)


def get_embedder(force_reload: bool = False):
    return EmbeddingWrapper().load(force_reload)


def get_llm(force_reload: bool = False):
    return LegalModelWrapper().load(force_reload)


# --------------------------------------------------------------------------
# Evaluation (matches the task description exactly)
# --------------------------------------------------------------------------
def evaluate_recall_precision(ground_truth: Dict, predictions: Dict, k: int = MAX_IDS) -> Dict:
    """Recall is primary, precision is the tiebreak.

    A question whose prediction exceeds k IDs scores 0 on both, and it still
    counts in the denominator - exactly as the organisers specify.
    """
    recalls, precisions, violated = [], [], 0
    for qid, item in ground_truth.items():
        gold = (item.get("answer") or []) if isinstance(item, dict) else item
        gold = set(map(str, gold or []))
        raw = predictions.get(str(qid), predictions.get(qid, {}))
        pred = raw.get("answer", []) if isinstance(raw, dict) else raw
        pred = list(map(str, pred or []))
        if not gold:
            continue
        if len(pred) > k:
            recalls.append(0.0)
            precisions.append(0.0)
            violated += 1
            continue
        hit = len(gold & set(pred))
        recalls.append(hit / len(gold))
        precisions.append(hit / len(pred) if pred else 0.0)
    r = float(np.mean(recalls)) if recalls else 0.0
    p = float(np.mean(precisions)) if precisions else 0.0
    return {"recall@5": r, "precision@5": p, "recall": r, "precision": p,
            "total_queries": len(recalls), "violated": violated}


def run_predictions(questions: Dict, retrieve_fn, reranker, top_k: int = MAX_IDS,
                    prior_weight: float = 0.0, desc: str = "Predicting") -> Dict:
    preds = {}
    for qid, item in tqdm(list(questions.items()), desc=desc):
        q = item["question"] if isinstance(item, dict) else str(item)
        preds[str(qid)] = {"answer": reranker.rerank(q, retrieve_fn(q), top_k=top_k,
                                                     prior_weight=prior_weight)}
    return preds


def save_submission(submission: Dict, filename: str = "submission.json",
                    zip_path: Optional[str] = None) -> str:
    for qid, v in submission.items():
        if len(v.get("answer", [])) > MAX_IDS:
            raise ValueError(f"Câu hỏi {qid} có nhiều hơn {MAX_IDS} document_id")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    if zip_path:
        import zipfile
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(filename, "submission.json")
    return filename
