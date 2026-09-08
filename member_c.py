"""
member_c.py - Reranker + Evaluation
Chiu trach nhiem: Reranker BAAI/bge-reranker-v2-m3, Evaluation metrics
"""

import os
import json
import pickle
import pathlib
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CKPT_DIR = pathlib.Path("/kaggle/working/C_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RERANKER_CKPT = CKPT_DIR / "reranker.pkl"
LLM_CKPT = CKPT_DIR / "legal_llm.pkl"
EVAL_CKPT = CKPT_DIR / "eval_fn.pkl"

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"


# ============ LEGAL LLM WRAPPER ============

class LegalModelWrapper:
    """
    Wrapper cho VLSP2025-LegalSML/qwen3-4b-legal-pretrain
    Su dung 4-bit quantization de tiet kiem memory
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self, force_reload: bool = False):
        """Load model voi checkpoint"""
        if not force_reload and LLM_CKPT.exists():
            logger.info(f"Load Legal LLM tu checkpoint: {LLM_CKPT}")
            with open(LLM_CKPT, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Loading {LLM_MODEL} with 4-bit quantization...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                LLM_MODEL,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            with open(LLM_CKPT, "wb") as f:
                pickle.dump(self, f)
            logger.info(f"Legal LLM checkpoint saved to {LLM_CKPT}")
            
        except Exception as e:
            logger.warning(f"Could not load LLM: {e}. Using fallback.")
            self.model = None
            self.tokenizer = None
        
        return self
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using Legal LLM"""
        if self.model is None:
            logger.warning("LLM not loaded!")
            return ""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return ""


# ============ EMBEDDING WRAPPER ============

class EmbeddingWrapper:
    """Wrapper cho BAAI/bge-m3 embedding model"""
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self, force_reload: bool = False):
        """Load embedding model"""
        if self.model is not None:
            return self
        
        logger.info("Loading embedding model: BAAI/bge-m3...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            logger.info(f"Embedding model loaded on {self.device}")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            self.model = None
        
        return self
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.model is None:
            raise ValueError("Embedding model not loaded!")
        return self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)


# ============ RERANKER ============

class LegalReranker:
    """
    Cross-encoder reranker using BAAI/bge-reranker-v2-m3
    Loc toi da 05 document_id (theo yeu cau cuoc thi)
    """
    
    def __init__(self):
        self.model_name = RERANKER_MODEL
        self.reranker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
    
    def load(self, force_reload: bool = False):
        """Load reranker voi checkpoint"""
        if not force_reload and RERANKER_CKPT.exists():
            logger.info(f"Load Reranker tu checkpoint: {RERANKER_CKPT}")
            with open(RERANKER_CKPT, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Loading {self.model_name} on {self.device}...")
        
        self.reranker = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
            trust_remote_code=True
        )
        
        with open(RERANKER_CKPT, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Reranker checkpoint saved to {RERANKER_CKPT}")
        
        return self
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[str]:
        """
        Rerank candidates va tra ve top_k document IDs
        
        Args:
            query: Cau hoi
            candidates: List [{"id": "doc1", "text": "..."}]
            top_k: So luong ket qua (mac dinh 5, MAX 5 theo luat thi)
        
        Returns:
            List[str]: Danh sach document IDs (toi da 5)
        """
        # LUAT THI: TOI DA 5 DOCUMENTS
        if top_k > 5:
            logger.warning(f"top_k={top_k} vuot qua 5, tu dong gioi han xuong 5")
            top_k = 5
        
        if self.reranker is None:
            self.load()
        
        if not candidates:
            return []
        
        # Chuan hoa candidates de tranh loi KeyError
        normalized = []
        for c in candidates:
            doc_id = c.get('doc_id') or c.get('id')
            text = c.get('passage') or c.get('text') or ''
            if doc_id and text:
                normalized.append({'id': str(doc_id), 'text': text})
        
        if not normalized:
            return []
        
        # Chuan bi pairs cho cross-encoder
        pairs = [(query, c["text"]) for c in normalized]
        
        try:
            scores = self.reranker.predict(pairs, batch_size=32)
            sorted_indices = np.argsort(scores)[::-1]
            actual_k = min(top_k, len(normalized))
            result_ids = [normalized[i]["id"] for i in sorted_indices[:actual_k]]
            
            logger.info(f"Reranked {len(normalized)} candidates -> {len(result_ids)} docs")
            return result_ids
            
        except Exception as e:
            logger.warning(f"Reranker error: {e}")
            return [c["id"] for c in normalized[:min(top_k, len(normalized))]]


# ============ EVALUATION ============

def recall_at_k(predicted: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Tinh Recall@k"""
    if not ground_truth:
        return 0.0
    hits = len(set(predicted[:k]) & set(ground_truth))
    return hits / len(ground_truth)


def precision_at_k(predicted: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Tinh Precision@k"""
    if not predicted:
        return 0.0
    hits = len(set(predicted[:k]) & set(ground_truth))
    return hits / min(k, len(predicted))


def evaluate_recall_precision(
    ground_truth: Dict, 
    predictions: Dict, 
    k: int = 5
) -> Dict:
    """
    Danh gia Recall@k va Precision@k
    
    QUAN TRONG: Neu query co >5 documents -> 0 diem cho query do
    """
    recall_scores = []
    precision_scores = []
    errors = []
    
    for qid, pred_docs in predictions.items():
        # Kiem tra: Khong duoc vuot qua 5 documents
        if len(pred_docs) > 5:
            error_msg = f"Query {qid} tra ve {len(pred_docs)} docs (toi da 5) -> 0 diem"
            errors.append(error_msg)
            logger.warning(error_msg)
            recall_scores.append(0.0)
            precision_scores.append(0.0)
            continue
        
        gt_docs = ground_truth.get(qid, [])
        r5 = recall_at_k(pred_docs, gt_docs, k=k)
        p5 = precision_at_k(pred_docs, gt_docs, k=k)
        recall_scores.append(r5)
        precision_scores.append(p5)
    
    avg_recall = np.mean(recall_scores) if recall_scores else 0.0
    avg_precision = np.mean(precision_scores) if precision_scores else 0.0
    
    result = {
        "recall": avg_recall,
        "precision": avg_precision,
        "recall@5": avg_recall,
        "precision@5": avg_precision,
        "total_queries": len(predictions),
        "violated": len(errors),
        "errors": errors
    }
    
    logger.info("="*50)
    logger.info("KET QUA DANH GIA")
    logger.info("="*50)
    logger.info(f"Recall@5:    {avg_recall:.4f}")
    logger.info(f"Precision@5: {avg_precision:.4f}")
    logger.info(f"Tong queries: {len(predictions)}")
    logger.info(f"Loi (>5 docs): {len(errors)}")
    logger.info("="*50)
    
    if avg_recall > 0.5:
        logger.info("Recall@5 > 0.5 - Dat yeu cau!")
    else:
        logger.warning("Recall@5 <= 0.5 - Can cai thien!")
    
    return result


# ============ HÀM PUBLIC ĐỂ NOTEBOOK GỌI ============

def get_reranker(force_reload: bool = False) -> LegalReranker:
    """Lay hoac load reranker tu checkpoint"""
    try:
        return LegalReranker().load(force_reload=force_reload)
    except Exception as e:
        logger.error(f"Failed to load reranker: {e}")
        reranker = LegalReranker()
        reranker.reranker = None
        return reranker


def get_llm(force_reload: bool = False) -> LegalModelWrapper:
    """Lay hoac load Legal LLM tu checkpoint"""
    return LegalModelWrapper().load(force_reload=force_reload)


def get_embedder(force_reload: bool = False) -> EmbeddingWrapper:
    """Lay hoac load embedding model"""
    return EmbeddingWrapper().load(force_reload=force_reload)


def save_submission(submission: Dict, filename: str = "submission.json"):
    """Luu submission ra file JSON va ZIP"""
    # Dinh dung format theo yeu cau BTC
    formatted = {}
    for qid, docs in submission.items():
        formatted[str(qid)] = {"answer": docs}
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)
    logger.info(f"Submission saved to {filename}")
    
    # Tao ZIP
    import zipfile
    zip_filename = filename.replace(".json", ".zip")
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename)
    logger.info(f"Submission ZIP saved to {zip_filename}")


def generate_submission(
    queries: List[Dict], 
    retrieve_func, 
    reranker=None,
    top_k: int = 5
) -> Dict:
    """
    Tao submission tu danh sach queries
    
    Args:
        queries: [{"id": "q1", "text": "..."}, ...]
        retrieve_func: Ham nhan query text, tra ve candidates
        reranker: Instance cua LegalReranker (neu None thi tu load)
        top_k: So luong ket qua (mac dinh 5)
    
    Returns:
        {query_id: [doc_ids]}
    """
    if reranker is None:
        reranker = get_reranker()
    
    submission = {}
    total = len(queries)
    
    logger.info(f"Dang tao submission cho {total} queries...")
    
    for i, item in enumerate(tqdm(queries, desc="Processing queries")):
        qid = item.get("id", f"q_{i}")
        query = item.get("text", item.get("question", ""))
        
        candidates = retrieve_func(query)
        top_docs = reranker.rerank(query, candidates, top_k=top_k)
        submission[str(qid)] = top_docs
    
    logger.info(f"Da tao submission voi {len(submission)} queries")
    return submission


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    print("="*50)
    print("Testing member_c.py...")
    print("="*50)
    
    # Test reranker
    try:
        print("\n1. Testing Reranker...")
        reranker = get_reranker()
        print(f"   Reranker loaded: {reranker.model_name}")
    except Exception as e:
        print(f"   Reranker failed: {e}")
    
    # Test evaluation
    print("\n2. Testing Evaluation...")
    gt = {"q1": ["doc1", "doc2"]}
    pred = {"q1": ["doc1", "doc3"]}
    result = evaluate_recall_precision(gt, pred)
    print(f"   Recall: {result['recall']:.4f}")
    print(f"   Precision: {result['precision']:.4f}")
    
    print("\n" + "="*50)
    print("Test complete!")