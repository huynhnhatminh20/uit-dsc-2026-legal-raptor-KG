"""
member_c.py - Reranker + Evaluation (Phiên bản sửa lỗi)
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

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/C_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RERANKER_CKPT = CKPT_DIR / "reranker.pkl"
LLM_CKPT = CKPT_DIR / "legal_llm.pkl"
EVAL_CKPT = CKPT_DIR / "eval_fn.pkl"

# ============ CONSTANTS ============
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
LLM_MODEL = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"


# ============ LEGAL LLM WRAPPER ============

class LegalModelWrapper:
    """
    Wrapper cho VLSP2025-LegalSML/qwen3-4b-legal-pretrain
    Sử dụng 4-bit quantization để tiết kiệm memory
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load(self, force_reload: bool = False):
        """Load model với checkpoint"""
        if not force_reload and LLM_CKPT.exists():
            logger.info(f" Load Legal LLM từ checkpoint: {LLM_CKPT}")
            with open(LLM_CKPT, "rb") as f:
                return pickle.load(f)
        
        logger.info(f" Loading {LLM_MODEL} with 4-bit quantization...")
        
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
            
            # Lưu checkpoint
            with open(LLM_CKPT, "wb") as f:
                pickle.dump(self, f)
            logger.info(f" Legal LLM checkpoint saved to {LLM_CKPT}")
            
        except Exception as e:
            logger.warning(f" Could not load LLM: {e}. Using fallback.")
            self.model = None
            self.tokenizer = None
        
        return self
    
    def generate(self, prompt: str, max_length: int = 256) -> str:
        """Generate text using Legal LLM"""
        if self.model is None:
            logger.warning(" LLM not loaded!")
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
            logger.warning(f" Generation failed: {e}")
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
        
        logger.info(f" Loading embedding model: BAAI/bge-m3...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("BAAI/bge-m3", device=self.device)
            logger.info(f" Embedding model loaded on {self.device}")
        except Exception as e:
            logger.warning(f" Could not load embedding model: {e}")
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
    Lọc tối đa 05 document_id (theo yêu cầu cuộc thi)
    """
    
    def __init__(self):
        self.model_name = RERANKER_MODEL
        self.reranker = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 512
    
    def load(self, force_reload: bool = False):
        """Load reranker với checkpoint"""
        if not force_reload and RERANKER_CKPT.exists():
            logger.info(f" Load Reranker từ checkpoint: {RERANKER_CKPT}")
            with open(RERANKER_CKPT, "rb") as f:
                return pickle.load(f)
        
        logger.info(f" Loading {self.model_name} on {self.device}...")
        
        self.reranker = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
            trust_remote_code=True
        )
        
        # Lưu checkpoint
        with open(RERANKER_CKPT, "wb") as f:
            pickle.dump(self, f)
        logger.info(f" Reranker checkpoint saved to {RERANKER_CKPT}")
        
        return self
    
    def rerank(self, query: str, candidates: List[Dict], top_k: int = 5) -> List[str]:
        """
        Rerank candidates và trả về top_k document IDs
        
        Args:
            query: Câu hỏi
            candidates: List [{"id": "doc1", "text": "..."}]
            top_k: Số lượng kết quả (mặc định 5, MAX 5 theo luật thi)
        
        Returns:
            List[str]: Danh sách document IDs (tối đa 5)
        """
        # LUẬT THI: TỐI ĐA 5 DOCUMENTS
        if top_k > 5:
            logger.warning(f" top_k={top_k} vượt quá 5, tự động giới hạn xuống 5")
            top_k = 5
        
        if self.reranker is None:
            self.load()
        
        if not candidates:
            return []
        
        # Chuẩn bị pairs cho cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        
        try:
            # Predict scores
            scores = self.reranker.predict(pairs, batch_size=32)
            
            # Sort by score descending
            sorted_indices = np.argsort(scores)[::-1]
            
            # Lấy top_k ID, khử trùng lặp (phòng hờ nếu candidates còn sót
            # 2 chunk khác nhau của CÙNG 1 document_id - chỉ giữ bản có
            # điểm rerank cao nhất, tránh lãng phí 1 trong 5 slot cho phép)
            result_ids = []
            seen = set()
            for i in sorted_indices:
                cid = candidates[i]["id"]
                if cid in seen:
                    continue
                seen.add(cid)
                result_ids.append(cid)
                if len(result_ids) >= top_k:
                    break
            
            logger.info(f" Reranked {len(candidates)} candidates -> {len(result_ids)} docs (unique)")
            return result_ids
            
        except Exception as e:
            logger.warning(f" Reranker error: {e}")
            # Fallback: return first candidates
            return [c["id"] for c in candidates[:min(top_k, len(candidates))]]


# ============ EVALUATION ============

def recall_at_k(predicted: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Tính Recall@k"""
    if not ground_truth:
        return 0.0
    hits = len(set(predicted[:k]) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0.0


def precision_at_k(predicted: List[str], ground_truth: List[str], k: int = 5) -> float:
    """Tính Precision@k"""
    if not predicted:
        return 0.0
    hits = len(set(predicted[:k]) & set(ground_truth))
    return hits / min(k, len(predicted)) if predicted else 0.0


def evaluate_recall_precision(
    ground_truth: Dict, 
    predictions: Dict, 
    k: int = 5
) -> Dict:
    """
    Đánh giá Recall@k và Precision@k
    
     QUAN TRỌNG: Nếu query có >5 documents → 0 điểm cho query đó
    
    Args:
        ground_truth: {query_id: [relevant_doc_ids]}
        predictions: {query_id: [predicted_doc_ids]}
        k: Số lượng kết quả (mặc định 5)
    
    Returns:
        Dict với recall@k, precision@k, và thông báo lỗi
    """
    recall_scores = []
    precision_scores = []
    errors = []
    
    for qid, pred_value in predictions.items():
        # Chấp nhận cả 2 dạng: {"qid": {"answer": [...]}} (đúng format
        # submission.json thật của BTC) và {"qid": [...]} (dạng rút gọn cũ)
        pred_docs = pred_value.get("answer", []) if isinstance(pred_value, dict) else pred_value

        # KIỂM TRA: Không được vượt quá 5 documents
        if len(pred_docs) > 5:
            error_msg = f" Query {qid} trả về {len(pred_docs)} docs (tối đa 5) -> 0 điểm"
            errors.append(error_msg)
            logger.warning(error_msg)
            recall_scores.append(0.0)
            precision_scores.append(0.0)
            continue
        
        # ground_truth có thể là train.json gốc {"qid": {"question":..., "answer":[...]}}
        # hoặc dạng rút gọn {"qid": [...]} -> chấp nhận cả 2.
        gt_value = ground_truth.get(qid, [])
        gt_docs = gt_value.get("answer", []) if isinstance(gt_value, dict) else gt_value
        r5 = recall_at_k(pred_docs, gt_docs, k=k)
        p5 = precision_at_k(pred_docs, gt_docs, k=k)
        recall_scores.append(r5)
        precision_scores.append(p5)
    
    # Tính trung bình
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
    
    # In kết quả
    logger.info("="*50)
    logger.info(" KẾT QUẢ ĐÁNH GIÁ")
    logger.info("="*50)
    logger.info(f" Recall@5:    {avg_recall:.4f}")
    logger.info(f" Precision@5: {avg_precision:.4f}")
    logger.info(f" Tổng queries: {len(predictions)}")
    logger.info(f" Lỗi (>5 docs): {len(errors)}")
    logger.info("="*50)
    
    if avg_recall > 0.5:
        logger.info(" Recall@5 > 0.5 - Đạt yêu cầu!")
    else:
        logger.warning(" Recall@5 <= 0.5 - Cần cải thiện!")
    
    return result


# ============ HÀM PUBLIC ĐỂ NOTEBOOK GỌI ============

def get_reranker(force_reload: bool = False) -> LegalReranker:
    """Lấy hoặc load reranker từ checkpoint"""
    try:
        return LegalReranker().load(force_reload=force_reload)
    except Exception as e:
        logger.error(f" Failed to load reranker: {e}")
        # Fallback: tạo instance mới
        reranker = LegalReranker()
        reranker.reranker = None
        return reranker


def get_llm(force_reload: bool = False) -> LegalModelWrapper:
    """Lấy hoặc load Legal LLM từ checkpoint"""
    return LegalModelWrapper().load(force_reload=force_reload)


def get_embedder(force_reload: bool = False) -> EmbeddingWrapper:
    """Lấy hoặc load embedding model"""
    return EmbeddingWrapper().load(force_reload=force_reload)


def save_submission(submission: Dict, filename: str = "submission.json"):
    """Lưu submission ra file JSON và ZIP"""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    logger.info(f" Submission saved to {filename}")
    
    # Tạo ZIP
    import zipfile
    zip_filename = filename.replace(".json", ".zip")
    with zipfile.ZipFile(zip_filename, "w") as zipf:
        zipf.write(filename)
    logger.info(f" Submission ZIP saved to {zip_filename}")


def generate_submission(
    queries: List[Dict], 
    retrieve_func, 
    reranker=None,
    top_k: int = 5
) -> Dict:
    """
    Tạo submission từ danh sách queries
    
    Args:
        queries: [{"id": "q1", "text": "..."}, ...]
        retrieve_func: Hàm nhận query text, trả về candidates
        reranker: Instance của LegalReranker (nếu None thì tự load)
        top_k: Số lượng kết quả (mặc định 5)
    
    Returns:
        {query_id: [doc_ids]}
    """
    if reranker is None:
        reranker = get_reranker()
    
    submission = {}
    total = len(queries)
    
    logger.info(f" Đang tạo submission cho {total} queries...")
    
    for i, item in enumerate(tqdm(queries, desc="Processing queries")):
        qid = item.get("id", f"q_{i}")
        query = item.get("text", item.get("question", ""))
        
        # Lấy candidates từ retrieval
        candidates = retrieve_func(query)
        
        # Rerank và lấy top_k
        top_docs = reranker.rerank(query, candidates, top_k=top_k)
        # QUAN TRỌNG: định dạng BTC yêu cầu là {"qid": {"answer": [...]}}.
        # Trước đây hàm này trả {"qid": [...]} (thiếu bọc "answer") ->
        # scoring.py của BTC sẽ lỗi vì cố đọc predictions[qid]['answer'].
        submission[str(qid)] = {"answer": top_docs}
    
    logger.info(f" Đã tạo submission với {len(submission)} queries")
    return submission


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    print("="*50)
    print(" Testing member_c.py...")
    print("="*50)
    
    # Test reranker
    try:
        print("\n1. Testing Reranker...")
        reranker = get_reranker()
        print(f"    Reranker loaded: {reranker.model_name}")
    except Exception as e:
        print(f"    Reranker failed: {e}")
    
    # Test evaluation
    print("\n2. Testing Evaluation...")
    gt = {"q1": ["doc1", "doc2"]}
    pred = {"q1": ["doc1", "doc3"]}
    result = evaluate_recall_precision(gt, pred)
    print(f"    Recall: {result['recall']:.4f}")
    print(f"    Precision: {result['precision']:.4f}")
    
    print("\n" + "="*50)
    print(" Test complete!")