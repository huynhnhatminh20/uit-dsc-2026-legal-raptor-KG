"""
member_c.py - Reranker + Evaluation (Phiên bản sửa lỗi)
"""

import os
import json
import pickle
import pathlib
import shutil
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import logging

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ TỰ ĐỘNG KHÔI PHỤC CHECKPOINT TỪ /kaggle/input ============

def _restore_checkpoint_from_input(folder_name: str, ckpt_dir: pathlib.Path):
    """
    Khoi phuc checkpoint da luu tu lan chay truoc, theo thu tu uu tien:
      1) Kaggle Input dataset - dung rglob (de qui MOI CAP thu muc con),
         vi Kaggle co the mount o /kaggle/input/<slug>/... (kieu cu) hoac
         /kaggle/input/datasets/<user>/<slug>/... (kieu moi, sau hon).
      2) GitHub repo da git clone - checkpoints/{folder_name} (kieu cu)
         hoac checkpoints/<ten_ngay>/{folder_name} (kieu moi, chon ngay
         moi nhat) - chi dung khi Kaggle Input khong co.
    Chi copy file CHUA CO o ckpt_dir (khong ghi de checkpoint moi hon dang co).
    """
    def _copy_missing(src: pathlib.Path) -> int:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in src.iterdir():
            dst = ckpt_dir / f.name
            if f.is_file() and not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        return copied

    # --- Nguon 1: Kaggle Input (de qui, khong quan tam sau bao nhieu cap) ---
    input_root = pathlib.Path("/kaggle/input")
    if input_root.exists():
        try:
            for src in sorted(input_root.rglob(folder_name)):
                if src.is_dir():
                    n = _copy_missing(src)
                    if n:
                        logger.info(f" Đã khôi phục {n} file checkpoint từ {src} -> {ckpt_dir}")
                    return
        except Exception as e:
            logger.warning(f" Lỗi khi quét /kaggle/input: {e}")

    # --- Nguon 2: GitHub repo da git clone (bo sung neu Kaggle Input chua co) ---
    repo_ckpt_root = pathlib.Path("uit-dsc-2026-legal-raptor-KG/checkpoints")
    if repo_ckpt_root.exists():
        try:
            candidates = [repo_ckpt_root] + sorted(
                (d for d in repo_ckpt_root.iterdir() if d.is_dir()),
                reverse=True,  # ten dang ngay-gio -> moi nhat truoc
            )
            for base in candidates:
                src = base / folder_name
                if src.is_dir():
                    n = _copy_missing(src)
                    if n:
                        logger.info(f" Đã khôi phục {n} file checkpoint từ {src} -> {ckpt_dir}")
                    return
        except Exception as e:
            logger.warning(f" Lỗi khi quét GitHub repo checkpoints: {e}")


# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/C_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_restore_checkpoint_from_input("C_checkpoint", CKPT_DIR)
RERANKER_CKPT = CKPT_DIR / "reranker.pkl"
LLM_CKPT = CKPT_DIR / "legal_llm.pkl"
EVAL_CKPT = CKPT_DIR / "eval_fn.pkl"
# Checkpoint TẠM cho vòng lặp tạo submission (retrieve + rerank từng query
# có thể rất chậm nếu có hàng trăm/nghìn câu hỏi) -> lưu định kỳ để resume,
# không phải chạy lại từ query đầu tiên nếu bị ngắt giữa chừng.
#
# [SỬA LỖI HIỆU NĂNG] Cùng họ bug với encode_with_checkpoint (member_a.py)
# và GRAPH_CKPT_INTERVAL (member_b.py): interval CỐ ĐỊNH (20) nghĩa là mỗi
# lần lưu, ta pickle lại TOÀN BỘ dict `submission` đã tích luỹ từ đầu, chứ
# không phải chỉ phần mới -> số lần ghi tỉ lệ n/20 và mỗi lần ghi có size
# tỉ lệ với số query đã xử lý => tổng I/O tăng kiểu O(n^2) nếu n (số câu
# hỏi) lớn. Với vài trăm/nghìn câu hỏi và mỗi entry chỉ ≤5 doc id thì ảnh
# hưởng không nghiêm trọng như embedding (member_a) hay graph (member_b),
# nhưng vẫn nên sửa để nhất quán và an toàn nếu tập câu hỏi lớn hơn dự kiến.
# Giải pháp: giãn interval theo tổng số query (giống _graph_ckpt_interval ở
# member_b) để tổng số lần ghi full-dict không tăng theo n.
SUBMISSION_CKPT = CKPT_DIR / "submission_partial.pkl"
SUBMISSION_SAVE_INTERVAL_MIN = 20  # tối thiểu vẫn lưu mỗi 20 query (tập nhỏ)


def _submission_save_interval(total_queries: int) -> int:
    """Khoảng cách giữa 2 lần lưu checkpoint tạm khi tạo submission.

    Luôn lưu tối thiểu SUBMISSION_SAVE_INTERVAL_MIN query/lần, nhưng nếu có
    nhiều câu hỏi thì giãn ra để tổng số lần ghi (mỗi lần ghi lại toàn bộ
    dict submission) không tăng theo n -> tránh I/O kiểu O(n^2).
    """
    return max(SUBMISSION_SAVE_INTERVAL_MIN, total_queries // 20)

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
            
            # [SỬA] device_map="auto" trên máy có NHIỀU GPU (vd Kaggle T4 x2)
            # có thể chia layer không đều giữa các GPU, khiến 1 GPU bị dồn
            # gần hết dung lượng trong lúc load dù model 4-bit lẽ ra chỉ cần
            # vài GB -> OOM ngay khi load dù GPU còn kia gần như trống. Ép
            # load hẳn vào 1 GPU duy nhất (đủ chỗ cho model 4-bit ~4B tham
            # số) để tránh kiểu chia lệch này. Nếu máy chỉ có 1 GPU hoặc
            # không có GPU, fallback về "auto"/CPU như cũ.
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                target_device_map = {"": 0}
                logger.info("   Nhiều GPU phát hiện được -> ép load Legal LLM vào GPU 0 "
                            "duy nhất (tránh device_map='auto' chia lệch gây OOM giả)")
            else:
                target_device_map = "auto"

            # Dọn cache GPU trước khi load (phòng trường hợp còn rác VRAM
            # từ model/pipeline chạy trước đó trong cùng process, dù không
            # phải nguyên nhân chính của lỗi OOM lần trước)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # [SỬA LỖI OOM] Thiếu low_cpu_mem_usage=True + torch_dtype tường minh
            # khiến from_pretrained() có xu hướng vật chất hoá trọng số ở
            # precision gốc (fp16/bf16, ~14-16GB cho model 4B) TRƯỚC khi
            # bitsandbytes kịp quantize xuống 4-bit (đáng lẽ chỉ ~2.5-3GB).
            # Trên GPU T4 15GB thì bị OOM ngay khi load, y hệt log 16:19:27
            # "Tried to allocate 40.00 MiB ... 14.42 GiB is allocated by PyTorch".
            # Giải pháp: ép low_cpu_mem_usage=True (stream từng shard thẳng
            # vào GPU rồi quantize ngay, không giữ bản full-precision trong
            # RAM/VRAM) + torch_dtype=bfloat16 tường minh cho phần compute
            # chưa quantize (embedding, norm...), tránh rơi về fp32 mặc định.
            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                quantization_config=bnb_config,
                device_map=target_device_map,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
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
        (Giữ nguyên để tương thích, batch thực tế dùng rerank_batched)
        """
        return self.rerank_batched([query], [candidates], top_k=top_k)[0]

    def rerank_batched(self, queries: List[str], candidates_list: List[List[Dict]], top_k: int = 5) -> List[List[str]]:
        """
        Giai đoạn A+B batch: gom tất cả pairs (60k với 1000q*60) predict 1 lần batch=256 + half
        Thay vì 1000 lần predict(30 pairs, batch=64) -> 1000 GPU launch -> 1 lần ~12-24s
        """
        if top_k > 5:
            logger.warning(f" top_k={top_k} vượt quá 5, tự động giới hạn xuống 5")
            top_k = 5
        if self.reranker is None:
            self.load()
        if not queries or not candidates_list:
            return []
        # Hỗ trợ cả 2 format
        def _get_text(c: Dict) -> str:
            return c.get("text") or c.get("passage") or c.get("content") or ""
        def _get_id(c: Dict) -> str:
            return str(c.get("id") or c.get("doc_id") or c.get("document_id") or "")
        # Gom pairs + index map
        all_pairs = []
        offsets = []  # (start, end) per query
        for q, cands in zip(queries, candidates_list):
            start = len(all_pairs)
            for c in (cands or []):
                all_pairs.append((q, _get_text(c)))
            offsets.append((start, len(all_pairs)))
        if not all_pairs:
            return [[] for _ in queries]
        try:
            # Half precision trên CUDA giảm 30% thời gian + VRAM
            try:
                if self.device == "cuda" and hasattr(self.reranker, "model"):
                    self.reranker.model.half()
            except Exception:
                pass
            scores = self.reranker.predict(all_pairs, batch_size=256, show_progress_bar=False)
            scores = __import__("numpy").asarray(scores)
            results = []
            for (start, end), cands in zip(offsets, candidates_list):
                if start >= end or not cands:
                    results.append([])
                    continue
                q_scores = scores[start:end]
                sorted_idx = __import__("numpy").argsort(q_scores)[::-1]
                seen=set(); res=[]
                for i in sorted_idx:
                    cid=_get_id(cands[i])
                    if not cid or cid in seen: continue
                    seen.add(cid); res.append(cid)
                    if len(res)>=top_k: break
                if not res:
                    seen=set()
                    for c in cands:
                        cid=_get_id(c)
                        if not cid or cid in seen: continue
                        seen.add(cid); res.append(cid)
                        if len(res)>=top_k: break
                results.append(res)
            logger.info(f" Reranked batch {len(queries)} queries {len(all_pairs)} pairs")
            return results
        except Exception as e:
            logger.warning(f" Reranker batch error: {e}")
            results=[]
            for cands in candidates_list:
                seen=set(); res=[]
                for c in (cands or []):
                    cid=_get_id(c)
                    if not cid or cid in seen: continue
                    seen.add(cid); res.append(cid)
                    if len(res)>=top_k: break
                results.append(res)
            return results


# ============ HyDE QUERY EXPANSION (Giai đoạn B -> 0.70) ============

def generate_hyde_docs(queries, legal=None, max_new_tokens=80, batch_size=8):
    """Sinh hypo document cho mỗi query bằng Legal LLM (HyDE). Batch theo loop, 80 tokens ~0.4s/q trên T4."""
    if legal is None or getattr(legal, "model", None) is None:
        logger.warning(" HyDE: legal LLM chưa load -> bỏ qua expansion, dùng query gốc")
        return ["" for _ in queries]
    hyde_docs=[]
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i+batch_size]
        for q in batch:
            prompt = f"Viet 1 doan van ban phap luat gia dinh (80 tu) tra loi cau hoi: {q[:300]}"
            try:
                hypo = legal.generate(prompt, max_length=max_new_tokens)
                # Cắt bỏ prompt nếu model trả cả prompt
                if prompt in hypo:
                    hypo = hypo.split(prompt)[-1].strip()
                hyde_docs.append(hypo[:500])
            except Exception as e:
                logger.warning(f" HyDE fail q{i}: {e}")
                hyde_docs.append("")
    logger.info(f" HyDE done {len(hyde_docs)}/{len(queries)} docs")
    return hyde_docs


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
     (theo docs/DSC2026_Task1_LegalIR_Data_Overview.docx:5 và LegalIR_Kaggle_Template_C.ipynb Cell 10)
    
    Args:
        ground_truth: {query_id: {"answer": [relevant_doc_ids]}} hoặc {query_id: [relevant_doc_ids]}
        predictions: {query_id: {"answer": [predicted_doc_ids]}} hoặc {query_id: [predicted_doc_ids]}
        k: Số lượng kết quả (mặc định 5)
    
    Returns:
        Dict với recall@k, precision@k, và thông báo lỗi
    """
    recalls, precisions = [], []
    errors = []
    violated = 0

    # Chuẩn hoá keys về str để tránh lệch int vs str (xungdot)
    # và duyệt theo GT (ground_truth-driven) thay vì predictions-driven
    # -> query có trong GT nhưng không có trong pred phải tính 0 điểm, không được bỏ qua
    gt_norm = {str(qid): val for qid, val in ground_truth.items()}
    pred_norm = {str(qid): val for qid, val in predictions.items()}

    for qid, gt_item in gt_norm.items():
        # GT: chấp nhận cả dạng {"answer": [...]} và [...] và {"question":..., "answer":[...]}
        # FIX: gt_item.get("answer", []) trả về None nếu JSON có "answer": null -> map(str, None) lỗi TypeError
        # Dùng (x or []) và kiểm tra kiểu list để tránh 'NoneType' object is not iterable
        if gt_item is None:
            gt = set()
        elif isinstance(gt_item, dict):
            raw = gt_item.get("answer")
            if raw is None:
                raw = []
            elif isinstance(raw, dict):
                raw = raw.get("answer")
                if raw is None:
                    raw = []
            if isinstance(raw, str):
                raw = [raw]
            elif isinstance(raw, (list, tuple, set)):
                # lọc None bên trong list (vd: ["doc1", null])
                raw = [x for x in raw if x is not None]
            elif raw is None:
                raw = []
            else:
                # kiểu lạ -> thử ép về list, nếu không được thì bỏ qua
                try:
                    raw = list(raw) if not isinstance(raw, str) else [raw]
                except Exception:
                    raw = []
            gt = set(map(str, raw))
        elif isinstance(gt_item, list):
            gt = set(map(str, [x for x in gt_item if x is not None]))
        else:
            gt = set()

        # Pred: tương thích notebook Cell 10 logic
        pred_entry = pred_norm.get(str(qid), [])
        # pred_entry có thể là {"answer": [...]}, [...] hoặc {} hoặc None
        if pred_entry is None:
            pred_raw = []
        elif isinstance(pred_entry, dict):
            pred_raw = pred_entry.get("answer")
            if pred_raw is None:
                pred_raw = []
            # fallback khi pred là {"qid": [...]} nhưng get nhầm dict rỗng
            if isinstance(pred_raw, dict):
                tmp = pred_raw.get("answer")
                pred_raw = tmp if tmp is not None else []
        else:
            pred_raw = pred_entry
        if isinstance(pred_raw, dict):
            tmp = pred_raw.get("answer")
            pred_raw = tmp if tmp is not None else []
        # đảm bảo list + lọc None
        if pred_raw is None:
            pred_raw = []
        if isinstance(pred_raw, str):
            pred_raw = [pred_raw]
        elif isinstance(pred_raw, (list, tuple, set)):
            pred_raw = [x for x in pred_raw if x is not None]
        elif not isinstance(pred_raw, list):
            try:
                pred_raw = list(pred_raw) if not isinstance(pred_raw, str) else [pred_raw]
                pred_raw = [x for x in pred_raw if x is not None]
            except Exception:
                pred_raw = []
        pred = list(map(str, pred_raw))

        # Ràng buộc cuộc thi: >k → 0 điểm (k=5)
        if len(pred) > k:
            error_msg = f" Query {qid} trả về {len(pred)} docs (tối đa {k}) -> 0 điểm"
            errors.append(error_msg)
            logger.warning(error_msg)
            recalls.append(0.0)
            precisions.append(0.0)
            violated += 1
            continue

        if len(gt) == 0:
            # Không có GT (như public-official) -> bỏ qua, không tính vào trung bình
            continue
        if len(pred) == 0:
            recalls.append(0.0)
            precisions.append(0.0)
            continue
        hit = len(gt & set(pred))
        recalls.append(hit / len(gt) if len(gt) > 0 else 0.0)
        precisions.append(hit / len(pred) if len(pred) > 0 else 0.0)

    # Tính trung bình
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    
    result = {
        "recall": avg_recall,
        "precision": avg_precision,
        "recall@5": avg_recall,
        "precision@5": avg_precision,
        "n": len(recalls),
        "total_queries": len(recalls),
        "violated": violated,
        "errors": errors
    }
    
    # In kết quả
    logger.info("="*50)
    logger.info(" KẾT QUẢ ĐÁNH GIÁ")
    logger.info("="*50)
    logger.info(f" Recall@5:    {avg_recall:.4f}")
    logger.info(f" Precision@5: {avg_precision:.4f}")
    logger.info(f" Tổng queries: {len(recalls)}")
    logger.info(f" Lỗi (>5 docs): {violated}")
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
    top_k: int = 5,
    ckpt_path: Optional[pathlib.Path] = None,
) -> Dict:
    """
    Tạo submission từ danh sách queries
    
    Args:
        queries: [{"id": "q1", "text": "..."}, ...]
        retrieve_func: Hàm nhận query text, trả về candidates
        reranker: Instance của LegalReranker (nếu None thì tự load)
        top_k: Số lượng kết quả (mặc định 5)
        ckpt_path: [MỚI] file checkpoint tạm dùng để resume. Mặc định
            SUBMISSION_CKPT (submission thật nộp BTC). Khi chạy đánh giá
            trên train.json (evaluate_on_train) ta truyền 1 checkpoint
            KHÁC (TRAIN_EVAL_CKPT) để không đè/lẫn với checkpoint của
            submission thật.
    
    Returns:
        {query_id: [doc_ids]}
    """
    if reranker is None:
        reranker = get_reranker()

    if ckpt_path is None:
        ckpt_path = SUBMISSION_CKPT

    # Resume từ checkpoint tạm nếu có (từ lần chạy trước bị ngắt giữa chừng)
    submission = {}
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            submission = pickle.load(f)
        logger.info(f" Resume submission từ checkpoint ({ckpt_path.name}): đã có {len(submission)} queries")

    total = len(queries)
    save_interval = _submission_save_interval(total)

    logger.info(f" Đang tạo submission cho {total} queries...")
    logger.info(f"   Lưu checkpoint tạm mỗi {save_interval} queries")
    
    for i, item in enumerate(tqdm(queries, desc="Processing queries")):
        qid = item.get("id", f"q_{i}")
        qid_str = str(qid)

        # Đã xử lý xong ở lần chạy trước -> bỏ qua, không làm lại
        if qid_str in submission:
            continue

        query = item.get("text", item.get("question", ""))
        
        # Lấy candidates từ retrieval
        candidates = retrieve_func(query)
        
        # Rerank và lấy top_k
        top_docs = reranker.rerank(query, candidates, top_k=top_k)
        # QUAN TRỌNG: định dạng BTC yêu cầu là {"qid": {"answer": [...]}}.
        # Trước đây hàm này trả {"qid": [...]} (thiếu bọc "answer") ->
        # scoring.py của BTC sẽ lỗi vì cố đọc predictions[qid]['answer'].
        submission[qid_str] = {"answer": top_docs}

        # Lưu checkpoint tạm định kỳ, không đợi xử lý hết mới lưu
        if (i + 1) % save_interval == 0:
            with open(ckpt_path, "wb") as f:
                pickle.dump(submission, f)
            logger.info(f"    Checkpoint submission tạm: {len(submission)}/{total} queries")

    # Lưu lần cuối để chắc chắn không sót query nào
    with open(ckpt_path, "wb") as f:
        pickle.dump(submission, f)
    
    logger.info(f" Đã tạo submission với {len(submission)} queries")
    return submission


# ============ ĐÁNH GIÁ TRÊN train.json (CÓ GROUND TRUTH THẬT) ============

# Checkpoint RIÊNG cho vòng chạy đánh giá trên train.json, tách biệt hẳn
# với SUBMISSION_CKPT (submission thật nộp BTC dựa trên public-official.json
# - vốn answer=null hết nên Recall/Precision luôn ra 0/0, không phản ánh
# chất lượng model). Tách file để 2 việc không đè checkpoint của nhau.
TRAIN_EVAL_CKPT = CKPT_DIR / "train_eval_partial.pkl"


def evaluate_on_train(
    train_json_path: str,
    retrieve_func,
    reranker=None,
    top_k: int = 5,
    sample_size: Optional[int] = None,
    seed: int = 42,
    ckpt_path: Optional[pathlib.Path] = None,
) -> Dict:
    """
    Đo Recall@5 / Precision@5 THẬT bằng cách chạy full pipeline
    (retrieve + rerank) trên train.json - file DUY NHẤT hiện có chứa
    ground truth thật (answer != null), khác với public-official.json
    (answer luôn null -> evaluate_recall_precision() sẽ luôn trả về
    Tổng queries = 0, không dùng để đánh giá model được).

    Args:
        train_json_path: đường dẫn tới train.json, format:
            {qid: {"question": "...", "answer": ["doc_id", ...]}, ...}
        retrieve_func: hàm nhận query text -> candidates (giống generate_submission)
        reranker: Instance LegalReranker (None thì tự load)
        top_k: số doc lấy ra mỗi câu (mặc định 5, đúng luật thi)
        sample_size: [MẶC ĐỊNH None = CHẠY FULL train.json]. Chỉ lấy mẫu
            ngẫu nhiên khi NOTEBOOK chủ động truyền một số cụ thể (vd 200)
            để có kết quả nhanh trong lúc thử nghiệm. Bản thân file
            member_c.py này không tự ý giới hạn mẫu - hàm mặc định chạy
            hết toàn bộ train.json (7000 câu, ước ~33 giờ theo tốc độ log
            trước ~17s/câu) trừ khi bị notebook yêu cầu lấy mẫu.
        seed: seed để lấy mẫu tái lập được (chỉ có tác dụng khi sample_size != None)
        ckpt_path: checkpoint resume riêng, mặc định TRAIN_EVAL_CKPT.
            LƯU Ý: nếu đổi sample_size/seed giữa các lần chạy, các qid
            trong checkpoint cũ (đã tính theo mẫu cũ) vẫn được giữ và dùng
            lại nếu trùng id với mẫu mới; nếu muốn chắc chắn mẫu mới chạy
            từ đầu, hãy xoá TRAIN_EVAL_CKPT trước khi gọi lại.

    Returns:
        Dict kết quả từ evaluate_recall_precision(), tức:
        {"recall", "precision", "recall@5", "precision@5",
         "n"/"total_queries", "violated", "errors"}
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_gt_full = json.load(f)

    total_available = len(train_gt_full)
    if sample_size is None or sample_size >= total_available:
        chosen_ids = list(train_gt_full.keys())
        logger.info(f" evaluate_on_train: dùng FULL {total_available} câu trong train.json "
                    f"(mặc định của member_c.py - không lấy mẫu trừ khi notebook truyền sample_size)")
    else:
        import random
        rng = random.Random(seed)
        chosen_ids = rng.sample(list(train_gt_full.keys()), sample_size)
        logger.info(f" evaluate_on_train: lấy mẫu {sample_size}/{total_available} câu "
                    f"(seed={seed}) để đánh giá nhanh")

    ground_truth = {qid: train_gt_full[qid] for qid in chosen_ids}
    queries = [
        {"id": qid, "text": train_gt_full[qid].get("question", "")}
        for qid in chosen_ids
    ]

    if ckpt_path is None:
        ckpt_path = TRAIN_EVAL_CKPT

    predictions = generate_submission(
        queries=queries,
        retrieve_func=retrieve_func,
        reranker=reranker,
        top_k=top_k,
        ckpt_path=ckpt_path,
    )

    result = evaluate_recall_precision(ground_truth, predictions, k=top_k)
    return result


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

    # 3. Cách gọi evaluate_on_train() trong notebook (KHÔNG chạy tự động ở
    # đây vì cần retrieve_func thật từ member_a/member_b). Ví dụ dùng:
    #
    # from member_c import evaluate_on_train
    # # Mặc định sample_size=None -> chạy FULL train.json (rất lâu, ~33h nếu 7000 câu).
    # # Muốn thử nhanh trước, notebook chủ động truyền sample_size=200 (~1h):
    # result = evaluate_on_train(
    #     train_json_path=TRAIN_JSON_PATH,
    #     retrieve_func=hybrid_retrieve,
    #     reranker=reranker,
    #     top_k=5,
    #     sample_size=200,   # bỏ dòng này (hoặc =None) để chạy full
    # )
    # print(result["recall@5"], result["precision@5"], result["total_queries"])