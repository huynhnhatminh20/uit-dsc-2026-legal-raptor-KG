"""
member_c.py - Thanh vien C: Model Infrastructure + Reranker + Evaluation
Repo: https://github.com/huynhnhatminh20/uit-dsc-2026-legal-raptor-KG
Notebook chi import file nay, khong paste code. Leader chi train.
"""
import os, pickle, pathlib, torch, json
from pathlib import Path

C_CKPT = Path("/kaggle/working/C_checkpoint")
C_CKPT.mkdir(parents=True, exist_ok=True)

# ============ Cell 8: Legal 4B + Embedding ============
LEGAL_ID = "VLSP2025-LegalSML/qwen3-4b-legal-pretrain"
FALLBACK_ID = "luanngo/Qwen3-4B-VietNamese-Legal-Chat"

class LegalModelWrapper:
    def __init__(self, model_id=LEGAL_ID):
        self.model_id = model_id
        self.tokenizer = None
        self.model = None
    def load(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        flag = C_CKPT / "legal_4b.flag"
        if flag.exists() and self.model is not None:
            return self
        print(f"Loading Legal 4B: {self.model_id} (4-bit)...")
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        except Exception as e:
            print(f"Fallback {FALLBACK_ID}: {e}")
            self.model_id = FALLBACK_ID
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        flag.write_text(self.model_id)
        print(f"Loaded {self.model_id}")
        return self
    def generate(self, prompt, max_new_tokens=256):
        if self.model is None: self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

class EmbeddingWrapper:
    def __init__(self, model_id="BAAI/bge-m3"):
        self.model_id = model_id
        self.model = None
    def load(self):
        if self.model is not None: return self
        from sentence_transformers import SentenceTransformer
        print(f"Loading {self.model_id}...")
        self.model = SentenceTransformer(self.model_id, device="cuda" if torch.cuda.is_available() else "cpu")
        (C_CKPT / "emb.flag").write_text(self.model_id)
        return self
    def encode(self, texts, batch_size=64):
        if self.model is None: self.load()
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

# ============ Cell 9: Reranker ============
class LegalReranker:
    def __init__(self, model_id="BAAI/bge-reranker-v2-m3", batch_size=16):
        self.model_id = model_id
        self.batch_size = batch_size
        self.model = None
    def load(self):
        if self.model is not None: return self
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = CrossEncoder(self.model_id, device=device, trust_remote_code=True)
        except Exception as e:
            print(f"Fallback bge-reranker-base: {e}")
            self.model = CrossEncoder("BAAI/bge-reranker-base", device=device)
            self.model_id = "BAAI/bge-reranker-base"
        (C_CKPT / "reranker.flag").write_text(self.model_id)
        print(f"Reranker {self.model_id} on {device}")
        return self
    def rerank(self, query, candidates, top_k=5):
        """candidates: [{doc_id, text/passage}] -> [doc_id] top_k<=5"""
        if not candidates: return []
        if self.model is None: self.load()
        pairs = [[query, c.get("text") or c.get("passage") or ""] for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        for c,s in zip(candidates, scores): c["rerank_score"] = float(s)
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return [str(c["doc_id"]) for c in ranked[:top_k]]

# ============ Cell 10: Evaluate ============
def evaluate_recall_precision(gt_dict, pred_dict, k=5):
    recalls, precisions = [], []
    violated = 0
    for qid, gt_item in gt_dict.items():
        gt = set(map(str, gt_item.get("answer", [])))
        pred_raw = pred_dict.get(str(qid), {})
        if isinstance(pred_raw, dict): pred_raw = pred_raw.get("answer", [])
        pred = list(map(str, pred_raw))
        if len(pred) > k:
            recalls.append(0.0); precisions.append(0.0); violated+=1; continue
        if len(gt)==0: continue
        if len(pred)==0:
            recalls.append(0.0); precisions.append(0.0); continue
        hit = len(gt & set(pred))
        recalls.append(hit/len(gt))
        precisions.append(hit/len(pred))
    return {"recall": sum(recalls)/len(recalls) if recalls else 0, "precision": sum(precisions)/len(precisions) if precisions else 0, "n": len(recalls), "violated": violated}
