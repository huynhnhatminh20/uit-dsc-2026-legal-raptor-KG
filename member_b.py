"""
member_b.py - Stub cho B (de C test duoc)
Khi B lam xong se thay the bang code that (Graph + BM25 + RRF).
"""
import pathlib, json, random
def build_graph(legal=None):
    print("[STUB B] build_graph() - mock")
    return None
def build_bm25():
    print("[STUB B] build_bm25() - mock")
    return None
def hybrid_retrieve(query, top_k=50):
    """
    Mock: tra ve candidates tu data_legalir neu co, khong thi tra mock.
    De C test reranker ma khong can B.
    """
    # Thu doc tu selected-contexts neu co tren Kaggle
    for p in [pathlib.Path("/kaggle/input/legalir/selected-contexts"),
              pathlib.Path("data_legalir/selected-contexts/selected-contexts"),
              pathlib.Path("/kaggle/input/data-legalir/selected-contexts")]:
        if p.exists():
            files = list(p.glob("context_*.json"))
            if files:
                import json, random
                cands = []
                for f in random.sample(files, min(5, len(files))):
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        for doc_id, doc in list(data.items())[:1]:
                            txt = doc.get("passage") or doc.get("text") or str(doc)[:500]
                            cands.append({"doc_id": str(doc_id), "text": txt[:1000]})
                    except: pass
                if cands:
                    return cands[:top_k]
    # Fallback mock cung
    return [
        {"doc_id": "177504", "text": "Nghi dinh 93/2021 ve van dong tiep nhan phan phoi nguon dong gop tu nguyen phai cong khai minh bach"},
        {"doc_id": "740", "text": "QUYET DINH 5868 Co cau to chuc Vu Trang thiet bi y te"},
        {"doc_id": "99999", "text": "Van ban nhieu khong lien quan"},
    ][:top_k]
