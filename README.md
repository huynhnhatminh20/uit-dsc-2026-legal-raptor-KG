# UIT DSC 2026 - LegalIR: RAPTOR + Knowledge Graph for Vietnamese Legal Retrieval

[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Legal Model](https://img.shields.io/badge/legal-Qwen3--4B--legal--pretrain-orange)](https://huggingface.co/VLSP2025-LegalSML/qwen3-4b-legal-pretrain)

> **Repo nộp bài:** `https://github.com/huynhnhatminh20/uit-dsc-2026-legal-raptor-KG` — nộp toàn bộ folder GitHub (Cách 2: notebook chỉ train & đọc KQ)

Hệ thống truy vấn pháp luật tiếng Việt cho UIT Data Science Challenge 2026 - Task 1 LegalIR. Với một câu hỏi pháp luật, hệ thống trả về tối đa 05 `document_id` chứa thông tin cần thiết (đánh giá chính `Recall@5`, phụ `Precision@5`).

> **Based on:** [incidentfox/OpenRag](https://github.com/incidentfox/OpenRag) — MIT License. Toàn bộ core RAPTOR, Vector Store, Knowledge Graph, Hybrid Retrieval được kế thừa từ OpenRag và điều chỉnh cho văn bản pháp luật tiếng Việt với Legal LLM.

---

## Mô hình vận hành (Cách 2)

**GitHub là Source of Truth + Kaggle Notebook chỉ để Train & Đọc Kết Quả** — Leader không paste code.

| Thành viên | Module trên GitHub | Nhiệm vụ |
|------------|-------------------|----------|
| **A** | `member_a.py` / `src/raptor/` | Chunk Điều→Khoản→Đoạn, build RAPTOR bằng Legal 4B, Vector Store `BAAI/bge-m3` → FAISS |
| **B** | `member_b.py` / `src/graph/` | KG (Legal 4B → NetworkX), BM25, Hybrid RRF |
| **C** | `member_c.py` / `src/reranker.py` | Wrapper Legal 4B 4-bit, `BAAI/bge-reranker-v2-m3` lọc top 5, `Recall@5/Precision@5` |

Kaggle `notebook/train.ipynb` chỉ có 5 cell: `git clone` → `import member_a/b/c` → `train` → `rerank` → `submission.json`. Xem chi tiết trong `docs/huong dẫn.docx`.

---

## Quick Start (Leader trên Kaggle)

### 1. Clone repo (trong Kaggle Notebook, Internet ON)
```bash
!rm -rf uit-dsc-2026-legal-raptor-KG
!git clone https://github.com/huynhnhatminh20/uit-dsc-2026-legal-raptor-KG.git
```

### 2. Cài đặt
```bash
pip install -r requirements.txt
# requirements chính: torch, transformers, accelerate, bitsandbytes, sentence-transformers, rank-bm25, networkx, faiss-cpu
```

### 3. Thêm dataset LegalIR vào Kaggle
Upload `data_legalir/` (33k văn bản, `train.json` 7000 query, `public-official.json` 1000 query) thành Kaggle Dataset riêng → Add Input vào notebook. Không push `data_legalir/` lên GitHub (vượt 100MB).

### 4. Train & Inference (Leader Run All)
```python
import sys; sys.path.append('uit-dsc-2026-legal-raptor-KG')
from member_a import build_raptor, build_vector_store
from member_b import build_graph, build_bm25, hybrid_retrieve
from member_c import LegalReranker

build_raptor()        # A: checkpoint /kaggle/working/A_checkpoint/
build_vector_store()  # A: FAISS
build_graph()         # B: /kaggle/working/B_checkpoint/
build_bm25()          # B
reranker = LegalReranker().load()  # C: bge-reranker-v2-m3
# for q in public-official.json: candidates = hybrid_retrieve(q); top5 = reranker.rerank(q, candidates)
```

Output: `/kaggle/working/submission.json` → zip `submission.zip` (chỉ chứa `submission.json`).

### 5. Đánh giá local
```python
from member_c import evaluate_recall_precision
evaluate_recall_precision(gt_dict, pred_dict)  # Recall@5, Precision@5, ràng buộc >5 → 0
```

---

## Model chốt (bắt buộc dùng Legal)

| Module | Model | Mục đích |
|--------|-------|----------|
| RAPTOR | `VLSP2025-LegalSML/qwen3-4b-legal-pretrain` (backup `luanngo/Qwen3-4B-VietNamese-Legal-Chat`) | Tóm tắt cụm chunk |
| Embedding | `BAAI/bge-m3` | Dense retrieval, clustering |
| Reranker | `BAAI/bge-reranker-v2-m3` | Chấm chéo [query, passage] → top 5 |
| KG | Legal 4B + NetworkX | Trích thực thể pháp lý |
| BM25 | `rank-bm25` | Khớp số hiệu điều luật |

---

## Cấu trúc repo khi nộp

```
uit-dsc-2026-legal-raptor-KG/
├── README.md
├── LICENSE (MIT - giữ nguyên từ OpenRag)
├── requirements.txt
├── member_a.py / src/raptor/    # A
├── member_b.py / src/graph/     # B
├── member_c.py / src/reranker.py # C
├── notebook/train.ipynb         # Leader chạy (5 cell)
├── docs/huong dẫn.docx          # Hướng dẫn Cách 2 đầy đủ
└── submission.json              # Sinh ra sau train
```

---

## Quy trình Git cho team 3

```bash
git checkout -b c/reranker  # mỗi người branch riêng
# code member_c.py
git add member_c.py && git commit -m "c: reranker" && git push origin c/reranker
# tạo Pull Request trên GitHub → Leader merge vào main
# Leader trên Kaggle: !git pull origin main → Run All
```

---

## License & Citation

MIT License — see [LICENSE](./LICENSE). Original work by [incidentfox/OpenRag](https://github.com/incidentfox/OpenRag).

```bibtex
@software{openrag_2026,
  title = {Multi-Strategy RAG for Multi-Hop Question Answering},
  author = {incidentfox},
  year = {2026},
  url = {https://github.com/incidentfox/OpenRag}
}
@software{legalir_2026,
  title = {UIT DSC 2026 LegalIR - RAPTOR + KG with Legal LLM},
  author = {huynhnhatminh20},
  year = {2026},
  url = {https://github.com/huynhnhatminh20/uit-dsc-2026-legal-raptor-KG}
}
```

## Acknowledgments
- [RAPTOR](https://arxiv.org/abs/2401.18059), [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3), [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [VLSP2025-LegalSML/qwen3-4b-legal-pretrain](https://huggingface.co/VLSP2025-LegalSML/qwen3-4b-legal-pretrain)
- Built on [OpenRag](https://github.com/incidentfox/OpenRag)
