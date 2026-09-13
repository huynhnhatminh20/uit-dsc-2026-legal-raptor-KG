"""
member_b.py - Knowledge Graph + BM25 + Hybrid Retrieval (Phiên bản tối ưu)
"""

import os
import json
import pickle
import pathlib
import shutil
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import logging
from collections import defaultdict

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
CKPT_DIR = pathlib.Path("/kaggle/working/B_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_restore_checkpoint_from_input("B_checkpoint", CKPT_DIR)
GRAPH_CKPT = CKPT_DIR / "graph.pkl"
BM25_CKPT = CKPT_DIR / "bm25.pkl"
ENTITY_CKPT = CKPT_DIR / "entities.pkl"
RELATION_CKPT = CKPT_DIR / "relations.pkl"
# Checkpoint TẠM cho vòng lặp build graph (đặc biệt tốn thời gian khi có
# LLM entity/relation extraction cho từng node) -> lưu định kỳ để resume,
# tránh mất hết tiến trình nếu Kaggle ngắt giữa chừng.
#
# [SỬA LỖI HIỆU NĂNG] GRAPH_CKPT_INTERVAL=200 CỐ ĐỊNH là bug: mỗi lần lưu,
# code pickle lại TOÀN BỘ graph G đã build từ đầu (không phải chỉ phần mới)
# -> với hàng trăm nghìn node, cứ 200 node lại ghi lại full graph 1 lần =
# I/O tăng theo kiểu O(n^2), y hệt bug encode_with_checkpoint ở member_a.py.
# Giờ đổi interval SCALE THEO TỔNG SỐ NODE (xem _graph_ckpt_interval) thay
# vì 1 số cố định nhỏ, để tổng số lần ghi full-graph không phụ thuộc n (chỉ
# ghi khoảng ~15-20 lần dù n lớn hay nhỏ).
GRAPH_PARTIAL_CKPT = CKPT_DIR / "graph_partial.pkl"
GRAPH_CKPT_INTERVAL_MIN = 500


def _graph_ckpt_interval(total_nodes: int) -> int:
    """Khoảng cách giữa 2 lần lưu checkpoint tạm khi build graph.

    Luôn lưu tối thiểu GRAPH_CKPT_INTERVAL_MIN node/lần, nhưng nếu graph có
    nhiều node thì giãn ra để tổng số lần ghi (mỗi lần ghi lại full graph)
    không tăng theo n -> tránh I/O kiểu O(n^2).
    """
    return max(GRAPH_CKPT_INTERVAL_MIN, total_nodes // 20)


# [SỬA LỖI] Nếu truyền `legal` (LLM) vào build_graph(), bản cũ sẽ gọi
# legal.generate(...) CHO TỪNG NODE để trích entity/relation. Với vài trăm
# nghìn node, gọi generate() (autoregressive, vài giây/lần) từng đó lần là
# BẤT KHẢ THI (mất hàng chục/hàng trăm giờ), dù có checkpoint cũng không
# cứu được vì bản chất là quá chậm chứ không phải do mất tiến trình. Giới
# hạn: chỉ thật sự dùng LLM để trích entity/relation khi tổng số node đủ
# nhỏ; ngược lại tự động dùng fallback bằng regex (nhanh, đã có sẵn) và log
# cảnh báo rõ ràng thay vì âm thầm treo máy.
LLM_ENTITY_EXTRACTION_MAX_NODES = 3000

# ============ CONSTANTS ============
RRF_K = 60
MAX_CANDIDATES = 50
EMBEDDING_MODEL = "BAAI/bge-m3"

# ============ CACHE (module-level) ============
# QUAN TRỌNG: trước đây get_dense_scores() load lại SentenceTransformer từ
# đầu MỖI câu hỏi (rất chậm), get_bm25_scores()/get_graph_scores() cũng
# unpickle lại BM25/Graph từ đĩa MỖI câu hỏi. Với ~1000 câu hỏi, việc này
# khiến toàn bộ vòng inference chậm hơn hàng chục-hàng trăm lần so với cần
# thiết. Cache 1 lần, dùng lại cho tất cả câu hỏi trong cùng session.
_DENSE_EMBEDDER_CACHE = None
_GRAPH_CACHE = None
_BM25_CACHE = None


def _get_dense_embedder():
    global _DENSE_EMBEDDER_CACHE
    if _DENSE_EMBEDDER_CACHE is None:
        _DENSE_EMBEDDER_CACHE = SentenceTransformer(EMBEDDING_MODEL)
    return _DENSE_EMBEDDER_CACHE


def _get_graph_cached():
    """Trả về dict {"graph": nx.Graph, "term_index": {term: [node_id,...]}}.

    [SỬA] Trước đây GRAPH_CKPT chỉ lưu thẳng đối tượng nx.Graph. Giờ lưu kèm
    `term_index` (inverted index term -> node_id) được build 1 LẦN DUY NHẤT
    khi build_graph(), để get_graph_scores() ở dưới không phải quét toàn bộ
    G.nodes() cho mỗi câu hỏi (xem giải thích chi tiết ở build_graph()).
    """
    global _GRAPH_CACHE
    if _GRAPH_CACHE is None and GRAPH_CKPT.exists():
        with open(GRAPH_CKPT, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict) and "graph" in loaded:
            _GRAPH_CACHE = loaded
        else:
            # Tương thích ngược nếu lỡ còn checkpoint kiểu cũ (chỉ có Graph
            # trần, chưa có term_index) -> vẫn dùng được nhưng graph retrieval
            # sẽ chậm hơn (fallback quét toàn bộ) cho tới khi build lại graph.
            logger.warning(" GRAPH_CKPT ở định dạng cũ (không có term_index) -> "
                            "nên xoá checkpoint và build_graph() lại để tăng tốc graph retrieval.")
            _GRAPH_CACHE = {"graph": loaded, "term_index": {}}
    return _GRAPH_CACHE


def _get_bm25_cached():
    global _BM25_CACHE
    if _BM25_CACHE is None and BM25_CKPT.exists():
        with open(BM25_CKPT, "rb") as f:
            _BM25_CACHE = pickle.load(f)
    return _BM25_CACHE


# ============ HÀM CHÍNH ============

def build_graph(legal=None):
    """
    Xây dựng Knowledge Graph với checkpoint và LLM entity extraction
    
    Args:
        legal: Legal LLM (dùng để trích xuất entity và relation)
    
    Returns:
        networkx.Graph: Knowledge Graph
    """
    # 1. Kiểm tra checkpoint
    if GRAPH_CKPT.exists():
        logger.info(f" Load Graph từ checkpoint: {GRAPH_CKPT}")
        with open(GRAPH_CKPT, "rb") as f:
            loaded = pickle.load(f)
            return loaded["graph"] if isinstance(loaded, dict) and "graph" in loaded else loaded
    
    logger.info(" Đang xây dựng Knowledge Graph tối ưu...")
    
    # 2. Lấy dữ liệu từ RAPTOR
    try:
        from member_a import get_raptor_nodes
        nodes = get_raptor_nodes()
    except Exception as e:
        logger.warning(f" Không thể import member_a: {e}")
        nodes = create_sample_nodes()
    
    if not nodes:
        logger.warning(" Không có nodes từ RAPTOR! Dùng sample nodes.")
        nodes = create_sample_nodes()
    
    logger.info(f" Có {len(nodes)} nodes từ RAPTOR")

    # [SỬA] Chỉ thật sự dùng LLM để trích entity/relation nếu số node đủ
    # nhỏ để chạy generate() cho từng node trong thời gian hợp lý. Xem giải
    # thích ở LLM_ENTITY_EXTRACTION_MAX_NODES phía trên.
    effective_legal = legal
    if legal is not None and len(nodes) > LLM_ENTITY_EXTRACTION_MAX_NODES:
        logger.warning(
            f" Có {len(nodes)} nodes > {LLM_ENTITY_EXTRACTION_MAX_NODES} -> "
            f"BỎ QUA LLM khi trích entity/relation (gọi LLM.generate() cho "
            f"từng node ở quy mô này sẽ mất hàng chục/hàng trăm giờ). Tự "
            f"động dùng fallback bằng regex (extract_entities_advanced/"
            f"extract_relations_from_text), vẫn nhanh và đủ dùng cho hầu "
            f"hết trường hợp."
        )
        effective_legal = None

    # 3. Xây dựng Graph với LLM (resume từ checkpoint tạm nếu có, để không
    #    phải chạy lại từ node đầu tiên nếu bị ngắt giữa chừng)
    resume_state = None
    if GRAPH_PARTIAL_CKPT.exists():
        logger.info(f" Tìm thấy checkpoint tạm, resume xây Graph: {GRAPH_PARTIAL_CKPT}")
        with open(GRAPH_PARTIAL_CKPT, "rb") as f:
            resume_state = pickle.load(f)

    G, term_index = build_knowledge_graph_optimized(nodes, effective_legal, resume_state=resume_state)
    
    # 4. Lưu checkpoint (kèm term_index để graph retrieval không phải quét
    #    toàn bộ node mỗi câu hỏi - xem _get_graph_cached()/get_graph_scores())
    with open(GRAPH_CKPT, "wb") as f:
        pickle.dump({"graph": G, "term_index": term_index}, f)
    logger.info(f" Graph checkpoint lưu tại {GRAPH_CKPT}")

    # Đã build xong graph hoàn chỉnh -> xóa checkpoint tạm, không cần nữa
    if GRAPH_PARTIAL_CKPT.exists():
        GRAPH_PARTIAL_CKPT.unlink()
    
    return G


def get_graph():
    """Trả về nx.Graph thuần (không kèm term_index), dùng cho code/test bên
    ngoài chỉ cần duyệt graph. build_graph() ở trên vẫn là API chính, trả về
    Graph để giữ tương thích ngược với chữ ký hàm cũ."""
    data = _get_graph_cached()
    return data["graph"] if data else None


def build_bm25():
    """
    Xây dựng BM25 index với checkpoint và từ điển đồng nghĩa
    """
    # 1. Kiểm tra checkpoint
    if BM25_CKPT.exists():
        logger.info(f" Load BM25 từ checkpoint: {BM25_CKPT}")
        with open(BM25_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info(" Đang xây dựng BM25 index tối ưu...")
    
    # 2. Lấy dữ liệu từ RAPTOR
    try:
        from member_a import get_raptor_nodes
        nodes = get_raptor_nodes()
    except Exception as e:
        logger.warning(f" Không thể import member_a: {e}")
        nodes = create_sample_nodes()
    
    if not nodes:
        logger.warning(" Không có nodes từ RAPTOR! Dùng sample nodes.")
        nodes = create_sample_nodes()
    
    # 3. Xây dựng BM25 với tokenization nâng cao
    corpus = [n["text"] for n in nodes]
    doc_ids = [n["id"] for n in nodes]
    tokenized_corpus = [tokenize_document_advanced(doc) for doc in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Lưu cả doc_ids để dùng sau
    bm25_data = {
        "bm25": bm25,
        "doc_ids": doc_ids,
        "corpus": corpus,
        "tokenized_corpus": tokenized_corpus
    }
    
    # 4. Lưu checkpoint
    with open(BM25_CKPT, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info(f" BM25 checkpoint lưu tại {BM25_CKPT}")
    
    return bm25_data


def hybrid_retrieve(query: str, top_k: int = 50) -> List[Dict]:
    """
    Hybrid Retrieval tối ưu: Dense + BM25 + Graph qua RRF
    
    Args:
        query: Câu hỏi truy vấn
        top_k: Số lượng kết quả trả về
    
    Returns:
        List[Dict]: [{"id": doc_id, "text": content, "score": score}, ...]
    """
    logger.info(f" Hybrid retrieval cho: {query[:50]}...")
    
    # 1. Lấy dữ liệu
    try:
        from member_a import get_raptor_nodes, get_vector_store
        nodes = get_raptor_nodes()
        vector_store = get_vector_store()
    except Exception as e:
        logger.warning(f" Lỗi import member_a: {e}")
        nodes = create_sample_nodes()
        vector_store = None
    
    if not nodes:
        return []
    
    node_dict = {n["id"]: n["text"] for n in nodes}
    
    # 2. BM25 retrieval (có boost cho số hiệu điều luật)
    bm25_scores = get_bm25_scores(query, top_k)
    
    # 3. Dense retrieval (FAISS)
    dense_scores = get_dense_scores(query, vector_store, top_k)
    
    # 4. Graph retrieval (có weighting)
    graph_scores = get_graph_scores(query, top_k)
    
    # 5. RRF (Reciprocal Rank Fusion) với weights khác nhau
    all_ids = list(set(bm25_scores.keys()) | set(dense_scores.keys()) | set(graph_scores.keys()))
    
    if not all_ids:
        logger.warning(" Không tìm thấy kết quả nào!")
        return []
    
    rrf_scores = {}
    k = RRF_K
    
    # Trọng số cho từng phương pháp
    weights = {
        "bm25": 1.0,
        "dense": 1.2,  # Dense retrieval có trọng số cao hơn
        "graph": 1.0
    }

    # [SỬA LỖI HIỆU NĂNG] Bản cũ gọi sorted(...).index(doc_id) BÊN TRONG
    # vòng lặp "for doc_id in all_ids" -> với mỗi doc_id lại sort lại toàn
    # bộ danh sách từ đầu (tốn O(m log m)) rồi mới .index() (tốn thêm O(m))
    # để tìm hạng của riêng nó -> tổng cộng O(m^2 log m) cho toàn bộ vòng
    # lặp thay vì chỉ cần O(m log m) nếu sort 1 lần. Giờ sort/đánh hạng 1
    # LẦN DUY NHẤT cho mỗi phương pháp trước khi vào vòng lặp.
    def _rank_dict(scores: Dict[str, float]) -> Dict[str, int]:
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return {doc_id: rank + 1 for rank, doc_id in enumerate(sorted_ids)}

    bm25_ranks = _rank_dict(bm25_scores)
    dense_ranks = _rank_dict(dense_scores)
    graph_ranks = _rank_dict(graph_scores)

    for doc_id in all_ids:
        score = 0

        if doc_id in bm25_ranks:
            score += weights["bm25"] * 1 / (k + bm25_ranks[doc_id])

        if doc_id in dense_ranks:
            score += weights["dense"] * 1 / (k + dense_ranks[doc_id])

        if doc_id in graph_ranks:
            score += weights["graph"] * 1 / (k + graph_ranks[doc_id])

        rrf_scores[doc_id] = score
    
    # QUAN TRỌNG: các "doc_id" ở trên thực chất là ID của NODE (chunk hoặc
    # cluster RAPTOR, VD "115374_a3_c22"), KHÔNG PHẢI document_id mà BTC
    # yêu cầu (VD "115374"). Phải tra ngược về document_id gốc bằng
    # get_node_doc_map() (chỉ node level 0 mới có map hợp lệ; cluster node
    # gộp nhiều văn bản nên bị loại vì không có 1 document_id duy nhất).
    try:
        from member_a import get_node_doc_map
        node_doc_map = get_node_doc_map()
    except Exception as e:
        logger.warning(f" Không lấy được node_doc_map: {e}")
        node_doc_map = {}

    # Sắp xếp theo điểm RRF (duyệt nhiều hơn top_k vì sẽ bị lọc bớt)
    sorted_node_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    result = []
    seen_doc_ids = set()
    for node_id in sorted_node_ids:
        real_doc_id = node_doc_map.get(node_id)
        if real_doc_id is None:
            # Node cluster (level 1/2) hoặc không xác định được văn bản gốc -> bỏ qua
            continue
        if real_doc_id in seen_doc_ids:
            # Đã có 1 chunk khác của cùng văn bản này trong candidate rồi,
            # không cần thêm bản trùng (giữ điểm/ text của chunk có rank cao nhất)
            continue
        seen_doc_ids.add(real_doc_id)
        result.append({
            "id": real_doc_id,
            "text": node_dict.get(node_id, ""),
            "score": rrf_scores.get(node_id, 0)
        })
        if len(result) >= top_k:
            break

    logger.info(f" Found {len(result)} candidates (đã quy về document_id, khử trùng lặp)")
    return result


# ============ TỐI ƯU BM25 ============

def get_bm25_scores(query: str, top_k: int) -> Dict[str, float]:
    """Lấy BM25 scores với tokenization nâng cao"""
    bm25_scores = {}
    
    bm25_data = _get_bm25_cached()
    if bm25_data is None:
        return bm25_scores
    
    bm25 = bm25_data["bm25"]
    doc_ids = bm25_data["doc_ids"]
    
    # Tokenization nâng cao cho query
    tokens = tokenize_document_advanced(query)
    
    # Boost cho số hiệu điều luật (ví dụ: "Điều 1", "Khoản 2")
    if re.search(r'điều\s+\d+', query.lower()):
        logger.debug("    Detected article number in query, boosting BM25...")
    
    scores = bm25.get_scores(tokens)
    
    top_indices = np.argsort(scores)[::-1][:top_k]
    for idx in top_indices:
        if idx < len(doc_ids):
            doc_id = doc_ids[idx]
            bm25_scores[doc_id] = float(scores[idx])
    
    return bm25_scores


def get_dense_scores(query: str, vector_store, top_k: int) -> Dict[str, float]:
    """Lấy dense retrieval scores từ FAISS"""
    dense_scores = {}
    
    if vector_store is None:
        return dense_scores
    
    try:
        embedder = _get_dense_embedder()
        q_emb = embedder.encode([query])
        
        index = vector_store["index"]
        node_ids = vector_store["node_ids"]
        
        distances, indices = index.search(
            np.array(q_emb).astype('float32'),
            min(top_k * 2, len(node_ids))
        )
        
        for i, idx in enumerate(indices[0]):
            if idx < len(node_ids):
                doc_id = node_ids[idx]
                # Chuyển distance thành similarity score (càng nhỏ càng gần)
                similarity = 1.0 / (1.0 + float(distances[0][i]))
                dense_scores[doc_id] = similarity
                
    except Exception as e:
        logger.warning(f" Dense retrieval lỗi: {e}")
    
    return dense_scores


def get_graph_scores(query: str, top_k: int) -> Dict[str, float]:
    """Lấy graph retrieval scores với multi-hop.

    [SỬA LỖI HIỆU NĂNG] Bản cũ duyệt `for node_id in G.nodes()` (TOÀN BỘ
    node trong graph) cho MỖI entity trích được từ câu hỏi -> với graph có
    hàng trăm nghìn node, một câu hỏi vài entity đã tốn hàng trăm nghìn *
    vài phép so sánh chuỗi, nhân với hàng trăm câu hỏi trong file inference
    thì cực chậm. Giờ dùng `term_index` (inverted index, build sẵn 1 lần khi
    build_graph()) để tra thẳng ra danh sách node ứng viên theo từng token,
    không cần quét toàn graph.
    """
    graph_scores = defaultdict(float)
    
    graph_data = _get_graph_cached()
    if graph_data is None:
        return dict(graph_scores)

    G = graph_data["graph"]
    term_index = graph_data.get("term_index", {})
    
    try:
        # Trích xuất entities từ query
        entities = extract_entities_advanced(query)
        
        for entity in entities:
            # extract_entities_advanced() trả về dict:
            # {'type': ..., 'value': ..., 'metadata': ...}
            entity_value = str(entity.get("value", ""))
            entity_lower = entity_value.lower()

            if not entity_lower:
                continue

            # Tra cứu qua term_index thay vì quét toàn bộ G.nodes(). Nếu
            # term_index rỗng (checkpoint cũ chưa có, xem _get_graph_cached)
            # thì fallback về quét toàn bộ như bản cũ để không mất kết quả.
            terms = tokenize_document_advanced(entity_value)
            if term_index:
                candidate_ids = set()
                for t in terms:
                    candidate_ids.update(term_index.get(t, ()))
            else:
                candidate_ids = G.nodes()

            for node_id in candidate_ids:
                if node_id not in G:
                    continue
                node_text = str(G.nodes[node_id].get("text", "")).lower()
                node_value = str(G.nodes[node_id].get("value", "")).lower()

                if entity_lower in node_text or entity_lower in node_value:
                    # Score cao hơn nếu match chính xác
                    if entity_lower == node_value:
                        graph_scores[node_id] += 2.0
                    else:
                        graph_scores[node_id] += 1.0

                    # Multi-hop: thêm score cho neighbors
                    for neighbor in G.neighbors(node_id):
                        graph_scores[neighbor] += 0.5
        
        # QUAN TRỌNG: chỉ giữ lại node kiểu "document" (= chunk level 0/1/2).
        # Node kiểu "entity" (VD "entity_Bộ Y tế_AGENCY") không map được về
        # 1 document_id cụ thể -> nếu lọt vào candidate list sẽ vừa vô ích
        # vừa làm reranker/submission nhận ID rác.
        graph_scores = {
            node_id: score for node_id, score in graph_scores.items()
            if G.nodes[node_id].get("type") == "document"
        }

        # Lấy top_k
        sorted_scores = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_scores)
        
    except Exception as e:
        logger.warning(f" Graph retrieval lỗi: {e}")
        return {}


# ============ TỐI ƯU KNOWLEDGE GRAPH ============

def _index_text(term_index: Dict[str, set], text: str, node_id: str):
    """Thêm node_id vào term_index cho mỗi token trong text (dùng chung
    tokenizer với BM25 để nhất quán). Đây là bước xây INVERTED INDEX 1 LẦN
    DUY NHẤT lúc build graph, để get_graph_scores() sau này tra cứu O(1) mỗi
    token thay vì phải quét qua TOÀN BỘ node trong graph cho mỗi câu hỏi
    (bug hiệu năng nghiêm trọng nhất của bản cũ ở quy mô lớn: n node lớn ->
    mỗi câu hỏi chậm tuyến tính theo n, hàng trăm câu hỏi thì nhân lên nữa).
    """
    for token in tokenize_document_advanced(text):
        term_index.setdefault(token, set()).add(node_id)


def build_knowledge_graph_optimized(
    nodes: List[Dict], legal=None, resume_state: Optional[Dict] = None
) -> Tuple[nx.Graph, Dict[str, List[str]]]:
    """
    Xây dựng Knowledge Graph từ nodes với LLM entity extraction (nếu có).

    Args:
        nodes: Danh sách RAPTOR nodes
        legal: Legal LLM (optional) - LƯU Ý: nơi gọi (build_graph()) đã tự
            động tắt LLM nếu số node quá lớn, xem LLM_ENTITY_EXTRACTION_MAX_NODES.
        resume_state: Nếu có (dict {"graph", "term_index", "all_entities",
            "all_relations", "processed_count"}), sẽ resume từ node thứ
            processed_count thay vì xây lại từ đầu.

    Returns:
        (G, term_index): term_index là {token: [node_id, ...]} dùng cho
        graph retrieval nhanh (xem get_graph_scores()).
    """
    if resume_state is not None:
        G = resume_state["graph"]
        all_entities = resume_state["all_entities"]
        all_relations = resume_state["all_relations"]
        start_idx = resume_state["processed_count"]
        # term_index có thể chưa có trong checkpoint cũ hơn -> tạo mới nếu thiếu.
        term_index = resume_state.get("term_index") or {}
        term_index = {k: set(v) for k, v in term_index.items()}
        logger.info(f" Resume xây Knowledge Graph từ node {start_idx}/{len(nodes)}...")
    else:
        logger.info(" Building Knowledge Graph tối ưu...")
        G = nx.Graph()
        all_entities = []
        all_relations = []
        term_index: Dict[str, set] = {}
        start_idx = 0

    ckpt_interval = _graph_ckpt_interval(len(nodes))

    # Thêm các document nodes
    for idx in tqdm(range(start_idx, len(nodes)), desc="Adding nodes",
                     initial=start_idx, total=len(nodes)):
        node = nodes[idx]
        node_id = node["id"]
        text = node["text"]
        level = node.get("level", 0)
        
        G.add_node(node_id, 
                   type="document",
                   text=text[:500],
                   level=level,
                   metadata=node.get("metadata", {}))
        _index_text(term_index, text, node_id)
        
        # Trích xuất entities bằng LLM nếu có (nơi gọi build_graph() đã tự
        # tắt legal khi số node quá lớn, nên nhánh legal is not None ở đây
        # chỉ chạy khi thật sự khả thi về thời gian)
        if legal is not None:
            entities = extract_entities_with_llm(text, legal)
            relations = extract_relations_with_llm(text, legal)
        else:
            entities = extract_entities_advanced(text)
            relations = extract_relations_from_text(text)
        
        all_entities.extend(entities)
        all_relations.extend(relations)
        
        # Thêm entity nodes và edges
        for entity in entities:
            entity_id = f"entity_{entity['value'][:30]}_{entity['type']}"
            if not G.has_node(entity_id):
                G.add_node(entity_id, 
                          type="entity",
                          entity_type=entity['type'],
                          value=entity['value'],
                          metadata=entity.get('metadata', {}))
                _index_text(term_index, entity['value'], entity_id)
            G.add_edge(node_id, entity_id, relation="contains", weight=1.0)

        # Lưu checkpoint TẠM định kỳ. Đây là vòng lặp tốn thời gian nhất và
        # dễ bị Kaggle ngắt giữa chừng nhất -> nếu không có checkpoint tạm,
        # mất là mất sạch. `ckpt_interval` GIÃN THEO tổng số node (thay vì
        # cố định 200) để tổng số lần ghi lại full-graph không phụ thuộc n.
        if (idx + 1) % ckpt_interval == 0:
            with open(GRAPH_PARTIAL_CKPT, "wb") as f:
                pickle.dump({
                    "graph": G,
                    "all_entities": all_entities,
                    "all_relations": all_relations,
                    "term_index": {k: list(v) for k, v in term_index.items()},
                    "processed_count": idx + 1,
                }, f)
            logger.info(f"    Checkpoint tạm Graph tại node {idx + 1}/{len(nodes)} (interval={ckpt_interval})")
    
    # Thêm relation edges
    for rel in all_relations:
        source_id = f"entity_{rel['source'][:30]}_{rel['source_type']}"
        target_id = f"entity_{rel['target'][:30]}_{rel['target_type']}"
        
        if G.has_node(source_id) and G.has_node(target_id):
            if G.has_edge(source_id, target_id):
                # Tăng weight nếu đã có edge
                G[source_id][target_id]['weight'] = G[source_id][target_id].get('weight', 1.0) + 0.5
            else:
                G.add_edge(source_id, target_id, 
                          relation=rel['type'],
                          weight=rel.get('weight', 1.0))
    
    # Kết nối các entity liên quan dựa trên đồng xuất hiện
    connect_related_entities(G, nodes)
    
    logger.info(f" Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
                f"term_index: {len(term_index)} terms")
    term_index_out = {k: list(v) for k, v in term_index.items()}
    return G, term_index_out


def extract_entities_with_llm(text: str, legal_llm) -> List[Dict]:
    """Trích xuất entities bằng Legal LLM"""
    try:
        prompt = f"""Trích xuất các thực thể pháp lý từ văn bản sau. 
Mỗi thực thể bao gồm loại (ARTICLE, CLAUSE, AGENCY, LEGAL_TERM) và giá trị.

Văn bản: {text[:500]}

Thực thể (định dạng: LOẠI: GIÁ_TRỊ):"""
        
        response = legal_llm.generate(prompt, max_length=200)
        
        entities = []
        for line in response.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    etype = parts[0].strip().upper()
                    evalue = parts[1].strip()
                    if evalue and len(evalue) > 2:
                        entities.append({
                            'type': etype,
                            'value': evalue[:100],
                            'metadata': {'source': 'llm'}
                        })
        
        # Fallback: dùng regex nếu LLM không trả về
        if not entities:
            entities = extract_entities_advanced(text)
        
        return entities
        
    except Exception as e:
        logger.warning(f" LLM entity extraction failed: {e}")
        return extract_entities_advanced(text)


def extract_relations_with_llm(text: str, legal_llm) -> List[Dict]:
    """Trích xuất relations bằng Legal LLM"""
    try:
        prompt = f"""Trích xuất các mối quan hệ pháp lý từ văn bản sau.
Mỗi quan hệ có dạng: (ENTITY1) - [RELATION_TYPE] -> (ENTITY2)

Văn bản: {text[:500]}

Quan hệ:"""
        
        response = legal_llm.generate(prompt, max_length=200)
        
        relations = []
        # Parse response đơn giản
        for line in response.split('\n'):
            if '->' in line or '—' in line:
                # Parse đơn giản
                parts = re.split(r'[-—>]+', line)
                if len(parts) >= 3:
                    relations.append({
                        'source': parts[0].strip(),
                        'source_type': 'ENTITY',
                        'target': parts[-1].strip(),
                        'target_type': 'ENTITY',
                        'type': 'RELATED_TO',
                        'weight': 1.0
                    })
        
        return relations
        
    except Exception as e:
        logger.warning(f" LLM relation extraction failed: {e}")
        return extract_relations_from_text(text)


def extract_entities_advanced(text: str) -> List[Dict]:
    """Trích xuất entities với regex nâng cao"""
    entities = []
    
    # 1. Điều luật
    article_pattern = re.compile(r'(Điều\s+(\d+[a-zA-Z]?)[\.\:\s]+)', re.IGNORECASE)
    for match in article_pattern.finditer(text):
        entities.append({
            'type': 'ARTICLE',
            'value': match.group(1).strip(),
            'metadata': {'number': match.group(2)}
        })
    
    # 2. Khoản
    clause_pattern = re.compile(r'(Khoản\s+(\d+[a-zA-Z]?)[\.\:\s]+)', re.IGNORECASE)
    for match in clause_pattern.finditer(text):
        entities.append({
            'type': 'CLAUSE',
            'value': match.group(1).strip(),
            'metadata': {'number': match.group(2)}
        })
    
    # 3. Điểm
    point_pattern = re.compile(r'(Điểm\s+([a-zâêôơưđ])[\.\:\s]+)', re.IGNORECASE)
    for match in point_pattern.finditer(text):
        entities.append({
            'type': 'POINT',
            'value': match.group(1).strip(),
            'metadata': {'letter': match.group(2)}
        })
    
    # 4. Cơ quan ban hành
    agency_pattern = re.compile(r'(Bộ|Sở|Ủy ban|Chính phủ|Quốc hội|Tòa án|Viện kiểm sát)\s+([A-ZÀ-Ỷ][a-zà-ỹ]+(?:\s+[A-ZÀ-Ỷ][a-zà-ỹ]+)*)')
    for match in agency_pattern.finditer(text):
        entities.append({
            'type': 'AGENCY',
            'value': f"{match.group(1)} {match.group(2)}",
            'metadata': {'agency_type': match.group(1)}
        })
    
    # 5. Từ khóa pháp lý
    legal_terms = [
        ("hợp đồng", "CONTRACT"),
        ("luật", "LAW"),
        ("nghị định", "DECREE"),
        ("thông tư", "CIRCULAR"),
        ("quyết định", "DECISION"),
        ("pháp lệnh", "ORDINANCE"),
        ("bộ luật", "CODE"),
        ("quyền", "RIGHT"),
        ("nghĩa vụ", "OBLIGATION"),
        ("trách nhiệm", "RESPONSIBILITY"),
        ("thanh tra", "INSPECTION"),
        ("kiểm toán", "AUDIT"),
        ("bảo hiểm", "INSURANCE"),
        ("đầu tư", "INVESTMENT")
    ]
    
    for term, term_type in legal_terms:
        if term in text.lower():
            entities.append({
                'type': 'LEGAL_TERM',
                'value': term,
                'metadata': {'term_type': term_type}
            })
    
    return entities


def extract_relations_from_text(text: str) -> List[Dict]:
    """Trích xuất relations bằng pattern"""
    relations = []
    
    relation_patterns = [
        (r'theo\s+quy\s+định\s+tại', 'REFERS_TO'),
        (r'căn\s+cứ\s+vào', 'BASED_ON'),
        (r'sửa\s+đổi', 'AMENDS'),
        (r'bổ\s+sung', 'SUPPLEMENTS'),
        (r'thay\s+thế', 'REPLACES'),
        (r'hướng\s+dẫn', 'GUIDES'),
        (r'quy\s+định', 'REGULATES'),
        (r'áp\s+dụng', 'APPLIES'),
    ]
    
    for pattern, rel_type in relation_patterns:
        if re.search(pattern, text.lower()):
            # Tìm entities gần pattern
            entities = extract_entities_advanced(text)
            if len(entities) >= 2:
                for i in range(len(entities)):
                    for j in range(i+1, len(entities)):
                        relations.append({
                            'source': entities[i]['value'],
                            'source_type': entities[i]['type'],
                            'target': entities[j]['value'],
                            'target_type': entities[j]['type'],
                            'type': rel_type,
                            'weight': 1.0
                        })
    
    return relations


def connect_related_entities(G: nx.Graph, nodes: List[Dict]):
    """Kết nối các entity liên quan dựa trên đồng xuất hiện"""
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "entity"]
    entity_terms = {}
    
    # Xây dựng term index cho entities
    for entity_id in entity_nodes:
        value = str(G.nodes[entity_id].get("value", "")).lower()
        for term in value.split():
            if len(term) > 2:
                if term not in entity_terms:
                    entity_terms[term] = []
                entity_terms[term].append(entity_id)
    
    # Kết nối các entity có chung term
    for term, entity_ids in entity_terms.items():
        if len(entity_ids) > 1:
            for i in range(len(entity_ids)):
                for j in range(i+1, len(entity_ids)):
                    if entity_ids[i] != entity_ids[j]:
                        G.add_edge(entity_ids[i], entity_ids[j], 
                                  relation="co_occur",
                                  weight=0.5,
                                  common_term=term)


# ============ TỐI ƯU TOKENIZATION ============

def tokenize_document_advanced(text: str) -> List[str]:
    """
    Tokenization nâng cao cho BM25:
    - Giữ nguyên số hiệu điều luật
    - Xử lý tiếng Việt có dấu
    - Loại bỏ stopwords
    """
    # Normalize text
    text = text.lower()
    
    # Thêm boost cho số hiệu điều luật
    text = re.sub(r'điều\s+(\d+)', r'điều_\1', text)
    text = re.sub(r'khoản\s+(\d+)', r'khoản_\1', text)
    text = re.sub(r'điểm\s+([a-z])', r'điểm_\1', text)
    
    # Tokenize
    tokens = re.findall(r'[a-zA-Z0-9À-ỹ_]+', text)
    
    # Stopwords tiếng Việt đơn giản
    stopwords = {'và', 'của', 'cho', 'với', 'trong', 'các', 'có', 'được', 'là', 
                 'tại', 'theo', 'từ', 'khi', 'để', 'mà', 'này', 'đó', 'như'}
    
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    
    return tokens


def create_sample_nodes() -> List[Dict]:
    """Tạo dữ liệu mẫu để test"""
    return [
        {"id": "doc_001", "text": "Điều 1: Quy định chung. Khoản 1: Phạm vi điều chỉnh của Luật này..."},
        {"id": "doc_002", "text": "Điều 2: Quyền và nghĩa vụ. Khoản 1: Quyền của người lao động..."},
        {"id": "doc_003", "text": "Điều 3: Trách nhiệm. Khoản 1: Trách nhiệm của người sử dụng lao động..."}
    ]


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    print("Testing member_b.py (optimized version)...")
    
    # Test BM25
    bm25_data = build_bm25()
    print(f"BM25 docs: {len(bm25_data.get('doc_ids', []))}")
    
    # Test Graph
    graph = build_graph()
    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    
    # Test Hybrid
    results = hybrid_retrieve("Quyền của người lao động", top_k=5)
    print(f"Hybrid results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r['id']}: score={r['score']:.4f}")
