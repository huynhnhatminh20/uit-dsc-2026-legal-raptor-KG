"""
member_b.py - Knowledge Graph + BM25 + Hybrid Retrieval
"""

import os
import json
import pickle
import pathlib
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import re
import logging

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/B_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CKPT = CKPT_DIR / "graph.pkl"
BM25_CKPT = CKPT_DIR / "bm25.pkl"
ENTITY_CKPT = CKPT_DIR / "entities.pkl"

# ============ HÀM CHÍNH ============

def build_graph(legal=None):
    """
    Xây dựng Knowledge Graph với checkpoint
    
    Args:
        legal: Legal LLM (dùng để trích xuất entity nếu có)
    
    Returns:
        networkx.Graph: Knowledge Graph
    """
    # 1. Kiểm tra checkpoint
    if GRAPH_CKPT.exists():
        logger.info(f" Load Graph từ checkpoint: {GRAPH_CKPT}")
        with open(GRAPH_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("⚙️ Đang xây dựng Knowledge Graph...")
    
    # 2. Lấy dữ liệu từ RAPTOR
    try:
        from member_a import get_raptor_nodes
        nodes = get_raptor_nodes()
    except Exception as e:
        logger.warning(f" Không thể import member_a: {e}")
        # Tạo dữ liệu mẫu nếu không có RAPTOR
        nodes = create_sample_nodes()
    
    if not nodes:
        logger.warning(" Không có nodes từ RAPTOR! Dùng sample nodes.")
        nodes = create_sample_nodes()
    
    logger.info(f"📊 Có {len(nodes)} nodes từ RAPTOR")
    
    # 3. Xây dựng Graph
    G = build_knowledge_graph(nodes, legal)
    
    # 4. Lưu checkpoint
    with open(GRAPH_CKPT, "wb") as f:
        pickle.dump(G, f)
    logger.info(f" Graph checkpoint lưu tại {GRAPH_CKPT}")
    
    return G


def build_bm25():
    """
    Xây dựng BM25 index với checkpoint
    
    Returns:
        BM25Okapi: BM25 index
    """
    # 1. Kiểm tra checkpoint
    if BM25_CKPT.exists():
        logger.info(f" Load BM25 từ checkpoint: {BM25_CKPT}")
        with open(BM25_CKPT, "rb") as f:
            data = pickle.load(f)
            return data
    
    logger.info(" Đang xây dựng BM25 index...")
    
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
    
    # 3. Xây dựng BM25
    corpus = [n["text"] for n in nodes]
    doc_ids = [n["id"] for n in nodes]
    tokenized_corpus = [tokenize_document(doc) for doc in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Lưu cả doc_ids để dùng sau
    bm25_data = {
        "bm25": bm25,
        "doc_ids": doc_ids,
        "corpus": corpus
    }
    
    # 4. Lưu checkpoint
    with open(BM25_CKPT, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info(f" BM25 checkpoint lưu tại {BM25_CKPT}")
    
    return bm25_data


def hybrid_retrieve(query: str, top_k: int = 50) -> List[Dict]:
    """
    Hybrid Retrieval: Dense + BM25 + Graph qua RRF
    
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
    
    # 2. BM25 retrieval
    bm25_scores = {}
    if BM25_CKPT.exists():
        with open(BM25_CKPT, "rb") as f:
            bm25_data = pickle.load(f)
        
        bm25 = bm25_data["bm25"]
        doc_ids = bm25_data["doc_ids"]
        tokenized_query = tokenize_document(query)
        scores = bm25.get_scores(tokenized_query)
        
        # Lấy top_k
        top_indices = np.argsort(scores)[::-1][:top_k]
        for idx in top_indices:
            if idx < len(doc_ids):
                doc_id = doc_ids[idx]
                bm25_scores[doc_id] = float(scores[idx])
    
    # 3. Dense retrieval (FAISS)
    dense_scores = {}
    if vector_store is not None:
        try:
            embedder = SentenceTransformer("BAAI/bge-m3")
            q_emb = embedder.encode([query])
            
            index = vector_store["index"]
            node_ids = vector_store["node_ids"]
            
            distances, indices = index.search(
                np.array(q_emb).astype('float32'),
                min(top_k, len(node_ids))
            )
            
            for i, idx in enumerate(indices[0]):
                if idx < len(node_ids):
                    doc_id = node_ids[idx]
                    dense_scores[doc_id] = float(distances[0][i])
        except Exception as e:
            logger.warning(f" Dense retrieval lỗi: {e}")
    
    # 4. Graph retrieval
    graph_scores = {}
    if GRAPH_CKPT.exists():
        try:
            with open(GRAPH_CKPT, "rb") as f:
                G = pickle.load(f)
            
            # Tìm entities trong query
            entities = extract_entities_from_text(query)
            for entity in entities:
                # Tìm nodes liên quan đến entity
                for node_id in G.nodes():
                    if entity.lower() in node_id.lower() or entity.lower() in str(G.nodes[node_id].get("text", "")).lower():
                        graph_scores[node_id] = graph_scores.get(node_id, 0) + 1
        except Exception as e:
            logger.warning(f" Graph retrieval lỗi: {e}")
    
    # 5. RRF (Reciprocal Rank Fusion)
    all_ids = list(set(bm25_scores.keys()) | set(dense_scores.keys()) | set(graph_scores.keys()))
    
    if not all_ids:
        logger.warning(" Không tìm thấy kết quả nào!")
        return []
    
    rrf_scores = {}
    k = 60  # RRF constant
    
    for doc_id in all_ids:
        score = 0
        
        # BM25 rank
        if doc_id in bm25_scores:
            rank = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True).index(doc_id) + 1
            score += 1 / (k + rank)
        
        # Dense rank
        if doc_id in dense_scores:
            rank = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True).index(doc_id) + 1
            score += 1 / (k + rank)
        
        # Graph rank
        if doc_id in graph_scores:
            rank = sorted(graph_scores.keys(), key=lambda x: graph_scores[x], reverse=True).index(doc_id) + 1
            score += 1 / (k + rank)
        
        rrf_scores[doc_id] = score
    
    # Sắp xếp và trả về top_k
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    result = []
    for doc_id in sorted_ids:
        result.append({
            "id": doc_id,
            "text": node_dict.get(doc_id, ""),
            "score": rrf_scores.get(doc_id, 0)
        })
    
    logger.info(f" Found {len(result)} candidates")
    return result


# ============ HÀM PHỤ TRỢ ============

def create_sample_nodes() -> List[Dict]:
    """Tạo dữ liệu mẫu để test"""
    return [
        {"id": "doc_001", "text": "Điều 1: Quy định chung. Khoản 1: Phạm vi điều chỉnh..."},
        {"id": "doc_002", "text": "Điều 2: Quyền và nghĩa vụ. Khoản 1: Quyền của người lao động..."},
        {"id": "doc_003", "text": "Điều 3: Trách nhiệm. Khoản 1: Trách nhiệm của người sử dụng lao động..."}
    ]


def tokenize_document(text: str) -> List[str]:
    """Tokenize văn bản cho BM25"""
    # Chuyển sang lowercase và tách từ
    import re
    tokens = re.findall(r'[a-zA-Z0-9À-ỹ]+', text.lower())
    return tokens


def build_knowledge_graph(nodes: List[Dict], legal=None) -> nx.Graph:
    """
    Xây dựng Knowledge Graph từ nodes
    
    Args:
        nodes: List các node từ RAPTOR
        legal: Legal LLM (dùng để trích xuất entity nếu có)
    
    Returns:
        networkx.Graph
    """
    logger.info(" Building Knowledge Graph...")
    G = nx.Graph()
    
    # Thêm các document nodes
    for node in tqdm(nodes, desc="Adding nodes"):
        node_id = node["id"]
        G.add_node(node_id, 
                   type="document",
                   text=node["text"][:200],
                   level=node.get("level", 0))
        
        # Trích xuất entities từ text
        entities = extract_entities_from_text(node["text"])
        
        for entity in entities:
            entity_id = f"entity_{entity[:30]}"
            if not G.has_node(entity_id):
                G.add_node(entity_id, type="entity", value=entity)
            G.add_edge(node_id, entity_id, relation="contains")
    
    # Kết nối các entity liên quan
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "entity"]
    for i, e1 in enumerate(entity_nodes):
        for e2 in entity_nodes[i+1:]:
            # Kết nối nếu entity có từ chung
            e1_val = G.nodes[e1].get("value", "")
            e2_val = G.nodes[e2].get("value", "")
            if len(set(e1_val.split()) & set(e2_val.split())) > 0:
                G.add_edge(e1, e2, relation="related")
    
    logger.info(f" Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def extract_entities_from_text(text: str) -> List[str]:
    """Trích xuất entities từ văn bản"""
    entities = []
    
    # Pattern cho Điều
    article_pattern = re.compile(r'(Điều\s+\d+[a-zA-Z]?\.?\s*)', re.IGNORECASE)
    articles = article_pattern.findall(text)
    for art in articles:
        entities.append(art.strip())
    
    # Pattern cho Khoản
    clause_pattern = re.compile(r'(Khoản\s+\d+[a-zA-Z]?\.?\s*)', re.IGNORECASE)
    clauses = clause_pattern.findall(text)
    for cl in clauses:
        entities.append(cl.strip())
    
    # Pattern cho Điểm
    point_pattern = re.compile(r'(Điểm\s+[a-zâêôơưđ]\.?\s*)', re.IGNORECASE)
    points = point_pattern.findall(text)
    for p in points:
        entities.append(p.strip())
    
    # Từ khóa pháp lý
    legal_terms = ["hợp đồng", "luật", "nghị định", "thông tư", "quyết định", 
                   "pháp lệnh", "bộ luật", "quyền", "nghĩa vụ", "trách nhiệm",
                   "thanh tra", "kiểm toán", "bảo hiểm", "đầu tư"]
    for term in legal_terms:
        if term in text.lower():
            entities.append(term)
    
    return list(set(entities))  # Loại bỏ duplicates


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    print("Testing member_b.py...")
    
    # Test BM25
    bm25_data = build_bm25()
    print(f"BM25 docs: {len(bm25_data.get('doc_ids', []))}")
    
    # Test Graph
    graph = build_graph()
    print(f"Graph nodes: {graph.number_of_nodes()}")
    
    # Test Hybrid
    results = hybrid_retrieve("Quyền của người lao động", top_k=5)
    print(f"Hybrid results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r['id']}: score={r['score']:.4f}")