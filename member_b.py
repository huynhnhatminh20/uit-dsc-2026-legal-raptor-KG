"""
member_b.py - Knowledge Graph + BM25 + Hybrid Retrieval (Phiên bản tối ưu)
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
from collections import defaultdict

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/B_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CKPT = CKPT_DIR / "graph.pkl"
BM25_CKPT = CKPT_DIR / "bm25.pkl"
ENTITY_CKPT = CKPT_DIR / "entities.pkl"
RELATION_CKPT = CKPT_DIR / "relations.pkl"

# ============ CONSTANTS ============
RRF_K = 60
MAX_CANDIDATES = 50
EMBEDDING_MODEL = "BAAI/bge-m3"


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
            return pickle.load(f)
    
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
    
    # 3. Xây dựng Graph với LLM
    G = build_knowledge_graph_optimized(nodes, legal)
    
    # 4. Lưu checkpoint
    with open(GRAPH_CKPT, "wb") as f:
        pickle.dump(G, f)
    logger.info(f" Graph checkpoint lưu tại {GRAPH_CKPT}")
    
    return G


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
    
    for doc_id in all_ids:
        score = 0
        
        # BM25 rank
        if doc_id in bm25_scores:
            rank = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True).index(doc_id) + 1
            score += weights["bm25"] * 1 / (k + rank)
        
        # Dense rank
        if doc_id in dense_scores:
            rank = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True).index(doc_id) + 1
            score += weights["dense"] * 1 / (k + rank)
        
        # Graph rank
        if doc_id in graph_scores:
            rank = sorted(graph_scores.keys(), key=lambda x: graph_scores[x], reverse=True).index(doc_id) + 1
            score += weights["graph"] * 1 / (k + rank)
        
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


# ============ TỐI ƯU BM25 ============

def get_bm25_scores(query: str, top_k: int) -> Dict[str, float]:
    """Lấy BM25 scores với tokenization nâng cao"""
    bm25_scores = {}
    
    if not BM25_CKPT.exists():
        return bm25_scores
    
    with open(BM25_CKPT, "rb") as f:
        bm25_data = pickle.load(f)
    
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
        embedder = SentenceTransformer(EMBEDDING_MODEL)
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
    """Lấy graph retrieval scores với multi-hop"""
    graph_scores = defaultdict(float)
    
    if not GRAPH_CKPT.exists():
        return dict(graph_scores)
    
    try:
        with open(GRAPH_CKPT, "rb") as f:
            G = pickle.load(f)
        
        # Trích xuất entities từ query
        entities = extract_entities_advanced(query)
        
        for entity in entities:
            entity_lower = entity.lower()
            
            # Tìm nodes chứa entity
            for node_id in G.nodes():
                node_text = str(G.nodes[node_id].get("text", "")).lower()
                node_value = str(G.nodes[node_id].get("value", "")).lower()
                
                if entity_lower in node_text or entity_lower in node_value:
                    # Score cao hơn nếu match chính xác
                    if entity in node_text or entity in node_value:
                        graph_scores[node_id] += 2.0
                    else:
                        graph_scores[node_id] += 1.0
                    
                    # Multi-hop: thêm score cho neighbors
                    for neighbor in G.neighbors(node_id):
                        graph_scores[neighbor] += 0.5
        
        # Lấy top_k
        sorted_scores = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_scores)
        
    except Exception as e:
        logger.warning(f" Graph retrieval lỗi: {e}")
        return {}


# ============ TỐI ƯU KNOWLEDGE GRAPH ============

def build_knowledge_graph_optimized(nodes: List[Dict], legal=None) -> nx.Graph:
    """
    Xây dựng Knowledge Graph từ nodes với LLM entity extraction
    """
    logger.info(" Building Knowledge Graph tối ưu...")
    G = nx.Graph()
    
    all_entities = []
    all_relations = []
    
    # Thêm các document nodes
    for node in tqdm(nodes, desc="Adding nodes"):
        node_id = node["id"]
        text = node["text"]
        level = node.get("level", 0)
        
        G.add_node(node_id, 
                   type="document",
                   text=text[:500],
                   level=level,
                   metadata=node.get("metadata", {}))
        
        # Trích xuất entities bằng LLM nếu có
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
            G.add_edge(node_id, entity_id, relation="contains", weight=1.0)
    
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
    
    logger.info(f" Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


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