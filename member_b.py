"""
member_b.py - Knowledge Graph + BM25 + Hybrid Retrieval (Phiên bản hoàn chỉnh)
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CKPT_DIR = pathlib.Path("/kaggle/working/B_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_CKPT = CKPT_DIR / "graph.pkl"
BM25_CKPT = CKPT_DIR / "bm25.pkl"
ENTITY_CKPT = CKPT_DIR / "entities.pkl"
RELATION_CKPT = CKPT_DIR / "relations.pkl"

RRF_K = 60
MAX_CANDIDATES = 50
EMBEDDING_MODEL = "BAAI/bge-m3"


# ============ HÀM PHỤ TRỢ ============

def create_sample_nodes() -> List[Dict]:
    return [
        {"id": "doc_001", "doc_id": "doc_001", "text": "Dieu 1: Quy dinh chung..."},
        {"id": "doc_002", "doc_id": "doc_002", "text": "Dieu 2: Quyen va nghia vu..."},
        {"id": "doc_003", "doc_id": "doc_003", "text": "Dieu 3: Trach nhiem..."}
    ]


def tokenize_document_advanced(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r'dieu\s+(\d+)', r'dieu_\1', text)
    text = re.sub(r'khoan\s+(\d+)', r'khoan_\1', text)
    text = re.sub(r'diem\s+([a-z])', r'diem_\1', text)
    tokens = re.findall(r'[a-zA-Z0-9_]+', text)
    stopwords = {'va', 'cua', 'cho', 'voi', 'trong', 'cac', 'co', 'duoc', 'la',
                 'tai', 'theo', 'tu', 'khi', 'de', 'ma', 'nay', 'do', 'nhu'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]


def extract_entities_advanced(text: str) -> List[Dict]:
    entities = []
    
    article_pattern = re.compile(r'(Dieu\s+(\d+)[\.\:\s]+)', re.IGNORECASE)
    for match in article_pattern.finditer(text):
        entities.append({'type': 'ARTICLE', 'value': match.group(1).strip()})
    
    clause_pattern = re.compile(r'(Khoan\s+(\d+)[\.\:\s]+)', re.IGNORECASE)
    for match in clause_pattern.finditer(text):
        entities.append({'type': 'CLAUSE', 'value': match.group(1).strip()})
    
    point_pattern = re.compile(r'(Diem\s+([a-z])[\.\:\s]+)', re.IGNORECASE)
    for match in point_pattern.finditer(text):
        entities.append({'type': 'POINT', 'value': match.group(1).strip()})
    
    legal_terms = [
        ("hop dong", "CONTRACT"), ("luat", "LAW"), ("nghi dinh", "DECREE"),
        ("thong tu", "CIRCULAR"), ("quyet dinh", "DECISION"),
        ("quyen", "RIGHT"), ("nghia vu", "OBLIGATION"),
        ("trach nhiem", "RESPONSIBILITY"), ("thanh tra", "INSPECTION"),
        ("kiem toan", "AUDIT"), ("bao hiem", "INSURANCE"), ("dau tu", "INVESTMENT")
    ]
    for term, term_type in legal_terms:
        if term in text.lower():
            entities.append({'type': 'LEGAL_TERM', 'value': term})
    
    return list({e['value']: e for e in entities}.values())


def extract_relations_from_text(text: str) -> List[Dict]:
    relations = []
    relation_patterns = [
        (r'theo\s+quy\s+dinh', 'REFERS_TO'),
        (r'can\s+cu\s+vao', 'BASED_ON'),
        (r'sua\s+doi', 'AMENDS'),
        (r'bo\s+sung', 'SUPPLEMENTS'),
        (r'thay\s+the', 'REPLACES'),
        (r'huong\s+dan', 'GUIDES'),
        (r'quy\s+dinh', 'REGULATES'),
        (r'ap\s+dung', 'APPLIES'),
    ]
    for pattern, rel_type in relation_patterns:
        if re.search(pattern, text.lower()):
            entities = extract_entities_advanced(text)
            if len(entities) >= 2:
                relations.append({
                    'source': entities[0]['value'],
                    'source_type': entities[0]['type'],
                    'target': entities[1]['value'],
                    'target_type': entities[1]['type'],
                    'type': rel_type,
                    'weight': 1.0
                })
    return relations


def extract_entities_with_llm(text: str, legal_llm) -> List[Dict]:
    try:
        prompt = f"Trich xuat thuc the phap ly tu: {text[:500]}"
        response = legal_llm.generate(prompt, max_length=100)
        return extract_entities_advanced(text)
    except:
        return extract_entities_advanced(text)


def extract_relations_with_llm(text: str, legal_llm) -> List[Dict]:
    try:
        prompt = f"Trich xuat quan he phap ly tu: {text[:500]}"
        response = legal_llm.generate(prompt, max_length=100)
        return extract_relations_from_text(text)
    except:
        return extract_relations_from_text(text)


# ============ HÀM CHÍNH ============

def build_graph(legal=None):
    if GRAPH_CKPT.exists():
        logger.info(f"Load Graph tu checkpoint: {GRAPH_CKPT}")
        with open(GRAPH_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("Dang xay dung Knowledge Graph toi uu...")
    
    try:
        from member_a import get_raptor_nodes
        nodes = get_raptor_nodes()
    except Exception as e:
        logger.warning(f"Khong the import member_a: {e}")
        nodes = create_sample_nodes()
    
    if not nodes:
        logger.warning("Khong co nodes tu RAPTOR! Dung sample nodes.")
        nodes = create_sample_nodes()
    
    logger.info(f"Co {len(nodes)} nodes tu RAPTOR")
    
    G = build_knowledge_graph_optimized(nodes, legal)
    
    with open(GRAPH_CKPT, "wb") as f:
        pickle.dump(G, f)
    logger.info(f"Graph checkpoint luu tai {GRAPH_CKPT}")
    
    return G


def build_bm25():
    if BM25_CKPT.exists():
        logger.info(f"Load BM25 tu checkpoint: {BM25_CKPT}")
        with open(BM25_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("Dang xay dung BM25 index toi uu...")
    
    try:
        from member_a import get_raptor_nodes
        nodes = get_raptor_nodes()
    except Exception as e:
        logger.warning(f"Khong the import member_a: {e}")
        nodes = create_sample_nodes()
    
    if not nodes:
        logger.warning("Khong co nodes tu RAPTOR! Dung sample nodes.")
        nodes = create_sample_nodes()
    
    corpus = [n["text"] for n in nodes]
    doc_ids = [n.get("doc_id", n["id"]) for n in nodes]
    tokenized_corpus = [tokenize_document_advanced(doc) for doc in corpus]
    
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_data = {
        "bm25": bm25,
        "doc_ids": doc_ids,
        "corpus": corpus,
        "tokenized_corpus": tokenized_corpus
    }
    
    with open(BM25_CKPT, "wb") as f:
        pickle.dump(bm25_data, f)
    logger.info(f"BM25 checkpoint luu tai {BM25_CKPT}")
    
    return bm25_data


def hybrid_retrieve(query: str, top_k: int = 50) -> List[Dict]:
    logger.info(f"Hybrid retrieval cho: {query[:50]}...")
    
    try:
        from member_a import get_raptor_nodes, get_vector_store
        nodes = get_raptor_nodes()
        vector_store = get_vector_store()
    except Exception as e:
        logger.warning(f"Loi import member_a: {e}")
        nodes = create_sample_nodes()
        vector_store = None
    
    if not nodes:
        return []
    
    node_dict = {}
    for n in nodes:
        doc_id = n.get('doc_id', n.get('metadata', {}).get('doc_id', n['id'].split('_')[0]))
        node_dict[n['id']] = {
            'text': n['text'],
            'doc_id': doc_id
        }
    
    bm25_scores = get_bm25_scores(query, top_k)
    dense_scores = get_dense_scores(query, vector_store, top_k)
    graph_scores = get_graph_scores(query, top_k)
    
    all_ids = list(set(bm25_scores.keys()) | set(dense_scores.keys()) | set(graph_scores.keys()))
    
    if not all_ids:
        logger.warning("Khong tim thay ket qua nao!")
        return []
    
    rrf_scores = {}
    k = RRF_K
    
    weights = {
        "bm25": 1.0,
        "dense": 1.2,
        "graph": 1.0
    }
    
    for doc_id in all_ids:
        score = 0
        
        if doc_id in bm25_scores:
            rank = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True).index(doc_id) + 1
            score += weights["bm25"] * 1 / (k + rank)
        
        if doc_id in dense_scores:
            rank = sorted(dense_scores.keys(), key=lambda x: dense_scores[x], reverse=True).index(doc_id) + 1
            score += weights["dense"] * 1 / (k + rank)
        
        if doc_id in graph_scores:
            rank = sorted(graph_scores.keys(), key=lambda x: graph_scores[x], reverse=True).index(doc_id) + 1
            score += weights["graph"] * 1 / (k + rank)
        
        rrf_scores[doc_id] = score
    
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
    
    result = []
    for chunk_id in sorted_ids:
        doc_info = node_dict.get(chunk_id, {})
        doc_id = doc_info.get('doc_id', chunk_id.split('_')[0])
        
        # Kiem tra trung lap doc_id
        existing_ids = [r['id'] for r in result]
        if doc_id not in existing_ids:
            result.append({
                "id": doc_id,
                "doc_id": doc_id,
                "text": doc_info.get('text', ''),
                "passage": doc_info.get('text', ''),
                "score": rrf_scores.get(chunk_id, 0)
            })
    
    logger.info(f"Tim thay {len(result)} candidates")
    return result


# ============ TỐI ƯU BM25 ============

def get_bm25_scores(query: str, top_k: int) -> Dict[str, float]:
    bm25_scores = {}
    if not BM25_CKPT.exists():
        return bm25_scores
    with open(BM25_CKPT, "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["bm25"]
    doc_ids = bm25_data["doc_ids"]
    tokens = tokenize_document_advanced(query)
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    for idx in top_indices:
        if idx < len(doc_ids):
            doc_id = doc_ids[idx]
            bm25_scores[doc_id] = float(scores[idx])
    return bm25_scores


def get_dense_scores(query: str, vector_store, top_k: int) -> Dict[str, float]:
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
                similarity = 1.0 / (1.0 + float(distances[0][i]))
                dense_scores[doc_id] = similarity
    except Exception as e:
        logger.warning(f"Dense retrieval loi: {e}")
    return dense_scores


def get_graph_scores(query: str, top_k: int) -> Dict[str, float]:
    graph_scores = defaultdict(float)
    if not GRAPH_CKPT.exists():
        return dict(graph_scores)
    try:
        with open(GRAPH_CKPT, "rb") as f:
            G = pickle.load(f)
        entities = extract_entities_advanced(query)
        for entity in entities:
            entity_lower = entity.lower()
            for node_id in G.nodes():
                node_text = str(G.nodes[node_id].get("text", "")).lower()
                node_value = str(G.nodes[node_id].get("value", "")).lower()
                if entity_lower in node_text or entity_lower in node_value:
                    if entity in node_text or entity in node_value:
                        graph_scores[node_id] += 2.0
                    else:
                        graph_scores[node_id] += 1.0
                    for neighbor in G.neighbors(node_id):
                        graph_scores[neighbor] += 0.5
        sorted_scores = sorted(graph_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_scores)
    except Exception as e:
        logger.warning(f"Graph retrieval loi: {e}")
        return {}


# ============ KNOWLEDGE GRAPH ============

def build_knowledge_graph_optimized(nodes: List[Dict], legal=None) -> nx.Graph:
    logger.info("Building Knowledge Graph toi uu...")
    G = nx.Graph()
    all_entities = []
    all_relations = []
    
    for node in tqdm(nodes, desc="Adding nodes"):
        node_id = node["id"]
        text = node["text"]
        level = node.get("level", 0)
        
        doc_id = node.get('doc_id', node.get('metadata', {}).get('doc_id', node_id.split('_')[0]))
        
        G.add_node(node_id, 
                   type="document",
                   doc_id=doc_id,
                   text=text[:500],
                   level=level,
                   metadata=node.get("metadata", {}))
        
        if legal is not None:
            entities = extract_entities_with_llm(text, legal)
            relations = extract_relations_with_llm(text, legal)
        else:
            entities = extract_entities_advanced(text)
            relations = extract_relations_from_text(text)
        
        all_entities.extend(entities)
        all_relations.extend(relations)
        
        for entity in entities:
            entity_id = f"entity_{entity['value'][:30]}_{entity['type']}"
            if not G.has_node(entity_id):
                G.add_node(entity_id, 
                          type="entity",
                          entity_type=entity['type'],
                          value=entity['value'],
                          metadata=entity.get('metadata', {}))
            G.add_edge(node_id, entity_id, relation="contains", weight=1.0)
    
    for rel in all_relations:
        source_id = f"entity_{rel['source'][:30]}_{rel['source_type']}"
        target_id = f"entity_{rel['target'][:30]}_{rel['target_type']}"
        if G.has_node(source_id) and G.has_node(target_id):
            if G.has_edge(source_id, target_id):
                G[source_id][target_id]['weight'] = G[source_id][target_id].get('weight', 1.0) + 0.5
            else:
                G.add_edge(source_id, target_id, 
                          relation=rel['type'],
                          weight=rel.get('weight', 1.0))
    
    # Ket noi cac entity co cung tu khoa
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "entity"]
    for i in range(len(entity_nodes)):
        for j in range(i+1, len(entity_nodes)):
            val1 = G.nodes[entity_nodes[i]].get('value', '')
            val2 = G.nodes[entity_nodes[j]].get('value', '')
            common_words = set(val1.split()) & set(val2.split())
            if len(common_words) > 0:
                G.add_edge(entity_nodes[i], entity_nodes[j], relation="co_occur", weight=0.5)
    
    logger.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


if __name__ == "__main__":
    print("Testing member_b.py...")
    
    bm25_data = build_bm25()
    print(f"BM25 docs: {len(bm25_data.get('doc_ids', []))}")
    
    graph = build_graph()
    print(f"Graph nodes: {graph.number_of_nodes()}")
    
    results = hybrid_retrieve("Quyen cua nguoi lao dong", top_k=5)
    print(f"Hybrid results: {len(results)}")
    for r in results[:3]:
        print(f"  - {r['id']}: score={r['score']:.4f}")