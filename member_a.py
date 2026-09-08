"""
member_a.py - RAPTOR + Vector Store (Phiên bản hoàn chỉnh)
"""

import os
import json
import pickle
import pathlib
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CKPT_DIR = pathlib.Path("/kaggle/working/A_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RAPTOR_CKPT = CKPT_DIR / "raptor_tree.pkl"
VECTOR_CKPT = CKPT_DIR / "vector_store.pkl"
CHUNK_CKPT = CKPT_DIR / "chunks.pkl"

CHUNK_SIZE = 512
OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
CLUSTER_THRESHOLD = 0.7


# ============ HÀM CHÍNH ============

def build_raptor(legal=None, emb=None):
    if RAPTOR_CKPT.exists():
        logger.info(f"Load RAPTOR tu checkpoint: {RAPTOR_CKPT}")
        with open(RAPTOR_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("Dang xay dung RAPTOR tree...")
    
    if emb is not None:
        embedder = emb
    else:
        logger.info("   Loading BAAI/bge-m3...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    documents = load_documents()
    if not documents:
        logger.warning("Khong co du lieu! Dung sample documents.")
        documents = create_sample_documents()
    
    chunks = chunk_documents_optimized(documents)
    logger.info(f"Da tao {len(chunks)} chunks")
    
    texts = [c["text"] for c in chunks]
    logger.info("   Encoding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    tree = build_raptor_tree_optimized(chunks, embeddings, embedder, legal)
    
    with open(RAPTOR_CKPT, "wb") as f:
        pickle.dump(tree, f)
    logger.info(f"RAPTOR checkpoint luu tai {RAPTOR_CKPT}")
    
    return tree


def build_vector_store(emb=None):
    if VECTOR_CKPT.exists():
        logger.info(f"Load Vector Store tu checkpoint: {VECTOR_CKPT}")
        with open(VECTOR_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("Dang xay dung Vector Store toi uu...")
    
    if not RAPTOR_CKPT.exists():
        logger.error("Chua co RAPTOR tree! Chay build_raptor() truoc.")
        return None
    
    with open(RAPTOR_CKPT, "rb") as f:
        tree = pickle.load(f)
    
    if emb is not None:
        embedder = emb
    else:
        logger.info("   Loading BAAI/bge-m3...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    all_nodes = tree.get("nodes", [])
    if not all_nodes:
        logger.error("Khong co nodes trong tree!")
        return None
    
    texts = []
    node_ids = []
    metadata_list = []
    
    for node in all_nodes:
        texts.append(node["text"])
        node_ids.append(node["id"])
        metadata_list.append(node.get("metadata", {}))
    
    logger.info(f"   Encoding {len(texts)} nodes...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    dim = embeddings.shape[1]
    
    if len(embeddings) > 1000:
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, min(100, len(embeddings) // 10))
        index.train(np.array(embeddings).astype('float32'))
        index.add(np.array(embeddings).astype('float32'))
        index_type = "IVF"
    else:
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype('float32'))
        index_type = "Flat"
    
    vector_store = {
        "index": index,
        "node_ids": node_ids,
        "texts": texts,
        "metadata": metadata_list,
        "dimension": dim,
        "num_nodes": len(node_ids),
        "index_type": index_type
    }
    
    with open(VECTOR_CKPT, "wb") as f:
        pickle.dump(vector_store, f)
    logger.info(f"Vector Store checkpoint luu tai {VECTOR_CKPT}")
    
    return vector_store


# ============ CHUNKING (CO LUU DOC_ID GOC) ============

def chunk_documents_optimized(documents: List[Dict]) -> List[Dict]:
    chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        content = doc.get('content', '') or doc.get('text', '')
        doc_id = doc.get('id', f"doc_{len(chunks)}")
        title = doc.get('title', doc.get('name', 'Van ban phap luat'))
        doc_type = doc.get('type', 'unknown')
        
        if not content:
            continue
        
        articles = split_by_article(content)
        
        if articles:
            for article in articles:
                article_num = article['num']
                article_text = article['text']
                
                clauses = split_by_clause(article_text)
                
                if clauses:
                    for clause in clauses:
                        clause_num = clause['num']
                        clause_text = clause['text']
                        
                        points = split_by_point(clause_text)
                        
                        if points:
                            for point in points:
                                if len(point['text'].strip()) > 20:
                                    chunks.append({
                                        'id': f"{doc_id}_a{article_num}_c{clause_num}_p{point['num']}",
                                        'doc_id': doc_id,
                                        'text': f"[{title}] Dieu {article_num}, Khoan {clause_num}, Diem {point['num']}: {point['text'][:CHUNK_SIZE]}",
                                        'level': 0,
                                        'metadata': {
                                            'doc_id': doc_id,
                                            'title': title,
                                            'doc_type': doc_type,
                                            'article': article_num,
                                            'clause': clause_num,
                                            'point': point['num'],
                                            'source': 'legal_document'
                                        }
                                    })
                        else:
                            if len(clause_text.strip()) > 20:
                                chunks.append({
                                    'id': f"{doc_id}_a{article_num}_c{clause_num}",
                                    'doc_id': doc_id,
                                    'text': f"[{title}] Dieu {article_num}, Khoan {clause_num}: {clause_text[:CHUNK_SIZE]}",
                                    'level': 0,
                                    'metadata': {
                                        'doc_id': doc_id,
                                        'title': title,
                                        'doc_type': doc_type,
                                        'article': article_num,
                                        'clause': clause_num,
                                        'source': 'legal_document'
                                    }
                                })
                else:
                    if len(article_text.strip()) > 20:
                        chunks.append({
                            'id': f"{doc_id}_a{article_num}",
                            'doc_id': doc_id,
                            'text': f"[{title}] Dieu {article_num}: {article_text[:CHUNK_SIZE]}",
                            'level': 0,
                            'metadata': {
                                'doc_id': doc_id,
                                'title': title,
                                'doc_type': doc_type,
                                'article': article_num,
                                'source': 'legal_document'
                            }
                        })
        else:
            paragraphs = content.split('\n\n')
            for p_idx, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append({
                        'id': f"{doc_id}_chunk_{p_idx}",
                        'doc_id': doc_id,
                        'text': f"[{title}] {para.strip()[:CHUNK_SIZE]}",
                        'level': 0,
                        'metadata': {
                            'doc_id': doc_id,
                            'title': title,
                            'doc_type': doc_type,
                            'source': 'legal_document'
                        }
                    })
        
        # Checkpoint moi 50 chunk
        if len(chunks) % 50 == 0 and len(chunks) > 0:
            with open(CHUNK_CKPT, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f"   Chunk checkpoint saved at {len(chunks)} chunks")
    
    # Luu chunk checkpoint cuoi cung
    with open(CHUNK_CKPT, "wb") as f:
        pickle.dump(chunks, f)
    
    logger.info(f"Da tao {len(chunks)} chunks")
    return chunks


def split_by_article(text: str) -> List[Dict]:
    pattern = r'(Điều\s+(\d+[a-zA-Z]?)[\.\:\s]+)'
    if not re.search(pattern, text):
        return []
    parts = re.split(pattern, text)
    articles = []
    for i in range(1, len(parts), 3):
        if i+1 < len(parts):
            num = parts[i+1].strip()
            content = parts[i+2] if i+2 < len(parts) else ""
            articles.append({"num": num, "text": content})
    return articles


def split_by_clause(text: str) -> List[Dict]:
    pattern = r'[Kk]hoản\s+(\d+[a-zA-Z]?)[\.\:\s]+'
    if not re.search(pattern, text):
        return []
    parts = re.split(pattern, text)
    clauses = []
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            num = parts[i].strip()
            content = parts[i+1] if i+1 < len(parts) else ""
            clauses.append({"num": num, "text": content})
    return clauses


def split_by_point(text: str) -> List[Dict]:
    pattern = r'([a-zâêôơưđ])\s*[\)\.]\s*'
    if not re.search(pattern, text):
        return []
    parts = re.split(pattern, text)
    points = []
    if parts and parts[0].strip():
        points.append({'num': '0', 'text': parts[0].strip()})
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            num = parts[i].strip()
            content = parts[i+1].strip()
            if content:
                points.append({'num': num, 'text': content})
    return points


def split_by_paragraph(text: str) -> List[str]:
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


# ============ RAPTOR TREE ============

def build_raptor_tree_optimized(chunks: List[Dict], embeddings: np.ndarray, embedder, legal_llm=None) -> Dict:
    logger.info("Xay dung RAPTOR tree toi uu...")
    
    tree = {"nodes": [], "levels": []}
    
    # Level 0
    level_0 = chunks.copy()
    tree["nodes"].extend(level_0)
    tree["levels"].append({"level": 0, "node_ids": [n["id"] for n in level_0]})
    logger.info(f"   Level 0: {len(level_0)} nodes")
    
    with open(RAPTOR_CKPT, "wb") as f:
        pickle.dump(tree, f)
    logger.info("   Checkpoint saved (level 0)")
    
    # Level 1
    if len(chunks) > 5:
        logger.info("   Building level 1 clusters...")
        try:
            from sklearn.cluster import AgglomerativeClustering
            n_clusters = min(max(3, len(chunks) // 4), 15)
            clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='cosine', linkage='average')
            labels = clustering.fit_predict(embeddings.astype('float32'))
            
            level_1 = []
            for cluster_idx in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_idx]
                cluster_chunks = [chunks[i] for i in cluster_indices]
                cluster_texts = [c["text"] for c in cluster_chunks]
                
                if len(cluster_texts) > 1:
                    if legal_llm is not None:
                        summary = summarize_cluster_with_llm(cluster_texts, legal_llm)
                    else:
                        summary = summarize_cluster_advanced(cluster_texts, embedder)
                    
                    level_1.append({
                        "id": f"cluster_{cluster_idx}",
                        "doc_id": cluster_chunks[0].get('doc_id', 'unknown'),
                        "text": summary,
                        "level": 1,
                        "metadata": {
                            "cluster_size": len(cluster_texts),
                            "children": [c["id"] for c in cluster_chunks],
                            "cluster_indices": cluster_indices
                        }
                    })
                elif cluster_texts:
                    idx = cluster_indices[0]
                    level_1.append(chunks[idx])
            
            tree["nodes"].extend(level_1)
            tree["levels"].append({"level": 1, "node_ids": [n["id"] for n in level_1]})
            logger.info(f"   Level 1: {len(level_1)} nodes")
            
            with open(RAPTOR_CKPT, "wb") as f:
                pickle.dump(tree, f)
            logger.info("   Checkpoint saved (level 1)")
            
            # Level 2
            if len(level_1) > 5:
                logger.info("   Building level 2 clusters...")
                level_1_texts = [n["text"] for n in level_1]
                level_1_embeddings = embedder.encode(level_1_texts, show_progress_bar=False)
                
                n_clusters_2 = min(max(2, len(level_1) // 3), 10)
                clustering_2 = AgglomerativeClustering(n_clusters=n_clusters_2, metric='cosine', linkage='average')
                labels_2 = clustering_2.fit_predict(level_1_embeddings.astype('float32'))
                
                level_2 = []
                for cluster_idx in range(n_clusters_2):
                    cluster_indices = [i for i, label in enumerate(labels_2) if label == cluster_idx]
                    cluster_nodes = [level_1[i] for i in cluster_indices]
                    cluster_texts = [n["text"] for n in cluster_nodes]
                    
                    if len(cluster_texts) > 1:
                        if legal_llm is not None:
                            summary = summarize_cluster_with_llm(cluster_texts, legal_llm)
                        else:
                            summary = summarize_cluster_advanced(cluster_texts, embedder)
                        
                        level_2.append({
                            "id": f"cluster_level2_{cluster_idx}",
                            "doc_id": cluster_nodes[0].get('doc_id', 'unknown'),
                            "text": summary,
                            "level": 2,
                            "metadata": {
                                "cluster_size": len(cluster_texts),
                                "children": [n["id"] for n in cluster_nodes]
                            }
                        })
                
                if level_2:
                    tree["nodes"].extend(level_2)
                    tree["levels"].append({"level": 2, "node_ids": [n["id"] for n in level_2]})
                    logger.info(f"   Level 2: {len(level_2)} nodes")
                    
                    with open(RAPTOR_CKPT, "wb") as f:
                        pickle.dump(tree, f)
                    logger.info("   Checkpoint saved (level 2)")
                    
        except Exception as e:
            logger.warning(f"Clustering failed: {e}")
    
    logger.info(f"RAPTOR tree completed: {len(tree['nodes'])} nodes, {len(tree['levels'])} levels")
    return tree


def summarize_cluster_advanced(texts: List[str], embedder) -> str:
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    try:
        embeddings = embedder.encode(texts, show_progress_bar=False)
        mean_emb = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)
        center_indices = np.argsort(distances)[:min(3, len(texts))]
        center_texts = [texts[i] for i in center_indices]
        if len(center_texts) == 1:
            return f"[Tom tat {len(texts)} van ban] {center_texts[0][:300]}"
        summary = f"[Tom tat {len(texts)} van ban]\n"
        for i, text in enumerate(center_texts, 1):
            summary += f"({i}) {text[:200]}...\n"
        return summary.strip()
    except Exception as e:
        logger.warning(f"Advanced summarization failed: {e}")
        return texts[0][:300] + "..."


def summarize_cluster_with_llm(texts: List[str], legal_llm) -> str:
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    try:
        sample_texts = texts[:5]
        combined_text = "\n---\n".join([t[:300] for t in sample_texts])
        prompt = f"""Tom tat cac van ban phap luat sau thanh mot doan ngan gon (toi da 100 tu), giu nguyen cac so hieu dieu luat quan trong va thuat ngu phap ly:

{combined_text}

Tom tat:"""
        response = legal_llm.generate(prompt, max_length=150)
        summary = response.strip()
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return f"[LLM Tom tat {len(texts)} van ban] {summary}"
    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}")
        return summarize_cluster_advanced(texts, None)


# ============ HÀM PHỤ TRỢ ============

def load_documents(data_dir: str = "data_legalir") -> List[Dict]:
    documents = []
    possible_paths = [
        data_dir,
        "/kaggle/input/legalir",
        "/kaggle/input/legalir-dataset",
        "../data_legalir",
    ]
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    if found_path is None:
        logger.warning("Khong tim thay thu muc du lieu.")
        return []
    logger.info(f"Doc du lieu tu: {found_path}")
    json_files = [f for f in os.listdir(found_path) if f.endswith('.json')]
    for filename in tqdm(json_files, desc="Loading files"):
        filepath = os.path.join(found_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    for doc_id, doc_content in data.items():
                        if "question" in doc_content:
                            documents.append({
                                "id": doc_id,
                                "question": doc_content.get("question", ""),
                                "answer": doc_content.get("answer", []),
                                "type": "query"
                            })
                        elif "content" in doc_content or "text" in doc_content:
                            title = doc_content.get("title", doc_content.get("name", "Van ban"))
                            documents.append({
                                "id": doc_id,
                                "title": title,
                                "content": doc_content.get("content", doc_content.get("text", "")),
                                "type": doc_content.get("type", "legal"),
                                "metadata": doc_content
                            })
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            doc_id = item.get("id", f"doc_{len(documents)}")
                            title = item.get("title", item.get("name", "Van ban"))
                            documents.append({
                                "id": doc_id,
                                "title": title,
                                "content": item.get("content", item.get("text", str(item))),
                                "type": item.get("type", "legal"),
                                "metadata": item
                            })
        except Exception as e:
            logger.warning(f"Loi doc {filename}: {e}")
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def create_sample_documents() -> List[Dict]:
    return [
        {"id": "doc_001", "title": "Luat Mau 1", "content": "Dieu 1: Quy dinh chung. Khoan 1: Pham vi dieu chinh..."},
        {"id": "doc_002", "title": "Luat Mau 2", "content": "Dieu 2: Quyen va nghia vu..."},
        {"id": "doc_003", "title": "Luat Mau 3", "content": "Dieu 3: Trach nhiem..."}
    ]


def get_raptor_nodes() -> List[Dict]:
    if not RAPTOR_CKPT.exists():
        logger.warning("Chua co RAPTOR tree!")
        return []
    with open(RAPTOR_CKPT, "rb") as f:
        tree = pickle.load(f)
    return tree.get("nodes", [])


def get_vector_store() -> Optional[Dict]:
    if not VECTOR_CKPT.exists():
        return None
    with open(VECTOR_CKPT, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    print("Testing member_a.py...")
    tree = build_raptor()
    print(f"Tree nodes: {len(tree.get('nodes', []))}")
    vector_store = build_vector_store()
    if vector_store:
        print(f"Vector store: {vector_store.get('num_nodes', 0)} vectors")