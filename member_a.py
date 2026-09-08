"""
member_a.py - RAPTOR + Vector Store
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

# ============ LOGGING ============
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/A_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RAPTOR_CKPT = CKPT_DIR / "raptor_tree.pkl"
VECTOR_CKPT = CKPT_DIR / "vector_store.pkl"
CHUNK_CKPT = CKPT_DIR / "chunks.pkl"

# ============ CONSTANTS ============
CHUNK_SIZE = 512
OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
CLUSTER_THRESHOLD = 0.7


# ============ HÀM CHÍNH ============

def build_raptor(legal=None, emb=None):
    """
    Xây dựng RAPTOR tree với checkpoint
    
    Args:
        legal: Legal LLM (VLSP2025-LegalSML/qwen3-4b-legal-pretrain)
        emb: Embedding model (BAAI/bge-m3)
    
    Returns:
        Dict: RAPTOR tree với nodes và levels
    """
    # 1. Kiểm tra checkpoint
    if RAPTOR_CKPT.exists():
        logger.info(f" Load RAPTOR từ checkpoint: {RAPTOR_CKPT}")
        with open(RAPTOR_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info(" Đang xây dựng RAPTOR tree...")
    
    # 2. Chuẩn bị embedder
    if emb is not None:
        embedder = emb
    else:
        logger.info("   Loading BAAI/bge-m3...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Đọc dữ liệu
    documents = load_documents()
    if not documents:
        logger.warning(" Không có dữ liệu! Dùng sample documents.")
        documents = create_sample_documents()
    
    # 4. Chunk văn bản
    chunks = chunk_documents(documents)
    logger.info(f" Đã tạo {len(chunks)} chunks")
    
    # 5. Tạo embeddings cho chunks
    texts = [c["text"] for c in chunks]
    logger.info("   Encoding chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # 6. Xây dựng cây RAPTOR
    tree = build_raptor_tree(chunks, embeddings, embedder)
    
    # 7. Lưu checkpoint cuối cùng
    with open(RAPTOR_CKPT, "wb") as f:
        pickle.dump(tree, f)
    logger.info(f" RAPTOR checkpoint lưu tại {RAPTOR_CKPT}")
    
    return tree


def build_vector_store(emb=None):
    """
    Xây dựng FAISS vector store với checkpoint
    
    Args:
        emb: Embedding model (BAAI/bge-m3)
    
    Returns:
        Dict: vector_store với index, node_ids, texts
    """
    if VECTOR_CKPT.exists():
        logger.info(f" Load Vector Store từ checkpoint: {VECTOR_CKPT}")
        with open(VECTOR_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info("⚙️ Đang xây dựng Vector Store...")
    
    # 1. Load RAPTOR tree
    if not RAPTOR_CKPT.exists():
        logger.error(" Chưa có RAPTOR tree! Chạy build_raptor() trước.")
        return None
    
    with open(RAPTOR_CKPT, "rb") as f:
        tree = pickle.load(f)
    
    # 2. Chuẩn bị embedder
    if emb is not None:
        embedder = emb
    else:
        logger.info("   Loading BAAI/bge-m3...")
        embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    # 3. Lấy tất cả nodes từ tree
    all_nodes = tree.get("nodes", [])
    if not all_nodes:
        logger.error(" Không có nodes trong tree!")
        return None
    
    # 4. Tạo embeddings
    texts = [n["text"] for n in all_nodes]
    node_ids = [n["id"] for n in all_nodes]
    
    logger.info(f"   Encoding {len(texts)} nodes...")
    embeddings = embedder.encode(texts, show_progress_bar=True)
    
    # 5. Tạo FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    
    vector_store = {
        "index": index,
        "node_ids": node_ids,
        "texts": texts,
        "dimension": dim,
        "num_nodes": len(node_ids)
    }
    
    # 6. Lưu checkpoint
    with open(VECTOR_CKPT, "wb") as f:
        pickle.dump(vector_store, f)
    logger.info(f" Vector Store checkpoint lưu tại {VECTOR_CKPT}")
    
    return vector_store


# ============ HÀM PHỤ TRỢ ============

def load_documents(data_dir: str = "data_legalir") -> List[Dict]:
    """
    Đọc dữ liệu từ thư mục data_legalir
    Hỗ trợ cả cấu trúc dữ liệu của BTC
    """
    documents = []
    
    # Các đường dẫn có thể chứa dữ liệu
    possible_paths = [
        data_dir,
        "/kaggle/input/legalir",
        "/kaggle/input/legalir-dataset",
        "../data_legalir",
    ]
    
    # Tìm đường dẫn hợp lệ
    found_path = None
    for path in possible_paths:
        if os.path.exists(path):
            found_path = path
            break
    
    if found_path is None:
        logger.warning(f" Không tìm thấy thư mục dữ liệu. Tìm trong {possible_paths}")
        return []
    
    logger.info(f" Đọc dữ liệu từ: {found_path}")
    
    # Đọc tất cả file JSON
    json_files = [f for f in os.listdir(found_path) if f.endswith('.json')]
    
    for filename in tqdm(json_files, desc="Loading files"):
        filepath = os.path.join(found_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Trường hợp 1: Dữ liệu BTC - dict với key là ID
                if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
                    for doc_id, doc_content in data.items():
                        # Dữ liệu train.json từ BTC
                        if "question" in doc_content:
                            documents.append({
                                "id": doc_id,
                                "question": doc_content.get("question", ""),
                                "answer": doc_content.get("answer", []),
                                "type": "query"
                            })
                        # Dữ liệu văn bản
                        elif "content" in doc_content or "text" in doc_content:
                            documents.append({
                                "id": doc_id,
                                "content": doc_content.get("content", doc_content.get("text", "")),
                                "metadata": doc_content
                            })
                        else:
                            documents.append({
                                "id": doc_id,
                                "content": str(doc_content)
                            })
                
                # Trường hợp 2: Dữ liệu dạng list
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            doc_id = item.get("id", f"doc_{len(documents)}")
                            documents.append({
                                "id": doc_id,
                                "content": item.get("content", item.get("text", str(item))),
                                "metadata": item
                            })
                
                # Trường hợp 3: Dữ liệu dạng dict khác
                else:
                    documents.append({
                        "id": f"file_{filename}",
                        "content": str(data)
                    })
                    
        except Exception as e:
            logger.warning(f"Lỗi đọc {filename}: {e}")
    
    logger.info(f" Loaded {len(documents)} documents")
    return documents


def create_sample_documents() -> List[Dict]:
    """Tạo dữ liệu mẫu để test"""
    return [
        {
            "id": "doc_001",
            "content": "Điều 1: Quy định chung. Khoản 1: Phạm vi điều chỉnh... Đoạn 1: Luật này quy định..."
        },
        {
            "id": "doc_002",
            "content": "Điều 2: Quyền và nghĩa vụ. Khoản 1: Quyền của người lao động..."
        },
        {
            "id": "doc_003",
            "content": "Điều 3: Trách nhiệm. Khoản 1: Trách nhiệm của người sử dụng lao động..."
        }
    ]


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Chunk văn bản theo Điều → Khoản → Đoạn
    """
    chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        # Lấy nội dung
        content = doc.get('content', '') or doc.get('text', '') or doc.get('question', '')
        doc_id = doc.get('id', f"doc_{len(chunks)}")
        
        if not content:
            continue
        
        # 1. Tách theo Điều
        articles = split_by_article(content)
        
        if articles:
            for article in articles:
                # 2. Tách theo Khoản
                clauses = split_by_clause(article["text"])
                
                if clauses:
                    for clause in clauses:
                        # 3. Tách theo Đoạn
                        paragraphs = split_by_paragraph(clause["text"])
                        
                        for para in paragraphs:
                            if len(para.strip()) > 20:  # Bỏ đoạn quá ngắn
                                chunks.append({
                                    'id': f"{doc_id}_{article['num']}_{clause['num']}_{len(chunks)}",
                                    'text': f"Điều {article['num']}, Khoản {clause['num']}: {para[:CHUNK_SIZE]}",
                                    'level': 0,
                                    'metadata': {
                                        'doc_id': doc_id,
                                        'article': article['num'],
                                        'clause': clause['num']
                                    }
                                })
                else:
                    # Không có Khoản, chunk theo Điều
                    if len(article["text"].strip()) > 20:
                        chunks.append({
                            'id': f"{doc_id}_{article['num']}_{len(chunks)}",
                            'text': f"Điều {article['num']}: {article['text'][:CHUNK_SIZE]}",
                            'level': 0,
                            'metadata': {'doc_id': doc_id, 'article': article['num']}
                        })
        else:
            # Không có Điều, chunk nguyên văn
            paragraphs = content.split('\n\n')
            for p_idx, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append({
                        'id': f"{doc_id}_chunk_{p_idx}",
                        'text': para.strip()[:CHUNK_SIZE],
                        'level': 0,
                        'metadata': {'doc_id': doc_id}
                    })
    
    # Lưu chunk checkpoint
    with open(CHUNK_CKPT, "wb") as f:
        pickle.dump(chunks, f)
    
    return chunks


def split_by_article(text: str) -> List[Dict]:
    """Tách văn bản thành các Điều"""
    # Pattern cho Điều
    pattern = r'(Điều\s+(\d+)[\.\:\s]+)'
    
    # Nếu không có Điều, trả về empty
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
    """Tách thành các Khoản"""
    # Pattern cho Khoản
    pattern = r'[Kk]hoản\s+(\d+)[\.\:\s]+'
    
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


def split_by_paragraph(text: str) -> List[str]:
    """Tách thành các Đoạn (theo dấu xuống dòng)"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def build_raptor_tree(chunks: List[Dict], embeddings: np.ndarray, embedder) -> Dict:
    """
    Xây dựng cây RAPTOR với checkpoint mỗi 50 chunk
    
    Args:
        chunks: List của chunks
        embeddings: Embeddings của chunks
        embedder: SentenceTransformer model
    
    Returns:
        Dict: RAPTOR tree
    """
    logger.info(" Xây dựng RAPTOR tree...")
    
    tree = {
        "nodes": [],
        "levels": []
    }
    
    # ============ LEVEL 0: Chunks gốc ============
    level_0 = chunks.copy()
    tree["nodes"].extend(level_0)
    tree["levels"].append({"level": 0, "node_ids": [n["id"] for n in level_0]})
    logger.info(f"   Level 0: {len(level_0)} nodes")
    
    # Checkpoint sau level 0
    with open(RAPTOR_CKPT, "wb") as f:
        pickle.dump(tree, f)
    logger.info("  Checkpoint saved (level 0)")
    
    # ============ LEVEL 1: Clusters ============
    if len(chunks) > 3:
        logger.info("   Building level 1 clusters...")
        
        try:
            from sklearn.cluster import KMeans
            
            # Tính số cluster
            n_clusters = min(max(2, len(chunks) // 5), 20)
            logger.info(f"   Number of clusters: {n_clusters}")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings.astype('float32'))
            
            # Tạo node tóm tắt cho mỗi cluster
            level_1 = []
            for cluster_idx in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_idx]
                cluster_texts = [chunks[i]["text"] for i in cluster_indices]
                
                if cluster_texts and len(cluster_texts) > 1:
                    # Tóm tắt cluster
                    summary = summarize_cluster(cluster_texts, embedder)
                    level_1.append({
                        "id": f"cluster_{cluster_idx}",
                        "text": summary,
                        "level": 1,
                        "metadata": {
                            "cluster_size": len(cluster_texts),
                            "children": [chunks[i]["id"] for i in cluster_indices]
                        }
                    })
                elif cluster_texts:
                    # Chỉ có 1 node, giữ nguyên
                    idx = cluster_indices[0]
                    level_1.append(chunks[idx])
            
            tree["nodes"].extend(level_1)
            tree["levels"].append({"level": 1, "node_ids": [n["id"] for n in level_1]})
            logger.info(f"   Level 1: {len(level_1)} nodes")
            
            # Checkpoint sau level 1
            with open(RAPTOR_CKPT, "wb") as f:
                pickle.dump(tree, f)
            logger.info("    Checkpoint saved (level 1)")
            
        except Exception as e:
            logger.warning(f" Clustering failed: {e}")
    
    logger.info(f" RAPTOR tree completed: {len(tree['nodes'])} nodes, {len(tree['levels'])} levels")
    return tree


def summarize_cluster(texts: List[str], embedder=None) -> str:
    """
    Tóm tắt một cụm văn bản
    
    Args:
        texts: List các văn bản trong cụm
        embedder: Embedding model (dùng để chọn representative)
    
    Returns:
        str: Văn bản tóm tắt
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    # Cách 1: Chọn câu dài nhất (đơn giản)
    longest = max(texts, key=len)
    
    # Cách 2: Nếu có embedder, chọn câu gần trung tâm nhất
    if embedder is not None:
        try:
            embeddings = embedder.encode(texts, show_progress_bar=False)
            mean_emb = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - mean_emb, axis=1)
            center_idx = np.argmin(distances)
            center_text = texts[center_idx]
            
            # Nếu center text dài hơn 200, cắt ngắn
            if len(center_text) > 300:
                return f"[Tóm tắt {len(texts)} văn bản] {center_text[:300]}..."
            return f"[Tóm tắt {len(texts)} văn bản] {center_text}"
            
        except Exception:
            pass
    
    # Fallback
    if len(longest) > 300:
        return f"[Tóm tắt {len(texts)} văn bản] {longest[:300]}..."
    return f"[Tóm tắt {len(texts)} văn bản] {longest}"


def get_raptor_nodes() -> List[Dict]:
    """Lấy tất cả nodes từ RAPTOR tree"""
    if not RAPTOR_CKPT.exists():
        logger.warning(" Chưa có RAPTOR tree!")
        return []
    
    with open(RAPTOR_CKPT, "rb") as f:
        tree = pickle.load(f)
    return tree.get("nodes", [])


def get_vector_store() -> Optional[Dict]:
    """Lấy vector store từ checkpoint"""
    if not VECTOR_CKPT.exists():
        return None
    
    with open(VECTOR_CKPT, "rb") as f:
        return pickle.load(f)


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    # Test nhanh
    print("Testing member_a.py...")
    tree = build_raptor()
    print(f"Tree nodes: {len(tree.get('nodes', []))}")
    
    vector_store = build_vector_store()
    if vector_store:
        print(f"Vector store: {vector_store.get('num_nodes', 0)} vectors")