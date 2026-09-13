"""
member_a.py - RAPTOR + Vector Store (Phiên bản tối ưu hoàn chỉnh)
"""

import os
import json
import pickle
import pathlib
import shutil
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


# ============ TỰ ĐỘNG KHÔI PHỤC CHECKPOINT TỪ /kaggle/input ============

def _restore_checkpoint_from_input(folder_name: str, ckpt_dir: pathlib.Path):
    """
    Nếu bạn đã Add Data một Dataset chứa checkpoint cũ (tải checkpoint về
    máy từ 1 lần chạy trước, upload lại thành Dataset, gắn vào notebook qua
    "Add Data"), hàm này tự tìm thư mục con tên đúng "{folder_name}" bên
    trong /kaggle/input và copy các file còn thiếu vào /kaggle/working,
    để build_...() nhận ra và resume tiếp mà không cần thao tác gì thêm.

    Chỉ copy file NÀO CHƯA CÓ ở ckpt_dir (không ghi đè checkpoint mới hơn
    đang có sẵn trong session hiện tại).
    """
    input_root = pathlib.Path("/kaggle/input")
    if not input_root.exists():
        return

    try:
        for dataset_dir in input_root.iterdir():
            if not dataset_dir.is_dir():
                continue
            # Tìm thư mục "{folder_name}" ngay trong dataset, hoặc lồng thêm 1 cấp
            candidates = [dataset_dir / folder_name] + list(dataset_dir.glob(f"*/{folder_name}"))
            for src in candidates:
                if src.is_dir():
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    copied = 0
                    for f in src.iterdir():
                        dst = ckpt_dir / f.name
                        if not dst.exists():
                            shutil.copy2(f, dst)
                            copied += 1
                    if copied:
                        logger.info(f" Đã khôi phục {copied} file checkpoint từ {src} -> {ckpt_dir}")
                    return
    except Exception as e:
        logger.warning(f" Lỗi khi quét /kaggle/input để khôi phục checkpoint: {e}")


# ============ CHECKPOINT CONFIG ============
CKPT_DIR = pathlib.Path("/kaggle/working/A_checkpoint")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
_restore_checkpoint_from_input("A_checkpoint", CKPT_DIR)
RAPTOR_CKPT = CKPT_DIR / "raptor_tree.pkl"
VECTOR_CKPT = CKPT_DIR / "vector_store.pkl"
CHUNK_CKPT = CKPT_DIR / "chunks.pkl"
# Checkpoint riêng cho bước encode (GPU, tốn thời gian nhất, dễ bị Kaggle
# ngắt giữa chừng nhất) -> lưu theo batch để có thể resume, không phải
# encode lại từ đầu nếu mất kết nối/hết giờ giữa chừng.
#
# [SỬA LỖI HIỆU NĂNG] Trước đây đây là 1 FILE .pkl duy nhất, và MỖI BATCH
# đều ghi đè lại TOÀN BỘ list embeddings đã encode từ đầu tới giờ. Với
# ~800k chunk, batch cuối phải ghi lại full ~800k vector MỖI LẦN -> tổng
# I/O tăng theo kiểu O(n^2), khiến encode càng về sau càng chậm dần (đây
# chính là nguyên nhân log train bị "đứng hình" ở bước Encoding chunks).
# Giờ đổi thành 1 THƯ MỤC chứa nhiều "shard" (mỗi batch = 1 file .npy
# riêng) -> mỗi batch chỉ ghi đúng phần MỚI của batch đó, tổng chi phí ghi
# đĩa là O(n), không tăng dần theo tiến trình. Resume chỉ cần đếm số shard
# đã có, không cần đọc lại nội dung cũ.
EMBED_CKPT_CHUNKS = CKPT_DIR / "embed_shards_chunks"
EMBED_CKPT_NODES = CKPT_DIR / "embed_shards_nodes"
# [SỬA LỖI HIỆU NĂNG - MỚI PHÁT HIỆN] build_vector_store() gọi
# encode_with_checkpoint() cho TẤT CẢ node của tree (bao gồm toàn bộ node
# level 0, tức chính là các chunk vừa được encode xong trong build_raptor()
# vài dòng phía trên) -> ENCODE LẠI TỪ ĐẦU đúng những gì đã encode, tốn
# gấp đôi thời gian GPU cho bước nặng nhất của cả pipeline (y hệt cảm giác
# "đứng ở bước encode" lần thứ 2, chỉ là ở build_vector_store thay vì
# build_raptor). Giờ lưu lại embeddings level 0 đã tính (theo id) vào 1
# file cache gọn, để build_vector_store() TÁI SỬ DỤNG thay vì encode lại;
# chỉ còn phải encode mới cho node level 1/2 (cluster summary, số lượng
# rất nhỏ, tối đa N_CLUSTERS_CAP_L1 + N_CLUSTERS_CAP_L2 ~ 65 node).
LEVEL0_EMBED_CACHE = CKPT_DIR / "level0_embeddings.npz"

# ============ CONSTANTS ============
CHUNK_SIZE = 512
OVERLAP = 50
EMBEDDING_MODEL = "BAAI/bge-m3"
CLUSTER_THRESHOLD = 0.7
# Có tối đa bao nhiêu LLM call khi tóm tắt cluster ở RAPTOR level 1/2. Số
# cluster luôn nhỏ (xem N_CLUSTERS_CAP) nên không cần giới hạn thêm, nhưng
# để tường minh ở 1 chỗ.
N_CLUSTERS_CAP_L1 = 50
N_CLUSTERS_CAP_L2 = 15


# ============ ENCODE CÓ CHECKPOINT (SHARD, O(n) KHÔNG PHẢI O(n^2)) ============

def encode_with_checkpoint(embedder, texts: List[str], shard_dir: pathlib.Path,
                            batch_size: int = 256, desc: str = "Encoding") -> np.ndarray:
    """
    Encode văn bản theo từng batch, MỖI BATCH LƯU RA 1 FILE SHARD (.npy)
    RIÊNG trong `shard_dir`, thay vì ghi đè lại toàn bộ danh sách embeddings
    mỗi lần (cách cũ gây I/O O(n^2), xem giải thích ở chỗ khai báo
    EMBED_CKPT_CHUNKS/NODES phía trên).

    Mỗi batch chỉ ghi đúng phần embeddings MỚI của batch đó -> tổng chi phí
    ghi đĩa tỉ lệ thuận O(n), không tăng dần theo tiến trình như trước.

    Resume: chỉ cần đếm số shard file `part_*.npy` đã có trong `shard_dir`
    (mỗi shard ứng đúng 1 batch, theo đúng thứ tự) để biết đã encode tới
    batch nào, không cần đọc lại nội dung cũ.

    Ghi an toàn: encode xong 1 batch mới ghi ra file `.tmp` rồi rename sang
    tên thật -> nếu bị ngắt đúng lúc đang ghi thì shard đó coi như chưa
    hoàn thành (không để lại file part_*.npy hỏng làm sai số đếm resume).
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    n_batches = (n + batch_size - 1) // batch_size

    existing_shards = sorted(shard_dir.glob("part_*.npy"))
    start_batch = 0
    while (shard_dir / f"part_{start_batch:06d}.npy").exists():
        start_batch += 1

    if start_batch >= n_batches:
        logger.info(f" Đã encode đủ {n} văn bản, load {start_batch} shard checkpoint: {shard_dir}")
    elif start_batch > 0:
        logger.info(f" Resume encoding từ batch {start_batch}/{n_batches} (shard checkpoint: {shard_dir})")

    for b in tqdm(range(start_batch, n_batches), desc=desc, initial=start_batch, total=n_batches):
        i = b * batch_size
        batch = texts[i:i + batch_size]
        batch_emb = np.asarray(embedder.encode(batch))

        shard_path = shard_dir / f"part_{b:06d}.npy"
        tmp_path = shard_dir / f"part_{b:06d}.tmp.npy"
        np.save(tmp_path, batch_emb)
        tmp_path.rename(shard_path)

    # Gộp toàn bộ shard lại thành 1 mảng duy nhất để trả về (chỉ ĐỌC, các
    # lần chạy sau không phải ghi lại phần đã có).
    shards = sorted(shard_dir.glob("part_*.npy"))
    embeddings = np.concatenate([np.load(s) for s in shards], axis=0)
    return embeddings


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
            tree = pickle.load(f)

        # Chỉ coi checkpoint là hoàn chỉnh khi đã có level 2.
        # Nếu mới có level 0/1 thì tiếp tục resume các level còn thiếu.
        levels = {x.get("level") for x in tree.get("levels", [])}
        if 2 in levels:
            logger.info(" RAPTOR checkpoint hoàn chỉnh -> load và return")
            return tree

        logger.info(f" RAPTOR checkpoint chưa hoàn chỉnh {sorted(levels)} -> resume")

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


    # 4. Chunk văn bản (phiên bản tối ưu)
    chunks = chunk_documents_optimized(documents)
    logger.info(f" Đã tạo {len(chunks)} chunks")

    # 5. Tạo embeddings cho chunks (có checkpoint theo batch, resume được
    #    nếu bị ngắt giữa chừng thay vì encode lại từ đầu)
    texts = [c["text"] for c in chunks]
    logger.info("   Encoding chunks...")
    embeddings = encode_with_checkpoint(embedder, texts, EMBED_CKPT_CHUNKS, desc="Encoding chunks")

    # Lưu cache embeddings level-0 theo id để build_vector_store() TÁI SỬ
    # DỤNG sau này thay vì encode lại toàn bộ chunk lần nữa (xem giải thích
    # ở khai báo LEVEL0_EMBED_CACHE phía trên).
    try:
        np.savez(
            str(LEVEL0_EMBED_CACHE),
            ids=np.array([c["id"] for c in chunks], dtype=object),
            vectors=embeddings.astype("float32"),
        )
        logger.info(f"   Đã lưu cache embedding level-0 ({len(chunks)} vectors) -> {LEVEL0_EMBED_CACHE}")
    except Exception as e:
        logger.warning(f" Không lưu được cache embedding level-0: {e}")

    # 6. Xây dựng cây RAPTOR (phiên bản tối ưu)
    tree = build_raptor_tree_optimized(chunks, embeddings, embedder, legal)

    # 7. Lưu checkpoint cuối cùng
    with open(RAPTOR_CKPT, "wb") as f:
        pickle.dump(tree, f)
    logger.info(f" RAPTOR checkpoint lưu tại {RAPTOR_CKPT}")

    # Đã build xong tree -> không cần checkpoint embedding trung gian nữa,
    # xóa để tránh chiếm dung lượng ổ đĩa Kaggle không cần thiết.
    if EMBED_CKPT_CHUNKS.exists():
        shutil.rmtree(EMBED_CKPT_CHUNKS)

    return tree


def build_vector_store(emb=None):
    """
    Xây dựng FAISS vector store với checkpoint (phiên bản tối ưu)
    
    Args:
        emb: Embedding model (BAAI/bge-m3)
    
    Returns:
        Dict: vector_store với index, node_ids, texts
    """
    if VECTOR_CKPT.exists():
        logger.info(f" Load Vector Store từ checkpoint: {VECTOR_CKPT}")
        with open(VECTOR_CKPT, "rb") as f:
            return pickle.load(f)
    
    logger.info(" Đang xây dựng Vector Store tối ưu...")
    
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
    
    # 4. Lấy text và metadata
    texts = []
    node_ids = []
    metadata_list = []
    
    for node in all_nodes:
        texts.append(node["text"])
        node_ids.append(node["id"])
        metadata_list.append(node.get("metadata", {}))
    
    # 5. Tạo embeddings - TÁI SỬ DỤNG embeddings level-0 đã encode sẵn từ
    #    build_raptor() (xem LEVEL0_EMBED_CACHE), CHỈ encode mới cho node
    #    chưa có trong cache (thực chất là node level 1/2 - cluster summary,
    #    số lượng rất nhỏ). [SỬA LỖI HIỆU NĂNG] Trước đây hàm này luôn gọi
    #    encode_with_checkpoint() cho TẤT CẢ node kể cả node level 0, tức
    #    encode lại lần 2 chính các chunk vừa encode xong ở build_raptor(),
    #    tốn gấp đôi thời gian GPU cho bước nặng nhất pipeline.
    cached_vectors: Dict[str, np.ndarray] = {}
    if LEVEL0_EMBED_CACHE.exists():
        try:
            cache_data = np.load(str(LEVEL0_EMBED_CACHE), allow_pickle=True)
            cached_vectors = dict(zip(cache_data["ids"].tolist(), cache_data["vectors"]))
            logger.info(f"   Tái sử dụng {len(cached_vectors)} embedding level-0 đã tính sẵn (không encode lại)")
        except Exception as e:
            logger.warning(f" Không đọc được cache embedding level-0, sẽ encode lại toàn bộ: {e}")

    missing_indices = [i for i, nid in enumerate(node_ids) if nid not in cached_vectors]

    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        logger.info(f"   Encoding {len(missing_texts)}/{len(texts)} nodes chưa có cache "
                    f"(cluster level 1/2, hoặc cache không khớp)...")
        # Số lượng còn lại thường rất nhỏ (vài chục node cluster) nên
        # không cần checkpoint theo shard như encode_with_checkpoint,
        # encode thẳng 1 lần cho gọn.
        missing_embeddings = np.asarray(embedder.encode(missing_texts, batch_size=64))
    else:
        missing_embeddings = None

    dim = (next(iter(cached_vectors.values())).shape[0] if cached_vectors
           else missing_embeddings.shape[1])
    embeddings = np.zeros((len(node_ids), dim), dtype="float32")
    for i, nid in enumerate(node_ids):
        if nid in cached_vectors:
            embeddings[i] = cached_vectors[nid]
    for j, i in enumerate(missing_indices):
        embeddings[i] = missing_embeddings[j]
    
    # 6. Tạo FAISS index (IVF cho tốc độ)
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
    
    # 7. Lưu checkpoint
    with open(VECTOR_CKPT, "wb") as f:
        pickle.dump(vector_store, f)
    logger.info(f" Vector Store checkpoint lưu tại {VECTOR_CKPT}")

    # Đã build xong vector store -> xóa checkpoint embedding trung gian
    if EMBED_CKPT_NODES.exists():
        shutil.rmtree(EMBED_CKPT_NODES)

    return vector_store


# ============ TỐI ƯU CHUNKING ============

def chunk_documents_optimized(documents: List[Dict]) -> List[Dict]:
    """
    Chunk văn bản thông minh:
    - Giữ nguyên cấu trúc Điều - Khoản - Điểm
    - Thêm metadata (tên văn bản, chương, loại văn bản)
    - Bảo toàn ngữ cảnh

    [SỬA LỖI HIỆU NĂNG] Bản trước LUÔN tách xuống tận cấp Điểm (a, b, c...)
    bất kể Khoản dài hay ngắn -> với dữ liệu thật (~8.5k văn bản) tạo ra
    ~811k chunk (mỗi Khoản 3-4 điểm lại thành 3-4 chunk riêng, phần lớn chỉ
    vài chục ký tự). Số chunk khổng lồ này kéo chậm MỌI bước phía sau: encode
    embedding, RAPTOR clustering, build Knowledge Graph, BM25...

    Giờ đổi chiến lược: MỘT KHOẢN GIỮ NGUYÊN LÀ 1 CHUNK (gồm cả các điểm bên
    trong) nếu đủ ngắn (<= CHUNK_SIZE ký tự). CHỈ tách tiếp xuống cấp Điểm khi
    Khoản đó dài hơn CHUNK_SIZE (thật sự cần chia nhỏ để không mất ngữ cảnh
    khi encode/rerank). Việc này giảm số chunk xuống nhiều lần (còn tuỳ dữ
    liệu, thường 5-10 lần) mà không mất thông tin, vì đa số Khoản trong văn
    bản pháp luật vốn đã ngắn hơn 512 ký tự.
    """
    chunks = []
    
    for doc in tqdm(documents, desc="Chunking documents"):
        content = doc.get('content', '') or doc.get('text', '')
        doc_id = doc.get('id', f"doc_{len(chunks)}")
        title = doc.get('title', doc.get('name', 'Văn bản pháp luật'))
        doc_type = doc.get('type', 'unknown')
        
        if not content:
            continue
        
        # 1. Tách theo Điều
        articles = split_by_article(content)
        
        if articles:
            for article in articles:
                article_num = article['num']
                article_text = article['text']
                
                # 2. Tách theo Khoản
                clauses = split_by_clause(article_text)
                
                if clauses:
                    for clause in clauses:
                        clause_num = clause['num']
                        clause_text = clause['text']
                        clause_stripped = clause_text.strip()

                        if not clause_stripped or len(clause_stripped) <= 20:
                            continue

                        if len(clause_stripped) <= CHUNK_SIZE:
                            # Khoản đã đủ ngắn -> giữ nguyên cả khoản (kể cả
                            # các điểm bên trong) làm 1 chunk duy nhất, KHÔNG
                            # tách vụn xuống cấp điểm.
                            chunks.append({
                                'id': f"{doc_id}_a{article_num}_c{clause_num}",
                                'text': f"[{title}] Điều {article_num}, Khoản {clause_num}: {clause_stripped[:CHUNK_SIZE]}",
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
                            continue

                        # 3. Khoản quá dài -> mới cần tách theo Điểm để
                        # không cắt cụt mất thông tin.
                        points = split_by_point(clause_text)

                        if points:
                            for point in points:
                                if len(point['text'].strip()) > 20:
                                    chunks.append({
                                        'id': f"{doc_id}_a{article_num}_c{clause_num}_p{point['num']}",
                                        'text': f"[{title}] Điều {article_num}, Khoản {clause_num}, Điểm {point['num']}: {point['text'][:CHUNK_SIZE]}",
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
                            # Khoản dài nhưng không tách được theo Điểm ->
                            # cắt cứng thành nhiều đoạn CHUNK_SIZE ký tự (có
                            # overlap) để không mất phần đuôi văn bản.
                            step = max(CHUNK_SIZE - OVERLAP, 1)
                            for seg_idx, start in enumerate(range(0, len(clause_stripped), step)):
                                segment = clause_stripped[start:start + CHUNK_SIZE]
                                if len(segment.strip()) > 20:
                                    chunks.append({
                                        'id': f"{doc_id}_a{article_num}_c{clause_num}_s{seg_idx}",
                                        'text': f"[{title}] Điều {article_num}, Khoản {clause_num}: {segment}",
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
                                if start + CHUNK_SIZE >= len(clause_stripped):
                                    break
                else:
                    # Không có Khoản, chunk theo Điều
                    if len(article_text.strip()) > 20:
                        chunks.append({
                            'id': f"{doc_id}_a{article_num}",
                            'text': f"[{title}] Điều {article_num}: {article_text[:CHUNK_SIZE]}",
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
            # Không có Điều, chunk theo đoạn
            paragraphs = content.split('\n\n')
            for p_idx, para in enumerate(paragraphs):
                if para.strip():
                    chunks.append({
                        'id': f"{doc_id}_chunk_{p_idx}",
                        'text': f"[{title}] {para.strip()[:CHUNK_SIZE]}",
                        'level': 0,
                        'metadata': {
                            'doc_id': doc_id,
                            'title': title,
                            'doc_type': doc_type,
                            'source': 'legal_document'
                        }
                    })
    
    # Lưu chunk checkpoint
    with open(CHUNK_CKPT, "wb") as f:
        pickle.dump(chunks, f)
    
    logger.info(f" Đã tạo {len(chunks)} chunks")
    return chunks


def split_by_article(text: str) -> List[Dict]:
    """Tách văn bản thành các Điều"""
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
    """Tách thành các Khoản"""
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
    """Tách thành các Điểm (a, b, c, ...)"""
    pattern = r'([a-zâêôơưđ])\s*[\)\.]\s*'
    
    if not re.search(pattern, text):
        return []
    
    parts = re.split(pattern, text)
    points = []
    
    # parts[0] là text trước điểm đầu tiên
    if parts and parts[0].strip():
        points.append({
            'num': '0',
            'text': parts[0].strip()
        })
    
    for i in range(1, len(parts), 2):
        if i+1 < len(parts):
            num = parts[i].strip()
            content = parts[i+1].strip()
            if content:
                points.append({
                    'num': num,
                    'text': content
                })
    
    return points


def split_by_paragraph(text: str) -> List[str]:
    """Tách thành các Đoạn (theo dấu xuống dòng)"""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


# ============ TỐI ƯU RAPTOR TREE ============

def build_raptor_tree_optimized(chunks: List[Dict], embeddings: np.ndarray, embedder, legal_llm=None) -> Dict:
    """
    Xây dựng cây RAPTOR tối ưu với:
    - AgglomerativeClustering thay vì KMeans
    - LLM summarization nếu có
    - 3 levels
    """
    logger.info(" Xây dựng RAPTOR tree tối ưu...")

    # ============ LOAD / RESUME TREE ============
    if RAPTOR_CKPT.exists():
        logger.info(f" Resume RAPTOR từ checkpoint: {RAPTOR_CKPT}")

        with open(RAPTOR_CKPT, "rb") as f:
            tree = pickle.load(f)

        completed_levels = {
            x.get("level") for x in tree.get("levels", [])
        }

        logger.info(
            f" Các level đã hoàn thành: {sorted(completed_levels)}"
        )
    else:
        tree = {
            "nodes": [],
            "levels": []
        }
        completed_levels = set()

    # ============ LEVEL 0 ============
    if 0 not in completed_levels:
        level_0 = chunks.copy()

        tree["nodes"].extend(level_0)
        tree["levels"].append({
            "level": 0,
            "node_ids": [n["id"] for n in level_0]
        })

        logger.info(f"   Level 0: {len(level_0)} nodes")

        with open(RAPTOR_CKPT, "wb") as f:
            pickle.dump(tree, f)

        logger.info("    Checkpoint saved (level 0)")

        completed_levels.add(0)

    # ============ LEVEL 1: Clusters ============
    level_1 = []

    # Nếu level 1 đã có trong checkpoint, khôi phục các node level 1
    # từ tree để có thể tiếp tục xây level 2.
    if 1 in completed_levels:
        level_1_ids = next(
            (x.get("node_ids", []) for x in tree.get("levels", [])
             if x.get("level") == 1),
            []
        )
        node_by_id = {n.get("id"): n for n in tree.get("nodes", [])}
        level_1 = [node_by_id[nid] for nid in level_1_ids if nid in node_by_id]

    if len(chunks) > 5 and 1 not in completed_levels:
        logger.info("   Building level 1 clusters...")

        try:
            from sklearn.cluster import MiniBatchKMeans

            # LƯU Ý: AgglomerativeClustering cần ma trận khoảng cách O(n^2) -
            # với hàng trăm nghìn chunk sẽ cần hàng trăm GB RAM và chắc chắn
            # crash/treo trên Kaggle. Dùng MiniBatchKMeans thay thế vì nó chỉ
            # cần O(n) bộ nhớ và scale tốt tới hàng triệu điểm.
            #
            # Số cluster: dùng heuristic sqrt(n) (phổ biến cho KMeans) thay vì
            # công thức cũ "//4 rồi cap 15" (cap 15 quá ít so với hàng trăm
            # nghìn chunk -> mỗi cluster chứa hàng chục nghìn chunk, tóm tắt
            # rất thô). Vẫn cap ở N_CLUSTERS_CAP_L1 để không tạo quá nhiều
            # cluster (mỗi cluster tốn 1 lần gọi LLM/summarize).
            n_clusters = min(max(3, int(len(chunks) ** 0.5)), N_CLUSTERS_CAP_L1)

            # Chuẩn hoá vector về độ dài 1 để KMeans (dùng khoảng cách Euclid)
            # xấp xỉ đúng hành vi của cosine similarity.
            emb_f32 = embeddings.astype('float32')
            norms = np.linalg.norm(emb_f32, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_normalized = emb_f32 / norms

            clustering = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(2048, len(chunks)),
                n_init=3,
                random_state=42,
            )
            labels = clustering.fit_predict(emb_normalized)

            level_1 = []
            for cluster_idx in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_idx]
                cluster_chunks = [chunks[i] for i in cluster_indices]
                cluster_texts = [c["text"] for c in cluster_chunks]
                # TÁI SỬ DỤNG embeddings đã encode sẵn ở bước trước (tham số
                # `embeddings` truyền vào hàm), KHÔNG gọi embedder.encode()
                # lại lần nữa cho từng cluster. [SỬA LỖI HIỆU NĂNG] bản cũ
                # gọi embedder.encode(cluster_texts) bên trong
                # summarize_cluster_advanced() cho MỖI cluster -> tổng cộng
                # encode lại gần như TOÀN BỘ chunk một lần nữa, tốn thêm
                # ngang bằng cả bước encode ban đầu một cách vô ích.
                cluster_embeddings = embeddings[cluster_indices]

                if len(cluster_texts) > 1:
                    if legal_llm is not None and getattr(legal_llm, "model", None) is not None:
                        summary = summarize_cluster_with_llm(cluster_texts, legal_llm)
                    else:
                        summary = summarize_cluster_advanced(cluster_texts, cluster_embeddings)

                    level_1.append({
                        "id": f"cluster_{cluster_idx}",
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

            completed_levels.add(1)

        except Exception as e:
            logger.warning(f" Clustering failed: {e}")

    # ============ LEVEL 2: Clusters of clusters ============
    # Đặt ngoài block level 1 để khi resume từ checkpoint level 1,
    # level 2 vẫn có thể tiếp tục chạy.
    if len(level_1) > 5 and 2 not in completed_levels:
        logger.info("   Building level 2 clusters...")

        try:
            from sklearn.cluster import MiniBatchKMeans

            # level_1 chỉ có tối đa N_CLUSTERS_CAP_L1 phần tử (là các bản tóm
            # tắt cluster, không phải toàn bộ chunk) nên encode lại ở đây rẻ,
            # không phải vấn đề hiệu năng như ở level 1.
            level_1_texts = [n["text"] for n in level_1]
            level_1_embeddings = embedder.encode(level_1_texts)

            n_clusters_2 = min(max(2, len(level_1) // 3), N_CLUSTERS_CAP_L2)
            l1_f32 = level_1_embeddings.astype('float32')
            l1_norms = np.linalg.norm(l1_f32, axis=1, keepdims=True)
            l1_norms[l1_norms == 0] = 1.0
            l1_normalized = l1_f32 / l1_norms
            clustering_2 = MiniBatchKMeans(
                n_clusters=n_clusters_2,
                batch_size=min(2048, len(level_1)),
                n_init=3,
                random_state=42,
            )
            labels_2 = clustering_2.fit_predict(l1_normalized)

            level_2 = []
            for cluster_idx in range(n_clusters_2):
                cluster_indices = [i for i, label in enumerate(labels_2) if label == cluster_idx]
                cluster_nodes = [level_1[i] for i in cluster_indices]
                cluster_texts = [n["text"] for n in cluster_nodes]
                cluster_embeddings_2 = level_1_embeddings[cluster_indices]

                if len(cluster_texts) > 1:
                    if legal_llm is not None and getattr(legal_llm, "model", None) is not None:
                        summary = summarize_cluster_with_llm(cluster_texts, legal_llm)
                    else:
                        summary = summarize_cluster_advanced(cluster_texts, cluster_embeddings_2)

                    level_2.append({
                        "id": f"cluster_level2_{cluster_idx}",
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

                completed_levels.add(2)

        except Exception as e:
            logger.warning(f" Clustering level 2 failed: {e}")

    logger.info(f" RAPTOR tree completed: {len(tree['nodes'])} nodes, {len(tree['levels'])} levels")
    return tree


def summarize_cluster_advanced(texts: List[str], embeddings: np.ndarray) -> str:
    """
    Tóm tắt cluster bằng cách lấy các đoạn đại diện (gần tâm cluster nhất).

    [SỬA LỖI HIỆU NĂNG] Tham số thứ 2 trước đây là `embedder` (model) và hàm
    tự gọi `embedder.encode(texts)` lại từ đầu cho mỗi cluster. Giờ nhận
    thẳng `embeddings` (mảng numpy) đã tính sẵn từ bước encode chunk ban đầu
    (được truyền vào qua cluster_indices ở nơi gọi) -> không encode lại.
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    try:
        mean_emb = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)
        
        center_indices = np.argsort(distances)[:min(3, len(texts))]
        center_texts = [texts[i] for i in center_indices]
        
        if len(center_texts) == 1:
            return f"[Tóm tắt {len(texts)} văn bản] {center_texts[0][:300]}"
        
        summary = f"[Tóm tắt {len(texts)} văn bản]\n"
        for i, text in enumerate(center_texts, 1):
            summary += f"({i}) {text[:200]}...\n"
        
        return summary.strip()
        
    except Exception as e:
        logger.warning(f" Advanced summarization failed: {e}")
        return texts[0][:300] + "..."


def summarize_cluster_with_llm(texts: List[str], legal_llm) -> str:
    """
    Tóm tắt cluster bằng Legal 4B LLM
    """
    if not texts:
        return ""
    
    if len(texts) == 1:
        return texts[0]
    
    try:
        sample_texts = texts[:5]
        combined_text = "\n---\n".join([t[:300] for t in sample_texts])
        
        prompt = f"""Tóm tắt các văn bản pháp luật sau thành một đoạn ngắn gọn (tối đa 100 từ), giữ nguyên các số hiệu điều luật quan trọng và thuật ngữ pháp lý:

{combined_text}

Tóm tắt:"""
        
        response = legal_llm.generate(prompt, max_length=150)
        summary = response.strip()
        
        if len(summary) > 500:
            summary = summary[:500] + "..."
        
        return f"[LLM Tóm tắt {len(texts)} văn bản] {summary}"
        
    except Exception as e:
        logger.warning(f" LLM summarization failed: {e}")
        return summarize_cluster_advanced(texts, None)


# ============ HÀM PHỤ TRỢ ============

def load_documents(data_dir: str = "data_legalir") -> List[Dict]:
    """Đọc dữ liệu từ thư mục chứa các file context_*.json của BTC.

    Định dạng thật của BTC: MỖI FILE = MỘT văn bản duy nhất, dạng phẳng:
        {"id": 740, "name": "...", "link": "...", "passage": "..."}
    (không phải 1 file = nhiều văn bản, và field nội dung là "passage",
    không phải "content"/"text").

    LƯU Ý: tham số data_dir cần được truyền đúng path dataset trên máy/
    Kaggle của người chạy (vd "/kaggle/input/ten-dataset-cua-ban"). KHÔNG
    auto-dò trong /kaggle/input, vì chỉ dựa vào "có file .json" là quá
    lỏng, dễ chọn nhầm dataset khác nếu notebook gắn nhiều dataset cùng
    lúc. Mỗi người trong nhóm tự sửa data_dir (hoặc truyền qua tham số khi
    gọi build_raptor(...)) cho khớp đúng dataset họ đã Add Data.

    Hàm này đọc cả 3 khả năng để không bị vỡ nếu định dạng thay đổi:
      1) File phẳng 1-văn-bản/file (định dạng thật của BTC) — ưu tiên.
      2) File dạng {"doc_id": {...}, ...} (nhiều văn bản/file).
      3) File dạng list các văn bản.
    """
    documents = []

    if not os.path.exists(data_dir):
        logger.warning(f" Không tìm thấy thư mục dữ liệu: {data_dir}")
        logger.warning("   -> Sửa lại data_dir cho đúng path dataset bạn đã Add Data,")
        logger.warning("      vd load_documents(data_dir=\"/kaggle/input/ten-dataset-cua-ban\")")
        return []

    found_path = data_dir

    # Nếu path chứa 1 thư mục con duy nhất (VD giải nén zip ra subfolder),
    # và thư mục gốc không có sẵn .json, thì đi vào thư mục con đó.
    if not any(f.endswith('.json') for f in os.listdir(found_path)):
        subdirs = [os.path.join(found_path, d) for d in os.listdir(found_path)
                   if os.path.isdir(os.path.join(found_path, d))]
        for sd in subdirs:
            if any(f.endswith('.json') for f in os.listdir(sd)):
                found_path = sd
                break

    logger.info(f" Đọc dữ liệu từ: {found_path}")

    json_files = [f for f in os.listdir(found_path) if f.endswith('.json')]
    n_empty = 0

    def _norm_id(x):
        return str(x)

    for filename in tqdm(json_files, desc="Loading files"):
        filepath = os.path.join(found_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Case 1: 1 file = 1 văn bản phẳng (định dạng thật của BTC)
            if isinstance(data, dict) and (
                "passage" in data or "content" in data or "text" in data
            ) and "id" in data:
                content = data.get("passage", "") or data.get("content", "") or data.get("text", "")
                if not content:
                    n_empty += 1
                    continue
                title = data.get("name", data.get("title", "Văn bản"))
                documents.append({
                    "id": _norm_id(data.get("id")),
                    "title": title,
                    "content": content,
                    "type": data.get("type", "legal"),
                    "metadata": data,
                })

            # Case 2: 1 file = nhiều văn bản, dạng {doc_id: {...}}
            elif isinstance(data, dict):
                for doc_id, doc_content in data.items():
                    if not isinstance(doc_content, dict):
                        continue
                    if "question" in doc_content:
                        documents.append({
                            "id": _norm_id(doc_id),
                            "question": doc_content.get("question", ""),
                            "answer": doc_content.get("answer", []),
                            "type": "query"
                        })
                    elif any(k in doc_content for k in ("content", "text", "passage")):
                        title = doc_content.get("title", doc_content.get("name", "Văn bản"))
                        content = doc_content.get("content", doc_content.get("text", doc_content.get("passage", "")))
                        if not content:
                            n_empty += 1
                            continue
                        documents.append({
                            "id": _norm_id(doc_content.get("id", doc_id)),
                            "title": title,
                            "content": content,
                            "type": doc_content.get("type", "legal"),
                            "metadata": doc_content
                        })

            # Case 3: 1 file = list các văn bản
            elif isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title", item.get("name", "Văn bản"))
                    content = item.get("content", item.get("text", item.get("passage", "")))
                    if not content:
                        n_empty += 1
                        continue
                    documents.append({
                        "id": _norm_id(item.get("id", f"doc_{len(documents)}")),
                        "title": title,
                        "content": content,
                        "type": item.get("type", "legal"),
                        "metadata": item
                    })
        except Exception as e:
            logger.warning(f"Lỗi đọc {filename}: {e}")

    logger.info(f" Loaded {len(documents)} documents (bỏ qua {n_empty} văn bản rỗng)")
    return documents


def create_sample_documents() -> List[Dict]:
    """Tạo dữ liệu mẫu để test"""
    return [
        {
            "id": "doc_001",
            "title": "Luật Mẫu 1",
            "content": "Điều 1: Quy định chung. Khoản 1: Phạm vi điều chỉnh... Đoạn 1: Luật này quy định..."
        },
        {
            "id": "doc_002",
            "title": "Luật Mẫu 2",
            "content": "Điều 2: Quyền và nghĩa vụ. Khoản 1: Quyền của người lao động..."
        },
        {
            "id": "doc_003",
            "title": "Luật Mẫu 3",
            "content": "Điều 3: Trách nhiệm. Khoản 1: Trách nhiệm của người sử dụng lao động..."
        }
    ]


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


_NODE_DOC_MAP_CACHE: Optional[Dict[str, str]] = None


def get_node_doc_map() -> Dict[str, str]:
    """
    Trả về dict {node_id: document_id}.

    QUAN TRỌNG: chỉ node level 0 (chunk gốc, VD "115374_a3_c22") mới ứng
    với ĐÚNG MỘT document_id (lưu trong metadata['doc_id']). Node level 1/2
    là cluster tóm tắt gộp chunk từ NHIỀU văn bản khác nhau -> không có 1
    document_id duy nhất, nên KHÔNG được đưa vào map này. Nơi dùng map này
    (hybrid_retrieve, reranker) phải tự loại các node không có trong map.

    Kết quả được cache trong bộ nhớ (module-level) để không phải quét lại
    toàn bộ RAPTOR tree (có thể ~387k node) mỗi lần gọi.
    """
    global _NODE_DOC_MAP_CACHE
    if _NODE_DOC_MAP_CACHE is not None:
        return _NODE_DOC_MAP_CACHE

    nodes = get_raptor_nodes()
    doc_map = {}
    for n in nodes:
        if n.get("level", 0) == 0:
            doc_id = n.get("metadata", {}).get("doc_id")
            if doc_id is not None:
                doc_map[n["id"]] = str(doc_id)

    _NODE_DOC_MAP_CACHE = doc_map
    return doc_map


# ============ KIỂM TRA NHANH ============
if __name__ == "__main__":
    print("Testing member_a.py (optimized version)...")
    tree = build_raptor()
    print(f"Tree nodes: {len(tree.get('nodes', []))}")
    print(f"Levels: {len(tree.get('levels', []))}")
    
    vector_store = build_vector_store()
    if vector_store:
        print(f"Vector store: {vector_store.get('num_nodes', 0)} vectors")
        print(f"Index type: {vector_store.get('index_type', 'unknown')}")
