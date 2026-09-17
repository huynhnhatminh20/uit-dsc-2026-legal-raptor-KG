"""High-recall document retrieval for UIT DSC 2026 LegalIR.

This deliberately indexes *only level-0 legal passages* and aggregates evidence
at document level before reranking.  It is a replacement for the RAPTOR/graph
route; the evaluation target is a document ID, not a chunk or a cluster summary.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import pickle
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
# NOTE: intentionally tqdm.std, not tqdm.auto. In Jupyter/Kaggle, tqdm.auto
# renders as an ipywidget progress bar, which draws directly in the notebook
# UI and writes nothing to stdout — so it is completely invisible in a text
# log (e.g. Kaggle's "Log" tab, or output piped/saved to a file). tqdm.std is
# a plain text bar (uses \r to update in place) that always shows up there.
from tqdm.std import tqdm

LOG = logging.getLogger("legalir")
EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Long enough to retain a legal provision; a cap prevents one exceptionally
# long source document from dominating the index and build time.
SEGMENT_CHARS = 1_600
SEGMENT_OVERLAP = 160
MAX_SEGMENTS_PER_DOCUMENT = 48
BM25_DEPTH = 2_500
DENSE_DEPTH = 2_500
# 120 is enough after document-level fusion and halves cross-encoder work
# relative to the old 200-candidate call.  Override only after validation.
DOC_CANDIDATES = int(os.environ.get("LEGALIR_RERANK_CANDIDATES", "120"))
# IVF makes 999 public queries practical with faiss-cpu.  The old code used
# IVF's default nprobe=1; 64 probes is deliberately set below for recall.
IVF_NPROBE = 64


def tokenize(text: str) -> List[str]:
    """Vietnamese-friendly lexical tokens; keep legal numbers and codes."""
    text = text.lower().replace("đ", "đ")
    return re.findall(r"[\wÀ-ỹ]+(?:[-/]\w+)*", text, flags=re.UNICODE)


def _fingerprint(context_dir: str) -> str:
    h = hashlib.sha256()
    for path in sorted(pathlib.Path(context_dir).glob("context_*.json")):
        stat = path.stat()
        h.update(path.name.encode())
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
    # NOTE: IVF_NPROBE is deliberately excluded here. It is a query-time-only
    # search parameter (how many IVF lists to probe); it does not affect the
    # segments, embeddings, or the trained/stored FAISS index. Including it in
    # the fingerprint used to force a full rebuild (re-segment + re-encode all
    # embeddings from scratch) every time nprobe was tuned for recall, even
    # though `self.index.nprobe = ...` already re-applies it correctly on load.
    h.update(f"{EMBED_MODEL}|{SEGMENT_CHARS}|{SEGMENT_OVERLAP}|cap{MAX_SEGMENTS_PER_DOCUMENT}".encode())
    return h.hexdigest()[:20]


def _legal_segments(text: str, title: str) -> Iterable[str]:
    """Lossless, bounded segmentation for heterogeneous legal source files.

    Do not split on generic ``1.`` / ``2.`` markers: in this corpus they also
    occur in dates, citations, tables, and footnotes, which previously exploded
    the corpus above one million segments.  Article boundaries are useful; all
    other long text is covered by overlapping windows.
    """
    clean = re.sub(r"\r\n?", "\n", text)
    clean = re.sub(r"[ \t]+", " ", clean)
    # Keep Điều with the following block.  Do NOT use every numeric clause as a
    # split boundary; it is not a reliable structural signal in these files.
    units = re.split(r"(?im)(?=^\s*Điều\s+\d+[A-Za-z]?\s*[\.:])", clean)
    all_parts = []
    for unit in units:
        unit = unit.strip()
        if len(unit) < 35:
            continue
        for start in range(0, len(unit), SEGMENT_CHARS - SEGMENT_OVERLAP):
            part = unit[start:start + SEGMENT_CHARS].strip()
            if len(part) >= 35:
                all_parts.append(f"[VĂN BẢN: {title}]\n{part}")
            if start + SEGMENT_CHARS >= len(unit):
                break
    # Preserve coverage across the whole document, including its tail, while
    # bounding pathological multi-megabyte contexts.
    if len(all_parts) <= MAX_SEGMENTS_PER_DOCUMENT:
        yield from all_parts
    else:
        positions = np.linspace(0, len(all_parts) - 1, MAX_SEGMENTS_PER_DOCUMENT, dtype=int)
        yield from (all_parts[i] for i in positions)


def _read_documents(context_dir: str) -> List[Tuple[str, str, str]]:
    docs = []
    paths = sorted(pathlib.Path(context_dir).glob("context_*.json"))
    if not paths:
        raise FileNotFoundError(f"Không có context_*.json trong {context_dir}")
    for path in tqdm(paths, desc="Reading contexts"):
        with path.open(encoding="utf-8") as f:
            item = json.load(f)
        doc_id, title, passage = str(item["id"]), item.get("name", ""), item.get("passage", "")
        if passage:
            docs.append((doc_id, title, passage))
    return docs


class LegalIRRetriever:
    def __init__(self, context_dir: str, cache_dir: str = "/kaggle/working/legalir_doc_index",
                 device: str | None = None):
        self.context_dir = context_dir
        self.cache_dir = pathlib.Path(cache_dir)
        self.device = device
        self.segments: List[Dict] = []
        self.index = None
        self.bm25 = None
        self.embedder = None
        self.reranker = None

    @property
    def _meta_path(self): return self.cache_dir / "metadata.pkl"
    @property
    def _index_path(self): return self.cache_dir / "segments.faiss"
    @property
    def _vectors_path(self): return self.cache_dir / "vectors.npy"

    def _encode_resumable(self, texts: List[str]) -> np.ndarray:
        """Encode in durable shards; a stopped Kaggle A run resumes at a shard."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        shard_dir = self.cache_dir / "embedding_shards"
        state_path = shard_dir / "state.json"
        state = {"count": len(texts), "model": EMBED_MODEL, "segment_chars": SEGMENT_CHARS,
                 "segment_overlap": SEGMENT_OVERLAP}
        matching_state = False
        if state_path.exists():
            with state_path.open(encoding="utf-8") as f:
                previous = json.load(f)
            if previous != state:
                # These are incomplete, local temporary artifacts only.
                import shutil
                shutil.rmtree(shard_dir)
            else:
                matching_state = True
        if self._vectors_path.exists() and matching_state:
            return np.load(self._vectors_path, mmap_mode="r")
        shard_dir.mkdir(exist_ok=True)
        if not state_path.exists():
            with state_path.open("w", encoding="utf-8") as f:
                json.dump(state, f)
        shard_size = 768
        batches = (len(texts) + shard_size - 1) // shard_size
        for b in tqdm(range(batches), desc="Encoding passage shards"):
            shard = shard_dir / f"part_{b:05d}.npy"
            if shard.exists():
                continue
            start, end = b * shard_size, min((b + 1) * shard_size, len(texts))
            vector = self.embedder.encode(texts[start:end], batch_size=128, show_progress_bar=False,
                                          normalize_embeddings=True).astype("float32")
            temporary = shard_dir / f"part_{b:05d}.tmp.npy"
            np.save(temporary, vector)
            os.replace(temporary, shard)
        first = np.load(shard_dir / "part_00000.npy", mmap_mode="r")
        temporary = self.cache_dir / "vectors.partial.npy"
        matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32",
                                           shape=(len(texts), first.shape[1]))
        for b in range(batches):
            start = b * shard_size
            values = np.load(shard_dir / f"part_{b:05d}.npy", mmap_mode="r")
            matrix[start:start + len(values)] = values
        matrix.flush()
        del matrix
        os.replace(temporary, self._vectors_path)
        return np.load(self._vectors_path, mmap_mode="r")

    def _adopt_input_checkpoint(self) -> None:
        """Use a completed Kaggle Input checkpoint in place, never copy it.

        Set LEGALIR_CHECKPOINT_DIR to the directory containing metadata.pkl and
        segments.faiss for deterministic selection.  The fallback supports the
        common workflow of uploading `legalir_doc_index/` as a Kaggle Dataset.
        """
        requested = os.environ.get("LEGALIR_CHECKPOINT_DIR")
        candidates = [pathlib.Path(requested)] if requested else []
        root = pathlib.Path("/kaggle/input")
        if not requested and root.exists():
            candidates.extend(root.rglob("legalir_doc_index"))
        for candidate in candidates:
            if candidate and (candidate / "metadata.pkl").is_file() and (candidate / "segments.faiss").is_file():
                LOG.info("Using mounted checkpoint directly: %s", candidate)
                self.cache_dir = candidate
                return

    def build_or_load(self, force_rebuild: bool = False) -> "LegalIRRetriever":
        if not force_rebuild and not self._meta_path.exists():
            self._adopt_input_checkpoint()
        fp = _fingerprint(self.context_dir)
        if not force_rebuild and self._meta_path.exists() and self._index_path.exists():
            with self._meta_path.open("rb") as f:
                saved = pickle.load(f)
            if saved.get("fingerprint") == fp:
                self.segments = saved["segments"]
                self.bm25 = BM25Okapi(saved["tokens"])
                self.index = faiss.read_index(str(self._index_path))
                # Critical for IVF: the FAISS default is nprobe=1, which is
                # recall-hostile.  64 probes is a measured-safe high-recall
                # setting without turning each public query into an exact scan.
                if hasattr(self.index, "nprobe"):
                    self.index.nprobe = min(IVF_NPROBE, self.index.nlist)
                LOG.info("Loaded %d passage segments", len(self.segments))
                return self

        docs = _read_documents(self.context_dir)
        self.segments = []
        for doc_id, title, passage in tqdm(docs, desc="Segmenting documents"):
            generated = list(_legal_segments(passage, title))
            if not generated:
                generated = [f"[VĂN BẢN: {title}]\n{passage[:SEGMENT_CHARS]}"]
            self.segments.extend({"doc_id": doc_id, "text": segment} for segment in generated)
        LOG.info("Built %d segments from %d documents", len(self.segments), len(docs))

        self._load_embedder()
        vectors = self._encode_resumable([x["text"] for x in self.segments])
        # IVF is required here: an exact CPU scan of hundreds of thousands of
        # 1024-d vectors for every one of 999 public questions is needlessly
        # slow. Unlike the old code, nprobe is explicitly raised on both build
        # and reload, so this does not silently degrade to nprobe=1.
        nlist = min(2048, max(128, int(np.sqrt(len(vectors)))))
        quantizer = faiss.IndexFlatIP(vectors.shape[1])
        self.index = faiss.IndexIVFFlat(quantizer, vectors.shape[1], nlist, faiss.METRIC_INNER_PRODUCT)
        train_vectors = vectors if len(vectors) <= 100_000 else vectors[:100_000]
        self.index.train(train_vectors)
        self.index.add(vectors)
        self.index.nprobe = min(IVF_NPROBE, nlist)
        tokens = [tokenize(x["text"]) for x in self.segments]
        self.bm25 = BM25Okapi(tokens, k1=1.25, b=0.62)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Never leave a half-written checkpoint that a later Kaggle session
        # mistakes for a valid A-stage artifact.
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        index_tmp = self.cache_dir / "segments.faiss.tmp"
        meta_tmp = self.cache_dir / "metadata.pkl.tmp"
        faiss.write_index(self.index, str(index_tmp))
        with meta_tmp.open("wb") as f:
            pickle.dump({"fingerprint": fp, "segments": self.segments, "tokens": tokens}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(index_tmp, self._index_path)
        os.replace(meta_tmp, self._meta_path)
        # The completed FAISS index supersedes temporary embeddings.  Shards
        # are retained only until a valid index+metadata pair exists.
        if self._vectors_path.exists():
            self._vectors_path.unlink()
        shard_dir = self.cache_dir / "embedding_shards"
        if shard_dir.exists():
            import shutil
            shutil.rmtree(shard_dir)
        return self

    def _load_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(EMBED_MODEL, device=self.device)

    def _load_reranker(self):
        if self.reranker is None:
            self.reranker = CrossEncoder(RERANK_MODEL, device=self.device, max_length=1024,
                                         trust_remote_code=True)

    @staticmethod
    def _rank(ids: Iterable[int]) -> Dict[int, int]:
        return {idx: rank for rank, idx in enumerate(ids, 1)}

    def candidate_documents(self, question: str, n_docs: int = DOC_CANDIDATES) -> List[Dict]:
        if self.index is None or self.bm25 is None:
            raise RuntimeError("Call build_or_load() first")
        n = len(self.segments)
        bm_depth, dense_depth = min(BM25_DEPTH, n), min(DENSE_DEPTH, n)
        bm_scores = self.bm25.get_scores(tokenize(question))
        # argpartition avoids sorting every segment for every query; only the
        # requested BM25 depth is sorted exactly.
        if bm_depth == n:
            bm_ids = np.argsort(bm_scores)[::-1].tolist()
        else:
            rough = np.argpartition(bm_scores, -bm_depth)[-bm_depth:]
            bm_ids = rough[np.argsort(bm_scores[rough])[::-1]].tolist()
        self._load_embedder()
        q = self.embedder.encode([question], normalize_embeddings=True).astype("float32")
        _, dense = self.index.search(q, dense_depth)
        dense_ids = [int(i) for i in dense[0] if i >= 0]

        # Fuse chunks, then aggregate by document.  Ranking chunks first and
        # de-duplicating only at the end wastes the candidate budget on one doc.
        evidence: Dict[str, Dict] = {}
        for weight, ranks in ((1.0, self._rank(bm_ids)), (1.35, self._rank(dense_ids))):
            for idx, rank in ranks.items():
                seg = self.segments[idx]
                score = weight / (30 + rank)
                entry = evidence.setdefault(seg["doc_id"], {"id": seg["doc_id"], "text": seg["text"], "score": 0.0})
                entry["score"] += score
                # Keep the best lexical/dense evidence as reranker passage.
                if score > entry.get("best", -1):
                    entry["best"] = score
                    entry["text"] = seg["text"]
        return sorted(evidence.values(), key=lambda x: x["score"], reverse=True)[:n_docs]

    def retrieve(self, question: str, max_docs: int = 5, candidate_docs: int = DOC_CANDIDATES) -> List[str]:
        candidates = self.candidate_documents(question, candidate_docs)
        self._load_reranker()
        scores = self.reranker.predict([(question, x["text"]) for x in candidates], batch_size=48,
                                       show_progress_bar=False)
        order = np.argsort(scores)[::-1]
        return [candidates[int(i)]["id"] for i in order[:max_docs]]

    def make_submission(self, questions: Dict, output_path: str, max_docs: int = 4) -> Dict:
        """max_docs=4 is the precision-safe default; never exceed the BTC cap of 5."""
        if not 1 <= max_docs <= 5:
            raise ValueError("BTC only accepts 1..5 document IDs")
        result = {}
        for qid, item in tqdm(questions.items(), desc="Retrieving"):
            question = item["question"] if isinstance(item, dict) else str(item)
            result[str(qid)] = {"answer": self.retrieve(question, max_docs=max_docs)}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return result
