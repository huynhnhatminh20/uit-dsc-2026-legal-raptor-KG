"""Fine-tuning on train.json - this is the step that moves recall past ~0.90.

Pipeline:
  1. mine_training_data()  : pick the best gold segment per question (weak
                             supervision) and mine hard negatives from the
                             current retriever's own mistakes.
  2. finetune_embedder()   : contrastive fine-tune of BAAI/bge-m3.
  3. finetune_reranker()   : cross-encoder fine-tune of bge-reranker-v2-m3.

After step 2 you MUST rebuild the index with the new model, because every
passage embedding changes:

    os.environ["LEGALIR_EMBED_MODEL"] = "/kaggle/working/bge-m3-legalir"
    os.environ["LEGALIR_CACHE_DIR"]   = "/kaggle/working/legalir_doc_index_ft"
    get_retriever(force_rebuild=True)

Hold out a validation split BEFORE mining and never train on it, otherwise your
local recall will look far better than the leaderboard.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.std import tqdm

from legalir_retriever import tokenize


# --------------------------------------------------------------------------
# 1. Mining
# --------------------------------------------------------------------------
def mine_training_data(retriever, train: Dict, out_path: str,
                       n_negatives: int = 12, candidate_depth: int = 120,
                       neg_rank_floor: int = 2, neg_rank_ceiling: int = 100,
                       max_queries: Optional[int] = None) -> str:
    """Write JSONL rows: {query, positive, negatives[]}.

    `neg_rank_floor` skips the very top non-gold documents: in this corpus many
    of those are near-duplicate amendments of the gold text, and training
    against them teaches the model noise instead of relevance.
    """
    seg_by_doc: Dict[str, List[int]] = {}
    for i, s in enumerate(retriever.segments):
        seg_by_doc.setdefault(s["doc_id"], []).append(i)

    items = list(train.items())
    if max_queries:
        items = items[:max_queries]

    written = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, item in tqdm(items, desc="Mining hard negatives"):
            query = item["question"]
            gold = set(map(str, item["answer"]))
            gold_segments = [i for d in gold for i in seg_by_doc.get(d, [])]
            if not gold_segments:
                skipped += 1
                continue

            bm, dense = retriever._segment_scores(query)
            combined = bm / (bm.max() or 1.0) + dense / (dense.max() or 1.0)

            local = combined[gold_segments]
            positive = retriever.segments[gold_segments[int(np.argmax(local))]]["text"]

            candidates = retriever.candidate_documents(query, n_docs=candidate_depth,
                                                       segments_per_doc=1)
            negatives = []
            for rank, c in enumerate(candidates, 1):
                if c["id"] in gold or rank < neg_rank_floor or rank > neg_rank_ceiling:
                    continue
                negatives.append(c["texts"][0])
                if len(negatives) >= n_negatives:
                    break
            if len(negatives) < 2:
                skipped += 1
                continue

            f.write(json.dumps({"qid": str(qid), "query": query,
                                "positive": positive, "negatives": negatives},
                               ensure_ascii=False) + "\n")
            written += 1
    print(f"Mined {written} rows, skipped {skipped} -> {out_path}")
    return out_path


def load_mined(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --------------------------------------------------------------------------
# 2. Embedder fine-tuning
# --------------------------------------------------------------------------
def finetune_embedder(mined_path: str, output_dir: str = "/kaggle/working/bge-m3-legalir",
                      base_model: str = "BAAI/bge-m3", epochs: int = 2,
                      batch_size: int = 4, negatives_per_query: int = 4,
                      max_seq_length: int = 512, lr: float = 1e-5,
                      use_lora: bool = False) -> str:
    """Contrastive fine-tune with in-batch plus explicit hard negatives.

    On a single T4, bge-m3 at seq-len 512 fits with batch_size 4 and gradient
    checkpointing.  With ~6.5k mined rows this is roughly 3-5 hours for 2 epochs.
    If you get OOM, lower `negatives_per_query` to 2 before lowering batch_size.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader
    import torch

    rows = load_mined(mined_path)
    model = SentenceTransformer(base_model)
    model.max_seq_length = max_seq_length
    if hasattr(model[0].auto_model, "gradient_checkpointing_enable"):
        model[0].auto_model.gradient_checkpointing_enable()

    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["query", "key", "value", "dense"])
            model[0].auto_model = get_peft_model(model[0].auto_model, cfg)
            print("LoRA enabled")
        except ImportError:
            print("peft not installed; training full weights")

    examples = []
    for r in rows:
        negs = r["negatives"][:negatives_per_query]
        examples.append(InputExample(texts=[r["query"], r["positive"]] + negs))

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)

    warmup = int(len(loader) * epochs * 0.1)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=warmup, optimizer_params={"lr": lr},
              use_amp=torch.cuda.is_available(), show_progress_bar=True,
              output_path=output_dir, checkpoint_save_steps=2000,
              checkpoint_path=str(pathlib.Path(output_dir) / "ckpt"),
              checkpoint_save_total_limit=2)
    model.save(output_dir)
    print("Saved embedder ->", output_dir)
    return output_dir


# --------------------------------------------------------------------------
# 3. Reranker fine-tuning
# --------------------------------------------------------------------------
def finetune_reranker(mined_path: str,
                      output_dir: str = "/kaggle/working/bge-reranker-legalir",
                      base_model: str = "BAAI/bge-reranker-v2-m3", epochs: int = 2,
                      batch_size: int = 8, negatives_per_query: int = 7,
                      max_length: int = 512, lr: float = 6e-6,
                      seed: int = 2026) -> str:
    """Binary cross-encoder fine-tune: gold passage = 1, hard negative = 0.

    The reranker is where the last few points of recall@5 come from, and it is
    much cheaper to train than the embedder because you do not have to re-encode
    the corpus afterwards. Train this first if you are short on time.
    """
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader
    import torch

    rows = load_mined(mined_path)
    rng = random.Random(seed)
    examples = []
    for r in rows:
        examples.append(InputExample(texts=[r["query"], r["positive"]], label=1.0))
        for neg in r["negatives"][:negatives_per_query]:
            examples.append(InputExample(texts=[r["query"], neg], label=0.0))
    rng.shuffle(examples)

    model = CrossEncoder(base_model, num_labels=1, max_length=max_length,
                         trust_remote_code=True)
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()

    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True)
    model.fit(train_dataloader=loader, epochs=epochs,
              warmup_steps=int(len(loader) * epochs * 0.1),
              optimizer_params={"lr": lr},
              use_amp=torch.cuda.is_available(),
              output_path=output_dir, show_progress_bar=True)
    model.save(output_dir)
    print("Saved reranker ->", output_dir)
    return output_dir


# --------------------------------------------------------------------------
# Helper: reproducible train/validation split
# --------------------------------------------------------------------------
def split_train(train: Dict, n_val: int = 700, seed: int = 2026) -> Tuple[Dict, Dict]:
    ids = list(train)
    random.Random(seed).shuffle(ids)
    val_ids = set(ids[:n_val])
    val = {k: train[k] for k in ids[:n_val]}
    fit = {k: train[k] for k in ids[n_val:]}
    return fit, val
