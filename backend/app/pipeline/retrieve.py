"""
retrieve.py
───────────
Hybrid retrieval: BM25 + dense bge-m3 (RRF fusion).
Loads chunks.jsonl and the pre-built embedding index from data/processed/.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from backend.app.config import settings

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
INDEX_DIR = PROCESSED_DIR / "index"

# ── Module-level singletons ───────────────────────────────────────────────
_chunks: list[dict] = []
_bm25 = None
_embedder = None
_embeddings: Optional[np.ndarray] = None


def _load_corpus():
    global _chunks, _bm25, _embeddings

    if _chunks:
        return

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    if not chunks_path.exists():
        logger.error("chunks.jsonl not found – run scripts/ingest.py first")
        return

    with chunks_path.open(encoding="utf-8") as f:
        _chunks = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Retrieval: loaded {len(_chunks)} chunks")

    # BM25
    try:
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in _chunks]
        _bm25 = BM25Okapi(tokenized)
        logger.success("BM25 index built.")
    except ImportError:
        logger.warning("rank-bm25 not installed – BM25 disabled")

    # Dense
    emb_path = INDEX_DIR / "embeddings.npy"
    if emb_path.exists():
        _embeddings = np.load(str(emb_path)).astype("float32")
        # L2-normalise for cosine via dot product
        norms = np.linalg.norm(_embeddings, axis=1, keepdims=True) + 1e-10
        _embeddings = _embeddings / norms
        logger.success(f"Dense index loaded: {_embeddings.shape}")
    else:
        logger.warning("Dense index not found – dense retrieval disabled")


def _load_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    # Suppress tqdm progress bars when running without a TTY (server / nohup / CI).
    # This prevents FlagEmbedding from hanging at startup in non-interactive environments.
    os.environ.setdefault("TQDM_DISABLE", "1" if not os.isatty(1) else "0")

    # NOTE: FlagEmbedding and sentence-transformers are imported lazily here
    # to avoid loading heavy model libraries at server startup when the dense
    # embedder is not pre-loaded (KEEP_EMBED_RESIDENT=false). This is an
    # intentional performance optimisation, not a circular-dependency workaround.
    try:
        from FlagEmbedding import BGEM3FlagModel  # noqa: PLC0415
        # fp16 halves bge-m3's footprint (2.3 → 1.1 GB); in fp32 it pushes Param-1 into swap on 16 GB Macs.
        _embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        logger.success("bge-m3 embedder loaded via FlagEmbedding.")
        return _embedder
    except Exception as exc:
        logger.warning(f"FlagEmbedding unavailable ({exc}), trying sentence-transformers …")

    # Fall back to sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        class _STWrapper:
            def __init__(self, model):
                self._model = model

            def encode(self, texts, **kwargs):
                vecs = self._model.encode(texts, normalize_embeddings=True)
                return {"dense_vecs": np.array(vecs)}

        _embedder = _STWrapper(SentenceTransformer("BAAI/bge-m3"))
        logger.success("bge-m3 embedder loaded via sentence-transformers.")
    except Exception as exc:
        logger.warning(f"Could not load any embedder: {exc}")
    return _embedder


def _bm25_scores(query: str, k: int) -> list[tuple[int, float]]:
    """Return [(chunk_idx, normalised_score), ...]."""
    if _bm25 is None or not _chunks:
        return []
    tokens = query.lower().split()
    raw = _bm25.get_scores(tokens)
    max_score = float(raw.max()) if raw.max() > 0 else 1.0
    top_k = np.argsort(raw)[::-1][:k]
    return [(int(i), float(raw[i]) / max_score) for i in top_k if raw[i] > 0]


def _dense_scores(query: str, k: int) -> list[tuple[int, float]]:
    """Return [(chunk_idx, cosine_sim), ...]."""
    if _embeddings is None or not _chunks:
        return []
    embedder = _load_embedder()
    if embedder is None:
        return []

    q_vec = embedder.encode([query])["dense_vecs"][0].astype("float32")
    q_vec /= (np.linalg.norm(q_vec) + 1e-10)
    sims = (_embeddings @ q_vec).tolist()
    top_k = sorted(enumerate(sims), key=lambda x: -x[1])[:k]
    return [(int(i), float(s)) for i, s in top_k if s > 0]


def _rrf_merge(
    bm25_hits: list[tuple[int, float]],
    dense_hits: list[tuple[int, float]],
    k_rrf: int = 60,
) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion."""
    scores: dict[int, float] = {}
    for rank, (idx, _) in enumerate(bm25_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    for rank, (idx, _) in enumerate(dense_hits):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
    # Normalise
    max_s = max(scores.values()) if scores else 1.0
    merged = sorted(scores.items(), key=lambda x: -x[1])
    return [(idx, s / max_s) for idx, s in merged]


def retrieve(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    mode: str = "hybrid",  # "bm25" | "dense" | "hybrid"
) -> list[dict]:
    """
    Retrieve the most relevant chunks for *query*.

    Returns a list of chunk dicts with an added 'score' key.
    """
    _load_corpus()

    if not _chunks:
        return []

    top_k = top_k or settings.rag_top_k
    min_score = min_score if min_score is not None else settings.rag_min_score

    t0 = time.perf_counter()

    search_k = top_k * 3  # fetch extra then filter by score

    if mode == "bm25":
        hits = _bm25_scores(query, search_k)
    elif mode == "dense":
        hits = _dense_scores(query, search_k)
    else:  # hybrid
        bm25_hits = _bm25_scores(query, search_k)
        dense_hits = _dense_scores(query, search_k)
        hits = _rrf_merge(bm25_hits, dense_hits)

    results = []
    for idx, score in hits[:top_k]:
        if score < min_score:
            continue
        chunk = dict(_chunks[idx])
        chunk["score"] = round(score, 4)
        results.append(chunk)

    latency = (time.perf_counter() - t0) * 1000
    logger.debug(f"Retrieve [{mode}] '{query[:60]}' → {len(results)} chunks ({latency:.0f}ms)")
    return results
