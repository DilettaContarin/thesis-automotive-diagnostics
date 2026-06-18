"""
retrieval.py
------------
Hybrid retrieval and cross-encoder reranking for the online pipeline.

Retrieval architecture:
  1. Dense search  — cosine similarity over all-mpnet-base-v2 embeddings in ChromaDB
  2. BM25 search   — keyword frequency matching via BM25Okapi
  3. RRF fusion    — modified RRF with k=0 and explicit 0.5/0.5 weighting
  4. Cross-encoder reranking — ms-marco-MiniLM-L-6-v2 joint query-document scoring

RRF variant note:
  Standard RRF uses k=60 (calibrated on large TREC benchmarks). This system
  uses k=0, making scores directly proportional to reciprocal rank: 1/rank.
  Rationale: with <1000 chunks in the knowledge base, top-ranked results from
  each retriever are reliable signals that should be preserved rather than
  dampened. The 0.5 multiplier ensures symmetric weighting between dense and
  sparse retrieval. The optimal k was not empirically tuned — future work.

Session state (managed by pipeline.py):
  This module uses module-level globals for the active vehicle's indexes.
  Call load_vehicle() from pipeline.py before any retrieval function.
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "all-mpnet-base-v2"
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K     = 5
# ─────────────────────────────────────────────────────────────────────────────

# ── Shared model instances (loaded once, reused across queries) ───────────────
_embedder      = None
_cross_encoder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print(f"Loading cross-encoder: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
    return _cross_encoder


# ── Session state (set by load_vehicle in pipeline.py) ───────────────────────
# These are set externally by the pipeline session management.
# Do not modify directly — use load_vehicle() instead.
chunks     = []        # list of chunk dicts from chunks.json
bm25       = None      # BM25Okapi index, rebuilt each session
collection = None      # ChromaDB collection for current vehicle


# ── Search functions ──────────────────────────────────────────────────────────

def dense_search(query, top_k=DEFAULT_TOP_K):
    """
    Semantic search over ChromaDB using all-mpnet-base-v2 embeddings.

    The same model used at indexing time is used here to ensure query and
    document vectors lie in the same 768-dimensional embedding space.
    Similarity is computed as 1 - cosine_distance (ChromaDB returns distances).
    """
    if collection is None:
        raise RuntimeError("No vehicle loaded — call load_vehicle() first")

    embedder = get_embedder()
    qvec     = embedder.encode([query])[0].tolist()
    results  = collection.query(
        query_embeddings = [qvec],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "metadata": m, "score": 1 - s}
        for d, m, s in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def bm25_search(query, top_k=DEFAULT_TOP_K):
    """
    Keyword-frequency search using BM25Okapi.

    BM25 performs exact term matching weighted by term frequency and inverse
    document frequency. Complements dense search for domain-specific terminology
    (e.g. "DPF", "Dualogic") that may not be well-represented in the embedding space.
    """
    if bm25 is None:
        raise RuntimeError("No vehicle loaded — call load_vehicle() first")

    tokens  = query.lower().split()
    scores  = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [
        {
            "text":     chunks[i]["text"],
            "metadata": chunks[i]["metadata"],
            "score":    float(scores[i]),
        }
        for i in top_idx
    ]


def retrieve_all(query, top_k=DEFAULT_TOP_K, timings=None, prefix=""):
    """
    Run dense and BM25 search independently and fuse with modified RRF.

    RRF variant: score = 0.5 * (1/rank_dense) + 0.5 * (1/rank_bm25)
    - k=0 (no smoothing): rank signal is preserved for small corpora
    - 0.5 weighting: symmetric balance between dense and sparse retrieval

    Args:
        query:   the query string (may be raw or enriched)
        top_k:   number of results to return after fusion
        timings: optional dict to record stage latencies {prefix+stage: seconds}
        prefix:  prefix for timing keys ("raw_" or "enriched_")

    Returns:
        dict with keys "bm25", "dense", "rrf" — each a list of result dicts
    """
    _t = timings if timings is not None else {}

    # BM25
    import time
    t0           = time.time()
    bm25_results = bm25_search(query, top_k=top_k * 2)
    _t[f"{prefix}bm25"] = time.time() - t0

    # Dense
    t0            = time.time()
    dense_results = dense_search(query, top_k=top_k * 2)
    _t[f"{prefix}dense"] = time.time() - t0

    # RRF fusion
    t0         = time.time()
    rrf_scores = {}
    for rank, r in enumerate(dense_results):
        key = r["text"][:50]
        rrf_scores[key] = rrf_scores.get(key, {"score": 0, "data": r})
        rrf_scores[key]["score"] += 0.5 * (1 / (rank + 1))
    for rank, r in enumerate(bm25_results):
        key = r["text"][:50]
        if key not in rrf_scores:
            rrf_scores[key] = {"score": 0, "data": r}
        rrf_scores[key]["score"] += 0.5 * (1 / (rank + 1))
    rrf_results = [
        r["data"] for r in sorted(
            rrf_scores.values(), key=lambda x: x["score"], reverse=True
        )
    ][:top_k]
    _t[f"{prefix}rrf"] = time.time() - t0

    return {
        "bm25":  bm25_results[:top_k],
        "dense": dense_results[:top_k],
        "rrf":   rrf_results,
    }


def rerank(query, candidates):
    """
    Re-score a candidate list using cross-encoder joint query-document scoring.

    Unlike the bi-encoder (which encodes query and document independently),
    the cross-encoder concatenates them as a single input:
        [CLS] query [SEP] document [SEP]
    Full bidirectional attention across both texts enables nuanced relevance
    assessment (causality, implicit reference, precise keyword interactions).

    Scores are unbounded logits — higher is more relevant, negatives are valid.

    Args:
        query:      the query string
        candidates: list of result dicts (from retrieve_all["rrf"])

    Returns:
        candidates sorted descending by rerank_score, with scores injected
    """
    cross_encoder = get_cross_encoder()
    pairs         = [(query, c["text"]) for c in candidates]
    scores        = cross_encoder.predict(pairs)

    for i, score in enumerate(scores):
        candidates[i]["rerank_score"] = float(score)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
