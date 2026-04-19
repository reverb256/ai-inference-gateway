"""
Lightweight Semantic Cache for Hybrid Search.

In-memory cache keyed by query embedding cosine similarity.
When a new query's embedding matches a cached query at >= threshold,
the cached result is returned instantly (zero network calls).

No Redis needed — just Python dict + numpy for cosine similarity.
Falls back gracefully if numpy is unavailable (exact text match only).
"""

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# In-memory store: hash -> (embedding, result, timestamp)
_cache: Dict[str, Tuple[List[float], Dict[str, Any], float]] = {}
_MAX_CACHE_SIZE = 500
_eviction_batch = 50


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _evict_expired(now: float, ttl: int):
    """Remove expired entries."""
    expired = [k for k, (_, _, ts) in _cache.items() if now - ts > ttl]
    for k in expired:
        del _cache[k]


def _evict_oldest():
    """Evict oldest entries when cache is full."""
    if len(_cache) <= _MAX_CACHE_SIZE:
        return
    sorted_keys = sorted(_cache.items(), key=lambda x: x[1][2])  # sort by timestamp
    for k, _ in sorted_keys[:_eviction_batch]:
        del _cache[k]


def _query_hash(query: str) -> str:
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


async def get_cached(
    query: str,
    query_embedding: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Look up a cached hybrid search result.

    If query_embedding is provided, does semantic similarity search.
    Otherwise falls back to exact text match.

    Returns cached result dict (with cache metadata added) or None.
    """
    if os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() != "true":
        return None

    ttl = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
    threshold = float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.95"))

    now = time.time()
    _evict_expired(now, ttl)

    qhash = _query_hash(query)

    # Exact match
    if qhash in _cache:
        emb, result, ts = _cache[qhash]
        if now - ts <= ttl:
            result["_cache_hit"] = "exact"
            result["_cache_age_seconds"] = round(now - ts, 1)
            logger.debug(f"Cache exact hit: '{query[:50]}'")
            return result

    # Semantic match (embedding similarity)
    if query_embedding and len(query_embedding) > 0:
        best_key = None
        best_score = 0.0
        for k, (emb, result, ts) in _cache.items():
            if now - ts > ttl:
                continue
            score = _cosine_sim(query_embedding, emb)
            if score > best_score:
                best_score = score
                best_key = k

        if best_score >= threshold and best_key is not None:
            _, result, ts = _cache[best_key]
            result["_cache_hit"] = "semantic"
            result["_cache_similarity"] = round(best_score, 4)
            result["_cache_age_seconds"] = round(now - ts, 1)
            logger.info(f"Cache semantic hit: '{query[:50]}' sim={best_score:.4f}")
            return result

    return None


async def set_cache(
    query: str,
    result: Dict[str, Any],
    query_embedding: Optional[List[float]] = None,
):
    """
    Store a hybrid search result in cache.

    If query_embedding is provided, stores it for semantic lookup.
    """
    if os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() != "true":
        return

    ttl = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))

    now = time.time()
    qhash = _query_hash(query)

    # Remove cache metadata from result before storing
    clean = {k: v for k, v in result.items() if not k.startswith("_cache_")}

    _cache[qhash] = (query_embedding or [], clean, now)
    _evict_oldest()

    logger.debug(f"Cached result for '{query[:50]}' ({len(_cache)} entries)")


def cache_stats() -> Dict[str, Any]:
    """Return cache statistics."""
    return {
        "entries": len(_cache),
        "max_size": _MAX_CACHE_SIZE,
        "enabled": os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true",
    }
