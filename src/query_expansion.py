"""
Query Expansion via LLM.

Generates alternative query formulations to improve recall
for short or ambiguous queries. Uses the local LLM (Qwen3.6).
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import urllib.request
from typing import List

logger = logging.getLogger(__name__)

# In-memory expansion cache: query_hash -> (expansions, timestamp)
_expansion_cache: dict = {}
_CACHE_TTL_SECONDS = 3600  # 1 hour


async def expand_query(
    query: str,
    gateway_url: str = "http://127.0.0.1:8080",
    max_expansions: int = 3,
    min_query_length: int = 50,
    skip_for_realtime: bool = True,
    intent: str = None,
) -> List[str]:
    """
    Expand a query using the local LLM.

    Only expands short queries (< min_query_length chars).
    Skips expansion for REALTIME intent queries (latency-sensitive).
    Returns list including original + expansions.
    """
    if len(query) > min_query_length:
        logger.debug(f"Query too long ({len(query)} chars), skipping expansion")
        return [query]
    enabled = os.getenv("QUERY_EXPANSION_ENABLED", "false").lower() == "true"
    if not enabled:
        return [query]
    if skip_for_realtime and intent == "realtime":
        logger.debug("Skipping expansion for REALTIME query")
        return [query]
    logger.info(f"Expanding query: '{query}'")

    # Check cache
    query_hash = hashlib.md5(query.encode()).hexdigest()
    if query_hash in _expansion_cache:
        cached, ts = _expansion_cache[query_hash]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return [query] + cached

    # Generate expansions via LLM (stdlib urllib, runs in thread)
    try:
        prompt = (
            f"Generate {max_expansions} alternative search queries that would find "
            f"the same information as this query, using different words and phrasing. "
            f"Return ONLY a JSON array of strings, no explanation.\n\n"
            f'Query: "{query}"'
        )

        payload = json.dumps({
            "model": os.getenv("QUERY_EXPANSION_MODEL", "qwen3.6-35b-a3b"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 200,
        }).encode()

        req = urllib.request.Request(
            f"{gateway_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        loop = asyncio.get_event_loop()
        resp_data = await loop.run_in_executor(
            None,
            lambda: urllib.request.urlopen(req, timeout=5).read()
        )
        data = json.loads(resp_data)
        content = data["choices"][0]["message"]["content"]

        # Strip <think/> tags from reasoning models
        content = re.sub(r'<think[^>]*>.*?</think\s*>', '', content, flags=re.DOTALL).strip()
        # Strip markdown code fences
        content = re.sub(r'^```(?:json)?\s*\n?', '', content)
        content = re.sub(r'\n?```\s*$', '', content)
        content = content.strip()

        # Parse expansions
        try:
            expansions = json.loads(content)
        except json.JSONDecodeError:
            expansions = re.findall(r'"([^"]{5,100})"', content)
            logger.info(f"JSON parse fallback: extracted {len(expansions)} from {content[:200]}")

        if not isinstance(expansions, list):
            expansions = [str(expansions)]
        expansions = [str(e).strip() for e in expansions[:max_expansions] if e and len(e) > 3]

        # Cache
        _expansion_cache[query_hash] = (expansions, time.time())
        # Evict old entries
        now = time.time()
        expired = [k for k, (_, ts) in _expansion_cache.items() if now - ts > _CACHE_TTL_SECONDS]
        for k in expired:
            del _expansion_cache[k]

        logger.info(f"Query expanded '{query}' -> {len(expansions)} variants: {expansions}")
        return [query] + expansions

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]
