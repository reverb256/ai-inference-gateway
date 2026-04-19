"""
Hybrid Search - RAG + SearXNG

Combines local knowledge base (RAG) with web search (SearXNG) for comprehensive results.

Features:
- Dual-source search (local vector DB + web)
- Intelligent result merging and deduplication
- Re-ranking based on query relevance and freshness
- Source prioritization (local vs. web)
- Unified response format
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse

from ai_inference_gateway.searxng_integration import SearxngIntegration
from ai_inference_gateway.query_expansion import expand_query
from ai_inference_gateway.search_cache import get_cached, set_cache

try:
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        QueryIntent,
    )
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False

logger = logging.getLogger(__name__)

# Intent → source weight multipliers
_INTENT_WEIGHTS = {
    QueryIntent.REALTIME: {"rag": 0.0, "web": 1.0},
    QueryIntent.CODE: {"rag": 1.5, "web": 0.8},
    QueryIntent.FACTUAL: {"rag": 1.5, "web": 0.8},
    QueryIntent.PROCEDURAL: {"rag": 1.3, "web": 0.9},
    QueryIntent.COMPARATIVE: {"rag": 1.0, "web": 1.0},
    QueryIntent.CONTEXTUAL: {"rag": 1.3, "web": 0.7},
    QueryIntent.UNKNOWN: {"rag": 1.0, "web": 1.0},
}


class HybridSearchEngine:
    """
    Hybrid search engine combining RAG and SearXNG.

    Provides unified search interface that:
    1. Searches local knowledge base (if available)
    2. Searches web via SearXNG
    3. Merges and deduplicates results
    4. Re-ranks based on relevance and freshness
    """

    def __init__(
        self,
        searxng: SearxngIntegration,
        rag_search: Optional[Any] = None
    ):
        self.searxng = searxng
        self.rag_search = rag_search
        self._reranker = None
        self._reranker_loaded = False

    async def search(
        self,
        query: str,
        max_results: int = 10,
        use_rag: bool = True,
        use_web: bool = True,
        collection: str = "default",
        rerank: bool = True,
        time_range: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search combining RAG and SearXNG.

        Args:
            query: Search query
            max_results: Maximum total results to return
            use_rag: Whether to search local knowledge base
            use_web: Whether to search web via SearXNG
            collection: RAG collection name
            rerank: Whether to re-rank combined results
            time_range: Optional time filter for web search

        Returns:
            Dict with:
            - results: Merged and ranked results
            - sources: Breakdown of results by source
            - metadata: Search metadata and timing
        """
        start_time = datetime.now()
        all_results = []
        source_counts = {"rag": 0, "web": 0}

        # Semantic cache lookup — return instantly if hit
        cached_result = await get_cached(query)
        if cached_result is not None:
            cached_result["metadata"]["cache_hit"] = cached_result.pop("_cache_hit", "exact")
            cached_result["metadata"]["cache_age_seconds"] = cached_result.pop("_cache_age_seconds", 0)
            if "_cache_similarity" in cached_result:
                cached_result["metadata"]["cache_similarity"] = cached_result.pop("_cache_similarity")
            return cached_result

        # Classify query intent for source weighting
        routing_intent = None
        intent_weights = {"rag": 1.0, "web": 1.0}
        if ROUTER_AVAILABLE:
            try:
                # Use classify directly with pre-compiled regex patterns
                query_lower = query.lower().strip()
                patterns = SemanticRouter._get_compiled_patterns()
                intent_scores = {}
                for intent, regex_list in patterns.items():
                    score = sum(1 for regex in regex_list if regex.search(query))
                    if score > 0:
                        intent_scores[intent] = score

                if intent_scores:
                    routing_intent = max(intent_scores, key=intent_scores.get)
                    weights = _INTENT_WEIGHTS.get(routing_intent, {"rag": 1.0, "web": 1.0})
                    intent_weights = weights
                    confidence = min(0.9, max(intent_scores.values()) * 0.2)
                    logger.info(
                        f"Query intent: {routing_intent.value} "
                        f"(confidence: {confidence:.2f}), "
                        f"weights: rag={weights['rag']:.1f} web={weights['web']:.1f}"
                    )
            except Exception as e:
                logger.warning(f"Semantic routing failed: {e}")

        # Apply intent-based source overrides
        effective_use_rag = use_rag and intent_weights["rag"] > 0
        effective_use_web = use_web and intent_weights["web"] > 0

        # Query expansion for short queries (skip for REALTIME intent)
        intent_str = routing_intent.value if routing_intent else None
        query_variants = await expand_query(query, intent=intent_str)

        # Adaptive local-first: run RAG first, conditionally run web
        rag_confidence = 0.0
        rag_result = None
        web_result = None
        all_expansion_results = []

        # Search all query variants (merge results from each)
        for variant in query_variants:
            variant_rag = None
            variant_web = None

            if effective_use_rag and self.rag_search:
                rag_task = self._search_rag(variant, collection, rerank)
                sr = await asyncio.gather(rag_task, return_exceptions=True)
                variant_rag = sr[0] if sr else None
                if isinstance(variant_rag, Exception):
                    variant_rag = None

            if effective_use_web and variant_rag is not None:
                top = variant_rag.get("results", [{}])[0] if variant_rag.get("results") else {}
                conf = top.get("dense_score", top.get("score", 0))
                intent_val = routing_intent.value if routing_intent else ""
                if intent_val != "realtime" and conf >= 0.7:
                    # High confidence — skip web for this and remaining variants
                    effective_use_web = False

            if effective_use_web:
                web_task = self._search_web(variant, time_range, max_results)
                wr = await asyncio.gather(web_task, return_exceptions=True)
                variant_web = wr[0] if wr else None
                if isinstance(variant_web, Exception):
                    variant_web = None

            # Collect results from this variant
            if variant_rag and variant_rag.get("results"):
                for r in variant_rag["results"]:
                    r["source"] = "rag"
                    r["source_type"] = "local_knowledge_base"
                    r["query_variant"] = variant
                all_expansion_results.extend(variant_rag["results"])

            if variant_web and variant_web.get("results"):
                for r in variant_web["results"]:
                    r["source"] = "web"
                    r["source_type"] = "searxng"
                    r["query_variant"] = variant
                all_expansion_results.extend(variant_web["results"])

            # Use first variant's RAG result for confidence tracking
            if rag_result is None and variant_rag and variant_rag.get("results"):
                rag_result = variant_rag
                top_rag = variant_rag["results"][0]
                rag_confidence = top_rag.get("dense_score", top_rag.get("score", 0))

            # Stop expanding if we already have enough results
            if len(all_expansion_results) >= max_results * 2:
                break

        all_results = all_expansion_results

        # Multi-hop: extract [[wiki-links]] from RAG results and follow them
        wiki_links = self._extract_wiki_links(all_results)
        if wiki_links and effective_use_rag and self.rag_search:
            for link_query in wiki_links:
                try:
                    hop_result = await self._search_rag(link_query, collection, False)
                    if hop_result and hop_result.get("results"):
                        for r in hop_result["results"]:
                            r["source"] = "rag"
                            r["source_type"] = "local_knowledge_base"
                            r["hop_origin"] = link_query
                        all_results.extend(hop_result["results"])
                        logger.info(f"Multi-hop: followed [[{link_query}]] -> {len(hop_result['results'])} results")
                except Exception as e:
                    logger.debug(f"Multi-hop failed for [[{link_query}]]: {e}")

        # Count sources
        for r in all_results:
            src = r.get("source", "")
            if src in source_counts:
                source_counts[src] += 1

        # Deduplicate results
        deduped_results = self._deduplicate_results(all_results)

        # Re-rank if requested
        if rerank:
            ranked_results = self._rerank_results(deduped_results, query)
        else:
            ranked_results = deduped_results

        # Synthesize LLM-ready context from results
        synthesized_context = self._synthesize_context(ranked_results, query, routing_intent)

        # Limit to max_results
        final_results = ranked_results[:max_results]

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        response = {
            "results": final_results,
            "sources": source_counts,
            "context": synthesized_context,
            "metadata": {
                "query": query,
                "total_found": len(deduped_results),
                "returned": len(final_results),
                "duration_ms": round(duration_ms, 2),
                "used_rag": effective_use_rag and self.rag_search is not None,
                "used_web": effective_use_web,
                "reranked": rerank,
                "routing_intent": routing_intent.value if routing_intent else None,
                "intent_weights": intent_weights,
                "rag_confidence": round(rag_confidence, 4),
                "timestamp": end_time.isoformat(),
            }
        }

        # Cache the result for future semantic lookups
        await set_cache(query, response)

        return response

    async def _search_rag(
        self,
        query: str,
        collection: str,
        rerank: bool
    ) -> Dict[str, Any]:
        """Search local knowledge base via RAG."""
        try:
            result = await self.rag_search.search(
                query=query,
                collection=collection,
                top_k=10,
                rerank=rerank
            )
            return result
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            raise

    async def _search_web(
        self,
        query: str,
        time_range: Optional[str],
        max_results: int
    ) -> Dict[str, Any]:
        """Search web via SearXNG."""
        try:
            result = await self.searxng.search(
                query=query,
                category="general",
                max_results=max_results,
                time_range=time_range,
                use_cache=True,
                learning_enabled=True
            )
            return result
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            raise

    def _deduplicate_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate results based on URL/title similarity.

        Returns:
            Deduplicated results with dedup_count tracking
        """
        seen_urls = set()
        seen_titles = set()
        deduped = []

        for result in results:
            url = result.get("url", "")
            title = result.get("title", "").lower().strip()

            # Check URL match
            if url and url in seen_urls:
                continue

            # Check title similarity
            if title in seen_titles:
                continue

            # Track and add
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)

            result["dedup_count"] = result.get("dedup_count", 0) + 1
            deduped.append(result)

        return deduped

    def _load_reranker(self):
        """Lazy-load cross-encoder reranker (once)."""
        if self._reranker_loaded:
            return
        self._reranker_loaded = True
        import os
        if os.getenv("RERANKER_ENABLED", "false").lower() != "true":
            logger.info("Cross-encoder reranker disabled (RERANKER_ENABLED != true)")
            return
        try:
            from sentence_transformers import CrossEncoder
            model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-base")
            logger.info(f"Loading cross-encoder reranker: {model}")
            self._reranker = CrossEncoder(model)
            logger.info("Cross-encoder reranker loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load cross-encoder reranker: {e}")

    def _rerank_cross_encoder(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """Rerank using cross-encoder (neural scoring)."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            # Get text content for each result
            pairs = []
            for r in results:
                text = r.get("title", "")
                content = r.get("content", r.get("snippet", ""))
                if content and content != text:
                    text = f"{text} {content}"
                pairs.append((query, text[:500]))  # Truncate to 500 chars

            # run_in_executor returns a Future — but this is called from
            # an async context, so we need to handle it synchronously here.
            # Use the loop's run_until_complete on a separate thread to avoid
            # "thread already running" errors.
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._reranker.predict, pairs)
                scores = future.result(timeout=30)

            for result, score in zip(results, scores):
                # Apply temporal decay: multiply by exp(-lambda * days_old)
                decay = self._temporal_decay(result)
                adjusted = float(score) * decay
                result["reranker_score"] = adjusted
                result["reranker_score_raw"] = float(score)
                result["temporal_decay"] = round(decay, 4)
                result["rerank_method"] = "cross_encoder"

            results.sort(key=lambda x: x.get("reranker_score", 0), reverse=True)
            logger.info(f"Cross-encoder reranked {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"Cross-encoder reranking failed: {e}, falling back to heuristic")
            return results

    def _temporal_decay(self, result: Dict[str, Any], half_life_days: float = 180.0) -> float:
        """
        Compute temporal decay multiplier: exp(-lambda * days_old).

        Args:
            result: Search result dict (may have date/timestamp metadata)
            half_life_days: Days for score to decay to 50% (default 180 = 6 months)

        Returns:
            Decay multiplier in [0, 1]. Returns 1.0 if no date found (no penalty).
        """
        import math
        lam = math.log(2) / half_life_days  # decay constant

        # Try to extract date from various metadata fields
        date_str = (
            result.get("date") or
            result.get("publishedDate") or
            result.get("published_date") or
            result.get("metadata", {}).get("date") or
            result.get("metadata", {}).get("timestamp") or
            ""
        )
        if not date_str:
            return 1.0  # no date = no decay

        try:
            # Parse ISO date (full or date-only)
            from datetime import timezone
            date_str = date_str[:19]  # truncate to YYYY-MM-DDTHH:MM:SS
            if "T" in date_str:
                doc_date = datetime.fromisoformat(date_str).replace(tzinfo=None)
            else:
                doc_date = datetime.strptime(date_str[:10], "%Y-%m-%d")

            days_old = (datetime.now() - doc_date).days
            if days_old <= 0:
                return 1.0  # future docs get no penalty
            decay = math.exp(-lam * days_old)
            return max(0.1, min(1.0, decay))  # floor at 10%
        except (ValueError, TypeError):
            return 1.0

    def _extract_wiki_links(self, results: List[Dict[str, Any]]) -> List[str]:
        """Extract [[wiki-links]] from RAG results for multi-hop retrieval."""
        import re
        links = set()
        for r in results:
            if r.get("source") != "rag":
                continue
            content = r.get("content", "")
            title = r.get("title", "")
            text = f"{title} {content}"
            found = re.findall(r'\[\[([^\]]+)\]\]', text)
            links.update(link.strip() for link in found if len(link.strip()) > 2)
        return list(links)[:5]  # cap at 5 hops to avoid explosion

    def _synthesize_context(
        self,
        results: List[Dict[str, Any]],
        query: str,
        intent: Optional[Any] = None,
    ) -> str:
        """
        Synthesize search results into LLM-ready context string.

        Formats results by source type with attribution, tailored to the
        detected query intent. This is the output that downstream LLM calls
        or agents can consume directly.
        """
        if not results:
            return ""

        intent_val = intent.value if intent else "unknown"

        # Group results by source
        rag_results = [r for r in results if r.get("source") == "rag"]
        web_results = [r for r in results if r.get("source") == "web"]

        parts = []

        # Header based on intent
        headers = {
            "realtime": "Current Information Retrieved:",
            "code": "Code Examples and Implementations:",
            "procedural": "Step-by-Step References:",
            "factual": "Factual Information Retrieved:",
            "comparative": "Comparisons and Alternatives:",
            "contextual": "Contextual Background:",
        }
        parts.append(headers.get(intent_val, "Search Results:"))

        # Format RAG results (local knowledge)
        if rag_results:
            parts.append(f"\n--- Local Knowledge ({len(rag_results)} results) ---")
            for i, r in enumerate(rag_results[:7], 1):
                title = r.get("title", "Untitled")
                content = r.get("content", "")[:300]
                score = r.get("reranker_score", r.get("dense_score", r.get("score", 0)))
                hop = r.get("hop_origin")
                hop_tag = f" [via [[{hop}]]]" if hop else ""
                parts.append(f"[{i}] {title} (relevance: {score:.2f}){hop_tag}")
                parts.append(f"    {content}")

        # Format web results
        if web_results:
            parts.append(f"\n--- Web Sources ({len(web_results)} results) ---")
            for i, r in enumerate(web_results[:5], 1):
                title = r.get("title", "Untitled")
                url = r.get("url", "")
                snippet = r.get("content", r.get("snippet", ""))[:200]
                parts.append(f"[{i}] {title}")
                if url:
                    parts.append(f"    {url}")
                parts.append(f"    {snippet}")

        # Source attribution
        sources = set()
        for r in rag_results:
            sources.add(f"brain-wiki ({len(rag_results)})")
            break
        for r in web_results:
            domain = urlparse(r.get("url", "")).netloc
            if domain:
                sources.add(domain)
        if sources:
            parts.append(f"\nSources: {', '.join(sorted(sources))}")

        return "\n".join(parts)

    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using RRF-inspired weighted scoring.

        If cross-encoder reranker is available, uses neural reranking.
        Falls back to heuristic text-matching scoring.

        Ranking factors (heuristic fallback):
        1. Original semantic/search score (normalized) — 50%
        2. Source priority (RAG > web) — 20%
        3. Text relevance to query — 20%
        4. Freshness (web results) — 10%

        Returns:
            Re-ranked results list
        """
        # Try cross-encoder reranker first
        self._load_reranker()
        if self._reranker is not None and len(results) > 1:
            return self._rerank_cross_encoder(results, query)

        # Fallback: heuristic scoring
        query_lower = query.lower()

        # Normalize scores across all results so we can compare RAG vs web
        # Use dense_score (cosine similarity) for RAG results if available,
        # otherwise fall back to the RRF score
        all_scores = []
        for r in results:
            ds = r.get("dense_score")
            if ds and ds > 0:
                all_scores.append(ds)
            else:
                s = r.get("score", 0)
                if s > 0:
                    all_scores.append(s)
        max_score = max(all_scores) if all_scores else 1.0

        def calculate_score(result: Dict[str, Any]) -> float:
            score = 0.0

            # 1. Original semantic/search score (50%) — normalized to [0, 1]
            # Use dense_score (cosine sim) for RAG, score for web
            if result.get("source") == "rag" and result.get("dense_score"):
                original = result["dense_score"]
            else:
                original = result.get("score", 0)
            if original > 0 and max_score > 0:
                score += (original / max_score) * 0.5

            # 2. Source priority (20%) — RAG gets higher base
            source = result.get("source", "")
            if source == "rag":
                score += 0.2
            else:
                score += 0.1

            # 3. Text relevance (20%)
            title = result.get("title", "").lower()
            content = result.get("content", result.get("snippet", "")).lower()

            # Exact phrase match
            if query_lower in title:
                score += 0.15
            elif any(word in title for word in query_lower.split() if len(word) > 3):
                score += 0.10

            if query_lower in content:
                score += 0.05

            # 4. Freshness for web results (10%)
            if source == "web":
                current_year = datetime.now().year
                text = f"{title} {content}"
                for year in [str(current_year), str(current_year - 1)]:
                    if year in text:
                        score += 0.1
                        break

            return score

        # Calculate scores and sort
        for result in results:
            result["rerank_score"] = calculate_score(result)

        ranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return ranked

    async def search_with_progressive_refinement(
        self,
        query: str,
        max_iterations: int = 3,
        min_results: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Progressive search that refines query if initial results are insufficient.

        Args:
            query: Initial search query
            max_iterations: Maximum refinement iterations
            min_results: Minimum results desired
            **kwargs: Additional arguments for search()

        Returns:
            Search results with refinement metadata
        """
        current_query = query
        all_results = []
        refinement_history = []

        for iteration in range(max_iterations):
            # Perform search
            result = await self.search(
                query=current_query,
                **kwargs
            )

            results = result.get("results", [])
            all_results.extend(results)
            refinement_history.append({
                "iteration": iteration + 1,
                "query": current_query,
                "result_count": len(results),
            })

            # Check if we have enough results
            if len(results) >= min_results:
                break

            # Refine query for next iteration
            if len(results) == 0:
                # No results, broaden search
                current_query = self._broaden_query(current_query)
            else:
                # Some results, try related terms
                current_query = self._refine_query(current_query, results[:3])

        # Deduplicate final results
        final_results = self._deduplicate_results(all_results)

        return {
            "results": final_results[:kwargs.get("max_results", 10)],
            "refinement_history": refinement_history,
            "total_iterations": len(refinement_history),
            "metadata": {
                **result.get("metadata", {}),
                "progressive_refinement": True,
            }
        }

    def _broaden_query(self, query: str) -> str:
        """Broaden search query to get more results."""
        # Remove specific terms, keep general ones
        words = query.lower().split()

        # Remove restrictive words
        restrictive = ["the", "a", "an", "exact", "specific", "precise"]
        broad_words = [w for w in words if w not in restrictive and len(w) > 2]

        # Return first 2-3 meaningful words
        return " ".join(broad_words[:3])

    def _refine_query(self, query: str, sample_results: List[Dict]) -> str:
        """Refine query based on sample results."""
        # Extract key terms from results
        terms = set()

        for result in sample_results:
            title = result.get("title", "").lower()
            content = result.get("content", result.get("snippet", "")).lower()

            # Extract significant words (length > 4, not common)
            words = title.split() + content.split()
            significant = [
                w for w in words
                if len(w) > 4 and w not in {
                    "this", "that", "with", "from", "they", "have", "been"
                }
            ]
            terms.update(significant[:3])

        # Add 1-2 new terms to original query
        new_terms = list(terms)[:2]
        if new_terms:
            return f"{query} {' '.join(new_terms)}"

        return query


# Global instance
_hybrid_search_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search(
    searxng: SearxngIntegration,
    rag_search: Optional[Any] = None
) -> HybridSearchEngine:
    """Get or create global hybrid search engine."""
    global _hybrid_search_engine
    if _hybrid_search_engine is None:
        _hybrid_search_engine = HybridSearchEngine(searxng, rag_search)
    return _hybrid_search_engine
