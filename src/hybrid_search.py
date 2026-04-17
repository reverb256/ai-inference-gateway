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

logger = logging.getLogger(__name__)


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

        # Parallel search of RAG and SearXNG
        tasks = []

        if use_rag and self.rag_search:
            tasks.append(self._search_rag(query, collection, rerank))

        if use_web:
            tasks.append(self._search_web(query, time_range, max_results))

        # Execute searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process RAG results
        if use_rag and self.rag_search:
            rag_result = search_results[0] if search_results else None
            if isinstance(rag_result, Exception):
                logger.warning(f"RAG search failed: {rag_result}")
            elif rag_result:
                rag_results = rag_result.get("results", [])
                for result in rag_results:
                    result["source"] = "rag"
                    result["source_type"] = "local_knowledge_base"
                all_results.extend(rag_results)
                source_counts["rag"] = len(rag_results)

        # Process web results
        if use_web:
            web_idx = 1 if (use_rag and self.rag_search) else 0
            web_result = search_results[web_idx] if web_idx < len(search_results) else None
            if isinstance(web_result, Exception):
                logger.warning(f"Web search failed: {web_result}")
            elif web_result:
                web_results = web_result.get("results", [])
                for result in web_results:
                    result["source"] = "web"
                    result["source_type"] = "searxng"
                all_results.extend(web_results)
                source_counts["web"] = len(web_results)

        # Deduplicate results
        deduped_results = self._deduplicate_results(all_results)

        # Re-rank if requested
        if rerank:
            ranked_results = self._rerank_results(deduped_results, query)
        else:
            ranked_results = deduped_results

        # Limit to max_results
        final_results = ranked_results[:max_results]

        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000

        return {
            "results": final_results,
            "sources": source_counts,
            "metadata": {
                "query": query,
                "total_found": len(deduped_results),
                "returned": len(final_results),
                "duration_ms": round(duration_ms, 2),
                "used_rag": use_rag and self.rag_search is not None,
                "used_web": use_web,
                "reranked": rerank,
                "timestamp": end_time.isoformat(),
            }
        }

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

    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results based on multiple factors.

        Ranking factors:
        1. Source priority (RAG > web for local knowledge)
        2. Text similarity to query
        3. Freshness (for web results)
        4. Quality score (if available)

        Returns:
            Re-ranked results list
        """
        query_lower = query.lower()

        def calculate_score(result: Dict[str, Any]) -> float:
            score = 0.0

            # 1. Source priority (40% of score)
            source = result.get("source", "")
            if source == "rag":
                score += 0.4  # Prioritize local knowledge
            else:  # web
                score += 0.2

            # 2. Text similarity (40% of score)
            title = result.get("title", "").lower()
            content = result.get("content", result.get("snippet", "")).lower()

            # Exact phrase match in title
            if query_lower in title:
                score += 0.3
            # Word matches in title
            elif any(word in title for word in query_lower.split() if len(word) > 3):
                score += 0.2

            # Content matches
            if query_lower in content:
                score += 0.1

            # 3. Freshness for web results (10% of score)
            if source == "web":
                current_year = datetime.now().year
                text = f"{title} {content}"
                for year in [str(current_year), str(current_year - 1)]:
                    if year in text:
                        score += 0.1
                        break

            # 4. Existing quality score (10% of score)
            if "quality_score" in result:
                score += result["quality_score"] * 0.1
            elif "score" in result:
                score += result["score"] * 0.1

            return score

        # Calculate scores and sort
        for result in results:
            result["rerank_score"] = calculate_score(result)

        # Sort by re-rank score
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
