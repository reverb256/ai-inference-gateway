"""
SearXNG Knowledge Source Adapter for Knowledge Fabric

Provides SearXNG metasearch integration for comprehensive web results.
Enhanced with RAG indexing, similarity search, and clustering capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import httpx

from ..core import (
    KnowledgeChunk,
    KnowledgeResult,
    SourceCapability,
    SourcePriority,
)

logger = logging.getLogger(__name__)

# Optional imports for enhanced features
try:
    from ai_inference_gateway.searxng_rag import SearXNGRAGIndexer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

try:
    from ai_inference_gateway.searxng_clustering import ResultClusterer
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False


@dataclass
class SearXNGKnowledgeSource:
    """
    SearXNG metasearch knowledge source.

    Provides aggregated search results from multiple search engines
    through SearXNG instance.

    Enhanced features:
    - Domain-aware routing
    - Quality scoring
    - Optional RAG indexing
    """
    searxng_url: str = "http://10.4.98.141:7777"  # Kubernetes service
    max_results: int = 5
    timeout: float = 30.0
    name: str = "searxng"
    description: str = "SearXNG metasearch"
    priority: SourcePriority = SourcePriority.MEDIUM
    capabilities: SourceCapability = (
        SourceCapability.REALTIME |
        SourceCapability.FACTUAL |
        SourceCapability.COMPARATIVE
    )
    enabled: bool = True
    # Enhanced features
    enable_domain_routing: bool = True
    enable_quality_scoring: bool = True
    enable_rag_indexing: bool = False
    min_quality_score: float = 0.3
    rag_collection: str = "searxng-results"
    # Injected dependencies (set at runtime)
    _rag_indexer: Optional[Any] = field(default=None, repr=False)
    _searxng_client: Optional[Any] = field(default=None, repr=False)
    # Domain hint for routing
    _domain_hint: Optional[str] = field(default=None, repr=False)

    def _get_domain_hint(self, query: str) -> Optional[str]:
        """Detect domain hint from query."""
        if not self.enable_domain_routing:
            return None

        query_lower = query.lower()

        domain_indicators = {
            "code": ["github", "gitlab", "function", "class", "api", "code", "implement"],
            "research": ["paper", "research", "study", "arxiv", "scholar", "academic"],
            "devops": ["docker", "kubernetes", "deploy", "terraform", "ansible"],
            "data": ["machine learning", "neural", "model", "dataset", "training"],
        }

        for domain, indicators in domain_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                return domain

        return None

    def _score_result_quality(self, result: Dict, domain: str) -> float:
        """Score result quality (0-1)."""
        score = 0.0
        url = result.get("url", "").lower()

        # Trusted domains
        trusted = {
            "code": ["github.com", "gitlab.com", "stackoverflow.com", "docs.rs"],
            "research": ["arxiv.org", "scholar.google", "semanticscholar.org"],
            "devops": ["docker.com", "kubernetes.io", "terraform.io"],
            "data": ["arxiv.org", "kaggle.com", "huggingface.co"],
            "general": ["wikipedia.org", "github.com"],
        }

        if any(t in url for t in trusted.get(domain, trusted["general"])):
            score += 0.4

        # Content length
        content = result.get("content", result.get("snippet", ""))
        if len(content) > 200:
            score += 0.3

        # HTTPS
        if url.startswith("https://"):
            score += 0.2

        # Title has query terms
        title = result.get("title", "").lower()
        if any(word in title for word in domain.split() if len(word) > 3):
            score += 0.1

        return min(score, 1.0)

    async def retrieve(
        self,
        query: str,
        domain: Optional[str] = None,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> KnowledgeResult:
        """
        Execute search via SearXNG API with enhanced features.

        Args:
            query: Search query
            domain: Optional domain hint for routing
            context: Request context for dependency injection

        Returns:
            KnowledgeResult with scored chunks
        """
        import time
        start = time.time()

        sanitized_query = query[:500]

        # Use provided domain or detect from query
        domain = domain or self._domain_hint or self._get_domain_hint(query)

        chunks = []
        metadata = {
            "tool": "SearXNG",
            "engine": "metasearch",
            "domain": domain or "general",
        }

        try:
            # Build request with domain-aware engine selection
            params = {
                "q": sanitized_query,
                "format": "json",
            }

            # Add domain-specific engines if domain routing enabled
            if self.enable_domain_routing and domain:
                domain_engines = {
                    "code": ["github", "gitlab", "stackoverflow"],
                    "research": ["google scholar", "arxiv", "semantic scholar"],
                    "devops": ["docker hub", "github", "stackoverflow"],
                    "data": ["github", "arxiv", "kaggle"],
                }
                engines = domain_engines.get(domain, [])
                if engines:
                    params["engines"] = ",".join(engines)
                    metadata["engines_selected"] = engines

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    self.searxng_url,
                    params=params,
                    headers={
                        "X-Forwarded-For": "10.0.0.102",
                        "X-Real-IP": "10.0.0.102",
                    },
                )

                if response.status_code == 200:
                    data = response.json()

                    if data.get("results"):
                        results = data["results"]

                        # Score and filter results if quality scoring enabled
                        if self.enable_quality_scoring:
                            for result in results:
                                result["quality_score"] = self._score_result_quality(
                                    result, domain or "general"
                                )

                            # Sort by quality score
                            results.sort(
                                key=lambda r: r.get("quality_score", 0), reverse=True
                            )

                            # Filter by minimum quality
                            results = [
                                r
                                for r in results
                                if r.get("quality_score", 0) >= self.min_quality_score
                            ]

                        # Create chunks from top results
                        for idx, result in enumerate(results[: self.max_results]):
                            title = result.get("title", "")
                            snippet = result.get("content", result.get("snippet", ""))
                            url = result.get("url", "")

                            content_text = f"{title}\n{snippet}"

                            chunk = KnowledgeChunk(
                                content=content_text,
                                source=self.name,
                                score=result.get("quality_score", 1.0 - (idx * 0.1)),
                                metadata={
                                    "url": url,
                                    "title": title,
                                    "engine": result.get("engine", ""),
                                    "quality_score": result.get("quality_score", 0.5),
                                    "domain": domain or "general",
                                },
                                capabilities=self.capabilities,
                            )
                            chunks.append(chunk)

                        metadata["total_results"] = len(chunks)
                        metadata["domain_detected"] = domain or "general"

                        # RAG indexing if enabled
                        if self.enable_rag_indexing and self._rag_indexer and context:
                            await self._index_results(query, results[: self.max_results], context)

                else:
                    metadata["error"] = f"HTTP {response.status_code}"
                    logger.warning(f"SearXNG returned status {response.status_code}")

        except httpx.ConnectError as e:
            metadata["error"] = "Cannot connect to SearXNG"
            metadata["error_type"] = "connection_error"
            logger.error(f"SearXNG connection error: {e}")

        except httpx.TimeoutException:
            metadata["error"] = "Request timeout"
            metadata["error_type"] = "timeout"
            logger.error(f"SearXNG timeout after {self.timeout}s")

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            logger.exception(f"SearXNG unexpected error: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )

    async def _index_results(
        self, query: str, results: List[Dict], context: Dict
    ) -> None:
        """Index search results in RAG system."""
        if not self._rag_indexer or not RAG_AVAILABLE:
            return

        try:
            # Get searxng client from context if available
            searxng_client = context.get("searxng_client")
            if not searxng_client:
                return

            await self._rag_indexer.search_and_index(
                query=query,
                searxng_client=searxng_client,
                collection=self.rag_collection,
                max_results=len(results),
            )

            logger.debug(f"Indexed {len(results)} results in RAG")

        except Exception as e:
            logger.error(f"RAG indexing failed: {e}")


def create_searxng_source(
    searxng_url: str = "http://10.4.98.141:7777",  # Kubernetes ClusterIP
    max_results: int = 5,
    enable_domain_routing: bool = True,
    enable_quality_scoring: bool = True,
    enable_rag_indexing: bool = False,
) -> SearXNGKnowledgeSource:
    """Factory function to create SearXNG knowledge source."""
    return SearXNGKnowledgeSource(
        searxng_url=searxng_url,
        max_results=max_results,
        enable_domain_routing=enable_domain_routing,
        enable_quality_scoring=enable_quality_scoring,
        enable_rag_indexing=enable_rag_indexing,
    )


# ========================================================================
# Enhanced Knowledge Sources
# ========================================================================

@dataclass
class SearxngSimilarityKnowledgeSource:
    """
    Knowledge source for finding content similar to a URL or text.

    Uses vector similarity search in Qdrant to find related content
    from previously indexed SearXNG results.
    """
    searxng_url: str = "http://10.4.98.141:7777"  # Kubernetes service
    max_results: int = 5
    similarity_threshold: float = 0.75
    name: str = "searxng-similarity"
    description: str = "SearXNG vector similarity search"
    priority: SourcePriority = SourcePriority.HIGH
    capabilities: SourceCapability = (
        SourceCapability.FACTUAL |
        SourceCapability.CONTEXTUAL
    )
    enabled: bool = True
    collection: str = "searxng-results"
    # Injected dependencies
    _rag_indexer: Optional[Any] = field(default=None, repr=False)
    _search_service: Optional[Any] = field(default=None, repr=False)

    async def retrieve(
        self,
        query: str,
        url: Optional[str] = None,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> KnowledgeResult:
        """
        Find content similar to query or URL.

        Args:
            query: Search query (used for text similarity if no URL)
            url: Optional URL to find similar content for
            context: Request context

        Returns:
            KnowledgeResult with similar content chunks
        """
        import time
        start = time.time()

        chunks = []
        metadata = {
            "tool": "SearXNG-Similarity",
            "similarity_threshold": self.similarity_threshold,
        }

        try:
            if url and self._rag_indexer:
                # Find similar to URL
                ingestion_service = context.get("ingestion_service") if context else None
                searxng_client = context.get("searxng_client") if context else None

                if ingestion_service and searxng_client:
                    result = await self._rag_indexer.search_similar_to_url(
                        url=url,
                        searxng_client=searxng_client,
                        ingestion_service=ingestion_service,
                        collection=self.collection,
                        max_results=self.max_results,
                        similarity_threshold=self.similarity_threshold,
                    )

                    for similar in result.get("similar_results", []):
                        chunks.append(
                            KnowledgeChunk(
                                content=similar.get("content", ""),
                                source=self.name,
                                score=similar.get("score", 0.0),
                                metadata={
                                    "url": similar.get("metadata", {}).get("url", ""),
                                    "title": similar.get("metadata", {}).get("title", ""),
                                },
                                capabilities=self.capabilities,
                            )
                        )

                    metadata["url_searched"] = url
                    metadata["similar_found"] = len(chunks)

            elif self._search_service:
                # Find similar to text
                result = await self._rag_indexer.search_similar_to_text(
                    text=query,
                    collection=self.collection,
                    max_results=self.max_results,
                    similarity_threshold=self.similarity_threshold,
                )

                for similar in result.get("similar_results", []):
                    chunks.append(
                        KnowledgeChunk(
                            content=similar.get("content", ""),
                            source=self.name,
                            score=similar.get("score", 0.0),
                            metadata={
                                "url": similar.get("metadata", {}).get("url", ""),
                                "title": similar.get("metadata", {}).get("title", ""),
                            },
                            capabilities=self.capabilities,
                        )
                    )

                metadata["text_searched"] = query[:100]
                metadata["similar_found"] = len(chunks)

        except Exception as e:
            metadata["error"] = str(e)
            logger.error(f"Similarity search failed: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )


@dataclass
class SearxngClusteringKnowledgeSource:
    """
    Knowledge source that clusters search results into semantic topics.

    Groups related results together and generates topic labels
    for better information organization.
    """
    searxng_url: str = "http://10.4.98.141:7777"  # Kubernetes service
    max_results: int = 15
    max_clusters: int = 5
    name: str = "searxng-clustering"
    description: str = "SearXNG clustered results"
    priority: SourcePriority = SourcePriority.MEDIUM
    capabilities: SourceCapability = (
        SourceCapability.FACTUAL |
        SourceCapability.CONTEXTUAL |
        SourceCapability.COMPARATIVE
    )
    enabled: bool = True
    # Injected dependencies
    _clusterer: Optional[Any] = field(default=None, repr=False)
    _searxng_client: Optional[Any] = field(default=None, repr=False)

    async def retrieve(
        self,
        query: str,
        context: Optional[Dict] = None,
        **kwargs,
    ) -> KnowledgeResult:
        """
        Search and cluster results by topic.

        Args:
            query: Search query
            context: Request context

        Returns:
            KnowledgeResult with clustered chunks
        """
        import time
        start = time.time()

        chunks = []
        metadata = {
            "tool": "SearXNG-Clustering",
            "max_clusters": self.max_clusters,
        }

        try:
            if not self._clusterer or not CLUSTERING_AVAILABLE:
                # Fallback to regular search
                metadata["error"] = "Clustering not available"
                return KnowledgeResult(
                    source_name=self.name,
                    chunks=[],
                    query=query,
                    retrieval_time=time.time() - start,
                    metadata=metadata,
                )

            # Get searxng client or do direct search
            results = []
            if self._searxng_client:
                search_result = await self._searxng_client.search_with_domain_routing(
                    query=query, max_results=self.max_results
                )
                results = search_result.get("results", [])
            else:
                # Direct search fallback
                async with httpx.AsyncClient(timeout=30.0) as client:
                    params = {"q": query, "format": "json"}
                    response = await client.get(
                        self.searxng_url,
                        params=params,
                        headers={
                            "X-Forwarded-For": "10.1.1.110",
                            "X-Real-IP": "10.1.1.110",
                        },
                    )
                    if response.status_code == 200:
                        data = response.json()
                        results = data.get("results", [])

            if not results:
                metadata["error"] = "No results to cluster"
                return KnowledgeResult(
                    source_name=self.name,
                    chunks=[],
                    query=query,
                    retrieval_time=time.time() - start,
                    metadata=metadata,
                )

            # Perform clustering
            cluster_summary = await self._clusterer.cluster_and_summarize(
                results=results, max_clusters=self.max_clusters
            )

            # Create chunks from clustered results
            for cluster_data in cluster_summary.get("clusters", []):
                label = cluster_data.get("label", "Unknown")
                keywords = cluster_data.get("keywords", [])
                cluster_results = cluster_data.get("sample_results", [])

                # Create a chunk for each cluster
                cluster_text = f"Topic: {label}\nKeywords: {', '.join(keywords)}\n\n"
                for result in cluster_results:
                    cluster_text += f"- {result.get('title', '')}\n"

                chunks.append(
                    KnowledgeChunk(
                        content=cluster_text,
                        source=self.name,
                        score=cluster_data.get("score", 0.5),
                        metadata={
                            "cluster_label": label,
                            "keywords": keywords,
                            "size": cluster_data.get("size", 0),
                        },
                        capabilities=self.capabilities,
                    )
                )

            metadata.update(
                {
                    "num_clusters": cluster_summary.get("num_clusters", 0),
                    "clustered_results": cluster_summary.get("clustered_results", 0),
                    "top_keywords": [kw[0] for kw in cluster_summary.get("top_keywords", [])[:5]],
                }
            )

        except Exception as e:
            metadata["error"] = str(e)
            logger.error(f"Clustering failed: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )


def create_similarity_source(
    searxng_url: str = "http://10.4.98.141:7777",  # Kubernetes ClusterIP
    max_results: int = 5,
    similarity_threshold: float = 0.75,
) -> SearxngSimilarityKnowledgeSource:
    """Factory function to create SearXNG similarity source."""
    return SearxngSimilarityKnowledgeSource(
        searxng_url=searxng_url,
        max_results=max_results,
        similarity_threshold=similarity_threshold,
    )


def create_clustering_source(
    searxng_url: str = "http://10.4.98.141:7777",  # Kubernetes ClusterIP
    max_results: int = 15,
    max_clusters: int = 5,
) -> SearxngClusteringKnowledgeSource:
    """Factory function to create SearXNG clustering source."""
    return SearxngClusteringKnowledgeSource(
        searxng_url=searxng_url,
        max_results=max_results,
        max_clusters=max_clusters,
    )
