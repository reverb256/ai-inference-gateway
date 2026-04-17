"""
RAG Knowledge Source Adapter

Integrates the HybridSearchService as a knowledge source for the Knowledge Fabric.
Provides semantic search over the RAG knowledge base with proper chunk formatting
and metadata mapping.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any

from ai_inference_gateway.rag.search import HybridSearchService

from ..core import (
    SourcePriority,
    SourceCapability,
    KnowledgeResult,
    KnowledgeChunk,
)


@dataclass
class RAGKnowledgeSource:
    """
    RAG knowledge source adapter for Knowledge Fabric.

    Wraps the HybridSearchService to provide semantic search capabilities
    over the knowledge base as a unified knowledge source.
    """
    search_service: HybridSearchService
    collection: str = "knowledge-base"
    max_chunks: int = 5
    name: str = "rag"
    description: str = "RAG knowledge base with semantic search"
    priority: SourcePriority = SourcePriority.HIGH
    capabilities: SourceCapability = (
        SourceCapability.FACTUAL | SourceCapability.CONTEXTUAL | SourceCapability.PROCEDURAL
    )
    enabled: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")
        # Note: search_service can be None for runtime injection
        # It will be validated during retrieve() if needed

    async def retrieve(self, query: str, **kwargs) -> KnowledgeResult:
        """
        Retrieve knowledge chunks from RAG knowledge base.
        """
        import time
        start = time.time()

        # Extract override parameters
        collection = kwargs.get("collection", self.collection)
        top_k = kwargs.get("top_k", self.max_chunks)
        rerank = kwargs.get("rerank", None)

        # Check if search_service is available
        if not self.search_service:
            return KnowledgeResult(
                source_name=self.name,
                chunks=[],
                query=query,
                retrieval_time=time.time() - start,
                metadata={
                    "note": "search_service not available (needs runtime injection)",
                },
            )

        try:
            # Perform hybrid search
            search_result = await self.search_service.search(
                query=query,
                collection=collection,
                top_k=top_k,
                rerank=rerank,
            )

            # Extract results
            results = search_result.get("results", [])
            total_results = search_result.get("total_results", 0)
            was_reranked = search_result.get("reranked", False)

            # Map to KnowledgeChunk format
            chunks = []
            for i, result in enumerate(results):
                content = result.get("content", "")
                score = result.get("score", 0.0)
                metadata = result.get("metadata", {})

                # Enhance metadata
                enhanced_metadata = {
                    **metadata,
                    "collection": collection,
                    "retrieval_method": "semantic_search",
                    "reranked": was_reranked,
                    "rank": i + 1,
                }

                chunk = KnowledgeChunk(
                    content=content,
                    source=self.name,
                    score=score,
                    metadata=enhanced_metadata,
                    capabilities=self.capabilities,
                )
                chunks.append(chunk)

            retrieval_time = time.time() - start

            return KnowledgeResult(
                source_name=self.name,
                chunks=chunks,
                query=query,
                retrieval_time=retrieval_time,
                metadata={
                    "collection": collection,
                    "total_available": total_results,
                    "retrieved": len(chunks),
                    "reranked": was_reranked,
                },
            )

        except Exception as e:
            retrieval_time = time.time() - start
            logging.getLogger(__name__).error(f"RAG search failed: {e}", exc_info=True)

            return KnowledgeResult(
                source_name=self.name,
                chunks=[],
                query=query,
                retrieval_time=retrieval_time,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )


def create_rag_source(
    search_service: HybridSearchService,
    collection: str = "knowledge-base",
    max_chunks: int = 5,
    **kwargs
) -> RAGKnowledgeSource:
    """Factory function to create RAG knowledge source."""
    return RAGKnowledgeSource(
        search_service=search_service,
        collection=collection,
        max_chunks=max_chunks,
        **kwargs
    )
