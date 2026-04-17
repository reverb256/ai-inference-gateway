"""
Code Search Knowledge Source Adapter for Knowledge Fabric

Provides code search integration using semantic and literal search.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List
import httpx

from ..core import (
    KnowledgeChunk,
    KnowledgeResult,
    SourceCapability,
    SourcePriority,
)

logger = logging.getLogger(__name__)


@dataclass
class CodeSearchKnowledgeSource:
    """
    Code search knowledge source.

    Provides semantic code search through embeddings-based retrieval.
    """
    search_url: str = "http://127.0.0.1:8080/mcp/call"
    max_results: int = 5
    timeout: float = 30.0
    search_paths: List[str] = field(default_factory=list)
    name: str = "code_search"
    description: str = "Semantic code search"
    priority: SourcePriority = SourcePriority.CRITICAL
    capabilities: SourceCapability = (
        SourceCapability.CODE |
        SourceCapability.PROCEDURAL
    )
    enabled: bool = True

    async def retrieve(self, query: str, **kwargs) -> KnowledgeResult:
        """
        Execute code search via embeddings.

        Returns relevant code snippets with semantic matching.
        """
        import time
        start = time.time()

        sanitized_query = query[:500]

        chunks = []
        metadata = {
            "tool": "code_search",
            "type": "semantic_search",
        }

        try:
            # This would integrate with a code search index
            # For now, return empty results as a placeholder
            metadata["total_results"] = 0
            metadata["note"] = "Code search not yet implemented"

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            logger.exception(f"Code search error: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )


def create_code_search_source(
    search_url: str = "http://127.0.0.1:8080/mcp/call",
    max_results: int = 5,
) -> CodeSearchKnowledgeSource:
    """Factory function to create code search knowledge source."""
    return CodeSearchKnowledgeSource(
        search_url=search_url,
        max_results=max_results,
    )
