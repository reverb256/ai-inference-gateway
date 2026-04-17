"""
Knowledge Source Adapters

Provides adapters for heterogeneous knowledge systems to integrate
with the Knowledge Fabric unified retrieval system.
"""

from .rag_source import RAGKnowledgeSource, create_rag_source
from .web_search_source import WebSearchKnowledgeSource, create_web_search_source
from .searxng_source import (
    SearXNGKnowledgeSource,
    create_searxng_source,
    SearxngSimilarityKnowledgeSource,
    create_similarity_source,
    SearxngClusteringKnowledgeSource,
    create_clustering_source,
)
from .code_search_source import CodeSearchKnowledgeSource, create_code_search_source
from .brain_source import BrainWikiSource, create_brain_source

__all__ = [
    "RAGKnowledgeSource",
    "create_rag_source",
    "WebSearchKnowledgeSource",
    "create_web_search_source",
    "SearXNGKnowledgeSource",
    "create_searxng_source",
    "SearxngSimilarityKnowledgeSource",
    "create_similarity_source",
    "SearxngClusteringKnowledgeSource",
    "create_clustering_source",
    "CodeSearchKnowledgeSource",
    "create_code_search_source",
    "BrainWikiSource",
    "create_brain_source",
]
