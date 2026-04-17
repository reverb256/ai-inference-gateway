"""
RAG (Retrieval Augmented Generation) Module for AI Inference Gateway.

Implements production-ready RAG with:
- BGE-M3 embeddings (multilingual, 8192 token context)
- Recursive character chunking (512 tokens, 10% overlap)
- Hybrid search (dense + sparse with RRF fusion)
- BGE-Reranker-v2 for result refinement
- Async Qdrant integration with gRPC
"""

from .config import (
    RAGConfig,
    EmbeddingConfig,
    ChunkingConfig,
    SearchConfig,
    RerankerConfig,
)
from .embeddings import EmbeddingService
from .chunker import DocumentChunker
from .qdrant_client import QdrantManager
from .search import HybridSearchService

__all__ = [
    "RAGConfig",
    "EmbeddingConfig",
    "ChunkingConfig",
    "SearchConfig",
    "RerankerConfig",
    "EmbeddingService",
    "DocumentChunker",
    "QdrantManager",
    "HybridSearchService",
]
