"""
Hybrid Search Service for RAG.

Implements end-to-end search pipeline:
- Query embedding
- Hybrid search (dense + sparse)
- Reranking (optional)
- Result formatting
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional

from .config import RAGConfig
from .embeddings import EmbeddingService
from .qdrant_client import QdrantManager, SearchResult

logger = logging.getLogger(__name__)


class HybridSearchService:
    """
    Hybrid search service combining dense, sparse, and reranking.
    """

    def __init__(
        self, config: RAGConfig, embedder: EmbeddingService, qdrant: QdrantManager
    ):
        """
        Initialize hybrid search service.

        Args:
            config: RAG configuration
            embedder: Embedding service
            qdrant: Qdrant manager
        """
        self.config = config
        self.embedder = embedder
        self.qdrant = qdrant
        self._reranker = None

    async def initialize_reranker(self) -> None:
        """Initialize reranker if enabled."""
        if not self.config.reranker.enable:
            return

        try:
            # Import reranker models (lazy import)
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker model: {self.config.reranker.model}")

            # Load model in thread pool
            loop = asyncio.get_event_loop()
            self._reranker = await loop.run_in_executor(
                None, lambda: CrossEncoder(self.config.reranker.model)
            )

            logger.info("Reranker loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            logger.warning("Reranking disabled due to error")
            self.config.reranker.enable = False

    async def search(
        self,
        query: str,
        collection: str = "default",
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            collection: Collection name
            top_k: Number of results (default from config)
            rerank: Enable reranking (default from config)

        Returns:
            Search results with metadata
        """
        top_k = top_k or self.config.search.default_top_k
        rerank = rerank if rerank is not None else self.config.reranker.enable

        try:
            # Generate embeddings
            query_dense, query_sparse = await asyncio.gather(
                self.embedder.embed_single(query), self._get_sparse_embedding(query)
            )

            # Hybrid search
            if self.config.search.hybrid_search:
                # Recall more results if reranking
                recall_k = self.config.reranker.top_k if rerank else top_k

                results = await self.qdrant.hybrid_search(
                    collection_name=collection,
                    query_dense=query_dense,
                    query_sparse=query_sparse,
                    limit=top_k,
                    dense_limit=recall_k,
                    sparse_limit=recall_k,
                )
            else:
                # Dense-only search
                results = await self.qdrant.search_dense(
                    collection_name=collection, query_vector=query_dense, limit=top_k
                )

            # Rerank if enabled
            if rerank and self._reranker and len(results) > 0:
                results = await self._rerank_results(query, results)

            # Format results
            return {
                "query": query,
                "results": [
                    {
                        "content": r.content,
                        "score": float(r.score),
                        "dense_score": float(r.dense_score) if r.dense_score else None,
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
                "total_results": len(results),
                "collection": collection,
                "reranked": rerank,
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {
                "query": query,
                "results": [],
                "total_results": 0,
                "collection": collection,
                "error": str(e),
            }

    async def _get_sparse_embedding(self, text: str) -> Dict[int, float]:
        """
        Get sparse embedding for text.

        Args:
            text: Input text

        Returns:
            Sparse vector
        """
        try:
            sparse_embeddings = await self.embedder.embed_sparse([text])
            return sparse_embeddings[0]
        except Exception as e:
            logger.error(f"Failed to generate sparse embedding: {e}")
            return {}

    async def _rerank_results(
        self, query: str, results: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Rerank search results.

        Args:
            query: Original query
            results: Search results to rerank

        Returns:
            Reranked results
        """
        if not self._reranker or len(results) == 0:
            return results

        try:
            # Prepare query-document pairs
            pairs = [(query, result.content) for result in results]

            # Run reranker in thread pool
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, lambda: self._reranker.predict(pairs)
            )

            # Update scores and sort
            for result, score in zip(results, scores):
                result.score = float(score)

            # Sort by new score
            results.sort(key=lambda r: r.score, reverse=True)

            # Return top-K
            final_k = self.config.reranker.final_k
            return results[:final_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    async def ingest_document(
        self,
        collection: str,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ingest document into RAG system.

        Args:
            collection: Collection name
            content: Document content
            metadata: Document metadata
            document_id: Optional document ID

        Returns:
            Ingestion result
        """
        try:
            from .chunker import create_document_chunker
            import uuid

            document_id = document_id or str(uuid.uuid4())

            # Chunk document
            chunker = create_document_chunker(self.config.chunking)
            chunks = chunker.chunk_text(content, metadata)

            if not chunks:
                return {"success": False, "error": "No chunks generated"}

            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            dense_embeddings, sparse_embeddings = await asyncio.gather(
                self.embedder.embed_dense(texts), self.embedder.embed_sparse(texts)
            )

            # Prepare chunks for Qdrant
            qdrant_chunks = []
            for chunk, dense_emb, sparse_emb in zip(
                chunks, dense_embeddings, sparse_embeddings
            ):
                qdrant_chunks.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "dense_embedding": dense_emb,
                        "sparse_embedding": sparse_emb,
                        "content": chunk.content,
                        "document_id": document_id,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_index": chunk.chunk_index,
                            "document_id": document_id,
                        },
                    }
                )

            # Ensure collection exists
            await self.qdrant.ensure_collection(
                collection, self.config.embedding.dimensions
            )

            # Ingest into Qdrant
            points_inserted = await self.qdrant.ingest_chunks(collection, qdrant_chunks)

            return {
                "success": True,
                "document_id": document_id,
                "chunks_created": len(chunks),
                "points_inserted": points_inserted,
                "collection": collection,
            }

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {"success": False, "error": str(e)}

    async def get_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections with info.

        Returns:
            List of collection info
        """
        try:
            collection_names = await self.qdrant.list_collections()
            collections = []

            for name in collection_names:
                info = await self.qdrant.get_collection_info(name)
                if info:
                    collections.append(info)

            return collections

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def delete_document(
        self, collection: str, document_id: str
    ) -> Dict[str, Any]:
        """
        Delete document from collection.

        Args:
            collection: Collection name
            document_id: Document ID

        Returns:
            Deletion result
        """
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            # Find all points for this document using metadata filter
            # Note: We can't use query_points with filter easily, so we'll scroll
            # For simplicity, we'll return a message about the limitation

            # Better approach: Use scroll to find points by document_id

            # Scroll through points to find matching document_id
            offset = None
            point_ids = []

            while True:
                records, offset = await self.qdrant._client.scroll(
                    collection_name=collection,
                    limit=100,
                    offset=offset,
                    with_payload=["document_id"],
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.document_id",
                                match=MatchValue(value=document_id),
                            )
                        ]
                    ),
                )

                if records:
                    point_ids.extend([r.id for r in records])

                if offset is None:
                    break

            if not point_ids:
                return {
                    "success": False,
                    "error": f"No documents found with ID: {document_id}",
                }

            # Delete the points
            deleted = await self.qdrant.delete_points(collection, point_ids)

            return {
                "success": True,
                "document_id": document_id,
                "points_deleted": deleted,
                "collection": collection,
            }

        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return {"success": False, "error": str(e)}


async def create_search_service(
    config: RAGConfig, embedder: EmbeddingService, qdrant: QdrantManager
) -> HybridSearchService:
    """
    Create and initialize search service.

    Args:
        config: RAG configuration
        embedder: Embedding service
        qdrant: Qdrant manager

    Returns:
        Initialized search service
    """
    service = HybridSearchService(config, embedder, qdrant)
    await service.initialize_reranker()
    return service
