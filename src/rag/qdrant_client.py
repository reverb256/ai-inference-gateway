"""
Qdrant Client Manager for RAG.

Manages Qdrant connection with:
- Async client with gRPC
- Connection pooling (singleton pattern)
- Collection management
- Hybrid search (dense + sparse)
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

from .config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A search result from Qdrant."""

    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    sparse_score: Optional[float] = None
    dense_score: Optional[float] = None


class QdrantManager:
    """
    Qdrant client manager with async support and connection pooling.

    Implements singleton pattern for efficient connection reuse.
    """

    _instance: Optional["QdrantManager"] = None
    _lock = asyncio.Lock()

    def __init__(self, config: RAGConfig):
        """
        Initialize Qdrant manager.

        Args:
            config: RAG configuration
        """
        self.config = config
        self._client: Optional[AsyncQdrantClient] = None
        self._initialized = False

    @classmethod
    async def get_instance(cls, config: RAGConfig) -> "QdrantManager":
        """
        Get singleton instance.

        Args:
            config: RAG configuration

        Returns:
            Qdrant manager instance
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
                await cls._instance.initialize()
            return cls._instance

    async def initialize(self) -> None:
        """Initialize Qdrant client."""
        if self._initialized:
            return

        try:
            logger.info(f"Connecting to Qdrant at {self.config.qdrant_url}")

            self._client = AsyncQdrantClient(
                url=self.config.qdrant_url,
                timeout=self.config.qdrant_timeout,
                prefer_grpc=self.config.prefer_grpc,
            )

            # Test connection
            collections = await self._client.get_collections()
            logger.info(
                f"Connected to Qdrant (collections: {len(collections.collections)})"
            )

            self._initialized = True

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def ensure_collection(self, collection_name: str, dimensions: int) -> None:
        """
        Ensure collection exists.

        Args:
            collection_name: Name of collection
            dimensions: Vector dimensions
        """
        try:
            # Check if collection exists
            collections = await self._client.get_collections()
            existing = [c.name for c in collections.collections]

            if collection_name not in existing:
                logger.info(f"Creating collection: {collection_name}")

                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=dimensions, distance=Distance.COSINE
                    ),
                    sparse_vectors_config={"text": SparseVectorParams()},
                )

                logger.info(f"Collection created: {collection_name}")
            else:
                logger.debug(f"Collection already exists: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure collection {collection_name}: {e}")
            raise

    async def ingest_chunks(
        self, collection_name: str, chunks: List[Dict[str, Any]]
    ) -> int:
        """
        Ingest document chunks into Qdrant.

        Args:
            collection_name: Name of collection
            chunks: List of chunks with embeddings and metadata

        Returns:
            Number of points ingested
        """
        try:
            points = []

            for chunk in chunks:
                # Build sparse vector only if it has content
                sparse_vector = None
                if chunk["sparse_embedding"]:
                    sparse_vector = SparseVector(
                        indices=list(chunk["sparse_embedding"].keys()),
                        values=list(chunk["sparse_embedding"].values()),
                    )

                point_data = {
                    "id": chunk["chunk_id"],
                    "vector": chunk["dense_embedding"],
                    "payload": {
                        "content": chunk["content"],
                        "document_id": chunk.get("document_id", ""),
                        "metadata": chunk.get("metadata", {}),
                    },
                }

                # Only add sparse vector if it has content
                if sparse_vector:
                    point_data["sparse_vectors"] = {"text": sparse_vector}

                point = PointStruct(**point_data)
                points.append(point)

            # Batch insert
            await self._client.upsert(collection_name=collection_name, points=points)

            logger.info(f"Inserted {len(points)} points into {collection_name}")
            return len(points)

        except Exception as e:
            logger.error(f"Failed to ingest chunks: {e}")
            raise

    async def search_dense(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Dense vector search.

        Args:
            collection_name: Name of collection
            query_vector: Query embedding
            limit: Number of results
            score_threshold: Minimum score threshold

        Returns:
            List of search results
        """
        try:
            response = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,  # Pass vector directly as query parameter
                limit=limit,
                score_threshold=score_threshold,
                with_payload=["content", "metadata"],
            )

            return [
                SearchResult(
                    id=str(hit.id),
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload.get("metadata", {}),
                )
                for hit in response.points
            ]

        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    async def search_sparse(
        self, collection_name: str, query_sparse: Dict[int, float], limit: int = 10
    ) -> List[SearchResult]:
        """
        Sparse vector search (BM25).

        Args:
            collection_name: Name of collection
            query_sparse: Sparse query vector
            limit: Number of results

        Returns:
            List of search results
        """
        try:
            # Convert dict to SparseVector object if not empty
            if not query_sparse:
                return []

            from qdrant_client.models import SparseVector

            sparse_vector = SparseVector(
                indices=list(query_sparse.keys()), values=list(query_sparse.values())
            )

            # Use query_points with sparse vector
            response = await self._client.query_points(
                collection_name=collection_name,
                query=sparse_vector,
                using="text",  # Use the sparse vector named "text"
                limit=limit,
                with_payload=["content", "metadata"],
            )

            return [
                SearchResult(
                    id=str(hit.id),
                    content=hit.payload.get("content", ""),
                    score=hit.score,
                    metadata=hit.payload.get("metadata", {}),
                    sparse_score=hit.score,
                )
                for hit in response.points
            ]

        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []

    async def hybrid_search(
        self,
        collection_name: str,
        query_dense: List[float],
        query_sparse: Dict[int, float],
        limit: int = 10,
        dense_limit: int = 30,
        sparse_limit: int = 30,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense and sparse.

        Performs parallel searches and fuses results using RRF.

        Args:
            collection_name: Name of collection
            query_dense: Dense query vector
            query_sparse: Sparse query vector
            limit: Final number of results
            dense_limit: Number of dense results to fetch
            sparse_limit: Number of sparse results to fetch

        Returns:
            Fused search results
        """
        try:
            # Parallel searches
            dense_results, sparse_results = await asyncio.gather(
                self.search_dense(collection_name, query_dense, dense_limit),
                self.search_sparse(collection_name, query_sparse, sparse_limit),
            )

            # Reciprocal Rank Fusion (RRF)
            fused = self._reciprocal_rank_fusion(
                dense_results, sparse_results, k=self.config.search.rrf_k
            )

            return fused[:limit]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[SearchResult],
        sparse_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Reciprocal Rank Fusion (RRF).

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results
            k: RRF constant

        Returns:
            Fused and sorted results
        """
        # Calculate RRF scores
        scores: Dict[str, Tuple[float, SearchResult, Optional[float]]] = {}

        for rank, result in enumerate(dense_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if result.id not in scores:
                scores[result.id] = (rrf_score, result, result.score)
            else:
                current_score, _, _ = scores[result.id]
                scores[result.id] = (current_score + rrf_score, result, result.score)

        for rank, result in enumerate(sparse_results, start=1):
            rrf_score = 1.0 / (k + rank)
            if result.id not in scores:
                scores[result.id] = (rrf_score, result, result.sparse_score)
            else:
                current_score, _, existing_dense = scores[result.id]
                scores[result.id] = (current_score + rrf_score, result, existing_dense)

        # Sort by RRF score
        sorted_results = sorted(
            [(score, result, dense) for score, result, dense in scores.values()],
            key=lambda x: x[0],
            reverse=True,
        )

        # Update scores in results, preserving dense_score for downstream reranking
        return [
            SearchResult(
                id=result.id,
                content=result.content,
                score=score,  # RRF score
                metadata=result.metadata,
                dense_score=dense,  # Original cosine similarity
            )
            for score, result, dense in sorted_results
        ]

    async def get_collection_info(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get collection information.

        Args:
            collection_name: Name of collection

        Returns:
            Collection info or None
        """
        try:
            info = await self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None

    async def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        try:
            collections = await self._client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    async def delete_points(self, collection_name: str, point_ids: List[str]) -> int:
        """
        Delete points from collection.

        Args:
            collection_name: Name of collection
            point_ids: List of point IDs

        Returns:
            Number of points deleted
        """
        try:
            await self._client.delete(
                collection_name=collection_name,
                points_selector=models.PointIdsList(points=point_ids),
            )
            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
            return len(point_ids)
        except Exception as e:
            logger.error(f"Failed to delete points: {e}")
            return 0

    async def close(self) -> None:
        """Close Qdrant connection."""
        if self._client:
            await self._client.close()
            self._initialized = False
            logger.info("Qdrant connection closed")


async def get_qdrant_manager(config: RAGConfig) -> QdrantManager:
    """
    Get Qdrant manager instance.

    Args:
        config: RAG configuration

    Returns:
        Qdrant manager instance
    """
    return await QdrantManager.get_instance(config)
