"""
Embedding Service for RAG.

Handles text embedding using BGE-M3 model with support for:
- Dense embeddings (semantic search)
- Sparse embeddings (BM25-like lexical search)
- GPU acceleration
- Batch processing
"""

import asyncio
import logging
from typing import List, Optional, Dict
from sentence_transformers import SentenceTransformer
import torch

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Embedding service using BGE-M3 model.

    BGE-M3 supports:
    - Dense embeddings (1024 dims)
    - Sparse embeddings (lexical search)
    - Multi-lingual (100+ languages)
    - 8192 token context window
    """

    def __init__(self, config: EmbeddingConfig):
        """
        Initialize embedding service.

        Args:
            config: Embedding configuration
        """
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the embedding model (lazy loading)."""
        async with self._lock:
            if self._model is not None:
                return

            try:
                logger.info(f"Loading embedding model: {self.config.model}")

                # Determine device
                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info(
                        f"Using CUDA for embeddings (GPU: {torch.cuda.get_device_name(0)})"
                    )
                else:
                    self._device = "cpu"
                    logger.info("Using CPU for embeddings")

                # Load model (blocking call - run in thread pool)
                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(self.config.model, device=self._device),
                )

                # Verify dimensions
                actual_dims = self._model.get_sentence_embedding_dimension()
                if actual_dims != self.config.dimensions:
                    logger.info(
                        f"Embedding model dimensions: {actual_dims} (configured: {self.config.dimensions}). "
                        f"Using actual dimensions from model."
                    )
                    self.config.dimensions = actual_dims

                logger.info(
                    f"Embedding model loaded successfully (dims: {actual_dims})"
                )

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    async def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """
        Generate dense embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if self._model is None:
            await self.initialize()

        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,  # Important for cosine similarity
                ),
            )

            # Convert to list of lists
            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Failed to generate dense embeddings: {e}")
            raise

    async def embed_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Generate sparse embeddings (BM25-like lexical features).

        BGE-M3 supports sparse embeddings natively.

        Args:
            texts: List of text strings

        Returns:
            List of sparse vectors (token_id -> weight mappings)
        """
        if self._model is None:
            await self.initialize()

        try:
            # Run in thread pool
            loop = asyncio.get_event_loop()

            # BGE-M3 sparse embeddings
            sparse_embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    output_value="sparse",
                    show_progress_bar=False,
                ),
            )

            # Convert sparse matrix to list of dicts
            results = []
            for i in range(len(texts)):
                # Get sparse row
                sparse_row = sparse_embeddings[i]
                # Convert to dict
                token_weights = {
                    int(idx): float(weight)
                    for idx, weight in zip(sparse_row.indices, sparse_row.data)
                }
                results.append(token_weights)

            return results

        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            # Fallback: return empty sparse embeddings
            logger.warning("Using empty sparse embeddings as fallback")
            return [{} for _ in texts]

    async def embed_single(self, text: str) -> List[float]:
        """
        Generate dense embedding for a single text.

        Args:
            text: Text string

        Returns:
            Embedding vector
        """
        embeddings = await self.embed_dense([text])
        return embeddings[0]

    def is_initialized(self) -> bool:
        """Check if model is initialized."""
        return self._model is not None

    async def shutdown(self) -> None:
        """Cleanup resources."""
        async with self._lock:
            if self._model is not None:
                # Free GPU memory if using CUDA
                if self._device == "cuda":
                    del self._model
                    torch.cuda.empty_cache()
                    logger.info("Freed CUDA memory for embedding model")

                self._model = None
                logger.info("Embedding service shutdown complete")


async def create_embedding_service(config: EmbeddingConfig) -> EmbeddingService:
    """
    Create and initialize embedding service.

    Args:
        config: Embedding configuration

    Returns:
        Initialized embedding service
    """
    service = EmbeddingService(config)
    await service.initialize()
    return service
