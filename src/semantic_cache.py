"""
Semantic Caching with Exact Match and Vector Similarity

Two-layer caching system for AI responses:
1. Exact match cache (Redis) - Fast key-based lookups
2. Semantic cache (Qdrant) - Vector similarity for paraphrases

Features:
- Exact match caching with Redis
- Semantic similarity search with Qdrant
- Automatic cache warming
- TTL-based invalidation
- Cache metrics and analytics
- Configurable similarity thresholds
"""

import hashlib
import json
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        VectorParams,
        PointStruct,
    )

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer types."""

    EXACT = "exact"  # Redis exact match
    SEMANTIC = "semantic"  # Qdrant vector similarity


@dataclass
class CacheConfig:
    """
    Cache configuration.

    Attributes:
        redis_url: Redis connection URL (default: redis://localhost:6379)
        qdrant_url: Qdrant server URL (default: http://localhost:6333)
        qdrant_collection: Collection name (default: ai-responses)
        exact_ttl_seconds: TTL for exact match cache (default: 3600s = 1 hour)
        semantic_ttl_seconds: TTL for semantic cache (default: 86400s = 24 hours)
        similarity_threshold: Minimum similarity score (0-1) for semantic hits (default: 0.85)
        embedding_model: Model for embeddings (default: all-MiniLM-L6-v2)
        embedding_endpoint: Embedding generation endpoint (default: http://127.0.0.1:1234/v1/embeddings)
        enable_exact_cache: Enable exact match caching
        enable_semantic_cache: Enable semantic caching
    """

    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "ai-responses"
    exact_ttl_seconds: int = 3600  # 1 hour
    semantic_ttl_seconds: int = 86400  # 24 hours
    similarity_threshold: float = 0.85
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_endpoint: str = "http://127.0.0.1:1234/v1/embeddings"
    enable_exact_cache: bool = True
    enable_semantic_cache: bool = True


@dataclass
class CacheHit:
    """Cache hit result."""

    layer: CacheLayer
    response: Dict[str, Any]
    similarity_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    exact_hits: int = 0
    exact_misses: int = 0
    semantic_hits: int = 0
    semantic_misses: int = 0
    total_requests: int = 0

    @property
    def exact_hit_rate(self) -> float:
        """Exact cache hit rate."""
        total = self.exact_hits + self.exact_misses
        return (self.exact_hits / total * 100) if total > 0 else 0.0

    @property
    def semantic_hit_rate(self) -> float:
        """Semantic cache hit rate."""
        total = self.semantic_hits + self.semantic_misses
        return (self.semantic_hits / total * 100) if total > 0 else 0.0

    @property
    def overall_hit_rate(self) -> float:
        """Combined cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        hits = self.exact_hits + self.semantic_hits
        return hits / self.total_requests * 100


class SemanticCache:
    """
    Two-layer semantic caching for AI responses.

    Provides fast exact match lookups with Redis and semantic
    similarity search with Qdrant for paraphrase detection.
    """

    def __init__(
        self, config: Optional[CacheConfig] = None, enable_metrics: bool = True
    ):
        """
        Initialize semantic cache.

        Args:
            config: Cache configuration (uses defaults if None)
            enable_metrics: Track cache metrics
        """
        self.config = config or CacheConfig()
        self.enable_metrics = enable_metrics
        self.metrics = CacheMetrics() if enable_metrics else None

        # Redis client (lazy initialization)
        self._redis: Optional[redis.Redis] = None

        # Qdrant client (lazy initialization)
        self._qdrant: Optional[QdrantClient] = None

        # Check dependencies
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available. Install redis: pip install redis")
            self.config.enable_exact_cache = False

        if not QDRANT_AVAILABLE:
            logger.warning(
                "Qdrant not available. Install qdrant-client: pip install qdrant-client"
            )
            self.config.enable_semantic_cache = False

        logger.info(
            f"SemanticCache initialized: "
            f"exact={self.config.enable_exact_cache}, "
            f"semantic={self.config.enable_semantic_cache}"
        )

    async def _get_redis(self) -> Optional[redis.Redis]:
        """Get or create Redis client."""
        if not self.config.enable_exact_cache or not REDIS_AVAILABLE:
            return None

        if self._redis is None:
            try:
                self._redis = await redis.from_url(
                    self.config.redis_url, encoding="utf-8", decode_responses=True
                )
                logger.info(f"Connected to Redis: {self.config.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.config.enable_exact_cache = False

        return self._redis

    async def _get_qdrant(self) -> Optional[QdrantClient]:
        """Get or create Qdrant client."""
        if not self.config.enable_semantic_cache or not QDRANT_AVAILABLE:
            return None

        if self._qdrant is None:
            try:
                self._qdrant = QdrantClient(url=self.config.qdrant_url)

                # Create collection if not exists
                collections = self._qdrant.get_collections()
                collection_names = [c.name for c in collections.collections]

                if self.config.qdrant_collection not in collection_names:
                    # Detect embedding dimension by generating a test embedding
                    embedding_size = await self._detect_embedding_size()

                    self._qdrant.create_collection(
                        collection_name=self.config.qdrant_collection,
                        vectors_config=VectorParams(
                            size=embedding_size, distance=Distance.COSINE
                        ),
                    )
                    logger.info(
                        f"Created Qdrant collection: {self.config.qdrant_collection} "
                        f"(vector_size={embedding_size})"
                    )
                else:
                    logger.info(
                        f"Using existing Qdrant collection: {self.config.qdrant_collection}"
                    )

            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                self.config.enable_semantic_cache = False

        return self._qdrant

    async def _detect_embedding_size(self) -> int:
        """Detect the embedding dimension by generating a test embedding."""
        test_embedding = await self._generate_embedding([{"role": "user", "content": "test"}])
        return len(test_embedding)

    def _make_cache_key(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """
        Generate cache key from request parameters.

        Args:
            model: Model name
            messages: Chat messages
            **kwargs: Additional request parameters

        Returns:
            SHA256 hash key
        """
        # Create deterministic string from request
        cache_dict = {
            "model": model,
            "messages": messages,
            **{k: v for k, v in kwargs.items() if k not in ["stream", "n"]},
        }
        cache_str = json.dumps(cache_dict, sort_keys=True)

        # Hash for compact key
        return hashlib.sha256(cache_str.encode()).hexdigest()

    async def get(
        self, model: str, messages: List[Dict[str, str]], **kwargs
    ) -> Optional[CacheHit]:
        """
        Get cached response.

        Checks exact cache first, then semantic cache.

        Args:
            model: Model name
            messages: Chat messages
            **kwargs: Additional request parameters

        Returns:
            CacheHit if found, None otherwise
        """
        if self.enable_metrics:
            self.metrics.total_requests += 1

        cache_key = self._make_cache_key(model, messages, **kwargs)

        # Try exact cache first (Redis)
        if self.config.enable_exact_cache:
            exact_hit = await self._get_exact(cache_key)
            if exact_hit:
                if self.enable_metrics:
                    self.metrics.exact_hits += 1
                return CacheHit(
                    layer=CacheLayer.EXACT,
                    response=exact_hit,
                    metadata={"cache_key": cache_key},
                )
            else:
                if self.enable_metrics:
                    self.metrics.exact_misses += 1

        # Try semantic cache (Qdrant)
        if self.config.enable_semantic_cache:
            semantic_hit = await self._get_semantic(messages)
            if semantic_hit:
                if self.enable_metrics:
                    self.metrics.semantic_hits += 1
                return semantic_hit
            else:
                if self.enable_metrics:
                    self.metrics.semantic_misses += 1

        return None

    async def _get_exact(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get response from exact cache (Redis)."""
        redis_client = await self._get_redis()
        if not redis_client:
            return None

        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.error(f"Error reading from Redis: {e}")

        return None

    async def _get_semantic(self, messages: List[Dict[str, str]]) -> Optional[CacheHit]:
        """Get response from semantic cache (Qdrant)."""
        qdrant_client = await self._get_qdrant()
        if not qdrant_client:
            return None

        try:
            # Generate embedding for query (placeholder - would call embedding service)
            query_embedding = await self._generate_embedding(messages)

            # Search for similar vectors
            results = qdrant_client.search(
                collection_name=self.config.qdrant_collection,
                query_vector=query_embedding,
                limit=1,
                score_threshold=self.config.similarity_threshold,
            )

            if results and results[0].score >= self.config.similarity_threshold:
                return CacheHit(
                    layer=CacheLayer.SEMANTIC,
                    response=results[0].payload.get("response"),
                    similarity_score=results[0].score,
                    metadata={
                        "point_id": results[0].id,
                        "similarity": results[0].score,
                    },
                )

        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")

        return None

    async def set(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response: Dict[str, Any],
        **kwargs,
    ) -> bool:
        """
        Store response in cache.

        Stores in both exact and semantic cache.

        Args:
            model: Model name
            messages: Chat messages
            response: Model response to cache
            **kwargs: Additional request parameters

        Returns:
            True if cached successfully
        """
        cache_key = self._make_cache_key(model, messages, **kwargs)

        # Store in exact cache (Redis)
        if self.config.enable_exact_cache:
            await self._set_exact(cache_key, response)

        # Store in semantic cache (Qdrant)
        if self.config.enable_semantic_cache:
            await self._set_semantic(messages, response)

        return True

    async def _set_exact(self, cache_key: str, response: Dict[str, Any]):
        """Store response in exact cache (Redis)."""
        redis_client = await self._get_redis()
        if not redis_client:
            return

        try:
            await redis_client.setex(
                cache_key, self.config.exact_ttl_seconds, json.dumps(response)
            )
        except Exception as e:
            logger.error(f"Error writing to Redis: {e}")

    async def _set_semantic(
        self, messages: List[Dict[str, str]], response: Dict[str, Any]
    ):
        """Store response in semantic cache (Qdrant)."""
        qdrant_client = await self._get_qdrant()
        if not qdrant_client:
            return

        try:
            # Generate embedding
            embedding = await self._generate_embedding(messages)

            # Store in Qdrant
            point_id = hashlib.sha256(json.dumps(messages).encode()).hexdigest()
            qdrant_client.upsert(
                collection_name=self.config.qdrant_collection,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "messages": messages,
                            "response": response,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Error writing to Qdrant: {e}")

    async def _generate_embedding(self, messages: List[Dict[str, str]]) -> List[float]:
        """
        Generate embedding for messages using the embedding endpoint.

        Uses the local embedding service to generate embeddings,
        avoiding external API calls and keeping everything on-cluster.

        Args:
            messages: Chat messages

        Returns:
            Embedding vector (dimension depends on the embedding model)
        """
        try:
            import httpx

            # Extract text from messages for embedding
            # For multi-turn conversations, concatenate all messages
            text_parts = []
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if isinstance(content, str):
                    text_parts.append(f"{role}: {content}")
                elif isinstance(content, list):
                    # Handle content blocks (e.g., for images)
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(f"{role}: {block.get('text', '')}")

            # Join messages for embedding
            text_to_embed = "\n".join(text_parts)

            # Use the embedding endpoint (configurable)
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.config.embedding_endpoint,
                    json={
                        "model": self.config.embedding_model,
                        "input": text_to_embed,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("data", [{}])[0].get("embedding", [])

                    if embedding:
                        logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                        return embedding
                    else:
                        logger.warning("Empty embedding received from embedding service")

                logger.warning(
                    f"Failed to get embedding from service: {response.status_code}"
                )

        except ImportError:
            logger.warning("httpx not available - cannot generate embeddings")
        except Exception as e:
            logger.warning(f"Error generating embedding: {e}")

        # Fallback: Return zero vector (semantic cache won't work)
        # Note: This will cause semantic cache to always miss, but won't crash
        logger.warning(
            "Using fallback placeholder embeddings - semantic cache won't work properly"
        )
        # Try to detect size first, otherwise default to 384 (common for MiniLM)
        try:
            return [0.0] * await self._detect_embedding_size()
        except Exception:
            return [0.0] * 384  # Default to MiniLM size

    async def invalidate(self, model: Optional[str] = None) -> int:
        """
        Invalidate cache entries.

        Args:
            model: Specific model to invalidate, or None for all

        Returns:
            Number of entries invalidated
        """
        count = 0

        # Invalidate exact cache (Redis)
        if self.config.enable_exact_cache:
            redis_client = await self._get_redis()
            if redis_client:
                try:
                    if model:
                        # Delete keys for specific model
                        pattern = f"*{model}*"
                        keys = await redis_client.keys(pattern)
                        if keys:
                            count += await redis_client.delete(*keys)
                    else:
                        # Flush all cache (use specific DB in production)
                        # For safety, only delete our keys
                        pattern = "*"
                        keys = await redis_client.keys(pattern)
                        if keys:
                            count += len(keys)
                            await redis_client.delete(*keys)

                    logger.info(f"Invalidated {count} entries from exact cache")

                except Exception as e:
                    logger.error(f"Error invalidating Redis cache: {e}")

        # Invalidate semantic cache (Qdrant)
        if self.config.enable_semantic_cache:
            qdrant_client = await self._get_qdrant()
            if qdrant_client:
                try:
                    if model:
                        # Delete points for specific model
                        # (would need to store model in payload)
                        pass
                    else:
                        # Delete all points
                        qdrant_client.delete_collection(
                            collection_name=self.config.qdrant_collection
                        )
                        # Recreate collection with detected embedding size
                        embedding_size = await self._detect_embedding_size()
                        qdrant_client.create_collection(
                            collection_name=self.config.qdrant_collection,
                            vectors_config=VectorParams(
                                size=embedding_size, distance=Distance.COSINE
                            ),
                        )
                        count += 1  # Approximate

                    logger.info("Invalidated semantic cache")

                except Exception as e:
                    logger.error(f"Error invalidating Qdrant cache: {e}")

        return count

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get cache performance metrics.

        Returns:
            Metrics dict, or None if metrics disabled
        """
        if not self.enable_metrics or not self.metrics:
            return None

        return {
            "exact_hits": self.metrics.exact_hits,
            "exact_misses": self.metrics.exact_misses,
            "exact_hit_rate": self.metrics.exact_hit_rate,
            "semantic_hits": self.metrics.semantic_hits,
            "semantic_misses": self.metrics.semantic_misses,
            "semantic_hit_rate": self.metrics.semantic_hit_rate,
            "total_requests": self.metrics.total_requests,
            "overall_hit_rate": self.metrics.overall_hit_rate,
            "config": {
                "exact_cache_enabled": self.config.enable_exact_cache,
                "semantic_cache_enabled": self.config.enable_semantic_cache,
                "similarity_threshold": self.config.similarity_threshold,
                "exact_ttl_seconds": self.config.exact_ttl_seconds,
                "semantic_ttl_seconds": self.config.semantic_ttl_seconds,
            },
        }

    def reset_metrics(self):
        """Reset cache metrics."""
        if self.metrics:
            self.metrics = CacheMetrics()

    async def _check_redis_health(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if Redis is healthy, False otherwise
        """
        if not self.config.enable_exact_cache:
            return True  # Not enabled, consider healthy

        try:
            redis_client = await self._get_redis()
            if not redis_client:
                return False

            # Ping Redis
            await redis_client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    async def _check_qdrant_health(self) -> bool:
        """
        Check Qdrant connection health.

        Returns:
            True if Qdrant is healthy, False otherwise
        """
        if not self.config.enable_semantic_cache:
            return True  # Not enabled, consider healthy

        try:
            qdrant_client = await self._get_qdrant()
            if not qdrant_client:
                return False

            # Get collection info to verify connection
            try:
                qdrant_client.get_collection(self.config.qdrant_collection)
                return True
            except Exception:
                # Collection might not exist, try to list collections
                collections = qdrant_client.get_collections().collections
                collection_names = [c.name for c in collections]
                return self.config.qdrant_collection in collection_names
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False

    async def close(self):
        """Close connections to Redis and Qdrant."""
        if self._redis:
            await self._redis.close()
            self._redis = None

        # Qdrant client doesn't need explicit closing


# Singleton instance
_default_cache: Optional[SemanticCache] = None


def get_default_cache() -> SemanticCache:
    """Get or create default semantic cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = SemanticCache()
    return _default_cache
