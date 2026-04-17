# modules/services/ai-inference/ai_inference_gateway/utils/redis_client.py
import asyncio
import time
import logging
from typing import Optional, Union
from redis.asyncio import Redis, ConnectionError


logger = logging.getLogger(__name__)


class InMemoryFallback:
    """
    In-memory fallback storage when Redis is unavailable.

    This provides graceful degradation by keeping data in memory
    when Redis cannot be reached. Note that data is not persisted
    across restarts and TTL expiration is approximate.
    """

    def __init__(self):
        self._storage: dict[str, tuple[str, Optional[float]]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """
        Set a key-value pair with optional expiry.

        Args:
            key: The key to set
            value: The value to store
            ex: Optional expiry time in seconds

        Returns:
            True if successful
        """
        async with self._lock:
            expire_at = time.time() + ex if ex else None
            self._storage[key] = (value, expire_at)
            return True

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value by key.

        Args:
            key: The key to retrieve

        Returns:
            The value if found and not expired, None otherwise
        """
        async with self._lock:
            if key not in self._storage:
                return None

            value, expire_at = self._storage[key]

            # Check if expired
            if expire_at and time.time() > expire_at:
                del self._storage[key]
                return None

            return value

    async def delete(self, key: str) -> bool:
        """
        Delete a key.

        Args:
            key: The key to delete

        Returns:
            True if the key was deleted, False if it didn't exist
        """
        async with self._lock:
            if key in self._storage:
                del self._storage[key]
                return True
            return False

    async def incrby(self, key: str, amount: int) -> int:
        """
        Increment a key by a given amount.

        Args:
            key: The key to increment
            amount: The amount to increment by

        Returns:
            The new value
        """
        async with self._lock:
            current = await self.get(key)
            if current is None:
                new_value = amount
            else:
                try:
                    new_value = int(current) + amount
                except ValueError:
                    new_value = amount

            await self.set(key, str(new_value))
            return new_value

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiry time for a key.

        Args:
            key: The key to set expiry for
            seconds: Seconds until expiry

        Returns:
            True if successful, False if key doesn't exist
        """
        async with self._lock:
            if key not in self._storage:
                return False

            value, _ = self._storage[key]
            expire_at = time.time() + seconds
            self._storage[key] = (value, expire_at)
            return True

    async def close(self):
        """Close the fallback storage (no-op for in-memory)."""
        pass


class RedisClient:
    """
    Redis client with automatic fallback to in-memory storage.

    This client automatically falls back to in-memory storage when
    Redis is unavailable, providing graceful degradation.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """
        Initialize the Redis client.

        Args:
            redis_url: Redis connection URL
        """
        self.redis_url = redis_url
        self._redis: Optional[Redis] = None
        self._fallback: Optional[InMemoryFallback] = None
        self._backend: Union[Redis, InMemoryFallback]
        self._connected = False

    async def connect(self) -> bool:
        """
        Connect to Redis, falling back to in-memory if unavailable.

        Returns:
            True if connected to Redis, False if using fallback
        """
        try:
            self._redis = Redis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            self._backend = self._redis
            self._connected = True
            logger.info(f"Connected to Redis at {self.redis_url}")
            return True
        except (ConnectionError, OSError) as e:
            logger.warning(
                f"Failed to connect to Redis at {self.redis_url}: {e}. "
                "Falling back to in-memory storage."
            )
            self._fallback = InMemoryFallback()
            self._backend = self._fallback
            self._redis = None
            self._connected = False
            return False

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """
        Set a key-value pair with optional expiry.

        Args:
            key: The key to set
            value: The value to store
            ex: Optional expiry time in seconds

        Returns:
            True if successful
        """
        if isinstance(self._backend, Redis):
            return await self._backend.set(key, value, ex=ex)
        else:
            return await self._backend.set(key, value, ex=ex)

    async def get(self, key: str) -> Optional[str]:
        """
        Get a value by key.

        Args:
            key: The key to retrieve

        Returns:
            The value if found, None otherwise
        """
        if isinstance(self._backend, Redis):
            return await self._backend.get(key)
        else:
            return await self._backend.get(key)

    async def delete(self, key: str) -> bool:
        """
        Delete a key.

        Args:
            key: The key to delete

        Returns:
            True if the key was deleted, False if it didn't exist
        """
        if isinstance(self._backend, Redis):
            result = await self._backend.delete(key)
            return result > 0
        else:
            return await self._backend.delete(key)

    async def incrby(self, key: str, amount: int) -> int:
        """
        Increment a key by a given amount.

        Args:
            key: The key to increment
            amount: The amount to increment by

        Returns:
            The new value
        """
        if isinstance(self._backend, Redis):
            return await self._backend.incrby(key, amount)
        else:
            return await self._backend.incrby(key, amount)

    async def expire(self, key: str, seconds: int) -> bool:
        """
        Set expiry time for a key.

        Args:
            key: The key to set expiry for
            seconds: Seconds until expiry

        Returns:
            True if successful, False if key doesn't exist
        """
        if isinstance(self._backend, Redis):
            return await self._backend.expire(key, seconds)
        else:
            return await self._backend.expire(key, seconds)

    async def close(self):
        """Close the Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._fallback:
            await self._fallback.close()

    @property
    def backend(self) -> Union[Redis, InMemoryFallback]:
        """Get the current backend (Redis or InMemoryFallback)."""
        return self._backend

    @property
    def is_using_fallback(self) -> bool:
        """Check if currently using in-memory fallback."""
        return isinstance(self._backend, InMemoryFallback)
