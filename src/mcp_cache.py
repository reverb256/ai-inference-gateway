"""
MCP Tool Schema Caching

Cache MCP tool schemas to reduce network calls and improve performance.
Implements TTL-based caching with warm-up and manual invalidation.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CachedSchema:
    """Cached tool schema with metadata."""

    schema: List[Dict[str, Any]]
    cached_at: datetime
    ttl: timedelta
    etag: Optional[str] = None
    server_name: str = ""

    def is_fresh(self) -> bool:
        """Check if cache entry is still fresh."""
        return datetime.now() < (self.cached_at + self.ttl)

    def age_seconds(self) -> float:
        """Get cache age in seconds."""
        return (datetime.now() - self.cached_at).total_seconds()


class ToolSchemaCache:
    """
    Cache MCP tool schemas with TTL-based invalidation.

    Features:
    - In-memory caching with configurable TTL (default 5 minutes)
    - Per-server cache keys
    - Cache hit/miss metrics tracking
    - Manual invalidation support
    - Warm-up on startup
    """

    def __init__(self, default_ttl_seconds: int = 300, enable_metrics: bool = True):
        """
        Initialize tool schema cache.

        Args:
            default_ttl_seconds: Default TTL for cache entries (default: 300s = 5 min)
            enable_metrics: Track cache hit/miss metrics
        """
        self.cache: Dict[str, CachedSchema] = {}
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        self.enable_metrics = enable_metrics

        # Metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_errors = 0

        logger.info(
            f"ToolSchemaCache initialized (TTL={default_ttl_seconds}s, "
            f"metrics={enable_metrics})"
        )

    def _make_cache_key(self, server_name: str) -> str:
        """Create cache key for a server."""
        return f"tools:{server_name}"

    async def get_tools(
        self, server_name: str, fetch_func: callable, force_refresh: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get tools from cache or fetch from server.

        Args:
            server_name: MCP server name
            fetch_func: Async function to fetch tools from server
            force_refresh: Bypass cache and fetch fresh data

        Returns:
            List of tool definitions, or None if fetch fails
        """
        cache_key = self._make_cache_key(server_name)

        # Check cache for fresh entry
        if not force_refresh and cache_key in self.cache:
            cached = self.cache[cache_key]

            if cached.is_fresh():
                self.cache_hits += 1
                age = cached.age_seconds()

                logger.debug(
                    f"Cache HIT: {server_name} "
                    f"(age={age:.1f}s, TTL={cached.ttl.total_seconds()}s)"
                )

                return cached.schema
            else:
                logger.debug(
                    f"Cache STALE: {server_name} "
                    f"(age={age:.1f}s > TTL={cached.ttl.total_seconds()}s)"
                )

        # Cache miss or stale - fetch from server
        self.cache_misses += 1
        logger.debug(f"Cache MISS: {server_name} - fetching from server")

        try:
            tools = await fetch_func()

            if tools:
                # Cache the fresh data
                self.cache[cache_key] = CachedSchema(
                    schema=tools,
                    cached_at=datetime.now(),
                    ttl=self.default_ttl,
                    server_name=server_name,
                )

                logger.info(
                    f"Cached {len(tools)} tools from {server_name} "
                    f"(TTL={self.default_ttl.total_seconds()}s)"
                )

                return tools
            else:
                logger.warning(f"No tools returned from {server_name}")
                self.cache_errors += 1
                return None

        except Exception as e:
            logger.error(f"Error fetching tools from {server_name}: {e}")
            self.cache_errors += 1

            # Return stale cache if available (fallback)
            if cache_key in self.cache:
                logger.info(f"Returning stale cache for {server_name}")
                return self.cache[cache_key].schema

            return None

    async def invalidate(self, server_name: str) -> bool:
        """
        Invalidate cache for a specific server.

        Args:
            server_name: Server to invalidate

        Returns:
            True if cache entry existed and was removed
        """
        cache_key = self._make_cache_key(server_name)

        if cache_key in self.cache:
            del self.cache[cache_key]
            logger.info(f"Invalidated cache for {server_name}")
            return True

        logger.debug(f"No cache to invalidate for {server_name}")
        return False

    async def invalidate_all(self) -> int:
        """
        Invalidate all cached schemas.

        Returns:
            Number of cache entries invalidated
        """
        count = len(self.cache)
        self.cache.clear()

        if count > 0:
            logger.info(f"Invalidated all cache entries ({count} servers)")

        return count

    async def warm_up(
        self, servers: Dict[str, callable], max_concurrency: int = 5
    ) -> Dict[str, bool]:
        """
        Pre-fetch tool schemas for all servers.

        Args:
            servers: Dict of {server_name: fetch_function}
            max_concurrency: Max concurrent fetches

        Returns:
            Dict of {server_name: success_status}
        """
        logger.info(f"Warming up cache for {len(servers)} servers...")

        results = {}
        semaphore = asyncio.Semaphore(max_concurrency)

        async def warm_server(name: str, fetch_func: callable):
            async with semaphore:
                try:
                    tools = await self.get_tools(name, fetch_func, force_refresh=True)
                    results[name] = tools is not None

                    if results[name]:
                        logger.debug(f"✓ Warmed up {name}")
                    else:
                        logger.warning(f"✗ Failed to warm up {name}")

                except Exception as e:
                    logger.error(f"✗ Error warming up {name}: {e}")
                    results[name] = False

        # Run all warm-ups in parallel
        tasks = [warm_server(name, fetch_func) for name, fetch_func in servers.items()]

        await asyncio.gather(*tasks, return_exceptions=True)

        successful = sum(1 for success in results.values() if success)
        logger.info(f"Cache warm-up complete: {successful}/{len(servers)} successful")

        return results

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cache performance metrics.

        Returns:
            Dict with hit_rate, miss_rate, cache size, etc.
        """
        total_requests = self.cache_hits + self.cache_misses

        if total_requests > 0:
            hit_rate = (self.cache_hits / total_requests) * 100
            miss_rate = (self.cache_misses / total_requests) * 100
        else:
            hit_rate = 0.0
            miss_rate = 0.0

        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_errors": self.cache_errors,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "miss_rate_percent": miss_rate,
            "cache_size": len(self.cache),
            "cached_servers": list(self.cache.keys()),
        }

    def get_cache_info(self, server_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed cache info for a specific server.

        Args:
            server_name: Server to query

        Returns:
            Cache info dict, or None if not cached
        """
        cache_key = self._make_cache_key(server_name)

        if cache_key not in self.cache:
            return None

        cached = self.cache[cache_key]

        return {
            "server_name": server_name,
            "cached_at": cached.cached_at.isoformat(),
            "age_seconds": cached.age_seconds(),
            "ttl_seconds": cached.ttl.total_seconds(),
            "is_fresh": cached.is_fresh(),
            "tool_count": len(cached.schema),
            "etag": cached.etag,
        }


# Singleton instance
_cache: Optional[ToolSchemaCache] = None


def get_cache(ttl_seconds: int = 300, enable_metrics: bool = True) -> ToolSchemaCache:
    """
    Get or create the singleton cache instance.

    Args:
        ttl_seconds: Default TTL for cache entries
        enable_metrics: Enable hit/miss metrics tracking

    Returns:
        ToolSchemaCache instance
    """
    global _cache

    if _cache is None:
        _cache = ToolSchemaCache(
            default_ttl_seconds=ttl_seconds, enable_metrics=enable_metrics
        )

    return _cache
