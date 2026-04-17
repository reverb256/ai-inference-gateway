"""
SearXNG Search History with Redis Backend

Provides persistent query and result history with Redis-backed storage.
Features TTL-based expiration and user-specific history tracking.

Features:
- Redis-backed query/result history
- 30-day TTL for history entries
- User-specific history isolation
- Search history aggregation
- Saved searches for recurring queries
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from hashlib import sha256

logger = logging.getLogger(__name__)

# History configuration
HISTORY_TTL_DAYS = 30
HISTORY_KEY_PREFIX = "searxng:history"
SAVED_SEARCHES_PREFIX = "searxng:saved"
USER_PREFIX = "user"

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None
    logger.warning("Redis not available - history features disabled")


@dataclass
class HistoryEntry:
    """A single search history entry."""

    query: str
    timestamp: float
    results_count: int
    domain: str
    engines_used: List[str]
    cached: bool
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    quality_scores: List[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["quality_scores"] = self.quality_scores or []
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SavedSearch:
    """A saved search for recurring queries."""

    name: str
    query: str
    domain: Optional[str] = None
    max_results: int = 10
    created_at: float = None
    last_run: Optional[float] = None
    run_count: int = 0
    user_id: Optional[str] = None
    tags: List[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedSearch":
        """Create from dictionary."""
        return cls(**data)


class SearXNGHistoryManager:
    """
    Manages SearXNG search history with Redis backend.

    Features:
    - Query/result history with TTL
    - User-specific isolation
    - Saved searches
    - History analytics
    """

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        ttl_days: int = HISTORY_TTL_DAYS,
        default_user: str = "default",
    ):
        """
        Initialize history manager.

        Args:
            redis_url: Redis connection URL
            ttl_days: Days to keep history entries
            default_user: Default user ID
        """
        self.redis_url = redis_url
        self.ttl = ttl_days * 86400  # Convert to seconds
        self.default_user = default_user
        self._redis: Optional[aioredis.Redis] = None

        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - history operations will be no-ops")

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        """Get or create Redis connection."""
        if not REDIS_AVAILABLE:
            return None

        if self._redis is None:
            try:
                self._redis = await aioredis.from_url(self.redis_url, decode_responses=True)
                logger.info(f"Connected to Redis at {self.redis_url}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                return None

        return self._redis

    def _make_history_key(self, user_id: str) -> str:
        """Generate history key for user."""
        return f"{HISTORY_KEY_PREFIX}:{USER_PREFIX}:{user_id}"

    def _make_entry_key(self, user_id: str, entry_id: str) -> str:
        """Generate key for specific history entry."""
        return f"{HISTORY_KEY_PREFIX}:{USER_PREFIX}:{user_id}:{entry_id}"

    def _make_saved_key(self, user_id: str, search_name: str) -> str:
        """Generate key for saved search."""
        return f"{SAVED_SEARCHES_PREFIX}:{USER_PREFIX}:{user_id}:{search_name}"

    def _make_saved_index_key(self, user_id: str) -> str:
        """Generate key for user's saved search index."""
        return f"{SAVED_SEARCHES_PREFIX}:{USER_PREFIX}:{user_id}"

    async def save_search(
        self,
        query: str,
        results: List[Dict[str, Any]],
        domain: str = "general",
        engines_used: List[str] = None,
        cached: bool = False,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """
        Save a search to history.

        Args:
            query: Search query
            results: Search results
            domain: Query domain
            engines_used: Engines that were queried
            cached: Whether results were from cache
            session_id: Optional session ID
            user_id: User ID (default_user if None)

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        if not redis:
            return False

        user_id = user_id or self.default_user

        # Create history entry
        entry = HistoryEntry(
            query=query,
            timestamp=time.time(),
            results_count=len(results),
            domain=domain,
            engines_used=engines_used or [],
            cached=cached,
            session_id=session_id,
            user_id=user_id,
            quality_scores=[r.get("quality_score", 0.0) for r in results if "quality_score" in r],
        )

        # Generate entry ID
        entry_id = sha256(f"{query}:{entry.timestamp}".encode()).hexdigest()[:16]

        try:
            # Store entry with TTL
            entry_key = self._make_entry_key(user_id, entry_id)
            await redis.setex(
                entry_key,
                self.ttl,
                json.dumps(entry.to_dict()),
            )

            # Add to user's history index (sorted set by timestamp)
            history_key = self._make_history_key(user_id)
            await redis.zadd(history_key, {entry_id: entry.timestamp})

            # Trim history to last 1000 entries
            await redis.zremrangebyrank(history_key, 0, -1001)

            logger.debug(f"Saved search '{query}' to history for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save search history: {e}")
            return False

    async def get_history(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
        since: Optional[float] = None,
        domain: Optional[str] = None,
    ) -> List[HistoryEntry]:
        """
        Get search history for user.

        Args:
            user_id: User ID
            limit: Maximum entries to return
            since: Only return entries after this timestamp
            domain: Filter by domain

        Returns:
            List of history entries
        """
        redis = await self._get_redis()
        if not redis:
            return []

        user_id = user_id or self.default_user
        history_key = self._make_history_key(user_id)

        try:
            # Get entry IDs (sorted by timestamp, newest first)
            entry_ids = await redis.zrevrange(history_key, 0, limit - 1, withscores=False)

            if not entry_ids:
                return []

            # Fetch entries
            entries = []
            for entry_id in entry_ids:
                entry_key = self._make_entry_key(user_id, entry_id)
                data = await redis.get(entry_key)

                if data:
                    entry = HistoryEntry.from_dict(json.loads(data))

                    # Apply filters
                    if since and entry.timestamp < since:
                        continue
                    if domain and entry.domain != domain:
                        continue

                    entries.append(entry)

                # Stop if we have enough
                if len(entries) >= limit:
                    break

            return entries

        except Exception as e:
            logger.error(f"Failed to get history: {e}")
            return []

    async def search_history(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[HistoryEntry]:
        """
        Search history by query text.

        Args:
            query: Search query to find in history
            user_id: User ID
            limit: Maximum results

        Returns:
            Matching history entries
        """
        redis = await self._get_redis()
        if not redis:
            return []

        user_id = user_id or self.default_user
        history_key = self._make_history_key(user_id)

        try:
            # Get all entry IDs
            entry_ids = await redis.zrevrange(history_key, 0, -1, withscores=False)

            if not entry_ids:
                return []

            # Search for matching entries
            query_lower = query.lower()
            matching_entries = []

            for entry_id in entry_ids:
                if len(matching_entries) >= limit:
                    break

                entry_key = self._make_entry_key(user_id, entry_id)
                data = await redis.get(entry_key)

                if data:
                    entry = HistoryEntry.from_dict(json.loads(data))

                    # Check if query matches
                    if query_lower in entry.query.lower():
                        matching_entries.append(entry)

            return matching_entries

        except Exception as e:
            logger.error(f"Failed to search history: {e}")
            return []

    async def save_search(
        self,
        name: str,
        query: str,
        domain: Optional[str] = None,
        max_results: int = 10,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: str = "",
    ) -> bool:
        """
        Save a search for future use.

        Args:
            name: Name for the saved search
            query: Search query
            domain: Optional domain hint
            max_results: Maximum results
            user_id: User ID
            tags: Optional tags for organization
            notes: Optional notes

        Returns:
            True if saved successfully
        """
        redis = await self._get_redis()
        if not redis:
            return False

        user_id = user_id or self.default_user

        # Create saved search
        saved = SavedSearch(
            name=name,
            query=query,
            domain=domain,
            max_results=max_results,
            user_id=user_id,
            tags=tags or [],
            notes=notes,
        )

        try:
            # Store saved search
            saved_key = self._make_saved_key(user_id, name)
            await redis.set(
                saved_key,
                json.dumps(saved.to_dict()),
            )

            # Add to index
            index_key = self._make_saved_index_key(user_id)
            await redis.sadd(index_key, name)

            logger.info(f"Saved search '{name}' for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save search: {e}")
            return False

    async def get_saved_searches(
        self, user_id: Optional[str] = None
    ) -> List[SavedSearch]:
        """
        Get all saved searches for user.

        Args:
            user_id: User ID

        Returns:
            List of saved searches
        """
        redis = await self._get_redis()
        if not redis:
            return []

        user_id = user_id or self.default_user
        index_key = self._make_saved_index_key(user_id)

        try:
            # Get saved search names
            names = await redis.smembers(index_key)

            if not names:
                return []

            # Fetch each saved search
            saved_searches = []
            for name in names:
                saved_key = self._make_saved_key(user_id, name)
                data = await redis.get(saved_key)

                if data:
                    saved_searches.append(SavedSearch.from_dict(json.loads(data)))

            # Sort by creation time
            saved_searches.sort(key=lambda s: s.created_at, reverse=True)

            return saved_searches

        except Exception as e:
            logger.error(f"Failed to get saved searches: {e}")
            return []

    async def run_saved_search(
        self,
        name: str,
        searxng_client: Any,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a saved search.

        Args:
            name: Name of saved search
            searxng_client: SearxngIntegration instance
            user_id: User ID

        Returns:
            Search results
        """
        redis = await self._get_redis()
        if not redis:
            return {"error": "Redis not available"}

        user_id = user_id or self.default_user
        saved_key = self._make_saved_key(user_id, name)

        try:
            data = await redis.get(saved_key)
            if not data:
                return {"error": f"Saved search '{name}' not found"}

            saved = SavedSearch.from_dict(json.loads(data))

            # Execute search
            result = await searxng_client.search_with_domain_routing(
                query=saved.query,
                domain=saved.domain,
                max_results=saved.max_results,
                use_cache=True,
            )

            # Update run stats
            saved.last_run = time.time()
            saved.run_count += 1

            # Save updated stats
            await redis.set(saved_key, json.dumps(saved.to_dict()))

            return {
                "name": name,
                "query": saved.query,
                "results": result.get("results", []),
                "run_count": saved.run_count,
                "last_run": saved.last_run,
            }

        except Exception as e:
            logger.error(f"Failed to run saved search: {e}")
            return {"error": str(e)}

    async def delete_saved_search(
        self, name: str, user_id: Optional[str] = None
    ) -> bool:
        """
        Delete a saved search.

        Args:
            name: Name of saved search
            user_id: User ID

        Returns:
            True if deleted successfully
        """
        redis = await self._get_redis()
        if not redis:
            return False

        user_id = user_id or self.default_user

        try:
            saved_key = self._make_saved_key(user_id, name)
            index_key = self._make_saved_index_key(user_id)

            # Remove from storage
            await redis.delete(saved_key)

            # Remove from index
            await redis.srem(index_key, name)

            logger.info(f"Deleted saved search '{name}' for user {user_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete saved search: {e}")
            return False

    async def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get history statistics for user.

        Args:
            user_id: User ID

        Returns:
            Statistics dictionary
        """
        redis = await self._get_redis()
        if not redis:
            return {}

        user_id = user_id or self.default_user
        history_key = self._make_history_key(user_id)
        index_key = self._make_saved_index_key(user_id)

        try:
            # Count history entries
            history_count = await redis.zcard(history_key)

            # Count saved searches
            saved_count = await redis.scard(index_key)

            # Get recent activity
            recent_entries = await self.get_history(user_id=user_id, limit=10)

            # Calculate stats
            domain_counts = {}
            total_results = 0
            cache_hits = 0

            for entry in recent_entries:
                domain_counts[entry.domain] = domain_counts.get(entry.domain, 0) + 1
                total_results += entry.results_count
                if entry.cached:
                    cache_hits += 1

            return {
                "user_id": user_id,
                "total_searches": history_count,
                "saved_searches": saved_count,
                "recent_domains": domain_counts,
                "recent_total_results": total_results,
                "recent_cache_hits": cache_hits,
                "recent_cache_rate": cache_hits / len(recent_entries) if recent_entries else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    async def clear_history(
        self, user_id: Optional[str] = None, before: Optional[float] = None
    ) -> int:
        """
        Clear search history.

        Args:
            user_id: User ID
            before: Only clear entries before this timestamp

        Returns:
            Number of entries cleared
        """
        redis = await self._get_redis()
        if not redis:
            return 0

        user_id = user_id or self.default_user
        history_key = self._make_history_key(user_id)

        try:
            if before is None:
                # Clear all history
                entry_ids = await redis.zrange(history_key, 0, -1)
                count = len(entry_ids)

                for entry_id in entry_ids:
                    entry_key = self._make_entry_key(user_id, entry_id)
                    await redis.delete(entry_key)

                await redis.delete(history_key)
                return count

            else:
                # Clear entries before timestamp
                entry_ids = await redis.zrangebyscore(history_key, 0, before)

                for entry_id in entry_ids:
                    entry_key = self._make_entry_key(user_id, entry_id)
                    await redis.delete(entry_key)
                    await redis.zrem(history_key, entry_id)

                return len(entry_ids)

        except Exception as e:
            logger.error(f"Failed to clear history: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None


def create_history_manager(
    redis_url: str = "redis://127.0.0.1:6379/0",
    ttl_days: int = HISTORY_TTL_DAYS,
    default_user: str = "default",
) -> SearXNGHistoryManager:
    """
    Create SearXNG history manager.

    Args:
        redis_url: Redis connection URL
        ttl_days: Days to keep history
        default_user: Default user ID

    Returns:
        Configured SearXNGHistoryManager
    """
    return SearXNGHistoryManager(
        redis_url=redis_url,
        ttl_days=ttl_days,
        default_user=default_user,
    )
