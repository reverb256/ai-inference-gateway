"""
Tests for MCP Tool Schema Caching.

Tests the ToolSchemaCache which provides TTL-based caching
for MCP tool schemas to reduce network calls.
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from ai_inference_gateway.mcp_cache import ToolSchemaCache, CachedSchema, get_cache


# ============================================================================
# Test Cache Basic Operations
# ============================================================================


class TestToolSchemaCacheBasic:
    """Tests for basic cache operations."""

    @pytest.fixture
    def cache(self):
        """Create cache instance for testing."""
        return ToolSchemaCache(default_ttl_seconds=60, enable_metrics=True)

    @pytest.mark.asyncio
    async def test_cache_hit(self, cache):
        """Test cache hit scenario."""
        server_name = "test-server"
        tools = [{"name": "tool1", "description": "Test tool"}]

        async def fetch_func():
            return tools

        # First call - cache miss
        result1 = await cache.get_tools(server_name, fetch_func)
        assert result1 == tools
        assert cache.cache_misses == 1

        # Second call - cache hit
        result2 = await cache.get_tools(server_name, fetch_func)
        assert result2 == tools
        assert cache.cache_hits == 1

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss scenario."""
        server_name = "test-server"
        tools = [{"name": "tool1", "description": "Test tool"}]

        async def fetch_func():
            return tools

        result = await cache.get_tools(server_name, fetch_func)
        assert result == tools
        assert cache.cache_misses == 1
        assert cache.cache_hits == 0

    @pytest.mark.asyncio
    async def test_cache_key_generation(self, cache):
        """Test that cache keys are generated correctly."""
        server_name1 = "server-1"
        server_name2 = "server-2"

        tools1 = [{"name": "tool1"}]
        tools2 = [{"name": "tool2"}]

        async def fetch1():
            return tools1

        async def fetch2():
            return tools2

        # Cache both servers
        await cache.get_tools(server_name1, fetch1)
        await cache.get_tools(server_name2, fetch2)

        # Should have 2 separate cache entries
        assert len(cache.cache) == 2
        assert f"tools:{server_name1}" in cache.cache
        assert f"tools:{server_name2}" in cache.cache


# ============================================================================
# Test TTL and Expiration
# ============================================================================


class TestCacheTTL:
    """Tests for TTL-based cache expiration."""

    @pytest.fixture
    def short_ttl_cache(self):
        """Create cache with short TTL for testing."""
        return ToolSchemaCache(
            default_ttl_seconds=1, enable_metrics=True  # 1 second TTL
        )

    @pytest.mark.asyncio
    async def test_cache_freshness(self, short_ttl_cache):
        """Test cache freshness check."""
        server_name = "test-server"
        tools = [{"name": "tool1"}]

        async def fetch_func():
            return tools

        # Store in cache
        await short_ttl_cache.get_tools(server_name, fetch_func)
        cache_key = short_ttl_cache._make_cache_key(server_name)

        # Check freshness immediately
        cached = short_ttl_cache.cache[cache_key]
        assert cached.is_fresh() is True

    @pytest.mark.asyncio
    async def test_cache_expiration(self, short_ttl_cache):
        """Test cache expiration after TTL."""

        server_name = "test-server"
        tools = [{"name": "tool1"}]
        fetch_count = [0]

        async def fetch_func():
            fetch_count[0] += 1
            return tools

        # First fetch
        await short_ttl_cache.get_tools(server_name, fetch_func)
        assert fetch_count[0] == 1

        # Wait for TTL to expire
        await asyncio.sleep(1.1)

        # Second fetch should trigger cache miss and refetch
        await short_ttl_cache.get_tools(server_name, fetch_func)
        assert fetch_count[0] == 2

    @pytest.mark.asyncio
    async def test_cache_age_calculation(self):
        """Test cache age calculation."""
        cache = ToolSchemaCache(default_ttl_seconds=60)
        server_name = "test-server"
        tools = [{"name": "tool1"}]

        async def fetch_func():
            return tools

        await cache.get_tools(server_name, fetch_func)
        cache_key = cache._make_cache_key(server_name)
        cached = cache.cache[cache_key]

        # Age should be very small (< 1 second)
        assert cached.age_seconds() < 1.0

        # Wait a bit
        await asyncio.sleep(0.1)

        # Age should increase
        age_after_wait = cached.age_seconds()
        assert age_after_wait >= 0.1


# ============================================================================
# Test Cache Invalidation
# ============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation operations."""

    @pytest.fixture
    def populated_cache(self):
        """Create cache with pre-populated data."""
        cache = ToolSchemaCache(default_ttl_seconds=60)

        # Pre-populate cache
        tools1 = [{"name": "tool1"}]
        tools2 = [{"name": "tool2"}]

        async def fetch1():
            return tools1

        async def fetch2():
            return tools2

        # Use _set directly to avoid async
        cache.cache["tools:server1"] = CachedSchema(
            schema=tools1,
            cached_at=datetime.now(),
            ttl=timedelta(seconds=60),
            server_name="server1",
        )
        cache.cache["tools:server2"] = CachedSchema(
            schema=tools2,
            cached_at=datetime.now(),
            ttl=timedelta(seconds=60),
            server_name="server2",
        )

        return cache

    @pytest.mark.asyncio
    async def test_invalidate_specific_server(self, populated_cache):
        """Test invalidating specific server cache."""
        # Invalidate server1
        success = await populated_cache.invalidate("server1")

        assert success is True
        assert "tools:server1" not in populated_cache.cache
        assert "tools:server2" in populated_cache.cache

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_server(self, populated_cache):
        """Test invalidating non-existent server."""
        success = await populated_cache.invalidate("nonexistent")

        assert success is False

    @pytest.mark.asyncio
    async def test_invalidate_all(self, populated_cache):
        """Test invalidating all cache entries."""
        count = await populated_cache.invalidate_all()

        assert count == 2
        assert len(populated_cache.cache) == 0


# ============================================================================
# Test Cache Metrics
# ============================================================================


class TestCacheMetrics:
    """Tests for cache metrics tracking."""

    @pytest.mark.asyncio
    async def test_metrics_initialization(self):
        """Test that metrics are initialized correctly."""
        cache = ToolSchemaCache(enable_metrics=True)

        metrics = cache.get_metrics()

        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0
        assert metrics["total_requests"] == 0
        assert metrics["hit_rate_percent"] == 0.0
        assert metrics["miss_rate_percent"] == 0.0

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Test that metrics track correctly."""
        cache = ToolSchemaCache(enable_metrics=True)
        tools = [{"name": "tool1"}]

        async def fetch_func():
            return tools

        # First call - miss
        await cache.get_tools("server1", fetch_func)
        metrics = cache.get_metrics()
        assert metrics["cache_misses"] == 1
        assert metrics["total_requests"] == 1

        # Second call - hit
        await cache.get_tools("server1", fetch_func)
        metrics = cache.get_metrics()
        assert metrics["cache_hits"] == 1
        assert metrics["total_requests"] == 2
        assert metrics["hit_rate_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        cache = ToolSchemaCache(enable_metrics=True)
        tools = [{"name": "tool1"}]

        async def fetch_func():
            return tools

        # 3 misses, 2 hits = 40% hit rate
        await cache.get_tools("server1", fetch_func)  # miss
        await cache.get_tools("server2", fetch_func)  # miss
        await cache.get_tools("server3", fetch_func)  # miss
        await cache.get_tools("server1", fetch_func)  # hit
        await cache.get_tools("server2", fetch_func)  # hit

        metrics = cache.get_metrics()
        assert metrics["hit_rate_percent"] == 40.0
        assert metrics["miss_rate_percent"] == 60.0

    @pytest.mark.asyncio
    async def test_metrics_disabled(self):
        """Test cache with metrics disabled."""
        cache = ToolSchemaCache(enable_metrics=False)

        metrics = cache.get_metrics()

        assert metrics is None

    @pytest.mark.asyncio
    async def test_reset_metrics(self):
        """Test resetting metrics."""
        cache = ToolSchemaCache(enable_metrics=True)
        tools = [{"name": "tool1"}]

        async def fetch_func():
            return tools

        # Generate some activity
        await cache.get_tools("server1", fetch_func)
        await cache.get_tools("server1", fetch_func)

        assert cache.cache_hits > 0

        # Reset
        cache.reset_metrics()

        metrics = cache.get_metrics()
        assert metrics["cache_hits"] == 0
        assert metrics["cache_misses"] == 0


# ============================================================================
# Test Cache Warm-up
# ============================================================================


class TestCacheWarmup:
    """Tests for cache warm-up functionality."""

    @pytest.mark.asyncio
    async def test_warm_up_success(self):
        """Test successful cache warm-up."""
        cache = ToolSchemaCache(enable_metrics=False)

        servers = {
            "server1": lambda: [{"name": "tool1"}],
            "server2": lambda: [{"name": "tool2"}],
            "server3": lambda: [{"name": "tool3"}],
        }

        # Wrap in async functions
        async def fetch1():
            return servers["server1"]()

        async def fetch2():
            return servers["server2"]()

        async def fetch3():
            return servers["server3"]()

        _servers_async = {"server1": fetch1, "server2": fetch2, "server3": fetch3}  # noqa: F841

        results = await cache.warm_up(servers)

        assert results["server1"] is True
        assert results["server2"] is True
        assert results["server3"] is True
        assert len(cache.cache) == 3

    @pytest.mark.asyncio
    async def test_warm_up_with_failures(self):
        """Test warm-up with some failures."""
        cache = ToolSchemaCache(enable_metrics=False)

        async def fetch_success():
            return [{"name": "tool1"}]

        async def fetch_failure():
            raise Exception("Fetch failed")

        servers = {"server1": fetch_success, "server2": fetch_failure}

        results = await cache.warm_up(servers)

        assert results["server1"] is True
        assert results["server2"] is False
        assert len(cache.cache) == 1

    @pytest.mark.asyncio
    async def test_warm_up_concurrency(self):
        """Test that warm-up runs concurrently."""

        cache = ToolSchemaCache(enable_metrics=False)

        async def slow_fetch():
            await asyncio.sleep(0.1)
            return [{"name": "tool"}]

        servers = {f"server{i}": slow_fetch for i in range(5)}

        start = datetime.now()
        await cache.warm_up(servers, max_concurrency=5)
        elapsed = (datetime.now() - start).total_seconds()

        # Should run concurrently, not sequentially
        # Sequential would be 5 * 0.1 = 0.5s
        # Concurrent should be ~0.1s
        assert elapsed < 0.3


# ============================================================================
# Test Error Handling
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in cache operations."""

    @pytest.mark.asyncio
    async def test_fetch_function_error(self):
        """Test handling of fetch function errors."""
        cache = ToolSchemaCache(enable_metrics=True)

        async def failing_fetch():
            raise ValueError("Fetch failed")

        with pytest.raises(ValueError):
            await cache.get_tools("server1", failing_fetch)

        # Should record error in metrics
        assert cache.cache_errors > 0

    @pytest.mark.asyncio
    async def test_fetch_returns_none(self):
        """Test handling when fetch returns None."""
        cache = ToolSchemaCache(enable_metrics=True)

        async def none_fetch():
            return None

        result = await cache.get_tools("server1", none_fetch)

        assert result is None
        assert cache.cache_errors > 0

    @pytest.mark.asyncio
    async def test_stale_cache_fallback(self):
        """Test stale cache fallback on error."""
        cache = ToolSchemaCache(default_ttl_seconds=1, enable_metrics=False)

        tools = [{"name": "tool1"}]

        async def initial_fetch():
            return tools

        async def failing_fetch():
            raise ValueError("Fetch failed")

        # Initial successful fetch
        result1 = await cache.get_tools("server1", initial_fetch)
        assert result1 == tools

        # Wait for cache to go stale
        await asyncio.sleep(1.1)

        # Failing fetch should return stale cache
        result2 = await cache.get_tools("server1", failing_fetch)
        assert result2 == tools  # Should return stale data


# ============================================================================
# Test Singleton
# ============================================================================


class TestSingleton:
    """Tests for singleton cache instance."""

    def test_get_cache_returns_singleton(self):
        """Test that get_cache returns same instance."""
        cache1 = get_cache()
        cache2 = get_cache()

        assert cache1 is cache2

    def test_get_cache_with_custom_ttl(self):
        """Test get_cache with custom TTL."""
        cache = get_cache(ttl_seconds=120)

        assert cache.default_ttl.total_seconds() == 120
