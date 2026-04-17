# modules/services/ai-inference/ai_inference_gateway/tests/test_redis_client.py
import pytest
from ai_inference_gateway.utils.redis_client import RedisClient, InMemoryFallback


@pytest.mark.asyncio
async def test_redis_client_set_get():
    """Test Redis client basic set and get operations"""
    client = RedisClient(redis_url="redis://localhost:6379")
    await client.connect()

    # Test set and get
    await client.set("test_key", "test_value", ex=60)
    value = await client.get("test_key")

    assert value == "test_value"

    # Cleanup
    await client.delete("test_key")
    await client.close()


@pytest.mark.asyncio
async def test_redis_client_delete():
    """Test Redis client delete operation"""
    client = RedisClient(redis_url="redis://localhost:6379")
    await client.connect()

    # Set a key
    await client.set("test_delete_key", "test_value")

    # Verify it exists
    value = await client.get("test_delete_key")
    assert value == "test_value"

    # Delete it
    await client.delete("test_delete_key")

    # Verify it's gone
    value = await client.get("test_delete_key")
    assert value is None

    await client.close()


@pytest.mark.asyncio
async def test_redis_client_incrby():
    """Test Redis client increment operation"""
    client = RedisClient(redis_url="redis://localhost:6379")
    await client.connect()

    # Set initial value
    await client.set("test_counter", "10")

    # Increment
    result = await client.incrby("test_counter", 5)
    assert result == 15

    # Verify
    value = await client.get("test_counter")
    assert value == "15"

    # Cleanup
    await client.delete("test_counter")
    await client.close()


@pytest.mark.asyncio
async def test_redis_client_expire():
    """Test Redis client expire operation"""
    client = RedisClient(redis_url="redis://localhost:6379")
    await client.connect()

    # Set a key without expiry
    await client.set("test_expire_key", "test_value")

    # Add expiry
    await client.expire("test_expire_key", 60)

    # Key should still exist
    value = await client.get("test_expire_key")
    assert value == "test_value"

    # Cleanup
    await client.delete("test_expire_key")
    await client.close()


@pytest.mark.asyncio
async def test_redis_client_fallback_to_memory():
    """Test fallback to in-memory when Redis is unavailable"""
    # Use an invalid Redis URL to force fallback
    client = RedisClient(redis_url="redis://invalid:9999")
    await client.connect()

    # Should fall back to in-memory
    assert isinstance(client.backend, InMemoryFallback)

    # Test basic operations still work
    await client.set("fallback_test", "fallback_value")
    value = await client.get("fallback_test")

    assert value == "fallback_value"

    await client.close()


@pytest.mark.asyncio
async def test_in_memory_fallback_set_get():
    """Test InMemoryFallback set and get operations"""
    fallback = InMemoryFallback()

    await fallback.set("memory_key", "memory_value", ex=60)
    value = await fallback.get("memory_key")

    assert value == "memory_value"


@pytest.mark.asyncio
async def test_in_memory_fallback_delete():
    """Test InMemoryFallback delete operation"""
    fallback = InMemoryFallback()

    await fallback.set("memory_delete_key", "value")
    value = await fallback.get("memory_delete_key")
    assert value == "value"

    await fallback.delete("memory_delete_key")
    value = await fallback.get("memory_delete_key")
    assert value is None


@pytest.mark.asyncio
async def test_in_memory_fallback_incrby():
    """Test InMemoryFallback increment operation"""
    fallback = InMemoryFallback()

    await fallback.set("memory_counter", "10")
    result = await fallback.incrby("memory_counter", 5)

    assert result == 15
    value = await fallback.get("memory_counter")
    assert value == "15"


@pytest.mark.asyncio
async def test_in_memory_fallback_expire():
    """Test InMemoryFallback expire operation (TTL not enforced)"""
    fallback = InMemoryFallback()

    await fallback.set("memory_expire", "value")
    await fallback.expire("memory_expire", 1)  # 1 second

    # InMemoryFallback doesn't enforce TTL, just stores it
    value = await fallback.get("memory_expire")
    assert value == "value"


@pytest.mark.asyncio
async def test_redis_client_get_nonexistent_key():
    """Test getting a nonexistent key returns None"""
    client = RedisClient(redis_url="redis://localhost:6379")
    await client.connect()

    value = await client.get("nonexistent_key")

    assert value is None

    await client.close()


@pytest.mark.asyncio
async def test_in_memory_fallback_get_nonexistent_key():
    """Test InMemoryFallback get nonexistent key returns None"""
    fallback = InMemoryFallback()

    value = await fallback.get("nonexistent_key")

    assert value is None
