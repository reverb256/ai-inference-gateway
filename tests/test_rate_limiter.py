# modules/services/ai-inference/ai_inference_gateway/tests/test_rate_limiter.py
import pytest
from unittest.mock import Mock
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.rate_limiter import RateLimiterMiddleware
from ai_inference_gateway.config import RateLimitingConfig


@pytest.mark.asyncio
async def test_rate_limiter_estimates_tokens():
    """Test that rate limiter estimates tokens correctly"""
    config = RateLimitingConfig(enabled=True)
    middleware = RateLimiterMiddleware(config)

    # Test token estimation (4 chars per token)
    text = "Hello, world!"  # 13 characters
    tokens = middleware.estimate_tokens(text)
    expected = 13 / 4  # ~3.25 tokens
    assert abs(tokens - expected) < 0.1


@pytest.mark.asyncio
async def test_rate_limiter_allows_within_limits():
    """Test that rate limiter allows requests within limits"""
    config = RateLimitingConfig(
        enabled=True, tokens_per_minute=100, tokens_per_hour=1000, tokens_per_day=10000
    )
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    context = {"request_body": {"messages": [{"role": "user", "content": "Hello"}]}}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_minute_limit():
    """Test that rate limiter blocks requests exceeding minute limit"""
    config = RateLimitingConfig(
        enabled=True,
        tokens_per_minute=10,  # Very low limit for testing
        tokens_per_hour=1000,
        tokens_per_day=10000,
    )
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    # First request should pass
    context1 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }
    should_continue, error = await middleware.process_request(request, context1)
    assert should_continue is True
    assert error is None

    # Second request should also pass
    context2 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }
    should_continue, error = await middleware.process_request(request, context2)
    assert should_continue is True
    assert error is None

    # Third request should exceed limit (15 tokens total > 10 limit)
    context3 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }
    should_continue, error = await middleware.process_request(request, context3)
    assert should_continue is False
    assert isinstance(error, HTTPException)
    assert error.status_code == 429
    assert "rate limit" in error.detail.lower()


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_hour_limit():
    """Test that rate limiter blocks requests exceeding hour limit"""
    config = RateLimitingConfig(
        enabled=True,
        tokens_per_minute=1000,
        tokens_per_hour=20,  # Very low limit for testing
        tokens_per_day=10000,
    )
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    # Make requests that exceed hour limit
    for i in range(5):  # 5 requests * 5 tokens = 25 tokens > 20 limit
        context = {
            "request_body": {
                "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
            }
        }
        should_continue, error = await middleware.process_request(request, context)

        if i < 4:  # First 4 requests should pass (20 tokens)
            assert should_continue is True, f"Request {i+1} should pass"
            assert error is None
        else:  # 5th request should fail (25 tokens > 20 limit)
            assert should_continue is False, "Request 5 should fail"
            assert isinstance(error, HTTPException)
            assert error.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_blocks_over_day_limit():
    """Test that rate limiter blocks requests exceeding day limit"""
    config = RateLimitingConfig(
        enabled=True,
        tokens_per_minute=1000,
        tokens_per_hour=1000,
        tokens_per_day=15,  # Very low limit for testing
    )
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    # Make requests that exceed day limit
    for i in range(4):  # 4 requests * 5 tokens = 20 tokens > 15 limit
        context = {
            "request_body": {
                "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
            }
        }
        should_continue, error = await middleware.process_request(request, context)

        if i < 3:  # First 3 requests should pass (15 tokens)
            assert should_continue is True, f"Request {i+1} should pass"
            assert error is None
        else:  # 4th request should fail (20 tokens > 15 limit)
            assert should_continue is False, "Request 4 should fail"
            assert isinstance(error, HTTPException)
            assert error.status_code == 429


@pytest.mark.asyncio
async def test_rate_limiter_per_api_key_limits():
    """Test that rate limiter tracks limits per API key"""
    config = RateLimitingConfig(enabled=True, tokens_per_minute=10)
    middleware = RateLimiterMiddleware(config)

    # Request with key1
    request1 = Mock(spec=Request)
    request1.headers = {"Authorization": "Bearer key1"}
    request1.state = Mock()

    context1 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }

    should_continue, error = await middleware.process_request(request1, context1)
    assert should_continue is True

    # Request with key2 should have independent limit
    request2 = Mock(spec=Request)
    request2.headers = {"Authorization": "Bearer key2"}
    request2.state = Mock()

    context2 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }

    should_continue, error = await middleware.process_request(request2, context2)
    assert should_continue is True

    # key1 should now be at limit
    context3 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }

    should_continue, error = await middleware.process_request(request1, context3)
    assert should_continue is False

    # But key2 should still have quota
    context4 = {
        "request_body": {
            "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
        }
    }

    should_continue, error = await middleware.process_request(request2, context4)
    assert should_continue is True


@pytest.mark.asyncio
async def test_rate_limiter_custom_quotas():
    """Test that rate limiter supports custom quotas per API key"""
    config = RateLimitingConfig(enabled=True, tokens_per_minute=10)
    middleware = RateLimiterMiddleware(config)

    # Set custom quota for specific key
    middleware.set_custom_quota("premium-key", tokens_per_minute=100)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer premium-key"}
    request.state = Mock()

    # Make many requests that would exceed default limit
    for i in range(15):  # 15 * 5 = 75 tokens, exceeds default 10 but under custom 100
        context = {
            "request_body": {
                "messages": [{"role": "user", "content": "x" * 20}]  # ~5 tokens
            }
        }
        should_continue, error = await middleware.process_request(request, context)
        assert should_continue is True, f"Request {i+1} should pass with custom quota"


@pytest.mark.asyncio
async def test_rate_limiter_disabled_skips_checking():
    """Test that disabled rate limiter skips all checks"""
    config = RateLimitingConfig(enabled=False)
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    # Make many requests without any limits
    for i in range(100):
        context = {
            "request_body": {"messages": [{"role": "user", "content": "x" * 1000}]}
        }
        should_continue, error = await middleware.process_request(request, context)
        assert should_continue is True
        assert error is None


def test_rate_limiter_enabled_property():
    """Test that enabled property reflects config"""
    config_enabled = RateLimitingConfig(enabled=True)
    middleware_enabled = RateLimiterMiddleware(config_enabled)
    assert middleware_enabled.enabled is True

    config_disabled = RateLimitingConfig(enabled=False)
    middleware_disabled = RateLimiterMiddleware(config_disabled)
    assert middleware_disabled.enabled is False


@pytest.mark.asyncio
async def test_rate_limiter_handles_missing_auth():
    """Test that rate limiter handles missing authentication"""
    config = RateLimitingConfig(enabled=True)
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {}  # No auth header
    request.state = Mock()

    context = {"request_body": {"messages": [{"role": "user", "content": "Hello"}]}}

    should_continue, error = await middleware.process_request(request, context)

    # Should use default key when no auth provided
    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_rate_limiter_handles_missing_request_body():
    """Test that rate limiter handles missing request body"""
    config = RateLimitingConfig(enabled=True)
    middleware = RateLimiterMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    context = {}  # No request_body

    should_continue, error = await middleware.process_request(request, context)

    # Should allow through with minimal token count
    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_rate_limiter_redis_fallback():
    """Test that rate limiter falls back to in-memory when Redis unavailable"""
    config = RateLimitingConfig(
        enabled=True,
        backend="redis",
        redis_url="redis://invalid:9999",  # Invalid URL to force fallback
    )
    middleware = RateLimiterMiddleware(config)
    await middleware.connect()

    # Should still work with in-memory fallback
    request = Mock(spec=Request)
    request.headers = {"Authorization": "Bearer test-key"}
    request.state = Mock()

    context = {"request_body": {"messages": [{"role": "user", "content": "Hello"}]}}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None

    await middleware.close()
