# modules/services/ai-inference/ai_inference_gateway/tests/test_circuit_breaker.py
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ai_inference_gateway.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerOpenError,
)
from ai_inference_gateway.config import CircuitBreakerConfig


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self):
        """Circuit breaker starts in CLOSED state."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=5, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        assert cb.state == CircuitBreakerState.CLOSED
        assert await cb.allow_request() is True

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        """Circuit opens after reaching failure threshold."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=3, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Record failures up to threshold
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        # Circuit should now be OPEN
        assert cb.state == CircuitBreakerState.OPEN
        assert await cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        """Circuit transitions to HALF_OPEN after timeout."""
        config = CircuitBreakerConfig(
            enabled=True,
            failure_threshold=2,
            success_threshold=2,
            timeout_seconds=1,  # Short timeout for testing
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(config.timeout_seconds + 0.1)

        # Next request should transition to HALF_OPEN
        result = await cb.allow_request()
        assert cb.state == CircuitBreakerState.HALF_OPEN
        assert result is True

    @pytest.mark.asyncio
    async def test_closes_after_success_threshold_in_half_open(self):
        """Circuit closes after reaching success threshold in HALF_OPEN."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=2, success_threshold=2, timeout_seconds=1
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)
        redis_client.delete = AsyncMock()

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        # Wait for timeout and trigger HALF_OPEN
        await asyncio.sleep(config.timeout_seconds + 0.1)
        await cb.allow_request()

        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record successes
        for _ in range(config.success_threshold):
            await cb.on_success()

        # Circuit should now be CLOSED
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_reopens_on_failure_in_half_open(self):
        """Circuit reopens if failure occurs in HALF_OPEN."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=2, success_threshold=2, timeout_seconds=1
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        # Wait for timeout and trigger HALF_OPEN
        await asyncio.sleep(config.timeout_seconds + 0.1)
        await cb.allow_request()

        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Record a failure
        await cb.on_failure()

        # Circuit should reopen
        assert cb.state == CircuitBreakerState.OPEN


class TestCircuitBreakerRedisIntegration:
    """Test circuit breaker integration with Redis."""

    @pytest.mark.asyncio
    async def test_loads_state_from_redis(self):
        """Circuit breaker loads its state from Redis on initialization."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=5, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        # Simulate OPEN state in Redis
        redis_client.get = AsyncMock(return_value="OPEN:1234567890")
        redis_client.set = AsyncMock()

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_saves_state_to_redis_on_failure(self):
        """Circuit breaker saves state to Redis on failure."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=1, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Trigger failure that opens circuit
        await cb.on_failure()

        # Verify state was saved to Redis
        redis_client.set.assert_called()
        call_args = redis_client.set.call_args_list[-1]
        assert "OPEN" in str(call_args)

    @pytest.mark.asyncio
    async def test_tracks_failure_count_in_redis(self):
        """Circuit breaker tracks failure count in Redis."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=5, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Record failures
        await cb.on_failure()
        await cb.on_failure()

        # Verify incrby was called for failure tracking
        assert redis_client.incrby.call_count >= 2


class TestCircuitBreakerAllowRequest:
    """Test allow_request method."""

    @pytest.mark.asyncio
    async def test_allows_request_when_closed(self):
        """allow_request returns True when circuit is CLOSED."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=5, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        assert await cb.allow_request() is True

    @pytest.mark.asyncio
    async def test_blocks_request_when_open(self):
        """allow_request returns False when circuit is OPEN."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=2, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        assert await cb.allow_request() is False

    @pytest.mark.asyncio
    async def test_allows_single_request_when_half_open(self):
        """allow_request allows single request in HALF_OPEN state."""
        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=2, success_threshold=2, timeout_seconds=1
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        # Wait for timeout
        await asyncio.sleep(config.timeout_seconds + 0.1)

        # First request should be allowed
        assert await cb.allow_request() is True
        assert cb.state == CircuitBreakerState.HALF_OPEN


class TestCircuitBreakerDisabled:
    """Test circuit breaker when disabled."""

    @pytest.mark.asyncio
    async def test_always_allows_when_disabled(self):
        """Circuit breaker always allows requests when disabled."""
        config = CircuitBreakerConfig(enabled=False)
        redis_client = Mock()

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        assert await cb.allow_request() is True
        # State operations should be no-ops
        await cb.on_success()
        await cb.on_failure()


class TestCircuitBreakerMiddlewareIntegration:
    """Test circuit breaker as middleware."""

    @pytest.mark.asyncio
    async def test_raises_error_when_open(self):
        """Middleware raises CircuitBreakerOpenError when circuit is open."""
        from fastapi import Request

        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=2, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)
        redis_client.set = AsyncMock()
        redis_client.incrby = AsyncMock(return_value=1)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Open the circuit
        for _ in range(config.failure_threshold):
            await cb.on_failure()

        # Create mock request
        request = Mock(spec=Request)
        request.state = Mock()

        # Process request should raise error
        with pytest.raises(CircuitBreakerOpenError):
            await cb.process_request(request, {})

    @pytest.mark.asyncio
    async def test_passes_through_when_closed(self):
        """Middleware allows requests through when circuit is closed."""
        from fastapi import Request

        config = CircuitBreakerConfig(
            enabled=True, failure_threshold=5, success_threshold=2, timeout_seconds=60
        )
        redis_client = Mock()
        redis_client.get = AsyncMock(return_value=None)

        cb = CircuitBreaker(
            service_id="test-service", config=config, redis_client=redis_client
        )

        # Create mock request
        request = Mock(spec=Request)
        request.state = Mock()

        # Process request should succeed
        should_continue, error = await cb.process_request(request, {})
        assert should_continue is True
        assert error is None
