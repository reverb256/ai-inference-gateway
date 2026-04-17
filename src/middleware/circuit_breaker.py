# modules/services/ai-inference/ai_inference_gateway/middleware/circuit_breaker.py
import asyncio
import logging
import time
from enum import Enum
from typing import Optional, Tuple

from fastapi import Request, HTTPException

from ai_inference_gateway.middleware.base import Middleware
from ai_inference_gateway.config import CircuitBreakerConfig


logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(HTTPException):
    """Error raised when circuit breaker is open."""

    def __init__(self, service_id: str, retry_after: Optional[int] = None):
        super().__init__(
            status_code=503, detail=f"Circuit breaker is open for service: {service_id}"
        )
        self.service_id = service_id
        self.retry_after = retry_after


class CircuitBreaker(Middleware):
    """
    Circuit breaker middleware with state machine.

    Implements a three-state circuit breaker:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests are blocked
    - HALF_OPEN: Testing if service has recovered

    State transitions:
    CLOSED -> OPEN: After failure_threshold failures
    OPEN -> HALF_OPEN: After timeout_seconds
    HALF_OPEN -> CLOSED: After success_threshold successes
    HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        service_id: str,
        config: CircuitBreakerConfig,
        redis_client: Optional[object] = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            service_id: Unique identifier for this service/backend
            config: Circuit breaker configuration
            redis_client: Optional Redis client for distributed state
        """
        self.service_id = service_id
        self.config = config
        self._redis = redis_client

        # State keys for Redis
        self._state_key = f"circuit_breaker:{service_id}:state"
        self._failure_count_key = f"circuit_breaker:{service_id}:failures"
        self._success_count_key = f"circuit_breaker:{service_id}:successes"
        self._last_failure_time_key = f"circuit_breaker:{service_id}:last_failure"

        # In-memory state (used when Redis unavailable)
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0

        # Load state from Redis if available
        if self._redis:
            asyncio.create_task(self._load_state())

    @property
    def enabled(self) -> bool:
        """Check if circuit breaker is enabled."""
        return self.config.enabled

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    async def _load_state(self):
        """Load circuit breaker state from Redis."""
        if not self._redis:
            return

        try:
            state_data = await self._redis.get(self._state_key)
            if state_data:
                parts = state_data.split(":")
                if len(parts) >= 2:
                    state_str, _timestamp = parts[0], parts[1]
                    self._state = CircuitBreakerState(state_str)

                    # Load counts
                    failures = await self._redis.get(self._failure_count_key)
                    if failures:
                        self._failure_count = int(failures)

                    successes = await self._redis.get(self._success_count_key)
                    if successes:
                        self._success_count = int(successes)

                    last_failure = await self._redis.get(self._last_failure_time_key)
                    if last_failure:
                        self._last_failure_time = float(last_failure)

                    logger.info(
                        f"Loaded circuit breaker state for {self.service_id}: "
                        f"{self._state.value}"
                    )
        except Exception as e:
            logger.warning(f"Failed to load circuit breaker state: {e}")

    async def _save_state(self):
        """Save circuit breaker state to Redis."""
        if not self._redis:
            return

        try:
            # Save state with timestamp
            state_data = f"{self._state.value}:{time.time()}"
            await self._redis.set(
                self._state_key, state_data, ex=self.config.timeout_seconds
            )

            # Save counts
            await self._redis.set(
                self._failure_count_key, str(self._failure_count), ex=3600
            )
            await self._redis.set(
                self._success_count_key, str(self._success_count), ex=3600
            )
            await self._redis.set(
                self._last_failure_time_key, str(self._last_failure_time), ex=3600
            )
        except Exception as e:
            logger.warning(f"Failed to save circuit breaker state: {e}")

    async def allow_request(self) -> bool:
        """
        Check if request should be allowed through.

        Returns:
            True if request should be allowed, False otherwise
        """
        if not self.enabled:
            return True

        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitBreakerState.OPEN:
            time_since_failure = time.time() - self._last_failure_time
            if time_since_failure >= self.config.timeout_seconds:
                logger.info(
                    f"Circuit breaker transitioning to HALF_OPEN for {self.service_id}"
                )
                self._state = CircuitBreakerState.HALF_OPEN
                self._success_count = 0
                await self._save_state()
                return True

            return False

        return True

    async def on_success(self):
        """
        Record a successful request.

        Called after a request completes successfully.
        """
        if not self.enabled:
            return

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            logger.info(
                f"Circuit breaker success in HALF_OPEN: {self._success_count}/"
                f"{self.config.success_threshold}"
            )

            if self._success_count >= self.config.success_threshold:
                logger.info(f"Circuit breaker closing for {self.service_id}")
                self._state = CircuitBreakerState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                await self._save_state()
            else:
                await self._save_state()
        elif self._state == CircuitBreakerState.CLOSED:
            # Reset failure count on success in CLOSED state
            if self._failure_count > 0:
                self._failure_count = 0
                await self._save_state()

    async def on_failure(self):
        """
        Record a failed request.

        Called after a request fails.
        """
        if not self.enabled:
            return

        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Immediately reopen on failure in HALF_OPEN
            logger.warning(f"Circuit breaker reopening for {self.service_id}")
            self._state = CircuitBreakerState.OPEN
            self._success_count = 0
            await self._save_state()
        elif self._state == CircuitBreakerState.CLOSED:
            logger.warning(
                f"Circuit breaker failure: {self._failure_count}/"
                f"{self.config.failure_threshold}"
            )

            if self._failure_count >= self.config.failure_threshold:
                logger.warning(f"Circuit breaker opening for {self.service_id}")
                self._state = CircuitBreakerState.OPEN
                await self._save_state()
            else:
                await self._save_state()

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request.

        Args:
            request: The FastAPI Request object
            context: A dict for passing state to other middleware

        Returns:
            Tuple of (should_continue, optional_error)
        """
        if not self.enabled:
            return True, None

        # Add circuit breaker info to context
        context["circuit_breaker"] = {
            "service_id": self.service_id,
            "state": self._state.value,
        }

        # Check if request should be allowed
        allowed = await self.allow_request()
        if not allowed:
            retry_after = max(
                0,
                self.config.timeout_seconds
                - int(time.time() - self._last_failure_time),
            )
            error = CircuitBreakerOpenError(self.service_id, retry_after)
            return False, error

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process outgoing response.

        Records success/failure based on response status.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict
        """
        if not self.enabled:
            return response

        # Check if request was successful
        status = response.get("status", None)
        if status is None:
            # Try to get status from choices (OpenAI format)
            choices = response.get("choices", [])
            if choices and len(choices) > 0:
                finish_reason = choices[0].get("finish_reason", "")
                # Consider it successful if we got a response
                if finish_reason in ("stop", "length", "content_filter"):
                    await self.on_success()
            else:
                # No status info, assume success
                await self.on_success()
        elif status < 500:
            # Success response
            await self.on_success()
        else:
            # Server error
            await self.on_failure()

        # Add circuit breaker state to response headers
        if "headers" not in response:
            response["headers"] = {}
        response["headers"]["X-Circuit-Breaker-State"] = self._state.value

        return response
