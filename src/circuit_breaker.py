"""
Circuit Breaker Pattern for Backend Reliability

Prevents cascading failures by automatically failing fast
when a backend is experiencing issues.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, Any
import httpx

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if backend recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close again
    timeout_seconds: int = 60  # How long to stay open
    half_open_max_calls: int = 3  # Test calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for backend reliability.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Backend failing, requests rejected immediately
    - HALF_OPEN: Testing if backend recovered
    """

    def __init__(
        self,
        backend_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.backend_name = backend_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self.state_changed_at = datetime.now()
        self._half_open_calls = 0

    async def call(self, func, *args, **kwargs):
        """
        Execute function through circuit breaker.

        Args:
            func: Async function to call
            *args: Arguments for function
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: If function fails
        """
        self.stats.total_calls += 1

        # Check if we should reject request
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                self.stats.rejected_calls += 1
                raise CircuitBreakerOpenError(
                    f"Circuit breaker OPEN for {self.backend_name}. "
                    f"Rejecting request. Try again later."
                )

        try:
            # Execute the function
            result = await func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        elapsed = datetime.now() - self.state_changed_at
        return elapsed >= timedelta(seconds=self.config.timeout_seconds)

    def _transition_to_half_open(self):
        """Transition from OPEN to HALF_OPEN state."""
        logger.info(
            f"Circuit breaker for {self.backend_name}: "
            f"OPEN → HALF_OPEN (testing recovery)"
        )
        self.state = CircuitState.HALF_OPEN
        self.state_changed_at = datetime.now()
        self._half_open_calls = 0

    def _transition_to_open(self):
        """Transition to OPEN state."""
        logger.warning(
            f"Circuit breaker for {self.backend_name}: "
            f"OPEN (failure threshold reached: {self.stats.consecutive_failures})"
        )
        self.state = CircuitState.OPEN
        self.state_changed_at = datetime.now()
        self.stats.last_failure_time = datetime.now()

    def _transition_to_closed(self):
        """Transition to CLOSED state."""
        logger.info(
            f"Circuit breaker for {self.backend_name}: "
            f"HALF_OPEN → CLOSED (backend recovered)"
        )
        self.state = CircuitState.CLOSED
        self.state_changed_at = datetime.now()
        self.stats.consecutive_failures = 0
        self._half_open_calls = 0

    def _on_success(self):
        """Handle successful call."""
        self.stats.successful_calls += 1
        self.stats.consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.last_success_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls >= self.config.half_open_max_calls:
                self._transition_to_closed()

    def _on_failure(self):
        """Handle failed call."""
        self.stats.failed_calls += 1
        self.stats.consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.last_failure_time = datetime.now()

        if self.state == CircuitState.HALF_OPEN:
            # Failed during testing, go back to OPEN
            self._transition_to_open()
        elif self.stats.consecutive_failures >= self.config.failure_threshold:
            self._transition_to_open()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "backend": self.backend_name,
            "state": self.state.value,
            "total_calls": self.stats.total_calls,
            "successful_calls": self.stats.successful_calls,
            "failed_calls": self.stats.failed_calls,
            "rejected_calls": self.stats.rejected_calls,
            "success_rate": (
                self.stats.successful_calls / self.stats.total_calls
                if self.stats.total_calls > 0
                else 0
            ),
            "consecutive_failures": self.stats.consecutive_failures,
            "last_failure": (
                self.stats.last_failure_time.isoformat()
                if self.stats.last_failure_time
                else None
            ),
            "state_changed_at": self.state_changed_at.isoformat(),
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different backends.
    """

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}

    def get_breaker(
        self,
        backend_name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create circuit breaker for backend."""
        if backend_name not in self.breakers:
            self.breakers[backend_name] = CircuitBreaker(backend_name, config)
        return self.breakers[backend_name]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self.breakers.items()
        }

    async def call_with_retry(
        self,
        backend_name: str,
        func,
        *args,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        fallback_backends: Optional[list] = None,
        **kwargs,
    ):
        """
        Call function with circuit breaker and retry logic.

        Args:
            backend_name: Name of primary backend
            func: Async function to call
            *args: Arguments for function
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
            fallback_backends: List of fallback backend names
            **kwargs: Keyword arguments for function

        Returns:
            Function result

        Raises:
            Exception: If all attempts fail
        """
        last_error = None
        backends_to_try = [backend_name]

        # Add fallback backends
        if fallback_backends:
            backends_to_try.extend(fallback_backends)

        for attempt, backend in enumerate(backends_to_try):
            breaker = self.get_breaker(backend)

            for retry in range(max_retries + 1):
                try:
                    if attempt > 0:
                        logger.info(f"Trying fallback backend: {backend}")
                        # Update kwargs to use fallback backend
                        kwargs = self._update_backend_kwargs(kwargs, backend)

                    result = await breaker.call(func, *args, **kwargs)
                    return result

                except CircuitBreakerOpenError:
                    logger.warning(f"Circuit breaker open for {backend}")
                    break  # Don't retry, move to next backend

                except Exception as e:
                    last_error = e
                    if retry < max_retries:
                        wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                        logger.warning(
                            f"Request to {backend} failed: {e}. "
                            f"Retrying in {wait_time}s... (attempt {retry + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All retries exhausted for {backend}")
                        break  # Move to next backend

        # All backends failed
        raise Exception(
            f"All backends failed for request. "
            f"Tried: {backends_to_try}. Last error: {last_error}"
        )

    def _update_backend_kwargs(self, kwargs: Dict, backend_name: str) -> Dict:
        """Update kwargs to use different backend."""
        # This is a placeholder - actual implementation depends on your backend structure
        # You might need to update URLs, client configs, etc.
        return kwargs


# Global circuit breaker manager instance
_circuit_breaker_manager = CircuitBreakerManager()


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager instance."""
    return _circuit_breaker_manager
