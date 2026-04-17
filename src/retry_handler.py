"""
Retry Handler with Exponential Backoff

Production-grade retry logic for HTTP requests to AI models.
Handles transient failures, rate limits, and network errors.

Features:
- Exponential backoff with jitter
- Rate limit detection (HTTP 429) with Retry-After header support
- Configurable retry conditions
- Integration with Circuit Breaker for cascading failures
- Metrics tracking for retry attempts
"""

import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from enum import Enum

import httpx

try:
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep_log,
        RetryError,
    )

    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False
    # Fallback implementations will be used

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    """Retry strategy types."""

    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """
    Retry configuration.

    Attributes:
        max_attempts: Maximum number of retry attempts (default: 5)
        base_wait_seconds: Base wait time for exponential backoff (default: 1s)
        max_wait_seconds: Maximum wait time between retries (default: 60s)
        exponential_base: Base for exponential backoff (default: 2)
        jitter: Add random jitter to wait times (default: True)
        retry_on_429: Retry on rate limit errors (default: True)
        retry_on_5xx: Retry on server errors (default: True)
        retry_on_timeout: Retry on timeout errors (default: True)
        retry_on_connection_error: Retry on connection errors (default: True)
    """

    max_attempts: int = 5
    base_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_429: bool = True
    retry_on_5xx: bool = True
    retry_on_timeout: bool = True
    retry_on_connection_error: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_wait_seconds < 0:
            raise ValueError("base_wait_seconds must be >= 0")
        if self.max_wait_seconds < self.base_wait_seconds:
            raise ValueError("max_wait_seconds must be >= base_wait_seconds")
        if self.exponential_base < 1.0:
            raise ValueError("exponential_base must be >= 1.0")


class RetryableError(Exception):
    """Base exception for errors that should trigger retry."""

    pass


class RateLimitError(RetryableError):
    """Rate limit exceeded (HTTP 429)."""

    def __init__(self, retry_after: Optional[float] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after: {retry_after}")


class ServerError(RetryableError):
    """Server error (HTTP 5xx)."""

    def __init__(self, status_code: int):
        self.status_code = status_code
        super().__init__(f"Server error: {status_code}")


class ConnectionError(RetryableError):
    """Connection error or timeout."""

    pass


class RetryMetrics:
    """Track retry metrics."""

    def __init__(self):
        self.total_attempts = 0
        self.total_retries = 0
        self.total_success = 0
        self.total_failures = 0
        self.retry_by_reason: Dict[str, int] = {}

    def record_attempt(self, reason: Optional[str] = None):
        """Record a retry attempt."""
        self.total_attempts += 1
        if reason:
            self.retry_by_reason[reason] = self.retry_by_reason.get(reason, 0) + 1

    def record_success(self):
        """Record a successful request after retries."""
        self.total_success += 1

    def record_failure(self):
        """Record a failed request after all retries."""
        self.total_failures += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return {
            "total_attempts": self.total_attempts,
            "total_retries": self.total_retries,
            "total_success": self.total_success,
            "total_failures": self.total_failures,
            "retry_by_reason": self.retry_by_reason,
            "success_rate": (
                self.total_success / (self.total_success + self.total_failures)
                if (self.total_success + self.total_failures) > 0
                else 0.0
            ),
        }


class RetryHandler:
    """
    Handle retries with exponential backoff for HTTP requests.

    Supports both tenacity-based and fallback implementations.
    """

    def __init__(
        self, config: Optional[RetryConfig] = None, enable_metrics: bool = True
    ):
        """
        Initialize retry handler.

        Args:
            config: Retry configuration (uses defaults if None)
            enable_metrics: Track retry metrics
        """
        self.config = config or RetryConfig()
        self.enable_metrics = enable_metrics
        self.metrics = RetryMetrics() if enable_metrics else None

        if not TENACITY_AVAILABLE:
            logger.warning(
                "Tenacity library not available. Using fallback retry implementation. "
                "Install tenacity for better retry handling: pip install tenacity"
            )

    async def execute_with_retry(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            RetryError: If all retry attempts exhausted
            Original exception: If exception is not retryable
        """
        if TENACITY_AVAILABLE:
            return await self._execute_with_tenacity(func, *args, **kwargs)
        else:
            return await self._execute_with_fallback(func, *args, **kwargs)

    async def _execute_with_tenacity(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute with tenacity-based retry logic."""

        def should_retry_exception(exception: Exception) -> bool:
            """Determine if exception should trigger retry."""
            if self.metrics:
                self.metrics.record_attempt(reason=type(exception).__name__)

            # HTTP status errors
            if isinstance(exception, httpx.HTTPStatusError):
                status_code = exception.response.status_code

                # Rate limit (429)
                if status_code == 429 and self.config.retry_on_429:
                    return True

                # Server errors (5xx)
                if 500 <= status_code < 600 and self.config.retry_on_5xx:
                    return True

                return False

            # Network errors
            if isinstance(exception, (httpx.ConnectError, httpx.RemoteProtocolError)):
                return self.config.retry_on_connection_error

            # Timeout errors
            if isinstance(exception, (httpx.TimeoutException, asyncio.TimeoutError)):
                return self.config.retry_on_timeout

            return False

        def before_sleep(retry_state):
            """Log before sleep."""
            if retry_state.outcome and retry_state.outcome.failed:
                exception = retry_state.outcome.exception()
                logger.warning(
                    f"Retry attempt {retry_state.attempt_number}/{self.config.max_attempts} "
                    f"after {exception.__class__.__name__}: {str(exception)[:100]}"
                )

        @retry(
            stop=stop_after_attempt(self.config.max_attempts),
            wait=wait_exponential(
                multiplier=self.config.base_wait_seconds,
                max=self.config.max_wait_seconds,
                exp_base=self.config.exponential_base,
            ),
            retry=retry_if_exception_type(Exception)
            & retry_if_exception_type(RetryableError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        async def _retry_wrapper():
            try:
                result = await func(*args, **kwargs)

                if self.metrics:
                    self.metrics.record_success()

                return result

            except httpx.HTTPStatusError as e:
                # Check if we should retry based on status code
                status_code = e.response.status_code

                if status_code == 429 and self.config.retry_on_429:
                    # Check for Retry-After header
                    retry_after = e.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            retry_after_seconds = float(retry_after)
                            logger.info(
                                f"Rate limited. Retry-After: {retry_after_seconds}s"
                            )
                            await asyncio.sleep(retry_after_seconds)
                        except ValueError:
                            pass

                    if self.metrics:
                        self.metrics.total_retries += 1

                    raise RateLimitError(retry_after) from e

                if 500 <= status_code < 600 and self.config.retry_on_5xx:
                    if self.metrics:
                        self.metrics.total_retries += 1

                    raise ServerError(status_code) from e

                # Don't retry on other status codes
                raise

            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if self.config.retry_on_connection_error:
                    if self.metrics:
                        self.metrics.total_retries += 1

                    raise ConnectionError(str(e)) from e

                raise

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                if self.config.retry_on_timeout:
                    if self.metrics:
                        self.metrics.total_retries += 1

                    raise ConnectionError(f"Timeout: {str(e)}") from e

                raise

        # Execute with retry
        try:
            return await _retry_wrapper()

        except RetryError as e:
            if self.metrics:
                self.metrics.record_failure()

            logger.error(
                f"All {self.config.max_attempts} retry attempts exhausted: {e}"
            )
            raise

        except Exception:
            if self.metrics:
                self.metrics.record_failure()

            raise

    async def _execute_with_fallback(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> Any:
        """Execute with fallback retry implementation (no tenacity)."""
        last_exception = None

        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await func(*args, **kwargs)

                if self.metrics:
                    self.metrics.record_success()

                return result

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Rate limit (429)
                if status_code == 429 and self.config.retry_on_429:
                    retry_after = e.response.headers.get("Retry-After")

                    if retry_after:
                        try:
                            wait_time = float(retry_after)
                        except ValueError:
                            wait_time = self._calculate_backoff(attempt)
                    else:
                        wait_time = self._calculate_backoff(attempt)

                    logger.warning(
                        f"Rate limited (attempt {attempt}/{self.config.max_attempts}). "
                        f"Waiting {wait_time:.1f}s before retry"
                    )

                    if attempt < self.config.max_attempts:
                        await asyncio.sleep(wait_time)
                        if self.metrics:
                            self.metrics.total_retries += 1
                            self.metrics.record_attempt("rate_limit")
                        last_exception = e
                        continue

                # Server errors (5xx)
                elif 500 <= status_code < 600 and self.config.retry_on_5xx:
                    wait_time = self._calculate_backoff(attempt)

                    logger.warning(
                        f"Server error {status_code} (attempt {attempt}/{self.config.max_attempts}). "
                        f"Waiting {wait_time:.1f}s before retry"
                    )

                    if attempt < self.config.max_attempts:
                        await asyncio.sleep(wait_time)
                        if self.metrics:
                            self.metrics.total_retries += 1
                            self.metrics.record_attempt("server_error")
                        last_exception = e
                        continue

                # Don't retry on other status codes
                if self.metrics:
                    self.metrics.record_failure()
                raise

            except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
                if self.config.retry_on_connection_error:
                    wait_time = self._calculate_backoff(attempt)

                    logger.warning(
                        f"Connection error (attempt {attempt}/{self.config.max_attempts}). "
                        f"Waiting {wait_time:.1f}s before retry: {str(e)[:100]}"
                    )

                    if attempt < self.config.max_attempts:
                        await asyncio.sleep(wait_time)
                        if self.metrics:
                            self.metrics.total_retries += 1
                            self.metrics.record_attempt("connection_error")
                        last_exception = e
                        continue

                if self.metrics:
                    self.metrics.record_failure()
                raise

            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                if self.config.retry_on_timeout:
                    wait_time = self._calculate_backoff(attempt)

                    logger.warning(
                        f"Timeout (attempt {attempt}/{self.config.max_attempts}). "
                        f"Waiting {wait_time:.1f}s before retry"
                    )

                    if attempt < self.config.max_attempts:
                        await asyncio.sleep(wait_time)
                        if self.metrics:
                            self.metrics.total_retries += 1
                            self.metrics.record_attempt("timeout")
                        last_exception = e
                        continue

                if self.metrics:
                    self.metrics.record_failure()
                raise

            except Exception:
                # Don't retry on other exceptions
                if self.metrics:
                    self.metrics.record_failure()
                raise

        # All retries exhausted
        if self.metrics:
            self.metrics.record_failure()

        if last_exception:
            raise RetryError(
                f"All {self.config.max_attempts} retry attempts exhausted"
            ) from last_exception

        raise RetryError(f"All {self.config.max_attempts} retry attempts exhausted")

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff with optional jitter.

        Args:
            attempt: Current attempt number (1-indexed)

        Returns:
            Wait time in seconds
        """
        # Exponential backoff: base_wait * exponential_base ^ (attempt - 1)
        wait_time = self.config.base_wait_seconds * (
            self.config.exponential_base ** (attempt - 1)
        )

        # Cap at max_wait_seconds
        wait_time = min(wait_time, self.config.max_wait_seconds)

        # Add jitter if enabled
        if self.config.jitter:
            import random

            jitter = wait_time * 0.1  # 10% jitter
            wait_time += random.uniform(-jitter, jitter)
            wait_time = max(0, wait_time)  # Ensure non-negative

        return wait_time

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get retry metrics.

        Returns:
            Metrics dict, or None if metrics disabled
        """
        if not self.enable_metrics or not self.metrics:
            return None

        return self.metrics.get_summary()

    def reset_metrics(self):
        """Reset retry metrics."""
        if self.metrics:
            self.metrics = RetryMetrics()


# Singleton instance
_default_handler: Optional[RetryHandler] = None


def get_default_handler() -> RetryHandler:
    """Get or create default retry handler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = RetryHandler()
    return _default_handler


async def execute_with_retry(
    func: Callable, *args: Any, config: Optional[RetryConfig] = None, **kwargs: Any
) -> Any:
    """
    Execute function with retry logic using default handler.

    Convenience function that uses default retry handler.

    Args:
        func: Async function to execute
        *args: Function arguments
        config: Optional retry configuration (uses default handler's config if None)
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    handler = get_default_handler()

    if config:
        # Create temporary handler with custom config
        handler = RetryHandler(config=config)

    return await handler.execute_with_retry(func, *args, **kwargs)
