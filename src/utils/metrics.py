# modules/services/ai-inference/ai_inference_gateway/utils/metrics.py
"""
Prometheus metrics helper for AI Inference Gateway.

Provides centralized metric definitions and update helpers for common metrics.
Integration with prometheus_client for metrics exposition.
"""

from typing import Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import logging
import time

logger = logging.getLogger(__name__)


class MetricsHelper:
    """
    Centralized Prometheus metrics management.

    Features:
    - Predefined common metrics for HTTP, latency, and errors
    - Thread-safe metric updates
    - Registry management
    - Metrics exposition endpoint support
    """

    # Common metric definitions
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics helper.

        Args:
            registry: Optional custom registry (defaults to global registry)
        """
        self.registry = registry or CollectorRegistry()

        # HTTP request metrics
        self.http_requests_total = Counter(
            "gateway_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            "gateway_http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            buckets=(
                0.005,
                0.01,
                0.025,
                0.05,
                0.075,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                2.5,
                5.0,
                7.5,
                10.0,
            ),
            registry=self.registry,
        )

        self.http_requests_in_progress = Gauge(
            "gateway_http_requests_in_progress",
            "HTTP requests currently in progress",
            ["method", "endpoint"],
            registry=self.registry,
        )

        self.http_response_size_bytes = Histogram(
            "gateway_http_response_size_bytes",
            "HTTP response size",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "gateway_errors_total",
            "Total errors",
            ["error_type", "middleware"],
            registry=self.registry,
        )

        # Middleware-specific metrics
        self.middleware_duration_seconds = Histogram(
            "gateway_middleware_duration_seconds",
            "Middleware processing duration",
            ["middleware_name", "stage"],  # stage: request or response
            registry=self.registry,
        )

        self.rate_limit_denied_total = Counter(
            "gateway_rate_limit_denied_total",
            "Total rate limit denials",
            ["limit_type"],  # tokens_per_minute, tokens_per_hour, etc
            registry=self.registry,
        )

        self.security_blocked_total = Counter(
            "gateway_security_blocked_total",
            "Total security filter blocks",
            ["block_reason"],  # pii, max_size, etc
            registry=self.registry,
        )

        # Cache metrics
        self.cache_hits_total = Counter(
            "gateway_cache_hits_total",
            "Total cache hits",
            ["cache_type"],  # semantic, exact
            registry=self.registry,
        )

        self.cache_misses_total = Counter(
            "gateway_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        # Circuit breaker metrics
        self.circuit_breaker_state = Gauge(
            "gateway_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half_open)",
            ["backend_name"],
            registry=self.registry,
        )

        self.circuit_breaker_failures_total = Counter(
            "gateway_circuit_breaker_failures_total",
            "Total circuit breaker failures",
            ["backend_name"],
            registry=self.registry,
        )

        # Backend metrics
        self.backend_requests_total = Counter(
            "gateway_backend_requests_total",
            "Total backend requests",
            ["backend_name", "status"],
            registry=self.registry,
        )

        self.backend_latency_seconds = Histogram(
            "gateway_backend_latency_seconds",
            "Backend request latency",
            ["backend_name"],
            registry=self.registry,
        )

        self.backend_health = Gauge(
            "gateway_backend_health",
            "Backend health status (1=healthy, 0=unhealthy)",
            ["backend_name"],
            registry=self.registry,
        )

        # Load balancer metrics
        self.load_balancer_selections_total = Counter(
            "gateway_load_balancer_selections_total",
            "Total backend selections",
            ["backend_name"],
            registry=self.registry,
        )

        logger.info("Metrics helper initialized")

    # HTTP Request Metrics
    def inc_http_requests(
        self, method: str, endpoint: str, status: int, amount: int = 1
    ) -> None:
        """Increment HTTP request counter."""
        self.http_requests_total.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc(amount)

    def observe_http_latency(self, method: str, endpoint: str, duration: float) -> None:
        """Observe HTTP request latency."""
        self.http_request_duration_seconds.labels(
            method=method, endpoint=endpoint
        ).observe(duration)

    def track_http_request_in_progress(
        self, method: str, endpoint: str, delta: int = 1
    ) -> None:
        """Track in-progress HTTP requests."""
        self.http_requests_in_progress.labels(method=method, endpoint=endpoint).inc(
            delta
        )

    def observe_response_size(
        self, method: str, endpoint: str, size_bytes: int
    ) -> None:
        """Observe HTTP response size."""
        self.http_response_size_bytes.labels(method=method, endpoint=endpoint).observe(
            size_bytes
        )

    # Error Metrics
    def inc_errors(self, error_type: str, middleware: str, amount: int = 1) -> None:
        """Increment error counter."""
        self.errors_total.labels(error_type=error_type, middleware=middleware).inc(
            amount
        )

    # Middleware Metrics
    def observe_middleware_duration(
        self, middleware_name: str, stage: str, duration: float
    ) -> None:
        """Observe middleware processing duration."""
        self.middleware_duration_seconds.labels(
            middleware_name=middleware_name, stage=stage
        ).observe(duration)

    # Rate Limiting Metrics
    def inc_rate_limit_denied(self, limit_type: str, amount: int = 1) -> None:
        """Increment rate limit denial counter."""
        self.rate_limit_denied_total.labels(limit_type=limit_type).inc(amount)

    # Security Metrics
    def inc_security_blocked(self, block_reason: str, amount: int = 1) -> None:
        """Increment security block counter."""
        self.security_blocked_total.labels(block_reason=block_reason).inc(amount)

    # Cache Metrics
    def inc_cache_hits(self, cache_type: str, amount: int = 1) -> None:
        """Increment cache hits counter."""
        self.cache_hits_total.labels(cache_type=cache_type).inc(amount)

    def inc_cache_misses(self, cache_type: str, amount: int = 1) -> None:
        """Increment cache misses counter."""
        self.cache_misses_total.labels(cache_type=cache_type).inc(amount)

    # Circuit Breaker Metrics
    def set_circuit_breaker_state(
        self, backend_name: str, state: int  # 0=closed, 1=open, 2=half_open
    ) -> None:
        """Set circuit breaker state."""
        self.circuit_breaker_state.labels(backend_name=backend_name).set(state)

    def inc_circuit_breaker_failures(self, backend_name: str, amount: int = 1) -> None:
        """Increment circuit breaker failure counter."""
        self.circuit_breaker_failures_total.labels(backend_name=backend_name).inc(
            amount
        )

    # Backend Metrics
    def inc_backend_requests(
        self, backend_name: str, status: str, amount: int = 1
    ) -> None:
        """Increment backend request counter."""
        self.backend_requests_total.labels(
            backend_name=backend_name, status=status
        ).inc(amount)

    def observe_backend_latency(self, backend_name: str, duration: float) -> None:
        """Observe backend request latency."""
        self.backend_latency_seconds.labels(backend_name=backend_name).observe(duration)

    def set_backend_health(self, backend_name: str, healthy: bool) -> None:
        """Set backend health status."""
        self.backend_health.labels(backend_name=backend_name).set(1 if healthy else 0)

    # Load Balancer Metrics
    def inc_load_balancer_selections(self, backend_name: str, amount: int = 1) -> None:
        """Increment load balancer selection counter."""
        self.load_balancer_selections_total.labels(backend_name=backend_name).inc(
            amount
        )

    # Utility Methods
    def get_metrics_text(self) -> bytes:
        """
        Get metrics in Prometheus text format.

        Returns:
            Metrics in Prometheus exposition format
        """
        return generate_latest(self.registry)

    def get_content_type(self) -> str:
        """
        Get content type for metrics endpoint.

        Returns:
            Content type string for metrics
        """
        return CONTENT_TYPE_LATEST

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        # Note: This is a simplified reset. In production, you might want
        # to create a new registry instead to avoid metric label collisions.
        logger.warning("Resetting all metrics")
        self.registry = CollectorRegistry()
        logger.info("Metrics reset complete")


# Context manager for timing
class MetricTimer:
    """Context manager for timing operations and updating histograms."""

    def __init__(self, metrics_helper: MetricsHelper, callback, **labels):
        """
        Initialize timer.

        Args:
            metrics_helper: MetricsHelper instance
            callback: Function to call with duration (e.g., observe_http_latency)
            **labels: Labels to pass to callback
        """
        self.metrics_helper = metrics_helper
        self.callback = callback
        self.labels = labels
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.callback(duration=duration, **self.labels)
        return False


def create_timer(metrics_helper: MetricsHelper, callback, **labels) -> MetricTimer:
    """
    Create a metric timer context manager.

    Args:
        metrics_helper: MetricsHelper instance
        callback: Function to call with duration
        **labels: Labels for the metric

    Returns:
        MetricTimer context manager

    Example:
        with create_timer(
            metrics,
            metrics.observe_http_latency,
            method="POST",
            endpoint="/v1/chat/completions"
        ):
            # Do work
            pass
    """
    return MetricTimer(metrics_helper, callback, **labels)
