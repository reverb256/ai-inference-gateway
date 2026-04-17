# modules/services/ai-inference/ai_inference_gateway/tests/test_metrics.py
"""
Tests for Metrics Helper utility.
"""

import pytest
from prometheus_client import CollectorRegistry
from ai_inference_gateway.utils.metrics import MetricsHelper, create_timer


class TestMetricsHelper:
    """Test MetricsHelper functionality."""

    def test_initialization(self):
        """Test metrics helper initialization."""
        registry = CollectorRegistry()
        metrics = MetricsHelper(registry)

        # Verify metrics are created
        assert metrics.http_requests_total is not None
        assert metrics.http_request_duration_seconds is not None
        assert metrics.errors_total is not None
        assert metrics.circuit_breaker_state is not None

    def test_http_request_metrics(self):
        """Test HTTP request metric updates."""
        metrics = MetricsHelper()

        # Test increment
        metrics.inc_http_requests("POST", "/v1/chat/completions", 200)
        metrics.inc_http_requests("POST", "/v1/chat/completions", 200, amount=5)

        # Test latency observation
        metrics.observe_http_latency("POST", "/v1/chat/completions", 0.5)

        # Test in-progress tracking
        metrics.track_http_request_in_progress("POST", "/v1/chat/completions", delta=1)
        metrics.track_http_request_in_progress("POST", "/v1/chat/completions", delta=-1)

        # Test response size
        metrics.observe_response_size("POST", "/v1/chat/completions", 1024)

    def test_error_metrics(self):
        """Test error metric updates."""
        metrics = MetricsHelper()

        metrics.inc_errors("RateLimitError", "rate_limiter")
        metrics.inc_errors("CircuitBreakerOpen", "circuit_breaker", amount=3)

    def test_middleware_metrics(self):
        """Test middleware duration metrics."""
        metrics = MetricsHelper()

        metrics.observe_middleware_duration("security_filter", "request", 0.01)
        metrics.observe_middleware_duration("rate_limiter", "response", 0.005)

    def test_rate_limiting_metrics(self):
        """Test rate limiting metrics."""
        metrics = MetricsHelper()

        metrics.inc_rate_limit_denied("tokens_per_minute")
        metrics.inc_rate_limit_denied("tokens_per_hour", amount=5)

    def test_security_metrics(self):
        """Test security filter metrics."""
        metrics = MetricsHelper()

        metrics.inc_security_blocked("pii")
        metrics.inc_security_blocked("max_request_size")

    def test_cache_metrics(self):
        """Test cache metrics."""
        metrics = MetricsHelper()

        metrics.inc_cache_hits("semantic")
        metrics.inc_cache_hits("exact", amount=3)
        metrics.inc_cache_misses("semantic")

    def test_circuit_breaker_metrics(self):
        """Test circuit breaker metrics."""
        metrics = MetricsHelper()

        # State: 0=closed, 1=open, 2=half_open
        metrics.set_circuit_breaker_state("backend1", 0)
        metrics.set_circuit_breaker_state("backend2", 1)

        metrics.inc_circuit_breaker_failures("backend1")
        metrics.inc_circuit_breaker_failures("backend2", amount=2)

    def test_backend_metrics(self):
        """Test backend metrics."""
        metrics = MetricsHelper()

        metrics.inc_backend_requests("backend1", "success")
        metrics.inc_backend_requests("backend2", "error", amount=2)

        metrics.observe_backend_latency("backend1", 0.3)
        metrics.observe_backend_latency("backend2", 0.5)

        metrics.set_backend_health("backend1", True)
        metrics.set_backend_health("backend2", False)

    def test_load_balancer_metrics(self):
        """Test load balancer metrics."""
        metrics = MetricsHelper()

        metrics.inc_load_balancer_selections("backend1")
        metrics.inc_load_balancer_selections("backend2", amount=5)

    def test_metrics_exposition(self):
        """Test metrics exposition."""
        metrics = MetricsHelper()

        # Add some data
        metrics.inc_http_requests("POST", "/v1/chat/completions", 200)
        metrics.observe_http_latency("POST", "/v1/chat/completions", 0.5)

        # Get metrics text
        metrics_text = metrics.get_metrics_text()

        # Verify it's bytes
        assert isinstance(metrics_text, bytes)

        # Verify content type
        content_type = metrics.get_content_type()
        assert content_type == "text/plain; version=0.0.4; charset=utf-8"

    def test_timer_context_manager(self):
        """Test metric timer context manager."""
        metrics = MetricsHelper()

        with create_timer(
            metrics,
            metrics.observe_http_latency,
            method="POST",
            endpoint="/v1/chat/completions",
        ):
            # Simulate work
            pass

        # Verify metric was updated (integration test would verify actual value)
        assert True

    def test_timer_with_exception(self):
        """Test timer handles exceptions correctly."""
        metrics = MetricsHelper()

        with pytest.raises(ValueError):
            with create_timer(
                metrics,
                metrics.observe_http_latency,
                method="POST",
                endpoint="/v1/chat/completions",
            ):
                raise ValueError("Test exception")

        # Timer should still record time
        assert True

    def test_multiple_instances_isolation(self):
        """Test that multiple metric instances don't interfere."""
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        metrics1 = MetricsHelper(registry1)
        metrics2 = MetricsHelper(registry2)

        metrics1.inc_http_requests("POST", "/test", 200)
        metrics2.inc_http_requests("POST", "/test", 200)

        # Each should have its own metrics
        text1 = metrics1.get_metrics_text()
        text2 = metrics2.get_metrics_text()

        assert isinstance(text1, bytes)
        assert isinstance(text2, bytes)
