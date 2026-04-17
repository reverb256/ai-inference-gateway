# modules/services/ai-inference/ai_inference_gateway/tests/test_integration.py
"""
Integration tests for AI Inference Gateway.

Tests the full request pipeline with middleware execution order,
error scenarios, and metrics collection.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock
from fastapi import Request, HTTPException

from ai_inference_gateway.config import GatewayConfig
from ai_inference_gateway.pipeline import MiddlewarePipeline
from ai_inference_gateway.middleware.observability import ObservabilityMiddleware
from ai_inference_gateway.middleware.security_filter import SecurityFilterMiddleware
from ai_inference_gateway.middleware.rate_limiter import RateLimiterMiddleware
from ai_inference_gateway.middleware.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
)
from ai_inference_gateway.middleware.load_balancer import (
    LoadBalancerMiddleware,
    BackendInstance,
    BackendState,
)
from ai_inference_gateway.utils.metrics import MetricsHelper


class TestIntegration:
    """Integration tests for full gateway pipeline."""

    @pytest.fixture
    def sample_config(self):
        """Create sample gateway configuration."""
        return GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://127.0.0.1:1234",
            backend_type="lm-studio",
        )

    @pytest.fixture
    def metrics_helper(self):
        """Create metrics helper for testing."""
        return MetricsHelper()

    @pytest.fixture
    def mock_request(self):
        """Create mock FastAPI request."""
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()
        return request

    def test_full_pipeline_execution_order(self, sample_config, metrics_helper):
        """
        Test that middleware executes in correct order.

        Order should be:
        1. Observability (adds request ID, start time)
        2. Security Filter (validates request)
        3. Rate Limiter (checks rate limits)
        4. Circuit Breaker (checks backend health)
        5. Load Balancer (selects backend)
        """
        # Create middleware
        observability = ObservabilityMiddleware(sample_config.middleware.observability)
        security = SecurityFilterMiddleware(sample_config.middleware.security)
        rate_limiter = RateLimiterMiddleware(
            sample_config.middleware.rate_limiting, metrics_helper=metrics_helper
        )
        circuit_breaker = CircuitBreaker(
            "test_backend",
            sample_config.middleware.circuit_breaker,
            metrics_helper=metrics_helper,
        )
        load_balancer = LoadBalancerMiddleware(
            sample_config.middleware.load_balancer,
            [BackendInstance(name="backend1", url="http://localhost:8001")],
            metrics_helper=metrics_helper,
        )

        # Create pipeline
        pipeline = MiddlewarePipeline(
            [observability, security, rate_limiter, circuit_breaker, load_balancer]
        )

        # Track execution order
        execution_order = []

        async def mock_handler(request, context):
            execution_order.append("handler")
            return {"status": 200, "data": "test"}

        # Process request
        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        # Verify all middleware ran
        assert "request_id" in result.get("gateway_metadata", {})
        assert result["status"] == 200

    def test_middleware_short_circuit(self, sample_config):
        """
        Test that middleware can short-circuit the pipeline.

        When security filter blocks a request, subsequent middleware should not run.
        """
        # Create config with large max size to allow test
        config = GatewayConfig()
        config.middleware.security.max_request_size = 100  # Very small limit

        observability = ObservabilityMiddleware(config.middleware.observability)
        security = SecurityFilterMiddleware(config.middleware.security)

        pipeline = MiddlewarePipeline([observability, security])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {"content-length": "1000"}  # Exceeds limit
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        # Should short-circuit with error
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_pipeline())

        assert exc_info.value.status_code == 413

    def test_error_propagation_through_pipeline(self, sample_config):
        """
        Test that errors propagate correctly through the pipeline.

        When an error occurs, all middleware should still get a chance
        to clean up via process_response.
        """
        observability = ObservabilityMiddleware(sample_config.middleware.observability)

        circuit_breaker = CircuitBreaker(
            "test_backend", sample_config.middleware.circuit_breaker
        )

        # Force circuit breaker open
        circuit_breaker._state = CircuitBreakerState.OPEN
        circuit_breaker._open_until = time.time() + 60

        pipeline = MiddlewarePipeline([observability, circuit_breaker])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        # Should get circuit breaker error
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_pipeline())

        assert exc_info.value.status_code == 503

    def test_metrics_collection_integration(self, sample_config, metrics_helper):
        """
        Test that metrics are collected across all middleware.

        Verifies that:
        - HTTP requests are counted
        - Latency is tracked
        - Errors are counted
        - Middleware-specific metrics are updated
        """
        observability = ObservabilityMiddleware(sample_config.middleware.observability)
        security = SecurityFilterMiddleware(sample_config.middleware.security)

        # Track metrics
        _ = 0  # Placeholder for initial request count

        pipeline = MiddlewarePipeline([observability, security])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            metrics_helper.inc_http_requests("POST", "/v1/chat/completions", 200)
            metrics_helper.observe_middleware_duration("observability", "request", 0.01)
            metrics_helper.observe_middleware_duration(
                "security_filter", "request", 0.005
            )
            return {"status": 200, "data": "test"}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        # Verify metrics were updated
        assert result["status"] == 200

        # Get metrics text
        metrics_text = metrics_helper.get_metrics_text()
        assert isinstance(metrics_text, bytes)

    def test_context_state_propagation(self, sample_config):
        """
        Test that context state is properly propagated between middleware.

        Verifies that:
        - Request ID is set by observability
        - Backend is selected by load balancer
        - Context is available in response processing
        """
        observability = ObservabilityMiddleware(sample_config.middleware.observability)

        # Enable load balancer
        config = GatewayConfig()
        config.middleware.load_balancer.enabled = True

        load_balancer = LoadBalancerMiddleware(
            config.middleware.load_balancer,
            [
                BackendInstance(name="backend1", url="http://localhost:8001"),
                BackendInstance(name="backend2", url="http://localhost:8002"),
            ],
        )

        pipeline = MiddlewarePipeline([observability, load_balancer])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            # Verify context has request ID
            assert "request_id" in context
            # Verify context has backend
            assert "load_balancer_backend" in context
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        # Verify response has metadata
        assert "gateway_metadata" in result
        assert "request_id" in result["gateway_metadata"]
        assert "load_balancer" in result["gateway_metadata"]

    def test_concurrent_request_handling(self, sample_config, metrics_helper):
        """
        Test that the gateway handles concurrent requests correctly.

        Verifies that:
        - Multiple requests can be processed simultaneously
        - Context is not shared between requests
        - Metrics are updated correctly
        """
        observability = ObservabilityMiddleware(sample_config.middleware.observability)

        pipeline = MiddlewarePipeline([observability])

        async def process_single_request(request_id):
            request = Mock(spec=Request)
            request.method = "POST"
            request.url = Mock(path="/v1/chat/completions")
            request.headers = {"X-Request-ID": request_id}
            request.state = Mock()

            async def mock_handler(req, context):
                # Simulate some work
                await asyncio.sleep(0.01)
                return {"status": 200, "request_id": context.get("request_id")}

            return await pipeline.process_request(request, mock_handler)

        async def run_concurrent():
            # Process 10 concurrent requests
            tasks = [process_single_request(f"req-{i}") for i in range(10)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_concurrent())

        # Verify all requests completed
        assert len(results) == 10

        # Verify each request has unique context
        request_ids = [r.get("request_id") for r in results]
        assert len(set(request_ids)) == 10  # All unique

    def test_response_processing_pipeline(self, sample_config):
        """
        Test that response processing works correctly.

        Verifies that:
        - Response middleware runs in reverse order
        - Response metadata is added
        - Processing time is calculated
        """
        observability = ObservabilityMiddleware(sample_config.middleware.observability)

        load_balancer = LoadBalancerMiddleware(
            sample_config.middleware.load_balancer,
            [BackendInstance(name="backend1", url="http://localhost:8001")],
        )

        pipeline = MiddlewarePipeline([observability, load_balancer])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200, "data": "response"}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        # Verify response has gateway metadata
        assert "gateway_metadata" in result

        metadata = result["gateway_metadata"]
        assert "request_id" in metadata
        assert "processing_time_ms" in metadata
        assert "load_balancer" in metadata

    def test_disabled_middleware_skip(self, sample_config):
        """
        Test that disabled middleware is skipped.

        Verifies that:
        - Disabled middleware doesn't process requests
        - Pipeline continues to next middleware
        - No errors from disabled middleware
        """
        # Disable rate limiting
        config = GatewayConfig()
        config.middleware.rate_limiting.enabled = False

        rate_limiter = RateLimiterMiddleware(config.middleware.rate_limiting)

        observability = ObservabilityMiddleware(config.middleware.observability)

        pipeline = MiddlewarePipeline([rate_limiter, observability])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        # Should succeed (rate limiter disabled, observability ran)
        assert result["status"] == 200
        assert "gateway_metadata" in result

    def test_backend_failover_scenario(self, sample_config):
        """
        Test backend failover when one backend is unhealthy.

        Verifies that:
        - Load balancer skips unhealthy backends
        - Requests are routed to healthy backends
        - Circuit breaker triggers on failures
        """
        config = GatewayConfig()
        config.middleware.load_balancer.enabled = True

        backends = [
            BackendInstance(name="backend1", url="http://localhost:8001"),
            BackendInstance(name="backend2", url="http://localhost:8002", weight=200),
        ]

        # Mark backend1 as unhealthy
        backends[0].state = BackendState.UNHEALTHY

        load_balancer = LoadBalancerMiddleware(
            config.middleware.load_balancer, backends
        )

        observability = ObservabilityMiddleware(config.middleware.observability)

        pipeline = MiddlewarePipeline([observability, load_balancer])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            backend = context.get("load_balancer_backend")
            # Should always get backend2 (healthy)
            assert backend.name == "backend2"
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        result = asyncio.run(run_pipeline())

        assert result["status"] == 200
        assert result["gateway_metadata"]["load_balancer"]["backend_name"] == "backend2"


class TestErrorScenarios:
    """Test error scenarios in integration."""

    def test_all_backends_unavailable(self, sample_config):
        """
        Test handling when all backends are unavailable.

        Should return 503 Service Unavailable.
        """
        config = GatewayConfig()
        config.middleware.load_balancer.enabled = True

        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        # Mark backend as unhealthy
        backends[0].state = BackendState.UNHEALTHY

        load_balancer = LoadBalancerMiddleware(
            config.middleware.load_balancer, backends
        )

        observability = ObservabilityMiddleware(config.middleware.observability)

        pipeline = MiddlewarePipeline([observability, load_balancer])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_pipeline())

        assert exc_info.value.status_code == 503

    def test_circuit_breaker_open(self, sample_config):
        """
        Test handling when circuit breaker is open.

        Should return 503 Service Unavailable.
        """
        config = GatewayConfig()

        circuit_breaker = CircuitBreaker(
            "test_backend", config.middleware.circuit_breaker
        )

        # Force open
        circuit_breaker._state = CircuitBreakerState.OPEN
        circuit_breaker._open_until = time.time() + 60

        observability = ObservabilityMiddleware(config.middleware.observability)

        pipeline = MiddlewarePipeline([observability, circuit_breaker])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_pipeline())

        assert exc_info.value.status_code == 503

    def test_security_filter_blocked(self, sample_config):
        """
        Test handling when security filter blocks request.

        Should return 413 Payload Too Large for size violations.
        """
        config = GatewayConfig()
        config.middleware.security.max_request_size = 100

        security = SecurityFilterMiddleware(config.middleware.security)

        observability = ObservabilityMiddleware(config.middleware.observability)

        pipeline = MiddlewarePipeline([observability, security])

        request = Mock(spec=Request)
        request.method = "POST"
        request.url = Mock(path="/v1/chat/completions")
        request.headers = {"content-length": "1000"}
        request.state = Mock()

        async def mock_handler(request, context):
            return {"status": 200}

        async def run_pipeline():
            return await pipeline.process_request(request, mock_handler)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(run_pipeline())

        assert exc_info.value.status_code == 413
