# modules/services/ai-inference/ai_inference_gateway/tests/test_load_balancer.py
"""
Tests for Load Balancer Middleware.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.load_balancer import (
    LoadBalancerMiddleware,
    BackendInstance,
    BackendState,
)
from ai_inference_gateway.config import LoadBalancerConfig


class TestBackendInstance:
    """Test BackendInstance dataclass."""

    def test_initialization(self):
        """Test backend instance initialization."""
        backend = BackendInstance(
            name="backend1",
            url="http://localhost:8001",
            weight=150,
            health_check_url="/health",
        )

        assert backend.name == "backend1"
        assert backend.url == "http://localhost:8001"
        assert backend.weight == 150
        assert backend.health_check_url == "/health"
        assert backend.state == BackendState.HEALTHY
        assert backend.current_connections == 0
        assert backend.is_available

    def test_is_available(self):
        """Test is_available property."""
        backend = BackendInstance(name="backend1", url="http://localhost:8001")

        assert backend.is_available

        # Mark as unhealthy
        backend.state = BackendState.UNHEALTHY
        assert not backend.is_available

        # Mark as healthy but at connection limit
        backend.state = BackendState.HEALTHY
        backend.current_connections = backend.max_concurrent_requests
        assert not backend.is_available

    def test_success_rate(self):
        """Test success rate calculation."""
        backend = BackendInstance(name="backend1", url="http://localhost:8001")

        # No requests yet
        assert backend.success_rate == 1.0

        # Some requests, no failures
        backend.total_requests = 10
        assert backend.success_rate == 1.0

        # Some failures
        backend.failed_requests = 2
        assert backend.success_rate == 0.8

    def test_record_request_start(self):
        """Test recording request start."""
        backend = BackendInstance(name="backend1", url="http://localhost:8001")

        backend.record_request_start()
        assert backend.total_requests == 1
        assert backend.current_connections == 1

    def test_record_request_success(self):
        """Test recording request success."""
        backend = BackendInstance(name="backend1", url="http://localhost:8001")

        backend.current_connections = 1
        backend.record_request_success(100.0)

        assert backend.current_connections == 0
        assert backend.average_latency_ms == 100.0

        # Test exponential moving average
        backend.current_connections = 1
        backend.record_request_success(200.0)
        # alpha=0.2, so: 0.2 * 200 + 0.8 * 100 = 120
        assert backend.average_latency_ms == 120.0

    def test_record_request_failure(self):
        """Test recording request failure."""
        backend = BackendInstance(name="backend1", url="http://localhost:8001")

        backend.current_connections = 1
        backend.record_request_failure("Connection refused")

        assert backend.current_connections == 0
        assert backend.failed_requests == 1
        assert backend.last_error == "Connection refused"


class TestLoadBalancerMiddleware:
    """Test LoadBalancerMiddleware."""

    def test_initialization(self):
        """Test middleware initialization."""
        config = LoadBalancerConfig(enabled=True)
        backends = [
            BackendInstance(name="backend1", url="http://localhost:8001"),
            BackendInstance(name="backend2", url="http://localhost:8002"),
        ]

        lb = LoadBalancerMiddleware(config, backends)

        assert len(lb.backends) == 2
        assert lb.enabled
        assert not lb._health_check_running

    def test_enabled_false(self):
        """Test middleware when disabled."""
        config = LoadBalancerConfig(enabled=False)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        assert not lb.enabled

    def test_select_backend_weighted_round_robin(self):
        """Test weighted round-robin backend selection."""
        config = LoadBalancerConfig(enabled=True)
        backends = [
            BackendInstance(name="backend1", url="http://localhost:8001", weight=100),
            BackendInstance(name="backend2", url="http://localhost:8002", weight=200),
        ]

        lb = LoadBalancerMiddleware(config, backends)

        # Select multiple backends
        selections = {}
        for _ in range(30):
            backend = lb.select_backend()
            assert backend is not None
            selections[backend.name] = selections.get(backend.name, 0) + 1

        # backend2 should be selected more often (higher weight)
        assert selections["backend2"] > selections["backend1"]

    def test_select_backend_no_available_backends(self):
        """Test backend selection when none are available."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        # Mark backend as unavailable
        backends[0].state = BackendState.UNHEALTHY

        backend = lb.select_backend()
        assert backend is None

    def test_select_backend_connection_limit(self):
        """Test backend selection respects connection limits."""
        config = LoadBalancerConfig(enabled=True)
        backends = [
            BackendInstance(
                name="backend1", url="http://localhost:8001", max_concurrent_requests=1
            ),
            BackendInstance(name="backend2", url="http://localhost:8002"),
        ]

        lb = LoadBalancerMiddleware(config, backends)

        # Fill up backend1
        backends[0].current_connections = 1

        # Should select backend2
        backend = lb.select_backend()
        assert backend.name == "backend2"

    async def test_process_request_success(self):
        """Test successful request processing."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await lb.process_request(request, context)

        assert should_continue
        assert error is None
        assert "load_balancer_backend" in context
        assert context["load_balancer_backend"].name == "backend1"
        assert "load_balancer_start_time" in context

    async def test_process_request_no_backends(self):
        """Test request processing when no backends available."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        # Mark backend as unavailable
        backends[0].state = BackendState.UNHEALTHY

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await lb.process_request(request, context)

        assert not should_continue
        assert isinstance(error, HTTPException)
        assert error.status_code == 503

    async def test_process_response_success(self):
        """Test successful response processing."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        backend = backends[0]
        backend.current_connections = 1

        response = {}
        context = {
            "load_balancer_backend": backend,
            "load_balancer_start_time": 0.0,  # Will calculate latency
        }

        result = await lb.process_response(response, context)

        assert "gateway_metadata" in result
        assert "load_balancer" in result["gateway_metadata"]
        assert result["gateway_metadata"]["load_balancer"]["backend_name"] == "backend1"
        assert backend.current_connections == 0  # Released
        assert backend.total_requests == 1

    async def test_process_response_error_status(self):
        """Test response processing with error status."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        backend = backends[0]
        backend.current_connections = 1

        response = {"status": 500}
        context = {"load_balancer_backend": backend, "load_balancer_start_time": 0.0}

        await lb.process_response(response, context)

        # Should record failure
        assert backend.failed_requests == 1

    def test_get_backend_stats(self):
        """Test getting backend statistics."""
        config = LoadBalancerConfig(enabled=True)
        backends = [
            BackendInstance(name="backend1", url="http://localhost:8001", weight=150),
            BackendInstance(name="backend2", url="http://localhost:8002"),
        ]

        lb = LoadBalancerMiddleware(config, backends)

        backends[0].total_requests = 100
        backends[0].failed_requests = 5
        backends[0].average_latency_ms = 50.5
        backends[0].state = BackendState.HEALTHY

        stats = lb.get_backend_stats()

        assert len(stats) == 2

        backend1_stats = next(s for s in stats if s["name"] == "backend1")
        assert backend1_stats["weight"] == 150
        assert backend1_stats["total_requests"] == 100
        assert backend1_stats["failed_requests"] == 5
        assert backend1_stats["success_rate"] == 0.95
        assert backend1_stats["average_latency_ms"] == 50.5
        assert backend1_stats["state"] == "healthy"
        assert backend1_stats["is_available"]

    @pytest.mark.asyncio
    async def test_health_check_loop(self):
        """Test health check loop runs periodically."""
        config = LoadBalancerConfig(enabled=True)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(
            config, backends, health_check_interval=1  # 1 second for testing
        )

        # Mock the health check method
        with patch.object(
            lb, "_check_all_backends", new_callable=AsyncMock
        ) as mock_check:
            # Start health checks
            await lb.start_health_checks()

            # Wait a bit
            await asyncio.sleep(1.5)

            # Stop health checks
            await lb.stop_health_checks()

            # Should have been called at least once
            assert mock_check.call_count >= 1

    async def test_disabled_middleware_passthrough(self):
        """Test that disabled middleware allows requests through."""
        config = LoadBalancerConfig(enabled=False)
        backends = [BackendInstance(name="backend1", url="http://localhost:8001")]

        lb = LoadBalancerMiddleware(config, backends)

        request = Mock(spec=Request)
        context = {}

        # Should pass through without selecting backend
        should_continue, error = await lb.process_request(request, context)
        assert should_continue
        assert error is None
        assert "load_balancer_backend" not in context

        # Response should pass through unchanged
        response = {"test": "data"}
        result = await lb.process_response(response, context)
        assert result == response
