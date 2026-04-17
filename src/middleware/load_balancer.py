# modules/services/ai-inference/ai_inference_gateway/middleware/load_balancer.py
"""
Load Balancer Middleware for AI Inference Gateway.

Implements weighted round-robin load balancing across multiple backend instances
with health checking and failover support.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum
import httpx
from fastapi import Request, HTTPException

from ai_inference_gateway.middleware.base import Middleware
from ai_inference_gateway.config import LoadBalancerConfig

logger = logging.getLogger(__name__)


class BackendState(Enum):
    """Backend health state."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"  # Temporarily not accepting new connections


@dataclass
class BackendInstance:
    """
    Represents a backend instance for load balancing.

    Attributes:
        name: Unique identifier for this backend
        url: Base URL for the backend
        weight: Weight for weighted round-robin (higher = more requests)
        health_check_url: URL path for health checks (default: /health)
        max_concurrent_requests: Max concurrent requests before marking as busy
        state: Current health state
        current_connections: Current active connections
        total_requests: Total requests sent to this backend
        failed_requests: Total failed requests
        average_latency_ms: Rolling average latency in milliseconds
        last_health_check: Timestamp of last health check
        last_error: Last error message (if any)
    """

    name: str
    url: str
    weight: int = 100
    health_check_url: str = "/health"
    max_concurrent_requests: int = 100
    state: BackendState = BackendState.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    average_latency_ms: float = 0.0
    last_health_check: float = 0.0
    last_error: Optional[str] = None

    # Internal tracking for weighted round-robin
    _current_weight: int = field(default=0, init=False, compare=False)

    @property
    def is_available(self) -> bool:
        """Check if backend is available for requests."""
        return (
            self.state == BackendState.HEALTHY
            and self.current_connections < self.max_concurrent_requests
        )

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

    def record_request_start(self) -> None:
        """Record that a request is starting."""
        self.total_requests += 1
        self.current_connections += 1

    def record_request_success(self, latency_ms: float) -> None:
        """Record that a request completed successfully."""
        self.current_connections -= 1
        # Update rolling average latency (exponential moving average)
        alpha = 0.2  # Smoothing factor
        if self.average_latency_ms == 0:
            self.average_latency_ms = latency_ms
        else:
            self.average_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.average_latency_ms
            )

    def record_request_failure(self, error: str) -> None:
        """Record that a request failed."""
        self.current_connections -= 1
        self.failed_requests += 1
        self.last_error = error

    def reset_current_weight(self) -> None:
        """Reset current weight for round-robin cycle."""
        self._current_weight = 0


class LoadBalancerMiddleware(Middleware):
    """
    Load balancer middleware with weighted round-robin selection.

    Features:
    - Weighted round-robin backend selection
    - Periodic health checks
    - Automatic failover to healthy backends
    - Connection limits per backend
    - Latency tracking
    - Metrics integration
    """

    def __init__(
        self,
        config: LoadBalancerConfig,
        backends: List[BackendInstance],
        health_check_interval: int = 30,
        health_check_timeout: int = 5,
        metrics_helper=None,
    ):
        """
        Initialize load balancer middleware.

        Args:
            config: Load balancer configuration
            backends: List of backend instances
            health_check_interval: Seconds between health checks
            health_check_timeout: Timeout for health check requests
            metrics_helper: Optional MetricsHelper for metrics
        """
        self.config = config
        self.backends = backends
        self.health_check_interval = health_check_interval
        self.health_check_timeout = health_check_timeout
        self.metrics_helper = metrics_helper

        # Health check task
        self._health_check_task = None
        self._health_check_running = False

        # Round-robin state
        self._last_selected_index = -1

        logger.info(
            f"Load balancer initialized with {len(backends)} backends: "
            f"{[b.name for b in backends]}"
        )

    async def start_health_checks(self) -> None:
        """Start background health check loop."""
        if self._health_check_running:
            logger.warning("Health checks already running")
            return

        self._health_check_running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health check loop started")

    async def stop_health_checks(self) -> None:
        """Stop background health check loop."""
        if not self._health_check_running:
            return

        self._health_check_running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health check loop stopped")

    async def _health_check_loop(self) -> None:
        """Background task to periodically check backend health."""
        while self._health_check_running:
            try:
                await self._check_all_backends()
            except Exception as e:
                logger.error(f"Health check error: {e}", exc_info=True)

            await asyncio.sleep(self.health_check_interval)

    async def _check_all_backends(self) -> None:
        """Check health of all backends."""
        async with httpx.AsyncClient() as client:
            tasks = [
                self._check_backend_health(backend, client) for backend in self.backends
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_backend_health(
        self, backend: BackendInstance, client: httpx.AsyncClient
    ) -> None:
        """Check health of a single backend."""
        health_url = backend.url.rstrip("/") + backend.health_check_url
        was_healthy = backend.state == BackendState.HEALTHY

        try:
            start_time = time.time()
            response = await client.get(health_url, timeout=self.health_check_timeout)
            latency_ms = (time.time() - start_time) * 1000

            if response.status_code == 200:
                backend.state = BackendState.HEALTHY
                backend.last_health_check = time.time()
                backend.last_error = None

                # Update metrics
                if self.metrics_helper:
                    self.metrics_helper.set_backend_health(backend.name, True)
                    self.metrics_helper.observe_backend_latency(
                        backend.name, latency_ms / 1000
                    )

                if not was_healthy:
                    logger.info(f"Backend {backend.name} is now healthy")
            else:
                backend.state = BackendState.UNHEALTHY
                backend.last_error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Backend {backend.name} health check failed: "
                    f"HTTP {response.status_code}"
                )

                if self.metrics_helper:
                    self.metrics_helper.set_backend_health(backend.name, False)

        except Exception as e:
            backend.state = BackendState.UNHEALTHY
            backend.last_error = str(e)
            logger.warning(f"Backend {backend.name} health check error: {e}")

            if self.metrics_helper:
                self.metrics_helper.set_backend_health(backend.name, False)

    def select_backend(self) -> Optional[BackendInstance]:
        """
        Select a backend using weighted round-robin algorithm.

        Returns:
            Selected backend instance or None if no backends available

        Algorithm:
        1. Filter to available backends
        2. Add weight to current_weight for each backend
        3. Select backend with highest current_weight
        4. Subtract its effective weight from current_weight
        """
        available_backends = [b for b in self.backends if b.is_available]

        if not available_backends:
            logger.error("No available backends")
            return None

        # Weighted round-robin (smooth weighted round-robin)
        # Based on Nginx's algorithm
        total_weight = sum(b.weight for b in available_backends)
        best_backend = None
        max_current_weight = -1

        for backend in available_backends:
            backend._current_weight += backend.weight

            if backend._current_weight > max_current_weight:
                max_current_weight = backend._current_weight
                best_backend = backend

        if best_backend:
            best_backend._current_weight -= total_weight

            # Update metrics
            if self.metrics_helper:
                self.metrics_helper.inc_load_balancer_selections(best_backend.name)

            logger.debug(
                f"Selected backend: {best_backend.name} "
                f"(weight={best_backend.weight}, "
                f"connections={best_backend.current_connections})"
            )

        return best_backend

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request to select backend.

        Adds selected backend to context for use in forwarding logic.

        Args:
            request: The FastAPI Request object
            context: Context dict for passing state

        Returns:
            Tuple of (should_continue=True, error=None if backend selected)
        """
        if not self.enabled:
            return True, None

        # Select backend
        backend = self.select_backend()

        if backend is None:
            logger.error("No healthy backends available")
            return False, HTTPException(
                status_code=503, detail="No healthy backends available"
            )

        # Store backend in context for later use
        context["load_balancer_backend"] = backend
        context["load_balancer_start_time"] = time.time()

        backend.record_request_start()

        logger.debug(
            f"Request assigned to backend: {backend.name} "
            f"(connections: {backend.current_connections})"
        )

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process outgoing response to track backend performance.

        Updates backend statistics based on response status.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict with backend metadata
        """
        if not self.enabled:
            return response

        backend = context.get("load_balancer_backend")
        start_time = context.get("load_balancer_start_time")

        if backend and start_time:
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Check if response indicates success
            status_code = response.get("status", 200)
            if 200 <= status_code < 400:
                backend.record_request_success(latency_ms)
            else:
                backend.record_request_failure(f"HTTP {status_code}")

            # Add backend metadata to response
            backend_metadata = {
                "backend_name": backend.name,
                "backend_url": backend.url,
                "backend_latency_ms": round(latency_ms, 2),
                "backend_connections": backend.current_connections,
            }

            # Add to gateway_metadata if it exists
            if "gateway_metadata" not in response:
                response["gateway_metadata"] = {}
            response["gateway_metadata"]["load_balancer"] = backend_metadata

        return response

    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return self.config.enabled and len(self.backends) > 0

    def get_backend_stats(self) -> List[Dict]:
        """
        Get statistics for all backends.

        Returns:
            List of backend statistics dictionaries
        """
        return [
            {
                "name": b.name,
                "url": b.url,
                "state": b.state.value,
                "weight": b.weight,
                "connections": b.current_connections,
                "total_requests": b.total_requests,
                "failed_requests": b.failed_requests,
                "success_rate": round(b.success_rate, 3),
                "average_latency_ms": round(b.average_latency_ms, 2),
                "last_error": b.last_error,
                "is_available": b.is_available,
            }
            for b in self.backends
        ]
