"""
Observability Plugin — OpenTelemetry + Prometheus integration.

Adds distributed tracing spans and Prometheus histogram metrics
for every request through the gateway.
"""

import logging
import time
from typing import Optional, Tuple

from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.plugin_base import GatewayPlugin

logger = logging.getLogger(__name__)


class ObservabilityPlugin(GatewayPlugin):
    """
    Records observability data for every request.

    - Prometheus histograms: request_duration_seconds, tokens_per_request
    - Structured log entries with trace context
    - OpenTelemetry-compatible span data (when OTel SDK available)
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._prometheus_available = False
        self._request_duration = None
        self._token_counter = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_startup(self, app) -> None:
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Histogram, Counter

            self._request_duration = Histogram(
                "gateway_request_duration_seconds",
                "Request duration in seconds",
                ["model", "backend", "specialization"],
                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            )
            self._token_counter = Counter(
                "gateway_tokens_total",
                "Total tokens processed",
                ["model", "backend", "direction"],  # direction: input|output
            )
            self._prometheus_available = True
            logger.info("ObservabilityPlugin initialized with Prometheus metrics")
        except ImportError:
            logger.info("prometheus_client not available — using log-only observability")

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        context["observability_start"] = time.time()
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        start_time = context.get("observability_start", time.time())
        duration = time.time() - start_time

        route_decision = context.get("route_decision")
        model = response.get("model", "unknown")
        backend = route_decision.backend if route_decision else "unknown"
        specialization = (
            route_decision.specialization.value
            if route_decision and route_decision.specialization
            else "general"
        )

        # Record Prometheus metrics
        if self._prometheus_available:
            self._request_duration.labels(
                model=model, backend=backend, specialization=specialization
            ).observe(duration)

            usage = response.get("usage", {})
            if usage.get("prompt_tokens"):
                self._token_counter.labels(
                    model=model, backend=backend, direction="input"
                ).inc(usage["prompt_tokens"])
            if usage.get("completion_tokens"):
                self._token_counter.labels(
                    model=model, backend=backend, direction="output"
                ).inc(usage["completion_tokens"])

        return response
