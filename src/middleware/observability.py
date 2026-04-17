# modules/services/ai-inference/ai_inference_gateway/middleware/observability.py
import time
import uuid
import logging
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware
from ai_inference_gateway.config import ObservabilityConfig


logger = logging.getLogger(__name__)


class ObservabilityMiddleware(Middleware):
    """
    Observability middleware for request tracking and logging.

    Features:
    - Generates or preserves X-Request-ID header for tracing
    - Adds gateway_metadata to responses with processing time
    - Tracks request processing time
    - Structured logging support
    """

    def __init__(self, config: ObservabilityConfig):
        """
        Initialize the observability middleware.

        Args:
            config: Observability configuration
        """
        self.config = config

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request to add tracking information.

        Generates or preserves request ID and records start time.

        Args:
            request: The FastAPI Request object
            context: Context dict for passing state to other middleware

        Returns:
            Tuple of (should_continue=True, error=None)
        """
        if not self.enabled:
            return True, None

        # Extract or generate request ID
        request_id_header = self.config.request_id_header
        request_id = request.headers.get(request_id_header)

        if request_id:
            # Preserve existing request ID (e.g., from load balancer)
            # Note: FastAPI headers are case-insensitive, but we use the exact header name
            context["request_id"] = request_id
            logger.debug(f"Using existing request ID: {request_id}")
        else:
            # Generate new request ID
            request_id = str(uuid.uuid4())
            context["request_id"] = request_id
            logger.debug(f"Generated new request ID: {request_id}")

        # Record start time for processing time tracking
        context["start_time"] = time.time()

        # Store in request state for access in endpoints
        if hasattr(request, "state"):
            request.state.request_id = request_id

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process outgoing response to add metadata.

        Adds gateway_metadata with request ID and processing time.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict with gateway_metadata
        """
        if not self.enabled:
            return response

        # Calculate processing time
        start_time = context.get("start_time")
        processing_time_ms = 0
        if start_time:
            processing_time_ms = (time.time() - start_time) * 1000

        # Build gateway metadata (merge with existing if present)
        metadata = response.get("gateway_metadata", {})
        metadata.update(
            {
                "request_id": context.get("request_id", "unknown"),
                "processing_time_ms": round(processing_time_ms, 2),
            }
        )

        # Add/merge metadata to response
        response["gateway_metadata"] = metadata

        if self.config.structured_logging:
            logger.info(
                "Request completed",
                extra={
                    "request_id": metadata["request_id"],
                    "processing_time_ms": metadata["processing_time_ms"],
                },
            )

        return response

    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return self.config.enabled
