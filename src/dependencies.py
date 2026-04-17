"""
Shared FastAPI dependencies for the AI Inference Gateway.

Provides dependency injection for route handlers to access
gateway state, configuration, and services.
"""

from fastapi import Request

from ai_inference_gateway.main import GatewayState


def get_gateway_state(request: Request) -> GatewayState:
    """Get the shared gateway application state."""
    return request.app.state.gateway
