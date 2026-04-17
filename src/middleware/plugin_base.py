"""
Gateway Plugin System — extends the existing middleware pipeline.

Plugins are middleware with additional lifecycle hooks (startup/shutdown)
and optional error handling. They follow the same process_request/process_response
interface as the existing Middleware base class.
"""

import logging
from typing import Optional, Tuple

from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)


class GatewayPlugin(Middleware):
    """
    Base class for gateway plugins.

    Extends Middleware with lifecycle hooks and structured error handling.
    All plugins must implement enabled, process_request, and process_response.
    on_startup, on_shutdown, and on_error are optional overrides.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    async def on_startup(self, app) -> None:
        """Called during gateway startup. Optional."""
        pass

    async def on_shutdown(self) -> None:
        """Called during gateway shutdown. Optional."""
        pass

    async def on_error(self, context: dict, error: Exception) -> None:
        """Called when a request fails. Optional."""
        logger.debug(f"[{self.name}] Error: {error}")

    async def safe_process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """Wraps process_request with error handling."""
        try:
            return await self.process_request(request, context)
        except Exception as e:
            await self.on_error(context, e)
            # Don't block on plugin errors — continue pipeline
            return True, None

    async def safe_process_response(self, response: dict, context: dict) -> dict:
        """Wraps process_response with error handling."""
        try:
            return await self.process_response(response, context)
        except Exception as e:
            await self.on_error(context, e)
            return response  # Return unmodified on error
