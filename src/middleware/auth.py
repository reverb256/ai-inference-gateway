"""JWT Authentication middleware stub.

Created for backup deployment on Sentry (Nexus down).
This is a no-op stub that allows the gateway to start without Casdoor auth.
"""

from typing import Optional, Tuple
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware


class JWTAuthMiddleware(Middleware):
    """JWT authentication middleware (backup no-op stub).

    Authenticates requests using JWKS from Casdoor.
    In backup mode this is a no-op - authentication is disabled.
    """

    def __init__(self, config) -> None:
        self._enabled = getattr(config, "enabled", False)

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        return response
