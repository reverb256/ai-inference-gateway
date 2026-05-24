"""Scope Enforcer middleware stub.

Created for backup deployment on Sentry (Nexus down).
This is a no-op stub that allows the gateway to start without Casdoor auth.
"""

from typing import Optional, Tuple
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware


class ScopeEnforcerMiddleware(Middleware):
    """Scope enforcement middleware (backup no-op stub).

    Enforces Casbin policy scopes on authenticated requests.
    In backup mode this is a no-op.
    """

    @property
    def enabled(self) -> bool:
        return False

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        return response
