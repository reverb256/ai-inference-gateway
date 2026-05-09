"""
Casbin Policy Enforcement Middleware.

Delegates authorization decisions to a remote Casbin enforcement API.
Uses local caching with TTL for performance. Fails open when the API
is unreachable for resilience.
"""

import logging
import time
from typing import Dict, Optional, Tuple

import httpx
from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)


class CasbinEnforcerMiddleware(Middleware):
    """
    Enforces Casbin policies via a remote enforcement API.

    For each request, sends (sub, obj, act) to the Casbin API and
    caches the result for the configured TTL. If the API is unreachable,
    the request is allowed (fail-open) to maintain availability.
    """

    def __init__(self, api_url: str = "https://auth.lan/api/enforce", cache_ttl: int = 60):
        """
        Initialize Casbin enforcer.

        Args:
            api_url: URL of the Casbin enforcement API endpoint
            cache_ttl: Cache TTL in seconds for enforcement decisions
        """
        self._api_url = api_url
        self._cache_ttl = cache_ttl
        self._cache: Dict[Tuple[str, str, str], Tuple[bool, float]] = {}
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def enabled(self) -> bool:
        """Casbin enforcer is always enabled when instantiated."""
        return True

    def _extract_obj(self, request: Request) -> str:
        """
        Extract the resource object from the request.

        Uses the URL path segments to determine the resource type.
        """
        path = request.url.path

        # Known resource patterns
        if "/v1/chat/completions" in path or "/v1/messages" in path:
            return "inference:chat"
        if "/v1/embeddings" in path:
            return "inference:embed"
        if "/v1/images/generations" in path:
            return "inference:vision"
        if "/v1/models" in path:
            return "models"
        if "/admin/keys" in path:
            return "admin:keys"

        # Default: use the path itself
        return path

    async def _check_policy(self, sub: str, obj: str, act: str) -> bool:
        """
        Check authorization via the Casbin API with local caching.

        Args:
            sub: Subject (authenticated user/identity)
            obj: Object (resource being accessed)
            act: Action (HTTP method)

        Returns:
            True if allowed, False if denied
        """
        cache_key = (sub, obj, act)
        now = time.time()

        # Check cache
        if cache_key in self._cache:
            allowed, timestamp = self._cache[cache_key]
            if now - timestamp < self._cache_ttl:
                return allowed

        # Query remote API
        try:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=5.0)

            response = await self._http_client.post(
                self._api_url,
                json={"sub": sub, "obj": obj, "act": act},
            )

            if response.status_code == 200:
                data = response.json()
                allowed = data.get("allowed", True)
                self._cache[cache_key] = (allowed, now)
                return allowed
            else:
                logger.warning(
                    f"CasbinEnforcer: API returned status {response.status_code}"
                )
                # Fail open
                return True

        except Exception as e:
            logger.warning(f"CasbinEnforcer: API unreachable ({e}), failing open")
            return True

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Enforce Casbin policy on the request.

        Gets subject from context (set by JWTAuthMiddleware), extracts
        object and action from the request, and checks against Casbin.
        """
        sub = context.get("auth_subject", "")
        if not sub:
            # No authenticated subject — skip (auth middleware handles 401)
            return True, None

        obj = self._extract_obj(request)
        act = request.method

        allowed = await self._check_policy(sub, obj, act)

        if not allowed:
            logger.warning(
                f"CasbinEnforcer: denied sub={sub} obj={obj} act={act}"
            )
            return False, HTTPException(
                status_code=403,
                detail={"error": "forbidden_by_policy"},
            )

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """Pass through response unchanged."""
        return response
