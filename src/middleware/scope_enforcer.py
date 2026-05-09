"""
Scope Enforcer Middleware.

Enforces endpoint-to-scope mapping for JWT-authenticated requests.
Skips scope checks for API key authentication (backward compatibility).
"""

import fnmatch
import logging
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)

# Default endpoint-to-scope mapping
DEFAULT_SCOPE_MAP: Dict[Tuple[str, str], str] = {
    ("POST", "/v1/chat/completions"): "inference:chat",
    ("POST", "/v1/messages"): "inference:chat",
    ("POST", "/v1/embeddings"): "inference:embed",
    ("POST", "/v1/images/generations"): "inference:vision",
    ("GET", "/v1/models"): "models:read",
    ("POST", "/admin/keys"): "keys:admin",
    ("GET", "/admin/keys"): "keys:admin",
}

# Wildcard prefix patterns: (method, fnmatch_pattern) -> required_scope
WILDCARD_SCOPE_MAP: Dict[Tuple[str, str], str] = {
    ("POST", "/admin/*"): "admin:gateway",
}


class ScopeEnforcerMiddleware(Middleware):
    """
    Enforces OAuth-style scope requirements on API endpoints.

    For JWT-authenticated requests, checks that the token's scopes include
    the scope required for the requested endpoint. API key auth bypasses
    scope checks for backward compatibility.
    """

    def __init__(self, scope_map: Optional[Dict[Tuple[str, str], str]] = None):
        """
        Initialize scope enforcer.

        Args:
            scope_map: Optional custom (method, path) -> scope mapping.
                       Defaults to built-in mapping.
        """
        self._exact_map = scope_map or DEFAULT_SCOPE_MAP

    @property
    def enabled(self) -> bool:
        """Scope enforcer is always enabled when instantiated."""
        return True

    def _find_required_scope(self, method: str, path: str) -> Optional[str]:
        """Find the required scope for a method+path combination."""
        # Check exact matches first
        scope = self._exact_map.get((method, path))
        if scope:
            return scope

        # Check wildcard patterns
        for (wm, wpattern), scope in WILDCARD_SCOPE_MAP.items():
            if method == wm and fnmatch.fnmatch(path, wpattern):
                return scope

        return None

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Check that the authenticated identity has the required scope.

        Skips enforcement for API key auth (backward compat).
        """
        auth_method = context.get("auth_method")

        # Skip scope check for API key auth (backward compat)
        if auth_method == "api_key":
            return True, None

        # If no auth method set, skip (auth middleware handles 401)
        if auth_method != "jwt":
            return True, None

        scopes = context.get("auth_scopes", [])
        required_scope = self._find_required_scope(request.method, request.url.path)

        # No scope mapping for this endpoint — allow
        if required_scope is None:
            return True, None

        # Check if required scope is in granted scopes
        if required_scope in scopes:
            return True, None

        # Deny
        logger.warning(
            f"ScopeEnforcer: insufficient_scope required={required_scope} "
            f"granted={scopes} path={request.url.path}"
        )
        return False, HTTPException(
            status_code=403,
            detail={
                "error": {
                    "code": -32001,
                    "message": "insufficient_scope",
                    "data": {
                        "required": required_scope,
                        "granted": scopes,
                    },
                }
            },
        )

    async def process_response(self, response: dict, context: dict) -> dict:
        """Pass through response unchanged."""
        return response
