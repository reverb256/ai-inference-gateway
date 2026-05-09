"""
JWT Authentication Middleware.

Validates JWT tokens via JWKS and falls back to API key authentication
via VirtualKeyManager. Gated behind jwt_auth.enabled=False by default.
"""

import asyncio
import base64
import logging
import time
from typing import Optional, Tuple

import httpx
import jwt
from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)

# Default path for virtual keys database
_DEFAULT_DB_PATH = "/tmp/ai-inference/virtual_keys.db"


def _build_rsa_public_key(n: int, e: int):
    """Build an RSA public key from JWKS modulus and exponent components."""
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicNumbers
    from cryptography.hazmat.backends import default_backend

    numbers = RSAPublicNumbers(e, n)
    return numbers.public_key(default_backend())


class JWTAuthMiddleware(Middleware):
    """
    JWT authentication middleware with JWKS validation and API key fallback.

    Validates Bearer tokens against a remote JWKS endpoint, then falls back
    to API key validation via VirtualKeyManager when JWT auth fails.
    """

    def __init__(self, config, virtual_key_manager=None):
        """
        Initialize JWT auth middleware.

        Args:
            config: JWTAuthConfig instance with jwks_url, issuer, audience, refresh_interval
            virtual_key_manager: Optional VirtualKeyManager for API key fallback
        """
        self.config = config
        self._jwks_keys: dict = {}  # kid -> public key
        self._http_client: Optional[httpx.AsyncClient] = None
        self._refresh_task: Optional[asyncio.Task] = None
        self._last_refresh: float = 0.0

        # Lazily instantiate VirtualKeyManager if not provided
        self._virtual_key_manager = virtual_key_manager
        self._vk_manager_instantiated = virtual_key_manager is not None

    def _get_virtual_key_manager(self):
        """Lazily instantiate VirtualKeyManager on first use."""
        if not self._vk_manager_instantiated:
            try:
                from ai_inference_gateway.services.virtual_keys import VirtualKeyManager

                self._virtual_key_manager = VirtualKeyManager(db_path=_DEFAULT_DB_PATH)
                logger.info("JWTAuthMiddleware: VirtualKeyManager instantiated for API key fallback")
            except Exception as e:
                logger.warning(f"JWTAuthMiddleware: Cannot instantiate VirtualKeyManager: {e}")
            self._vk_manager_instantiated = True
        return self._virtual_key_manager

    @property
    def enabled(self) -> bool:
        """Middleware is enabled when jwt_auth is enabled."""
        return getattr(self.config, "enabled", False)

    async def _fetch_jwks(self) -> dict:
        """Fetch JWKS from the configured endpoint."""
        url = self.config.jwks_url
        try:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=10.0)

            response = await self._http_client.get(url)
            response.raise_for_status()
            data = response.json()
            keys = data.get("keys", [])
            logger.info(f"JWTAuthMiddleware: Fetched {len(keys)} JWKS keys from {url}")
            return data
        except Exception as e:
            logger.warning(f"JWTAuthMiddleware: Failed to fetch JWKS from {url}: {e}")
            return {"keys": []}

    async def _parse_jwks_keys(self, jwks_data: dict) -> dict:
        """Parse JWKS data into a dict of kid -> public key."""
        keys = {}
        for key_data in jwks_data.get("keys", []):
            try:
                kid = key_data.get("kid")
                if not kid:
                    continue

                n = int.from_bytes(
                    base64.urlsafe_b64decode(key_data["n"] + "=="), "big"
                )
                e = int.from_bytes(
                    base64.urlsafe_b64decode(key_data["e"] + "=="), "big"
                )
                public_key = _build_rsa_public_key(n, e)
                keys[kid] = public_key
                logger.debug(f"JWTAuthMiddleware: Parsed JWKS key kid={kid}")
            except Exception as e:
                logger.warning(f"JWTAuthMiddleware: Failed to parse JWKS key: {e}")
        return keys

    async def _refresh_jwks_loop(self):
        """Background loop to periodically refresh JWKS keys."""
        while True:
            try:
                await asyncio.sleep(self.config.refresh_interval)
                jwks_data = await self._fetch_jwks()
                if jwks_data.get("keys"):
                    new_keys = await self._parse_jwks_keys(jwks_data)
                    if new_keys:
                        self._jwks_keys = new_keys
                        self._last_refresh = time.time()
                        logger.info(f"JWTAuthMiddleware: Refreshed {len(new_keys)} JWKS keys")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"JWTAuthMiddleware: JWKS refresh error: {e}")

    async def _get_signing_key(self, token: str):
        """Extract kid from token header and return the matching public key."""
        try:
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            if kid and kid in self._jwks_keys:
                return self._jwks_keys[kid]
            # Try any key if no kid match
            if self._jwks_keys:
                return next(iter(self._jwks_keys.values()))
        except Exception as e:
            logger.debug(f"JWTAuthMiddleware: Failed to extract token header: {e}")
        return None

    async def _validate_jwt(self, token: str) -> Optional[dict]:
        """Validate a JWT token and return its payload."""
        signing_key = await self._get_signing_key(token)
        if signing_key is None:
            return None

        try:
            payload = jwt.decode(
                token,
                key=signing_key,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
                options={"verify_exp": True},
            )
            return payload
        except jwt.InvalidTokenError as e:
            logger.debug(f"JWTAuthMiddleware: JWT validation failed: {e}")
            return None

    async def _try_api_key(self, request: Request, context: dict) -> bool:
        """Try API key authentication. Returns True if successful."""
        # Check x-api-key header
        api_key = request.headers.get("x-api-key")
        # Also check Authorization header value (non-Bearer)
        if not api_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header and not auth_header.lower().startswith("bearer "):
                api_key = auth_header

        if not api_key:
            return False

        vkm = self._get_virtual_key_manager()
        if vkm is None:
            return False

        key = vkm.validate_key(api_key)
        if key is None:
            return False

        context["auth_method"] = "api_key"
        context["auth_subject"] = key.name
        context["auth_scopes"] = ["admin:gateway"]  # Virtual keys get admin scope by default
        logger.debug(f"JWTAuthMiddleware: API key auth succeeded for {key.name}")
        return True

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request: validate JWT or fall back to API key.

        On first request, fetches JWKS and starts background refresh loop.
        """
        # Initialize JWKS on first request
        if not self._jwks_keys:
            jwks_data = await self._fetch_jwks()
            self._jwks_keys = await self._parse_jwks_keys(jwks_data)
            self._last_refresh = time.time()
            if not self._refresh_task:
                self._refresh_task = asyncio.create_task(self._refresh_jwks_loop())

        # Step 1: Get Authorization header
        auth_header = request.headers.get("authorization", "")

        # Step 2: Try JWT validation if Bearer token present
        jwt_authenticated = False
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if token:
                payload = await self._validate_jwt(token)
                if payload:
                    context["auth_method"] = "jwt"
                    context["auth_subject"] = payload.get("sub", "")
                    context["auth_scopes"] = payload.get("scope", "").split()
                    context["auth_app"] = payload.get("aud", "")
                    logger.debug(f"JWTAuthMiddleware: JWT auth succeeded for {payload.get('sub', 'unknown')}")
                    jwt_authenticated = True

        # Step 3: If JWT failed, try API key
        if not jwt_authenticated:
            if await self._try_api_key(request, context):
                return True, None

        # Step 4: Neither auth method worked
        if not jwt_authenticated:
            return False, HTTPException(
                status_code=401,
                detail={"error": "authentication_required"},
            )

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """Pass through response unchanged."""
        return response
