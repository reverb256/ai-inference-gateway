# modules/services/ai-inference/ai_inference_gateway/middleware/rate_limiter.py
import logging
import time
import hashlib
from typing import Optional, Tuple, Dict
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware
from ai_inference_gateway.config import RateLimitingConfig
from ai_inference_gateway.utils.redis_client import RedisClient


logger = logging.getLogger(__name__)


class RateLimiterMiddleware(Middleware):
    """
    Token-based rate limiting middleware with Redis backing.

    Features:
    - Token estimation (4 characters per token)
    - Sliding windows: minute, hour, day
    - Per-API-key tracking
    - Custom quotas per API key
    - Redis-backed with in-memory fallback
    """

    # Approximate characters per token (varies by model)
    CHARS_PER_TOKEN = 4

    # Rate limit keys
    MINUTE_KEY = "tokens_minute"
    HOUR_KEY = "tokens_hour"
    DAY_KEY = "tokens_day"

    def __init__(self, config: RateLimitingConfig):
        """
        Initialize the rate limiter middleware.

        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self._client: Optional[RedisClient] = None
        self._custom_quotas: Dict[str, dict] = {}
        self._memory_limits: Dict[tuple, Tuple[float, float]] = {}

    async def connect(self):
        """Connect to Redis backend (with automatic fallback)."""
        if self.config.backend == "redis":
            self._client = RedisClient(
                redis_url=getattr(self.config, "redis_url", "redis://localhost:6379")
            )
            await self._client.connect()

    async def close(self):
        """Close the Redis connection."""
        if self._client:
            await self._client.close()

    def estimate_tokens(self, text: str) -> float:
        """
        Estimate token count for text.

        Uses a simple heuristic: 4 characters per token.
        This is approximate but works for rate limiting purposes.

        Args:
            text: The text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return len(text) / self.CHARS_PER_TOKEN

    def _get_api_key(self, request: Request) -> str:
        """
        Extract API key from request.

        Args:
            request: The FastAPI Request object

        Returns:
            API key string (or "default" if not found)
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()
        else:
            api_key = "default"

        # Hash the key for privacy
        return hashlib.sha256(api_key.encode()).hexdigest()[:16]

    def _get_quota(self, api_key: str) -> dict:
        """
        Get quota limits for an API key.

        Args:
            api_key: The API key identifier

        Returns:
            Dict with tokens_per_minute, tokens_per_hour, tokens_per_day
        """
        if api_key in self._custom_quotas:
            return self._custom_quotas[api_key]

        return {
            "tokens_per_minute": self.config.tokens_per_minute,
            "tokens_per_hour": self.config.tokens_per_hour,
            "tokens_per_day": self.config.tokens_per_day,
        }

    def set_custom_quota(
        self,
        api_key: str,
        tokens_per_minute: Optional[int] = None,
        tokens_per_hour: Optional[int] = None,
        tokens_per_day: Optional[int] = None,
    ):
        """
        Set custom quota for a specific API key.

        Args:
            api_key: The API key identifier
            tokens_per_minute: Custom minute limit
            tokens_per_hour: Custom hour limit
            tokens_per_day: Custom day limit
        """
        quota = {}

        if tokens_per_minute is not None:
            quota["tokens_per_minute"] = tokens_per_minute
        if tokens_per_hour is not None:
            quota["tokens_per_hour"] = tokens_per_hour
        if tokens_per_day is not None:
            quota["tokens_per_day"] = tokens_per_day

        self._custom_quotas[api_key] = quota

    async def _check_and_increment_limit(
        self, api_key: str, window: str, tokens: float, limit: int, expiry: int
    ) -> Tuple[bool, int]:
        """
        Check and increment rate limit for a window.

        Args:
            api_key: The API key identifier
            window: Window type (minute, hour, day)
            tokens: Token count to add
            limit: Maximum tokens allowed
            expiry: Key expiry in seconds

        Returns:
            Tuple of (allowed, current_usage)
        """
        key = f"ratelimit:{api_key}:{window}"

        if self._client:
            # Use Redis
            current = await self._client.get(key)
            current_tokens = float(current) if current else 0

            if current_tokens + tokens > limit:
                return False, int(current_tokens)

            # Increment counter
            await self._client.incrby(key, int(tokens))
            await self._client.expire(key, expiry)

            return True, int(current_tokens + tokens)
        else:
            # In-memory tracking (not thread-safe, but fallback)
            if not hasattr(self, "_memory_limits"):
                self._memory_limits = {}

            memory_key = (api_key, window)
            current_tokens, last_reset = self._memory_limits.get(memory_key, (0, 0))

            # Reset if window expired
            now = time.time()
            if now - last_reset > expiry:
                current_tokens = 0

            if current_tokens + tokens > limit:
                return False, int(current_tokens)

            self._memory_limits[memory_key] = (current_tokens + tokens, now)
            return True, int(current_tokens + tokens)

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request for rate limiting.

        Estimates tokens and checks against rate limits.

        Args:
            request: The FastAPI Request object
            context: Context dict for passing state to other middleware

        Returns:
            Tuple of (should_continue, optional_error)
        """
        if not self.enabled:
            return True, None

        # Get API key
        api_key = self._get_api_key(request)

        # Estimate token count
        request_body = context.get("request_body", {})
        messages = request_body.get("messages", [])

        total_tokens = 0
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                total_tokens += self.estimate_tokens(content)

        # Add small buffer for system tokens (2 tokens)
        total_tokens += 2  # Reserve minimal tokens for system prompt overhead

        # Get quota for this API key
        quota = self._get_quota(api_key)

        # Check minute limit
        allowed, usage = await self._check_and_increment_limit(
            api_key, self.MINUTE_KEY, total_tokens, quota["tokens_per_minute"], 60
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded (minute): {api_key} used {usage}/{quota['tokens_per_minute']}"
            )
            error = HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {usage}/{quota['tokens_per_minute']} tokens per minute",
            )
            return False, error

        # Check hour limit
        allowed, usage = await self._check_and_increment_limit(
            api_key, self.HOUR_KEY, total_tokens, quota["tokens_per_hour"], 3600
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded (hour): {api_key} used {usage}/{quota['tokens_per_hour']}"
            )
            error = HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {usage}/{quota['tokens_per_hour']} tokens per hour",
            )
            return False, error

        # Check day limit
        allowed, usage = await self._check_and_increment_limit(
            api_key, self.DAY_KEY, total_tokens, quota["tokens_per_day"], 86400
        )

        if not allowed:
            logger.warning(
                f"Rate limit exceeded (day): {api_key} used {usage}/{quota['tokens_per_day']}"
            )
            error = HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded: {usage}/{quota['tokens_per_day']} tokens per day",
            )
            return False, error

        # Add rate limit info to context
        context["rate_limit"] = {
            "api_key": api_key,
            "tokens_used": total_tokens,
            "quota_remaining": {
                "minute": quota["tokens_per_minute"] - usage,
                "hour": quota["tokens_per_hour"] - usage,
                "day": quota["tokens_per_day"] - usage,
            },
        }

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process outgoing response (no-op for rate limiter).

        Args:
            response: The response dict
            context: State from request processing

        Returns:
            Unmodified response dict
        """
        # Rate limiter doesn't modify responses
        return response

    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return self.config.enabled
