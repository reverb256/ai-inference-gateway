"""
Provider-aware rate limiting with automatic discovery.

Tracks rate limits per provider/backend and respects them.
When 429s are encountered, updates discovered limits.
"""
import asyncio
import logging
import time
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProviderLimit:
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    last_429_at: Optional[float] = None
    cooldown_seconds: int = 60

    def is_in_cooldown(self) -> bool:
        if self.last_429_at is None:
            return False
        return time.time() - self.last_429_at < self.cooldown_seconds

    def get_cooldown_remaining(self) -> float:
        if self.last_429_at is None:
            return 0
        elapsed = time.time() - self.last_429_at
        return max(0, self.cooldown_seconds - elapsed)


class ProviderRateLimiter:
    """
    Tracks and enforces per-provider rate limits.

    Features:
    - Per-provider request tracking
    - Automatic cooldown on 429
    - Configurable limits per provider type
    - Local backends (vllm-local, llama-cpp) bypass limits
    """

    def __init__(self):
        self._limits: Dict[str, ProviderLimit] = {}
        self._request_counts: Dict[str, Dict[str, Tuple[int, float]]] = {}
        self._lock = asyncio.Lock()

    def get_config_for_provider(self, provider: str, model_id: str = "") -> ProviderLimit:
        """Get default limits for a provider."""
        if provider in self._limits:
            return self._limits[provider]

        limit = ProviderLimit()

        if provider in ("vllm-local", "llama-cpp"):
            limit.requests_per_minute = None
            limit.requests_per_hour = None
        elif provider == "kilo":
            if ":free" in model_id or "free" in model_id.lower():
                limit.requests_per_minute = 3
                limit.requests_per_hour = 200
            else:
                limit.requests_per_minute = 60
        elif provider == "nvidia":
            limit.requests_per_minute = 30
            limit.requests_per_hour = 1000
        elif provider == "openrouter":
            limit.requests_per_minute = 20
            limit.requests_per_hour = 500
            limit.requests_per_day = 5000
        elif provider in ("zai", "pollinations"):
            limit.requests_per_minute = 60
        else:
            limit.requests_per_minute = 60

        self._limits[provider] = limit
        return limit

    async def check_limit(
        self,
        provider: str,
        model_id: str = "",
        window_seconds: int = 60,
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if request to provider is allowed.

        Args:
            provider: Backend provider name
            model_id: Model ID (for free tier detection)
            window_seconds: Time window to check (60, 3600, or 86400)

        Returns:
            Tuple of (allowed, cooldown_remaining_seconds)
        """
        if provider in ("vllm-local", "llama-cpp"):
            return True, None

        async with self._lock:
            limit = self.get_config_for_provider(provider, model_id)
            now = time.time()

            if limit.is_in_cooldown():
                remaining = limit.get_cooldown_remaining()
                logger.warning(f"Provider {provider} in cooldown, {remaining:.0f}s remaining")
                return False, remaining

            window_key = f"{window_seconds}s"
            if provider not in self._request_counts:
                self._request_counts[provider] = {}

            counts = self._request_counts[provider]
            if window_key not in counts:
                counts[window_key] = (0, now)
            else:
                count, window_start = counts[window_key]
                if now - window_start >= window_seconds:
                    counts[window_key] = (0, now)

            count, window_start = counts[window_key]

            if window_seconds == 60 and limit.requests_per_minute:
                if count >= limit.requests_per_minute:
                    return False, None
            elif window_seconds == 3600 and limit.requests_per_hour:
                if count >= limit.requests_per_hour:
                    return False, None
            elif window_seconds == 86400 and limit.requests_per_day:
                if count >= limit.requests_per_day:
                    return False, None

            counts[window_key] = (count + 1, window_start)
            return True, None

    async def record_request(self, provider: str):
        """Record a request to provider (for tracking)."""
        async with self._lock:
            now = time.time()
            if provider not in self._request_counts:
                self._request_counts[provider] = {}

            counts = self._request_counts[provider]
            for window_key in ["60s", "3600s", "86400s"]:
                if window_key not in counts:
                    counts[window_key] = (0, now)
                else:
                    count, window_start = counts[window_key]
                    window_sec = int(window_key.rstrip("s"))
                    if now - window_start >= window_sec:
                        counts[window_key] = (0, now)

    async def record_429(self, provider: str, retry_after: Optional[int] = None):
        """
        Record a 429 response from provider.

        Sets cooldown and updates discovered limit.
        """
        async with self._lock:
            limit = self.get_config_for_provider(provider)
            limit.last_429_at = time.time()
            if retry_after:
                limit.cooldown_seconds = retry_after
            else:
                limit.cooldown_seconds = 60
            logger.warning(f"Recorded 429 from {provider}, cooldown: {limit.cooldown_seconds}s")

    def update_limit(
        self,
        provider: str,
        requests_per_minute: Optional[int] = None,
        requests_per_hour: Optional[int] = None,
        requests_per_day: Optional[int] = None,
    ):
        """Manually update limits for a provider."""
        limit = self.get_config_for_provider(provider)
        if requests_per_minute is not None:
            limit.requests_per_minute = requests_per_minute
        if requests_per_hour is not None:
            limit.requests_per_hour = requests_per_hour
        if requests_per_day is not None:
            limit.requests_per_day = requests_per_day

    def get_stats(self, provider: str) -> dict:
        """Get rate limit stats for a provider."""
        limit = self.get_config_for_provider(provider)
        stats = {
            "provider": provider,
            "limits": {
                "rpm": limit.requests_per_minute,
                "rph": limit.requests_per_hour,
                "rpd": limit.requests_per_day,
            },
            "in_cooldown": limit.is_in_cooldown(),
        }
        if limit.last_429_at:
            stats["last_429_ago_seconds"] = time.time() - limit.last_429_at
        return stats


GLOBAL_PROVIDER_LIMITER = ProviderRateLimiter()


async def check_provider_limit(provider: str, model_id: str = "") -> Tuple[bool, Optional[float]]:
    """Check if request to provider is allowed. Returns (allowed, cooldown_remaining)."""
    return await GLOBAL_PROVIDER_LIMITER.check_limit(provider, model_id, 60)


async def record_provider_request(provider: str):
    """Record a request to provider."""
    await GLOBAL_PROVIDER_LIMITER.record_request(provider)


async def record_provider_429(provider: str, retry_after: Optional[int] = None):
    """Record a 429 from provider."""
    await GLOBAL_PROVIDER_LIMITER.record_429(provider, retry_after)


def get_provider_stats(provider: str) -> dict:
    """Get stats for a provider."""
    return GLOBAL_PROVIDER_LIMITER.get_stats(provider)