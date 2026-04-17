"""
Concurrency Limiter Middleware.

Implements soft concurrency limits with graceful degradation.
Rather than blocking requests when limit is reached, allows them through
with warnings and metrics for monitoring.
"""

import logging
import asyncio
from typing import Dict, Tuple, Optional
from fastapi import Request, HTTPException
from .base import Middleware

logger = logging.getLogger(__name__)


class ConcurrencyLimiter(Middleware):
    """
    Soft concurrency limiter with graceful degradation.

    Tracks concurrent requests per model and logs warnings when limits
    are exceeded, but doesn't hard-block requests. This allows the
    system to handle load spikes gracefully.
    """

    def __init__(self, max_concurrency: int = 1):
        """
        Initialize concurrency limiter.

        Args:
            max_concurrency: Maximum concurrent requests per model (soft limit)
        """
        self.max_concurrency = max_concurrency
        # Map of model name to (semaphore, current_count)
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        self.counters: Dict[str, int] = {}
        # Lock to protect access to dicts
        self.lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return True

    async def _get_semaphore(self, model: str) -> asyncio.Semaphore:
        """
        Get or create semaphore for a model.

        Args:
            model: Model name

        Returns:
            Semaphore for the model
        """
        async with self.lock:
            if model not in self.semaphores:
                self.semaphores[model] = asyncio.Semaphore(self.max_concurrency)
                self.counters[model] = 0
                logger.info(
                    f"Created semaphore for model: {model} (max_concurrency={self.max_concurrency})"
                )
            return self.semaphores[model]

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Track concurrency with graceful degradation.

        Rather than blocking when limit is exceeded, allows request through
        with a warning. This enables graceful degradation under load.

        Args:
            request: FastAPI request object
            context: Request context

        Returns:
            Tuple of (should_continue, optional_error) - always continues
        """
        # Get model from context (set by endpoint)
        model = context.get("model", "default")

        # Get semaphore for this model
        semaphore = await self._get_semaphore(model)

        # Check current concurrency level
        async with self.lock:
            current_count = self.counters.get(model, 0)

        if current_count >= self.max_concurrency:
            # Over capacity - but allow through with warning (graceful degradation)
            logger.warning(
                f"Concurrency limit exceeded for model: {model} "
                f"({current_count}/{self.max_concurrency} active). "
                f"Allowing request through (graceful degradation)."
            )
            # Store degradation flag in context
            context["_concurrency_degraded"] = True
        else:
            # Within capacity
            logger.info(
                f"Concurrency within limit for model: {model} ({current_count}/{self.max_concurrency})"
            )
            context["_concurrency_degraded"] = False

        # Acquire permit
        await semaphore.acquire()

        # Increment counter
        async with self.lock:
            self.counters[model] = self.counters.get(model, 0) + 1

        logger.info(
            f"Acquired concurrency permit for model: {model} (active: {self.counters[model]})"
        )

        # Store permit in context for release later
        context["_concurrency_permit"] = (semaphore, model)
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Release concurrency permit after request completes.

        Args:
            response: Response dict
            context: Request context

        Returns:
            Modified response (not context!)
        """
        # Release permit if it was acquired
        if "_concurrency_permit" in context:
            semaphore, model = context["_concurrency_permit"]
            semaphore.release()

            # Decrement counter
            async with self.lock:
                self.counters[model] = self.counters.get(model, 1) - 1

            logger.info(
                f"Released concurrency permit for model: {model} (active: {self.counters[model]})"
            )
            context.pop("_concurrency_permit", None)

        return response
