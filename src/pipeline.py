# modules/services/ai-inference/ai_inference_gateway/pipeline.py
import logging
from typing import List, Optional, Tuple

from fastapi import Request, HTTPException

from ai_inference_gateway.middleware.base import Middleware


logger = logging.getLogger(__name__)


class MiddlewarePipeline:
    """
    Orchestrates execution of middleware components.

    The pipeline processes requests in order (first to last) and responses
    in reverse order (last to first). This allows middleware to:
    - Pre-process requests before they reach the backend
    - Post-process responses before they reach the client

    If any middleware blocks a request (returns should_continue=False),
    the pipeline short-circuits and subsequent middleware are not executed.
    """

    def __init__(self):
        """Initialize an empty middleware pipeline."""
        self._middleware: List[Middleware] = []

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """
        Add middleware to the pipeline.

        Middleware are executed in the order they are added for requests,
        and in reverse order for responses.

        Args:
            middleware: The middleware to add

        Returns:
            Self for method chaining
        """
        self._middleware.append(middleware)
        logger.debug(f"Added middleware: {middleware.__class__.__name__}")
        return self

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process an incoming request through all middleware.

        Middleware are executed in the order they were added.
        If any middleware returns should_continue=False, the pipeline
        short-circuits and no further middleware are executed.

        Args:
            request: The FastAPI Request object
            context: A dict for passing state between middleware

        Returns:
            Tuple of (should_continue, optional_error):
            - should_continue: False if pipeline was short-circuited
            - optional_error: HTTPException if request was blocked
        """
        logger.debug(f"Processing request through {len(self._middleware)} middleware")

        for middleware in self._middleware:
            # Skip disabled middleware
            if not middleware.enabled:
                logger.debug(
                    f"Skipping disabled middleware: {middleware.__class__.__name__}"
                )
                continue

            try:
                should_continue, error = await middleware.process_request(
                    request, context
                )

                if not should_continue:
                    logger.info(
                        f"Request blocked by middleware: {middleware.__class__.__name__}"
                    )
                    return False, error
            except Exception as e:
                logger.error(
                    f"Error in middleware {middleware.__class__.__name__}: {e}",
                    exc_info=True,
                )
                # On exception, block the request
                return False, HTTPException(
                    status_code=500, detail=f"Middleware error: {str(e)}"
                )

        logger.debug("Request processed successfully through all middleware")
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process an outgoing response through all middleware.

        Middleware are executed in reverse order (last to first).
        All middleware execute regardless of what happened during request
        processing, allowing cleanup and response modification.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict
        """
        logger.debug("Processing response through middleware")

        # Execute in reverse order
        for middleware in reversed(self._middleware):
            # Skip disabled middleware
            if not middleware.enabled:
                continue

            try:
                response = await middleware.process_response(response, context)
            except Exception as e:
                logger.error(
                    f"Error in middleware {middleware.__class__.__name__} "
                    f"response processing: {e}",
                    exc_info=True,
                )
                # Continue processing other middleware on error

        logger.debug("Response processed successfully")
        return response

    def clear(self):
        """Remove all middleware from the pipeline."""
        self._middleware.clear()
        logger.debug("Cleared all middleware from pipeline")

    @property
    def count(self) -> int:
        """Get the number of middleware in the pipeline."""
        return len(self._middleware)

    @property
    def middleware(self) -> List[Middleware]:
        """Get the list of middleware in the pipeline."""
        return self._middleware.copy()
