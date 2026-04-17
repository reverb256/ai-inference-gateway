# modules/services/ai-inference/ai_inference_gateway/middleware/base.py
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from fastapi import Request, HTTPException


class Middleware(ABC):
    """
    Abstract base class for all middleware components.

    All middleware must implement process_request and process_response methods.
    This enables a clean pipeline architecture where each middleware can:
    - Inspect and modify incoming requests
    - Block requests with HTTP exceptions
    - Modify outgoing responses
    - Track state via the context dict
    """

    @abstractmethod
    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process an incoming request.

        Args:
            request: The FastAPI Request object
            context: A dict for passing state to other middleware

        Returns:
            Tuple of (should_continue, optional_error):
            - should_continue: False to short-circuit the pipeline
            - optional_error: HTTPException if blocking the request
        """
        pass

    @abstractmethod
    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process an outgoing response.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict
        """
        pass

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """
        Check if this middleware is enabled.

        Returns:
            True if middleware should process requests/responses
        """
        pass
