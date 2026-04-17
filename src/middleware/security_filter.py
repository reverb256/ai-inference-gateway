# modules/services/ai-inference/ai_inference_gateway/middleware/security_filter.py
import re
import json
import logging
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware
from ai_inference_gateway.config import SecurityConfig


logger = logging.getLogger(__name__)


class SecurityFilterMiddleware(Middleware):
    """
    Security filter middleware for input validation and PII redaction.

    Features:
    - Detects and blocks prompt injection attempts
    - Redacts PII (email, phone, SSN, API keys, credit cards)
    - Enforces request size limits
    - Configurable enable/disable
    """

    # Prompt injection patterns (case-insensitive)
    INJECTION_PATTERNS = [
        r"ignore\s+all\s+previous\s+instructions",
        r"disregard\s+everything\s+above",
        r"forget\s+the\s+above",
        r"override\s+your\s+instructions",
        r"pretend\s+you\s+are\s+not",
        r"act\s+as\s+if\s+you\s+are",
    ]

    # PII redaction patterns
    PII_PATTERNS = {
        "EMAIL": [
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        ],
        "PHONE": [
            r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",  # US phone formats
        ],
        "SSN": [
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # SSN format
        ],
        "API_KEY": [
            r"\b(sk-|pk-|api_)[A-Za-z0-9_-]{20,}\b",  # Common API key formats
            r"\b[A-Za-z0-9]{32,}\b",  # Long alphanumeric strings
        ],
        "CREDIT_CARD": [
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card format
        ],
    }

    def __init__(self, config: SecurityConfig):
        """
        Initialize the security filter middleware.

        Args:
            config: Security filter configuration
        """
        self.config = config
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        # Compile injection patterns (case-insensitive)
        self._injection_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.INJECTION_PATTERNS
        ]

        # Compile PII patterns
        self._pii_patterns = {}
        if self.config.pii_redaction:
            for pii_type, patterns in self.PII_PATTERNS.items():
                self._pii_patterns[pii_type] = [
                    re.compile(pattern, re.IGNORECASE) for pattern in patterns
                ]

    def _calculate_request_size(self, request_body: dict) -> int:
        """
        Calculate the size of a request body in bytes.

        Args:
            request_body: The request body dict

        Returns:
            Size in bytes
        """
        try:
            return len(json.dumps(request_body).encode("utf-8"))
        except Exception:
            # If we can't serialize, estimate
            return len(str(request_body))

    def _detect_injection(self, text: str) -> bool:
        """
        Detect prompt injection attempts in text.

        Args:
            text: The text to check

        Returns:
            True if injection detected
        """
        for pattern in self._injection_patterns:
            if pattern.search(text):
                logger.warning(f"Prompt injection detected: {pattern.pattern}")
                return True
        return False

    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text.

        Args:
            text: The text to redact

        Returns:
            Text with PII redacted
        """
        redacted_text = text

        for pii_type, patterns in self._pii_patterns.items():
            for pattern in patterns:
                redacted_text = pattern.sub(f"[REDACTED:{pii_type}]", redacted_text)

        return redacted_text

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request for security validation.

        Checks for prompt injection, enforces size limits, and redacts PII.

        Args:
            request: The FastAPI Request object
            context: Context dict for passing state to other middleware

        Returns:
            Tuple of (should_continue, optional_error)
        """
        if not self.enabled:
            return True, None

        request_body = context.get("request_body")
        if not request_body:
            # No request body to validate
            return True, None

        # Check request size
        request_size = self._calculate_request_size(request_body)
        if request_size > self.config.max_request_size:
            logger.warning(
                f"Request size {request_size} exceeds limit "
                f"{self.config.max_request_size}"
            )
            error = HTTPException(
                status_code=413,
                detail=f"Request entity too large: {request_size} bytes",
            )
            return False, error

        # Check for prompt injection in messages
        messages = request_body.get("messages", [])
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str) and self._detect_injection(content):
                error = HTTPException(
                    status_code=400, detail="Request blocked: prompt injection detected"
                )
                return False, error

        # Redact PII if enabled
        if self.config.pii_redaction:
            for message in messages:
                content = message.get("content", "")
                if isinstance(content, str):
                    redacted = self._redact_pii(content)
                    message["content"] = redacted

                    # Log if redaction occurred
                    if redacted != content:
                        logger.info(f"PII redacted from message: {redacted[:50]}...")

        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process outgoing response (no-op for security filter).

        Args:
            response: The response dict
            context: State from request processing

        Returns:
            Unmodified response dict
        """
        # Security filter doesn't modify responses
        return response

    @property
    def enabled(self) -> bool:
        """Check if this middleware is enabled."""
        return self.config.enabled
