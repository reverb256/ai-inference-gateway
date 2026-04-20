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

    # Comprehensive prompt injection patterns (case-insensitive)
    # Covers: instruction override, role play, system prompt extraction,
    # jailbreak, hidden HTML, encoded payloads, multi-language
    INJECTION_PATTERNS = [
        # --- Instruction override ---
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(everything\s+)?(above|before|prior)",
        r"forget\s+(the\s+)?(above|previous|prior|everything)",
        r"override\s+(your\s+)?instructions",
        r"pretend\s+you\s+are\s+not",
        r"act\s+as\s+if\s+you\s+are",
        r"you\s+are\s+now\s+(a\s+)?(DAN|evil|unfiltered|uncensored)",
        r"stop\s+being\s+(an?\s+)?(AI|assistant|helpful)",
        r"new\s+instructions?\s*:",
        r"system\s*(override|update|reset|instruction)\s*:",
        # --- Role play / persona ---
        r"simulate\s+(being|a|an)\s+(?!a\s+professional)",
        r"role[\-\s]?play\s+as",
        r"pretend\s+to\s+be\s+(?!a\s+professional)",
        r"you\s+are\s+no\s+longer\s+(an?\s+)?(AI|assistant|LLM)",
        r"(always|never)\s+(respond|answer|reply|comply|refuse)",
        # --- System prompt extraction ---
        r"(what\s+are\s+your|show\s+me\s+your|reveal\s+your)\s+(system|initial|original)\s+(prompt|instructions)",
        r"repeat\s+(your|the)\s+(system|initial|original)\s+(prompt|instructions)",
        r"output\s+(your|the)\s+system\s+prompt",
        # --- Credential/secret requests ---
        r"(show|reveal|give|send|share|display|print)\s+(me\s+)?(your\s+)?(api\s+key|secret|password|token|private\s+key|seed|credential)",
        r"(cat|type|read|print|echo)\s+.*(\.env|\.ssh|\.config|/etc/passwd|/etc/shadow)",
        r"\$\(\s*cat\s+",
        r"\b(exec|eval|system|subprocess|os\.system|os\.popen)\s*\(",
        # --- Hidden HTML/invisible content ---
        r"<!--\s*(system|important|instruction|override|admin)",
        r"<style[^>]*>[\s\S]*?(position\s*:\s*absolute|visibility\s*:\s*hidden|display\s*:\s*none|opacity\s*:\s*0|font-size\s*:\s*0|color\s*:\s*transparent)",
        r"<img\s+[^>]*(?:alt|title|src)\s*=\s*[\"'][^\"']*(?:ignore|override|system|instruction)[^\"']*[\"']",
        r"data:text/html[,;].*(?:ignore|override|system)",
        # --- Encoded payloads ---
        r"(base64|b64|atob|btoa)\s*[\(\[]",
        r"\\x[0-9a-fA-F]{2}.*\\x[0-9a-fA-F]{2}",
        r"\\u[0-9a-fA-F]{4}.*\\u[0-9a-fA-F]{4}",
        r"&#x[0-9a-fA-F]+;",
        # --- Shell injection in content ---
        r"(curl|wget|bash|sh|python|perl|ruby|node)\s+.*\|.*\|",
        r"(curl|wget)\s+https?://\S+\s*\|\s*(bash|sh|python)",
        r"`[^`]*(?:rm|curl|wget|bash|sh|chmod|chown)[^`]*`",
        # --- Multi-language (Korean, Chinese, Japanese, Russian) ---
        r"이전\s*(모든\s*)?지시",
        r"무시하고",
        r"API\s*키\s*(보여|알려|전달)",
        r"忽略\s*(之前|所有|上述)\s*(指令|指示)",
        r"假装\s*(你|您)\s*(不是|是)",
        r"前\s*の\s*指示\s*を\s*無視",
        r"игнорир(уй|овать)\s*(все\s+)?предыдущ",
        r"покажи\s+(мне\s+)?(свой\s+)?(api|ключ|пароль)",
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

    # Trusted internal IPs - skip injection detection for local traffic
    TRUSTED_IPS = [
        "10.1.1.",       # Internal cluster network
        "127.0.0.",      # Loopback
        "::1",           # IPv6 loopback
        "10.244.",       # K8s pod network
        "172.16.",       # Docker networks
    ]

    def _is_trusted_source(self, context: dict) -> bool:
        """Check if the request comes from a trusted internal IP."""
        client_ip = context.get("client_ip", "")
        for trusted in self.TRUSTED_IPS:
            if client_ip.startswith(trusted):
                return True
        return False

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

        messages = request_body.get("messages", [])

        # Check for prompt injection in messages (skip for trusted internal sources)
        if not (context and self._is_trusted_source(context)):
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
