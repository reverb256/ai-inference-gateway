"""
OTel Span Attribute Sanitizer.

Sanitizes PII from OpenTelemetry span attributes before export to
Tempo/Grafana. Selectively targets only string-valued attributes in
keys likely to carry LLM payload content, leaving metrics and numeric
attributes untouched.
"""

import logging
from typing import Any, Dict

from ai_inference_gateway.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)

# Attribute key prefixes/names that carry PII-bearing LLM content.
_PII_KEY_PREFIXES = ("llm.",)
_PII_KEY_EXACT = frozenset(
    {
        "input",
        "output",
        "content",
        "message",
        "prompt",
        "response",
        "reasoning",
        "tool",
        "arguments",
    }
)


class OTelSpanSanitizer:
    """
    Sanitizes PII from OTel span attributes.

    Only string values in PII-bearing keys are redacted. Numeric, bool,
    and other non-string values pass through unchanged. Non-matching
    keys (metrics, counts, latencies) are never touched.
    """

    def __init__(self, redactor: PIIRedactor):
        """
        Initialize the OTel span sanitizer.

        Args:
            redactor: PIIRedactor instance for regex-based redaction.
        """
        self._redactor = redactor

    def sanitize_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return a new dict with PII sanitized from relevant attributes.

        Does not mutate the input dict. Only string values in keys
        matching the PII-bearing key set are redacted.

        Args:
            attributes: OTel span attributes dict.

        Returns:
            New dict with sanitized string values where applicable.
        """
        if not attributes:
            return dict(attributes) if attributes else {}

        sanitized_count = 0
        result = {}

        for key, value in attributes.items():
            if self._is_pii_key(key) and isinstance(value, str):
                redacted = self._redactor.redact(value)
                if redacted != value:
                    sanitized_count += 1
                result[key] = redacted
            else:
                result[key] = value

        if sanitized_count > 0:
            logger.debug(
                "OTel span sanitizer: %d attribute(s) had PII redacted",
                sanitized_count,
            )

        return result

    @staticmethod
    def _is_pii_key(key: str) -> bool:
        """
        Check if an attribute key is likely to carry PII-bearing content.

        Args:
            key: The attribute key to check.

        Returns:
            True if the key matches a PII-bearing pattern.
        """
        if key in _PII_KEY_EXACT:
            return True
        for prefix in _PII_KEY_PREFIXES:
            if key.startswith(prefix):
                return True
        return False
