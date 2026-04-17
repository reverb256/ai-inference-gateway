"""
PII (Personally Identifiable Information) Redaction

Detects and redacts sensitive information from text to protect user privacy.

Features:
- Email addresses
- Phone numbers (international formats)
- SSN (Social Security Numbers)
- Credit card numbers
- IP addresses
- API keys and tokens
- Custom patterns
- Configurable redaction modes
"""

import re
import logging
from typing import List, Dict, Any, Optional, Pattern
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RedactionMode(Enum):
    """Redaction behavior modes."""

    REDACT = "redact"  # Replace with placeholder (e.g., [EMAIL])
    HASH = "hash"  # Replace with hash
    MASK = "mask"  # Partial masking (e.g., j***@example.com)
    REMOVE = "remove"  # Remove completely


@dataclass
class PIIPattern:
    """PII pattern definition."""

    name: str
    pattern: Pattern
    mode: RedactionMode = RedactionMode.REDACT
    description: str = ""
    examples: List[str] = field(default_factory=list)


class PIIRedactor:
    """
    Detect and redact PII from text.

    Supports common PII types with extensible pattern matching.
    """

    # Default PII patterns
    DEFAULT_PATTERNS = [
        PIIPattern(
            name="email",
            pattern=re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            mode=RedactionMode.REDACT,
            description="Email addresses",
            examples=["user@example.com", "john.doe@company.co.uk"],
        ),
        PIIPattern(
            name="phone",
            pattern=re.compile(
                r"(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}", re.IGNORECASE
            ),
            mode=RedactionMode.MASK,
            description="Phone numbers (US/international)",
            examples=["555-123-4567", "+1 (555) 123-4567"],
        ),
        PIIPattern(
            name="ssn",
            pattern=re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", re.IGNORECASE),
            mode=RedactionMode.REDACT,
            description="Social Security Numbers",
            examples=["123-45-6789", "123 45 6789"],
        ),
        PIIPattern(
            name="credit_card",
            pattern=re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b", re.IGNORECASE),
            mode=RedactionMode.REDACT,
            description="Credit card numbers",
            examples=["4111-1111-1111-1111", "4111111111111111"],
        ),
        PIIPattern(
            name="ip_address",
            pattern=re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", re.IGNORECASE),
            mode=RedactionMode.REDACT,
            description="IP addresses",
            examples=["192.168.1.1", "10.0.0.1"],
        ),
        PIIPattern(
            name="api_key",
            pattern=re.compile(r"\b[A-Za-z0-9]{32,}\b", re.IGNORECASE),
            mode=RedactionMode.REDACT,
            description="API keys and tokens (32+ chars)",
            examples=["abcd1234efgh5678ijkl9012mnop3456"],
        ),
        PIIPattern(
            name="bearer_token",
            pattern=re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
            mode=RedactionMode.REDACT,
            description="Bearer tokens",
            examples=["Bearer abcd1234efgh5678"],
        ),
        PIIPattern(
            name="password",
            pattern=re.compile(
                r'password["\s:=]+[^\s"\']+',  # password="xxx" or password: xxx
                re.IGNORECASE,
            ),
            mode=RedactionMode.REDACT,
            description="Passwords in config format",
            examples=['password="secret123"', "password: mypass"],
        ),
    ]

    def __init__(
        self,
        patterns: Optional[List[PIIPattern]] = None,
        enabled_patterns: Optional[List[str]] = None,
        redaction_char: str = "*",
    ):
        """
        Initialize PII redactor.

        Args:
            patterns: Custom PII patterns (uses defaults if None)
            enabled_patterns: List of pattern names to enable (all if None)
            redaction_char: Character for masking (default: *)
        """
        self.patterns = patterns or self.DEFAULT_PATTERNS
        self.redaction_char = redaction_char

        # Filter enabled patterns
        if enabled_patterns:
            self.patterns = [p for p in self.patterns if p.name in enabled_patterns]

        logger.info(
            f"PII Redactor initialized with {len(self.patterns)} patterns: "
            f"{', '.join(p.name for p in self.patterns)}"
        )

    def redact(self, text: str, mode: Optional[RedactionMode] = None) -> str:
        """
        Redact PII from text.

        Args:
            text: Input text
            mode: Override default redaction mode

        Returns:
            Text with PII redacted
        """
        if not text:
            return text

        result = text

        for pii_pattern in self.patterns:
            redaction_mode = mode or pii_pattern.mode

            if redaction_mode == RedactionMode.REDACT:
                result = pii_pattern.pattern.sub(
                    f"[{pii_pattern.name.upper()}]", result
                )

            elif redaction_mode == RedactionMode.HASH:
                result = self._hash_replace(result, pii_pattern)

            elif redaction_mode == RedactionMode.MASK:
                result = self._mask_replace(result, pii_pattern)

            elif redaction_mode == RedactionMode.REMOVE:
                result = pii_pattern.pattern.sub("", result)

        return result

    def _hash_replace(self, text: str, pii_pattern: PIIPattern) -> str:
        """Replace with hash."""

        def replacement(match):
            import hashlib

            value = match.group(0)
            hash_val = hashlib.sha256(value.encode()).hexdigest()[:8]
            return f"[{pii_pattern.name.upper()}:{hash_val}]"

        return pii_pattern.pattern.sub(replacement, text)

    def _mask_replace(self, text: str, pii_pattern: PIIPattern) -> str:
        """Replace with partial masking."""

        def replacement(match):
            value = match.group(0)

            # Keep first and last characters, mask middle
            if len(value) <= 4:
                return self.redaction_char * len(value)

            return value[0] + self.redaction_char * (len(value) - 2) + value[-1]

        return pii_pattern.pattern.sub(replacement, text)

    def detect(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect PII in text without redacting.

        Args:
            text: Input text

        Returns:
            Dict with detected PII instances
        """
        detections = {}

        for pii_pattern in self.patterns:
            matches = []

            for match in pii_pattern.pattern.finditer(text):
                matches.append(
                    {
                        "match": match.group(0),
                        "start": match.start(),
                        "end": match.end(),
                        "type": pii_pattern.name,
                    }
                )

            if matches:
                detections[pii_pattern.name] = matches

        return detections

    def redact_messages(
        self,
        messages: List[Dict[str, str]],
        redact_user: bool = True,
        redact_assistant: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Redact PII from chat messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            redact_user: Redact user messages
            redact_assistant: Redistant assistant messages

        Returns:
            Messages with PII redacted
        """
        redacted = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Apply redaction based on role
            if (
                (role == "user" and redact_user)
                or (role == "assistant" and redact_assistant)
                or role not in ("user", "assistant")
            ):
                content = self.redact(content)

            redacted.append({**msg, "content": content})

        return redacted

    def get_patterns(self) -> List[Dict[str, Any]]:
        """
        Get information about configured patterns.

        Returns:
            List of pattern info dicts
        """
        return [
            {
                "name": p.name,
                "description": p.description,
                "mode": p.mode.value,
                "examples": p.examples,
            }
            for p in self.patterns
        ]


# Singleton instance
_default_redactor: Optional[PIIRedactor] = None


def get_default_redactor() -> PIIRedactor:
    """Get or create default PII redactor."""
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = PIIRedactor()
    return _default_redactor


def redact_text(text: str, **kwargs) -> str:
    """
    Redact PII from text using default redactor.

    Convenience function.

    Args:
        text: Input text
        **kwargs: Arguments passed to redactor

    Returns:
        Redacted text
    """
    redactor = get_default_redactor()
    return redactor.redact(text, **kwargs)
