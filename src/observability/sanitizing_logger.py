"""
PII Sanitizing Logging Filter.

Intercepts log records before export and sanitizes PII from message
text and format arguments. Prevents raw PII from flowing into
Loki/Tempo/Grafana via the OTel logging pipeline.
"""

import logging
import threading
from typing import Dict

from ai_inference_gateway.pii_redactor import PIIRedactor

# Short-circuit threshold: messages shorter than this with no digits and
# no "@" cannot contain the PII patterns the redactor checks.
_MIN_SANITIZE_LENGTH = 20


class PIISanitizingFilter(logging.Filter):
    """
    Logging filter that sanitizes PII from log messages.

    Attaches to any stdlib logger and runs before the record is emitted
    to handlers. Sanitizes ``record.msg`` and string elements in
    ``record.args``.
    """

    def __init__(self, redactor: PIIRedactor, name: str = ""):
        """
        Initialize the PII sanitizing filter.

        Args:
            redactor: PIIRedactor instance for regex-based redaction.
            name: Optional filter name (passed to logging.Filter).
        """
        super().__init__(name)
        self._redactor = redactor
        self._lock = threading.Lock()
        self._total_messages = 0
        self._messages_sanitized = 0
        self._redactions_performed = 0

    # ------------------------------------------------------------------
    # Core filter
    # ------------------------------------------------------------------

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Sanitize a log record in-place.

        Always returns True (never suppresses records).

        Args:
            record: The log record to sanitize.
        """
        with self._lock:
            self._total_messages += 1

        sanitized_this_record = False

        # --- Sanitize the message string ---
        if isinstance(record.msg, str) and self._should_sanitize(record.msg):
            original = record.msg
            record.msg = self._redactor.redact(original)
            if record.msg != original:
                sanitized_this_record = True

        # --- Sanitize format arguments ---
        if record.args is not None:
            args = record.args

            # args is typically a tuple or dict
            if isinstance(args, dict):
                new_args = {}
                for key, value in args.items():
                    if isinstance(value, str) and self._should_sanitize(value):
                        redacted = self._redactor.redact(value)
                        if redacted != value:
                            sanitized_this_record = True
                        new_args[key] = redacted
                    else:
                        new_args[key] = value
                record.args = new_args
            elif isinstance(args, tuple):
                new_args = []
                for value in args:
                    if isinstance(value, str) and self._should_sanitize(value):
                        redacted = self._redactor.redact(value)
                        if redacted != value:
                            sanitized_this_record = True
                        new_args.append(redacted)
                    else:
                        new_args.append(value)
                record.args = tuple(new_args)

        if sanitized_this_record:
            with self._lock:
                self._messages_sanitized += 1

        return True

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """
        Return redaction statistics.

        Returns:
            Dict with total_messages, messages_sanitized, and
            redactions_performed counts.
        """
        with self._lock:
            return {
                "total_messages": self._total_messages,
                "messages_sanitized": self._messages_sanitized,
                "redactions_performed": self._messages_sanitized,
            }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _should_sanitize(text: str) -> bool:
        """
        Quick heuristic to skip obviously clean short messages.

        Messages under the length threshold that contain no digits and
        no "@" symbol cannot match the PIIRedactor's email, phone, SSN,
        credit card, IP, or API key patterns.

        Args:
            text: The message to evaluate.

        Returns:
            True if the message should be passed through the redactor.
        """
        if len(text) < _MIN_SANITIZE_LENGTH:
            if not any(c.isdigit() for c in text) and "@" not in text:
                return False
        return True
