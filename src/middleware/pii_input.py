"""
PII Input Sanitization Middleware.

Two-pass input sanitization: regex fast path (always) + ML thorough path
(when PII detected or sensitivity flag set).

- Pass 1: PIIRedactor regex-based redaction
- Pass 2: PrivacyFilterClient ML-based redaction (graceful fallback)
- Thread-safe, async throughout
- Config-driven via PrivacyFilterConfig
- No-op when disabled
"""

import copy
import json
import logging
import threading
from typing import Any, Dict, List, Optional

from ai_inference_gateway.pii_redactor import PIIRedactor;
from ai_inference_gateway.privacy_filter_client import PrivacyFilterClient;

logger = logging.getLogger(__name__)


class PIIInputMiddleware:
    """
    Comprehensive PII input sanitization middleware.

    Provides two-pass sanitization for all incoming message content:
    1. Regex-based fast path via PIIRedactor (always runs)
    2. ML-based thorough path via PrivacyFilterClient (when flagged or PII detected)

    Gracefully falls back to regex-only when the ML service is unavailable.
    """

    def __init__(
        self,
        pii_redactor: Optional[PIIRedactor] = None,
        privacy_filter_client: Optional[PrivacyFilterClient] = None,
        enabled: bool = True,
    ):
        """
        Initialize PII input sanitization middleware.

        Args:
            pii_redactor: PIIRedactor instance for regex-based redaction (or None).
            privacy_filter_client: PrivacyFilterClient instance for ML-based redaction (or None).
            enabled: Whether sanitization is active. False = no-op passthrough.
        """
        self._pii_redactor = pii_redactor;
        self._privacy_filter_client = privacy_filter_client;
        self._enabled = enabled;
        self._lock = threading.Lock();

        logger.info(
            "PIIInputMiddleware initialized: enabled=%s, regex=%s, ml=%s",
            self._enabled,
            pii_redactor is not None,
            privacy_filter_client is not None,
        )

    @property
    def enabled(self) -> bool:
        """Whether this middleware is active."""
        return self._enabled;

    def _has_regex_redactor(self) -> bool:
        """Check if regex-based redactor is available."""
        return self._pii_redactor is not None;

    def _has_ml_client(self) -> bool:
        """Check if ML privacy filter client is available and circuit is closed."""
        return (
            self._privacy_filter_client is not None
            and self._privacy_filter_client.is_available
        );

    def _regex_redact(self, text: str) -> str:
        """Run regex-based PII redaction (Pass 1)."""
        if not self._has_regex_redactor():
            return text;
        return self._pii_redactor.redact(text);

    async def _ml_redact(self, text: str) -> str:
        """
        Run ML-based PII redaction (Pass 2).

        Returns original text if ML service is unavailable.
        """
        if not self._has_ml_client():
            return text;
        try:
            redacted_text, entities = await self._privacy_filter_client.redact(text);
            return redacted_text;
        except Exception as exc:
            logger.warning("ML redaction failed, falling back to regex: %s", exc);
            return text;

    async def sanitize_text(self, text: str, use_ml: bool = False) -> str:
        """
        Sanitize a single text string with two-pass redaction.

        Pass 1: regex-based fast path (always when enabled).
        Pass 2: ML-based thorough path (when use_ml=True and client available).

        Args:
            text: Input text to sanitize.
            use_ml: Whether to invoke ML-based redaction after regex pass.

        Returns:
            Sanitized text with PII redacted.
        """
        if not self._enabled or not text:
            return text;

        # Pass 1: regex fast path
        result = self._regex_redact(text);

        # Detect if PII was found in regex pass
        has_pii = result != text;

        if has_pii:
            logger.info("PII detected and redacted in input (regex pass)");

        # Pass 2: ML thorough path (if flagged or PII detected)
        if use_ml or has_pii:
            ml_result = await self._ml_redact(result);
            if ml_result != result:
                logger.info("PII further redacted in input (ML pass)");
            result = ml_result;

        return result;

    async def sanitize_messages(self, messages: list[dict]) -> list[dict]:
        """
        Sanitize ALL content in a messages array.

        Handles:
        - user messages: regex + ML (PII detected triggers ML)
        - tool messages: sanitize tool_call arguments (JSON strings)
        - system messages: sanitize but preserve structure
        - assistant messages: sanitize content (PII leak from context)
        - Content as string OR as content array (multi-modal, type "text")

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            Deep-copied list of messages with sanitized content.
        """
        if not self._enabled or not messages:
            return messages;

        sanitized = [];

        for msg in messages:
            sanitized_msg = copy.deepcopy(msg);
            role = msg.get("role", "");
            content = sanitized_msg.get("content");

            if content is None:
                sanitized.append(sanitized_msg);
                continue;

            # Handle content as string
            if isinstance(content, str):
                use_ml = role == "user";
                sanitized_msg["content"] = await self.sanitize_text(content, use_ml=use_ml);

            # Handle content as array (multi-modal)
            elif isinstance(content, list):
                sanitized_parts = [];
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = part.get("text", "");
                        use_ml = role == "user";
                        part["text"] = await self.sanitize_text(text, use_ml=use_ml);
                    sanitized_parts.append(part);
                sanitized_msg["content"] = sanitized_parts;

            # Sanitize tool_call arguments for tool messages
            if role == "tool":
                if "tool_call" in sanitized_msg:
                    tool_call = sanitized_msg["tool_call"];
                    if isinstance(tool_call, dict) and "arguments" in tool_call:
                        args = tool_call["arguments"];
                        if isinstance(args, str):
                            try:
                                parsed = json.loads(args);
                            except (json.JSONDecodeError, ValueError):
                                parsed = args;
                            sanitized_args = await self._sanitize_tool_arguments(parsed);
                            if isinstance(parsed, dict):
                                tool_call["arguments"] = json.dumps(sanitized_args);
                            else:
                                tool_call["arguments"] = str(sanitized_args);

                # Also sanitize content field of tool messages
                if isinstance(sanitized_msg.get("content"), str):
                    sanitized_msg["content"] = await self.sanitize_text(
                        sanitized_msg["content"], use_ml=False
                    );

            sanitized.append(sanitized_msg);

        return sanitized;

    async def _sanitize_tool_arguments(self, args: Any) -> Any:
        """
        Recursively sanitize tool call arguments.

        Handles dicts, lists, and string values.
        """
        if isinstance(args, str):
            return await self.sanitize_text(args, use_ml=False);
        elif isinstance(args, dict):
            return {
                key: await self._sanitize_tool_arguments(value)
                for key, value in args.items()
            };
        elif isinstance(args, list):
            return [await self._sanitize_tool_arguments(item) for item in args];
        return args;

    async def sanitize_embedding_input(self, input_data) -> list[str]:
        """
        Sanitize embedding input (string or list of strings).

        Args:
            input_data: Either a single string or a list of strings.

        Returns:
            List of sanitized strings.
        """
        if not self._enabled:
            if isinstance(input_data, list):
                return input_data;
            return [input_data] if isinstance(input_data, str) else [];

        if isinstance(input_data, str):
            sanitized = await self.sanitize_text(input_data, use_ml=True);
            return [sanitized];

        elif isinstance(input_data, list):
            results = [];
            for item in input_data:
                if isinstance(item, str):
                    sanitized = await self.sanitize_text(item, use_ml=True);
                    results.append(sanitized);
                else:
                    results.append(str(item));
            return results;

        return [];

    async def sanitize_search_query(self, query: str) -> str:
        """
        Sanitize search query text.

        Args:
            query: Search query string.

        Returns:
            Sanitized query with PII redacted.
        """
        if not self._enabled or not query:
            return query;

        # Always use ML for search queries to catch nuanced PII
        return await self.sanitize_text(query, use_ml=True);
