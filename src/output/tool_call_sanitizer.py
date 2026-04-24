"""
Tool Call Sanitizer.

Sanitizes PII from LLM tool call arguments across OpenAI and Anthropic
formats. Tool call arguments are a high-risk leakage channel because LLMs
frequently echo user PII into function parameters (e.g., send_email arguments
containing email addresses and SSNs) and reasoning models never self-filter
this channel.

Supported formats:
- OpenAI: tool_calls[].function.arguments (JSON string)
- Anthropic: tool_use content blocks with input dict
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from ai_inference_gateway.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)

_MAX_RECURSION_DEPTH = 10


class ToolCallSanitizer:
    """
    Sanitizes PII from tool call arguments.

    Recursively walks all string values in tool call arguments, applying
    PII redaction. Handles nested JSON, arrays, mixed types, and escaped
    strings. Malformed JSON is passed through safely after redacting the
    raw string.
    """

    def __init__(self, pii_redactor: PIIRedactor):
        """
        Initialize the tool call sanitizer.

        Args:
            pii_redactor: PIIRedactor instance for regex-based redaction.
        """
        self.pii_redactor = pii_redactor

    def sanitize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize PII from a list of tool call dicts.

        Handles both OpenAI format (function.arguments as JSON string) and
        Anthropic format (input dict or arguments dict).

        Args:
            tool_calls: List of tool call dicts.

        Returns:
            New list with sanitized tool calls. Original list is not mutated.
        """
        if not tool_calls:
            return tool_calls

        sanitized_calls = []
        sanitized_count = 0

        for tool_call in tool_calls:
            call = deepcopy(tool_call)
            was_modified = False

            # OpenAI format: function.arguments is a JSON string
            function = call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                args = function["arguments"]
                result = self._sanitize_arguments_string(args)
                if result is not None:
                    function["arguments"] = result
                    was_modified = True

            # Anthropic format: direct arguments dict
            elif "arguments" in call and isinstance(call["arguments"], dict):
                original = call["arguments"]
                sanitized = self._sanitize_value(original, depth=0)
                if sanitized is not original:
                    call["arguments"] = sanitized
                    was_modified = True

            # Anthropic tool_use format: input dict
            elif "input" in call and isinstance(call["input"], dict):
                original = call["input"]
                sanitized = self._sanitize_value(original, depth=0)
                if sanitized is not original:
                    call["input"] = sanitized
                    was_modified = True

            if was_modified:
                sanitized_count += 1

            sanitized_calls.append(call)

        if sanitized_count > 0:
            logger.info(
                "Tool call sanitization: %d/%d calls had arguments redacted",
                sanitized_count,
                len(tool_calls),
            )

        return sanitized_calls

    def _sanitize_arguments_string(self, args: Any) -> Optional[str]:
        """
        Sanitize arguments that may be a JSON string or dict.

        Returns None if no modification was needed, otherwise the sanitized
        JSON string or raw string.
        """
        if args is None:
            return None

        if isinstance(args, str):
            if not args.strip():
                return None

            try:
                parsed = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                # Malformed JSON — redact the raw string
                redacted = self.pii_redactor.redact(args)
                if redacted != args:
                    return redacted
                return None

            sanitized = self._sanitize_value(parsed, depth=0)
            if sanitized is not parsed:
                return json.dumps(sanitized, ensure_ascii=False)
            return None

        if isinstance(args, dict):
            sanitized = self._sanitize_value(args, depth=0)
            if sanitized is not args:
                return json.dumps(sanitized, ensure_ascii=False)
            return None

        return None

    def _sanitize_value(self, value: Any, depth: int) -> Any:
        """
        Recursively sanitize PII from an arbitrary value.

        Walks dicts and lists, applying PII redaction to all string values.

        Args:
            value: The value to sanitize (any type).
            depth: Current recursion depth.

        Returns:
            Sanitized value of the same type.
        """
        if depth > _MAX_RECURSION_DEPTH:
            logger.warning(
                "Tool call sanitization hit max recursion depth %d",
                _MAX_RECURSION_DEPTH,
            )
            return value

        if isinstance(value, str):
            redacted = self.pii_redactor.redact(value)
            return redacted

        if isinstance(value, dict):
            return {
                k: self._sanitize_value(v, depth + 1)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [self._sanitize_value(item, depth + 1) for item in value]

        # Numbers, bools, None — return as-is
        return value
