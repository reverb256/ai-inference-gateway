"""
Reasoning Trace Sanitizer.

Sanitizes PII from LLM reasoning/thinking traces across three provider
formats. Reasoning content is a high-risk leakage channel because models
frequently reproduce user PII verbatim in chain-of-thought output, and
no current provider applies PII filtering to this channel.

Supported formats:
- OpenAI: response["reasoning_content"] string
- Anthropic: content[] blocks with type="thinking" and "thinking" field
- DeepSeek: <think...>...</think*> tags embedded in content strings
"""

import logging
import re
from copy import deepcopy
from typing import Any, Dict

from ai_inference_gateway.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)

# DeepSeek-style embedded thinking tags
_THINK_TAG_RE = re.compile(
    r"(<think[^>]*>)(.*?)(</think[^>]*>)",
    re.DOTALL | re.IGNORECASE,
)


class ReasoningSanitizer:
    """
    Sanitizes PII from reasoning/thinking traces in LLM responses.

    Detects and handles OpenAI, Anthropic, and DeepSeek reasoning formats.
    Preserves response structure — only PII values within reasoning content
    are modified.
    """

    def __init__(self, pii_redactor: PIIRedactor):
        """
        Initialize the reasoning sanitizer.

        Args:
            pii_redactor: PIIRedactor instance for regex-based redaction.
        """
        self.pii_redactor = pii_redactor

    def sanitize_reasoning(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize PII from reasoning traces in an LLM response dict.

        Detects and handles all three reasoning formats:
        1. OpenAI: reasoning_content key in message or at top level
        2. Anthropic: content array with type="thinking" blocks
        3. DeepSeek: <think...>...</think*> tags in content strings

        Args:
            response: Raw LLM response dict.

        Returns:
            New response dict with reasoning traces sanitized. Original
            dict is not mutated.
        """
        if not response:
            return response

        result = deepcopy(response)
        redaction_count = 0

        # --- OpenAI format: choices[].message.reasoning_content ---
        choices = result.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue

                # OpenAI reasoning_content
                for key in ("reasoning_content", "reasoning"):
                    rc = message.get(key)
                    if isinstance(rc, str) and rc:
                        redacted = self.pii_redactor.redact(rc)
                        if redacted != rc:
                            message[key] = redacted
                            redaction_count += 1

                # Anthropic-style content blocks inside message
                content = message.get("content")
                if isinstance(content, list):
                    count = self._sanitize_anthropic_blocks(content)
                    redaction_count += count

                # DeepSeek tags in string content
                elif isinstance(content, str) and content:
                    redacted = self._sanitize_deepseek_tags(content)
                    if redacted != content:
                        message["content"] = redacted
                        redaction_count += 1

        # --- Anthropic format: top-level content array (no choices key) ---
        anthropic_content = result.get("content")
        if isinstance(anthropic_content, list) and "choices" not in result:
            count = self._sanitize_anthropic_blocks(anthropic_content)
            redaction_count += count

        # --- Top-level reasoning_content (some proxy formats) ---
        for key in ("reasoning_content", "reasoning"):
            rc = result.get(key)
            if isinstance(rc, str) and rc and "choices" not in result:
                redacted = self.pii_redactor.redact(rc)
                if redacted != rc:
                    result[key] = redacted
                    redaction_count += 1

        if redaction_count > 0:
            logger.info(
                "Reasoning sanitization: %d trace(s) had PII redacted",
                redaction_count,
            )

        return result

    def _sanitize_anthropic_blocks(
        self,
        content_blocks: list,
    ) -> int:
        """
        Sanitize PII in Anthropic-style content blocks in-place.

        Only modifies blocks with type="thinking" — redacts the "thinking"
        field. Text blocks and other types are left untouched.

        Args:
            content_blocks: List of content block dicts (modified in-place).

        Returns:
            Number of blocks that had PII redacted.
        """
        count = 0

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "thinking":
                thinking = block.get("thinking")
                if isinstance(thinking, str) and thinking:
                    redacted = self.pii_redactor.redact(thinking)
                    if redacted != thinking:
                        block["thinking"] = redacted
                        count += 1

        return count

    def _sanitize_deepseek_tags(self, content: str) -> str:
        """
        Sanitize PII inside DeepSeek <think...>...</think*> tags.

        Preserves the tag structure, only redacting PII within the tag
        content.

        Args:
            content: Content string possibly containing thinking tags.

        Returns:
            Content with PII redacted inside thinking sections.
        """
        if not content or "<think" not in content.lower():
            return content

        def _redact_match(match: re.Match) -> str:
            open_tag = match.group(1)
            think_content = match.group(2)
            close_tag = match.group(3)
            redacted = self.pii_redactor.redact(think_content)
            return f"{open_tag}{redacted}{close_tag}"

        return _THINK_TAG_RE.sub(_redact_match, content)
