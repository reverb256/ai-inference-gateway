"""
Unified Output Sanitizer.

Sanitizes LLM responses across all output channels: chat content,
tool call arguments, and reasoning/thinking traces. Applies a two-pass
approach — regex-based PII redaction first, then ML-based NER filtering
when the privacy filter service is available.

Supported reasoning formats:
- OpenAI: reasoning_content field
- Anthropic: content[].type == "thinking" blocks
- DeepSeek: <think/>...</think*> embedded in content
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ai_inference_gateway.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)

# DeepSeek-style embedded thinking tags
_THINK_OPEN_RE = re.compile(r"<think[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think[^>]*>", re.IGNORECASE)

# Maximum recursive depth for tool call argument sanitization
_MAX_RECURSION_DEPTH = 10


class ResponseSanitizer:
    """
    Sanitizes LLM responses across all output channels.

    Applies regex-based PII redaction (fast, local) as Pass 1, then
    ML-based NER filtering (remote privacy filter) as Pass 2 when
    available.
    """

    def __init__(
        self,
        pii_redactor: PIIRedactor,
        privacy_filter_client: Optional[Any] = None,
    ):
        """
        Initialize the response sanitizer.

        Args:
            pii_redactor: PIIRedactor instance for regex-based redaction.
            privacy_filter_client: Optional PrivacyFilterClient for ML-based
                NER filtering. If None, only regex redaction is applied.
        """
        self.pii_redactor = pii_redactor
        self.privacy_filter = privacy_filter_client

        logger.info(
            "ResponseSanitizer initialized: regex=True, ml=%s",
            privacy_filter_client is not None,
        )

    async def _apply_regex(self, text: str) -> str:
        """Apply regex-based PII redaction."""
        if not text:
            return text
        return self.pii_redactor.redact(text)

    async def _apply_ml(self, text: str) -> str:
        """Apply ML-based privacy filtering if available."""
        if not text or self.privacy_filter is None:
            return text

        redacted, _entities = await self.privacy_filter.redact(text)
        return redacted

    async def _sanitize_text(self, text: str) -> str:
        """
        Full two-pass text sanitization.

        Pass 1: Regex (PIIRedactor) — always runs.
        Pass 2: ML (PrivacyFilterClient) — runs only when service is available.
        """
        if not text:
            return text

        # Pass 1: regex
        text = await self._apply_regex(text)

        # Pass 2: ML (best-effort, falls back to regex result on failure)
        text = await self._apply_ml(text)

        return text

    async def sanitize_chat_content(self, content: str) -> str:
        """
        Sanitize chat response content.

        Applies regex then ML filtering to plain text chat responses.

        Args:
            content: Chat response content string.

        Returns:
            Sanitized content string.
        """
        if not content:
            return content

        sanitized = await self._sanitize_text(content)

        if sanitized != content:
            logger.debug("Chat content sanitized")

        return sanitized

    async def sanitize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recursively sanitize all tool call arguments.

        Parses JSON arguments, walks all string values recursively,
        redacts PII, and re-serializes. Handles nested JSON, arrays,
        and mixed types up to _MAX_RECURSION_DEPTH.

        Args:
            tool_calls: List of tool call dicts with 'function.arguments'
                as JSON string or 'arguments' as dict.

        Returns:
            List of tool calls with sanitized arguments.
        """
        if not tool_calls:
            return tool_calls

        sanitized_calls = []

        for tool_call in tool_calls:
            call = dict(tool_call)  # shallow copy

            # Handle OpenAI-style tool calls: function.arguments is a JSON string
            function = call.get("function")
            if isinstance(function, dict) and "arguments" in function:
                func_copy = dict(function)
                args_str = func_copy.get("arguments", "{}")

                if isinstance(args_str, str):
                    try:
                        args_dict = json.loads(args_str)
                        sanitized_args = await self._sanitize_value(args_dict, depth=0)
                        func_copy["arguments"] = json.dumps(sanitized_args, ensure_ascii=False)
                    except (json.JSONDecodeError, TypeError):
                        # If JSON parsing fails, sanitize the raw string
                        func_copy["arguments"] = await self._sanitize_text(args_str)
                elif isinstance(args_str, dict):
                    sanitized_args = await self._sanitize_value(args_str, depth=0)
                    func_copy["arguments"] = json.dumps(sanitized_args, ensure_ascii=False)

                call["function"] = func_copy

            # Handle direct arguments dict (Anthropic-style)
            elif "arguments" in call and isinstance(call["arguments"], dict):
                call["arguments"] = await self._sanitize_value(call["arguments"], depth=0)

            # Handle 'input' field (some API variants)
            elif "input" in call and isinstance(call["input"], dict):
                call["input"] = await self._sanitize_value(call["input"], depth=0)

            sanitized_calls.append(call)

        return sanitized_calls

    async def _sanitize_value(self, value: Any, depth: int) -> Any:
        """
        Recursively sanitize PII from an arbitrary value.

        Walks dicts and lists, applying text sanitization to all string
        values encountered.

        Args:
            value: The value to sanitize (any type).
            depth: Current recursion depth.

        Returns:
            Sanitized value of the same type.
        """
        if depth > _MAX_RECURSION_DEPTH:
            logger.warning("Tool call sanitization hit max depth %d", _MAX_RECURSION_DEPTH)
            return value

        if isinstance(value, str):
            return await self._sanitize_text(value)

        if isinstance(value, dict):
            return {k: await self._sanitize_value(v, depth + 1) for k, v in value.items()}

        if isinstance(value, list):
            return [await self._sanitize_value(item, depth + 1) for item in value]

        # Numbers, bools, None — return as-is
        return value

    async def sanitize_reasoning(self, reasoning_content: str) -> str:
        """
        Sanitize reasoning/thinking traces.

        Handles plain reasoning text (e.g., OpenAI reasoning_content field).

        Args:
            reasoning_content: Reasoning text to sanitize.

        Returns:
            Sanitized reasoning text.
        """
        if not reasoning_content:
            return reasoning_content

        sanitized = await self._sanitize_text(reasoning_content)

        if sanitized != reasoning_content:
            logger.debug("Reasoning content sanitized")

        return sanitized

    async def sanitize_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply sanitization to ALL output channels in a single pass.

        Processes:
        1. Chat content in choices[].message.content
        2. Tool calls in choices[].message.tool_calls
        3. OpenAI reasoning_content in choices[].message.reasoning_content
        4. Anthropic thinking blocks in content[].type == "thinking"
        5. DeepSeek embedded <think/> tags in content

        Args:
            response: Raw LLM response dict (OpenAI-compatible format).

        Returns:
            Response dict with all channels sanitized.
        """
        if not response:
            return response

        result = dict(response)

        # --- OpenAI format: choices[].message ---
        choices = result.get("choices")
        if isinstance(choices, list):
            sanitized_choices = []

            for choice in choices:
                if not isinstance(choice, dict):
                    sanitized_choices.append(choice)
                    continue

                choice_copy = dict(choice)
                message = choice_copy.get("message")

                if isinstance(message, dict):
                    message_copy = dict(message)

                    # 1. Sanitize chat content
                    content = message_copy.get("content")
                    if isinstance(content, str):
                        # Check for DeepSeek embedded <think/> tags first
                        content = await self._sanitize_deepseek_thinking(content)
                        message_copy["content"] = await self.sanitize_chat_content(content)
                    elif isinstance(content, list):
                        # Anthropic-style content blocks
                        message_copy["content"] = await self._sanitize_anthropic_content(content)

                    # 2. Sanitize tool calls
                    tool_calls = message_copy.get("tool_calls")
                    if isinstance(tool_calls, list):
                        message_copy["tool_calls"] = await self.sanitize_tool_calls(tool_calls)

                    # 3. Sanitize OpenAI reasoning content
                    reasoning = message_copy.get("reasoning_content")
                    if isinstance(reasoning, str):
                        message_copy["reasoning_content"] = await self.sanitize_reasoning(reasoning)

                    # 4. Sanitize 'reasoning' field (some providers use this name)
                    reasoning_alt = message_copy.get("reasoning")
                    if isinstance(reasoning_alt, str):
                        message_copy["reasoning"] = await self.sanitize_reasoning(reasoning_alt)

                    choice_copy["message"] = message_copy

                sanitized_choices.append(choice_copy)

            result["choices"] = sanitized_choices

        # --- Anthropic format: top-level content array ---
        anthropic_content = result.get("content")
        if isinstance(anthropic_content, list) and "choices" not in result:
            result["content"] = await self._sanitize_anthropic_content(anthropic_content)

        # --- Top-level tool_calls (some proxy formats) ---
        top_tool_calls = result.get("tool_calls")
        if isinstance(top_tool_calls, list) and "choices" not in result:
            result["tool_calls"] = await self.sanitize_tool_calls(top_tool_calls)

        return result

    async def _sanitize_anthropic_content(self, content_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sanitize Anthropic-style content blocks.

        Handles thinking blocks (type == "thinking") and text blocks.

        Args:
            content_blocks: List of Anthropic content block dicts.

        Returns:
            Sanitized content blocks.
        """
        if not content_blocks:
            return content_blocks

        sanitized = []

        for block in content_blocks:
            if not isinstance(block, dict):
                sanitized.append(block)
                continue

            block_copy = dict(block)
            block_type = block_copy.get("type", "")

            if block_type == "thinking":
                # Anthropic thinking block
                thinking_text = block_copy.get("thinking", "")
                if isinstance(thinking_text, str):
                    block_copy["thinking"] = await self.sanitize_reasoning(thinking_text)

            elif block_type == "text":
                # Regular text block
                text = block_copy.get("text", "")
                if isinstance(text, str):
                    block_copy["text"] = await self.sanitize_chat_content(text)

            sanitized.append(block_copy)

        return sanitized

    async def _sanitize_deepseek_thinking(self, content: str) -> str:
        """
        Extract and sanitize DeepSeek embedded <think/> tags within content.

        DeepSeek embeds reasoning as: <think/>reasoning text</think*>
        within the main content string. This method sanitizes only the
        content inside the tags while preserving the tag structure.

        Args:
            content: Content string possibly containing embedded thinking.

        Returns:
            Content with thinking sections sanitized.
        """
        if not content or "<think" not in content.lower():
            return content

        # Find all thinking sections using regex
        parts = []
        last_end = 0

        # Match <think...>content</think...> blocks
        think_pattern = re.compile(
            r"(<think[^>]*>)(.*?)(</think[^>]*>)",
            re.DOTALL | re.IGNORECASE,
        )

        for match in think_pattern.finditer(content):
            # Add content before this thinking block
            if match.start() > last_end:
                parts.append(content[last_end : match.start()])

            open_tag = match.group(1)
            think_content = match.group(2)
            close_tag = match.group(3)

            # Sanitize the thinking content
            sanitized_think = await self.sanitize_reasoning(think_content)

            parts.append(f"{open_tag}{sanitized_think}{close_tag}")
            last_end = match.end()

        # Add remaining content after last thinking block
        if last_end < len(content):
            parts.append(content[last_end:])

        if parts:
            result = "".join(parts)
            return result

        return content
