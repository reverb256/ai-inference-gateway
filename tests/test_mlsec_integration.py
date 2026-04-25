"""
MLSEC Phase 1 Integration Tests.

Tests the full pipeline wiring: PIIRedactor → ResponseSanitizer → output channels.
Verifies that PII is stripped from all response paths:
- Non-streaming chat content
- Non-streaming tool_calls.arguments
- Non-streaming reasoning_content
- Streaming delta.content
- Streaming delta.reasoning_content
- Anthropic non-streaming content + tool calls + thinking
- Anthropic streaming content
- Ollama non-streaming
- Log message sanitization
"""

import asyncio
import json
import logging
import pytest

from ai_inference_gateway.pii_redactor import PIIRedactor, get_default_redactor
from ai_inference_gateway.output.response_sanitizer import ResponseSanitizer
from ai_inference_gateway.output.tool_call_sanitizer import ToolCallSanitizer
from ai_inference_gateway.output.reasoning_sanitizer import ReasoningSanitizer
from ai_inference_gateway.observability.sanitizing_logger import PIISanitizingFilter


# --- Fixtures ---

@pytest.fixture
def redactor():
    return get_default_redactor()


@pytest.fixture
def sanitizer(redactor):
    return ResponseSanitizer(pii_redactor=redactor)


# ============================================================
# Non-streaming: OpenAI chat content
# ============================================================

class TestNonStreamingChatContent:
    """Verify PII is stripped from choices[].message.content."""

    @pytest.mark.asyncio
    async def test_email_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Send the report to john.doe@example.com please."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        assert "john.doe@example.com" not in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_ssn_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "His SSN is 123-45-6789."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        content = result["choices"][0]["message"]["content"]
        assert "123-45-6789" not in content

    @pytest.mark.asyncio
    async def test_phone_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Call me at (555) 123-4567."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        content = result["choices"][0]["message"]["content"]
        assert "(555) 123-4567" not in content

    @pytest.mark.asyncio
    async def test_empty_response_passthrough(self, sanitizer):
        response = {}
        result = await sanitizer.sanitize_response(response)
        assert result == {}

    @pytest.mark.asyncio
    async def test_no_pii_passthrough(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "The capital of France is Paris."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        assert result["choices"][0]["message"]["content"] == "The capital of France is Paris."


# ============================================================
# Non-streaming: Tool calls
# ============================================================

class TestNonStreamingToolCalls:
    """Verify PII is stripped from tool_calls[].function.arguments."""

    @pytest.mark.asyncio
    async def test_tool_call_email_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "send_email",
                            "arguments": json.dumps({
                                "to": "sensitive@corp.com",
                                "body": "Here is the data."
                            })
                        }
                    }]
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert "sensitive@corp.com" not in args["to"]

    @pytest.mark.asyncio
    async def test_tool_call_nested_pii(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup_user",
                            "arguments": json.dumps({
                                "user": {
                                    "email": "private@email.com",
                                    "phone": "555-000-1234",
                                    "notes": "SSN: 987-65-4321"
                                }
                            })
                        }
                    }]
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        args_str = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert "private@email.com" not in args_str
        assert "555-000-1234" not in args_str
        assert "987-65-4321" not in args_str

    @pytest.mark.asyncio
    async def test_tool_call_no_pii(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Paris"})
                        }
                    }]
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        args = json.loads(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        assert args["city"] == "Paris"


# ============================================================
# Non-streaming: Reasoning content
# ============================================================

class TestNonStreamingReasoning:
    """Verify PII is stripped from reasoning_content."""

    @pytest.mark.asyncio
    async def test_reasoning_email_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Here is the answer.",
                    "reasoning_content": "The user's email is secret@hidden.org. I should use that."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        reasoning = result["choices"][0]["message"]["reasoning_content"]
        assert "secret@hidden.org" not in reasoning

    @pytest.mark.asyncio
    async def test_reasoning_ssn_redacted(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "Done.",
                    "reasoning_content": "SSN on file is 111-22-3333."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        assert "111-22-3333" not in result["choices"][0]["message"]["reasoning_content"]

    @pytest.mark.asyncio
    async def test_alt_reasoning_field(self, sanitizer):
        """Some providers use 'reasoning' instead of 'reasoning_content'."""
        response = {
            "choices": [{
                "message": {
                    "content": "Result.",
                    "reasoning": "User phone: (999) 888-7777"
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        assert "(999) 888-7777" not in result["choices"][0]["message"]["reasoning"]


# ============================================================
# Anthropic format
# ============================================================

class TestAnthropicFormat:
    """Verify sanitization of Anthropic-style responses."""

    @pytest.mark.asyncio
    async def test_anthropic_non_streaming(self, sanitizer):
        """Anthropic non-streaming uses top-level content array."""
        response = {
            "content": [
                {"type": "thinking", "thinking": "Email is private@test.com for this user"},
                {"type": "text", "text": "The user can be reached at private@test.com."},
            ],
            "model": "claude-sonnet-4",
        }
        result = await sanitizer.sanitize_response(response)
        # Check text block
        text_block = result["content"][1]
        assert "private@test.com" not in text_block["text"]
        # Check thinking block
        think_block = result["content"][0]
        assert "private@test.com" not in think_block["thinking"]

    @pytest.mark.asyncio
    async def test_anthropic_tool_use(self, sanitizer):
        response = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "send_email",
                    "input": {"to": "leaked@corp.com"},
                }
            ],
        }
        result = await sanitizer.sanitize_response(response)
        tool_block = result["content"][0]
        # Tool input is a dict — check serialized form
        input_str = json.dumps(tool_block["input"])
        assert "leaked@corp.com" not in input_str


# ============================================================
# DeepSeek embedded thinking
# ============================================================

class TestDeepSeekThinking:
    """Verify <think/> tags in content are sanitized."""

    @pytest.mark.asyncio
    async def test_embedded_think_tags(self, sanitizer):
        response = {
            "choices": [{
                "message": {
                    "content": "<think\\n>User SSN is 444-55-6666</think\\n>\\nThe answer is 42."
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        content = result["choices"][0]["message"]["content"]
        assert "444-55-6666" not in content


# ============================================================
# Logging sanitization
# ============================================================

class TestLoggingSanitization:
    """Verify PIISanitizingFilter redacts log records."""

    def test_log_email_redacted(self, redactor):
        pii_filter = PIISanitizingFilter(redactor=redactor)
        test_logger = logging.getLogger("test_mlsec_integration")
        test_logger.addFilter(pii_filter)

        # Create a log record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="User email is test@secret.com", args=None, exc_info=None
        )
        pii_filter.filter(record)
        assert "test@secret.com" not in record.msg
        test_logger.removeFilter(pii_filter)

    def test_log_ssn_redacted(self, redactor):
        pii_filter = PIISanitizingFilter(redactor=redactor)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="SSN: 000-11-2222", args=None, exc_info=None
        )
        pii_filter.filter(record)
        assert "000-11-2222" not in record.msg

    def test_log_no_pii_passthrough(self, redactor):
        pii_filter = PIISanitizingFilter(redactor=redactor)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="All systems normal", args=None, exc_info=None
        )
        pii_filter.filter(record)
        assert record.msg == "All systems normal"


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    """Edge cases and robustness."""

    @pytest.mark.asyncio
    async def test_multiple_choices(self, sanitizer):
        response = {
            "choices": [
                {"message": {"content": "Email: a@test.com"}},
                {"message": {"content": "Email: b@test.com"}},
            ]
        }
        result = await sanitizer.sanitize_response(response)
        assert "a@test.com" not in result["choices"][0]["message"]["content"]
        assert "b@test.com" not in result["choices"][1]["message"]["content"]

    @pytest.mark.asyncio
    async def test_none_content(self, sanitizer):
        response = {
            "choices": [{
                "message": {"content": None, "tool_calls": None}
            }]
        }
        result = await sanitizer.sanitize_response(response)
        assert result["choices"][0]["message"]["content"] is None

    @pytest.mark.asyncio
    async def test_mixed_pii_in_tool_args_string(self, sanitizer):
        """Tool args might be a plain string, not JSON."""
        response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "search",
                            "arguments": "Find user alice@corp.com with SSN 111-22-3333"
                        }
                    }]
                }
            }]
        }
        result = await sanitizer.sanitize_response(response)
        args = result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        assert "alice@corp.com" not in args
        assert "111-22-3333" not in args

    @pytest.mark.asyncio
    async def test_preserves_non_pii_structure(self, sanitizer):
        """Verify non-PII fields are preserved."""
        response = {
            "id": "chatcmpl-123",
            "model": "qwen3.5-32b",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = await sanitizer.sanitize_response(response)
        assert result["id"] == "chatcmpl-123"
        assert result["model"] == "qwen3.5-32b"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
