"""
Tests for Phase 1 ML Security sanitizers.

Covers:
1. PIISanitizingFilter (observability/sanitizing_logger.py)
2. OTelSpanSanitizer (observability/otel_sanitizer.py)
3. ToolCallSanitizer (output/tool_call_sanitizer.py) -- conditional
4. ReasoningSanitizer (output/reasoning_sanitizer.py) -- conditional
5. ResponseSanitizer (output/response_sanitizer.py) -- conditional
"""

import json
import logging
import threading

import pytest

from ai_inference_gateway.observability.sanitizing_logger import PIISanitizingFilter
from ai_inference_gateway.observability.otel_sanitizer import OTelSpanSanitizer
from ai_inference_gateway.pii_redactor import PIIRedactor

# ---------------------------------------------------------------------------
# Conditional imports for sanitizers created by other agents
# ---------------------------------------------------------------------------

try:
    from ai_inference_gateway.output.tool_call_sanitizer import ToolCallSanitizer

    HAS_TOOL_SANITIZER = True
except ImportError:
    HAS_TOOL_SANITIZER = False

try:
    from ai_inference_gateway.output.reasoning_sanitizer import ReasoningSanitizer

    HAS_REASONING_SANITIZER = True
except ImportError:
    HAS_REASONING_SANITIZER = False

try:
    from ai_inference_gateway.output.response_sanitizer import ResponseSanitizer

    HAS_RESPONSE_SANITIZER = True
except ImportError:
    HAS_RESPONSE_SANITIZER = False


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redactor():
    """Fresh PIIRedactor instance."""
    return PIIRedactor()


@pytest.fixture
def log_filter(redactor):
    """PIISanitizingFilter wired to a fresh redactor."""
    return PIISanitizingFilter(redactor)


@pytest.fixture
def otel_sanitizer(redactor):
    """OTelSpanSanitizer wired to a fresh redactor."""
    return OTelSpanSanitizer(redactor)


def _make_record(msg, *args, name="test", level=logging.INFO):
    """Create a LogRecord without going through the logging machinery."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=msg,
        args=args or None,
        exc_info=None,
    )
    return record


# ============================================================================
# 1. PIISanitizingFilter
# ============================================================================


class TestPIISanitizingFilter:
    """Tests for the PII sanitizing logging filter."""

    # --- Email ---

    def test_filter_redacts_email_in_message(self, log_filter):
        """Email addresses in log messages are replaced with [EMAIL]."""
        record = _make_record("User user@example.com logged in")
        log_filter.filter(record)

        assert "[EMAIL]" in record.msg
        assert "user@example.com" not in record.msg

    def test_filter_redacts_email_in_tuple_args(self, log_filter):
        """Email addresses in format-string tuple args are redacted."""
        record = _make_record("User %s logged in", "admin@corp.com")
        log_filter.filter(record)

        assert record.args is not None
        assert "[EMAIL]" in record.args[0]
        assert "admin@corp.com" not in record.args[0]

    def test_filter_redacts_email_in_dict_args(self, log_filter):
        """Email addresses in dict-style format args are redacted."""
        record = _make_record("User %(email)s logged in", {"email": "bob@test.org"})
        log_filter.filter(record)

        assert "[EMAIL]" in record.args["email"]
        assert "bob@test.org" not in record.args["email"]

    # --- SSN ---

    def test_filter_redacts_ssn(self, log_filter):
        """SSN patterns in log messages are replaced with [SSN]."""
        record = _make_record("SSN on file: 123-45-6789")
        log_filter.filter(record)

        assert "[SSN]" in record.msg
        assert "123-45-6789" not in record.msg

    # --- Phone ---

    def test_filter_redacts_phone(self, log_filter):
        """Phone numbers in log messages are masked."""
        record = _make_record("Call back at 555-123-4567 please")
        log_filter.filter(record)

        # Phone uses MASK mode -- digits are partially hidden
        assert "555-123-4567" not in record.msg

    # --- Credit Card ---

    def test_filter_redacts_credit_card(self, log_filter):
        """Credit card numbers in log messages are replaced."""
        record = _make_record("Payment with 4111-1111-1111-1111")
        log_filter.filter(record)

        assert "4111-1111-1111-1111" not in record.msg
        # PIIRedactor masks digits in-place (e.g. 41******1)
        assert record.msg != "Payment with 4111-1111-1111-1111"

    # --- Short clean messages (performance shortcut) ---

    def test_short_clean_message_passes_through(self, log_filter):
        """Messages < 20 chars with no digits and no @ skip sanitization."""
        msg = "Hello world"
        record = _make_record(msg)
        log_filter.filter(record)

        assert record.msg == msg

    def test_short_message_with_digit_is_sanitized(self, log_filter):
        """Short messages containing digits still get sanitized."""
        # "SSN: 123-45-6789" is 15 chars but has digits -> passes the check
        record = _make_record("SSN: 123-45-6789")
        log_filter.filter(record)

        assert "[SSN]" in record.msg

    def test_short_message_with_at_sign_is_sanitized(self, log_filter):
        """Short messages containing @ still get sanitized."""
        record = _make_record("a@b.com is short")
        log_filter.filter(record)

        assert "[EMAIL]" in record.msg
        assert "a@b.com" not in record.msg

    # --- Non-string args pass through ---

    def test_non_string_args_pass_through(self, log_filter):
        """Non-string format args (ints, floats) pass through unchanged."""
        record = _make_record("Count: %d, Rate: %.2f", 42, 3.14)
        log_filter.filter(record)

        assert record.args == (42, 3.14)

    # --- Always returns True ---

    def test_filter_always_returns_true(self, log_filter):
        """Filter never suppresses log records."""
        record = _make_record("anything")
        result = log_filter.filter(record)

        assert result is True

    # --- None msg ---

    def test_filter_handles_non_string_msg(self, log_filter):
        """Non-string msg values don't crash the filter."""
        record = _make_record("template")
        record.msg = 12345  # not a string
        log_filter.filter(record)

        assert record.msg == 12345

    # --- Stats ---

    def test_stats_initial(self, log_filter):
        """Stats start at zero."""
        stats = log_filter.get_stats()

        assert stats["total_messages"] == 0
        assert stats["messages_sanitized"] == 0
        assert stats["redactions_performed"] == 0

    def test_stats_increment_on_clean_message(self, log_filter):
        """total_messages increments even for clean messages."""
        record = _make_record("Hello world")
        log_filter.filter(record)

        stats = log_filter.get_stats()
        assert stats["total_messages"] == 1
        assert stats["messages_sanitized"] == 0

    def test_stats_increment_on_pii_message(self, log_filter):
        """Both total_messages and messages_sanitized increment for PII."""
        record = _make_record("Email: user@example.com")
        log_filter.filter(record)

        stats = log_filter.get_stats()
        assert stats["total_messages"] == 1
        assert stats["messages_sanitized"] == 1

    def test_stats_accumulate_across_calls(self, log_filter):
        """Stats accumulate correctly across multiple filter calls."""
        log_filter.filter(_make_record("clean message here"))
        log_filter.filter(_make_record("Email: a@b.com"))
        log_filter.filter(_make_record("also clean"))

        stats = log_filter.get_stats()
        assert stats["total_messages"] == 3
        assert stats["messages_sanitized"] == 1

    # --- Thread safety ---

    def test_filter_is_thread_safe(self, redactor):
        """Concurrent filter calls don't corrupt stats counters."""
        filt = PIISanitizingFilter(redactor)
        errors = []

        def worker(msg, count):
            try:
                for _ in range(count):
                    filt.filter(_make_record(msg))
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=("Email: a@b.com", 50)),
            threading.Thread(target=worker, args=("clean message", 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = filt.get_stats()
        assert stats["total_messages"] == 100


# ============================================================================
# 2. OTelSpanSanitizer
# ============================================================================


class TestOTelSpanSanitizer:
    """Tests for the OTel span attribute sanitizer."""

    def test_sanitize_email_in_content_key(self, otel_sanitizer):
        """String values in 'content' key are redacted."""
        attrs = {"content": "Email user@example.com for details"}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert "[EMAIL]" in result["content"]
        assert "user@example.com" not in result["content"]

    def test_sanitize_email_in_llm_prefixed_key(self, otel_sanitizer):
        """String values in 'llm.*' keys are redacted."""
        attrs = {"llm.prompt": "Send to admin@corp.com"}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert "[EMAIL]" in result["llm.prompt"]
        assert "admin@corp.com" not in result["llm.prompt"]

    def test_sanitize_ssn_in_input_key(self, otel_sanitizer):
        """SSN in 'input' key is redacted."""
        attrs = {"input": "SSN: 123-45-6789"}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert "[SSN]" in result["input"]
        assert "123-45-6789" not in result["input"]

    def test_sanitize_credit_card_in_response_key(self, otel_sanitizer):
        """Credit card in 'response' key is redacted."""
        attrs = {"response": "Card 4111-1111-1111-1111 charged"}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert "4111-1111-1111-1111" not in result["response"]
        # PIIRedactor masks digits in-place (e.g. 41******1)
        assert result["response"] != "Card 4111-1111-1111-1111 charged"

    def test_non_matching_keys_pass_through(self, otel_sanitizer):
        """Keys not in the PII set are never touched."""
        attrs = {
            "request_count": 42,
            "latency_ms": 123.5,
            "model_name": "gpt-4",
        }
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["request_count"] == 42
        assert result["latency_ms"] == 123.5
        assert result["model_name"] == "gpt-4"

    def test_numeric_values_pass_through(self, otel_sanitizer):
        """Numeric values in PII keys pass through unchanged."""
        attrs = {"llm.token_count": 150}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["llm.token_count"] == 150

    def test_bool_values_pass_through(self, otel_sanitizer):
        """Bool values in PII keys pass through unchanged."""
        attrs = {"llm.stream": True}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["llm.stream"] is True

    def test_none_values_pass_through(self, otel_sanitizer):
        """None values in PII keys pass through unchanged."""
        attrs = {"content": None}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["content"] is None

    def test_does_not_mutate_input(self, otel_sanitizer):
        """Original attributes dict is not mutated."""
        attrs = {"content": "Email user@example.com"}
        original_content = attrs["content"]

        otel_sanitizer.sanitize_attributes(attrs)

        assert attrs["content"] == original_content

    def test_mixed_attributes_selective_sanitization(self, otel_sanitizer):
        """PII keys get sanitized, metric keys pass through, in one call."""
        attrs = {
            "content": "Contact admin@corp.com",
            "llm.prompt": "SSN 123-45-6789",
            "request_count": 100,
            "latency_ms": 45.2,
            "model": "claude-3",
            "reasoning": "User email bob@test.org found",
            "service.version": "1.0.0",
        }
        result = otel_sanitizer.sanitize_attributes(attrs)

        # PII keys sanitized
        assert "[EMAIL]" in result["content"]
        assert "[SSN]" in result["llm.prompt"]
        assert "[EMAIL]" in result["reasoning"]

        # Metric/non-matching keys untouched
        assert result["request_count"] == 100
        assert result["latency_ms"] == 45.2
        assert result["model"] == "claude-3"
        assert result["service.version"] == "1.0.0"

    def test_all_pii_exact_keys(self, otel_sanitizer):
        """Every key in the exact-match set triggers sanitization."""
        pii_value = "Email user@example.com"
        keys = [
            "input", "output", "content", "message",
            "prompt", "response", "reasoning", "tool", "arguments",
        ]

        for key in keys:
            attrs = {key: pii_value}
            result = otel_sanitizer.sanitize_attributes(attrs)
            assert "[EMAIL]" in result[key], f"Key '{key}' was not sanitized"

    def test_empty_attributes(self, otel_sanitizer):
        """Empty dict returns empty dict."""
        result = otel_sanitizer.sanitize_attributes({})
        assert result == {}

    def test_clean_string_in_pii_key(self, otel_sanitizer):
        """Clean string in a PII key passes through unchanged."""
        attrs = {"content": "The weather is nice today"}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["content"] == "The weather is nice today"

    def test_list_value_in_pii_key_passes_through(self, otel_sanitizer):
        """Non-string values (list) in PII keys pass through."""
        attrs = {"content": [1, 2, 3]}
        result = otel_sanitizer.sanitize_attributes(attrs)

        assert result["content"] == [1, 2, 3]


# ============================================================================
# 3. ToolCallSanitizer (conditional)
# ============================================================================


@pytest.mark.skipif(not HAS_TOOL_SANITIZER, reason="ToolCallSanitizer not available")
class TestToolCallSanitizer:
    """Tests for the tool call argument sanitizer."""

    @pytest.fixture
    def tool_sanitizer(self, redactor):
        return ToolCallSanitizer(redactor)

    def test_sanitize_openai_tool_call_email(self, tool_sanitizer):
        """OpenAI-format tool call arguments with email are sanitized."""
        tool_calls = [
            {
                "function": {
                    "name": "send_email",
                    "arguments": json.dumps({
                        "to": "user@example.com",
                        "body": "Hello",
                    }),
                }
            }
        ]
        result = tool_sanitizer.sanitize_tool_calls(tool_calls)

        args = json.loads(result[0]["function"]["arguments"])
        assert "[EMAIL]" in args["to"]
        assert "user@example.com" not in args["to"]

    def test_sanitize_anthropic_input_ssn(self, tool_sanitizer):
        """Anthropic-format tool call input with SSN is sanitized."""
        tool_calls = [
            {
                "name": "lookup",
                "input": {"ssn": "123-45-6789"},
            }
        ]
        result = tool_sanitizer.sanitize_tool_calls(tool_calls)

        assert "[SSN]" in result[0]["input"]["ssn"]
        assert "123-45-6789" not in result[0]["input"]["ssn"]

    def test_does_not_mutate_original(self, tool_sanitizer):
        """Original tool call list is not mutated."""
        original_args = json.dumps({"email": "a@b.com"})
        tool_calls = [{"function": {"arguments": original_args}}]

        tool_sanitizer.sanitize_tool_calls(tool_calls)

        assert tool_calls[0]["function"]["arguments"] == original_args

    def test_empty_tool_calls(self, tool_sanitizer):
        """Empty list returns empty list."""
        result = tool_sanitizer.sanitize_tool_calls([])
        assert result == []

    def test_none_tool_calls(self, tool_sanitizer):
        """None input returns None."""
        result = tool_sanitizer.sanitize_tool_calls(None)
        assert result is None


# ============================================================================
# 4. ReasoningSanitizer (conditional)
# ============================================================================


@pytest.mark.skipif(not HAS_REASONING_SANITIZER, reason="ReasoningSanitizer not available")
class TestReasoningSanitizer:
    """Tests for the reasoning/thinking trace sanitizer."""

    @pytest.fixture
    def reasoning_sanitizer(self, redactor):
        return ReasoningSanitizer(redactor)

    def test_sanitize_openai_reasoning_content(self, reasoning_sanitizer):
        """OpenAI reasoning_content with email is sanitized."""
        response = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "User email is admin@corp.com",
                    }
                }
            ]
        }
        result = reasoning_sanitizer.sanitize_reasoning(response)

        rc = result["choices"][0]["message"]["reasoning_content"]
        assert "[EMAIL]" in rc
        assert "admin@corp.com" not in rc

    def test_sanitize_anthropic_thinking_block(self, reasoning_sanitizer):
        """Anthropic thinking block with SSN is sanitized."""
        response = {
            "content": [
                {"type": "thinking", "thinking": "SSN 123-45-6789 found"},
                {"type": "text", "text": "Here is the answer"},
            ]
        }
        result = reasoning_sanitizer.sanitize_reasoning(response)

        assert "[SSN]" in result["content"][0]["thinking"]
        assert "123-45-6789" not in result["content"][0]["thinking"]
        # Non-thinking blocks untouched
        assert result["content"][1]["text"] == "Here is the answer"

    def test_anthropic_no_thinking_blocks(self, reasoning_sanitizer):
        """Content without thinking blocks passes through unchanged."""
        response = {
            "content": [
                {"type": "text", "text": "No PII here"},
                {"type": "text", "text": "Also clean"},
            ]
        }
        result = reasoning_sanitizer.sanitize_reasoning(response)
        assert result["content"][0]["text"] == "No PII here"
        assert result["content"][1]["text"] == "Also clean"