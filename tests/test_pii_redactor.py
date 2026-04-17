"""
Tests for PII Redaction.

Tests the PIIRedactor which detects and redacts personally
identifiable information from text.
"""

from ai_inference_gateway.pii_redactor import (
    PIIRedactor,
    RedactionMode,
)


# ============================================================================
# Test Email Redaction
# ============================================================================


class TestEmailRedaction:
    """Tests for email address detection and redaction."""

    def test_redact_email_basic(self):
        """Test basic email redaction."""
        redactor = PIIRedactor()
        text = "Contact me at user@example.com for help."

        result = redactor.redact(text)

        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_redact_multiple_emails(self):
        """Test redacting multiple emails."""
        redactor = PIIRedactor()
        text = "Email john@example.com or jane@company.co.uk"

        result = redactor.redact(text)

        assert result.count("[EMAIL]") == 2

    def test_redact_email_with_subdomains(self):
        """Test email with subdomains."""
        redactor = PIIRedactor()
        text = "user@mail.example.com"

        result = redactor.redact(text)

        assert "[EMAIL]" in result

    def test_no_false_positives(self):
        """Test that email-like patterns in code are not false positives."""
        redactor = PIIRedactor()
        text = "Use example.com without the @ symbol"

        result = redactor.redact(text)

        # Should not redact if it's just a domain
        assert "example.com" in result


# ============================================================================
# Test Phone Number Redaction
# ============================================================================


class TestPhoneRedaction:
    """Tests for phone number detection and redaction."""

    def test_redact_us_phone_format(self):
        """Test US phone number formats."""
        redactor = PIIRedactor()
        text = "Call me at 555-123-4567"

        result = redactor.redact(text)

        # Mask mode shows partial
        assert "555***567" in result

    def test_redact_phone_with_parentheses(self):
        """Test phone with area code in parentheses."""
        redactor = PIIRedactor()
        text = "Call (555) 123-4567"

        result = redactor.redact(text)

        assert "5***567" in result

    def test_redact_international_phone(self):
        """Test international phone format."""
        redactor = PIIRedactor()
        text = "Call +1 555-123-4567"

        result = redactor.redact(text)

        # Should be masked
        assert "5***567" in result


# ============================================================================
# Test SSN Redaction
# ============================================================================


class TestSSNRedaction:
    """Tests for Social Security Number redaction."""

    def test_redact_ssn_with_dashes(self):
        """Test SSN format with dashes."""
        redactor = PIIRedactor()
        text = "My SSN is 123-45-6789"

        result = redactor.redact(text)

        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_redact_ssn_with_spaces(self):
        """Test SSN format with spaces."""
        redactor = PIIRedactor()
        text = "My SSN is 123 45 6789"

        result = redactor.redact(text)

        assert "[SSN]" in result

    def test_redact_ssn_without_separators(self):
        """Test SSN without separators."""
        redactor = PIIRedactor()
        text = "SSN: 123456789"

        result = redactor.redact(text)

        assert "[SSN]" in result


# ============================================================================
# Test Credit Card Redaction
# ============================================================================


class TestCreditCardRedaction:
    """Tests for credit card number redaction."""

    def test_redact_credit_card_with_dashes(self):
        """Test credit card with dashes."""
        redactor = PIIRedactor()
        text = "Card: 4111-1111-1111-1111"

        result = redactor.redact(text)

        assert "[CREDIT_CARD]" in result

    def test_redact_credit_card_without_spaces(self):
        """Test credit card without separators."""
        redactor = PIIRedactor()
        text = "Card: 4111111111111111"

        result = redactor.redact(text)

        assert "[CREDIT_CARD]" in result


# ============================================================================
# Test IP Address Redaction
# ============================================================================


class TestIPRedaction:
    """Tests for IP address redaction."""

    def test_redact_ipv4_address(self):
        """Test IPv4 address redaction."""
        redactor = PIIRedactor()
        text = "Server at 192.168.1.1"

        result = redactor.redact(text)

        assert "[IP_ADDRESS]" in result
        assert "192.168.1.1" not in result

    def test_redact_local_ip(self):
        """Test local IP address redaction."""
        redactor = PIIRedactor()
        text = "Localhost is 127.0.0.1"

        result = redactor.redact(text)

        assert "[IP_ADDRESS]" in result


# ============================================================================
# Test API Key Redaction
# ============================================================================


class TestAPIKeyRedaction:
    """Tests for API key and token redaction."""

    def test_redact_long_alphanumeric(self):
        """Test detection of long alphanumeric strings (API keys)."""
        redactor = PIIRedactor()
        api_key = "abcd1234efgh5678ijkl9012mnop3456"
        text = f"API key: {api_key}"

        result = redactor.redact(text)

        assert "[API_KEY]" in result
        assert api_key not in result

    def test_redact_bearer_token(self):
        """Test Bearer token redaction."""
        redactor = PIIRedactor()
        text = "Authorization: Bearer abcd1234efgh5678"

        result = redactor.redact(text)

        assert "[API_KEY]" in result


# ============================================================================
# Test Password Redaction
# ============================================================================


class TestPasswordRedaction:
    """Tests for password redaction."""

    def test_redact_password_with_equals(self):
        """Test password in key=value format."""
        redactor = PIIRedactor()
        text = 'password="secret123"'

        result = redactor.redact(text)

        assert "[PASSWORD]" in result
        assert "secret123" not in result

    def test_redact_password_with_colon(self):
        """Test password in key: value format."""
        redactor = PIIRedactor()
        text = "password: mypass123"

        result = redactor.redact(text)

        assert "[PASSWORD]" in result


# ============================================================================
# Test Redaction Modes
# ============================================================================


class TestRedactionModes:
    """Tests for different redaction modes."""

    def test_react_mode_placeholder(self):
        """Test redact mode uses placeholders."""
        redactor = PIIRedactor()
        text = "Email: user@example.com"

        result = redactor.redact(text, mode=RedactionMode.REDACT)

        assert "[EMAIL]" in result

    def test_mask_mode_partial(self):
        """Test mask mode shows partial."""
        redactor = PIIRedactor()
        text = "user@example.com"

        result = redactor.redact(text, mode=RedactionMode.MASK)

        # Should show first and last chars
        assert result[0] == "u"
        assert result[-1] != "@"

    def test_remove_mode_completely(self):
        """Test remove mode deletes PII."""
        redactor = PIIRedactor()
        text = "Contact user@example.com today"

        result = redactor.redact(text, mode=RedactionMode.REMOVE)

        assert "user@example.com" not in result
        assert "Contact  today" in result

    def test_hash_mode_obfuscated(self):
        """Test hash mode obfuscates with hash."""
        redactor = PIIRedactor()
        text = "Email: user@example.com"

        result = redactor.redact(text, mode=RedactionMode.HASH)

        # Should contain hash
        assert "[EMAIL:" in result


# ============================================================================
# Test Message Redaction
# ============================================================================


class TestMessageRedaction:
    """Tests for redacting chat messages."""

    def test_redact_user_messages_only(self, sample_pii_text):
        """Test redacting only user messages."""
        redactor = PIIRedactor()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": sample_pii_text},
            {"role": "assistant", "content": "Here's my email: assistant@example.com"},
        ]

        result = redactor.redact_messages(
            messages, redact_user=True, redact_assistant=False
        )

        # User message should be redacted
        assert "[EMAIL]" in result[1]["content"]

        # Assistant message should NOT be redacted
        assert "assistant@example.com" in result[2]["content"]

    def test_redact_all_messages(self, sample_pii_text):
        """Test redacting all messages."""
        redactor = PIIRedactor()
        messages = [
            {"role": "user", "content": sample_pii_text},
            {"role": "assistant", "content": "Email: assistant@example.com"},
        ]

        result = redactor.redact_messages(
            messages, redact_user=True, redact_assistant=True
        )

        # Both should be redacted
        assert result[0]["content"].count("[EMAIL]") >= 1
        assert "[EMAIL]" in result[1]["content"]


# ============================================================================
# Test PII Detection
# ============================================================================


class TestPIIDetection:
    """Tests for PII detection without redaction."""

    def test_detect_email(self):
        """Test email detection."""
        redactor = PIIRedactor()
        text = "Contact user@example.com"

        detections = redactor.detect(text)

        assert "email" in detections
        assert len(detections["email"]) == 1
        assert detections["email"][0]["match"] == "user@example.com"

    def test_detect_multiple_pii_types(self, sample_pii_text):
        """Test detecting multiple PII types."""
        redactor = PIIRedactor()

        detections = redactor.detect(sample_pii_text)

        # Should detect multiple types
        assert len(detections) > 1
        assert "email" in detections
        assert "phone" in detections

    def test_detect_returns_positions(self):
        """Test that detection includes position information."""
        redactor = PIIRedactor()
        text = "Email: user@example.com"

        detections = redactor.detect(text)

        email_match = detections["email"][0]
        assert "start" in email_match
        assert "end" in email_match


# ============================================================================
# Test Pattern Filtering
# ============================================================================


class TestPatternFiltering:
    """Tests for enabling/disabling specific patterns."""

    def test_enable_specific_patterns(self):
        """Test enabling only specific patterns."""
        redactor = PIIRedactor(enabled_patterns=["email", "phone"])
        text = "Email: user@example.com, SSN: 123-45-6789"

        result = redactor.redact(text)

        # Email and phone should be redacted
        assert "[EMAIL]" in result or "5***567" in result

        # SSN should NOT be redacted (not in enabled patterns)
        assert "123-45-6789" in result

    def test_empty_enabled_patterns_allows_none(self):
        """Test that empty enabled patterns redacts nothing."""
        redactor = PIIRedactor(enabled_patterns=[])
        text = "Email: user@example.com, Phone: 555-123-4567"

        result = redactor.redact(text)

        # Nothing should be redacted
        assert "user@example.com" in result
        assert "555-123-4567" in result


# ============================================================================
# Test Complex Scenarios
# ============================================================================


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def test_redact_multiple_pii_types(self, sample_pii_text):
        """Test redacting text with multiple PII types."""
        redactor = PIIRedactor()

        result = redactor.redact(sample_pii_text)

        # Should contain redaction markers
        assert "[EMAIL]" in result
        assert "[PHONE]" in result or "***" in result
        assert "[SSN]" in result
        assert "[CREDIT_CARD]" in result
        assert "[IP_ADDRESS]" in result

    def test_preserve_non_pii_content(self, sample_pii_text):
        """Test that non-PII content is preserved."""
        redactor = PIIRedactor()
        text = f"Hello, {sample_pii_text}, please call me."

        result = redactor.redact(text)

        # Non-PII content should be preserved
        assert "Hello" in result
        assert "please call me" in result

    def test_realistic_scenario(self):
        """Test realistic user message with PII."""
        redactor = PIIRedactor()
        text = """
        Hi, my name is John and I need help with my account.
        You can reach me at john.doe@example.com or 555-123-4567.
        My SSN is 123-45-6789 for verification.
        I'm connecting from IP 192.168.1.1.
        """

        result = redactor.redact(text)

        # All PII should be redacted
        assert "john.doe@example.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result
        assert "192.168.1.1" not in result

        # Non-PII content preserved
        assert "John" in result
        assert "account" in result


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test redacting empty text."""
        redactor = PIIRedactor()

        result = redactor.redact("")

        assert result == ""

    def test_no_pii_in_text(self):
        """Test text without PII."""
        redactor = PIIRedactor()
        text = "Hello, how are you today?"

        result = redactor.redact(text)

        assert result == text

    def test_unicode_handling(self):
        """Test handling of unicode characters."""
        redactor = PIIRedactor()
        text = "Email: tëst@example.com"

        result = redactor.redact(text)

        assert "[EMAIL]" in result

    def test_very_long_text(self):
        """Test redacting very long text."""
        redactor = PIIRedactor()
        text = "Contact me at " + "user@example.com " * 1000

        result = redactor.redact(text)

        # Should handle without error
        assert "[EMAIL]" in result
