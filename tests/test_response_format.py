"""
Tests for response_format transformation (JSON Schema Mode).

Tests the ResponseFormatTransformer which converts OpenAI response_format
requests to LM Studio-compatible system prompts.
"""

from ai_inference_gateway.response_format import (
    ResponseFormatTransformer,
    transform_request,
    validate_response,
    get_transformer,
)


# ============================================================================
# Test JSON Object Mode
# ============================================================================


class TestJSONObjectMode:
    """Tests for json_object response_format mode."""

    def test_transform_json_object_request(self, json_object_request):
        """Test transformation of json_object request."""
        result = transform_request(json_object_request)

        # Check that response_format was removed
        assert "response_format" not in result

        # Check that system prompt was added
        assert result["messages"][0]["role"] == "system"
        assert "valid JSON" in result["messages"][0]["content"]
        assert "no markdown" in result["messages"][0]["content"].lower()

    def test_json_object_system_prompt_content(self):
        """Test that json_object system prompt contains required instructions."""
        request = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "json_object"},
        }

        result = transform_request(request)
        system_prompt = result["messages"][0]["content"]

        assert "ONLY with valid JSON" in system_prompt
        assert "NOT use markdown" in system_prompt
        assert "NOT include any text outside the JSON" in system_prompt

    def test_json_object_with_strict_mode(self):
        """Test json_object mode with strict mode enabled."""
        transformer = ResponseFormatTransformer(strict_mode=True)
        request = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "json_object"},
        }

        result = transformer.transform_request(request)
        system_prompt = result["messages"][0]["content"]

        # Check for additional strict mode requirements
        assert "double quotes" in system_prompt.lower()
        assert "Validate JSON" in system_prompt

    def test_validate_json_object_success(self):
        """Test validation of valid JSON response."""
        response_format = {"type": "json_object"}
        content = '{"name": "John", "age": 30}'

        is_valid, error = validate_response(content, response_format)

        assert is_valid is True
        assert error is None

    def test_validate_json_object_failure(self):
        """Test validation of invalid JSON response."""
        response_format = {"type": "json_object"}
        content = "This is not JSON"

        is_valid, error = validate_response(content, response_format)

        assert is_valid is False
        assert error is not None
        assert "Invalid JSON" in error


# ============================================================================
# Test JSON Schema Mode
# ============================================================================


class TestJSONSchemaMode:
    """Tests for json_schema response_format mode."""

    def test_transform_json_schema_request(self, json_schema_request):
        """Test transformation of json_schema request."""
        result = transform_request(json_schema_request)

        # Check that response_format was removed
        assert "response_format" not in result

        # Check that system prompt was added
        assert result["messages"][0]["role"] == "system"

    def test_json_schema_system_prompt_content(self, json_schema_request):
        """Test that json_schema system prompt contains schema."""
        result = transform_request(json_schema_request)
        system_prompt = result["messages"][0]["content"]

        # Should contain schema information
        assert "user_profile" in system_prompt
        assert "name" in system_prompt
        assert "age" in system_prompt
        assert "email" in system_prompt

    def test_json_schema_strict_mode_instructions(self, json_schema_request):
        """Test json_schema with strict mode."""
        transformer = ResponseFormatTransformer(strict_mode=True)
        result = transformer.transform_request(json_schema_request)
        system_prompt = result["messages"][0]["content"]

        assert "Strict Mode: Yes" in system_prompt
        assert "Field types must match exactly" in system_prompt

    def test_validate_json_schema_success(self, json_schema_request):
        """Test validation of valid JSON Schema response."""
        response_format = json_schema_request["response_format"]
        content = '{"name": "John", "age": 30, "email": "john@example.com"}'

        is_valid, error = validate_response(content, response_format)

        assert is_valid is True
        assert error is None

    def test_validate_json_schema_missing_required_field(self, json_schema_request):
        """Test validation fails when required field is missing."""
        response_format = json_schema_request["response_format"]
        content = '{"name": "John"}'  # Missing required 'age'

        is_valid, error = validate_response(content, response_format)

        assert is_valid is False
        assert "Missing required fields" in error

    def test_validate_json_schema_wrong_type(self, json_schema_request):
        """Test validation fails when field has wrong type."""
        response_format = json_schema_request["response_format"]
        content = '{"name": "John", "age": "thirty", "email": "john@example.com"}'

        is_valid, error = validate_response(content, response_format)

        assert is_valid is False
        assert "expected integer" in error.lower()


# ============================================================================
# Test Text Mode
# ============================================================================


class TestTextMode:
    """Tests for text mode (no transformation)."""

    def test_text_mode_no_transformation(self):
        """Test that text mode doesn't modify messages."""
        request = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "text"},
        }

        result = transform_request(request)

        # response_format should be removed but messages unchanged
        assert "response_format" not in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_no_response_format_no_transformation(self):
        """Test that missing response_format doesn't modify request."""
        request = {"model": "test", "messages": [{"role": "user", "content": "test"}]}

        result = transform_request(request)

        # Should be unchanged
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_unknown_response_format_type(self):
        """Test handling of unknown response_format type."""
        request = {
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "unknown_type"},
        }

        # Should not raise exception, just log warning
        result = transform_request(request)
        assert "response_format" not in result

    def test_empty_messages(self):
        """Test transformation with empty messages list."""
        request = {
            "model": "test",
            "messages": [],
            "response_format": {"type": "json_object"},
        }

        result = transform_request(request)

        # System prompt should be added
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "system"

    def test_existing_system_message(self):
        """Test transformation with existing system message."""
        request = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "test"},
            ],
            "response_format": {"type": "json_object"},
        }

        result = transform_request(request)

        # JSON mode instructions should be inserted first
        assert result["messages"][0]["role"] == "system"
        assert "valid JSON" in result["messages"][0]["content"]
        assert len(result["messages"]) == 3  # Added system prompt

    def test_multiple_messages_preserved(self):
        """Test that all messages are preserved during transformation."""
        request = {
            "model": "test",
            "messages": [
                {"role": "system", "content": "Original system"},
                {"role": "user", "content": "Question 1"},
                {"role": "assistant", "content": "Answer 1"},
                {"role": "user", "content": "Question 2"},
            ],
            "response_format": {"type": "json_object"},
        }

        result = transform_request(request)

        # All original messages should be preserved
        assert len(result["messages"]) == 6  # 5 original + 1 JSON mode instruction
        assert result["messages"][1]["content"] == "Original system"


# ============================================================================
# Test Singleton
# ============================================================================


class TestSingleton:
    """Tests for singleton transformer instance."""

    def test_get_transformer_returns_singleton(self):
        """Test that get_transformer returns same instance."""
        transformer1 = get_transformer()
        transformer2 = get_transformer()

        assert transformer1 is transformer2

    def test_get_transformer_with_different_strict_mode(self):
        """Test that different strict_mode creates new instance."""
        transformer1 = get_transformer(strict_mode=False)
        transformer2 = get_transformer(strict_mode=True)

        assert transformer1 is not transformer2
        assert transformer1.strict_mode is False
        assert transformer2.strict_mode is True
