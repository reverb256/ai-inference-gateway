"""
Tests for Phase 3 MLSEC: Observability Hardening — OTel tracing with PII-sanitized spans.

Covers:
1. SanitizingExporter — PII redaction in span attributes
2. setup_tracing() — initialization, env var config, graceful degradation
3. OTelSpanSanitizer integration — email, SSN, phone, credit card, API key redaction
4. Metric attributes pass-through
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from ai_inference_gateway.observability.otel_sanitizer import OTelSpanSanitizer
from ai_inference_gateway.observability.tracing import SanitizingExporter
from ai_inference_gateway.pii_redactor import PIIRedactor

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def redactor():
    """PIIRedactor instance."""
    return PIIRedactor()


@pytest.fixture
def sanitizer(redactor):
    """OTelSpanSanitizer wrapping a PIIRedactor."""
    return OTelSpanSanitizer(redactor)


# ============================================================================
# TestSanitizingExporter
# ============================================================================


class TestSanitizingExporter:
    """Test the SanitizingExporter class."""

    def _make_exporter(self, sanitizer):
        """Build a SanitizingExporter with a mock delegate."""
        delegate = MagicMock()
        delegate.export.return_value = 0  # SpanExportResult.SUCCESS
        exporter = SanitizingExporter(delegate=delegate, sanitizer=sanitizer)
        return exporter, delegate

    def _make_span(self, attributes):
        """Create a mock ReadableSpan with given attributes."""
        span = MagicMock()
        span.attributes = dict(attributes) if attributes else None
        return span

    def test_email_in_span_attributes_is_redacted(self, sanitizer):
        """PII (email) in span attributes must be redacted before export."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span({"content": "Contact user" + "@example.com for info"})

        exporter.export([span])

        delegate.export.assert_called_once()
        called_spans = delegate.export.call_args[0][0]
        assert ("user" + "@example.com") not in str(called_spans[0].attributes.get("content", ""))

    def test_ssn_in_span_attributes_is_redacted(self, sanitizer):
        """SSN in span attributes must be redacted."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span({"prompt": "My SSN is 123" + "-45-6789"})

        exporter.export([span])
        called_spans = delegate.export.call_args[0][0]
        assert ("123" + "-45-6789") not in str(called_spans[0].attributes.get("prompt", ""))

    def test_phone_in_span_attributes_is_redacted(self, sanitizer):
        """Phone number in span attributes must be redacted."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span({"response": "Call +1-555" + "-123-4567"})

        exporter.export([span])
        called_spans = delegate.export.call_args[0][0]
        assert ("555" + "-123-4567") not in str(called_spans[0].attributes.get("response", ""))

    def test_non_pii_numeric_attributes_pass_through(self, sanitizer):
        """Numeric/metric attributes must pass through unchanged."""
        exporter, delegate = self._make_exporter(sanitizer)
        original_attrs = {
            "duration_ms": 150,
            "token_count": 42,
            "status_code": 200,
            "is_success": True,
        }
        span = self._make_span(original_attrs)

        exporter.export([span])
        called_spans = delegate.export.call_args[0][0]
        attrs = called_spans[0].attributes
        assert attrs["duration_ms"] == 150
        assert attrs["token_count"] == 42
        assert attrs["status_code"] == 200
        assert attrs["is_success"] is True

    def test_empty_attributes_dict_passes(self, sanitizer):
        """Empty attributes dict must not cause errors."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span({})

        exporter.export([span])
        delegate.export.assert_called_once()

    def test_none_attributes_passes(self, sanitizer):
        """None attributes must not cause errors."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span(None)

        exporter.export([span])
        delegate.export.assert_called_once()

    def test_spans_without_pii_keys_untouched(self, sanitizer):
        """Spans with only non-PII keys remain unchanged."""
        exporter, delegate = self._make_exporter(sanitizer)
        original = {"http.method": "POST", "http.url": "/v1/chat/completions"}
        span = self._make_span(original)

        exporter.export([span])
        called_spans = delegate.export.call_args[0][0]
        attrs = called_spans[0].attributes
        assert attrs["http.method"] == "POST"
        assert attrs["http.url"] == "/v1/chat/completions"

    def test_mixed_pii_and_metric_attributes(self, sanitizer):
        """PII-bearing keys are sanitized while metrics are untouched."""
        exporter, delegate = self._make_exporter(sanitizer)
        span = self._make_span(
            {
                "input": "Email me at test" + "@example.com",
                "duration_ms": 300,
                "token_count": 100,
                "output": "Sure, I will email test" + "@example.com",
            }
        )

        exporter.export([span])
        called_spans = delegate.export.call_args[0][0]
        attrs = called_spans[0].attributes
        assert ("test" + "@example.com") not in str(attrs.get("input", ""))
        assert ("test" + "@example.com") not in str(attrs.get("output", ""))
        assert attrs["duration_ms"] == 300
        assert attrs["token_count"] == 100

    def test_shutdown_delegates(self, sanitizer):
        """shutdown() must forward to the delegate."""
        exporter, delegate = self._make_exporter(sanitizer)
        exporter.shutdown()
        delegate.shutdown.assert_called_once()

    def test_force_flush_delegates(self, sanitizer):
        """force_flush() must forward to the delegate."""
        exporter, delegate = self._make_exporter(sanitizer)
        delegate.force_flush.return_value = True
        result = exporter.force_flush(5000)
        delegate.force_flush.assert_called_once_with(5000)
        assert result is True


# ============================================================================
# TestTracingSetup
# ============================================================================


class TestTracingSetup:
    """Test setup_tracing() initialization and config."""

    def test_returns_none_when_otel_sdk_unavailable(self):
        """setup_tracing must return None when OTel SDK is not installed."""
        from ai_inference_gateway.observability import tracing

        with patch.object(tracing, "_OTEL_SDK_AVAILABLE", False):
            result = tracing.setup_tracing(app=None, redactor=PIIRedactor())
            assert result is None

    def test_returns_none_when_exporter_unavailable(self):
        """setup_tracing must return None when OTLP exporter is not installed."""
        from ai_inference_gateway.observability import tracing

        # SDK available but exporter not — need to also mock Resource etc.
        # so the function doesn't NameError on the code path past the check.
        mock_resource_cls = MagicMock()
        mock_provider = MagicMock()

        with (
            patch.object(tracing, "_OTEL_SDK_AVAILABLE", True),
            patch.object(tracing, "_OTEL_EXPORTER_AVAILABLE", False),
            patch.object(tracing, "Resource", mock_resource_cls, create=True),
        ):
            result = tracing.setup_tracing(app=None, redactor=PIIRedactor())
            assert result is None

    def test_custom_service_name_env_var(self):
        """OTEL_SERVICE_NAME env var must be respected."""
        from ai_inference_gateway.observability import tracing

        mock_resource_cls = MagicMock()
        mock_provider = MagicMock()
        mock_tracer_provider_cls = MagicMock(return_value=mock_provider)
        mock_exporter_cls = MagicMock()

        with (
            patch.object(tracing, "_OTEL_SDK_AVAILABLE", True),
            patch.object(tracing, "_OTEL_EXPORTER_AVAILABLE", True),
            patch.object(tracing, "_FASTAPI_INSTRUMENTOR_AVAILABLE", False),
            patch.object(tracing, "Resource", mock_resource_cls, create=True),
            patch.object(tracing, "TracerProvider", mock_tracer_provider_cls, create=True),
            patch.object(tracing, "OTLPSpanExporter", mock_exporter_cls, create=True),
            patch.object(tracing, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(tracing, "trace", MagicMock(), create=True),
            patch.dict(
                os.environ,
                {"OTEL_SERVICE_NAME": "custom-gateway", "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317"},
            ),
        ):
            tracing.setup_tracing(app=None, redactor=PIIRedactor())

            mock_resource_cls.create.assert_called_once_with({"service.name": "custom-gateway"})

    def test_default_service_name(self):
        """Default service name must be 'ai-inference-gateway'."""
        from ai_inference_gateway.observability import tracing

        mock_resource_cls = MagicMock()
        mock_provider = MagicMock()

        with (
            patch.object(tracing, "_OTEL_SDK_AVAILABLE", True),
            patch.object(tracing, "_OTEL_EXPORTER_AVAILABLE", True),
            patch.object(tracing, "_FASTAPI_INSTRUMENTOR_AVAILABLE", False),
            patch.object(tracing, "Resource", mock_resource_cls, create=True),
            patch.object(tracing, "TracerProvider", MagicMock(return_value=mock_provider), create=True),
            patch.object(tracing, "OTLPSpanExporter", MagicMock(), create=True),
            patch.object(tracing, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(tracing, "trace", MagicMock(), create=True),
        ):
            # Clear service name to test default
            env = os.environ.copy()
            env.pop("OTEL_SERVICE_NAME", None)
            with patch.dict(os.environ, env, clear=True):
                tracing.setup_tracing(app=None, redactor=PIIRedactor())

                mock_resource_cls.create.assert_called_once_with({"service.name": "ai-inference-gateway"})

    def test_fastapi_instrumentation_called_when_available(self):
        """FastAPI app must be instrumented when FastAPIInstrumentor is available."""
        from ai_inference_gateway.observability import tracing

        mock_app = MagicMock()
        mock_provider = MagicMock()
        mock_instrumentor = MagicMock()

        with (
            patch.object(tracing, "_OTEL_SDK_AVAILABLE", True),
            patch.object(tracing, "_OTEL_EXPORTER_AVAILABLE", True),
            patch.object(tracing, "_FASTAPI_INSTRUMENTOR_AVAILABLE", True),
            patch.object(tracing, "FastAPIInstrumentor", mock_instrumentor, create=True),
            patch.object(tracing, "Resource", MagicMock(), create=True),
            patch.object(tracing, "TracerProvider", MagicMock(return_value=mock_provider), create=True),
            patch.object(tracing, "OTLPSpanExporter", MagicMock(), create=True),
            patch.object(tracing, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(tracing, "trace", MagicMock(), create=True),
        ):
            tracing.setup_tracing(app=mock_app, redactor=PIIRedactor())

            mock_instrumentor.instrument_app.assert_called_once_with(mock_app)

    def test_uses_default_redactor_when_none_provided(self):
        """When redactor is None, get_default_redactor() must be called."""
        from ai_inference_gateway.observability import tracing

        with (
            patch.object(tracing, "_OTEL_SDK_AVAILABLE", True),
            patch.object(tracing, "_OTEL_EXPORTER_AVAILABLE", True),
            patch.object(tracing, "_FASTAPI_INSTRUMENTOR_AVAILABLE", False),
            patch.object(tracing, "Resource", MagicMock(), create=True),
            patch.object(tracing, "TracerProvider", MagicMock(return_value=MagicMock()), create=True),
            patch.object(tracing, "OTLPSpanExporter", MagicMock(), create=True),
            patch.object(tracing, "BatchSpanProcessor", MagicMock(), create=True),
            patch.object(tracing, "trace", MagicMock(), create=True),
            patch("ai_inference_gateway.pii_redactor.get_default_redactor") as mock_get,
        ):
            mock_get.return_value = PIIRedactor()
            tracing.setup_tracing(app=None, redactor=None)
            mock_get.assert_called_once()


# ============================================================================
# TestOTelSanitizerIntegration
# ============================================================================


class TestOTelSanitizerIntegration:
    """Test OTelSpanSanitizer catches all PII types relevant to span attributes."""

    def test_email_redaction(self, sanitizer):
        """Email in PII-bearing keys is redacted."""
        result = sanitizer.sanitize_attributes({"content": "Reach me at admin@corp.com"})
        assert "admin@corp.com" not in result["content"]

    def test_ssn_redaction(self, sanitizer):
        """SSN in PII-bearing keys is redacted."""
        result = sanitizer.sanitize_attributes({"prompt": "SSN: 987-65-4321"})
        assert "987-65-4321" not in result["prompt"]

    def test_phone_redaction(self, sanitizer):
        """Phone number in PII-bearing keys is redacted."""
        result = sanitizer.sanitize_attributes({"message": "Phone: +1-800-555-0199"})
        assert "800-555-0199" not in result["message"]

    def test_credit_card_redaction(self, sanitizer):
        """Credit card number in PII-bearing keys is redacted."""
        result = sanitizer.sanitize_attributes({"input": "Card: 4532-1234-5678-9012"})
        assert "4532-1234-5678-9012" not in result["input"]

    def test_api_key_redaction_long_alphanumeric(self, sanitizer):
        """Long 32+ char alphanumeric API key pattern is redacted."""
        result = sanitizer.sanitize_attributes({"arguments": "key=abcdefghijklmnopqrstuvwx1234567890ABCD"})
        assert "abcdefghijklmnopqrstuvwx1234567890ABCD" not in result["arguments"]

    def test_bearer_token_redaction(self, sanitizer):
        """Bearer token pattern is redacted."""
        result = sanitizer.sanitize_attributes({"message": "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"})
        assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in result["message"]

    def test_metric_duration_ms_untouched(self, sanitizer):
        """duration_ms is not a PII key and passes through."""
        result = sanitizer.sanitize_attributes({"duration_ms": 42})
        assert result["duration_ms"] == 42

    def test_metric_token_count_untouched(self, sanitizer):
        """token_count is not a PII key and passes through."""
        result = sanitizer.sanitize_attributes({"token_count": 1024})
        assert result["token_count"] == 1024

    def test_metric_status_code_untouched(self, sanitizer):
        """status_code is not a PII key and passes through."""
        result = sanitizer.sanitize_attributes({"status_code": 200})
        assert result["status_code"] == 200

    def test_llm_prefixed_key_sanitized(self, sanitizer):
        """Keys starting with 'llm.' are treated as PII-bearing."""
        result = sanitizer.sanitize_attributes({"llm.user_input": "user@example.com"})
        assert "user@example.com" not in result["llm.user_input"]

    def test_non_pii_string_key_untouched(self, sanitizer):
        """Non-PII string keys with non-PII content pass through."""
        result = sanitizer.sanitize_attributes({"http.method": "GET"})
        assert result["http.method"] == "GET"

    def test_empty_dict(self, sanitizer):
        """Empty attributes dict returns empty dict."""
        result = sanitizer.sanitize_attributes({})
        assert result == {}

    def test_none_returns_empty(self, sanitizer):
        """None attributes returns empty dict."""
        result = sanitizer.sanitize_attributes(None)
        assert result == {}

    def test_numeric_value_in_pii_key_passes(self, sanitizer):
        """Non-string values in PII keys are not redacted (they're not strings)."""
        result = sanitizer.sanitize_attributes({"token_count": 42, "content": 99})
        assert result["content"] == 99

    def test_multiple_pii_types_in_one_value(self, sanitizer):
        """Multiple PII types in one string are all redacted."""
        result = sanitizer.sanitize_attributes({"message": "Email user@test.com, SSN 111-22-3333, phone 555-000-1111"})
        val = result["message"]
        assert "user@test.com" not in val
        assert "111-22-3333" not in val
        assert "555-000-1111" not in val
