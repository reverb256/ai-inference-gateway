"""
OTel Tracing Setup with PII-Sanitized Span Export.

Wraps the OpenTelemetry SDK pipeline so that all span attributes pass
through OTelSpanSanitizer before leaving the process.  Falls back to a
no-op when opentelemetry packages are not installed.
"""

import logging
import os
from typing import Any, Optional

from ai_inference_gateway.observability.otel_sanitizer import OTelSpanSanitizer
from ai_inference_gateway.pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy OTel imports — guarded so the module loads even without the packages.
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

    _OTEL_SDK_AVAILABLE = True
except ImportError:
    # Stash placeholder names so tests / type checkers can still import the
    # module.  These are never used at runtime when the SDK is absent.
    Resource = None  # type: ignore[assignment,misc]
    TracerProvider = None  # type: ignore[assignment,misc]
    BatchSpanProcessor = None  # type: ignore[assignment,misc]
    SpanExporter = object  # type: ignore[assignment,misc]
    trace = None  # type: ignore[assignment]
    _OTEL_SDK_AVAILABLE = False

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    _OTEL_EXPORTER_AVAILABLE = True
except ImportError:
    OTLPSpanExporter = None  # type: ignore[assignment,misc]
    _OTEL_EXPORTER_AVAILABLE = False

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    _FASTAPI_INSTRUMENTOR_AVAILABLE = True
except ImportError:
    FastAPIInstrumentor = None  # type: ignore[assignment]
    _FASTAPI_INSTRUMENTOR_AVAILABLE = False


# ---------------------------------------------------------------------------
# SanitizingExporter — wraps any SpanExporter and redacts PII before export.
# Always defined so it can be tested and imported without OTel installed.
# ---------------------------------------------------------------------------
class _SanitizedSpanView:
    """Lightweight proxy that overrides ``attributes`` with sanitized values.

    Proxies all other attribute access to the original span so the OTel
    SDK (and tests) can read ``.name``, ``.context``, etc. as usual.
    """

    __slots__ = ("_orig", "attributes")

    def __init__(self, orig: Any, attributes: dict) -> None:
        self._orig = orig
        self.attributes = attributes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class SanitizingExporter(SpanExporter):  # type: ignore[misc]
    """
    Delegating SpanExporter that runs every span's attributes through
    OTelSpanSanitizer before forwarding to the real exporter.
    """

    def __init__(self, delegate: Any, sanitizer: OTelSpanSanitizer):
        self._delegate = delegate
        self._sanitizer = sanitizer

    def export(self, spans: Any) -> Any:
        sanitized_spans = []
        for span in spans:
            if span.attributes:
                cleaned = self._sanitizer.sanitize_attributes(dict(span.attributes))
                span = _SanitizedSpanView(span, cleaned)
            sanitized_spans.append(span)
        return self._delegate.export(sanitized_spans)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> Any:
        return self._delegate.force_flush(timeout_millis)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def setup_tracing(
    app: Optional[Any] = None,
    redactor: Optional[PIIRedactor] = None,
) -> Optional[Any]:
    """
    Initialize OpenTelemetry tracing with PII sanitization.

    * Creates a ``TracerProvider`` whose exporter pipeline includes
      ``SanitizingExporter`` so no PII leaves the process.
    * Instruments the FastAPI *app* when ``FastAPIInstrumentor`` is available.
    * Reads config from environment variables:

      - ``OTEL_EXPORTER_OTLP_ENDPOINT`` (default ``http://localhost:4317``)
      - ``OTEL_SERVICE_NAME`` (default ``ai-inference-gateway``)

    Args:
        app: FastAPI application instance (optional — instrumentation skipped if None).
        redactor: PIIRedactor instance (optional — falls back to ``get_default_redactor()``).

    Returns:
        The ``TracerProvider`` on success, or ``None`` if OTel SDK is unavailable.
    """
    if not _OTEL_SDK_AVAILABLE:
        logger.warning(
            "OpenTelemetry SDK not installed — tracing disabled. "
            "Install opentelemetry-sdk and opentelemetry-exporter-otlp-proto-grpc."
        )
        return None

    # Resolve redactor -------------------------------------------------------
    if redactor is None:
        from ai_inference_gateway.pii_redactor import get_default_redactor

        redactor = get_default_redactor()

    sanitizer = OTelSpanSanitizer(redactor)

    # Build exporter chain ---------------------------------------------------
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    service_name = os.environ.get("OTEL_SERVICE_NAME", "ai-inference-gateway")

    resource = Resource.create({"service.name": service_name})  # type: ignore[union-attr]

    if _OTEL_EXPORTER_AVAILABLE:
        real_exporter = OTLPSpanExporter(endpoint=endpoint)  # type: ignore[misc]
    else:
        logger.warning(
            "OTLP exporter not installed — spans will not be exported. Install opentelemetry-exporter-otlp-proto-grpc."
        )
        return None

    # Wrap the real exporter with PII sanitization
    sanitizing_exporter = SanitizingExporter(real_exporter, sanitizer)

    provider = TracerProvider(resource=resource)  # type: ignore[misc]
    provider.add_span_processor(BatchSpanProcessor(sanitizing_exporter))  # type: ignore[misc]

    # Register globally
    trace.set_tracer_provider(provider)  # type: ignore[union-attr]

    # Instrument FastAPI if app given and instrumentor available
    if app is not None and _FASTAPI_INSTRUMENTOR_AVAILABLE:
        FastAPIInstrumentor.instrument_app(app)  # type: ignore[misc]
        logger.info("FastAPI instrumented for OTel tracing")
    elif app is not None:
        logger.warning("FastAPIInstrumentor not available — skipping app instrumentation")

    logger.info(
        "OTel tracing initialized (service=%s, endpoint=%s, sanitization=enabled)",
        service_name,
        endpoint,
    )
    return provider
