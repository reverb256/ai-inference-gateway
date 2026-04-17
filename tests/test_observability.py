# modules/services/ai-inference/ai_inference_gateway/tests/test_observability.py
import asyncio
import pytest
from unittest.mock import Mock
from fastapi import Request
from ai_inference_gateway.middleware.observability import ObservabilityMiddleware
from ai_inference_gateway.config import ObservabilityConfig


@pytest.mark.asyncio
async def test_observability_generates_request_id():
    """Test that observability middleware generates request ID when not present"""
    config = ObservabilityConfig(enabled=True)
    middleware = ObservabilityMiddleware(config)

    # Create mock request without X-Request-ID
    request = Mock(spec=Request)
    request.headers = {}
    request.state = Mock()

    context = {}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None
    assert "request_id" in context
    assert len(context["request_id"]) > 0
    assert "start_time" in context


@pytest.mark.asyncio
async def test_observability_preserves_existing_request_id():
    """Test that observability middleware preserves existing request ID"""
    config = ObservabilityConfig(enabled=True, request_id_header="X-Request-ID")
    middleware = ObservabilityMiddleware(config)

    # Create mock request with X-Request-ID
    request = Mock(spec=Request)
    request.headers = {"X-Request-ID": "existing-request-id-123"}
    request.state = Mock()

    context = {}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None
    assert context["request_id"] == "existing-request-id-id-123"


@pytest.mark.asyncio
async def test_observability_tracks_processing_time():
    """Test that observability middleware tracks processing time"""
    config = ObservabilityConfig(enabled=True)
    middleware = ObservabilityMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {}
    request.state = Mock()

    context = {}

    # Process request
    await middleware.process_request(request, context)

    # Simulate some processing time
    await asyncio.sleep(0.01)

    # Process response
    response = {"data": "test"}
    result = await middleware.process_response(response, context)

    assert "gateway_metadata" in result
    assert "request_id" in result["gateway_metadata"]
    assert "processing_time_ms" in result["gateway_metadata"]
    assert result["gateway_metadata"]["processing_time_ms"] >= 10  # At least 10ms


@pytest.mark.asyncio
async def test_observability_adds_gateway_metadata():
    """Test that observability middleware adds gateway metadata to response"""
    config = ObservabilityConfig(enabled=True)
    middleware = ObservabilityMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {}
    request.state = Mock()

    context = {}
    await middleware.process_request(request, context)

    response = {"choices": [{"message": {"content": "test"}}]}
    result = await middleware.process_response(response, context)

    assert "gateway_metadata" in result
    assert "request_id" in result["gateway_metadata"]
    assert "processing_time_ms" in result["gateway_metadata"]
    assert len(result["gateway_metadata"]["request_id"]) > 0
    assert result["gateway_metadata"]["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_observability_preserves_existing_response_data():
    """Test that observability middleware preserves existing response data"""
    config = ObservabilityConfig(enabled=True)
    middleware = ObservabilityMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {}
    request.state = Mock()

    context = {}
    await middleware.process_request(request, context)

    response = {
        "choices": [{"message": {"content": "test response"}}],
        "usage": {"total_tokens": 100},
    }
    result = await middleware.process_response(response, context)

    # Original data should be preserved
    assert result["choices"] == response["choices"]
    assert result["usage"] == response["usage"]
    # Metadata should be added
    assert "gateway_metadata" in result


@pytest.mark.asyncio
async def test_observability_disabled_skips_processing():
    """Test that disabled observability middleware skips processing"""
    config = ObservabilityConfig(enabled=False)
    middleware = ObservabilityMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {}
    request.state = Mock()

    context = {}

    # Process request
    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None
    # Should not add context when disabled
    assert "request_id" not in context
    assert "start_time" not in context

    # Process response
    response = {"data": "test"}
    result = await middleware.process_response(response, context)

    # Should not modify response when disabled
    assert result == response
    assert "gateway_metadata" not in result


def test_observability_enabled_property():
    """Test that enabled property reflects config"""
    config_enabled = ObservabilityConfig(enabled=True)
    middleware_enabled = ObservabilityMiddleware(config_enabled)
    assert middleware_enabled.enabled is True

    config_disabled = ObservabilityConfig(enabled=False)
    middleware_disabled = ObservabilityMiddleware(config_disabled)
    assert middleware_disabled.enabled is False


@pytest.mark.asyncio
async def test_observability_custom_request_id_header():
    """Test custom request ID header name"""
    config = ObservabilityConfig(enabled=True, request_id_header="X-Custom-Request-ID")
    middleware = ObservabilityMiddleware(config)

    request = Mock(spec=Request)
    request.headers = {"X-Custom-Request-ID": "custom-id-456"}
    request.state = Mock()

    context = {}
    await middleware.process_request(request, context)

    assert context["request_id"] == "custom-id-456"


@pytest.mark.asyncio
async def test_observability_generates_unique_request_ids():
    """Test that each request gets a unique request ID"""
    config = ObservabilityConfig(enabled=True)
    middleware = ObservabilityMiddleware(config)

    request_ids = set()
    for _ in range(10):
        request = Mock(spec=Request)
        request.headers = {}
        request.state = Mock()

        context = {}
        await middleware.process_request(request, context)

        request_ids.add(context["request_id"])

    # All request IDs should be unique
    assert len(request_ids) == 10
