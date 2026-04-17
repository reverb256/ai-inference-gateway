# modules/services/ai-inference/ai_inference_gateway/tests/test_middleware_base.py
import pytest
from fastapi import HTTPException
from ai_inference_gateway.middleware.base import Middleware


class MockMiddleware(Middleware):
    """Mock middleware for testing"""

    def __init__(self, enabled=True):
        self._enabled = enabled

    async def process_request(self, request, context):
        if not self._enabled:
            return False, HTTPException(503, "Middleware disabled")
        return True, None

    async def process_response(self, response, context):
        response["mock_processed"] = True
        return response

    @property
    def enabled(self) -> bool:
        return self._enabled


@pytest.mark.asyncio
async def test_middleware_interface_process_request():
    """Test middleware can process requests"""
    middleware = MockMiddleware(enabled=True)

    # Create a mock request
    class MockRequest:
        pass

    request = MockRequest()
    context = {}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_middleware_interface_process_response():
    """Test middleware can process responses"""
    middleware = MockMiddleware(enabled=True)

    response = {"data": "test"}
    context = {}

    result = await middleware.process_response(response, context)

    assert result["mock_processed"] is True
    assert result["data"] == "test"


def test_middleware_enabled_property():
    """Test middleware enabled property"""
    middleware_enabled = MockMiddleware(enabled=True)
    middleware_disabled = MockMiddleware(enabled=False)

    assert middleware_enabled.enabled is True
    assert middleware_disabled.enabled is False


@pytest.mark.asyncio
async def test_middleware_can_block_request():
    """Test middleware can block requests"""
    middleware = MockMiddleware(enabled=False)

    class MockRequest:
        pass

    request = MockRequest()
    context = {}

    should_continue, error = await middleware.process_request(request, context)

    assert should_continue is False
    assert isinstance(error, HTTPException)
    assert error.status_code == 503
