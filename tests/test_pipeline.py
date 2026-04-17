# modules/services/ai-inference/ai_inference_gateway/tests/test_pipeline.py
import pytest
from unittest.mock import Mock
from fastapi import Request, HTTPException

from ai_inference_gateway.pipeline import MiddlewarePipeline
from ai_inference_gateway.middleware.base import Middleware


class DummyMiddleware(Middleware):
    """Dummy middleware for testing."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process_request(self, request: Request, context: dict):
        context[f"{self.name}_request"] = True
        return True, None

    async def process_response(self, response: dict, context: dict):
        response[f"{self.name}_response"] = True
        return response


class FailingMiddleware(Middleware):
    """Middleware that blocks requests."""

    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process_request(self, request: Request, context: dict):
        context[f"{self.name}_request"] = True
        return False, HTTPException(status_code=403, detail=f"Blocked by {self.name}")

    async def process_response(self, response: dict, context: dict):
        response[f"{self.name}_response"] = True
        return response


class DisabledMiddleware(Middleware):
    """Middleware that is disabled."""

    @property
    def enabled(self) -> bool:
        return False

    async def process_request(self, request: Request, context: dict):
        raise Exception("Should not be called")

    async def process_response(self, response: dict, context: dict):
        raise Exception("Should not be called")


class TestMiddlewarePipeline:
    """Test middleware pipeline execution."""

    @pytest.mark.asyncio
    async def test_add_middleware(self):
        """Can add middleware to pipeline."""
        pipeline = MiddlewarePipeline()
        middleware = DummyMiddleware("test")

        pipeline.add(middleware)

        assert len(pipeline._middleware) == 1
        assert pipeline._middleware[0] == middleware

    @pytest.mark.asyncio
    async def test_add_multiple_middleware(self):
        """Can add multiple middleware in order."""
        pipeline = MiddlewarePipeline()
        m1 = DummyMiddleware("first")
        m2 = DummyMiddleware("second")
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        assert len(pipeline._middleware) == 3
        assert pipeline._middleware[0] == m1
        assert pipeline._middleware[1] == m2
        assert pipeline._middleware[2] == m3

    @pytest.mark.asyncio
    async def test_process_request_in_order(self):
        """Process request executes middleware in order (first to last)."""
        pipeline = MiddlewarePipeline()
        m1 = DummyMiddleware("first")
        m2 = DummyMiddleware("second")
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await pipeline.process_request(request, context)

        # All middleware should have been executed
        assert context["first_request"] is True
        assert context["second_request"] is True
        assert context["third_request"] is True
        assert should_continue is True
        assert error is None

    @pytest.mark.asyncio
    async def test_process_response_in_reverse_order(self):
        """Process response executes middleware in reverse (last to first)."""
        pipeline = MiddlewarePipeline()
        m1 = DummyMiddleware("first")
        m2 = DummyMiddleware("second")
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        response = {}
        context = {}

        result = await pipeline.process_response(response, context)

        # All middleware should have modified response
        assert result["first_response"] is True
        assert result["second_response"] is True
        assert result["third_response"] is True

    @pytest.mark.asyncio
    async def test_short_circuit_on_error(self):
        """Pipeline stops processing when middleware returns error."""
        pipeline = MiddlewarePipeline()
        m1 = DummyMiddleware("first")
        m2 = FailingMiddleware("second")
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await pipeline.process_request(request, context)

        # First middleware executed, second failed, third didn't run
        assert context["first_request"] is True
        assert context["second_request"] is True
        assert "third_request" not in context
        assert should_continue is False
        assert error is not None
        assert error.status_code == 403

    @pytest.mark.asyncio
    async def test_disabled_middleware_skipped(self):
        """Disabled middleware is skipped during execution."""
        pipeline = MiddlewarePipeline()
        m1 = DummyMiddleware("first")
        m2 = DisabledMiddleware()
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await pipeline.process_request(request, context)

        # Enabled middleware executed, disabled one skipped
        assert context["first_request"] is True
        assert "disabled_request" not in context
        assert context["third_request"] is True
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_context_passed_between_middleware(self):
        """Context dict is shared between middleware."""

        class ContextMiddleware(Middleware):
            def __init__(self, name: str, key: str, value: any):
                self.name = name
                self.key = key
                self.value = value

            @property
            def enabled(self) -> bool:
                return True

            async def process_request(self, request: Request, context: dict):
                context[self.key] = self.value
                return True, None

            async def process_response(self, response: dict, context: dict):
                return response

        pipeline = MiddlewarePipeline()
        m1 = ContextMiddleware("first", "key1", "value1")
        m2 = ContextMiddleware("second", "key2", "value2")
        m3 = ContextMiddleware("third", "key3", "value3")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        await pipeline.process_request(request, context)

        # All context should be available
        assert context["key1"] == "value1"
        assert context["key2"] == "value2"
        assert context["key3"] == "value3"

    @pytest.mark.asyncio
    async def test_empty_pipeline(self):
        """Empty pipeline allows requests through."""
        pipeline = MiddlewarePipeline()

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await pipeline.process_request(request, context)

        assert should_continue is True
        assert error is None

        response = {}
        result = await pipeline.process_response(response, context)
        assert result == response

    @pytest.mark.asyncio
    async def test_response_modifications_accumulate(self):
        """Response modifications from all middleware accumulate."""

        class ResponseModifier(Middleware):
            def __init__(self, name: str, header: str):
                self.name = name
                self.header = header

            @property
            def enabled(self) -> bool:
                return True

            async def process_request(self, request: Request, context: dict):
                return True, None

            async def process_response(self, response: dict, context: dict):
                if "headers" not in response:
                    response["headers"] = {}
                response["headers"][self.header] = self.name
                return response

        pipeline = MiddlewarePipeline()
        m1 = ResponseModifier("first", "X-First")
        m2 = ResponseModifier("second", "X-Second")
        m3 = ResponseModifier("third", "X-Third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        response = {}
        context = {}

        result = await pipeline.process_response(response, context)

        # All headers should be present
        assert result["headers"]["X-First"] == "first"
        assert result["headers"]["X-Second"] == "second"
        assert result["headers"]["X-Third"] == "third"

    @pytest.mark.asyncio
    async def test_request_stops_at_first_error(self):
        """Pipeline stops at first middleware that returns error."""
        pipeline = MiddlewarePipeline()
        m1 = FailingMiddleware("first")
        m2 = FailingMiddleware("second")
        m3 = DummyMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        should_continue, error = await pipeline.process_request(request, context)

        # Only first middleware executed
        assert context["first_request"] is True
        assert "second_request" not in context
        assert "third_request" not in context
        assert should_continue is False
        assert error.detail == "Blocked by first"

    @pytest.mark.asyncio
    async def test_response_execution_order(self):
        """Verify response processing order is reverse of request order."""
        execution_order = []

        class OrderTrackingMiddleware(Middleware):
            def __init__(self, name: str):
                self.name = name

            @property
            def enabled(self) -> bool:
                return True

            async def process_request(self, request: Request, context: dict):
                execution_order.append(f"{self.name}_request")
                return True, None

            async def process_response(self, response: dict, context: dict):
                execution_order.append(f"{self.name}_response")
                return response

        pipeline = MiddlewarePipeline()
        m1 = OrderTrackingMiddleware("first")
        m2 = OrderTrackingMiddleware("second")
        m3 = OrderTrackingMiddleware("third")

        pipeline.add(m1)
        pipeline.add(m2)
        pipeline.add(m3)

        request = Mock(spec=Request)
        context = {}

        await pipeline.process_request(request, context)
        await pipeline.process_response({}, context)

        # Request order: first, second, third
        # Response order: third, second, first
        expected = [
            "first_request",
            "second_request",
            "third_request",
            "third_response",
            "second_response",
            "first_response",
        ]
        assert execution_order == expected
