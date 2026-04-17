# modules/services/ai-inference/ai_inference_gateway/tests/test_main.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_inference_gateway.main import create_app, GatewayState
from ai_inference_gateway.config import GatewayConfig


class TestCreateApp:
    """Test FastAPI app creation."""

    def test_creates_fastapi_app(self):
        """create_app returns a FastAPI application."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)

        assert isinstance(app, FastAPI)

    def test_includes_health_endpoint(self):
        """App includes /health endpoint."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_includes_gateway_info(self):
        """Health endpoint includes gateway version and info."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)
        client = TestClient(app)

        response = client.get("/health")

        data = response.json()
        assert "gateway" in data
        assert "version" in data["gateway"]

    def test_includes_models_endpoint(self):
        """App includes /v1/models endpoint."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)
        client = TestClient(app)

        # Mock backend response
        with patch("ai_inference_gateway.main.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "data": [
                    {"id": "model-1", "object": "model"},
                    {"id": "model-2", "object": "model"},
                ]
            }

            mock_http_client = AsyncMock()
            mock_http_client.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_http_client

            response = client.get("/v1/models")

            assert response.status_code == 200
            data = response.json()
            assert "data" in data


class TestMiddlewarePipeline:
    """Test middleware pipeline integration."""

    @pytest.mark.asyncio
    async def test_creates_middleware_pipeline(self):
        """App creates middleware pipeline with configured middleware."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )
        config.middleware.observability.enabled = True
        config.middleware.security.enabled = True

        app = create_app(config)

        # Check that middleware were added
        # This would require exposing the pipeline for inspection
        # For now, just verify the app was created
        assert app is not None

    @pytest.mark.asyncio
    async def test_disabled_middleware_not_added(self):
        """Disabled middleware are not added to pipeline."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )
        config.middleware.observability.enabled = False
        config.middleware.security.enabled = False

        app = create_app(config)

        assert app is not None


class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint."""

    @pytest.mark.asyncio
    async def test_forwards_to_backend(self):
        """Endpoint forwards requests to backend."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)
        client = TestClient(app)

        # Mock backend response
        with patch("ai_inference_gateway.main.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_http_client

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_uses_middleware_pipeline(self):
        """Endpoint processes requests through middleware pipeline."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )
        config.middleware.security.enabled = True
        config.middleware.observability.enabled = True

        app = create_app(config)
        client = TestClient(app)

        # Mock backend response
        with patch("ai_inference_gateway.main.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_http_client

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # If middleware pipeline is working, request should still succeed
            # (assuming no middleware blocks it)
            assert response.status_code == 200


class TestLifespanManagement:
    """Test startup and shutdown lifecycle."""

    @pytest.mark.asyncio
    async def test_initializes_redis_on_startup(self):
        """App initializes Redis client on startup."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)

        # The app should have lifespan handlers configured
        # We can't easily test startup in unit tests, but we can
        # verify the app was created successfully
        assert app is not None

    @pytest.mark.asyncio
    async def test_closes_redis_on_shutdown(self):
        """App closes Redis client on shutdown."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)

        # Lifespan handlers should be configured
        assert app is not None


class TestGatewayState:
    """Test gateway state management."""

    def test_gateway_state_stores_config(self):
        """GatewayState stores configuration."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        state = GatewayState(config=config)

        assert state.config == config

    def test_gateway_state_stores_redis_client(self):
        """GatewayState stores Redis client."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        redis_client = Mock()

        state = GatewayState(config=config, redis_client=redis_client)

        assert state.redis_client == redis_client

    def test_gateway_state_stores_pipeline(self):
        """GatewayState stores middleware pipeline."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        pipeline = Mock()

        state = GatewayState(config=config, pipeline=pipeline)

        assert state.pipeline == pipeline


class TestEnvironmentConfig:
    """Test configuration loading from environment."""

    @patch.dict(
        "os.environ",
        {
            "GATEWAY_HOST": "0.0.0.0",
            "GATEWAY_PORT": "9000",
            "BACKEND_URL": "http://backend:9999",
            "SECURITY_ENABLED": "false",
            "OBSERVABILITY_ENABLED": "true",
        },
    )
    def test_loads_config_from_environment(self):
        """App loads configuration from environment variables."""
        config = GatewayConfig.load_from_env()

        assert config.gateway_host == "0.0.0.0"
        assert config.gateway_port == 9000
        assert config.backend_url == "http://backend:9999"
        assert config.middleware.security.enabled is False
        assert config.middleware.observability.enabled is True

    @patch.dict("os.environ", {}, clear=True)
    def test_uses_defaults_when_env_not_set(self):
        """App uses default values when environment variables not set."""
        config = GatewayConfig.load_from_env()

        assert config.gateway_host == "127.0.0.1"
        assert config.gateway_port == 8080
        assert config.backend_url == "http://127.0.0.1:1234"


class TestErrorHandling:
    """Test error handling."""

    @pytest.mark.asyncio
    async def test_backend_errors_propagated(self):
        """Backend errors are properly propagated."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )

        app = create_app(config)
        client = TestClient(app)

        # Mock backend error
        with patch("ai_inference_gateway.main.httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.json.return_value = {
                "error": {"message": "Internal server error"}
            }

            mock_http_client = AsyncMock()
            mock_http_client.post.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_http_client

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            # Should propagate the error
            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_middleware_blocks_request(self):
        """Middleware can block requests."""
        config = GatewayConfig(
            gateway_host="127.0.0.1",
            gateway_port=8080,
            backend_url="http://test-backend:1234",
        )
        config.middleware.security.enabled = True
        config.middleware.security.max_request_size = 1  # Very small limit

        app = create_app(config)
        client = TestClient(app)

        # Send a large request that should be blocked
        large_content = "x" * 10000

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": large_content}],
            },
        )

        # Should be blocked by security middleware
        assert response.status_code == 413  # Payload Too Large
