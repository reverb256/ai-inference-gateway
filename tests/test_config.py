# modules/services/ai-inference/ai_inference_gateway/tests/test_config.py
import os
from ai_inference_gateway.config import GatewayConfig, MiddlewareConfig


def test_load_config_from_env():
    """Test loading configuration from environment variables"""
    # Set environment variables
    os.environ["GATEWAY_HOST"] = "0.0.0.0"
    os.environ["GATEWAY_PORT"] = "9000"
    os.environ["RATE_LIMIT_ENABLED"] = "true"
    os.environ["RATE_LIMIT_RPM"] = "100"

    config = GatewayConfig.load_from_env()

    assert config.gateway_host == "0.0.0.0"
    assert config.gateway_port == 9000
    assert config.middleware.rate_limiting.enabled is True
    assert config.middleware.rate_limiting.rpm == 100

    # Cleanup
    del os.environ["GATEWAY_HOST"]
    del os.environ["GATEWAY_PORT"]
    del os.environ["RATE_LIMIT_ENABLED"]
    del os.environ["RATE_LIMIT_RPM"]


def test_middleware_config_defaults():
    """Test middleware config has sensible defaults"""
    config = MiddlewareConfig()

    assert config.rate_limiting.enabled is False  # Default disabled
    assert config.security.enabled is True
    assert config.cache.enabled is False
    assert config.circuit_breaker.enabled is True


def test_load_config_defaults_without_env():
    """Test loading config with defaults when no env vars set"""
    # Clear any existing env vars
    for key in ["GATEWAY_HOST", "GATEWAY_PORT", "RATE_LIMIT_ENABLED", "RATE_LIMIT_RPM"]:
        if key in os.environ:
            del os.environ[key]

    config = GatewayConfig.load_from_env()

    assert config.gateway_host == "127.0.0.1"  # Default
    assert config.gateway_port == 8080  # Default
    assert config.middleware.rate_limiting.enabled is False  # Default
    assert config.middleware.rate_limiting.rpm == 60  # Default


def test_boolean_env_parsing():
    """Test that boolean env vars are parsed correctly"""
    test_cases = [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
    ]

    for value, expected in test_cases:
        os.environ["RATE_LIMIT_ENABLED"] = value
        config = GatewayConfig.load_from_env()
        assert (
            config.middleware.rate_limiting.enabled is expected
        ), f"Failed for value: {value}"

    # Cleanup
    if "RATE_LIMIT_ENABLED" in os.environ:
        del os.environ["RATE_LIMIT_ENABLED"]
