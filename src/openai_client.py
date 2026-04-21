"""
OpenAI SDK client wrapper for AI Gateway.

This module provides OpenAI client instances configured for different backends:
- llama.cpp (local OpenAI-compatible server)
- ZAI (cloud OpenAI-compatible API)
- Pollinations (free OpenAI-compatible API)
- nvidia: NVIDIA NIM API for cloud-hosted models

The OpenAI SDK handles:
- Automatic header management (User-Agent, etc.)
- Proper authentication (Bearer tokens)
- Request/response formatting
- Streaming support
- Error handling
"""

import logging
from typing import Optional, Dict, Any
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = logging.getLogger(__name__)


class OpenAIBackendError(Exception):
    """Exception raised when backend request fails."""

    pass


class OpenAIClientWrapper:
    """
    Wrapper for OpenAI SDK clients with automatic backend failover.

    Manages multiple OpenAI clients for different backends and provides
    a unified interface for chat completions with automatic failover.
    """

    def __init__(
        self,
        primary_url: str,
        primary_api_key: Optional[str],
        fallback_url: Optional[str] = None,
        fallback_api_key: Optional[str] = None,
        timeout: float = 120.0,
        zai_models: Optional[list[str]] = None,
        nvidia_url: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        local_backend_url: Optional[str] = None,
        local_backend_model: Optional[str] = None,
        secondary_backend_url: Optional[str] = None,
        secondary_backend_model: Optional[str] = None,
    ):
        """
        Initialize OpenAI client wrapper.

        Args:
            primary_url: Primary backend URL (e.g., llama.cpp)
            primary_api_key: API key for primary backend (optional for local)
            fallback_url: Fallback backend URL (e.g., ZAI)
            fallback_api_key: API key for fallback backend
            timeout: Request timeout in seconds
            zai_models: List of ZAI models to try (in order)
            nvidia_url: NVIDIA NIM API base URL
            nvidia_api_key: NVIDIA NIM API key
            local_backend_url: Local backend URL (e.g., sentry ROCm)
            local_backend_model: Model name on local backend
            secondary_backend_url: Secondary backend URL (e.g., 3060Ti)
            secondary_backend_model: Model name on secondary backend
        """
        self.primary_url = primary_url.rstrip("/")
        # For local servers (llama.cpp), use placeholder if no key provided
        # OpenAI SDK requires api_key to be set, but local servers don't validate it
        if primary_api_key and primary_api_key.strip():
            self.primary_api_key = primary_api_key
        else:
            self.primary_api_key = "not-needed"  # Placeholder for local servers
        self.fallback_url = fallback_url.rstrip("/") if fallback_url else None
        self.fallback_api_key = fallback_api_key
        self.timeout = timeout
        # ZAI models to try in order (from fastest to most capable)
        self.zai_models = zai_models or ["glm-4.6", "glm-4.7", "glm-5"]

        # Initialize primary client
        self.primary_client = AsyncOpenAI(
            base_url=f"{self.primary_url}/v1",
            api_key=self.primary_api_key,
            timeout=timeout,
        )

        # Initialize fallback client if configured
        self.fallback_client: Optional[AsyncOpenAI] = None
        if self.fallback_url and self.fallback_api_key:
            # ZAI uses /api/coding/paas/v4 without /v1 prefix
            self.fallback_client = AsyncOpenAI(
                base_url=self.fallback_url,
                api_key=self.fallback_api_key,
                timeout=timeout,
            )
            logger.info(f"Initialized ZAI fallback client: {self.fallback_url}")
            logger.info(f"ZAI model fallback order: {self.zai_models}")

        # Initialize NVIDIA NIM client if configured
        self.nvidia_client: Optional[AsyncOpenAI] = None
        self.nvidia_url = nvidia_url
        if nvidia_url and nvidia_api_key:
            self.nvidia_client = AsyncOpenAI(
                base_url=nvidia_url,
                api_key=nvidia_api_key,
                timeout=timeout,
            )
            logger.info(f"Initialized NVIDIA NIM client: {nvidia_url}")

        # Initialize local backend client (e.g., sentry ROCm) if configured
        self.local_client: Optional[AsyncOpenAI] = None
        if local_backend_url and local_backend_url.strip():
            local_url = local_backend_url.rstrip("/")
            self.local_client = AsyncOpenAI(
                base_url=f"{local_url}/v1",
                api_key="not-needed",
                timeout=timeout,
            )
            logger.info(f"Initialized local backend client: {local_url}")

        # Initialize secondary backend client (e.g., 3060Ti) if configured
        self.secondary_client: Optional[AsyncOpenAI] = None
        if secondary_backend_url and secondary_backend_url.strip():
            sec_url = secondary_backend_url.rstrip("/")
            self.secondary_client = AsyncOpenAI(
                base_url=f"{sec_url}/v1",
                api_key="not-needed",
                timeout=timeout,
            )
            logger.info(f"Initialized secondary backend client: {sec_url}")

        # Build model -> client mapping for llama-cpp routing
        # Maps lowercase model name substrings to the appropriate client
        self.local_model_map: Dict[str, AsyncOpenAI] = {}
        if self.secondary_client:
            # Secondary backend (3060Ti) hosts SuperGemma
            self.local_model_map["supergemma"] = self.secondary_client
            if secondary_backend_model and secondary_backend_model.strip():
                self.local_model_map[secondary_backend_model.lower()] = self.secondary_client
        if self.local_client:
            # Local backend (sentry ROCm) hosts Qwen3.5-4B
            self.local_model_map["qwen3.5-4b"] = self.local_client
            if local_backend_model and local_backend_model.strip():
                self.local_model_map[local_backend_model.lower()] = self.local_client
        if self.local_model_map:
            logger.info(f"Local model routing map: {list(self.local_model_map.keys())} -> clients")

    async def chat_completion(
        self,
        messages: list[Dict[str, Any]],
        model: str,
        stream: bool = False,
        backend: Optional[str] = None,
        **kwargs,
    ):
        """
        Create chat completion with automatic failover.

        Args:
            messages: Chat messages
            model: Model name
            stream: Whether to stream response
            backend: Backend to use ("llama-cpp", "zai", or None for auto-detection)
            **kwargs: Additional OpenAI parameters
        Returns:
            ChatCompletion or AsyncStream of ChatCompletionChunk

        Raises:
            OpenAIBackendError: If all backends fail
        """
        # Remove 'stream' from kwargs to avoid duplicate parameter error
        kwargs.pop("stream", None)

        # Move chat_template_kwargs into extra_body for OpenAI SDK compatibility
        # llama-server supports this via --jinja but the Python SDK doesn't accept it as a top-level kwarg
        if "chat_template_kwargs" in kwargs:
            extra_body = kwargs.get("extra_body", {})
            extra_body["chat_template_kwargs"] = kwargs.pop("chat_template_kwargs")
            kwargs["extra_body"] = extra_body

        # Move enable_thinking into extra_body for OpenAI SDK compatibility
        # ZAI's API accepts this as a JSON field but the Python SDK rejects it as a typed kwarg
        if "enable_thinking" in kwargs:
            extra_body = kwargs.get("extra_body", {})
            extra_body["enable_thinking"] = kwargs.pop("enable_thinking")
            kwargs["extra_body"] = extra_body

        # Filter out parameters not supported by OpenAI SDK
        # These are used by llama.cpp/Qwen models but not supported by OpenAI Python SDK
        unsupported_params = [
            "top_k",          # llama.cpp sampling parameter
            "repeat_penalty", # Qwen-specific penalty
            "thinking",       # Qwen thinking mode flag
            "thinking_enabled", # Qwen thinking mode
            "supports_thinking_toggle", # Qwen capability flag
            "backend",        # Gateway routing parameter (not for SDK)
        ]
        for param in unsupported_params:
            kwargs.pop(param, None)

        # Strip 'think' from extra_body — Hermes sends think=false
        # when reasoning_effort=none, but NVIDIA NIM rejects it (HTTP 400)
        extra_body = kwargs.get("extra_body", {})
        if isinstance(extra_body, dict):
            extra_body.pop("think", None)
            if extra_body:
                kwargs["extra_body"] = extra_body
            else:
                kwargs.pop("extra_body", None)

        # If backend is specified, use it directly
        if backend == "zai" and self.fallback_client:
            logger.info(f"Using ZAI backend directly for model: {model}")
            try:
                response = await self.fallback_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    **kwargs,
                )
                # Strip markdown fences from non-streaming ZAI responses
                # Models like GLM wrap JSON in ```json fences, breaking consumers (Vane)
                if not stream and hasattr(response, 'choices'):
                    from ai_inference_gateway.utils import strip_markdown_json_fences
                    for choice in response.choices:
                        if choice.message and choice.message.content:
                            raw = choice.message.content
                            stripped = strip_markdown_json_fences(raw)
                            if stripped != raw:
                                logger.info(f"ZAI fence strip: removed markdown fences from response")
                                choice.message.content = stripped
                logger.info(f"ZAI backend succeeded with model: {model}")
                return response
            except Exception as e:
                logger.error(f"ZAI backend failed: {str(e)}")
                raise OpenAIBackendError(f"ZAI backend error: {str(e)}")
        elif backend == "llama-cpp":
            logger.info(f"Using llama.cpp backend directly for model: {model}")
            try:
                # Model-aware local routing: pick the right client based on model name
                client = self._resolve_local_client(model)
                response = await client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    **kwargs,
                )
                logger.info(f"llama.cpp backend succeeded with model: {model} via {client.base_url}")
                return response
            except Exception as e:
                logger.error(f"llama.cpp backend failed: {str(e)}")
                raise OpenAIBackendError(f"llama.cpp backend error: {str(e)}")
        elif backend == "nvidia" and self.nvidia_client:
            logger.info(f"Using NVIDIA NIM backend directly for model: {model}")
            try:
                response = await self.nvidia_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    **kwargs,
                )
                # Strip markdown fences from non-streaming NIM responses
                if not stream and hasattr(response, 'choices'):
                    from ai_inference_gateway.utils import strip_markdown_json_fences
                    for choice in response.choices:
                        if choice.message and choice.message.content:
                            raw = choice.message.content
                            stripped = strip_markdown_json_fences(raw)
                            if stripped != raw:
                                logger.info(f"NVIDIA fence strip: removed markdown fences from response")
                                choice.message.content = stripped
                logger.info(f"NVIDIA NIM backend succeeded with model: {model}")
                return response
            except Exception as e:
                logger.error(f"NVIDIA NIM backend failed: {str(e)}")
                raise OpenAIBackendError(f"NVIDIA NIM backend error: {str(e)}")
        elif backend == "pollinations":
            logger.info(f"Using Pollinations backend directly for model: {model}")
            try:
                response = await self.primary_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=stream,
                    **kwargs,
                )
                logger.info(f"Pollinations backend succeeded with model: {model}")
                return response
            except Exception as e:
                logger.error(f"Pollinations backend failed: {str(e)}")
                raise OpenAIBackendError(f"Pollinations backend error: {str(e)}")

        # Auto-detect: try primary backend first
        try:
            logger.info(
                f"Attempting primary backend: {self.primary_url} with model: {model}"
            )
            response = await self.primary_client.chat.completions.create(
                messages=messages,
                model=model,
                stream=stream,
                **kwargs,
            )
            logger.info(f"Primary backend succeeded with model: {model}")
            return response

        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Primary backend failed: {error_msg}")

            # Check if it's a connection error (should failover)
            # or an application error (should not failover)
            if self._should_failover(error_msg):
                if self.fallback_client:
                    # For streaming requests, try ZAI models with rotation
                    if stream:
                        last_error = None
                        for zai_model in [model] + [m for m in self.zai_models if m != model]:
                            try:
                                logger.info(
                                    f"Attempting ZAI streaming fallback with model: {zai_model}"
                                )
                                response = (
                                    await self.fallback_client.chat.completions.create(
                                        messages=messages,
                                        model=zai_model,
                                        stream=stream,
                                        **kwargs,
                                    )
                                )
                                logger.info(f"ZAI streaming fallback succeeded with model: {zai_model}")
                                return response
                            except Exception as fallback_error:
                                last_error = fallback_error
                                model_error_msg = str(fallback_error)
                                logger.warning(
                                    f"ZAI streaming model {zai_model} failed: {model_error_msg}"
                                )
                                if not self._should_try_next_model(model_error_msg):
                                    raise OpenAIBackendError(
                                        f"ZAI backend error: {model_error_msg}"
                                    )
                        raise OpenAIBackendError(
                            f"All ZAI streaming models failed. Last error: {str(last_error)}"
                        )
                    else:
                        # For non-streaming requests, try multiple ZAI models in sequence
                        last_error = None
                        for zai_model in self.zai_models:
                            try:
                                logger.info(
                                    f"Attempting ZAI fallback with model: {zai_model}"
                                )
                                response = (
                                    await self.fallback_client.chat.completions.create(
                                        messages=messages,
                                        model=zai_model,
                                        stream=stream,
                                        **kwargs,
                                    )
                                )
                                logger.info(
                                    f"ZAI fallback succeeded with model: {zai_model}"
                                )
                                return response
                            except Exception as model_error:
                                last_error = model_error
                                model_error_msg = str(model_error)
                                logger.warning(
                                    f"ZAI model {zai_model} failed: {model_error_msg}"
                                )

                                # Don't try more models if it's not a retryable error
                                if not self._should_try_next_model(model_error_msg):
                                    logger.error(
                                        f"ZAI model {zai_model} failed with non-retryable error, stopping fallback"
                                    )
                                    raise OpenAIBackendError(
                                        f"ZAI backend error: {model_error_msg}"
                                    )

                        # All models failed
                        logger.error(
                            f"All ZAI models failed. Last error: {str(last_error)}"
                        )
                        raise OpenAIBackendError(
                            f"All ZAI models failed. Last error: {str(last_error)}"
                        )
                else:
                    logger.warning("No fallback backend configured")
                    raise OpenAIBackendError(f"Primary backend failed: {error_msg}")
            else:
                # Application error (4xx/5xx) - don't failover
                raise OpenAIBackendError(f"Backend error: {error_msg}")

    def _should_failover(self, error_message: str) -> bool:
        """
        Determine if an error should trigger failover.

        Only connection errors should trigger failover, not application errors.
        This prevents cascading bad requests across all backends.

        Args:
            error_message: Error message string

        Returns:
            True if error should trigger failover
        """
        # Connection errors - should failover
        connection_errors = [
            "connect",
            "timeout",
            "connection refused",
            "connection reset",
            "host unreachable",
            "network unreachable",
            "all connection attempts failed",
        ]

        error_lower = error_message.lower()
        return any(err in error_lower for err in connection_errors)

    def _should_try_next_model(self, error_message: str) -> bool:
        """
        Determine if we should try the next ZAI model.

        Retryable errors that suggest trying a different model:
        - Rate limiting (429)
        - Model unloaded
        - Insufficient balance
        - Model-specific errors

        Non-retryable errors that should stop immediately:
        - Authentication errors (401)
        - Invalid request format (400)
        - Context length exceeded
        - Other client errors

        Args:
            error_message: Error message string

        Returns:
            True if we should try the next model
        """
        # Retryable errors - try next model
        retryable_errors = [
            "rate limit",  # 429 rate limiting
            "429",  # HTTP 429 status code
            "too many requests",  # Rate limiting message
            "model unloaded",  # Model not loaded
            "no models loaded",  # No models available
            "insufficient balance",  # Balance issues
            "error code: 400",  # ZAI returns 400 for model issues
            "unknown model",  # Model doesn't exist
        ]

        # Non-retryable errors - stop immediately
        non_retryable_errors = [
            "401",  # Authentication - won't help to switch models
            "unauthorized",  # Auth failure
            "invalid api key",  # Auth failure
            "context length",  # Request too long - won't help to switch models
            "maximum context length",  # Request too long
            "this model's maximum context length",  # Request too long
        ]

        error_lower = error_message.lower()

        # Check non-retryable first (these should stop immediately)
        if any(err in error_lower for err in non_retryable_errors):
            return False

        # Check retryable (these should try next model)
        return any(err in error_lower for err in retryable_errors)

    def _resolve_local_client(self, model: str) -> AsyncOpenAI:
        """
        Resolve the correct local backend client based on model name.

        Checks the local_model_map for substring matches against the model name.
        Falls back to primary_client if no match found.

        Args:
            model: Model name from the request

        Returns:
            AsyncOpenAI client to use for this model
        """
        model_lower = model.lower()
        for pattern, client in self.local_model_map.items():
            if pattern in model_lower:
                logger.info(f"Model '{model}' matched pattern '{pattern}' -> routed to {client.base_url}")
                return client
        # Default: use primary client (3090)
        return self.primary_client

    async def close(self):
        """Close all client connections."""
        await self.primary_client.close()
        if self.fallback_client:
            await self.fallback_client.close()
        if self.nvidia_client:
            await self.nvidia_client.close()
        if self.local_client:
            await self.local_client.close()
        if self.secondary_client:
            await self.secondary_client.close()


def create_openai_client(config) -> OpenAIClientWrapper:
    """
    Create OpenAI client wrapper from gateway configuration.

    Args:
        config: GatewayConfig instance

    Returns:
        OpenAIClientWrapper instance
    """
    # Get primary backend credentials
    primary_api_key = None
    if config.backend_type == "zai":
        primary_api_key = config.get_zai_api_key()
    elif config.backend_type == "pollinations":
        primary_api_key = config.get_pollinations_api_key()
    # llama-cpp doesn't require authentication

    # Get fallback backend credentials
    fallback_url = None
    fallback_api_key = None
    fallback_urls = config.get_backend_fallback_urls()
    if fallback_urls:
        fallback_url = fallback_urls[0]
        fallback_api_key = config.get_zai_api_key()

    # Get NVIDIA NIM credentials
    nvidia_api_key = config.get_nvidia_nim_api_key()
    nvidia_url = config.nvidia_nim_base_url if nvidia_api_key else None

    return OpenAIClientWrapper(
        primary_url=config.backend_url,
        primary_api_key=primary_api_key,
        fallback_url=fallback_url,
        fallback_api_key=fallback_api_key,
        nvidia_url=nvidia_url,
        nvidia_api_key=nvidia_api_key,
        local_backend_url=config.local_backend_url,
        local_backend_model=config.local_backend_model,
        secondary_backend_url=config.secondary_backend_url,
        secondary_backend_model=config.secondary_backend_model,
    )
