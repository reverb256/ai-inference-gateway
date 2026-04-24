# modules/services/ai-inference/ai_inference_gateway/config.py
"""
Configuration module for AI Inference Gateway using Pydantic for validation.

This module provides production-grade configuration with:
- Automatic environment variable loading
- Runtime validation
- Type coercion
- Secret field protection
- Schema generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration with validation"""

    enabled: bool = Field(default=False, description="Enable rate limiting")
    backend: str = Field(default="memory", pattern="^(memory|redis)$", description="Rate limit backend")
    tokens_per_minute: int = Field(default=10000, ge=0, description="Tokens per minute limit")
    tokens_per_hour: int = Field(default=50000, ge=0, description="Tokens per hour limit")
    tokens_per_day: int = Field(default=500000, ge=0, description="Tokens per day limit")
    rpm: int = Field(default=60, ge=1, le=10000, description="Requests per minute")

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        """Validate rate limit backend"""
        if v not in ["memory", "redis"]:
            raise ValueError('backend must be "memory" or "redis"')
        return v


class SecurityConfig(BaseModel):
    """Security filter configuration with validation"""

    enabled: bool = Field(default=True, description="Enable security filtering")
    pii_redaction: bool = Field(default=True, description="Enable PII redaction")
    max_request_size: int = Field(
        default=10485760,  # 10MB
        ge=1,
        le=104857600,  # Max 100MB
        description="Maximum request size in bytes",
    )

    @field_validator("max_request_size")
    @classmethod
    def validate_size(cls, v: int) -> int:
        """Ensure reasonable size limits"""
        if v < 1024:  # 1KB minimum
            raise ValueError("max_request_size must be at least 1KB")
        return v


class CacheConfig(BaseModel):
    """Cache configuration with validation"""

    enabled: bool = Field(default=False, description="Enable caching")
    backend: str = Field(default="memory", pattern="^(memory|redis)$", description="Cache backend")
    default_ttl: int = Field(
        default=3600,  # 1 hour
        ge=1,
        le=86400,  # Max 24 hours
        description="Default cache TTL in seconds",
    )


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration with validation"""

    enabled: bool = Field(default=True, description="Enable circuit breaker")
    failure_threshold: int = Field(default=5, ge=1, le=100, description="Number of failures before opening circuit")
    success_threshold: int = Field(default=2, ge=1, le=10, description="Number of successes before closing circuit")
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description="Seconds to wait before trying half-open state",
    )
    health_check_interval: int = Field(default=10, ge=5, le=60, description="Seconds between health checks")


class RequestQueueConfig(BaseModel):
    """Request queue configuration with validation"""

    enabled: bool = Field(default=False, description="Enable request queuing")
    max_concurrent: int = Field(default=10, ge=1, le=1000, description="Maximum concurrent requests")


class LoadBalancerConfig(BaseModel):
    """Load balancer configuration"""

    enabled: bool = Field(default=False, description="Enable load balancing")
    # Backend weights are configured dynamically


class ConcurrencyLimiterConfig(BaseModel):
    """Concurrency limiter configuration"""

    enabled: bool = Field(default=False, description="Enable concurrency limiting")
    max_concurrency: int = Field(default=1, ge=1, le=100, description="Maximum concurrent requests per model")


class KnowledgeFabricConfig(BaseSettings):
    """Knowledge Fabric middleware configuration"""

    model_config = SettingsConfigDict(
        env_prefix="",  # No prefix for env vars
        case_sensitive=False,
        extra="ignore",
    )

    enabled: bool = Field(default=False, description="Enable Knowledge Fabric middleware")
    rrf_k: int = Field(default=60, ge=1, le=100, description="RRF constant for fusion")
    rag_enabled: bool = Field(default=False, description="Enable RAG source")
    code_search_enabled: bool = Field(default=True, description="Enable code search source")
    searxng_enabled: bool = Field(default=False, description="Enable SearXNG source")
    web_search_enabled: bool = Field(default=False, description="Enable MCP web search source")
    code_search_paths: List[str] = Field(default_factory=lambda: ["/etc/nixos"], description="Paths to search for code")
    rag_top_k: int = Field(default=5, ge=1, le=20, description="RAG top-K results")
    searxng_url: str = Field(default="http://10.4.98.141:7777", description="SearXNG URL")
    mcp_url: str = Field(default="http://127.0.0.1:8080/mcp/call", description="MCP broker URL")
    web_max_results: int = Field(default=5, ge=1, le=20, description="Web search max results")
    searxng_max_results: int = Field(default=5, ge=1, le=20, description="SearXNG max results")
    code_max_results: int = Field(default=5, ge=1, le=20, description="Code search max results")
    brain_wiki_enabled: bool = Field(default=True, description="Enable brain wiki source")
    brain_wiki_path: str = Field(
        default=str(Path.home() / "brain" / "wiki"),
        description="Path to brain wiki directory",
    )
    brain_wiki_max_results: int = Field(default=5, ge=1, le=20, description="Brain wiki max results")
    brain_wiki_max_chunk_chars: int = Field(
        default=2000,
        ge=100,
        le=10000,
        description="Max chars per brain wiki chunk",
    )


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""

    name: str = Field(..., description="Server name")
    type: str = Field(
        default="local",
        pattern="^(local|remote)$",
        description="Server type (local or remote)",
    )
    command: Optional[List[str]] = Field(default=None, description="Command for local servers")
    url: Optional[str] = Field(default=None, description="URL for remote servers")
    headers: Dict[str, str] = Field(default_factory=dict, description="HTTP headers for remote servers")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables for local servers")


class MCPConfig(BaseModel):
    """MCP broker configuration."""

    enabled: bool = Field(default=False, description="Enable MCP broker")
    servers: List[MCPServerConfig] = Field(default_factory=list, description="Configured MCP servers")
    # Exclude servers from env var parsing - we handle it manually in the validator
    model_config = SettingsConfigDict(extra="ignore")


class SystemPromptsConfig(BaseModel):
    """System prompts configuration for different request types."""

    enabled: bool = Field(default=False, description="Enable custom system prompts")
    default: str = Field(default="", description="Default system prompt for all requests")
    coding: str = Field(
        default="You are an expert coding assistant. Write clean, efficient, and well-documented code.",
        description="System prompt for coding-related requests",
    )
    reasoning: str = Field(
        default="You are an expert reasoning assistant. Think step-by-step and provide clear explanations.",
        description="System prompt for reasoning-related requests",
    )
    analysis: str = Field(
        default="You are an expert analysis assistant. Provide thorough and structured analysis.",
        description="System prompt for analysis-related requests",
    )
    agentic: str = Field(
        default="You are an autonomous agent capable of multi-step planning and execution.",
        description="System prompt for agentic/workflow requests",
    )
    fast: str = Field(
        default="You are a fast and efficient assistant. Provide concise, direct answers.",
        description="System prompt for fast response requests",
    )
    custom: Dict[str, str] = Field(default_factory=dict, description="Custom system prompts by name")

    def get_prompt(self, category: str) -> Optional[str]:
        """
        Get system prompt for a specific category.

        Args:
            category: One of 'default', 'coding', 'reasoning', 'analysis', 'agentic', 'fast', or custom name

        Returns:
            System prompt string or None if not found
        """
        if not self.enabled:
            return None

        # Check built-in categories
        if hasattr(self, category):
            value = getattr(self, category)
            if value:
                return value

        # Check custom prompts
        if category in self.custom and self.custom[category]:
            return self.custom[category]

        # Fall back to default
        if self.default:
            return self.default

        return None


class SentryConfig(BaseModel):
    """Sentry error tracking configuration."""

    enabled: bool = Field(default=False, description="Enable Sentry error tracking")
    dsn: Optional[str] = Field(default=None, repr=False, exclude=True, description="Sentry DSN")
    dsn_file: Optional[str] = Field(default=None, description="Path to file containing Sentry DSN")
    environment: str = Field(
        default="production",
        pattern="^(development|staging|production)$",
        description="Sentry environment",
    )
    traces_sample_rate: float = Field(default=0.1, ge=0.0, le=1.0, description="Sample rate for performance tracing")

    def get_dsn(self) -> Optional[str]:
        """
        Get Sentry DSN value.

        Priority:
        1. Environment variable SENTRY_DSN
        2. File specified in SENTRY_DSN_FILE
        """
        # Try direct value first
        if self.dsn:
            return self.dsn

        # Try file
        if self.dsn_file:
            try:
                with open(self.dsn_file, "r") as f:
                    return f.read().strip()
            except Exception:
                return None

        return None


class ObservabilityConfig(BaseModel):
    """Observability and logging configuration with validation"""

    enabled: bool = Field(default=True, description="Enable observability")
    structured_logging: bool = Field(default=True, description="Use structured JSON logging")
    request_id_header: str = Field(
        default="X-Request-ID",
        min_length=1,
        max_length=100,
        description="Header name for request ID tracking",
    )
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Logging level",
    )
    log_format: str = Field(default="json", pattern="^(json|text)$", description="Log output format")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper


class MiddlewareConfig(BaseSettings):
    """Complete middleware configuration - inherits BaseSettings for env var support"""

    model_config = SettingsConfigDict(
        env_prefix="",  # No prefix for env vars
        case_sensitive=False,
        extra="ignore",
    )

    rate_limiting: RateLimitingConfig = Field(
        default_factory=RateLimitingConfig, description="Rate limiting configuration"
    )
    security: SecurityConfig = Field(default_factory=SecurityConfig, description="Security filter configuration")
    cache: CacheConfig = Field(default_factory=CacheConfig, description="Cache configuration")
    circuit_breaker: CircuitBreakerConfig = Field(
        default_factory=CircuitBreakerConfig,
        description="Circuit breaker configuration",
    )
    request_queue: RequestQueueConfig = Field(
        default_factory=RequestQueueConfig, description="Request queue configuration"
    )
    load_balancer: LoadBalancerConfig = Field(
        default_factory=LoadBalancerConfig, description="Load balancer configuration"
    )
    concurrency_limiter: ConcurrencyLimiterConfig = Field(
        default_factory=ConcurrencyLimiterConfig,
        description="Concurrency limiter configuration",
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig, description="Observability configuration"
    )
    mcp: MCPConfig = Field(default_factory=MCPConfig, description="MCP broker configuration")

    knowledge_fabric: KnowledgeFabricConfig = Field(
        default_factory=KnowledgeFabricConfig,
        description="Knowledge Fabric middleware configuration",
    )

    # Override knowledge_fabric using model_validator to parse env vars
    @model_validator(mode="after")
    def override_knowledge_fabric_from_env(self):
        """Parse Knowledge Fabric env vars that use MIDDLEWARE__ prefix"""
        import os
        import json

        # Read env vars with MIDDLEWARE__KNOWLEDGE_FABRIC__ prefix
        enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__ENABLED", "").lower()
        rrf_k = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__RRF_K", "")
        rag_enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__RAG_ENABLED", "").lower()
        searxng_enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__SEARXNG_ENABLED", "").lower()
        searxng_url = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__SEARXNG_URL", "http://10.4.98.141:7777")
        code_search_enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__CODE_SEARCH_ENABLED", "").lower()
        web_search_enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__WEB_SEARCH_ENABLED", "").lower()
        code_search_paths_raw = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__CODE_SEARCH_PATHS", '["/etc/nixos"]')
        rag_top_k = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__RAG_TOP_K", "5")
        searxng_max_results = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__SEARXNG_MAX_RESULTS", "5")
        code_max_results = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__CODE_MAX_RESULTS", "5")
        web_max_results = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__WEB_MAX_RESULTS", "5")

        # Parse JSON paths
        try:
            code_search_paths = json.loads(code_search_paths_raw)
        except:
            code_search_paths = ["/etc/nixos"]

        # Override knowledge_fabric fields
        self.knowledge_fabric.enabled = enabled == "true"
        self.knowledge_fabric.rrf_k = int(rrf_k) if rrf_k else 60
        self.knowledge_fabric.rag_enabled = rag_enabled == "true"
        self.knowledge_fabric.searxng_enabled = searxng_enabled == "true"
        self.knowledge_fabric.searxng_url = searxng_url
        self.knowledge_fabric.code_search_enabled = code_search_enabled == "true"
        self.knowledge_fabric.web_search_enabled = web_search_enabled == "true"
        self.knowledge_fabric.code_search_paths = code_search_paths
        self.knowledge_fabric.rag_top_k = int(rag_top_k) if rag_top_k else 5
        self.knowledge_fabric.searxng_max_results = int(searxng_max_results) if searxng_max_results else 5
        self.knowledge_fabric.code_max_results = int(code_max_results) if code_max_results else 5
        self.knowledge_fabric.web_max_results = int(web_max_results) if web_max_results else 5

        return self

    # Override mcp using model_validator to parse env vars
    @model_validator(mode="after")
    def override_mcp_from_env(self):
        """Parse MCP env vars"""
        import os
        import json

        # Read MCP_ENABLED env var
        mcp_enabled = os.environ.get("MCP_ENABLED", "").lower()

        # Override mcp.enabled field
        if mcp_enabled:
            self.mcp.enabled = mcp_enabled == "true"

        # Read MCP_SERVERS env var (JSON string)
        mcp_servers_json = os.environ.get("MCP_SERVERS", "[]")
        if mcp_servers_json and mcp_servers_json != "[]":
            try:
                servers_data = json.loads(mcp_servers_json)
                # Convert dict to MCPServerConfig objects
                # The dict key is the server name, value contains the config
                servers = []
                for server_name, server_config in servers_data.items():
                    if not server_config.get("enabled", True):
                        continue

                    # Extract fields explicitly to avoid unexpected kwargs
                    server_type = server_config.get("type", "local")
                    command = server_config.get("command")
                    url = server_config.get("url")
                    headers = server_config.get("headers", {})
                    environment = server_config.get("environment", {})

                    # Create MCPServerConfig with explicit fields
                    server = MCPServerConfig(
                        name=server_name,
                        type=server_type,
                        command=command,
                        url=url,
                        headers=headers,
                        environment=environment,
                    )
                    servers.append(server)

                self.mcp.servers = servers
            except Exception as e:
                import logging

                logging.warning(f"Failed to parse MCP_SERVERS JSON: {e}")

        return self

    # RAG configuration (optional - loaded from environment variables)
    # These use the exact env var names from gateway.nix for compatibility
    RAG_ENABLED: bool = Field(default=False, description="Enable RAG functionality")
    QDRANT_URL: str = Field(default="http://127.0.0.1:6333", description="Qdrant URL")
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-m3", description="Embedding model")
    CHUNK_SIZE: int = Field(default=512, description="Chunk size")
    CHUNK_OVERLAP: int = Field(default=50, description="Chunk overlap")
    RAG_TOP_K: int = Field(default=5, description="Default top-K results")
    HYBRID_SEARCH_ENABLED: bool = Field(default=True, description="Enable hybrid search")
    RERANKER_ENABLED: bool = Field(default=True, description="Enable reranking")


class GatewayConfig(BaseSettings):
    """
    Main gateway configuration with automatic environment variable loading.

    Environment variables are automatically loaded:
    - GATEWAY_HOST: Gateway listen host
    - GATEWAY_PORT: Gateway listen port
    - BACKEND_URL: Backend service URL
    - BACKEND_TYPE: Backend type (llama-cpp, vllm, sglang, zai, pollinations)
    - ZAI_API_KEY: ZAI API key (or ZAI_API_KEY_FILE)
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - STRUCTURED_LOGGING: Enable structured logging (true/false)
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # No prefix for env vars
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )

    # Core settings with validation
    gateway_host: str = Field(default="127.0.0.1", description="Gateway listen host")
    gateway_port: int = Field(default=8080, ge=1, le=65535, description="Gateway listen port")

    # Backend settings with validation
    backend_url: str = Field(default="http://127.0.0.1:1234", description="Primary backend service URL")
    backend_fallback_urls: str = Field(default="", description="Fallback backend URLs (comma-separated)")
    backend_type: str = Field(
        default="llama-cpp",
        pattern="^(llama-cpp|vllm|sglang|zai|pollinations|nvidia-nim)$",
        description="Primary backend type",
    )

    # Local backend (always-on llama-cpp)
    local_backend_url: str = Field(default="http://127.0.0.1:1235", description="Local llama-cpp backend URL")
    local_backend_model: str = Field(default="gemma-4-e2b-it", description="Default model on local backend")

    # Secondary local backend (3060Ti)
    secondary_backend_url: str = Field(default="", description="Secondary local backend URL (e.g., 3060Ti)")
    secondary_backend_model: str = Field(default="", description="Model name on secondary backend")

    # NVIDIA NIM backend
    nvidia_nim_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1", description="NVIDIA NIM API base URL"
    )
    nvidia_nim_api_key: Optional[SecretStr] = Field(
        default=None, repr=False, exclude=True, description="NVIDIA NIM API key"
    )
    nvidia_nim_api_key_file: Optional[str] = Field(
        default=None, description="Path to file containing NVIDIA NIM API key"
    )

    def get_backend_fallback_urls(self) -> list[str]:
        """Get backend fallback URLs as a list."""
        if not self.backend_fallback_urls:
            return []
        return [url.strip() for url in self.backend_fallback_urls.split(",") if url.strip()]

    def get_backend_fallback_types(self) -> list[str]:
        """Get backend fallback types inferred from URLs."""
        fallback_urls = self.get_backend_fallback_urls()
        fallback_types = []
        for url in fallback_urls:
            url_lower = url.lower()
            if "api.z.ai" in url_lower or "zai" in url_lower:
                fallback_types.append("zai")
            elif "nvidia.com" in url_lower or "api.nvidia" in url_lower:
                fallback_types.append("nvidia-nim")
            elif "pollinations" in url_lower:
                fallback_types.append("pollinations")
            elif "vllm" in url_lower:
                fallback_types.append("vllm")
            elif "sglang" in url_lower:
                fallback_types.append("sglang")
            else:
                # Default to llama-cpp for local URLs
                fallback_types.append("llama-cpp")
        return fallback_types

    def get_nvidia_nim_api_key(self) -> Optional[str]:
        """Get NVIDIA NIM API key from file or direct value."""
        if self.nvidia_nim_api_key_file:
            try:
                with open(self.nvidia_nim_api_key_file) as f:
                    return f.read().strip()
            except Exception:
                return None
        if self.nvidia_nim_api_key:
            return self.nvidia_nim_api_key.get_secret_value()
        return None

    # API Keys (marked as secrets - won't appear in logs or repr)
    zai_api_key: Optional[SecretStr] = Field(default=None, repr=False, exclude=True, description="ZAI API key")
    zai_api_key_file: Optional[str] = Field(default=None, description="Path to file containing ZAI API key")

    pollinations_api_key: Optional[SecretStr] = Field(
        default=None, repr=False, exclude=True, description="Pollinations API key"
    )
    pollinations_api_key_file: Optional[str] = Field(
        default=None, description="Path to file containing Pollinations API key"
    )

    # Middleware configuration
    middleware: MiddlewareConfig = Field(default=MiddlewareConfig(), description="Middleware configuration")

    # Sentry error tracking configuration
    sentry: SentryConfig = Field(default_factory=SentryConfig, description="Sentry error tracking configuration")

    # System prompts configuration
    system_prompts: SystemPromptsConfig = Field(
        default_factory=SystemPromptsConfig, description="System prompts configuration"
    )

    @field_validator("backend_url")
    @classmethod
    def validate_backend_url(cls, v: str) -> str:
        """Ensure backend URL is valid"""
        if not v.startswith(("http://", "https://")):
            raise ValueError("backend_url must start with http:// or https://")
        return v

    @field_validator("gateway_host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """Validate host address"""
        if not v:
            raise ValueError("gateway_host cannot be empty")
        return v

    def get_zai_api_key(self) -> Optional[str]:
        """
        Get ZAI API key value.

        Priority:
        1. Environment variable ZAI_API_KEY
        2. File specified in ZAI_API_KEY_FILE
        """
        # Try secret field first (but not empty strings)
        if self.zai_api_key:
            value = self.zai_api_key.get_secret_value()
            if value and value.strip():
                return value

        # Try file
        if self.zai_api_key_file:
            try:
                with open(self.zai_api_key_file, "r") as f:
                    return f.read().strip()
            except Exception:
                return None

        return None

    def get_pollinations_api_key(self) -> Optional[str]:
        """
        Get Pollinations API key value.

        Priority:
        1. Environment variable POLLINATIONS_API_KEY
        2. File specified in POLLINATIONS_API_KEY_FILE
        """
        # Try secret field first (but not empty strings)
        if self.pollinations_api_key:
            value = self.pollinations_api_key.get_secret_value()
            if value and value.strip():
                return value

        # Try file
        if self.pollinations_api_key_file:
            try:
                with open(self.pollinations_api_key_file, "r") as f:
                    return f.read().strip()
            except Exception:
                return None

        return None
