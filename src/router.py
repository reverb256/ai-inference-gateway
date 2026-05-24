"""
Intelligent Router for AI Inference Gateway.

Routes requests to appropriate models based on:
- Token count estimation
- Task type detection (coding, agentic, general, fast, large_context)
- Latency tracking and overload detection
- Model specialization matching
- Cost tier considerations
- Category-based routing (inspired by oh-my-opencode)
- Autonomous model selection based on benchmarks
"""

from .contexts import LLAMA_SERVER_CONTEXT, CLOUD_MODEL_CONTEXT, QWEN_FAMILY_CONTEXT, MAX_OUTPUT_TOKENS, get_context_length, get_max_tokens
from .model_benchmark import ModelBenchmark, AutonomousModelSelector, get_benchmark, get_selector
from .provider_rate_limiter import check_provider_limit, record_provider_429, get_provider_stats
import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Token threshold for forcing local routing.
# Cloud providers (ZAI, NIM) timeout around 60-90s on large requests.
# Local llama.cpp has no API timeout — just KV cache limits.
# Requests exceeding this threshold get a cloud penalty to prefer local.
TOKEN_THRESHOLD_LOCAL=60000  # tokens


# Prefill optimization config for faster TTFT on base models
# Extended context variants get full context, base models get aggressive limits
MODEL_PREFILL_CONFIG = {
    # Haiku - no base/extended distinction, just fast
    "claude-haiku-4": {
        "max_input_tokens": 30_000,
        "max_history_messages": 10,
    },
    "claude-haiku-4-20250514": {
        "max_input_tokens": 30_000,
        "max_history_messages": 10,
    },
    # Sonnet base (200K equivalent) - aggressive limits for fast prefill
    "claude-sonnet-4-20250514": {
        "max_input_tokens": 50_000,
        "max_history_messages": 20,
    },
    # Sonnet extended (256K equivalent) - full context
    "claude-sonnet-4-20250514-1m": {
        "max_input_tokens": 200_000,
        "max_history_messages": None,  # No trimming
    },
    # Opus base (200K equivalent) - aggressive limits for fast prefill
    "claude-opus-4-20250514": {
        "max_input_tokens": 50_000,
        "max_history_messages": 20,
    },
    # Opus extended (256K equivalent) - full context
    "claude-opus-4-20250514-1m": {
        "max_input_tokens": 256_000,
        "max_history_messages": None,  # No trimming
    },
}


# Qwen3.5 model-specific configuration
# Based on: https://unsloth.ai/docs/models/qwen3.5
QWEN_MODEL_CONFIG = {
    # ========== ULTRA-TINY MODELS (0.8B-2B) ==========
    # Fastest models, suitable for simple tasks and quick responses
    "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 8192,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "simple_qa"],
    },
    "qwen3.5-0.8b": {
        "max_tokens": 8192,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "simple_qa"],
    },
    "qwen3.5-2b": {
        "max_tokens": 8192,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "summarization"],
    },
    "qwen3.5-2b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 8192,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "fast",
        "recommended_for": ["fast", "reasoning"],
    },

    # ========== SMALL MODELS (4B-9B) ==========
    # Good balance of speed and quality
    "qwen3.5-4b": {
        "max_tokens": 8192,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "fast",
        "recommended_for": ["general", "chat"],
    },
    "qwen3.5-4b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 16384,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "fast",
        "recommended_for": ["coding", "reasoning"],
    },
    "qwen3.5-4b-claude-4.6-opus-distilled-32k@q8_0": {
        "max_tokens": 16384,
        "context_length": LLAMA_SERVER_CONTEXT["qwen3.5-0.8b"],
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "fast",
        "recommended_for": ["general", "chat"],
    },
    "qwen3.5-9b": {
        "max_tokens": 16384,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "balanced",
        "recommended_for": ["general", "coding"],
    },
    "qwen3.5-9b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 16384,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "balanced",
        "recommended_for": ["coding", "complex_reasoning"],
    },
    "qwen3.5-9b-claude-4.6-opus-distilled-32k": {
        "max_tokens": 16384,
        "context_length": LLAMA_SERVER_CONTEXT["qwen3.5-0.8b"],
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "balanced",
        "recommended_for": ["general", "long_context"],
    },

    # ========== LARGE MODELS (27B-35B) ==========
    # Highest quality, slower but more capable
    "qwen3.5-27b": {
        "max_tokens": 32768,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "slow",
        "recommended_for": ["complex_reasoning", "creative"],
    },
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 32768,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "slow",
        "recommended_for": ["complex_reasoning", "agentic"],
    },
    "qwen3.5-35b-a3b": {
        "max_tokens": 32768,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "slow",
        "recommended_for": ["complex_reasoning", "agentic", "analysis"],
    },
}


# Optimal parameters for Qwen3.5 thinking vs non-thinking modes
# Based on: https://unsloth.ai/docs/models/qwen3.5
# Note: Only includes parameters supported by llama.cpp's OpenAI-compatible API
QWEN_OPTIMAL_PARAMS = {
    "thinking": {
        # General purpose with thinking enabled
        "general": {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 1.5,
        },
        # Coding tasks with thinking
        "coding": {
            "temperature": 0.7,
            "top_p": 0.8,
            "presence_penalty": 1.5,
        },
        # Agentic tasks (tool-calling, multi-step reasoning)
        "agentic": {
            "temperature": 0.8,
            "top_p": 0.9,
            "presence_penalty": 1.2,
        },
        # Fast responses (still with thinking enabled but lower temp)
        "fast": {
            "temperature": 0.5,
            "top_p": 0.85,
            "presence_penalty": 0.5,
        },
    },
    "non_thinking": {
        # General purpose without thinking
        "general": {
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 0.0,
        },
        # Heavy reasoning without explicit thinking mode
        "reasoning": {
            "temperature": 1.0,
            "top_p": 0.95,
            "presence_penalty": 0.0,
        },
        # Coding without thinking mode
        "coding": {
            "temperature": 0.5,
            "top_p": 0.85,
            "presence_penalty": 0.0,
        },
        # Agentic tasks without thinking
        "agentic": {
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.8,
        },
        # Fast responses
        "fast": {
            "temperature": 0.4,
            "top_p": 0.8,
            "presence_penalty": 0.0,
        },
    },
}


def get_qwen_model_config(model_id: str) -> Dict:
    """Get Qwen model configuration, with fallback to defaults."""
    model_key = model_id.split("/")[-1] if "/" in model_id else model_id
    return QWEN_MODEL_CONFIG.get(model_key, {
        "max_tokens": 4096,
        "context_length": QWEN_FAMILY_CONTEXT,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
    })


def get_optimal_qwen_params(
    model_id: str = "",
    thinking_enabled: bool = False,
    task_type: str = "general",
) -> Dict:
    """
    Get optimal parameters for Qwen3.5 models.

    Args:
        model_id: Qwen model identifier
        thinking_enabled: Whether thinking/reasoning mode is enabled
        task_type: Task type ("general", "coding", "agentic", "fast", or "reasoning")

    Returns:
        Dict of optimal parameters for the model and task type
    """
    mode = "thinking" if thinking_enabled else "non_thinking"
    valid_tasks = {"general", "coding", "agentic", "fast", "reasoning"}
    task = task_type if task_type in valid_tasks else "general"
    return QWEN_OPTIMAL_PARAMS.get(mode, {}).get(task, {})
# Context windows live in contexts.py — single source of truth.
# from .contexts import LLAMA_SERVER_CONTEXT, CLOUD_MODEL_CONTEXT





class TaskSpecialization(Enum):
    """Task specialization types for intelligent routing."""

    CODING = "coding"
    AGENTIC = "agentic"
    GENERAL = "general"
    FAST = "fast"
    LARGE_CONTEXT = "large_context"
    VISION = "vision"
    REASONING = "reasoning"
    CLASSIFICATION = "classification"
    ROUTING = "routing"


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    context_length: int = 131072  # Safe default; use get_context_length() for actual
    priority: int = 0
    specializations: List[TaskSpecialization] = field(default_factory=list)
    cost_tier: int = 1
    estimated_tokens_per_second: float = 50.0
    backend: str = "llama-cpp"  # llama-cpp, zai, etc.


@dataclass
class ModelCandidate:
    """Candidate model for reranking."""

    model: str
    backend: str
    score: float
    reason: str
    specialization: TaskSpecialization
    expected_latency_ms: float


@dataclass
class RouteDecision:
    """Routing decision with metadata."""

    model: str
    confidence: float
    reason: str
    estimated_tokens: int
    backend: str
    specialization: Optional[TaskSpecialization] = None
    expected_latency_ms: Optional[float] = None


class LatencyTracker:
    """Track model response times for latency-aware routing."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies: Dict[str, List[float]] = {}
        self._lock = asyncio.Lock()

    async def record_latency(self, model: str, latency_ms: float):
        """Record a latency measurement."""
        async with self._lock:
            if model not in self.latencies:
                self.latencies[model] = []
            self.latencies[model].append(latency_ms)
            if len(self.latencies[model]) > self.window_size:
                self.latencies[model] = self.latencies[model][-self.window_size :]

    async def get_avg_latency(self, model: str) -> Optional[float]:
        """Get average latency for a model."""
        async with self._lock:
            if model not in self.latencies or not self.latencies[model]:
                return None
            return sum(self.latencies[model]) / len(self.latencies[model])

    async def is_overloaded(self, model: str, threshold_ms: float = 5000.0) -> bool:
        """Check if a model is overloaded based on recent latencies."""
        avg = await self.get_avg_latency(model)
        return avg is not None and avg > threshold_ms


class Router:
    """
    Intelligent router for model selection.

    Analyzes requests and routes to appropriate models based on
    token count, task type, and current model performance.

    Now with auto-discovery of llama-server and LM Studio models!
    """

    def __init__(
        self,
        models: List[ModelInfo],
        latency_tracker: Optional[LatencyTracker] = None,
        model_discovery=None,
        cloud_discovery=None,
    ):
        """
        Initialize router.

        Args:
            models: List of available models
            latency_tracker: Optional latency tracker for performance-based routing
            model_discovery: Optional ModelDiscovery instance for auto-discovery
            cloud_discovery: Optional CloudModelRegistry for cloud model lookup
        """
        self.models = {model.id: model for model in models}
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.model_discovery = model_discovery
        self.cloud_discovery = cloud_discovery
        self.claude_model_mapping = self._build_claude_mapping()
        # Active request tracking for smart load balancing
        self.active_requests: Dict[str, Dict] = (
            {}
        )  # request_id -> {model, backend, stream, start_time}
        self.max_concurrent_streams = 1  # Backend can handle 1 stream at a time

        # Per-backend concurrency semaphores — 1 request per GPU at a time
        self.backend_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._init_backend_semaphores()

        # Round-robin counter for local backend selection
        self._rr_index: int = 0

        # Backend health cache
        self._backend_health: Dict[str, bool] = {
            "llama-cpp": True,
            "vllm-local": True,
            "zai": True,
            "nvidia": True,
            "kilo": True,
            "openrouter": True,
            "opencode-go": True,
            "opencode-zen": True,
        }
        self._backend_health_check_time: Dict[str, float] = {}
        self._health_check_ttl: float = 10.0  # Check health every 10 seconds

    # Backend health check — uses configured BACKEND_URL from env
    BACKEND_PORTS = {
        "llama-cpp": 1235,  # Updated: local llama-cpp on 3060 Ti
        "llama-server": 1235,
    }

    async def get_backend_load(self, backend: str) -> Dict:
        """
        Get current load on a backend.

        Args:
            backend: Backend name (llama-cpp, zai, etc.)

        Returns:
            Dict with load information
        """
        active = sum(
            1 for r in self.active_requests.values() if r.get("backend") == backend
        )
        is_streaming = any(
            r.get("stream")
            for r in self.active_requests.values()
            if r.get("backend") == backend
        )
        return {
            "backend": backend,
            "active_requests": active,
            "is_streaming": is_streaming,
            "at_capacity": active >= self.max_concurrent_streams,
        }

    def track_request_start(
        self, request_id: str, model: str, backend: str, stream: bool
    ):
        """Track the start of a request."""
        import time

        self.active_requests[request_id] = {
            "model": model,
            "backend": backend,
            "stream": stream,
            "start_time": time.time(),
        }
        logger.debug(
            f"Tracking request {request_id}: model={model}, backend={backend}, stream={stream}"
        )

    def track_request_end(self, request_id: str):
        """Track the end of a request."""
        if request_id in self.active_requests:
            del self.active_requests[request_id]
            logger.debug(f"Stopped tracking request {request_id}")

    def _init_backend_semaphores(self):
        """Initialize per-backend concurrency semaphores from discovery."""
        if self.model_discovery:
            for name in self.model_discovery.BACKENDS:
                self.backend_semaphores[name] = asyncio.Semaphore(1)
                logger.info(f"Initialized semaphore for backend: {name}")

        # Initialize vllm-local semaphore (not in model discovery)
        if "vllm-local" not in self.backend_semaphores:
            self.backend_semaphores["vllm-local"] = asyncio.Semaphore(1)
            logger.info("Initialized semaphore for backend: vllm-local")

    def is_backend_busy(self, backend_name: str) -> bool:
        """Check if a backend is currently processing a request."""
        sem = self.backend_semaphores.get(backend_name)
        if sem is None:
            return False  # Unknown backend — let it through
        return sem._value == 0

    async def acquire_backend(self, backend_name: str) -> bool:
        """
        Try to acquire a backend for exclusive use. Non-blocking.
        Returns True if acquired, False if busy.
        """
        sem = self.backend_semaphores.get(backend_name)
        if sem is None:
            logger.debug(f"No semaphore for {backend_name}, allowing request")
            return True
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.01)
            logger.info(f"Acquired backend: {backend_name}")
            return True
        except asyncio.TimeoutError:
            logger.info(f"Backend busy: {backend_name}")
            return False

    def release_backend(self, backend_name: str):
        """Release a backend after request completes."""
        sem = self.backend_semaphores.get(backend_name)
        if sem is not None:
            sem.release()
            logger.info(f"Released backend: {backend_name}")

    def get_idle_local_backend(self) -> Optional[str]:
        """
        Get the next idle local backend using round-robin.
        Returns backend name or None if all busy.
        """
        if not self.backend_semaphores:
            return None

        backends = list(self.backend_semaphores.keys())
        # Try each backend starting from rr_index
        for i in range(len(backends)):
            idx = (self._rr_index + i) % len(backends)
            name = backends[idx]
            if not self.is_backend_busy(name):
                self._rr_index = (idx + 1) % len(backends)
                return name
        return None

    def get_backend_status(self) -> Dict[str, Dict]:
        """Get status of all local backends."""
        status = {}
        for name, sem in self.backend_semaphores.items():
            active = sem._value == 0
            model = "unknown"
            if self.model_discovery:
                backend_info = self.model_discovery.BACKENDS.get(name)
                if backend_info and backend_info.models:
                    model = backend_info.models[0]
            status[name] = {
                "active": active,
                "model": model,
            }
        return status

    def get_backend_for_model(self, model_id: str) -> Optional[str]:
        """
        Get the backend name for a given model ID.

        Uses model discovery for auto-discovered models, falls back to
        static backend mapping for cloud models.

        Args:
            model_id: Model ID to look up

        Returns:
            Backend name (llama-3090, llama-3060ti, zai, nvidia, etc.)
        """
        # Try model discovery first (for llama-servers, LM Studio)
        if self.model_discovery:
            backend = self.model_discovery.get_backend_for_model(model_id)
            if backend:
                logger.debug(f"Model {model_id} → discovered backend {backend}")
                return backend

        # Fallback to static backend mapping (cloud models)
        model_info = self.models.get(model_id)
        if model_info:
            return model_info.backend

        # Default to llama-cpp for unknown models
        logger.warning(f"Unknown model {model_id}, defaulting to llama-cpp backend")
        return "llama-cpp"

    def get_backend_url(self, model_id: str) -> Optional[str]:
        """
        Get the backend URL for a given model ID.

        Args:
            model_id: Model ID to look up

        Returns:
            Full backend URL (http://host:port/v1)
        """
        backend = self.get_backend_for_model(model_id)

        # Use model discovery for local backends
        if self.model_discovery and backend in ["llama-3090", "llama-3060ti", "llama-sentry", "lmstudio"]:
            return self.model_discovery.get_backend_url(backend)

        # Fallback to static BACKEND_PORTS for legacy compatibility
        port = self.BACKEND_PORTS.get(backend)
        if port:
            return f"http://localhost:{port}/v1"

        # Cloud backends use environment variables
        import os
        if backend == "vllm-local":
            return os.getenv("VLLM_LOCAL_BASE_URL", "http://10.1.1.110:8040/v1")
        elif backend == "zai":
            return os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
        elif backend == "nvidia":
            return os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
        elif backend == "openrouter":
            return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        elif backend == "kilo":
            return os.getenv("KILO_BASE_URL", "https://api.kilo.ai/api/gateway")

        logger.error(f"Cannot determine URL for backend {backend}")
        return None

    def route_by_category(
        self,
        headers: Dict[str, str],
        query_params: Dict[str, str],
        content: Optional[str] = None,
    ) -> RouteDecision:
        """
        Route request using category-based routing (oh-my-opencode style).

        This method is called when X-Task-Category header or category query param
        is present. It provides intelligent model selection based on task categories.

        Categories:
        - quick: Fast, lightweight tasks
        - ultrabrain: Deep logical reasoning, architecture decisions
        - deep: Complex algorithms, business logic
        - unspecified-high: High uncertainty, high quality needed
        - unspecified-low: Medium complexity with clear requirements
        - visual-engineering: UI/UX, design (vision models)
        - artistry: Creative work
        - writing: Documentation, prose

        Args:
            headers: HTTP headers from request
            query_params: Query parameters from request
            content: Request body content for auto-detection

        Returns:
            RouteDecision with selected model
        """
        try:
            # Lazy import to avoid circular dependency
            from ai_inference_gateway.category_router import (
                CategoryRouter,
                TaskCategory,
            )

            # Create category router if not exists
            if not hasattr(self, "_category_router"):
                models_list = list(self.models.values())
                self._category_router = CategoryRouter(
                    models=self.models,
                    default_category=TaskCategory.UNSPECIFIED_LOW,
                    enable_auto_detection=True,
                )

            # Route using category router
            decision = self._category_router.route(headers, query_params, content)

            # Verify model exists
            if decision.model not in self.models:
                logger.warning(f"Category router selected unknown model: {decision.model}")
                # Fallback to default routing
                return self.route(
                    messages=[],
                    requested_model=None,
                    headers=headers,
                )

            return decision

        except Exception as e:
            logger.error(f"Category routing failed: {e}, falling back to default routing")
            # Fallback to default routing
            return self.route(
                messages=[],
                requested_model=None,
                headers=headers,
            )

    def get_category_info(self) -> Dict[str, dict]:
        """
        Get information about available categories.

        Returns:
            Dictionary mapping category names to their configurations
        """
        try:
            if not hasattr(self, "_category_router"):
                from ai_inference_gateway.category_router import (
                    CategoryRouter,
                    TaskCategory,
                )

                models_list = list(self.models.values())
                self._category_router = CategoryRouter(
                    models=self.models,
                    default_category=TaskCategory.UNSPECIFIED_LOW,
                    enable_auto_detection=True,
                )

            return self._category_router.get_category_info()
        except Exception as e:
            logger.error(f"Failed to get category info: {e}")
            return {}

    async def check_backend_health(self, backend: str, force_check: bool = False) -> bool:
        """
        Check if a backend is healthy.

        Uses cached health status with TTL to avoid excessive health checks.
        For llama-cpp, we check if the backend is accepting connections.
        For zai, we assume it's healthy (cloud service).

        Args:
            backend: Backend name (llama-cpp, zai, etc.)
            force_check: Force a new health check, bypassing cache

        Returns:
            True if backend is healthy, False otherwise
        """
        import time

        # ZAI is assumed healthy (cloud service with own failover)
        if backend == "zai":
            return True

        # NVIDIA NIM is assumed healthy (cloud service)
        if backend == "nvidia":
            return True

        # Determine port for this backend
        port = self.BACKEND_PORTS.get(backend)
        if port is None:
            logger.warning(f"Unknown backend type '{backend}', assuming healthy")
            return True

        # Check cache for local backends
        now = time.time()
        last_check = self._backend_health_check_time.get(backend, 0)

        if not force_check and (now - last_check) < self._health_check_ttl:
            return self._backend_health.get(backend, True)

        # Perform health check for local backends
        try:
            import httpx

            # Try to connect to the backend
            # Use a short timeout to avoid blocking
            headers = {}  # llama-cpp doesn't require authentication

            async with httpx.AsyncClient(timeout=2.0) as client:
                # Try the health endpoint or models endpoint
                for endpoint in ["/v1/models", "/health"]:
                    try:
                        # Use configured BACKEND_URL from env (supports remote backends)
                        import os
                        backend_url = os.environ.get(
                            "BACKEND_URL", f"http://127.0.0.1:{port}"
                        )

                        response = await client.get(
                            f"{backend_url}{endpoint}",
                            headers=headers,
                            timeout=1.0,
                        )
                        is_healthy = response.status_code == 200
                        self._backend_health[backend] = is_healthy
                        self._backend_health_check_time[backend] = now

                        if is_healthy:
                            logger.debug(f"Backend {backend} is healthy")
                        else:
                            logger.warning(
                                f"Backend {backend} health check returned {response.status_code}"
                            )

                        return is_healthy
                    except Exception:
                        continue

            # All health checks failed
            logger.warning(f"Backend {backend} health check failed")
            self._backend_health[backend] = False
            self._backend_health_check_time[backend] = now
            return False

        except Exception as e:
            logger.error(f"Error checking backend {backend} health: {e}")
            self._backend_health[backend] = False
            self._backend_health_check_time[backend] = now
            return False

    async def is_backend_healthy(self, backend: str) -> bool:
        """
        Check if backend is healthy (cached result).

        Args:
            backend: Backend name

        Returns:
            True if healthy, False otherwise
        """
        return self._backend_health.get(backend, True)

    def _build_claude_mapping(self) -> Dict[str, str]:
        """Build mapping from Anthropic Claude model names to available models.

        LOCAL-FIRST STRATEGY: Maps Claude models to local Opus-distilled variants.

        Model mapping (5 Claude options → 3 underlying local models):
        - Opus → qwen3.5-35b-a3b (largest, highest quality)
        - Opus (1M context) → qwen3.5-35b-a3b (same model, extended context variant)
        - Sonnet → qwen3.5-9b-claude-4.6-opus-reasoning-distilled (balanced)
        - Sonnet (1M context) → qwen3.5-9b-claude-4.6-opus-reasoning-distilled (same model, extended context variant)
        - Haiku → qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled (fastest)

        Note: "1M" context variants map to the same underlying model since Qwen models
        support up to 256K context. The distinction is client-side metadata.

        Cloud fallback chain (when Local backend down/capacity): NIM → OpenRouter
        """
        return {
            # Haiku tier → NIM Nano 30B (fastest reliable NIM model)
            "claude-haiku-4": "nvidia/nemotron-3-nano-30b-a3b",
            "claude-haiku-4-20250514": "nvidia/nemotron-3-nano-30b-a3b",
            # Sonnet tier → NIM Super 120B (balanced, primary coding model)
            "claude-sonnet-4-20250514": "nvidia/nemotron-3-super-120b-a12b",
            "claude-sonnet-4": "nvidia/nemotron-3-super-120b-a12b",
            # Sonnet extended context variant
            "claude-sonnet-4-20250514-1m": "nvidia/nemotron-3-super-120b-a12b",
            # Opus tier → Mistral Large 3 675B (best quality NIM model)
            "claude-opus-4-20250514": "mistralai/mistral-large-3-675b-instruct-2512",
            "claude-opus-4": "mistralai/mistral-large-3-675b-instruct-2512",
            # Opus extended context variant
            "claude-opus-4-20250514-1m": "mistralai/mistral-large-3-675b-instruct-2512",
        }

    # ========================================================================
    # Autonomous Model Selection (Benchmark-based)
    # ========================================================================

    def get_benchmark(self) -> ModelBenchmark:
        """Get the model benchmark instance."""
        return get_benchmark()

    def get_selector(self) -> AutonomousModelSelector:
        """Get the autonomous model selector."""
        return get_selector()

    async def select_model_by_benchmark(
        self,
        requirements: Dict[str, any],
        available_models: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Select best model using benchmark data.

        Requirements:
        - estimated_input_tokens: int
        - estimated_output_tokens: int
        - task_specialization: TaskSpecialization
        - max_ttft_ms: Optional[float]
        - min_throughput_tps: Optional[float]
        - needs_concurrency: Optional[int]

        Returns:
            Best model ID or None if no suitable model found
        """
        selector = self.get_selector()

        if available_models is None:
            available_models = list(self.models.keys())

        return await selector.select_best_model(requirements, available_models)

    def get_model_rankings(
        self,
        requirements: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Get ranked models based on benchmark data.

        Returns list of {model_id, rank, score, strengths, weaknesses, best_for}
        """
        benchmark = self.get_benchmark()
        rankings = benchmark.rank_all_models(requirements)

        return [
            {
                "model_id": r.model_id,
                "rank": r.rank,
                "score": r.score,
                "strengths": r.strengths,
                "weaknesses": r.weaknesses,
                "best_for": r.best_for,
                "avg_ttft_ms": r.avg_ttft_ms,
                "avg_throughput_tps": r.avg_throughput_tps,
                "actual_context_window": r.actual_context_window,
            }
            for r in rankings
        ]

    def get_model_recommendations(self, task_type: str) -> List[Dict]:
        """
        Get model recommendations for a task type.

        Args:
            task_type: fast-chat, coding, long-context, rag, analysis, batch, high-volume

        Returns:
            List of {model_id, score, reason}
        """
        selector = self.get_selector()
        return selector.get_recommendations(task_type)

    async def start_auto_benchmark(
        self,
        backend_configs: List[Tuple[str, str, str, Optional[str]]],
    ) -> Dict[str, any]:
        """
        Start automatic benchmarking of all models.

        Args:
            backend_configs: List of (model_id, backend, backend_url, api_key)

        Returns:
            Benchmark results summary
        """
        benchmark = self.get_benchmark()

        logger.info(f"Starting auto-benchmark of {len(backend_configs)} models")
        results = await benchmark.auto_benchmark_all(backend_configs)

        return {
            "total_models": len(backend_configs),
            "successful": sum(1 for r in results.values() if r.success),
            "failed": sum(1 for r in results.values() if not r.success),
            "results": {
                model_id: {
                    "success": r.success,
                    "metrics": {k.value: v for k, v in r.metrics.items()},
                }
                for model_id, r in results.items()
            },
        }

    # ========================================================================
    # Token Estimation & Analysis
    # ========================================================================

    def estimate_tokens(self, messages: List[Dict]) -> int:
        """
        Estimate token count for messages.

        Args:
            messages: List of message dicts with 'content' field

        Returns:
            Estimated token count
        """
        CHARS_PER_TOKEN = 4
        CHARS_PER_TOKEN_CODE = 6

        total_chars = 0
        has_code = False

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
                # Detect code blocks
                if "```" in content or "def " in content or "function " in content:
                    has_code = True

        divisor = CHARS_PER_TOKEN_CODE if has_code else CHARS_PER_TOKEN
        return max(1, total_chars // divisor)

    def apply_prefill_limits(
        self,
        messages: List[Dict],
        claude_model: Optional[str] = None,
    ) -> List[Dict]:
        """
        Apply prefill optimization limits to reduce time-to-first-token.

        Base models get aggressive limits (fewer input tokens = faster prefill).
        Extended context variants get full context.

        Args:
            messages: List of message dicts
            claude_model: Original Claude model ID requested

        Returns:
            Potentially trimmed messages list
        """
        if not claude_model or claude_model not in MODEL_PREFILL_CONFIG:
            return messages

        config = MODEL_PREFILL_CONFIG[claude_model]
        trimmed_messages = messages
        max_tokens = config.get("max_input_tokens")
        max_history = config.get("max_history_messages")

        # Apply history message limit if set
        if max_history is not None and len(messages) > max_history:
            # Keep system messages and trim user/assistant history
            system_msgs = [m for m in messages if m.get("role") == "system"]
            history_msgs = [m for m in messages if m.get("role") != "system"]

            # Keep most recent history messages
            trimmed_history = history_msgs[-max_history:]
            trimmed_messages = system_msgs + trimmed_history

            logger.debug(
                f"Prefill trim: {len(messages)} → {len(trimmed_messages)} messages "
                f"for {claude_model}"
            )

        # Apply token limit if set
        if max_tokens is not None:
            estimated = self.estimate_tokens(trimmed_messages)
            if estimated > max_tokens:
                logger.warning(
                    f"Prefill limit: {estimated} > {max_tokens} tokens for {claude_model}, "
                    f"consider using extended context variant"
                )
                # Could truncate here, but for now just warn
                # Truncation could break conversation flow

        return trimmed_messages

    def detect_specialization(self, messages: List[Dict]) -> TaskSpecialization:
        """
        Detect task type from messages.

        Args:
            messages: List of message dicts

        Returns:
            Detected task specialization
        """
        # Check for vision content FIRST (highest priority)
        try:
            from ai_inference_gateway.vision import detect_vision_content

            if detect_vision_content(messages):
                logger.info("Vision content detected in request")
                return TaskSpecialization.VISION
        except ImportError:
            logger.warning("Vision module not available, skipping vision detection")

        # Combine all message content
        text = " ".join(
            msg.get("content", "")
            for msg in messages
            if isinstance(msg.get("content", ""), str)
        ).lower()

        # Check for code/programming
        code_patterns = [
            r"```\w*",
            r"def\s+\w+",
            r"function\s+\w+",
            r"class\s+\w+",
            r"import\s+\w+",
            r"from\s+\w+\s+import",
            r"λ\s*->",
            r"=>\s*{",
            r"@\[|for\s+\w+\s+in",
        ]
        if any(re.search(pattern, text) for pattern in code_patterns):
            return TaskSpecialization.CODING

        # Check for agentic/multi-step tasks
        agentic_keywords = [
            "agent",
            "workflow",
            "multi-step",
            "step by step",
            "plan",
            "analyze then",
        ]
        if any(keyword in text for keyword in agentic_keywords):
            return TaskSpecialization.AGENTIC

        # Check for urgency/fast mode
        fast_keywords = ["quickly", "asap", "fast", "brief", "short", "quick"]
        if any(keyword in text for keyword in fast_keywords):
            return TaskSpecialization.FAST

        # Check for large context needs
        if len(text) > 10000:  # Large input
            return TaskSpecialization.LARGE_CONTEXT

        return TaskSpecialization.GENERAL

    async def route(
        self,
        messages: List[Dict],
        requested_model: Optional[str] = None,
        urgency: str = "normal",
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, str]] = None,
    ) -> RouteDecision:
        """
        Route a request to the best model.

        Args:
            messages: List of messages
            requested_model: Optional model requested by client
            urgency: Urgency level (fast, normal, quality)
            headers: Optional HTTP headers for category-based routing
            query_params: Optional query parameters for category-based routing

        Returns:
            Routing decision with model and metadata
        """
        # Initialize headers/params if None
        headers = headers or {}
        query_params = query_params or {}

        # Check for category-based routing hints (oh-my-opencode style)
        category_hint = headers.get("X-Task-Category") or query_params.get("category")
        if category_hint:
            # Combine message content for category detection
            content = " ".join(
                msg.get("content", "")
                for msg in messages
                if isinstance(msg.get("content", ""), str)
            )
            # Use category-based routing
            try:
                decision = self.route_by_category(headers, query_params, content)
                logger.info(
                    f"Category-based routing selected model: {decision.model} "
                    f"(category: {category_hint}, confidence: {decision.confidence})"
                )
                return decision
            except Exception as e:
                logger.warning(f"Category routing failed, falling back to default: {e}")

        # Check if llama.cpp is healthy before routing
        local_backend_healthy = await self.check_backend_health("llama-cpp")

        # If Local backend (llama-cpp) is down, failover to cloud backends
        if not local_backend_healthy:
            logger.info("Local backend (llama-cpp) is down, using cloud fallback")
            estimated_tokens = self.estimate_tokens(messages)

            # Get available cloud models (NIM, ZAI, OpenRouter)
            cloud_models = [m for m in self.models.values()
                           if m.backend in ["nvidia", "zai", "openrouter"] and m.priority > 0]

            if cloud_models:
                # Sort by priority and pick the best one
                best_cloud = max(cloud_models, key=lambda m: m.priority)
                specialization = self.detect_specialization(messages)

                return RouteDecision(
                    model=best_cloud.id,
                    confidence=0.95,
                    reason="Local backend down (auto-failover to cloud)",
                    estimated_tokens=estimated_tokens,
                    backend=best_cloud.backend,
                    specialization=specialization,
                    expected_latency_ms=best_cloud.estimated_tokens_per_second
                    * estimated_tokens
                    / 1000,
                )
            else:
                # No cloud models available - this is an error condition
                logger.error("Local backend down and no cloud fallback available!")
                # Return default model anyway (will likely fail)
                return RouteDecision(
                    model="qwen/qwen3.5-9b",
                    confidence=0.1,
                    reason="Local backend down, no cloud fallback available",
                    estimated_tokens=estimated_tokens,
                    backend="llama-cpp",
                )

        # Check if llama.cpp is busy with streaming requests
        local_backend_load = await self.get_backend_load("llama-cpp")

        # If llama.cpp is at capacity (processing streams), route to ZAI
        if local_backend_load["at_capacity"] and local_backend_load["is_streaming"]:
            logger.info(
                f"Local backend busy ({local_backend_load['active_requests']} active requests, "
                f"streaming: {local_backend_load['is_streaming']}), auto-offloading to ZAI"
            )
            # Find best ZAI model for the request
            estimated_tokens = self.estimate_tokens(messages)

            # If client requested a specific model, check if we can map it to ZAI
            if requested_model:
                # Check if it's a Claude model that maps to ZAI
                if requested_model in self.claude_model_mapping:
                    mapped_model = self.claude_model_mapping[requested_model]
                    model_info = self.models.get(mapped_model)
                    if model_info and model_info.backend == "zai":
                        return RouteDecision(
                            model=mapped_model,
                            confidence=1.0,
                            reason=f"llama.cpp at capacity, using ZAI fallback for {requested_model}",
                            estimated_tokens=estimated_tokens,
                            backend="zai",
                            expected_latency_ms=model_info.estimated_tokens_per_second
                            * estimated_tokens
                            / 1000,
                        )

            # Otherwise, find best ZAI model based on specialization
            zai_models = [m for m in self.models.values() if m.backend == "zai"]
            if zai_models:
                # Sort by priority and pick the best one
                best_zai = max(zai_models, key=lambda m: m.priority)
                specialization = self.detect_specialization(messages)
                return RouteDecision(
                    model=best_zai.id,
                    confidence=0.9,
                    reason="llama.cpp at capacity (auto-failover to ZAI)",
                    estimated_tokens=estimated_tokens,
                    backend="zai",
                    specialization=specialization,
                    expected_latency_ms=best_zai.estimated_tokens_per_second
                    * estimated_tokens
                    / 1000,
                )

        # Estimate tokens
        estimated_tokens = self.estimate_tokens(messages)

        # Check if client requested a specific model
        if requested_model:
            # Check if it's a Claude model name
            if requested_model in self.claude_model_mapping:
                mapped_model = self.claude_model_mapping[requested_model]
                model_info = self.models.get(mapped_model)
                if model_info:
                    return RouteDecision(
                        model=mapped_model,
                        confidence=1.0,
                        reason=f"Claude model mapped to {mapped_model}",
                        estimated_tokens=estimated_tokens,
                        backend=model_info.backend,
                        expected_latency_ms=model_info.estimated_tokens_per_second
                        * estimated_tokens
                        / 1000,
                    )
            # Check if it's a direct model ID
            elif requested_model in self.models:
                model_info = self.models[requested_model]
                return RouteDecision(
                    model=requested_model,
                    confidence=1.0,
                    reason=f"Requested model {requested_model}",
                    estimated_tokens=estimated_tokens,
                    backend=model_info.backend,
                    expected_latency_ms=model_info.estimated_tokens_per_second
                    * estimated_tokens
                    / 1000,
                )
            # Check if it's a discovered model (not in static models but in discovery registry)
            elif self.model_discovery:
                disc_backend = self.model_discovery.get_backend_for_model(requested_model)
                if disc_backend:
                    return RouteDecision(
                        model=requested_model,
                        confidence=1.0,
                        reason=f"Requested discovered model {requested_model}",
                        estimated_tokens=estimated_tokens,
                        backend=disc_backend,
                    )
            # Cloud discovery fallback: model not in router but known to cloud registry
            elif self.cloud_discovery:
                cloud_model = self.cloud_discovery.get_model(requested_model)
                if cloud_model:
                    backend_map = {
                        "openrouter": "openrouter",
                        "opencode-go": "opencode-go",
                        "opencode-zen": "opencode-zen",
                        "nim": "nvidia",
                        "zai": "zai",
                    }
                    backend = backend_map.get(cloud_model.provider, "openrouter")
                    logger.info(
                        f"Cloud discovery fallback: {requested_model} -> {backend}"
                    )
                    return RouteDecision(
                        model=requested_model,
                        confidence=0.9,
                        reason=f"Cloud-discovered model ({cloud_model.provider})",
                        estimated_tokens=estimated_tokens,
                        backend=backend,
                        expected_latency_ms=80.0 * estimated_tokens / 1000,
                    )

        # Detect task specialization
        specialization = self.detect_specialization(messages)

        # Generate candidates
        candidates = await self._generate_candidates(
            estimated_tokens=estimated_tokens,
            specialization=specialization,
            urgency=urgency,
        )

        # Rank candidates
        ranked_candidates = await self._rank_candidates(
            candidates=candidates,
            specialization=specialization,
            urgency=urgency,
            estimated_tokens=estimated_tokens,
        )

        if not ranked_candidates:
            # Fallback to default model (use fast model for quick responses)
            default_model = "qwen3.5-4b"
            model_info = self.models.get(default_model)
            if model_info:
                return RouteDecision(
                    model=default_model,
                    confidence=0.5,
                    reason="No suitable candidates, using default",
                    estimated_tokens=estimated_tokens,
                    backend=model_info.backend,
                )
            else:
                # Ultimate fallback if even default is unavailable
                fallback_model = list(self.models.keys())[0]
                model_info = self.models[fallback_model]
                return RouteDecision(
                    model=fallback_model,
                    confidence=0.3,
                    reason="Default model unavailable, using fallback",
                    estimated_tokens=estimated_tokens,
                    backend=model_info.backend,
                )

        # Select best candidate
        best = ranked_candidates[0]
        return RouteDecision(
            model=best.model,
            confidence=best.score,
            reason=best.reason,
            estimated_tokens=estimated_tokens,
            backend=best.backend,
            specialization=best.specialization,
            expected_latency_ms=best.expected_latency_ms,
        )

    async def _generate_candidates(
        self,
        estimated_tokens: int,
        specialization: TaskSpecialization,
        urgency: str,
    ) -> List[ModelCandidate]:
        """Generate candidate models for the request."""
        candidates = []

        for model_id, model_info in self.models.items():
            # Filter by context length
            if estimated_tokens > model_info.context_length:
                continue

            # Check if model is overloaded
            if await self.latency_tracker.is_overloaded(model_id):
                logger.warning(f"Model {model_id} is overloaded, skipping")
                continue

            # Base score from priority
            score = float(model_info.priority)

            # Penalize models on busy local backends
            backend = model_info.backend
            if backend and backend.startswith("llama-") and self.is_backend_busy(backend):
                score -= 5.0
                logger.debug(f"Backend {backend} busy, penalizing model {model_id}")

            # Boost for specialization match
            if specialization in model_info.specializations:
                score += 1.5

            # Estimate latency
            expected_latency_ms = (
                estimated_tokens / model_info.estimated_tokens_per_second
            ) * 1000

            candidates.append(
                ModelCandidate(
                    model=model_id,
                    backend=model_info.backend,
                    score=score,
                    reason=f"Priority {model_info.priority}, specialization {specialization.value if specialization in model_info.specializations else 'none'}",
                    specialization=specialization,
                    expected_latency_ms=expected_latency_ms,
                )
            )

        return candidates

    async def _rank_candidates(
        self,
        candidates: List[ModelCandidate],
        specialization: TaskSpecialization,
        urgency: str,
        estimated_tokens: int = 0,
    ) -> List[ModelCandidate]:
        """Rank candidates by multiple factors."""
        for candidate in candidates:
            # Apply specialization boost
            model_info = self.models[candidate.model]
            if specialization in model_info.specializations:
                candidate.score *= 1.5

            # Adjust for latency
            avg_latency = await self.latency_tracker.get_avg_latency(candidate.model)
            if avg_latency:
                if avg_latency > 3000:  # > 3s
                    candidate.score *= 0.5
                elif avg_latency > 1000:  # > 1s
                    candidate.score *= 0.7

            # Token-aware routing: penalize cloud backends for large requests
            # Cloud APIs timeout on 60k+ token requests; local has no timeout
            if estimated_tokens > TOKEN_THRESHOLD_LOCAL:
                cloud_backends = ("zai", "nvidia")
                if candidate.backend in cloud_backends:
                    # Progressive penalty: at 60k it's mild, at 100k it's severe
                    overshoot = (estimated_tokens - TOKEN_THRESHOLD_LOCAL) / TOKEN_THRESHOLD_LOCAL
                    penalty = max(0.1, 1.0 - overshoot * 0.5)
                    candidate.score *= penalty
                    if overshoot > 0.5:
                        logger.warning(
                            f"Token-aware routing: {candidate.model} ({candidate.backend}) "
                            f"penalized {penalty:.2f}x for {estimated_tokens} tokens"
                        )

            # Urgency adjustment
            if urgency == "fast":
                # Prefer faster models
                candidate.score /= candidate.expected_latency_ms / 1000
            elif urgency == "quality":
                # Prefer higher cost tier (better quality)
                candidate.score *= 1 + model_info.cost_tier * 0.1

        # Sort by score descending
        return sorted(candidates, key=lambda c: c.score, reverse=True)


def create_default_router(model_discovery=None, cloud_discovery=None) -> Router:
    """Create router with default model configuration.

    Args:
        model_discovery: Optional ModelDiscovery instance for auto-discovering local models
        cloud_discovery: Optional CloudModelRegistry for auto-discovering cloud models
    """
    # Mapping from string specializations (from cloud API) to TaskSpecialization enum
    _SPECIALIZATION_MAP = {
        "coding": TaskSpecialization.CODING,
        "agentic": TaskSpecialization.AGENTIC,
        "general": TaskSpecialization.GENERAL,
        "fast": TaskSpecialization.FAST,
        "large_context": TaskSpecialization.LARGE_CONTEXT,
        "vision": TaskSpecialization.VISION,
        "reasoning": TaskSpecialization.REASONING,
        "classification": TaskSpecialization.CLASSIFICATION,
        "routing": TaskSpecialization.ROUTING,
    }

    def _map_specializations(specializations: List[str]) -> List[TaskSpecialization]:
        """Convert cloud model specialization strings to TaskSpecialization enum values."""
        result = []
        for spec in specializations:
            enum_val = _SPECIALIZATION_MAP.get(spec.lower())
            if enum_val:
                result.append(enum_val)
            else:
                logger.debug(f"Unknown specialization '{spec}' from cloud model, using GENERAL")
                result.append(TaskSpecialization.GENERAL)
        return result if result else [TaskSpecialization.GENERAL]

    # If model discovery is available, use it to dynamically discover models
    # Otherwise, fall back to hardcoded model list
    if model_discovery:
        # Query backends for available models (synchronously, before event loop)
        try:
            discovered = model_discovery.refresh_all_backends_sync()
        except Exception as e:
            logger.warning(f"Model discovery failed: {e}")
            discovered = {}

        # Build model list from discovered models
        models = []
        for backend_name, backend_info in discovered.items():
            if backend_info.models:
                for model_id in backend_info.models:
                    # Determine model characteristics from model ID
                    priority = backend_info.priority
                    backend = backend_name  # "llama-3090", "llama-3060ti", "llama-sentry"

                    # Add model info
                    models.append(ModelInfo(
                        id=model_id,
                        name=model_id,  # Use full model ID as name
                        context_length=get_context_length(model_id),
                        priority=priority,
                        specializations=[TaskSpecialization.GENERAL],
                        cost_tier=0,  # Free (local)
                        estimated_tokens_per_second=50.0,
                        backend=backend,
                    ))
                    logger.info(f"Discovered model {model_id} on {backend_name}")

        # Fall back to hardcoded models if discovery failed
        if not models:
            logger.warning("Model discovery failed, using hardcoded model list")
            models = _get_hardcoded_models()
    else:
        models = _get_hardcoded_models()

    # Merge cloud-discovered models (OpenRouter, NIM, ZAI)
    # These replace hardcoded cloud models with auto-discovered equivalents
    if cloud_discovery and cloud_discovery.models:
        existing_ids = {m.id for m in models}
        cloud_count = 0
        for mid, cloud_model in cloud_discovery.models.items():
            if mid in existing_ids:
                continue  # Don't override local models with same ID

            # Map provider to backend name
            backend_map = {
                "openrouter": "openrouter",
                "opencode-go": "opencode-go",
                "opencode-zen": "opencode-zen",
                "nim": "nvidia",
                "zai": "zai",
            }
            backend = backend_map.get(cloud_model.provider, "openrouter")

            models.append(ModelInfo(
                id=mid,
                name=cloud_model.name,
                context_length=cloud_model.context_length,
                priority=cloud_model.priority,
                specializations=_map_specializations(cloud_model.specializations),
                cost_tier=cloud_model.cost_tier,
                estimated_tokens_per_second=80.0,  # Cloud models are fast
                backend=backend,
            ))
            cloud_count += 1

        if cloud_count:
            logger.info(f"Added {cloud_count} cloud-discovered models, total: {len(models)}")

    # Merge hardcoded cloud models (NIM, ZAI, OpenRouter) with discovered models
    # Hardcoded entries OVERRIDE cloud-discovered ones for the same model ID
    # because they have accurate backend assignments (not auto-detected).
    try:
        hardcoded = _get_hardcoded_models() or []
        if hardcoded and models:
            existing_ids = {m.id for m in models}
            # Replace cloud-discovered models with hardcoded ones (better backend mapping)
            models = [m for m in models if m.id not in {h.id for h in hardcoded}]
            cloud_models = [m for m in hardcoded if m.backend in ("nvidia", "zai", "openrouter", "kilo", "opencode-go", "opencode-zen")]
            models.extend(cloud_models)
            if cloud_models:
                logger.warning(f"Added {len(cloud_models)} hardcoded cloud models, total: {len(models)}")
        elif hardcoded and not models:
            models = hardcoded
            logger.warning(f"No discovered models, using {len(models)} hardcoded models")
    except Exception as e:
        logger.warning(f"Failed to merge cloud models: {e}")

    return Router(models, model_discovery=model_discovery, cloud_discovery=cloud_discovery)


def _get_hardcoded_models() -> List[ModelInfo]:
    models = [
        # ========================================================================
        # Local llama.cpp models - routed by model name to correct GPU
        # ========================================================================
        # Qwen3.6 35B A3B Abliterated - Primary (3090, zephyr:1237)
        ModelInfo(
            id="qwen3.6-35b",
            name="Qwen 3.6 35B A3B Abliterated",
            context_length=LLAMA_SERVER_CONTEXT["qwen3.6-35b"],
            priority=12,  # Highest priority local model
            specializations=[
                TaskSpecialization.LARGE_CONTEXT,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=3,
            estimated_tokens_per_second=35.0,
            backend="llama-cpp",
        ),
        # SuperGemma4 Q5_K_M - Secondary (3060Ti, zephyr:1236)
        ModelInfo(
            id="supergemma4-Q5_K_M.gguf",
            name="Supergemma4 E4B (Local 3060Ti)",
            context_length=LLAMA_SERVER_CONTEXT["supergemma4"],
            priority=11,  # High priority for local-fast routing
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=0,  # Free (local)
            estimated_tokens_per_second=80.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 4B - Local ROCm (sentry:1235)
        ModelInfo(
            id="qwen3.5-4b",
            name="Qwen 3.5 4B (Local ROCm)",
            context_length=LLAMA_SERVER_CONTEXT["qwen3.5-4b"],
            priority=10,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,  # Free (local)
            estimated_tokens_per_second=80.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 2B AWQ - vLLM (3060Ti, zephyr:8040)
        ModelInfo(
            id="qwen3.5-2b-awq",
            name="Qwen 3.5 2B AWQ (vLLM Local)",
            context_length=32768,
            priority=9,  # Highest priority for fast tasks (classification, routing)
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=0,  # Free (local)
            estimated_tokens_per_second=564.0,
            backend="vllm-local",
        ),
        # ========================================================================
        # ZAI models - Cloud fallback
        # ========================================================================
        ModelInfo(
            id="glm-5.1",
            name="GLM-5.1",
            context_length=CLOUD_MODEL_CONTEXT["glm-5.1"],
            priority=7,  # Highest priority ZAI model
            specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.GENERAL, TaskSpecialization.CODING],
            cost_tier=4,
            estimated_tokens_per_second=40.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-5-turbo",
            name="GLM-5 Turbo",
            context_length=CLOUD_MODEL_CONTEXT["glm-5-turbo"],
            priority=7,  # Same as glm-5.1 (turbo variant)
            specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.GENERAL, TaskSpecialization.CODING],
            cost_tier=4,
            estimated_tokens_per_second=50.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-5",
            name="GLM-5",
            context_length=CLOUD_MODEL_CONTEXT["glm-5"],
            priority=6,
            specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.GENERAL],
            cost_tier=4,
            estimated_tokens_per_second=40.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.7",
            name="GLM-4.7",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.7"],
            priority=5,
            specializations=[TaskSpecialization.CODING, TaskSpecialization.GENERAL],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.6v",
            name="GLM-4.6v",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.6v"],
            priority=5,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.FAST,
                TaskSpecialization.VISION,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.6",
            name="GLM-4.6",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.6"],
            priority=4,
            specializations=[TaskSpecialization.GENERAL, TaskSpecialization.CODING],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.7-flash",
            name="GLM-4.7 Flash",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.7-flash"],
            priority=4,
            specializations=[TaskSpecialization.FAST, TaskSpecialization.GENERAL],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.5",
            name="GLM-4.5",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.5"],
            priority=3,
            specializations=[TaskSpecialization.GENERAL],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.5-flash",
            name="GLM-4.5 Flash",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.5-flash"],
            priority=3,
            specializations=[TaskSpecialization.FAST],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.5-air",
            name="GLM-4.5 Air",
            context_length=CLOUD_MODEL_CONTEXT["glm-4.5-air"],
            priority=5,
            specializations=[TaskSpecialization.FAST],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="zai",
        ),
        # ========================================================================
        # NVIDIA NIM models - Cloud-hosted via NVIDIA NIM API (priority 8)
        # ========================================================================
        # Existing working NIM models (kept)
        ModelInfo(
            id="nvidia/llama-3.3-nemotron-super-49b-v1",
            name="Nemotron-Super-49B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["nvidia/llama-3.3-nemotron-super-49b-v1"],
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-coder-480b-a35b-instruct",
            name="Qwen3 Coder 480B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["qwen/qwen3-coder-480b-a35b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="meta/llama-3.1-405b-instruct",
            name="Llama 3.1 405B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["meta/llama-3.1-405b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="z-ai/glm-5.1",
            name="GLM-5.1 (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["z-ai/glm-5.1"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        # New NIM models (2026-04-25 batch)
        ModelInfo(
            id="nvidia/nemotron-3-super-120b-a12b",
            name="Nemotron-3 Super 120B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["nvidia/nemotron-3-super-120b-a12b"],
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/nemotron-mini-4b-instruct",
            name="Nemotron Mini 4B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["nvidia/nemotron-mini-4b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=1,
            estimated_tokens_per_second=90.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3.5-397b-a17b",
            name="Qwen3.5 397B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["qwen/qwen3.5-397b-a17b"],
            priority=8,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3.5-122b-a10b",
            name="Qwen3.5 122B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["qwen/qwen3.5-122b-a10b"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-next-80b-a3b-instruct",
            name="Qwen3 Next 80B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["qwen/qwen3-next-80b-a3b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="deepseek-ai/deepseek-v4-flash",
            name="DeepSeek V4 Flash 284B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["deepseek-ai/deepseek-v4-flash"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="meta/llama-3.3-70b-instruct",
            name="Llama 3.3 70B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["meta/llama-3.3-70b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="meta/llama-4-maverick-17b-128e-instruct",
            name="Llama 4 Maverick MoE (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["meta/llama-4-maverick-17b-128e-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="meta/llama-3.2-90b-vision-instruct",
            name="Llama 3.2 90B Vision (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["meta/llama-3.2-90b-vision-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="moonshotai/kimi-k2-instruct",
            name="Kimi K2 Instruct (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["moonshotai/kimi-k2-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="moonshotai/kimi-k2-thinking",
            name="Kimi K2 Thinking (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["moonshotai/kimi-k2-thinking"],
            priority=8,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=3,
            estimated_tokens_per_second=35.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="mistralai/mistral-large-3-675b-instruct-2512",
            name="Mistral Large 3 675B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["mistralai/mistral-large-3-675b-instruct-2512"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.CODING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="mistralai/devstral-2-123b-instruct-2512",
            name="Devstral 2 123B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["mistralai/devstral-2-123b-instruct-2512"],
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="mistralai/mistral-small-4-119b-2603",
            name="Mistral Small 4 119B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["mistralai/mistral-small-4-119b-2603"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="mistralai/magistral-small-2506",
            name="Magistral Small (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["mistralai/magistral-small-2506"],
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="mistralai/mixtral-8x22b-instruct-v0.1",
            name="Mixtral 8x22B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["mistralai/mixtral-8x22b-instruct-v0.1"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="minimaxai/minimax-m2.5",
            name="MiniMax M2.5 230B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["minimaxai/minimax-m2.5"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="minimaxai/minimax-m2.7",
            name="MiniMax M2.7 (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["minimaxai/minimax-m2.7"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="openai/gpt-oss-120b",
            name="GPT-OSS 120B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["openai/gpt-oss-120b"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="z-ai/glm4.7",
            name="GLM-4.7 (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["z-ai/glm4.7"],
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="google/gemma-3-27b-it",
            name="Gemma 3 27B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["google/gemma-3-27b-it"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.FAST,
            ],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="microsoft/phi-4-multimodal-instruct",
            name="Phi-4 Multimodal (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["microsoft/phi-4-multimodal-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="stepfun-ai/step-3.5-flash",
            name="Step 3.5 Flash (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["stepfun-ai/step-3.5-flash"],
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="bytedance/seed-oss-36b-instruct",
            name="Seed OSS 36B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["bytedance/seed-oss-36b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=1,
            estimated_tokens_per_second=65.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="stockmark/stockmark-2-100b-instruct",
            name="Stockmark 2 100B (NIM)",
            context_length=CLOUD_MODEL_CONTEXT["stockmark/stockmark-2-100b-instruct"],
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        # ========================================================================
        # KILO AI free models (https://api.kilo.ai/api/gateway)
        # ========================================================================
        ModelInfo(
            id="kilo-auto/free",
            name="KILO Auto Free (Best available free model)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=50.0,
            backend="kilo",
        ),
        ModelInfo(
            id="bytedance-seed/dola-seed-2.0-pro:free",
            name="Dola Seed 2.0 Pro (KILO Free)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=0,
            estimated_tokens_per_second=40.0,
            backend="kilo",
        ),
        ModelInfo(
            id="x-ai/grok-code-fast-1:optimized:free",
            name="Grok Code Fast 1 Optimized (KILO Free)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.CODING,
            ],
            cost_tier=0,
            estimated_tokens_per_second=60.0,
            backend="kilo",
        ),
        ModelInfo(
            id="nvidia/nemotron-3-super-120b-a12b:free",
            name="Nemotron 3 Super 120B (KILO Free)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.CODING,
            ],
            cost_tier=0,
            estimated_tokens_per_second=35.0,
            backend="kilo",
        ),
        ModelInfo(
            id="arcee-ai/trinity-large-thinking:free",
            name="Trinity Large Thinking (KILO Free)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=30.0,
            backend="kilo",
        ),
        ModelInfo(
            id="openrouter/free",
            name="OpenRouter Free (KILO)",
            context_length=128000,
            priority=5,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=45.0,
            backend="kilo",
        ),
        # ========================================================================
        # OpenCode Go models (subscription, 5h+ weekly cap)
        # ========================================================================
        ModelInfo(
            id="opencode/deepseek-v4-flash",
            name="DeepSeek V4 Flash (OpenCode Go)",
            context_length=1000000,
            priority=7,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="opencode-go",
        ),
        ModelInfo(
            id="opencode/nemotron-3-super-120b-a12b",
            name="Nemotron 3 Super (OpenCode Go)",
            context_length=1000000,
            priority=7,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.CODING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="opencode-go",
        ),
        # ========================================================================
        # OpenCode Zen models (free, daily quota 7PM reset)
        # ========================================================================
        ModelInfo(
            id="deepseek-v4-flash",
            name="DeepSeek V4 Flash (OpenCode Zen Free)",
            context_length=1000000,
            priority=6,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=50.0,
            backend="opencode-zen",
        ),
        ModelInfo(
            id="nvidia/nemotron-3-super-120b-a12b:free",
            name="Nemotron 3 Super (OpenCode Zen Free, 128K)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=35.0,
            backend="opencode-zen",
        ),
        ModelInfo(
            id="nvidia/nemotron-3-nano-30b-a3b:free",
            name="Nemotron 3 Nano (OpenCode Zen Free, 128K)",
            context_length=128000,
            priority=6,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,
            estimated_tokens_per_second=65.0,
            backend="opencode-zen",
        ),
        # ========================================================================
        # Additional NIM models (2026-05-02 — new from /v1/models discovery)
        # ========================================================================
        # Vision models
        ModelInfo(
            id="z-ai/glm-5v-turbo",
            name="GLM 5V Turbo (Z.AI Vision)",
            context_length=131072,
            priority=7,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.GENERAL,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="zai",
        ),
        ModelInfo(
            id="z-ai/glm-4.5v",
            name="GLM 4.5V (Z.AI Vision)",
            context_length=131072,
            priority=5,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.FAST,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="zai",
        ),
        ModelInfo(
            id="z-ai/glm-4.6v",
            name="GLM 4.6V (Z.AI Vision)",
            context_length=131072,
            priority=5,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.CODING,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="zai",
        ),
        # Nemotron new additions
        ModelInfo(
            id="nvidia/nemotron-4-340b-instruct",
            name="Nemotron 4 340B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/nemotron-3-nano-30b-a3b",
            name="Nemotron 3 Nano 30B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.FAST,
            ],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
            name="Nemotron 3 Nano Omni 30B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.REASONING,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/nemotron-nano-9b-v2",
            name="Nemotron Nano 9B v2 (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=1,
            estimated_tokens_per_second=90.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/llama-3.1-nemotron-70b-instruct",
            name="Nemotron 70B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="nvidia/nemotron-mini-4b-instruct",
            name="Nemotron Mini 4B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CLASSIFICATION,
            ],
            cost_tier=1,
            estimated_tokens_per_second=100.0,
            backend="nvidia",
        ),
        # Kimi / Moonshot new
        ModelInfo(
            id="moonshotai/kimi-k2.6",
            name="Kimi K2.6 (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="moonshotai/kimi-k2.5",
            name="Kimi K2.5 (NIM)",
            context_length=262144,
            priority=8,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.LARGE_CONTEXT,
                TaskSpecialization.VISION,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="nvidia",
        ),
        # DeepSeek V4 Pro
        ModelInfo(
            id="deepseek-ai/deepseek-v4-pro",
            name="DeepSeek V4 Pro (NIM)",
            context_length=1048576,
            priority=8,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.CODING,
                TaskSpecialization.LARGE_CONTEXT,
            ],
            cost_tier=4,
            estimated_tokens_per_second=40.0,
            backend="nvidia",
        ),
        # Google Gemma 4
        ModelInfo(
            id="google/gemma-4-31b-it",
            name="Gemma 4 31B (NIM)",
            context_length=262144,
            priority=8,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.REASONING,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="google/gemma-4-26b-a4b-it",
            name="Gemma 4 26B MoE (NIM)",
            context_length=262144,
            priority=8,
            specializations=[
                TaskSpecialization.VISION,
                TaskSpecialization.REASONING,
                TaskSpecialization.GENERAL,
                TaskSpecialization.FAST,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        # Microsoft Phi
        ModelInfo(
            id="microsoft/phi-4",
            name="Phi 4 (NIM)",
            context_length=16384,
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.REASONING,
                TaskSpecialization.CODING,
            ],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="nvidia",
        ),
        # Qwen Coder variants
        ModelInfo(
            id="qwen/qwen3-coder-plus",
            name="Qwen3 Coder Plus (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
            ],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-coder-next",
            name="Qwen3 Coder Next (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.REASONING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-coder-flash",
            name="Qwen3 Coder Flash (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.FAST,
            ],
            cost_tier=2,
            estimated_tokens_per_second=70.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-coder-30b-a3b-instruct",
            name="Qwen3 Coder 30B MoE (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=65.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3-next-80b-a3b-thinking",
            name="Qwen3 Next 80B Thinking (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.CODING,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=3,
            estimated_tokens_per_second=45.0,
            backend="nvidia",
        ),
        # Qwen 3.5 variants
        ModelInfo(
            id="qwen/qwen3.5-35b-a3b",
            name="Qwen 3.5 35B MoE (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3.5-27b",
            name="Qwen 3.5 27B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
        ModelInfo(
            id="qwen/qwen3.5-9b",
            name="Qwen 3.5 9B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=1,
            estimated_tokens_per_second=75.0,
            backend="nvidia",
        ),
        # Qwen 3.5 Plus
        ModelInfo(
            id="qwen/qwen3.5-plus-20260420",
            name="Qwen 3.5 Plus (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="nvidia",
        ),
        # OpenRouter
        ModelInfo(
            id="openrouter/owl-alpha",
            name="Owl Alpha (OpenRouter)",
            context_length=262144,
            priority=6,
            specializations=[
                TaskSpecialization.REASONING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.LARGE_CONTEXT,
            ],
            cost_tier=3,
            estimated_tokens_per_second=40.0,
            backend="openrouter",
        ),
        # OpenAI GPT-OSS 20B
        ModelInfo(
            id="openai/gpt-oss-20b",
            name="GPT-OSS 20B (NIM)",
            context_length=131072,
            priority=8,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.CODING,
            ],
            cost_tier=1,
            estimated_tokens_per_second=75.0,
            backend="nvidia",
        ),
    ]
    return models
