"""
Intelligent Router for AI Inference Gateway.

Routes requests to appropriate models based on:
- Token count estimation
- Task type detection (coding, agentic, general, fast, large_context)
- Latency tracking and overload detection
- Model specialization matching
- Cost tier considerations
- Category-based routing (inspired by oh-my-opencode)
"""

import logging
import re
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


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
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "simple_qa"],
    },
    "qwen3.5-0.8b": {
        "max_tokens": 8192,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "simple_qa"],
    },
    "qwen3.5-2b": {
        "max_tokens": 8192,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "ultra_fast",
        "recommended_for": ["fast", "summarization"],
    },
    "qwen3.5-2b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 8192,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "fast",
        "recommended_for": ["fast", "reasoning"],
    },

    # ========== SMALL MODELS (4B-9B) ==========
    # Good balance of speed and quality
    "qwen3.5-4b": {
        "max_tokens": 8192,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "fast",
        "recommended_for": ["general", "chat"],
    },
    "qwen3.5-4b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 16384,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "fast",
        "recommended_for": ["coding", "reasoning"],
    },
    "qwen3.5-4b-claude-4.6-opus-distilled-32k@q8_0": {
        "max_tokens": 16384,
        "context_length": 32768,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "fast",
        "recommended_for": ["general", "chat"],
    },
    "qwen3.5-9b": {
        "max_tokens": 16384,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "balanced",
        "recommended_for": ["general", "coding"],
    },
    "qwen3.5-9b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 16384,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "balanced",
        "recommended_for": ["coding", "complex_reasoning"],
    },
    "qwen3.5-9b-claude-4.6-opus-distilled-32k": {
        "max_tokens": 16384,
        "context_length": 32768,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "balanced",
        "recommended_for": ["general", "long_context"],
    },

    # ========== LARGE MODELS (27B-35B) ==========
    # Highest quality, slower but more capable
    "qwen3.5-27b": {
        "max_tokens": 32768,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
        "speed_tier": "slow",
        "recommended_for": ["complex_reasoning", "creative"],
    },
    "qwen3.5-27b-claude-4.6-opus-reasoning-distilled": {
        "max_tokens": 32768,
        "context_length": 262144,
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "speed_tier": "slow",
        "recommended_for": ["complex_reasoning", "agentic"],
    },
    "qwen3.5-35b-a3b": {
        "max_tokens": 32768,
        "context_length": 262144,
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
        "context_length": 262144,
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


class TaskSpecialization(Enum):
    """Task specialization types for intelligent routing."""

    CODING = "coding"
    AGENTIC = "agentic"
    GENERAL = "general"
    FAST = "fast"
    LARGE_CONTEXT = "large_context"
    VISION = "vision"


@dataclass
class ModelInfo:
    """Information about an available model."""

    id: str
    name: str
    context_length: int = 262144  # Qwen3.5 supports 256K!
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
    """

    def __init__(
        self,
        models: List[ModelInfo],
        latency_tracker: Optional[LatencyTracker] = None,
    ):
        """
        Initialize router.

        Args:
            models: List of available models
            latency_tracker: Optional latency tracker for performance-based routing
        """
        self.models = {model.id: model for model in models}
        self.latency_tracker = latency_tracker or LatencyTracker()
        self.claude_model_mapping = self._build_claude_mapping()
        # Active request tracking for smart load balancing
        self.active_requests: Dict[str, Dict] = (
            {}
        )  # request_id -> {model, backend, stream, start_time}
        self.max_concurrent_streams = 1  # Backend can handle 1 stream at a time

        # Backend health cache
        self._backend_health: Dict[str, bool] = {
            "llama-cpp": True,
            "zai": True,
            "nvidia": True,
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

        ZAI fallback chain (when Local backend down/capacity): glm-5 → glm-4.7 → glm-4.5-air
        """
        return {
            # Haiku tier → Local 0.8B Opus reasoning distilled (fastest)
            "claude-haiku-4": "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled",
            "claude-haiku-4-20250514": "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled",
            # Sonnet tier → Local 9B Opus reasoning distilled
            "claude-sonnet-4-20250514": "qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
            "claude-sonnet-4": "qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
            # Sonnet extended context variant (same underlying model)
            "claude-sonnet-4-20250514-1m": "qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
            # Opus tier → Local 35B (best quality local)
            "claude-opus-4-20250514": "qwen3.5-35b-a3b",
            "claude-opus-4": "qwen3.5-35b-a3b",
            # Opus extended context variant (same underlying model)
            "claude-opus-4-20250514-1m": "qwen3.5-35b-a3b",
        }

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

        # If Local backend (llama-cpp) is down, route directly to ZAI
        if not local_backend_healthy:
            logger.info("Local backend (llama-cpp) is down, auto-failing over to ZAI")
            estimated_tokens = self.estimate_tokens(messages)

            # Get available ZAI models
            zai_models = [m for m in self.models.values() if m.backend == "zai"]
            if zai_models:
                # Sort by priority and pick the best one
                best_zai = max(zai_models, key=lambda m: m.priority)
                specialization = self.detect_specialization(messages)

                # If client requested a specific model, try to map it
                if requested_model and requested_model in self.claude_model_mapping:
                    mapped_model = self.claude_model_mapping[requested_model]
                    model_info = self.models.get(mapped_model)
                    if model_info and model_info.backend == "zai":
                        return RouteDecision(
                            model=mapped_model,
                            confidence=1.0,
                            reason=f"Local backend down, using ZAI fallback for {requested_model}",
                            estimated_tokens=estimated_tokens,
                            backend="zai",
                            specialization=specialization,
                            expected_latency_ms=model_info.estimated_tokens_per_second
                            * estimated_tokens
                            / 1000,
                        )

                return RouteDecision(
                    model=best_zai.id,
                    confidence=0.95,
                    reason="Local backend down (auto-failover to ZAI)",
                    estimated_tokens=estimated_tokens,
                    backend="zai",
                    specialization=specialization,
                    expected_latency_ms=best_zai.estimated_tokens_per_second
                    * estimated_tokens
                    / 1000,
                )
            else:
                # No ZAI models available - this is an error condition
                logger.error("Local backend down and no ZAI models available!")
                # Return default model anyway (will likely fail)
                return RouteDecision(
                    model="qwen/qwen3.5-9b",
                    confidence=0.1,
                    reason="Local backend down, no ZAI fallback available",
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

            # Urgency adjustment
            if urgency == "fast":
                # Prefer faster models
                candidate.score /= candidate.expected_latency_ms / 1000
            elif urgency == "quality":
                # Prefer higher cost tier (better quality)
                candidate.score *= 1 + model_info.cost_tier * 0.1

        # Sort by score descending
        return sorted(candidates, key=lambda c: c.score, reverse=True)


def create_default_router() -> Router:
    """Create router with default model configuration."""
    models = [
        # ========================================================================
        # Local llama.cpp models - Primary backends
        # ========================================================================
        # Gemma 4 E2B - Always-on fallback on 3060 Ti (port 1235)
        ModelInfo(
            id="gemma-4-e2b-it",
            name="Gemma 4 E2B Instruct",
            context_length=65536,  # 65K with Q4_0 KV
            priority=11,  # Highest priority — always available
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=0,  # Free, local
            estimated_tokens_per_second=60.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 35B A3B - Largest local model, best for complex tasks
        ModelInfo(
            id="qwen3.5-35b-a3b",
            name="Qwen 3.5 35B A3B",
            context_length=262144,  # 256K
            priority=10,  # Highest priority local model
            specializations=[
                TaskSpecialization.LARGE_CONTEXT,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=3,
            estimated_tokens_per_second=35.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 27B - Large context, general purpose
        ModelInfo(
            id="qwen3.5-27b",
            name="Qwen 3.5 27B",
            context_length=262144,  # 256K
            priority=9,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.LARGE_CONTEXT,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=2,
            estimated_tokens_per_second=45.0,
            backend="llama-cpp",
        ),
        # CROW 9B Opus 4.6 Distill - Claude Opus distilled for reasoning
        ModelInfo(
            id="crow-9b-opus-4.6-distill-heretic_qwen3.5-i1",
            name="CROW 9B Opus 4.6 Distill Heretic",
            context_length=32768,  # 32K
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=55.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 9B Claude 4.6 Opus Reasoning Distilled
        ModelInfo(
            id="qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
            name="Qwen 3.5 9B Claude Opus Reasoning",
            context_length=32768,  # 32K
            priority=8,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=2,
            estimated_tokens_per_second=58.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 9B - Standard base model
        ModelInfo(
            id="qwen3.5-9b",
            name="Qwen 3.5 9B",
            context_length=262144,  # 256K
            priority=7,
            specializations=[
                TaskSpecialization.GENERAL,
                TaskSpecialization.FAST,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=1,
            estimated_tokens_per_second=65.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 4B Claude 4.6 Opus Distilled (q8_0 - higher quality)
        ModelInfo(
            id="qwen3.5-4b-claude-4.6-opus-distilled-32k@q8_0",
            name="Qwen 3.5 4B Claude Opus Distilled (q8)",
            context_length=32768,  # 32K
            priority=7,
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=1,
            estimated_tokens_per_second=70.0,
            backend="llama-cpp",
        ),
        # CROW 4B Opus 4.6 Distill - Small but capable
        ModelInfo(
            id="crow-4b-opus-4.6-distill-heretic_qwen3.5-i1",
            name="CROW 4B Opus 4.6 Distill Heretic",
            context_length=32768,  # 32K
            priority=6,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.CODING,
            ],
            cost_tier=1,
            estimated_tokens_per_second=75.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 4B - Fast general purpose
        ModelInfo(
            id="qwen3.5-4b",
            name="Qwen 3.5 4B",
            context_length=32768,  # 32K
            priority=6,
            specializations=[
                TaskSpecialization.FAST,
                TaskSpecialization.GENERAL,
                TaskSpecialization.VISION,  # All Qwen 3.5 support vision
            ],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 2B - Very fast
        ModelInfo(
            id="qwen3.5-2b-claude-4.6-opus-reasoning-distilled",
            name="Qwen 3.5 2B Claude Reasoning",
            context_length=32768,  # 32K
            priority=5,
            specializations=[TaskSpecialization.FAST, TaskSpecialization.VISION],  # All Qwen 3.5 support vision
            cost_tier=1,
            estimated_tokens_per_second=90.0,
            backend="llama-cpp",
        ),
        ModelInfo(
            id="qwen3.5-2b",
            name="Qwen 3.5 2B",
            context_length=32768,  # 32K
            priority=5,
            specializations=[TaskSpecialization.FAST, TaskSpecialization.VISION],  # All Qwen 3.5 support vision
            cost_tier=1,
            estimated_tokens_per_second=95.0,
            backend="llama-cpp",
        ),
        # Qwen 3.5 0.8B - Tiny, fastest
        ModelInfo(
            id="qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled",
            name="Qwen 3.5 0.8B Claude Reasoning",
            context_length=32768,  # 32K
            priority=4,
            specializations=[TaskSpecialization.FAST, TaskSpecialization.VISION],  # All Qwen 3.5 support vision
            cost_tier=1,
            estimated_tokens_per_second=100.0,
            backend="llama-cpp",
        ),
        ModelInfo(
            id="qwen3.5-0.8b",
            name="Qwen 3.5 0.8B",
            context_length=32768,  # 32K
            priority=4,
            specializations=[TaskSpecialization.FAST, TaskSpecialization.VISION],  # All Qwen 3.5 support vision
            cost_tier=1,
            estimated_tokens_per_second=110.0,
            backend="llama-cpp",
        ),
        # ZAI models - Fallback priority order: glm-5 → glm-4.7 → glm-4.5-air
        ModelInfo(
            id="glm-5",
            name="GLM-5",
            context_length=200000,
            priority=9,  # Highest ZAI priority (Opus tier fallback)
            specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.GENERAL],
            cost_tier=4,
            estimated_tokens_per_second=40.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-5.1",
            name="GLM-5.1",
            context_length=200000,
            priority=10,  # Highest priority — primary ZAI model
            specializations=[TaskSpecialization.AGENTIC, TaskSpecialization.GENERAL, TaskSpecialization.CODING],
            cost_tier=4,
            estimated_tokens_per_second=40.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.7",
            name="GLM-4.7",
            context_length=200000,
            priority=8,  # Second ZAI priority (Sonnet tier fallback)
            specializations=[TaskSpecialization.CODING, TaskSpecialization.GENERAL],
            cost_tier=3,
            estimated_tokens_per_second=50.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.5-air",
            name="GLM-4.5 Air",
            context_length=132000,
            priority=7,  # Third ZAI priority (Haiku tier fallback)
            specializations=[TaskSpecialization.FAST],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="zai",
        ),
        ModelInfo(
            id="glm-4.6v",
            name="GLM-4.6v",
            context_length=200000,
            priority=6,  # Lower priority (vision specialist)
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
            id="glm-4-flash",
            name="GLM-4 Flash",
            context_length=128000,
            priority=5,  # Lowest ZAI priority
            specializations=[TaskSpecialization.FAST],
            cost_tier=1,
            estimated_tokens_per_second=80.0,
            backend="zai",
        ),
        # ========================================================================
        # NVIDIA NIM models - Cloud-hosted via NVIDIA NIM API
        # ========================================================================
        ModelInfo(
            id="nvidia/llama-3.3-nemotron-super-49b-v1",
            name="Nemotron-Super-49B (NIM)",
            context_length=32768,
            priority=15,  # High priority for JSON/extraction tasks
            specializations=[
                TaskSpecialization.CODING,
                TaskSpecialization.AGENTIC,
                TaskSpecialization.GENERAL,
            ],
            cost_tier=2,
            estimated_tokens_per_second=60.0,
            backend="nvidia",
        ),
    ]

    return Router(models=models)
