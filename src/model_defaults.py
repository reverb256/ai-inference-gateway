"""
Model-specific default parameters for Qwen3.5 models.

Provides optimal temperature, top_p, max_tokens, and other parameters
based on model size and capabilities.

Based on:
- docs/qwen3.5-best-practices.md
- https://unsloth.ai/docs/models/qwen3.5
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# Qwen3.5 optimal parameters for thinking vs non-thinking modes
# Based on Unsloth documentation: https://unsloth.ai/docs/models/qwen3.5
QWEN_THINKING_MODE_PARAMS = {
    "thinking": {
        "general": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },
        "coding": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
        },
    },
    "non_thinking": {
        "general": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        },
        "reasoning": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        },
    },
}


# Vision-specific temperature overrides
VISION_TEMPERATURE_OVERRIDES = {
    "0.8b": 0.7,  # Vision requires lower temp for consistency
    "2b": 0.7,
}


# Model parameter patterns
# These map model name patterns to their optimal defaults
# Based on Unsloth documentation: https://unsloth.ai/docs/models/qwen3.5
MODEL_DEFAULTS = {
    # 35B-A3B (Mixture-of-Experts) - Best for long context, hybrid reasoning
    # Maximum context: 262,144 tokens (256K), can extend to 1M via YaRN
    # Adequate output length: 32,768 tokens for most queries
    "35b-a3b": {
        "temperature": 1.0,  # Thinking mode default
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,  # Thinking mode default (0.0-2.0 range)
        "max_tokens": 32768,  # Adequate output length
        "context_length": 262144,  # 256K
        "thinking_enabled_default": True,
        "supports_thinking_toggle": True,
        "description": "Hybrid reasoning MoE, optimal for long-context tasks",
        "use_case": "long_context_reasoning",
        "quantization": "Q4_K_M",
    },
    # 27B - Dense quality priority
    "27b": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 32768,
        "context_length": 262144,  # 256K with KV cache
        "thinking_enabled_default": True,
        "supports_thinking_toggle": True,
        "description": "Dense quality priority, more accurate than 35B",
        "use_case": "high_quality",
        "quantization": "Q4_K_M",
    },
    # 9B models (base + distilled) - General reasoning
    # Small models have reasoning DISABLED by default, enable via enable_thinking
    "9b": {
        "temperature": 0.6,  # Non-thinking default
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,  # Non-thinking default
        "max_tokens": 16384,  # Adequate for 9B
        "context_length": 262144,  # 256K context
        "thinking_enabled_default": False,  # Small models: disabled by default
        "supports_thinking_toggle": True,
        "description": "General reasoning, enable_thinking for reasoning mode",
        "use_case": "general",
        "quantization": "IQ4_NL",
    },
    # Distilled 9B variants (Claude-style) - Reasoning distilled
    "9b-claude": {
        "temperature": 1.0,  # Thinking mode
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 16384,
        "context_length": 262144,
        "thinking_enabled_default": True,  # Reasoning-distilled, enabled by default
        "supports_thinking_toggle": True,
        "description": "Claude-distilled reasoning, thinking enabled by default",
        "use_case": "reasoning",
        "quantization": "IQ4_NL",
        "prompt_style": "claude",
    },
    # CROW 9B distill
    "crow-9b": {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": 16384,
        "context_length": 262144,
        "thinking_enabled_default": True,
        "supports_thinking_toggle": True,
        "description": "Jackrong's CROW distill, CoT with <think> tags",
        "use_case": "reasoning",
        "quantization": "IQ4_NL",
        "prompt_style": "cot",
    },
    # 4B - Multimodal agents, modest GPUs
    "4b": {
        "temperature": 0.6,  # Non-thinking
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": 8192,
        "context_length": 262144,  # 256K context
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "description": "Multimodal agents, 8GB GPUs, enable_thinking for reasoning",
        "use_case": "multimodal",
        "quantization": "Q4_K_S",
    },
    # 2B - Edge devices, basic tasks
    "2b": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": 4096,
        "context_length": 262144,  # 256K context
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "description": "Edge devices, basic tasks",
        "use_case": "edge",
        "quantization": "IQ4_NL",
    },
    # 0.8B - Edge devices, simple tasks
    "0.8b": {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "presence_penalty": 0.0,
        "max_tokens": 2048,
        "context_length": 262144,  # 256K context
        "thinking_enabled_default": False,
        "supports_thinking_toggle": True,
        "description": "Edge devices, simple tasks, enable_thinking for reasoning",
        "use_case": "edge",
        "quantization": "IQ4_NL",
    },
}


def get_model_defaults(model_id: str) -> Dict[str, Any]:
    """
    Get optimal default parameters for a model.

    Args:
        model_id: Model identifier (e.g., "qwen3.5-9b", "qwen/qwen3.5-35b-a3b")

    Returns:
        Dict with default parameters:
        - temperature: float
        - top_p: float
        - max_tokens: int
        - context_length: int
        - description: str
        - use_case: str
    """
    model_lower = model_id.lower()

    # Match model pattern
    for pattern, defaults in MODEL_DEFAULTS.items():
        if pattern in model_lower:
            logger.debug(f"Matched model '{model_id}' to pattern '{pattern}'")
            return defaults.copy()

    # Fallback to sensible defaults
    logger.warning(
        f"No specific defaults for model '{model_id}', using generic defaults"
    )
    return {
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 4096,
        "context_length": 8192,
        "description": "Generic defaults",
        "use_case": "unknown",
        "thinking_enabled_default": False,
        "supports_thinking_toggle": False,
    }


def get_qwen_thinking_params(
    thinking_enabled: bool,
    task_type: str = "general",
) -> Dict[str, Any]:
    """
    Get optimal Qwen3.5 parameters based on thinking mode and task type.

    Based on Unsloth documentation:
    https://unsloth.ai/docs/models/qwen3.5

    Args:
        thinking_enabled: Whether thinking/reasoning mode is enabled
        task_type: Task type ("general", "coding", "reasoning")

    Returns:
        Dict with optimal parameters for the mode
    """
    mode = "thinking" if thinking_enabled else "non_thinking"
    task = task_type if task_type in ("general", "coding", "reasoning") else "general"
    return QWEN_THINKING_MODE_PARAMS.get(mode, {}).get(task, {})


def detect_task_type(messages: list) -> str:
    """
    Detect task type from messages for optimal parameter selection.

    Args:
        messages: List of message dicts with 'content' and 'role'

    Returns:
        Task type: "coding", "general", or "reasoning"
    """
    # Combine all message content
    all_content = ""
    for msg in messages:
        if isinstance(msg.get("content"), str):
            all_content += msg["content"].lower()
        elif isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "text":
                    all_content += block.get("text", "").lower()

    # Coding indicators
    coding_keywords = [
        "code", "function", "class", "def ", "import ",
        "programming", "debug", "syntax", "algorithm",
        "implement", "refactor", "variable", "```",
    ]

    # Reasoning indicators
    reasoning_keywords = [
        "analyze", "compare", "evaluate", "explain why",
        "reasoning", "logic", "inference", "conclusion",
        "premise", "argument", "deduction",
    ]

    coding_score = sum(1 for kw in coding_keywords if kw in all_content)
    reasoning_score = sum(1 for kw in reasoning_keywords if kw in all_content)

    if coding_score >= 2:
        return "coding"
    elif reasoning_score >= 2:
        return "reasoning"
    return "general"


def apply_model_defaults(
    model_id: str,
    request_params: Dict[str, Any],
    override: bool = False,
    is_vision_request: bool = False,
) -> Dict[str, Any]:
    """
    Apply model-specific defaults to request parameters.

    For Qwen3.5 models, also applies optimal parameters based on thinking mode.

    Args:
        model_id: Model identifier
        request_params: Original request parameters
        override: If True, defaults override user params.
                   If False (default), only fill missing values.
        is_vision_request: If True, use vision-specific temperature

    Returns:
        Updated request parameters with defaults applied
    """
    defaults = get_model_defaults(model_id)
    result = request_params.copy()

    # Detect if this is a Qwen model with thinking support
    is_qwen = "qwen" in model_id.lower()
    thinking_enabled = False

    # Determine thinking mode conditionally based on task and model
    if is_qwen:
        # Check if thinking is explicitly requested
        if "thinking" in result:
            thinking_cfg = result["thinking"]
            if isinstance(thinking_cfg, dict):
                # Check for enable_thinking or type="enabled"
                thinking_enabled = thinking_cfg.get("enable_thinking",
                    thinking_cfg.get("type") == "enabled")
        else:
            # Conditional thinking: Enable only for complex reasoning tasks on larger models
            model_size = None
            if "0.8b" in model_id or "1b" in model_id:
                model_size = "tiny"
            elif "2b" in model_id or "3b" in model_id:
                model_size = "small"
            elif "4b" in model_id or "6b" in model_id or "7b" in model_id:
                model_size = "medium"
            elif "9b" in model_id or "14b" in model_id:
                model_size = "large"
            elif "27b" in model_id or "32b" in model_id or "35b" in model_id:
                model_size = "xlarge"

            # Detect task type from messages
            messages = result.get("messages", [])
            task_type = detect_task_type(messages)

            # Enable thinking only for:
            # 1. Large/XL models on reasoning tasks
            # 2. Explicit model default (if set to True)
            model_default_thinking = defaults.get("thinking_enabled_default", False)

            if model_default_thinking:
                # Model explicitly wants thinking on by default
                thinking_enabled = True
            elif model_size in ["large", "xlarge"] and task_type == "reasoning":
                # Large models doing complex reasoning benefit from thinking
                thinking_enabled = True
                logger.debug(
                    f"Conditionally enabled thinking for {model_id} "
                    f"(size={model_size}, task={task_type})"
                )
            else:
                # Default: thinking disabled for fast responses
                thinking_enabled = False

    # For vision requests, override temperature to be more conservative
    if is_vision_request:
        for pattern, vision_temp in VISION_TEMPERATURE_OVERRIDES.items():
            if pattern in model_id.lower():
                defaults["temperature"] = vision_temp
                logger.debug(
                    f"Using vision-specific temperature {vision_temp} for {model_id}"
                )
                break

    # Apply Qwen thinking mode optimal parameters
    if is_qwen and "qwen" in model_id.lower():
        messages = result.get("messages", [])
        task_type = detect_task_type(messages)
        thinking_params = get_qwen_thinking_params(thinking_enabled, task_type)

        # Apply thinking mode parameters (only if not already set)
        for param, value in thinking_params.items():
            if override or param not in result:
                result[param] = value
                logger.debug(
                    f"Applied Qwen thinking mode param: {param}={value} "
                    f"(thinking={thinking_enabled}, task={task_type})"
                )

    # Apply defaults
    if override:
        # Force defaults
        result.setdefault("temperature", defaults["temperature"])
        result.setdefault("top_p", defaults["top_p"])
        result.setdefault("top_k", defaults.get("top_k", 20))
        result.setdefault("max_tokens", defaults["max_tokens"])
    else:
        # Only fill missing values
        if "temperature" not in result:
            result["temperature"] = defaults["temperature"]
        if "top_p" not in result:
            result["top_p"] = defaults["top_p"]
        if "top_k" not in result:
            result["top_k"] = defaults.get("top_k", 20)
        if "max_tokens" not in result:
            result["max_tokens"] = defaults["max_tokens"]

    # For small Qwen models, add enable_thinking to thinking dict if needed
    if is_qwen and thinking_enabled:
        if "thinking" not in result:
            result["thinking"] = {}
        if isinstance(result["thinking"], dict):
            # Ensure enable_thinking is set for small models
            if "enable_thinking" not in result["thinking"]:
                result["thinking"]["enable_thinking"] = True

    # Log what we're using
    log_parts = [
        f"Model '{model_id}'",
        f"temperature={result.get('temperature')}",
        f"top_p={result.get('top_p')}",
        f"max_tokens={result.get('max_tokens')}",
    ]
    if is_qwen:
        log_parts.append(f"thinking={thinking_enabled}")
    if is_vision_request:
        log_parts.append("[vision request]")

    logger.info(", ".join(log_parts))

    return result


def get_model_recommendation(model_id: str) -> Dict[str, str]:
    """
    Get recommendation details for a model.

    Args:
        model_id: Model identifier

    Returns:
        Dict with:
        - description: What the model is best for
        - use_case: Recommended use case
        - quantization: Recommended quantization
        - prompt_style: Recommended prompt style (if applicable)
    """
    defaults = get_model_defaults(model_id)

    return {
        "description": defaults.get("description", "Unknown"),
        "use_case": defaults.get("use_case", "general"),
        "quantization": defaults.get("quantization", "Q4_K_M"),
        "prompt_style": defaults.get("prompt_style", "standard"),
        "context_length": defaults.get("context_length", 8192),
    }


def suggest_model_for_task(task: str, context_length: int = 4096) -> str:
    """
    Suggest the best Qwen3.5 model for a given task.

    Args:
        task: Task description (e.g., "reasoning", "chat", "code")
        context_length: Required context length in tokens

    Returns:
        Recommended model ID
    """
    task_lower = task.lower()

    # Long context required
    if context_length > 128000:
        return "qwen3.5-35b-a3b"  # Best for 256K context

    # Complex reasoning
    if any(
        keyword in task_lower
        for keyword in ["reasoning", "complex", "analysis", "cortex"]
    ):
        if context_length > 64000:
            return "qwen3.5-35b-a3b"
        return "qwen3.5-9b-claude-4.6-opus-distilled-32k"

    # Code generation
    if "code" in task_lower or "programming" in task_lower:
        return "qwen3.5-9b"  # Good balance of quality/speed

    # Fast responses
    if any(keyword in task_lower for keyword in ["fast", "quick", "simple"]):
        return "qwen3.5-4b"

    # Edge deployment
    if "edge" in task_lower or "mobile" in task_lower:
        return "qwen3.5-0.8b"

    # Default: balanced option
    return "qwen3.5-9b"


# Export for use in other modules
__all__ = [
    "get_model_defaults",
    "apply_model_defaults",
    "get_model_recommendation",
    "suggest_model_for_task",
    "get_qwen_thinking_params",
    "detect_task_type",
    "MODEL_DEFAULTS",
    "QWEN_THINKING_MODE_PARAMS",
]
