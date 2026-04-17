"""
Anthropic API compatibility service.

Handles translation between OpenAI and Anthropic API formats,
including extended thinking, effort levels, and response mapping.
"""

import logging
from typing import Optional


logger = logging.getLogger(__name__)

# Effort level to budget_tokens mapping
EFFORT_BUDGET_MAP = {
    "low": 5_000,       # Quick responses, minimal reasoning
    "medium": 15_000,   # Balanced reasoning
    "high": 50_000,     # Deep analysis, extensive reasoning
}


def parse_thinking_params(body: dict) -> tuple:
    """
    Parse thinking/effort parameters from an Anthropic request.

    Returns:
        (thinking_budget, thinking_intensity, thinking_type)
    """
    thinking_budget = None
    thinking_intensity = None
    thinking_type = None

    if "thinking" in body:
        thinking = body["thinking"]
        if isinstance(thinking, dict):
            thinking_intensity = thinking.get("intensity")
            thinking_budget = thinking.get("budget_tokens")
            thinking_type = thinking.get("type", "enabled")
        elif isinstance(thinking, str):
            thinking_intensity = thinking
            thinking_type = "enabled"
    elif "thinking_intensity" in body:
        thinking_intensity = body["thinking_intensity"]
        thinking_type = "enabled"

    # Map effort levels to budget_tokens if not explicitly set
    if thinking_intensity and not thinking_budget:
        thinking_budget = EFFORT_BUDGET_MAP.get(thinking_intensity)
        logger.info(
            f"Thinking intensity '{thinking_intensity}' → budget_tokens={thinking_budget}"
        )

    return thinking_budget, thinking_intensity, thinking_type


def apply_thinking_to_body(body: dict, thinking_budget, thinking_type, thinking_intensity) -> dict:
    """Apply parsed thinking parameters back to the request body."""
    if thinking_budget is not None or thinking_type:
        if "thinking" not in body or not isinstance(body["thinking"], dict):
            body["thinking"] = {}
        if thinking_type:
            body["thinking"]["type"] = thinking_type
        if thinking_budget is not None:
            body["thinking"]["budget_tokens"] = thinking_budget
        if thinking_intensity:
            body["thinking"]["intensity"] = thinking_intensity
    return body


def detect_system_prompt_category(messages: list, system_prompts_config=None) -> Optional[str]:
    """
    Detect which system prompt category to use based on message content.

    Returns:
        Category name or None
    """
    text = " ".join(
        msg.get("content", "")
        for msg in messages
        if isinstance(msg.get("content", ""), str)
    ).lower()

    if any(kw in text for kw in ["def ", "class ", "function", "import ", "code", "function(", "return "]):
        return "coding"
    if any(kw in text for kw in ["agent", "workflow", "multi-step", "plan", "execute"]):
        return "agentic"
    if any(kw in text for kw in ["reason", "think", "step", "explain", "why", "how"]):
        return "reasoning"
    if any(kw in text for kw in ["quickly", "asap", "fast", "brief", "short"]):
        return "fast"
    return None
