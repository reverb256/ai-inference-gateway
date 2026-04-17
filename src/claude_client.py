"""
Claude API Client - Anthropic SDK compatibility layer

Translates Anthropic API requests to local model calls while supporting:
- Extended thinking (thinking_tokens, thinking_budget)
- Tool use with Claude format
- Artifacts and structured outputs
- Native Claude response formats
"""

import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ThinkingIntensity(Enum):
    """Thinking intensity levels mapping to model tiers."""
    AUTO = "auto"
    LOW = "low"      # 0.8B model (Haiku tier)
    MEDIUM = "medium"  # 9B model (Sonnet tier)
    HIGH = "high"    # 35B model (Opus tier)


@dataclass
class ClaudeRequest:
    """Normalized Claude API request."""
    messages: List[Dict[str, Any]]
    model: str
    max_tokens: int = 4096
    thinking_intensity: Optional[str] = None
    thinking_budget: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 1.0
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False
    system: Optional[str] = None


@dataclass
class ClaudeResponse:
    """Claude API formatted response."""
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: Dict[str, int] = field(default_factory=dict)
    thinking: Optional[Dict[str, Any]] = None


class ClaudeClient:
    """
    Client for handling Claude API requests with local model fallback.

    Translates Anthropic-style requests to the appropriate local model
    based on thinking intensity and requested model tier.
    """

    # Model tier mapping for thinking intensity
    INTENSITY_MODEL_MAP = {
        ThinkingIntensity.LOW: "qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled",
        ThinkingIntensity.MEDIUM: "qwen3.5-9b-claude-4.6-opus-reasoning-distilled",
        ThinkingIntensity.HIGH: "qwen3.5-35b-a3b",
        ThinkingIntensity.AUTO: "qwen3.5-35b-a3b",  # Default to highest tier
    }

    # Anthropic model ID to local model mapping
    # Model mapping (5 Claude options → 3 underlying local models):
    CLAUDE_MODEL_MAPPING = {
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

    def __init__(
        self,
        openai_client,
        default_model: str = "qwen3.5-4b",
    ):
        """
        Initialize Claude client.

        Args:
            openai_client: OpenAI client for backend communication
            default_model: Default model when no specific model requested
        """
        self.openai_client = openai_client
        self.default_model = default_model

    def resolve_model(
        self,
        requested_model: str,
        thinking_intensity: Optional[str] = None,
    ) -> str:
        """
        Resolve the Anthropic model ID to the actual local model.

        Args:
            requested_model: Anthropic model ID (e.g., "claude-opus-4")
            thinking_intensity: Optional thinking intensity override

        Returns:
            Local model ID to use
        """
        # If thinking intensity is specified, it overrides the model tier
        if thinking_intensity:
            try:
                intensity = ThinkingIntensity(thinking_intensity)
                return self.INTENSITY_MODEL_MAP[intensity]
            except ValueError:
                logger.warning(f"Invalid thinking intensity: {thinking_intensity}")

        # Map Anthropic model to local model
        if requested_model in self.CLAUDE_MODEL_MAPPING:
            return self.CLAUDE_MODEL_MAPPING[requested_model]

        # Otherwise use the requested model directly (might be local model ID)
        return requested_model or self.default_model

    def translate_to_openai_request(
        self,
        claude_request: ClaudeRequest,
    ) -> Dict[str, Any]:
        """
        Translate Claude API request to OpenAI format.

        Args:
            claude_request: Normalized Claude request

        Returns:
            OpenAI-formatted request dict
        """
        # Resolve the actual model to use
        model = self.resolve_model(
            claude_request.model,
            claude_request.thinking_intensity,
        )

        # Build OpenAI-compatible request
        openai_request = {
            "model": model,
            "messages": claude_request.messages,
            "max_tokens": claude_request.max_tokens,
            "temperature": claude_request.temperature,
            "top_p": claude_request.top_p,
            "stream": claude_request.stream,
        }

        # Add system prompt if provided
        if claude_request.system:
            openai_request["messages"] = [
                {"role": "system", "content": claude_request.system}
            ] + openai_request["messages"]

        # Add tools if provided
        if claude_request.tools:
            openai_request["tools"] = claude_request.tools
            if claude_request.tool_choice:
                openai_request["tool_choice"] = claude_request.tool_choice

        # Add thinking metadata for tracking
        openai_request["thinking_metadata"] = {
            "intensity": claude_request.thinking_intensity,
            "budget": claude_request.thinking_budget,
            "original_model": claude_request.model,
            "resolved_model": model,
        }

        return openai_request

    def translate_to_claude_response(
        self,
        openai_response: Dict[str, Any],
        original_model: str,
        thinking_metadata: Optional[Dict] = None,
    ) -> ClaudeResponse:
        """
        Translate OpenAI response to Claude API format.

        Args:
            openai_response: Response from OpenAI backend
            original_model: Original Anthropic model requested
            thinking_metadata: Thinking parameters from request

        Returns:
            Claude-formatted response
        """
        # Extract content from OpenAI response
        content = []
        thinking = None

        # Handle thinking content if present
        if "reasoning_content" in openai_response:
            thinking = {
                "thinking": openai_response.get("reasoning_content", ""),
                "tokens": openai_response.get("reasoning_tokens", 0),
            }

        # Main message content
        message = openai_response.get("choices", [{}])[0].get("message", {})
        if message.get("content"):
            content.append({
                "type": "text",
                "text": message["content"]
            })

        # Tool calls
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                content.append({
                    "type": "tool_use",
                    "id": tool_call.get("id", ""),
                    "name": tool_call.get("function", {}).get("name", ""),
                    "input": tool_call.get("function", {}).get("arguments", "{}"),
                })

        return ClaudeResponse(
            id=openai_response.get("id", ""),
            type="message",
            role="assistant",
            content=content,
            model=original_model,
            stop_reason=openai_response.get("choices", [{}])[0].get("finish_reason"),
            usage=openai_response.get("usage", {}),
            thinking=thinking,
        )

    async def create(
        self,
        claude_request: ClaudeRequest,
    ) -> ClaudeResponse:
        """
        Create a Claude API completion.

        Args:
            claude_request: Normalized Claude request

        Returns:
            Claude-formatted response
        """
        # Translate to OpenAI format
        openai_request = self.translate_to_openai_request(claude_request)
        thinking_metadata = openai_request.pop("thinking_metadata", None)

        logger.debug(
            f"Claude request: model={claude_request.model} "
            f"→ resolved={openai_request['model']} "
            f"intensity={thinking_metadata.get('intensity') if thinking_metadata else None}"
        )

        # Make request through OpenAI client
        openai_response = await self.openai_client.create(
            **openai_request
        )

        # Translate back to Claude format
        return self.translate_to_claude_response(
            openai_response,
            claude_request.model,
            thinking_metadata,
        )

    async def create_stream(
        self,
        claude_request: ClaudeRequest,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Create a streaming Claude API completion.

        Args:
            claude_request: Normalized Claude request

        Yields:
            Claude-formatted streaming response chunks
        """
        # Translate to OpenAI format
        openai_request = self.translate_to_openai_request(claude_request)
        thinking_metadata = openai_request.pop("thinking_metadata", None)
        openai_request["stream"] = True

        # Stream through OpenAI client
        async for chunk in await self.openai_client.stream(**openai_request):
            # Translate streaming chunk to Claude format
            if chunk.get("type") == "content_block_delta":
                yield {
                    "type": "content_block_delta",
                    "index": chunk.get("index", 0),
                    "delta": chunk.get("delta", {}),
                }
            elif chunk.get("type") == "content_block_stop":
                yield {
                    "type": "content_block_stop",
                    "index": chunk.get("index", 0),
                }
            else:
                yield chunk


def create_claude_client(openai_client, **kwargs) -> ClaudeClient:
    """Factory function to create Claude client."""
    return ClaudeClient(openai_client, **kwargs)
