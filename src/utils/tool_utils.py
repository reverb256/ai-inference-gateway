"""
Tool calling utility functions for AI Gateway.

Provides helper functions for detecting and handling tool calls in
both OpenAI and Anthropic response formats, enabling agentic workflows.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def has_tool_calls_openai(response: Dict[str, Any]) -> bool:
    """
    Check if an OpenAI-format response contains tool calls.

    OpenAI tool calls appear in choices[].message.tool_calls:
    {
        "choices": [{
            "message": {
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }]
            }
        }],
        "finish_reason": "tool_calls"
    }

    Args:
        response: OpenAI API response dictionary

    Returns:
        True if response contains tool calls, False otherwise
    """
    if not response or not isinstance(response, dict):
        return False

    choices = response.get("choices", [])
    if not choices:
        return False

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return False

    message = first_choice.get("message", {})
    tool_calls = message.get("tool_calls")

    # Check if tool_calls exists and is non-empty
    if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
        return True

    # Also check finish_reason
    finish_reason = first_choice.get("finish_reason", "")
    return finish_reason == "tool_calls"


def has_tool_calls_anthropic(response: Dict[str, Any]) -> bool:
    """
    Check if an Anthropic-format response contains tool use blocks.

    Anthropic tool use appears in content array:
    {
        "content": [
            {"type": "text", "text": "..."},
            {"type": "tool_use", "id": "toolu_abc123", "name": "get_weather", "input": {}}
        ],
        "stop_reason": "tool_use"
    }

    Args:
        response: Anthropic Messages API response dictionary

    Returns:
        True if response contains tool_use blocks, False otherwise
    """
    if not response or not isinstance(response, dict):
        return False

    content = response.get("content", [])
    if not content:
        return False

    # Check for tool_use blocks
    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return True

    # Also check stop_reason
    stop_reason = response.get("stop_reason", "")
    return stop_reason == "tool_use"


def has_tool_calls(response: Dict[str, Any], response_format: str = "openai") -> bool:
    """
    Generic tool call detection - auto-detects response format.

    Args:
        response: API response dictionary
        response_format: Either "openai" or "anthropic" (default: "openai")

    Returns:
        True if response contains tool calls, False otherwise
    """
    if response_format == "anthropic":
        return has_tool_calls_anthropic(response)
    return has_tool_calls_openai(response)


def extract_tool_calls_openai(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool calls from OpenAI-format response.

    Returns:
        List of tool call dictionaries with keys: id, type, name, arguments (as string)
    """
    if not response or not isinstance(response, dict):
        return []

    choices = response.get("choices", [])
    if not choices:
        return []

    first_choice = choices[0]
    message = first_choice.get("message", {})
    tool_calls = message.get("tool_calls", [])

    extracted_calls = []
    for tool_call in tool_calls:
        extracted = {
            "id": tool_call.get("id"),
            "type": tool_call.get("type", "function"),
            "name": tool_call.get("function", {}).get("name"),
            "arguments": tool_call.get("function", {}).get("arguments", "{}"),
        }
        extracted_calls.append(extracted)

    return extracted_calls


def extract_tool_calls_anthropic(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract tool use blocks from Anthropic-format response.

    Returns:
        List of tool use dictionaries with keys: id, name, input
    """
    if not response or not isinstance(response, dict):
        return []

    content = response.get("content", [])
    extracted_calls = []

    for block in content:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            extracted = {
                "id": block.get("id"),
                "name": block.get("name"),
                "input": block.get("input", {}),
            }
            extracted_calls.append(extracted)

    return extracted_calls


def create_tool_result_openai(
    tool_call_id: str,
    result: str,
    tool_type: str = "function"
) -> Dict[str, Any]:
    """
    Create a tool result message in OpenAI format.

    Args:
        tool_call_id: The ID of the tool call being responded to
        result: The result content (string or dict)
        tool_type: The type of tool (usually "function")

    Returns:
        OpenAI-formatted tool result message
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": str(result),
        "name": tool_type,
    }


def create_tool_result_anthropic(
    tool_use_id: str,
    result: Any
) -> Dict[str, Any]:
    """
    Create a tool result block in Anthropic format.

    Args:
        tool_use_id: The ID of the tool_use block being responded to
        result: The result content (typically a string or dict with "type": "text")

    Returns:
        Anthropic-formatted tool result content block
    """
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {"type": "text", "text": str(result)}
        ]
    }


def is_tool_response_format(message: Dict[str, Any]) -> bool:
    """
    Check if a message is a tool response (not a tool call).

    Used to distinguish between user messages and tool result messages
    in the conversation history.

    Args:
        message: Message dictionary to check

    Returns:
        True if the message is a tool response, False otherwise
    """
    if not message:
        return False

    role = message.get("role", "")
    return role == "tool"
