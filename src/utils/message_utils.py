"""
Message utility functions for AI Gateway.

Provides helper functions for extracting and manipulating message content
from various API formats (OpenAI, Anthropic, etc.).
"""

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def extract_last_user_message(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract the last user message from a message list.

    This is a common pattern across multiple middleware and endpoints.
    Messages are processed in reverse order to find the most recent user message.

    Args:
        messages: List of message dictionaries with 'role' and 'content' fields

    Returns:
        The last user message dict, or None if no user message found

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "First question"},
        ...     {"role": "assistant", "content": "First answer"},
        ...     {"role": "user", "content": "Second question"}
        ... ]
        >>> extract_last_user_message(messages)
        {"role": "user", "content": "Second question"}
    """
    if not messages or not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg

    return None


def extract_message_content(message: Dict[str, Any]) -> Optional[str]:
    """
    Extract text content from a message, handling multiple content formats.

    Supports:
    - Plain string content
    - Multi-modal content arrays (text, image, etc.)
    - Content with annotations

    Args:
        message: Message dictionary with 'content' field

    Returns:
        Extracted text content as a string, or None if no text content found

    Example:
        >>> msg = {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        >>> extract_message_content(msg)
        "Hello"
    """
    if not message or not isinstance(message, dict):
        return None

    content = message.get("content", "")

    # Plain string content
    if isinstance(content, str):
        return content

    # Multi-modal content array
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                text_parts.append(text)

        return " ".join(text_parts) if text_parts else None

    return None


def extract_user_query_from_messages(
    messages: List[Dict[str, Any]], default: Optional[str] = None
) -> Optional[str]:
    """
    Extract the user's query text from a message list.

    This combines finding the last user message and extracting its content,
    which is the most common pattern for query extraction.

    Args:
        messages: List of message dictionaries
        default: Optional default value if no user message found

    Returns:
        The user's query text, or default if not found

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "What is NixOS?"}
        ... ]
        >>> extract_user_query_from_messages(messages)
        "What is NixOS?"
    """
    last_user_msg = extract_last_user_message(messages)

    if last_user_msg:
        content = extract_message_content(last_user_msg)
        if content:
            return content

    return default


def extract_user_query_from_request_body(body: Dict[str, Any]) -> Optional[str]:
    """
    Extract user query from a request body dictionary.

    Handles both OpenAI chat completions format and completions format.

    Args:
        body: Request body dictionary

    Returns:
        Extracted user query, or None if not found

    Example:
        >>> body = {"messages": [{"role": "user", "content": "Hello"}]}
        >>> extract_user_query_from_request_body(body)
        "Hello"

        >>> body = {"prompt": "Explain quantum computing"}
        >>> extract_user_query_from_request_body(body)
        "Explain quantum computing"
    """
    if not body or not isinstance(body, dict):
        return None

    # Try messages format first (chat completions)
    if "messages" in body:
        messages = body.get("messages")
        if messages and isinstance(messages, list):
            return extract_user_query_from_messages(messages)

    # Try prompt format (completions)
    if "prompt" in body:
        prompt = body.get("prompt")
        if prompt:
            return str(prompt)

    return None


async def parse_request_body_safely(request) -> Optional[Dict[str, Any]]:
    """
    Parse request body with consistent error handling.

    This is a foundation utility that can be gradually adopted across
    the 35+ endpoints that currently parse request bodies independently.

    Args:
        request: FastAPI Request object

    Returns:
        Parsed request body as dict, or None if parsing fails

    Note:
        This function does NOT raise HTTPException to allow middleware
        to handle parsing failures gracefully. Callers should check the
        return value and handle None appropriately.

    Example:
        >>> body = await parse_request_body_safely(request)
        >>> if body is None:
        ...     return {"error": "Invalid JSON"}
        >>> process_request(body)
    """
    try:
        return await request.json()
    except Exception as e:
        logger.debug(f"Failed to parse request body: {e}")
        return None
