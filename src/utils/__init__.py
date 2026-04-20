from .redis_client import RedisClient

__all__ = ["RedisClient"]

# Message utilities for content extraction
try:
    from .message_utils import (
        extract_last_user_message,
        extract_message_content,
        extract_user_query_from_messages,
        extract_user_query_from_request_body,
        parse_request_body_safely,
    )

    __all__.extend([
        "extract_last_user_message",
        "extract_message_content",
        "extract_user_query_from_messages",
        "extract_user_query_from_request_body",
        "parse_request_body_safely",
    ])
except ImportError:
    pass

# Tool utilities for agentic workflows
try:
    from .tool_utils import (
        has_tool_calls,
        has_tool_calls_openai,
        has_tool_calls_anthropic,
        extract_tool_calls_openai,
        extract_tool_calls_anthropic,
        create_tool_result_openai,
        create_tool_result_anthropic,
        is_tool_response_format,
        ToolUtils,
    )

    __all__.extend([
        "has_tool_calls",
        "has_tool_calls_openai",
        "has_tool_calls_anthropic",
        "extract_tool_calls_openai",
        "extract_tool_calls_anthropic",
        "create_tool_result_openai",
        "create_tool_result_anthropic",
        "is_tool_response_format",
        "ToolUtils",
    ])
except ImportError:
    pass

# Optional imports
try:
    from .metrics import MetricsHelper  # noqa: F401

    __all__.append("MetricsHelper")
except ImportError:
    pass

def strip_markdown_json_fences(text: str) -> str:
    """Strip markdown code fences from LLM responses that should be raw JSON.
    
    Many models (Qwen, GLM, Gemma) wrap JSON in ```json ... ``` fences.
    This strips those fences so tools like Vane/Perplexica can parse the response.
    """
    if not text or not isinstance(text, str):
        return text
    
    import re
    stripped = text.strip()
    
    # Match ```json ... ``` or ``` ... ``` wrapping
    pattern = r'^```(?:json|JSON)?\s*\n?(.*?)\n?\s*```$'
    match = re.match(pattern, stripped, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    return text
