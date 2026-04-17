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
