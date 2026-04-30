import logging
from typing import Optional, List


logger = logging.getLogger(__name__)


def normalize_roles(messages: List[dict]) -> List[dict]:
    """
    Normalize message roles for backend compatibility.

    Converts 'developer' role to 'system' role for OMP compatibility.
    The 'developer' role is used by some MCP clients but not supported by llama.cpp.
    """
    normalized = []
    for msg in messages:
        normalized_msg = msg.copy()
        if msg.get("role") == "developer":
            normalized_msg["role"] = "system"
            logger.debug("Converted 'developer' role to 'system' for backend compatibility")
        normalized.append(normalized_msg)
    return normalized


class RequestValidationMiddleware:
    """Validates incoming API request bodies before forwarding to providers.

    Message count and content length limits are NOT enforced here — the model
    is the authority on what it can handle. If it can't, the gateway's circuit
    breaker and fallback chain handle it (try next, larger model).
    """

    MAX_REQUEST_SIZE = 10 * 1024 * 1024;  # 10MB — actual abuse prevention
    MAX_TOOLS = 250;
    ALLOWED_ROLES = {"system", "user", "assistant", "tool", "function", "developer"};

    async def validate_chat_request(self, body: dict) -> Optional[List[str]]:
        """Validate /v1/chat/completions request body."""
        errors: List[str] = [];

        if not isinstance(body.get("model"), str) or not body["model"]:
            errors.append("field 'model' is required and must be a non-empty string");

        messages = body.get("messages");
        if not isinstance(messages, list):
            errors.append("field 'messages' is required and must be an array");
        else:
            for idx, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"message at index {idx} must be an object");
                    continue;
                if not isinstance(msg.get("role"), str):
                    errors.append(f"message at index {idx} missing or invalid 'role'");
                elif msg["role"] not in self.ALLOWED_ROLES:
                    errors.append(
                        f"message at index {idx} has invalid role '{msg['role']}'"
                    );

        tools = body.get("tools");
        if tools is not None:
            if not isinstance(tools, list):
                errors.append("field 'tools' must be an array if provided");
            elif len(tools) > self.MAX_TOOLS:
                errors.append(
                    f"too many tools: {len(tools)} exceeds limit of {self.MAX_TOOLS}"
                );

        temperature = body.get("temperature");
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                errors.append("field 'temperature' must be a number if provided");
            elif not (0.0 <= temperature <= 2.0):
                errors.append("field 'temperature' must be between 0.0 and 2.0");

        top_p = body.get("top_p");
        if top_p is not None:
            if not isinstance(top_p, (int, float)):
                errors.append("field 'top_p' must be a number if provided");
            elif not (0.0 <= top_p <= 1.0):
                errors.append("field 'top_p' must be between 0.0 and 1.0");

        if errors:
            logger.warning("chat request validation failed: %s", errors);
            return errors;
        return None;

    async def validate_embedding_request(self, body: dict) -> Optional[List[str]]:
        """Validate /v1/embeddings request body."""
        errors: List[str] = [];

        inp = body.get("input");
        if inp is None:
            errors.append("field 'input' is required");
        elif isinstance(inp, str):
            if not inp:
                errors.append("field 'input' must not be empty");
            elif len(inp) > 8192:
                errors.append(
                    f"input text length {len(inp)} exceeds limit of 8192"
                );
        elif isinstance(inp, list):
            if len(inp) == 0:
                errors.append("field 'input' list must not be empty");
            elif len(inp) > 2048:
                errors.append(
                    f"input list length {len(inp)} exceeds limit of 2048"
                );
            else:
                for idx, item in enumerate(inp):
                    if isinstance(item, str):
                        if len(item) > 8192:
                            errors.append(
                                f"input[{idx}] text length {len(item)} exceeds limit of 8192"
                            );
                    elif isinstance(item, (int, float)):
                        pass;  # token IDs are fine
                    else:
                        errors.append(
                            f"input[{idx}] must be a string or number"
                        );
        else:
            errors.append("field 'input' must be a string or array");

        if errors:
            logger.warning("embedding request validation failed: %s", errors);
            return errors;
        return None;

    async def validate_search_request(self, body: dict) -> Optional[List[str]]:
        """Validate search request body."""
        errors: List[str] = [];

        query = body.get("query");
        if not isinstance(query, str) or not query:
            errors.append("field 'query' is required and must be a non-empty string");
        elif len(query) > 1000:
            errors.append(
                f"query length {len(query)} exceeds limit of 1000 characters"
            );

        max_results = body.get("max_results");
        if max_results is not None:
            if not isinstance(max_results, int) or isinstance(max_results, bool):
                errors.append("field 'max_results' must be a positive integer");
            elif max_results < 1:
                errors.append("field 'max_results' must be a positive integer");

        category = body.get("category");
        if category is not None:
            if not isinstance(category, str) or not category:
                errors.append("field 'category' must be a non-empty string if provided");

        if errors:
            logger.warning("search request validation failed: %s", errors);
            return errors;
        return None;

    async def validate_anthropic_request(self, body: dict) -> Optional[List[str]]:
        """Validate /v1/messages (Anthropic) request body."""
        errors: List[str] = [];

        if not isinstance(body.get("model"), str) or not body["model"]:
            errors.append("field 'model' is required and must be a non-empty string");

        messages = body.get("messages");
        if not isinstance(messages, list):
            errors.append("field 'messages' is required and must be an array");

        max_tokens = body.get("max_tokens");
        if max_tokens is None:
            errors.append("field 'max_tokens' is required");
        elif not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
            errors.append("field 'max_tokens' must be a positive integer");
        elif max_tokens < 1:
            errors.append("field 'max_tokens' must be a positive integer");

        if errors:
            logger.warning("anthropic request validation failed: %s", errors);
            return errors;
        return None;
