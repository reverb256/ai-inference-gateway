"""
response_format transformation for OpenAI compatibility.

Transforms OpenAI response_format requests into backend-compatible
system prompts that enforce JSON output constraints.

Supported modes:
- json_object: Ensures valid JSON response
- json_schema: Enforces JSON Schema compliance
- text: No transformation (default)
"""

import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ResponseFormatTransformer:
    """
    Transform OpenAI response_format to backend instructions.

    OpenAI clients may send:
    {
        "response_format": {
            "type": "json_object"
        }
    }

    Or with structured outputs:
    {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "schema_name",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }
    }

    The backend doesn't natively support response_format, so we transform
    these into system prompt instructions that guide the model to produce
    compliant JSON responses.
    """

    # Pre-computed system prompts for common modes
    JSON_OBJECT_PROMPT = """You must respond ONLY with valid JSON.

Requirements:
- Respond with valid JSON syntax
- Do NOT use markdown code blocks
- Do NOT include any text outside the JSON
- Do NOT provide explanations
- Your entire response must be parseable as JSON"""

    # PII patterns to redact from schemas (just in case)
    PII_PATTERNS = [r"password", r"secret", r"api[_-]?key", r"token", r"credential"]

    def __init__(self, strict_mode: bool = False):
        """
        Initialize transformer.

        Args:
            strict_mode: If True, add additional validation constraints
        """
        self.strict_mode = strict_mode

    def transform_request(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform request body to add JSON mode instructions.

        Args:
            request_body: Original request body

        Returns:
            Modified request body with system prompt added
        """
        # Clone to avoid mutating original
        body = request_body.copy()
        body["messages"] = [msg.copy() for msg in body.get("messages", [])]

        response_format = body.get("response_format")

        if not response_format:
            # No response_format specified
            return body

        format_type = response_format.get("type")

        if format_type == "json_object":
            self._add_json_object_instructions(body)
            # Remove response_format as backend doesn't use it
            del body["response_format"]

        elif format_type == "json_schema":
            self._add_json_schema_instructions(
                body, response_format.get("json_schema", {})
            )
            del body["response_format"]

        elif format_type == "text":
            # No transformation needed for text mode
            if "response_format" in body:
                del body["response_format"]

        else:
            logger.warning(f"Unknown response_format type: {format_type}")

        return body

    def _add_json_object_instructions(self, body: Dict[str, Any]) -> None:
        """
        Add instructions for basic JSON mode.

        Args:
            body: Request body to modify
        """
        system_prompt = self.JSON_OBJECT_PROMPT

        if self.strict_mode:
            system_prompt += "\n\nAdditional requirements:\n"
            system_prompt += "- Use double quotes for all strings\n"
            system_prompt += "- Ensure proper escaping\n"
            system_prompt += "- Validate JSON before responding\n"
            system_prompt += "- No trailing commas in objects/arrays"

        # Insert as first system message
        body["messages"].insert(0, {"role": "system", "content": system_prompt})

        logger.debug("Added JSON object mode instructions")

    def _add_json_schema_instructions(
        self, body: Dict[str, Any], json_schema: Dict[str, Any]
    ) -> None:
        """
        Add instructions for structured output (JSON Schema mode).

        Args:
            body: Request body to modify
            json_schema: JSON Schema specification
        """
        schema = json_schema.get("schema", {})
        schema_name = json_schema.get("name", "output")
        strict = json_schema.get("strict", False)

        # Build comprehensive instructions
        instructions = f"""You must respond ONLY with valid JSON matching this schema:

Schema Name: {schema_name}
Strict Mode: {'Yes' if strict else 'No'}

```json
{json.dumps(schema, indent=2)}
```

Requirements:
- Respond ONLY with valid JSON
- Your response must match the schema above exactly
- Do NOT use markdown code blocks (```json ... ```)
- Do NOT include any explanations outside the JSON
- All required fields must be present"""

        if strict:
            instructions += "\n- Field types must match exactly\n"
            instructions += "- Required fields cannot be omitted\n"
            instructions += "- No extra fields beyond the schema\n"
            instructions += "- Enums must match exact values"

        # Add format examples if schema has specific patterns
        if "properties" in schema:
            instructions += "\n\nField Details:\n"
            for prop_name, prop_def in schema["properties"].items():
                required = prop_name in schema.get("required", [])
                instructions += f"- {prop_name}: {prop_def.get('type', 'any')}"
                if required:
                    instructions += " (required)"
                instructions += "\n"

        if self.strict_mode:
            instructions += "\n\nCRITICAL: Validate your output before responding."

        body["messages"].insert(0, {"role": "system", "content": instructions})

        logger.debug(f"Added JSON Schema mode instructions for {schema_name}")

    def validate_response(
        self, response_content: str, response_format: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that response matches the requested format.

        Args:
            response_content: Model response content
            response_format: Original response_format request

        Returns:
            (is_valid, error_message)
        """
        format_type = response_format.get("type")

        if format_type == "json_object":
            return self._validate_json_object(response_content)

        elif format_type == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            return self._validate_json_schema(response_content, schema)

        # No validation for text mode
        return True, None

    def _validate_json_object(self, content: str) -> tuple[bool, Optional[str]]:
        """Validate response is valid JSON."""
        try:
            json.loads(content)
            return True, None
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"

    def _validate_json_schema(
        self, content: str, schema: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate response matches JSON Schema."""
        try:
            parsed = json.loads(content)

            # Basic type check
            schema_type = schema.get("type")
            if schema_type == "object" and not isinstance(parsed, dict):
                return False, f"Expected object, got {type(parsed).__name__}"

            elif schema_type == "array" and not isinstance(parsed, list):
                return False, f"Expected array, got {type(parsed).__name__}"

            # Required fields check
            required = schema.get("required", [])
            missing_fields = [field for field in required if field not in parsed]

            if missing_fields:
                return False, f"Missing required fields: {missing_fields}"

            # Properties type check
            properties = schema.get("properties", {})
            for prop_name, prop_value in parsed.items():
                if prop_name not in properties:
                    # Extra field (warn if strict)
                    logger.warning(f"Extra field in response: {prop_name}")
                    continue

                prop_def = properties[prop_name]
                prop_type = prop_def.get("type")

                if prop_type == "string" and not isinstance(prop_value, str):
                    return (
                        False,
                        f"Field {prop_name}: expected string, got {type(prop_value).__name__}",
                    )

                elif prop_type == "number" and not isinstance(prop_value, (int, float)):
                    return (
                        False,
                        f"Field {prop_name}: expected number, got {type(prop_value).__name__}",
                    )

                elif prop_type == "boolean" and not isinstance(prop_value, bool):
                    return (
                        False,
                        f"Field {prop_name}: expected boolean, got {type(prop_value).__name__}",
                    )

                elif prop_type == "array" and not isinstance(prop_value, list):
                    return (
                        False,
                        f"Field {prop_name}: expected array, got {type(prop_value).__name__}",
                    )

                elif prop_type == "object" and not isinstance(prop_value, dict):
                    return (
                        False,
                        f"Field {prop_name}: expected object, got {type(prop_value).__name__}",
                    )

            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {str(e)}"


# Singleton instance
_transformer = None


def get_transformer(strict_mode: bool = False) -> ResponseFormatTransformer:
    """Get or create the singleton transformer instance."""
    global _transformer
    if _transformer is None or _transformer.strict_mode != strict_mode:
        _transformer = ResponseFormatTransformer(strict_mode=strict_mode)
    return _transformer


async def transform_request(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform request to add JSON mode instructions.

    Convenience function that uses the singleton transformer.

    Args:
        body: Request body

    Returns:
        Transformed request body
    """
    transformer = get_transformer()
    return transformer.transform_request(body)


async def validate_response(
    content: str, response_format: Dict[str, Any]
) -> tuple[bool, Optional[str]]:
    """
    Validate response matches requested format.

    Convenience function that uses the singleton transformer.

    Args:
        content: Response content
        response_format: Requested format

    Returns:
        (is_valid, error_message)
    """
    transformer = get_transformer()
    return transformer.validate_response(content, response_format)
