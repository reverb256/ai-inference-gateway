"""
Vision support for multimodal models.

Handles detection and processing of image content in messages for
Qwen3.5 vision-capable models.

Vision-capable models (those with mmproj files):
- qwen3.5-35b-a3b (mmproj-F32.gguf)
- qwen3.5-27b (mmproj-F32.gguf)
- qwen3.5-9b (mmproj-F32.gguf)
- qwen3.5-4b (mmproj-F32.gguf)
- crow-9b-opus-4.6-distill-heretic_qwen3.5 (mmproj-f16.gguf)
"""

import logging
import base64
import re
from typing import Dict, List, Optional, Any, Union

logger = logging.getLogger(__name__)


# Models with vision support (have mmproj files)
VISION_CAPABLE_MODELS = {
    "qwen3.5-35b-a3b",
    "qwen3.5-27b",
    "qwen3.5-9b",
    "qwen3.5-4b",
    "crow-9b-opus-4.6-distill-heretic_qwen3.5",
}


def detect_vision_content(messages: List[Dict[str, Any]]) -> bool:
    """
    Detect if messages contain image content.

    Args:
        messages: List of messages with content field

    Returns:
        True if any message contains image_url content
    """
    for message in messages:
        content = message.get("content")

        # Handle list content (OpenAI multimodal format)
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "image_url":
                    return True

        # Handle string content (check for embedded base64)
        elif isinstance(content, str):
            # Look for data:image/ in content
            if "data:image/" in content:
                return True

    return False


def extract_images_from_message(
    content: Union[str, List[Dict]],
) -> List[Dict[str, str]]:
    """
    Extract image URLs from message content.

    Args:
        content: Message content (string or list of blocks)

    Returns:
        List of image URLs in data URI format
    """
    images = []

    if isinstance(content, list):
        for block in content:
            if block.get("type") == "image_url":
                url = block.get("image_url", {})
                if isinstance(url, dict):
                    url = url.get("url", "")
                images.append(url)
            elif block.get("type") == "text":
                # Text blocks don't contain images
                continue

    elif isinstance(content, str):
        # Extract base64 images from text
        # Pattern: data:image/[format];base64,[data]
        pattern = r"data:image/([^;]+);base64,([A-Za-z0-9+/=]+)"
        matches = re.findall(pattern, content)
        for img_format, b64_data in matches:
            images.append(f"data:image/{img_format};base64,{b64_data}")

    return images


def convert_to_backend_vision_format(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI multimodal messages to backend format.

    Backend expects:
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "..."},
        {"type": "image", "image_url": {"url": "data:image/...;base64,..."}}
      ]
    }

    Args:
        messages: OpenAI format messages

    Returns:
        Backend format messages with preserved images
    """
    backend_messages = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content")

        if isinstance(content, str):
            # Simple text message
            backend_messages.append({"role": role, "content": content})

        elif isinstance(content, list):
            # Multimodal message - preserve structure
            multimodal_content = []

            for block in content:
                block_type = block.get("type")

                if block_type == "text":
                    multimodal_content.append(
                        {"type": "text", "text": block.get("text", "")}
                    )

                elif block_type == "image_url":
                    # Extract URL (might be nested dict)
                    url = block.get("image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")

                    multimodal_content.append(
                        {"type": "image", "image_url": {"url": url}}
                    )

            backend_messages.append({"role": role, "content": multimodal_content})

    return backend_messages


def is_vision_capable_model(model_id: str) -> bool:
    """
    Check if a model supports vision.

    Args:
        model_id: Model identifier

    Returns:
        True if model has vision capabilities (mmproj file)
    """
    model_lower = model_id.lower()

    for vision_model in VISION_CAPABLE_MODELS:
        if vision_model in model_lower:
            return True

    return False


def recommend_vision_model(
    available_models: List[str], quality_priority: bool = False
) -> Optional[str]:
    """
    Recommend the best vision model from available options.

    Args:
        available_models: List of available model IDs
        quality_priority: If True, prefer larger models (35B-A3B)
                        If False, prefer faster models (4B)

    Returns:
        Recommended model ID or None
    """
    # Filter to vision-capable models
    vision_models = [m for m in available_models if is_vision_capable_model(m)]

    if not vision_models:
        return None

    if quality_priority:
        # Prefer larger models
        priority_order = [
            "35b-a3b",  # Best quality, 256K context
            "27b",  # High quality
            "9b",  # Balanced
            "4b",  # Faster
        ]

        for priority in priority_order:
            for model in vision_models:
                if priority in model.lower():
                    return model
    else:
        # Prefer faster models
        priority_order = [
            "4b",  # Fastest
            "9b",  # Balanced
            "27b",  # High quality
            "35b-a3b",  # Best quality
        ]

        for priority in priority_order:
            for model in vision_models:
                if priority in model.lower():
                    return model

    # Fallback to first available
    return vision_models[0]


def validate_image_url(url: str) -> bool:
    """
    Validate an image URL/data URI.

    Args:
        url: Image URL or data URI

    Returns:
        True if valid format
    """
    if not url:
        return False

    # Check for data URI format
    if url.startswith("data:image/"):
        try:
            # Basic validation: should have format and base64 data
            if ";base64," not in url:
                return False

            # Extract and validate base64 portion
            _, b64_data = url.split(";base64,", 1)
            if not b64_data:
                return False

            # Try to decode to ensure valid base64
            base64.b64decode(b64_data, validate=True)
            return True

        except Exception:
            logger.warning("Invalid base64 image data")
            return False

    # Check for HTTP/HTTPS URL
    elif url.startswith("http://") or url.startswith("https://"):
        return True

    return False


def get_vision_model_recommendation(task_description: str = "") -> str:
    """
    Get vision model recommendation based on task.

    Args:
        task_description: Optional task description

    Returns:
        Recommended model ID
    """
    task_lower = task_description.lower()

    # Complex analysis or high quality required
    if any(
        keyword in task_lower
        for keyword in ["detailed", "analyze", "complex", "professional"]
    ):
        return "qwen3.5-35b-a3b"

    # General vision tasks
    if any(
        keyword in task_lower for keyword in ["describe", "what", "identify", "ocr"]
    ):
        return "qwen3.5-9b"

    # Fast responses needed
    if any(keyword in task_lower for keyword in ["fast", "quick", "simple"]):
        return "qwen3.5-4b"

    # Default: balanced option
    return "qwen3.5-9b"


# Export for use in other modules
__all__ = [
    "detect_vision_content",
    "extract_images_from_message",
    "convert_to_backend_vision_format",
    "is_vision_capable_model",
    "recommend_vision_model",
    "validate_image_url",
    "get_vision_model_recommendation",
    "VISION_CAPABLE_MODELS",
]
