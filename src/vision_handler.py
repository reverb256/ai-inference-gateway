#!/usr/bin/env python3
"""
Qwen3-Vision Handler for Image Understanding

Supports visual question answering using Qwen3-VL models:
- Qwen/Qwen2-VL-7B-Instruct
- Qwen/Qwen2-VL-2B-Instruct
- Qwen/Qwen2-VL-7B

Based on:
- https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
- OpenAI Vision API reference: https://platform.openai.com/docs/guides/vision
"""

import logging
import asyncio
import io
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from dataclasses import dataclass

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS (OpenAI compatible)
# ============================================================================

class ImageUrl(BaseModel):
    """Image URL or base64 data."""
    url: str
    detail: Optional[str] = Field("auto", description="Detail level: auto, low, high")


class ImageContent(BaseModel):
    """Image content for vision requests."""
    type: str = Field("image_url")
    image_url: ImageUrl


class VisionMessage(BaseModel):
    """Message in a vision conversation."""
    role: str
    content: Union[str, List[Union[str, ImageContent]]]


class VisionRequest(BaseModel):
    """Vision API request (OpenAI compatible format)."""

    model: str = Field(..., description="Vision model to use")
    messages: List[VisionMessage] = Field(..., description="Conversation messages with images")
    max_tokens: int = Field(default=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class VisionResponse(BaseModel):
    """Vision API response."""
    id: str = Field(description="Response identifier")
    object: str = Field(default="chat.completion", description="Object type")
    created: int = Field(description="Unix timestamp")
    model: str = Field(description="Model used")
    choices: List[Dict] = Field(description="Generated choices")
    usage: Dict = Field(description="Token usage")


# ============================================================================
# QWEN3-VISION MODEL CONFIGURATION
# ============================================================================

QWEN3_VISION_MODELS = {
    "qwen2-vl-7b-instruct": {
        "model_id": "Qwen/Qwen2-VL-7B-Instruct",
        "max_tokens": 32768,
        "image_size": 512,
        "description": "Qwen2-VL 7B instruction-tuned vision-language model",
        "quality": "high",
        "supports_video": False,
        "max_images": 16,
    },
    "qwen2-vl-2b-instruct": {
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "max_tokens": 32768,
        "image_size": 512,
        "description": "Qwen2-VL 2B instruction-tuned (lighter, faster)",
        "quality": "standard",
        "supports_video": False,
        "max_images": 16,
    },
    "qwen2-vl-7b": {
        "model_id": "Qwen/Qwen2-VL-7B",
        "max_tokens": 32768,
        "image_size": 512,
        "description": "Qwen2-VL 7B base vision-language model",
        "quality": "high",
        "supports_video": False,
        "max_images": 16,
    },
    "qwen2-vl-72b-instruct": {
        "model_id": "Qwen/Qwen2-VL-72B-Instruct",
        "max_tokens": 32768,
        "image_size": 512,
        "description": "Qwen2-VL 72B instruction-tuned (highest quality)",
        "quality": "very_high",
        "supports_video": False,
        "max_images": 16,
    },
}

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["png", "jpeg", "jpg", "gif", "webp", "bmp"]
MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB


# =============================================================================
# VISION HANDLER
# =============================================================================

@dataclass
class Qwen3VisionHandler:
    """Handler for Qwen3-VL models for image understanding."""

    backend_url: str
    timeout: int = 120
    enable_local_models: bool = False

    def __post_init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._loaded_models: Dict[str, Any] = {}

    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialized async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                follow_redirects=True
            )
        return self._client

    async def close(self):
        """Close the HTTP client and unload models."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._loaded_models.clear()

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a vision model."""
        model_key = model_name.lower().replace("qwen/qwen2-", "qwen2-")
        model_key = model_key.replace("qwen/qwen-", "qwen-")

        if model_key in QWEN3_VISION_MODELS:
            return QWEN3_VISION_MODELS[model_key]

        for key, config in QWEN3_VISION_MODELS.items():
            if model_key in key or key in model_key:
                return config

        return None

    async def analyze_image(
        self,
        image_data: bytes,
        image_format: str,
        prompt: str,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Analyze an image using Qwen3-VL models.

        Returns:
            Analysis response with generated text
        """
        model_config = self.get_model_config(model)
        if not model_config:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported vision model: {model}. Available: {list(QWEN3_VISION_MODELS.keys())}"
            )

        model_id = model_config["model_id"]

        try:
            if self.enable_local_models:
                return await self._analyze_local(
                    image_data=image_data,
                    image_format=image_format,
                    prompt=prompt,
                    model_id=model_id,
                    model_config=model_config,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            else:
                return await self._analyze_via_backend(
                    image_data=image_data,
                    image_format=image_format,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Vision analysis failed: {str(e)}"
            )

    async def _analyze_local(
        self,
        image_data: bytes,
        image_format: str,
        prompt: str,
        model_id: str,
        model_config: Dict[str, Any],
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Analyze image using local Qwen3-VL model."""
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from PIL import Image
        import io

        logger.info(f"Loading Qwen3-VL model: {model_id}")

        loop = asyncio.get_event_loop()

        def load_model():
            if model_id not in self._loaded_models:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_id)
                self._loaded_models[model_id] = (model, processor)
            return self._loaded_models[model_id]

        model, processor = await loop.run_in_executor(None, load_model)

        # Load image
        def load_image():
            image = Image.open(io.BytesIO(image_data))
            # Convert to RGB if necessary
            if image.mode != "RGB":
                image = image.convert("RGB")
            return image

        image = await loop.run_in_executor(None, load_image)

        # Prepare inputs
        def prepare_inputs():
            # Qwen2-VL expects messages format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            )
            return inputs

        inputs = await loop.run_in_executor(None, prepare_inputs)

        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate response
        def generate():
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 0.7,
                    top_p=0.9,
                )
            return output_ids

        output_ids = await loop.run_in_executor(None, generate)

        # Decode output
        def decode_output():
            generated_ids = [
                output_ids[len(inp_ids):]
                for inp_ids, output_ids in zip(inputs.input_ids, output_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            return response_text

        response_text = await loop.run_in_executor(None, decode_output)

        # Build response
        return {
            "id": f"chatcmpl-{model_id[:10]}",
            "object": "chat.completion",
            "created": int(asyncio.get_event_loop().time()),
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(output_ids[0]) - inputs["input_ids"].shape[1],
                "total_tokens": len(output_ids[0])
            }
        }

    async def _analyze_via_backend(
        self,
        image_data: bytes,
        image_format: str,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Analyze image via cloud backend."""
        # For now, indicate local models are preferred
        raise HTTPException(
            status_code=501,
            detail="Cloud vision API not yet configured. Please enable local models with enable_local_models=True."
        )


# =============================================================================
# VISION SERVICE (singleton)
# =============================================================================

_vision_handler: Optional[Qwen3VisionHandler] = None


def get_vision_handler(backend_url: str) -> Qwen3VisionHandler:
    """Get or create the vision handler singleton."""
    global _vision_handler
    if _vision_handler is None or _vision_handler.backend_url != backend_url:
        _vision_handler = Qwen3VisionHandler(backend_url=backend_url)
    return _vision_handler


async def close_vision_handler():
    """Close the vision handler connection."""
    global _vision_handler
    if _vision_handler:
        await _vision_handler.close()
        _vision_handler = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def read_image_from_url(url: str) -> tuple[bytes, str]:
    """
    Read image from URL (http/https or base64 data URL).

    Returns:
        (image_data, format)
    """
    if url.startswith("data:image/"):
        # Base64 encoded image
        header, data = url.split(",", 1)
        format_part = header.split("/")[1].split(";")[0]
        image_data = base64.b64decode(data)
        return image_data, format_part

    else:
        # HTTP/HTTPS URL
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url)
            response.raise_for_status()
            image_data = response.content

            # Detect format from content type or URL
            content_type = response.headers.get("content-type", "")
            if "image/" in content_type:
                image_format = content_type.split("/")[1]
            else:
                # Try to extract from URL
                for fmt in SUPPORTED_IMAGE_FORMATS:
                    if url.endswith(f".{fmt}"):
                        image_format = fmt
                        break
                else:
                    image_format = "png"

            return image_data, image_format


def encode_image_to_base64(image_data: bytes, image_format: str) -> str:
    """Encode image data to base64 data URL."""
    b64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:image/{image_format};base64,{b64_data}"
