#!/usr/bin/env python3
"""
Qwen3-Audio Handler for Speech-to-Text (STT)

Supports audio transcription and translation using Qwen3-Audio models:
- Qwen2-Audio-7B-Instruct
- Qwen/Qwen2-Audio-7B

Based on:
- https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- OpenAI Whisper API reference: https://platform.openai.com/docs/api-reference/audio/createTranscription
"""

import logging
import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Union, TYPE_CHECKING
from dataclasses import dataclass

import httpx
from fastapi import HTTPException, UploadFile
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS (OpenAI compatible)
# ============================================================================

AudioResponseFormat = Literal["text", "json", "verbose_json", "srt"]


class TranscriptionRequest(BaseModel):
    """Audio transcription request (OpenAI compatible format)."""

    file: str = Field(..., description="Audio file to transcribe (base64 or path)")
    model: str = Field(..., description="Model to use for transcription")
    language: Optional[str] = Field(None, description="Language code (e.g., 'en', 'zh')")
    prompt: Optional[str] = Field(None, description="Optional text to guide transcription")
    response_format: AudioResponseFormat = Field(default="json", description="Response format")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")
    timestamp_granularities: Optional[list[Literal["word"]]] = Field(
        None, description="Timestamp granularity"
    )


class TranslationRequest(BaseModel):
    """Audio translation request (transcribes and translates to English)."""

    file: str = Field(..., description="Audio file to translate")
    model: str = Field(..., description="Model to use for translation")
    prompt: Optional[str] = Field(None, description="Optional text to guide translation")
    response_format: AudioResponseFormat = Field(default="json", description="Response format")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Sampling temperature")


class TranscriptionResponse(BaseModel):
    """Transcription response (OpenAI compatible format)."""

    text: str = Field(description="Transcribed text")
    task: str = Field(default="transcribe", description="Task performed")
    language: str = Field(description="Detected language code")
    duration: float = Field(description="Audio duration in seconds")
    words: Optional[list] = Field(default=None, description="Timestamps per word")
    segments: Optional[list] = Field(default=None, description="Transcription segments")


# ============================================================================
# QWEN3-AUDIO MODEL CONFIGURATION
# ============================================================================

QWEN3_AUDIO_MODELS = {
    "qwen2-audio-7b-instruct": {
        "model_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "sample_rate": 16000,
        "max_duration": 600,  # 10 minutes max
        "languages": ["en", "zh", "es", "fr", "de", "ja", "ko"],
        "description": "Qwen2-Audio 7B instruction-tuned model for speech understanding",
        "quality": "high",
        "supports_translation": True,
        "supports_timestamps": True,
    },
    "qwen2-audio-7b": {
        "model_id": "Qwen/Qwen2-Audio-7B",
        "sample_rate": 16000,
        "max_duration": 600,
        "languages": ["en", "zh", "es", "fr", "de", "ja", "ko"],
        "description": "Qwen2-Audio 7B base model for speech understanding",
        "quality": "high",
        "supports_translation": True,
        "supports_timestamps": False,
    },
    "qwen-audio": {
        "model_id": "Qwen/Qwen-Audio",
        "sample_rate": 16000,
        "max_duration": 300,  # 5 minutes
        "languages": ["en", "zh"],
        "description": "Qwen-Audio for speech understanding (lighter)",
        "quality": "standard",
        "supports_translation": True,
        "supports_timestamps": False,
    },
}

# Pollinations TTS backend URL (for speech synthesis if needed)
POLLINATIONS_STT_URL = "https://image.pollinations.ai"

# Audio format support
SUPPORTED_AUDIO_FORMATS = [
    "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm", "ogg", "flac", "aac", "opus"
]


# =============================================================================
# AUDIO HANDLER
# =============================================================================

@dataclass
class Qwen3AudioHandler:
    """Handler for Qwen3-Audio models for speech-to-text."""

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
        # Clear loaded models to free memory
        self._loaded_models.clear()

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for an audio model."""
        model_key = model_name.lower().replace("qwen/qwen2-", "qwen2-")
        model_key = model_key.replace("qwen/qwen-", "qwen-")

        if model_key in QWEN3_AUDIO_MODELS:
            return QWEN3_AUDIO_MODELS[model_key]

        for key, config in QWEN3_AUDIO_MODELS.items():
            if model_key in key or key in model_key:
                return config

        return None

    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str,
        model: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        temperature: float = 0.0,
        response_format: str = "json",
        timestamp_granularities: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Qwen3-Audio models.

        Returns:
            Transcription response with text, language, duration, and optionally timestamps
        """
        model_config = self.get_model_config(model)
        if not model_config:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio model: {model}. Available: {list(QWEN3_AUDIO_MODELS.keys())}"
            )

        model_id = model_config["model_id"]

        try:
            # Try local model first if enabled
            if self.enable_local_models:
                return await self._transcribe_local(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    model_id=model_id,
                    model_config=model_config,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format,
                    timestamp_granularities=timestamp_granularities
                )
            else:
                # Use cloud backend via llama.cpp or API
                return await self._transcribe_via_backend(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    model=model,
                    model_config=model_config,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio transcription failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(e)}"
            )

    async def _transcribe_local(
        self,
        audio_data: bytes,
        audio_format: str,
        model_id: str,
        model_config: Dict[str, Any],
        language: Optional[str],
        prompt: Optional[str],
        temperature: float,
        response_format: str,
        timestamp_granularities: Optional[list]
    ) -> Dict[str, Any]:
        """Transcribe using local Qwen3-Audio model."""
        import torch
        import numpy as np
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        import torchaudio

        logger.info(f"Loading Qwen3-Audio model: {model_id}")

        loop = asyncio.get_event_loop()

        def load_model():
            if model_id not in self._loaded_models:
                model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                processor = AutoProcessor.from_pretrained(model_id)
                self._loaded_models[model_id] = (model, processor)
            return self._loaded_models[model_id]

        model, processor = await loop.run_in_executor(None, load_model)

        # Prepare audio input
        def prepare_audio():
            # Save audio data to temp file for torchaudio to load
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_path = temp_file.name

            try:
                # Load audio with torchaudio
                waveform, sample_rate = torchaudio.load(temp_path)

                # Resample if needed (Qwen2-Audio expects 16kHz)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000

                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                return waveform, sample_rate
            finally:
                Path(temp_path).unlink(missing_ok=True)

        waveform, sample_rate = await loop.run_in_executor(None, prepare_audio)

        # Calculate duration
        duration = waveform.shape[1] / sample_rate

        # Prepare inputs for the model
        def prepare_inputs():
            # Qwen2-Audio expects audio in a specific format
            inputs = processor(
                audio=waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            )
            return inputs

        inputs = await loop.run_in_executor(None, prepare_inputs)

        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate transcription
        def generate():
            # Build prompt
            instruction = "Transcribe the following audio."
            if prompt:
                instruction += f" Context: {prompt}"
            if language:
                instruction += f" The audio is in {language}."

            # Generate transcription
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 0.7,
                )

            return output_ids

        output_ids = await loop.run_in_executor(None, generate)

        # Decode output
        def decode_output():
            transcription = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            return transcription

        transcription_text = await loop.run_in_executor(None, decode_output)

        # Detect language (simple heuristic or use model's detection)
        detected_language = language or self._detect_language(transcription_text)

        # Build response
        response = {
            "text": transcription_text,
            "task": "transcribe",
            "language": detected_language,
            "duration": duration
        }

        # Add word-level timestamps if requested and supported
        if timestamp_granularities and "word" in timestamp_granularities and model_config.get("supports_timestamps"):
            response["words"] = self._generate_word_timestamps(transcription_text, duration)

        return response

    async def _transcribe_via_backend(
        self,
        audio_data: bytes,
        audio_format: str,
        model: str,
        model_config: Dict[str, Any],
        language: Optional[str],
        prompt: Optional[str],
        temperature: float,
        response_format: str
    ) -> Dict[str, Any]:
        """Transcribe using cloud backend (llama.cpp or compatible API)."""
        # For now, we'll use a placeholder that indicates the feature needs local models
        # In production, this could call llama.cpp's audio endpoint or another STT API
        raise HTTPException(
            status_code=501,
            detail="Cloud STT not yet configured. Please enable local models with enable_local_models=True "
                  "or use OpenAI's Whisper API as an alternative."
        )

    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        # Check for Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return "zh"

        # Check for Japanese characters
        japanese_chars = sum(1 for c in text if '\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff')
        if japanese_chars > len(text) * 0.2:
            return "ja"

        # Check for Korean characters
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7af')
        if korean_chars > len(text) * 0.2:
            return "ko"

        # Default to English for Latin scripts
        return "en"

    def _generate_word_timestamps(self, text: str, duration: float) -> list:
        """Generate estimated word timestamps."""
        import re
        words = re.findall(r"\b[\w']+\b", text)
        if not words:
            return []

        avg_time_per_word = duration / len(words)
        timestamps = []

        for i, word in enumerate(words):
            timestamps.append({
                "word": word,
                "start": round(i * avg_time_per_word, 3),
                "end": round((i + 1) * avg_time_per_word, 3)
            })

        return timestamps


# =============================================================================
# AUDIO SERVICE (singleton)
# =============================================================================

_audio_handler: Optional[Qwen3AudioHandler] = None


def get_audio_handler(backend_url: str) -> Qwen3AudioHandler:
    """Get or create the audio handler singleton."""
    global _audio_handler
    if _audio_handler is None or _audio_handler.backend_url != backend_url:
        _audio_handler = Qwen3AudioHandler(backend_url=backend_url)
    return _audio_handler


async def close_audio_handler():
    """Close the audio handler connection."""
    global _audio_handler
    if _audio_handler:
        await _audio_handler.close()
        _audio_handler = None


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

async def read_audio_file(file: UploadFile, max_size: int = 25 * 1024 * 1024) -> tuple[bytes, str]:
    """
    Read audio file from UploadFile.

    Returns:
        (audio_data, format)
    """
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large. Maximum size: {max_size // (1024*1024)}MB"
        )

    # Detect format from filename or content type
    filename = file.filename or ""
    content_type = file.content_type or ""

    audio_format = "wav"  # Default
    for fmt in SUPPORTED_AUDIO_FORMATS:
        if filename.endswith(f".{fmt}") or fmt in content_type:
            audio_format = fmt
            break

    return content, audio_format
