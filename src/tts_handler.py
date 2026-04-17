#!/usr/bin/env python3
"""
Text-to-Speech Handler

Supports multiple TTS backends:
- Pollinations.ai (free, cloud-based, multiple voices)
- Qwen3-TTS models (local, via official qwen-tts package)

Based on:
- Pollinations TTS API: https://text.pollinations.ai
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
"""

import logging
import asyncio
import io
from pathlib import Path
from typing import Optional, Dict, Any, Literal, Union, TYPE_CHECKING
from dataclasses import dataclass
from urllib.parse import quote

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, Field

# Type checking imports - these are only imported when actually used
if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS (OpenAI compatible)
# ============================================================================

TTSAudioFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm16"]


class TTSRequest(BaseModel):
    """Text-to-speech request (OpenAI compatible format)."""

    model: str = Field(
        description="TTS model to use",
        pattern="^(tts-1|tts-1-hd|pollinations-tts|qwen3-tts|Qwen3-TTS|Qwen/Qwen3-TTS).*"
    )
    input: str = Field(..., description="Text to synthesize")
    voice: str = Field(
        default="alloy",
        description="Voice to use (for compatibility, mapped to internal voices)"
    )
    response_format: TTSAudioFormat = Field(
        default="mp3",
        description="Audio output format"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of audio playback"
    )


class TTSResponse(BaseModel):
    """TTS response (OpenAI compatible format)."""

    id: str = Field(description="Unique identifier for the speech generation")
    object: Literal["speech"] = "speech"
    created: int = Field(description="Unix timestamp of creation")
    model: str = Field(description="Model used")
    content: bytes = Field(description="Audio data (for binary response)")
    # For JSON response with base64
    b64_json: Optional[str] = Field(default=None, description="Base64 encoded audio")


# ============================================================================
# TTS MODEL CONFIGURATION
# ============================================================================

QWEN3_TTS_MODELS = {
    # Pollinations.ai TTS (cloud, free, OpenAI-compatible voices)
    "tts-1": {
        "model_id": "pollinations-tts-1",
        "sample_rate": 24000,
        "max_tokens": 4096,
        "description": "Pollinations TTS - OpenAI compatible, free cloud service",
        "quality": "standard",
        "language": "en",
        "backend": "pollinations",
    },
    "tts-1-hd": {
        "model_id": "pollinations-tts-1-hd",
        "sample_rate": 24000,
        "max_tokens": 4096,
        "description": "Pollinations TTS HD - Higher quality",
        "quality": "high",
        "language": "en",
        "backend": "pollinations",
    },
    "pollinations-tts": {
        "model_id": "pollinations-tts",
        "sample_rate": 24000,
        "max_tokens": 4096,
        "description": "Pollinations.ai free TTS service",
        "quality": "standard",
        "language": "en",
        "backend": "pollinations",
    },
    # Qwen3-TTS CustomVoice models (9 premium speakers, instruction control)
    "qwen3-tts-customvoice-0.6b": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "sample_rate": 24000,
        "max_tokens": 2048,
        "description": "Qwen3-TTS CustomVoice 0.6B - 9 premium speakers with instruction control",
        "quality": "standard",
        "supports_speakers": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
        "supports_instruction": True,
        "streaming": True,
        "backend": "qwen3-tts",
    },
    "qwen3-tts-customvoice-1.7b": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "sample_rate": 24000,
        "max_tokens": 2048,
        "description": "Qwen3-TTS CustomVoice 1.7B - 9 premium speakers, highest quality",
        "quality": "high",
        "supports_speakers": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"],
        "supports_instruction": True,
        "streaming": True,
        "backend": "qwen3-tts",
    },
    # Qwen3-TTS VoiceDesign model (text-to-voice synthesis)
    "qwen3-tts-voicedesign-1.7b": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "sample_rate": 24000,
        "max_tokens": 2048,
        "description": "Qwen3-TTS VoiceDesign - Create voices from natural language descriptions",
        "quality": "high",
        "supports_voice_design": True,
        "supports_instruction": True,
        "streaming": True,
        "backend": "qwen3-tts",
    },
    # Qwen3-TTS Base models (3-second voice cloning)
    "qwen3-tts-base-0.6b": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "sample_rate": 24000,
        "max_tokens": 2048,
        "description": "Qwen3-TTS Base 0.6B - 3-second rapid voice cloning",
        "quality": "standard",
        "supports_voice_cloning": True,
        "streaming": True,
        "backend": "qwen3-tts",
    },
    "qwen3-tts-base-1.7b": {
        "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "sample_rate": 24000,
        "max_tokens": 2048,
        "description": "Qwen3-TTS Base 1.7B - 3-second rapid voice cloning, best quality",
        "quality": "high",
        "supports_voice_cloning": True,
        "streaming": True,
        "backend": "qwen3-tts",
    },
}

# Qwen3-TTS CustomVoice speaker descriptions (for API documentation)
QWEN3_TTS_SPEAKERS = {
    "Vivian": "Bright, slightly edgy young female voice (Chinese native)",
    "Serena": "Warm, gentle young female voice (Chinese native)",
    "Uncle_Fu": "Seasoned male voice with low, mellow timbre (Chinese native)",
    "Dylan": "Youthful Beijing male voice with clear, natural timbre (Beijing dialect)",
    "Eric": "Lively Chengdu male voice with slightly husky brightness (Sichuan dialect)",
    "Ryan": "Dynamic male voice with strong rhythmic drive (English native)",
    "Aiden": "Sunny American male voice with clear midrange (English native)",
    "Ono_Anna": "Playful Japanese female voice with light, nimble timbre (Japanese native)",
    "Sohee": "Warm Korean female voice with rich emotion (Korean native)",
}

# Qwen3-TTS supported languages
QWEN3_TTS_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
    "Auto",  # Auto-detect from text
]


# Voice mapping (OpenAI voices → Pollinations voices)
# Pollinations supports many voices including:
# - OpenAI voices: alloy, echo, fable, onyx, nova, shimmer
# - Additional voices: various male/female options
VOICE_MAPPING = {
    "alloy": "alloy",
    "echo": "echo",
    "fable": "fable",
    "onyx": "onyx",
    "nova": "nova",
    "shimmer": "shimmer",
    # Additional Pollinations voices
    "rachel": "rachel",
    "drew": "drew",
    "clyde": "clyde",
    "sarah": "sarah",
    "jessie": "jessie",
    "joe": "joe",
    "emily": "emily",
    "bill": "bill",
    "michael": "michael",
    "matthew": "matthew",
    "linda": "linda",
    "charlotte": "charlotte",
    "james": "james",
    "daniel": "daniel",
    "kate": "kate",
    "elizabeth": "elizabeth",
    # Male voices
    "adam": "adam",
    "narrator": "narrator",
    "fin": "fin",
    # Female voices
    "bfem": "bfem",
    "bfm": "bfm",
    "gfem": "gfem",
    "gfm": "gfm",
    "mfem": "mfem",
    "mfm": "mfm",
}

# Pollinations TTS backend URL
POLLINATIONS_TTS_URL = "https://text.pollinations.ai"


# =============================================================================
# TTS HANDLER
# =============================================================================

@dataclass
class TTSHandler:
    """Handler for TTS models via Pollinations.ai or local models."""

    backend_url: str
    pollinations_url: str = POLLINATIONS_TTS_URL
    timeout: int = 60
    enable_local_models: bool = False

    def __post_init__(self):
        self._client: Optional[httpx.AsyncClient] = None

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
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a TTS model."""
        # Normalize model name
        model_key = model_name.lower().replace("qwen/qwen3-tts-", "qwen3-tts-")
        model_key = model_key.replace("qwen/qwen3-", "qwen3-")

        # Try exact match first
        if model_key in QWEN3_TTS_MODELS:
            return QWEN3_TTS_MODELS[model_key]

        # Try fuzzy match
        for key, config in QWEN3_TTS_MODELS.items():
            if model_key in key or key in model_key:
                return config

        # Default to pollinations if no match
        return {
            "model_id": "pollinations-tts",
            "sample_rate": 24000,
            "max_tokens": 4096,
            "description": "Pollinations.ai free TTS service",
            "quality": "standard",
            "language": "en",
            "backend": "pollinations",
        }

    async def generate_speech_pollinations(
        self,
        text: str,
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3"
    ) -> tuple[bytes, str, int]:
        """
        Generate speech from text using Pollinations.ai TTS.

        Returns:
            (audio_data, content_type, sample_rate)
        """
        # Normalize voice name
        if voice not in VOICE_MAPPING and voice not in ["default"]:
            logger.warning(f"Unknown voice '{voice}', using 'alloy'")
            voice = "alloy"
        elif voice == "default":
            voice = "alloy"

        # URL encode the text
        encoded_text = quote(text, safe='')

        # Pollinations TTS URL format:
        # https://text.pollinations.ai/{encoded_text}?model=openai&voice={voice}
        tts_url = f"{self.pollinations_url}/{encoded_text}"

        # Build query parameters
        params = {
            "model": "openai",  # Use OpenAI voice model
            "voice": voice,
        }

        # Add speed if not default
        if speed != 1.0:
            params["speed"] = str(speed)

        logger.info(f"Pollinations TTS request: voice={voice}, speed={speed}, text_len={len(text)}")

        try:
            response = await self.client.get(tts_url, params=params)
            response.raise_for_status()

            audio_data = response.content
            content_type = response.headers.get("content-type", get_content_type(response_format))

            # Pollinations typically returns 24kHz audio
            sample_rate = 24000

            logger.info(f"Pollinations TTS success: {len(audio_data)} bytes, content-type={content_type}")

            return audio_data, content_type, sample_rate

        except httpx.HTTPStatusError as e:
            logger.error(f"Pollinations TTS HTTP error: {e.response.status_code} - {e.response.text[:200]}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Pollinations TTS error: {e.response.text[:200]}"
            )
        except Exception as e:
            logger.error(f"Pollinations TTS error: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(e)}"
            )

    async def generate_speech(
        self,
        text: str,
        model: str,
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[Any] = None,
    ) -> tuple[bytes, str, int]:
        """
        Generate speech from text using the best available backend.

        Args:
            text: Text to synthesize
            model: Model name or ID
            voice: Voice name (for CustomVoice models)
            speed: Speech speed multiplier
            response_format: Output audio format
            ref_audio: Reference audio URL for voice cloning (Base models)
            ref_text: Transcript of reference audio for voice cloning
            voice_clone_prompt: Pre-computed voice clone prompt for reuse

        Returns:
            (audio_data, content_type, sample_rate)
        """
        # Get model config to determine backend
        model_config = self.get_model_config(model)
        backend = model_config.get("backend", "pollinations")

        if backend == "pollinations":
            return await self.generate_speech_pollinations(text, voice, speed, response_format)

        # Qwen3-TTS via official qwen-tts package
        if backend == "qwen3-tts":
            return await self.generate_speech_qwen3_official(
                text=text,
                model=model,
                model_config=model_config,
                voice=voice,
                speed=speed,
                response_format=response_format,
                ref_audio=ref_audio,
                ref_text=ref_text,
                voice_clone_prompt=voice_clone_prompt,
            )

        # Legacy fallback
        raise HTTPException(
            status_code=500,
            detail=f"Unknown TTS backend: {backend}"
        )

    async def generate_speech_qwen3_official(
        self,
        text: str,
        model: str,
        model_config: Dict[str, Any],
        voice: str = "alloy",
        speed: float = 1.0,
        response_format: str = "mp3",
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        voice_clone_prompt: Optional[Any] = None,
    ) -> tuple[bytes, str, int]:
        """
        Generate speech using official Qwen3-TTS package via qwen-tts.

        Supports three model types:
        - CustomVoice: Pre-built speakers with instruction control
        - VoiceDesign: Text-to-voice synthesis
        - Base/VoiceClone: 3-second rapid voice cloning

        Returns:
            (audio_data, content_type, sample_rate)
        """
        import torch
        import numpy as np
        import soundfile as sf

        model_id = model_config["model_id"]
        sample_rate = model_config["sample_rate"]

        logger.info(f"Loading Qwen3-TTS model: {model_id}")

        try:
            # Import the official Qwen3-TTS package
            from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer

            # Determine model type from model_id
            model_type = None
            if "CustomVoice" in model_id:
                model_type = "custom_voice"
            elif "VoiceDesign" in model_id:
                model_type = "voice_design"
            elif "Base" in model_id:
                model_type = "voice_clone"

            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def load_model():
                # Determine device and dtype
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

                # Try flash_attention_2 if available
                try:
                    return Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=device,
                        dtype=dtype,
                        attn_implementation="flash_attention_2",
                    )
                except Exception:
                    # Fallback without flash attention
                    return Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=device,
                        dtype=dtype,
                    )

            qwen_model = await loop.run_in_executor(None, load_model)
            logger.info(f"Model loaded ({model_type}), generating speech for text: {text[:50]}...")

            # Generate audio based on model type
            def generate_audio():
                if model_type == "custom_voice":
                    # CustomVoice: generate_custom_voice(text, language, speaker, instruct)
                    # Map OpenAI voice names to Qwen speakers
                    speaker = self._map_to_qwen_speaker(voice)
                    language = self._detect_language(text)  # Auto-detect or use default

                    # Use instruct parameter for emotion/style if provided in voice name
                    # Format: "voice_name:instruct" e.g., "alloy:speak happily"
                    instruct = None
                    if ":" in speaker:
                        speaker, instruct = speaker.split(":", 1)

                    wavs, sr = qwen_model.generate_custom_voice(
                        text=text,
                        language=language,
                        speaker=speaker,
                        instruct=instruct or "",
                    )
                    return wavs[0], sr

                elif model_type == "voice_design":
                    # VoiceDesign: generate_voice_design(text, language, instruct)
                    # Use voice parameter as the instruct for voice design
                    language = self._detect_language(text)
                    instruct = voice  # The "voice" parameter becomes the design instruction

                    wavs, sr = qwen_model.generate_voice_design(
                        text=text,
                        language=language,
                        instruct=instruct,
                    )
                    return wavs[0], sr

                elif model_type == "voice_clone":
                    # VoiceClone: generate_voice_clone(text, language, ref_audio, ref_text)
                    language = self._detect_language(text)

                    if voice_clone_prompt is not None:
                        # Reuse pre-computed voice clone prompt
                        wavs, sr = qwen_model.generate_voice_clone(
                            text=text,
                            language=language,
                            voice_clone_prompt=voice_clone_prompt,
                        )
                    elif ref_audio:
                        # Create new voice clone prompt from reference
                        wavs, sr = qwen_model.generate_voice_clone(
                            text=text,
                            language=language,
                            ref_audio=ref_audio,
                            ref_text=ref_text or "",
                        )
                    else:
                        # Fallback to CustomVoice if no reference provided
                        speaker = self._map_to_qwen_speaker(voice)
                        wavs, sr = qwen_model.generate_custom_voice(
                            text=text,
                            language=language,
                            speaker=speaker,
                        )
                    return wavs[0], sr

                else:
                    raise ValueError(f"Unknown Qwen3-TTS model type: {model_type}")

            audio_array, sr = await loop.run_in_executor(None, generate_audio)

            # Convert to numpy if needed
            if not isinstance(audio_array, np.ndarray):
                if torch.is_tensor(audio_array):
                    audio_array = audio_array.cpu().numpy()
                else:
                    audio_array = np.array(audio_array)

            # Normalize if floating point
            if audio_array.dtype in [np.float32, np.float64]:
                if np.max(np.abs(audio_array)) > 1.0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                audio_array = (audio_array * 32767).astype(np.int16)

            # Convert to requested format
            audio_bytes, content_type = self._convert_audio_format(
                audio_array, sr, response_format
            )

            logger.info(f"Qwen3-TTS success: {len(audio_bytes)} bytes, format={response_format}")

            return audio_bytes, content_type, sr

        except ImportError as e:
            logger.error(f"Missing qwen-tts package: {e}")
            raise HTTPException(
                status_code=500,
                detail="Qwen3-TTS requires the qwen-tts package. Install with: pip install qwen-tts"
            )
        except OSError as e:
            logger.error(f"Failed to load Qwen3-TTS model: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load TTS model. Check internet connection for model download: {e}"
            )
        except Exception as e:
            logger.error(f"Qwen3-TTS generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"TTS generation failed: {str(e)}"
            )

    def _map_to_qwen_speaker(self, voice: str) -> str:
        """Map OpenAI/Custom voice names to Qwen3-TTS CustomVoice speakers."""
        voice_lower = voice.lower()

        # Direct Qwen speaker names
        qwen_speakers = ["vivian", "serena", "uncle_fu", "dylan", "eric", "ryan", "aiden", "ono_anna", "sohee"]
        if voice_lower in [s.replace("_", "") for s in qwen_speakers]:
            return voice_lower.replace("_", "")

        # Map OpenAI voices to similar Qwen speakers
        openai_to_qwen = {
            "alloy": "Ryan",      # Dynamic female → Dynamic male
            "echo": "Aiden",       # Clear male → Sunny male
            "fable": "Ono_Anna",  # Playful female → Playful Japanese
            "onyx": "Uncle_Fu",   # Deep male → Seasoned male
            "nova": "Serena",     # Warm female → Warm female
            "shimmer": "Vivian",  # Bright female → Bright female
        }

        return openai_to_qwen.get(voice_lower, "Ryan")  # Default to Ryan

    def _detect_language(self, text: str) -> str:
        """
        Detect language from text or return Auto.

        Simple heuristic: check for CJK characters, etc.
        """
        import re

        # Check for Chinese characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return "Chinese"

        # Check for Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "Japanese"

        # Check for Korean characters
        if re.search(r'[\uac00-\ud7af]', text):
            return "Korean"

        # Default to Auto for let the model decide
        return "Auto"

    async def create_voice_clone_prompt(
        self,
        model: str,
        ref_audio: str,
        ref_text: str,
        x_vector_only_mode: bool = False,
    ) -> Any:
        """
        Create a reusable voice clone prompt from reference audio.

        This allows you to extract speaker features once and reuse them
        for multiple generations, avoiding recomputing features each time.

        Args:
            model: Base model ID (e.g., "qwen3-tts-base-1.7b")
            ref_audio: Reference audio URL or file path
            ref_text: Transcript of reference audio
            x_vector_only_mode: Use only speaker embedding (faster but lower quality)

        Returns:
            Voice clone prompt object for use with generate_voice_clone
        """
        import torch
        from qwen_tts import Qwen3TTSModel

        model_config = self.get_model_config(model)
        model_id = model_config["model_id"]

        if model_config.get("backend") != "qwen3-tts":
            raise HTTPException(
                status_code=400,
                detail="Voice cloning is only supported by Qwen3-TTS Base models"
            )

        try:
            loop = asyncio.get_event_loop()

            def load_and_create_prompt():
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

                qwen_model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=dtype,
                )

                return qwen_model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=x_vector_only_mode,
                )

            return await loop.run_in_executor(None, load_and_create_prompt)

        except Exception as e:
            logger.error(f"Failed to create voice clone prompt: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create voice clone prompt: {str(e)}"
            )

    def get_supported_speakers(self, model: str = None) -> list[str]:
        """
        Get list of supported speakers for Qwen3-TTS CustomVoice models.

        Args:
            model: Optional model ID (uses default CustomVoice model if not specified)

        Returns:
            List of speaker names
        """
        return list(QWEN3_TTS_SPEAKERS.keys())

    def get_supported_languages(self, model: str = None) -> list[str]:
        """
        Get list of supported languages for Qwen3-TTS.

        Args:
            model: Optional model ID

        Returns:
            List of language names
        """
        return QWEN3_TTS_LANGUAGES.copy()

    def _convert_audio_format(
        self,
        audio_array: any,
        sample_rate: int,
        output_format: str
    ) -> tuple[bytes, str]:
        """
        Convert audio array to the requested format.

        Returns:
            (audio_bytes, content_type)
        """
        import numpy as np
        import io

        # Ensure audio_array is a numpy array
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.int16)

        if output_format == "wav":
            # Write WAV file
            buffer = io.BytesIO()
            import wave
            with wave.open(buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_array.tobytes())
            return buffer.getvalue(), "audio/wav"

        elif output_format == "mp3":
            # Try to use pydub for MP3 conversion (requires ffmpeg)
            try:
                from pydub import AudioSegment

                # Create AudioSegment from raw audio data
                audio_segment = AudioSegment(
                    data=audio_array.tobytes(),
                    sample_width=audio_array.dtype.itemsize,
                    frame_rate=sample_rate,
                    channels=1
                )

                buffer = io.BytesIO()
                audio_segment.export(buffer, format="mp3", bitrate="128k")
                return buffer.getvalue(), "audio/mpeg"

            except ImportError:
                # Fallback to WAV if pydub not available
                logger.warning("pydub not available, falling back to WAV format")
                return self._convert_audio_format(audio_array, sample_rate, "wav")

        elif output_format == "flac":
            # Try to use soundfile for FLAC
            try:
                import soundfile
                buffer = io.BytesIO()
                soundfile.write(buffer, audio_array, sample_rate, format='FLAC')
                return buffer.getvalue(), "audio/flac"
            except ImportError:
                # Fallback to WAV
                logger.warning("soundfile not available, falling back to WAV format")
                return self._convert_audio_format(audio_array, sample_rate, "wav")

        else:
            # Default to WAV for unsupported formats
            return self._convert_audio_format(audio_array, sample_rate, "wav")


# =============================================================================
# TTS SERVICE (singleton)
# =============================================================================

_tts_handler: Optional[TTSHandler] = None


def get_tts_handler(backend_url: str) -> TTSHandler:
    """Get or create the TTS handler singleton."""
    global _tts_handler
    if _tts_handler is None or _tts_handler.backend_url != backend_url:
        _tts_handler = TTSHandler(backend_url=backend_url)
    return _tts_handler


async def close_tts_handler():
    """Close the TTS handler connection."""
    global _tts_handler
    if _tts_handler:
        await _tts_handler.close()
        _tts_handler = None


# =============================================================================
# CONTENT TYPE MAPPING
# =============================================================================

AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus;codecs=opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm16": "audio/pcm;codec=pcm-16",
}

AUDIO_EXTENSIONS = {
    "mp3": ".mp3",
    "opus": ".opus",
    "aac": ".aac",
    "flac": ".flac",
    "wav": ".wav",
    "pcm16": ".pcm",
}


def get_content_type(format: str) -> str:
    """Get content type for audio format."""
    return AUDIO_CONTENT_TYPES.get(format, "audio/wav")


def get_audio_extension(format: str) -> str:
    """Get file extension for audio format."""
    return AUDIO_EXTENSIONS.get(format, ".wav")
