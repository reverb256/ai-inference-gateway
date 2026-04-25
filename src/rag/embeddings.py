"""
Embedding Service for RAG.

Handles text and multimodal embedding using BidirLM-Omni or BGE-M3:
- Dense embeddings (semantic search)
- Multimodal embeddings (text + image, BidirLM-Omni only)
- GPU acceleration
- Batch processing
"""

import asyncio
import base64
import io
import json
import logging
import os
import sys
import types
from typing import List, Optional, Dict, Union
import torch

from .config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _patch_sentence_transformers_compat():
    """Compatibility shim for BidirLM-Omni with sentence-transformers v5.3.0.

    BidirLM-Omni (built for st 5.4 + transformers 5.5) needs three fixes
    to load on our st 5.3 + transformers 5.3 stack:
    1. sentence_transformers.base.modules.Transformer (moved to .models in v5)
    2. Transformer.__init__(modality_config=...) (added in v5.4)
    3. self.model -> self.auto_model (renamed in v5)
    """
    try:
        from sentence_transformers import models
        from sentence_transformers.models.Transformer import Transformer

        # Shim 1: old import path alias
        base_pkg = sys.modules.get("sentence_transformers.base")
        if base_pkg is None:
            base_pkg = types.ModuleType("sentence_transformers.base")
            base_pkg.__path__ = []
            sys.modules["sentence_transformers.base"] = base_pkg
        modules_pkg = sys.modules.get("sentence_transformers.base.modules")
        if modules_pkg is None:
            modules_pkg = types.ModuleType("sentence_transformers.base.modules")
            sys.modules["sentence_transformers.base.modules"] = modules_pkg
        if not hasattr(modules_pkg, "Transformer"):
            modules_pkg.Transformer = Transformer

        # Shim 2: accept modality_config and similar kwargs added in v5.4+
        original_init = Transformer.__init__
        if not hasattr(Transformer, "_compat_patched"):
            import inspect

            known_params = set()
            sig = inspect.signature(original_init)
            for p in sig.parameters.values():
                if p.name != "self":
                    known_params.add(p.name)

            def _compat_init(self, *args, **kwargs):
                extra = {k: v for k, v in kwargs.items() if k not in known_params}
                if extra:
                    logger.debug(f"Transformer compat: filtered unknown kwargs: {list(extra.keys())}")
                filtered = {k: v for k, v in kwargs.items() if k in known_params}
                original_init(self, *args, **filtered)

            Transformer.__init__ = _compat_init

            # Shim 3 removed: sentence-transformers v5 already has both
            # .model and .auto_model — adding a property creates infinite recursion
            # (auto_model -> model -> auto_model -> ...)

            Transformer._compat_patched = True
    except ImportError:
        pass


def _patch_tokenizer_config(model_name: str):
    """Remove fix_mistral_regex from tokenizer_config for transformers < 5.5 compat."""
    try:
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(
            repo_id=model_name, filename="tokenizer_config.json"
        )
        with open(config_path) as f:
            cfg = json.load(f)
        if "fix_mistral_regex" in cfg:
            del cfg["fix_mistral_regex"]
            with open(config_path, "w") as f:
                json.dump(cfg, f, indent=2)
            logger.debug("Removed fix_mistral_regex from tokenizer config")
    except Exception:
        pass


_patch_sentence_transformers_compat()

from sentence_transformers import SentenceTransformer  # noqa: E402


def _load_image(image_input: Union[str, bytes]) -> any:
    """Load image from base64 string, URL, or bytes."""
    from PIL import Image

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))

    # Base64 data URI
    if image_input.startswith("data:"):
        _, b64_data = image_input.split(",", 1)
        img_bytes = base64.b64decode(b64_data)
        return Image.open(io.BytesIO(img_bytes))

    # Raw base64
    try:
        img_bytes = base64.b64decode(image_input)
        return Image.open(io.BytesIO(img_bytes))
    except Exception:
        pass

    # File path
    if os.path.exists(image_input):
        return Image.open(image_input)

    raise ValueError(f"Cannot load image from input (len={len(image_input)})")


class EmbeddingService:
    """
    Embedding service supporting BidirLM-Omni (multimodal, 2048d) or BGE-M3 (text, 1024d).
    """

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model: Optional[SentenceTransformer] = None
        self._device: Optional[str] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the embedding model (lazy loading)."""
        async with self._lock:
            if self._model is not None:
                return

            try:
                logger.info(f"Loading embedding model: {self.config.model}")

                if self.config.trust_remote_code:
                    _patch_tokenizer_config(self.config.model)
                    # ST5 tries top-level import for custom modules — add model dir to sys.path
                    try:
                        from huggingface_hub import snapshot_download
                        model_dir = snapshot_download(
                            self.config.model,
                            local_files_only=True,
                        )
                        if model_dir not in sys.path:
                            sys.path.insert(0, model_dir)
                            logger.info(f"Added model dir to sys.path: {model_dir}")
                    except Exception as path_err:
                        logger.warning(f"Could not add model dir to sys.path: {path_err}")

                if self.config.device == "cuda" and torch.cuda.is_available():
                    self._device = "cuda"
                    logger.info(f"Using CUDA for embeddings (GPU: {torch.cuda.get_device_name(0)})")
                else:
                    self._device = "cpu"
                    logger.info("Using CPU for embeddings")

                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: SentenceTransformer(
                        self.config.model,
                        device=self._device,
                        trust_remote_code=self.config.trust_remote_code,
                    ),
                )

                actual_dims = self._model.get_sentence_embedding_dimension()
                if actual_dims != self.config.dimensions:
                    logger.info(
                        f"Embedding dimensions: {actual_dims} (configured: {self.config.dimensions}). "
                        f"Using actual dimensions from model."
                    )
                    self.config.dimensions = actual_dims

                logger.info(f"Embedding model loaded (dims: {actual_dims})")

            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise

    async def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for texts."""
        if self._model is None:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate dense embeddings: {e}")
            raise

    async def embed_multimodal(
        self, texts: List[str], images: Optional[List[Union[str, bytes]]] = None
    ) -> List[List[float]]:
        """Generate embeddings for text+image pairs (BidirLM-Omni only).

        If images are provided, each image is paired with the text at the same index.
        Images can be base64 strings, data URIs, file paths, or bytes.
        """
        if self._model is None:
            await self.initialize()

        if not images:
            return await self.embed_dense(texts)

        try:
            inputs = []
            for i, text in enumerate(texts):
                img = images[i] if i < len(images) else None
                if img is not None:
                    pil_img = _load_image(img)
                    inputs.append({"text": text, "image": pil_img})
                else:
                    inputs.append(text)

            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    inputs,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                ),
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to generate multimodal embeddings: {e}")
            raise

    async def embed_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """Generate sparse embeddings (BM25-like). BGE-M3 only; BidirLM-Omni returns empty."""
        if self._model is None:
            await self.initialize()

        try:
            loop = asyncio.get_event_loop()
            sparse_embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    output_value="sparse",
                    show_progress_bar=False,
                ),
            )
            results = []
            for i in range(len(texts)):
                sparse_row = sparse_embeddings[i]
                token_weights = {
                    int(idx): float(weight)
                    for idx, weight in zip(sparse_row.indices, sparse_row.data)
                }
                results.append(token_weights)
            return results
        except Exception as e:
            logger.error(f"Failed to generate sparse embeddings: {e}")
            logger.warning("Using empty sparse embeddings as fallback")
            return [{} for _ in texts]

    async def embed_single(self, text: str) -> List[float]:
        """Generate dense embedding for a single text."""
        embeddings = await self.embed_dense([text])
        return embeddings[0]

    def is_initialized(self) -> bool:
        return self._model is not None

    async def shutdown(self) -> None:
        async with self._lock:
            if self._model is not None:
                if self._device == "cuda":
                    del self._model
                    torch.cuda.empty_cache()
                    logger.info("Freed CUDA memory for embedding model")
                self._model = None
                logger.info("Embedding service shutdown complete")


async def create_embedding_service(config: EmbeddingConfig) -> EmbeddingService:
    service = EmbeddingService(config)
    await service.initialize()
    return service
