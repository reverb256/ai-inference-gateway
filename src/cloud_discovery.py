"""
Cloud Model Discovery — OpenRouter, NVIDIA NIM, ZAI.

Auto-discovers available cloud models at startup and periodically refreshes.
Provides a curated registry with routing hints (cost tier, priority, specializations).

Architecture:
  - fetch_models(): Hit provider /v1/models endpoints
  - CuratedModelRegistry: Merges auto-discovered models with curated config
  - Routing hints: cost_tier, priority, specializations from contexts.py + curation config

Usage:
  from cloud_discovery import CloudModelRegistry
  registry = CloudModelRegistry(openrouter_key="...", nim_key="...", zai_key="...")
  await registry.start()  # starts background refresh
  models = registry.get_available_models()
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

import httpx

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CloudModelInfo:
    """A cloud model with routing metadata."""
    id: str
    name: str
    provider: str          # "openrouter", "nim", "zai"
    context_length: int
    max_output_tokens: int = 16384
    cost_tier: int = 3     # 1=cheapest, 5=most expensive
    priority: int = 5      # higher = preferred
    specializations: List[str] = field(default_factory=lambda: ["general"])
    pricing: Dict[str, float] = field(default_factory=dict)  # prompt/completion per 1M tokens
    free: bool = False
    available: bool = True
    last_checked: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "cost_tier": self.cost_tier,
            "priority": self.priority,
            "specializations": self.specializations,
            "pricing": self.pricing,
            "free": self.free,
            "available": self.available,
        }


# ── Curation rules ──────────────────────────────────────────────────────────
# Models we WANT in the gateway. Auto-discovery fills in context_length and pricing.
# We control priority, cost_tier, and specializations here.

# Provider prefixes that indicate "cloud" (vs local llama-servers)
CLOUD_PROVIDERS = {"openrouter", "nim", "zai"}

# Models to EXCLUDE from auto-discovery (dead, deprecated, duplicates)
EXCLUDED_MODELS: Set[str] = {
    "glm-4-flash",          # Dead on ZAI API
    "glm-4.7-flashx",       # 429 on coding plan
}

# Family-level curation — applied when a model matches a prefix
FAMILY_DEFAULTS = {
    # OpenRouter free models — high priority since they cost nothing
    ":free": {"cost_tier": 1, "priority": 8, "free": True},
    # ZAI coding plan models
    "glm-5.1": {"cost_tier": 4, "priority": 7, "specializations": ["agentic", "coding", "general"]},
    "glm-5-turbo": {"cost_tier": 4, "priority": 7, "specializations": ["agentic", "coding"]},
    "glm-5": {"cost_tier": 4, "priority": 6, "specializations": ["agentic", "general"]},
    "glm-4.7": {"cost_tier": 3, "priority": 5, "specializations": ["coding", "general"]},
    "glm-4.6": {"cost_tier": 2, "priority": 4, "specializations": ["general", "coding"]},
    "glm-4.5": {"cost_tier": 2, "priority": 3, "specializations": ["general"]},
    # NIM models — routed through OpenRouter or NIM endpoint
    "deepseek-ai/deepseek-v4": {"cost_tier": 3, "priority": 6, "specializations": ["coding", "reasoning"]},
    "meta/llama-3.1-405b": {"cost_tier": 4, "priority": 5, "specializations": ["general", "reasoning"]},
    "meta/llama-3.3-70b": {"cost_tier": 3, "priority": 5, "specializations": ["general", "coding"]},
    "meta/llama-4": {"cost_tier": 3, "priority": 5, "specializations": ["general"]},
    "qwen/qwen3-coder": {"cost_tier": 2, "priority": 6, "specializations": ["coding"]},
    "qwen/qwen3.5": {"cost_tier": 3, "priority": 5, "specializations": ["general", "reasoning"]},
    "qwen/qwen3-next": {"cost_tier": 3, "priority": 4, "specializations": ["general"]},
    "moonshotai/kimi": {"cost_tier": 3, "priority": 5, "specializations": ["general", "reasoning"]},
    "mistralai/": {"cost_tier": 3, "priority": 4, "specializations": ["general"]},
    "google/gemma-3-27b": {"cost_tier": 2, "priority": 4, "specializations": ["general"]},
    "google/gemma-4": {"cost_tier": 2, "priority": 5, "specializations": ["general"]},
    "openai/gpt-oss": {"cost_tier": 2, "priority": 5, "specializations": ["general"]},
    "nvidia/nemotron": {"cost_tier": 2, "priority": 4, "specializations": ["general"]},
    "nvidia/nvidia-nemotron": {"cost_tier": 2, "priority": 5, "specializations": ["general", "reasoning"]},
    "nvidia/llama-3": {"cost_tier": 2, "priority": 4, "specializations": ["general", "reasoning"]},
    "minimaxai/": {"cost_tier": 3, "priority": 4, "specializations": ["general"]},
    "z-ai/glm": {"cost_tier": 3, "priority": 5, "specializations": ["coding"]},
}

# Which models to INCLUDE from auto-discovery.
# If non-empty, ONLY these patterns are included. If empty, include all (minus EXCLUDED).
INCLUDE_PATTERNS: List[str] = [
    # All ZAI coding plan models
    "glm-5.1", "glm-5-turbo", "glm-5", "glm-4.7", "glm-4.7-flash",
    "glm-4.6", "glm-4.6v", "glm-4.5", "glm-4.5-flash", "glm-4.5-air",
    # NIM models we use
    "deepseek-ai/deepseek-v4",
    "meta/llama-3.1-405b", "meta/llama-3.3-70b", "meta/llama-4-maverick",
    "meta/llama-3.2-90b",
    "qwen/qwen3-coder", "qwen/qwen3.5", "qwen/qwen3-next",
    "moonshotai/kimi",
    "mistralai/mistral-large", "mistralai/devstral", "mistralai/mistral-small",
    "nvidia/nemotron", "nvidia/nvidia-nemotron", "nvidia/llama-3",
    "minimaxai/minimax",
    "openai/gpt-oss",
    "z-ai/glm",
    "google/gemma-3-27b", "google/gemma-4",
    "microsoft/phi-4",
    "stepfun-ai/", "bytedance/", "stockmark/",
    # Free models from OpenRouter
    ":free",
]


# ── Discovery functions ──────────────────────────────────────────────────────

def _match_patterns(model_id: str, patterns: List[str]) -> bool:
    """Check if model_id matches any include pattern."""
    if not patterns:
        return True
    for pat in patterns:
        if pat == ":free":
            continue  # handled separately
        if pat in model_id:
            return True
    return False


def _apply_curation(model_id: str) -> dict:
    """Apply family-level curation defaults based on model ID prefix."""
    defaults = {"cost_tier": 3, "priority": 5, "specializations": ["general"], "free": False}
    for prefix, overrides in FAMILY_DEFAULTS.items():
        if model_id.startswith(prefix) or (prefix == ":free" and ":free" in model_id):
            defaults.update(overrides)
    return defaults


async def fetch_openrouter_models(api_key: str) -> List[CloudModelInfo]:
    """Fetch available models from OpenRouter /api/v1/models."""
    models = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid in EXCLUDED_MODELS:
                    continue
                if not _match_patterns(mid, INCLUDE_PATTERNS):
                    # Also check free models
                    pricing = m.get("pricing", {})
                    is_free = pricing.get("prompt", "1") == "0"
                    if not is_free:
                        continue

                pricing = m.get("pricing", {})
                is_free = pricing.get("prompt", "1") == "0"
                curation = _apply_curation(mid)
                if is_free:
                    curation.update({"cost_tier": 1, "priority": 8, "free": True})
                # Don't duplicate 'free' — set via curation
                curation["free"] = is_free if not curation.get("free") else True

                models.append(CloudModelInfo(
                    id=mid,
                    name=m.get("name", mid),
                    provider="openrouter",
                    context_length=m.get("context_length", 8192),
                    pricing={
                        "prompt_per_1m": float(pricing.get("prompt", 0) or 0),
                        "completion_per_1m": float(pricing.get("completion", 0) or 0),
                    },
                    **{k: v for k, v in curation.items() if k != "free"},
                    free=curation["free"],
                ))
            logger.info(f"OpenRouter: discovered {len(models)} curated models")
    except Exception as e:
        logger.error(f"OpenRouter discovery failed: {e}")
    return models


async def fetch_nim_models(api_key: str) -> List[CloudModelInfo]:
    """Fetch available models from NVIDIA NIM /v1/models."""
    models = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid in EXCLUDED_MODELS:
                    continue
                if not _match_patterns(mid, INCLUDE_PATTERNS):
                    continue

                curation = _apply_curation(mid)
                models.append(CloudModelInfo(
                    id=mid,
                    name=m.get("id", mid),
                    provider="nim",
                    context_length=m.get("max_model_len") or m.get("context_length", 8192),
                    **curation,
                ))
            logger.info(f"NIM: discovered {len(models)} curated models")
    except Exception as e:
        logger.error(f"NIM discovery failed: {e}")
    return models


async def fetch_zai_models(api_key: str) -> List[CloudModelInfo]:
    """Fetch available models from Z.AI coding plan /api/coding/paas/v4/models."""
    models = []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                "https://api.z.ai/api/coding/paas/v4/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()

            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid in EXCLUDED_MODELS:
                    continue

                curation = _apply_curation(mid)
                models.append(CloudModelInfo(
                    id=mid,
                    name=mid,
                    provider="zai",
                    context_length=curation.get("context_length", 131072),
                    **curation,
                ))
            logger.info(f"ZAI: discovered {len(models)} models")
    except Exception as e:
        logger.error(f"ZAI discovery failed: {e}")
    return models


# ── Main registry ────────────────────────────────────────────────────────────

class CloudModelRegistry:
    """
    Auto-discovers cloud models from OpenRouter, NIM, and ZAI.
    Merges with curated routing config from contexts.py.
    """

    def __init__(
        self,
        openrouter_key: Optional[str] = None,
        nim_key: Optional[str] = None,
        zai_key: Optional[str] = None,
        refresh_interval: int = 3600,  # 1 hour
    ):
        self.openrouter_key = openrouter_key
        self.nim_key = nim_key
        self.zai_key = zai_key
        self.refresh_interval = refresh_interval
        self.models: Dict[str, CloudModelInfo] = {}
        self._refresh_task: Optional[asyncio.Task] = None

    async def start(self):
        """Initial discovery + start background refresh."""
        await self.refresh()
        self._refresh_task = asyncio.create_task(self._refresh_loop())
        logger.info(f"Cloud discovery started ({len(self.models)} models, refresh={self.refresh_interval}s)")

    async def stop(self):
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None

    async def _refresh_loop(self):
        while True:
            try:
                await asyncio.sleep(self.refresh_interval)
                await self.refresh()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cloud discovery refresh error: {e}")
                await asyncio.sleep(300)

    async def refresh(self):
        """Fetch models from all configured providers and merge."""
        tasks = []
        if self.openrouter_key:
            tasks.append(fetch_openrouter_models(self.openrouter_key))
        if self.nim_key:
            tasks.append(fetch_nim_models(self.nim_key))
        if self.zai_key:
            tasks.append(fetch_zai_models(self.zai_key))

        if not tasks:
            logger.warning("No cloud API keys configured, skipping cloud discovery")
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge: OpenRouter enriches NIM models with pricing/context,
        # ZAI models take priority for ZAI-registered IDs
        new_registry: Dict[str, CloudModelInfo] = {}

        # Priority: zai > openrouter > nim (for same model_id)
        provider_order = {"nim": 0, "openrouter": 1, "zai": 2}

        for result in results:
            if isinstance(result, Exception):
                continue
            for model in result:
                existing = new_registry.get(model.id)
                if existing is None or provider_order.get(model.provider, 0) > provider_order.get(existing.provider, 0):
                    model.last_checked = datetime.now()
                    new_registry[model.id] = model

        # Enrich with context lengths from contexts.py if available
        try:
            from .contexts import get_context_length, get_max_tokens
            for mid, model in new_registry.items():
                discovered_ctx = model.context_length
                curated_ctx = get_context_length(mid)
                # Use the larger of discovered vs curated
                if curated_ctx and curated_ctx > discovered_ctx:
                    model.context_length = curated_ctx
                model.max_output_tokens = get_max_tokens(mid)
        except ImportError:
            pass

        self.models = new_registry
        logger.info(f"Cloud registry: {len(self.models)} models from {len(tasks)} providers")

    def get_available_models(self) -> Dict[str, CloudModelInfo]:
        return {k: v for k, v in self.models.items() if v.available}

    def get_model(self, model_id: str) -> Optional[CloudModelInfo]:
        return self.models.get(model_id)

    def get_models_by_provider(self, provider: str) -> Dict[str, CloudModelInfo]:
        return {k: v for k, v in self.models.items() if v.provider == provider}

    def get_models_by_specialization(self, spec: str) -> Dict[str, CloudModelInfo]:
        return {k: v for k, v in self.models.items() if spec in v.specializations}

    def get_free_models(self) -> Dict[str, CloudModelInfo]:
        return {k: v for k, v in self.models.items() if v.free}

    def get_backend_for_model(self, model_id: str) -> Optional[str]:
        """Return the backend name for routing (openrouter/nim/zai)."""
        model = self.models.get(model_id)
        if model:
            return model.provider
        return None

    def get_base_url(self, provider: str) -> str:
        """Get the API base URL for a provider."""
        urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "nim": "https://integrate.api.nvidia.com/v1",
            "zai": os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
        }
        return urls.get(provider, "")

    def get_api_key(self, provider: str) -> Optional[str]:
        """Get the API key for a provider."""
        keys = {
            "openrouter": self.openrouter_key,
            "nim": self.nim_key,
            "zai": self.zai_key,
        }
        return keys.get(provider)

    def to_dict(self) -> dict:
        """Serialize registry for /v1/models endpoint and debugging."""
        return {
            "total": len(self.models),
            "providers": {
                p: len([m for m in self.models.values() if m.provider == p])
                for p in {"openrouter", "nim", "zai"}
            },
            "free_count": len([m for m in self.models.values() if m.free]),
            "models": {k: v.to_dict() for k, v in sorted(self.models.items())},
        }
