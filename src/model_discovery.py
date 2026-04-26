"""
Model Discovery Module

Auto-discovers available models from llama-servers, LM Studio, and other backends.
Maintains a registry of model_id → backend_url mappings for intelligent routing.

Backends are queried at startup and periodically refreshed to pick up model changes.
"""

import asyncio
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx


logger = logging.getLogger(__name__)


@dataclass
class BackendInfo:
    """Information about a model backend."""
    name: str
    base_url: str
    priority: int = 0
    enabled: bool = True
    models: Optional[List[str]] = None
    last_checked: Optional[datetime] = None
    health_status: str = "unknown"


class ModelDiscovery:
    """
    Auto-discovers models from multiple backends.

    Queries configured backends for their available models and maintains
    a registry for intelligent routing.
    """

    # Backend configurations (using K8s service names)
    BACKENDS = {
        "llama-3090": BackendInfo(
            name="llama-3090",
            base_url="http://llama-server-zephyr.ai-inference.svc.cluster.local:1235/v1",
            priority=10,  # Highest priority (24GB VRAM)
        ),
        "llama-3060ti": BackendInfo(
            name="llama-3060ti",
            base_url="http://llama-server-zephyr-3060ti.ai-inference.svc.cluster.local:1236/v1",
            priority=9,   # Secondary (8GB VRAM)
        ),
        "lmstudio": BackendInfo(
            name="lmstudio",
            base_url="http://10.1.1.110:1234/v1",  # Host-only, not in K8s
            priority=8,   # Desktop client
        ),
        "llama-sentry": BackendInfo(
            name="llama-sentry",
            base_url="http://llama-server-sentry.ai-inference.svc.cluster.local:1235/v1",
            priority=7,   # AMD GPU (8GB)
        ),
    }

    def __init__(self, refresh_interval: int = 300):
        """
        Initialize model discovery.

        Args:
            refresh_interval: Seconds between backend refreshes (default 5 min)
        """
        self.refresh_interval = refresh_interval
        self.model_registry: Dict[str, str] = {}  # model_id → backend name
        self.backend_registry: Dict[str, BackendInfo] = {}
        self._refresh_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

    async def start(self):
        """Start background refresh task."""
        if self._refresh_task is None:
            self._client = httpx.AsyncClient(timeout=10.0)
            self._refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Model discovery started")

    async def stop(self):
        """Stop background refresh task."""
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Model discovery stopped")

    async def _refresh_loop(self):
        """Periodically refresh model registry."""
        while True:
            try:
                await self.refresh_all_backends()
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def refresh_all_backends(self) -> Dict[str, BackendInfo]:
        """
        Query all backends and update model registry.

        Returns:
            Dict of backend name → BackendInfo with discovered models
        """
        logger.info("Refreshing model registry...")
        discovered = {}

        for backend_name, backend_info in self.BACKENDS.items():
            if not backend_info.enabled:
                continue

            models = await self._discover_backend_models(backend_info)
            if models:
                backend_info.models = models
                backend_info.last_checked = datetime.now()
                backend_info.health_status = "healthy"
                discovered[backend_name] = backend_info

                # Update model registry
                for model_id in models:
                    # Use exact model ID from backend
                    self.model_registry[model_id] = backend_name
                    logger.debug(f"Discovered {model_id} on {backend_name}")

        self.backend_registry = discovered

        # Log summary
        total_models = sum(len(b.models) for b in discovered.values())
        logger.info(
            f"Model discovery complete: {len(discovered)} backends, "
            f"{total_models} models total"
        )

        return discovered

    async def _discover_backend_models(self, backend: BackendInfo) -> Optional[List[str]]:
        """
        Query a backend for available models.

        Args:
            backend: Backend to query

        Returns:
            List of model IDs, or None if query failed
        """
        try:
            if not self._client:
                return None

            response = await self._client.get(
                f"{backend.base_url}/models",
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()

            data = response.json()
            if "data" not in data:
                logger.warning(f"Backend {backend.name} returned unexpected format")
                return None

            models = [item["id"] for item in data["data"]]
            logger.debug(f"Discovered {len(models)} models on {backend.name}")
            return models

        except httpx.HTTPError as e:
            logger.warning(f"Failed to query {backend.name}: {e}")
            backend.health_status = "unreachable"
            return None
        except Exception as e:
            logger.error(f"Error querying {backend.name}: {e}")
            backend.health_status = "error"
            return None

    def get_backend_for_model(self, model_id: str) -> Optional[str]:
        """
        Get the backend name for a given model ID.

        Args:
            model_id: Model ID to look up

        Returns:
            Backend name, or None if model not found
        """
        return self.model_registry.get(model_id)

    def get_backend_url(self, backend_name: str) -> Optional[str]:
        """
        Get the base URL for a backend.

        Args:
            backend_name: Backend name

        Returns:
            Backend URL, or None if backend not found
        """
        backend = self.backend_registry.get(backend_name)
        if backend:
            return backend.base_url
        return self.BACKENDS.get(backend_name, {}).base_url if backend_name in self.BACKENDS else None

    def get_all_models(self) -> Dict[str, str]:
        """
        Get all discovered models.

        Returns:
            Dict of model_id → backend_name
        """
        return self.model_registry.copy()

    def get_backend_status(self) -> Dict[str, Dict]:
        """
        Get status of all backends.

        Returns:
            Dict of backend_name → status info
        """
        return {
            name: {
                "base_url": info.base_url,
                "health": info.health_status,
                "models": info.models or [],
                "last_checked": info.last_checked.isoformat() if info.last_checked else None,
            }
            for name, info in {
                **self.backend_registry,
                **{k: v for k, v in self.BACKENDS.items() if v.enabled}
            }.items()
        }
