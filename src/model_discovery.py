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


# Synchronous client for initial discovery (before event loop is ready)
_sync_client = httpx.Client(timeout=10.0)


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

    # Backend configurations — use K8s DNS service names.
    # ClusterIPs change on redeploy; K8s DNS is stable.
    BACKENDS = {
        "llama-3090": BackendInfo(
            name="llama-3090",
            base_url="http://llama-server-zephyr-3090-moe.ai-inference.svc.cluster.local:1237/v1",
            priority=11,  # Highest — 35B MoE model on RTX 3090 24GB (flex slot)
        ),
        "llama-3060ti": BackendInfo(
            name="llama-3060ti",
            base_url="http://llama-server-zephyr-3060ti.ai-inference.svc.cluster.local:1236/v1",
            priority=10,  # Primary (9B model, zephyr RTX 3060 Ti 8GB)
        ),
        "llama-sentry": BackendInfo(
            name="llama-sentry",
            base_url="http://llama-server-sentry.ai-inference.svc.cluster.local:1235/v1",
            priority=9,   # Secondary (4B model, sentry RX 5600 XT 8GB AMD)
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

    def _discover_backend_models_sync(self, backend: BackendInfo, retries: int = 3) -> Optional[List[str]]:
        """
        Query a backend for available models (synchronous with retries).

        Args:
            backend: Backend to query
            retries: Number of retry attempts

        Returns:
            List of model IDs, or None if query failed
        """
        import time

        for attempt in range(retries):
            try:
                response = _sync_client.get(
                    f"{backend.base_url}/models",
                    headers={"Accept": "application/json"},
                    timeout=5.0  # Short timeout for faster failure detection
                )
                response.raise_for_status()

                data = response.json()
                if "data" not in data:
                    logger.warning(f"Backend {backend.name} returned unexpected format")
                    return None

                models = [item["id"] for item in data["data"]]
                logger.debug(f"Discovered {len(models)} models on {backend.name}")
                return models

            except httpx.ConnectError as e:
                # Connection refused - backend might be starting up
                if attempt < retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.debug(f"Backend {backend.name} connection failed (attempt {attempt + 1}/{retries}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                logger.warning(f"Failed to query {backend.name} after {retries} attempts: {e}")
                backend.health_status = "unreachable"
                return None
            except httpx.HTTPError as e:
                logger.warning(f"Failed to query {backend.name}: {e}")
                backend.health_status = "unreachable"
                return None
            except Exception as e:
                logger.error(f"Error querying {backend.name}: {e}")
                backend.health_status = "error"
                return None

        return None

    def refresh_all_backends_sync(self) -> Dict[str, BackendInfo]:
        """
        Query all backends synchronously (for initial discovery).

        Returns:
            Dict of backend name → BackendInfo with discovered models
        """
        logger.info("Refreshing model registry (synchronous)...")
        discovered = {}

        for backend_name, backend_info in self.BACKENDS.items():
            if not backend_info.enabled:
                continue

            models = self._discover_backend_models_sync(backend_info)
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
