"""
Model Discovery Module

Auto-discovers available models from llama-servers, LM Studio, and other backends.
Maintains a registry of model_id -> backend_url mappings for intelligent routing.

Backends are queried at startup and periodically refreshed to pick up model changes.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx


logger = logging.getLogger(__name__)


_sync_client = httpx.Client(timeout=10.0)


@dataclass
class BackendInfo:
    name: str
    base_url: str
    priority: int = 0
    enabled: bool = True
    models: Optional[List[str]] = None
    last_checked: Optional[datetime] = None
    health_status: str = "unknown"
    model_filter: Optional[str] = None


class ModelDiscovery:

    BACKENDS = {
        "llama-3090": BackendInfo(
            name="llama-3090",
            base_url="http://llama-server-zephyr-3090-moe.ai-inference.svc.cluster.local:1237/v1",
            priority=11,
        ),
        "llama-3060ti": BackendInfo(
            name="llama-3060ti",
            base_url="http://llama-server-zephyr-3060ti.ai-inference.svc.cluster.local:1236/v1",
            priority=10,
        ),
        "llama-sentry": BackendInfo(
            name="llama-sentry",
            base_url="http://llama-server-sentry.ai-inference.svc.cluster.local:1235/v1",
            priority=9,
        ),
    }

    def __init__(
        self,
        refresh_interval: int = 300,
        extra_backends: Optional[Dict] = None,
        disabled_models: Optional[Set[str]] = None,
    ):
        self.refresh_interval = refresh_interval
        self.model_registry: Dict[str, str] = {}
        self.backend_registry: Dict[str, BackendInfo] = {}
        self.disabled_models: Set[str] = disabled_models or set()
        self._refresh_task: Optional[asyncio.Task] = None
        self._client: Optional[httpx.AsyncClient] = None

        if extra_backends:
            for name, info in extra_backends.items():
                if isinstance(info, dict):
                    info_copy = dict(info)
                    info_copy.pop("provider", None)
                    model_filter = info_copy.pop("model", None)
                    if model_filter:
                        info_copy["model_filter"] = model_filter
                    self.BACKENDS[name] = BackendInfo(**info_copy)
                else:
                    self.BACKENDS[name] = info
            logger.info(f"Merged {len(extra_backends)} extra backend(s) from config")

        if self.disabled_models:
            logger.info(f"Disabled models configured: {len(self.disabled_models)} entries")

    def _is_model_disabled(self, model_id: str) -> bool:
        return model_id in self.disabled_models

    def _log_skipped(self, model_id: str, backend_name: str, reason: str):
        logger.debug(f"Skipping {model_id} on {backend_name} ({reason})")

    async def start(self):
        if self._refresh_task is None:
            self._client = httpx.AsyncClient(timeout=10.0)
            self._refresh_task = asyncio.create_task(self._refresh_loop())
            logger.info("Model discovery started")

    async def stop(self):
        if self._refresh_task:
            self._refresh_task.cancel()
            self._refresh_task = None
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Model discovery stopped")

    async def _refresh_loop(self):
        while True:
            try:
                await self.refresh_all_backends()
                await asyncio.sleep(self.refresh_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh loop: {e}")
                await asyncio.sleep(60)

    async def refresh_all_backends(self) -> Dict[str, BackendInfo]:
        logger.info("Refreshing model registry...")
        discovered = {}
        total_skipped = 0

        for backend_name, backend_info in self.BACKENDS.items():
            if not backend_info.enabled:
                continue

            models = await self._discover_backend_models(backend_info)
            if models:
                backend_info.models = models
                backend_info.last_checked = datetime.now()
                backend_info.health_status = "healthy"
                discovered[backend_name] = backend_info

                accepted = 0
                for model_id in models:
                    if self._is_model_disabled(model_id):
                        self._log_skipped(model_id, backend_name, "disabled")
                        total_skipped += 1
                        continue
                    if backend_info.model_filter and model_id != backend_info.model_filter:
                        self._log_skipped(model_id, backend_name, f"filter: {backend_info.model_filter}")
                        continue
                    self.model_registry[model_id] = backend_name
                    accepted += 1
                    logger.debug(f"Discovered {model_id} on {backend_name}")

                if backend_info.model_filter and accepted == 0:
                    logger.warning(f"model_filter '{backend_info.model_filter}' not found on {backend_name}")

        self.backend_registry = discovered

        total_models = sum(len(b.models) for b in discovered.values()) if discovered else 0
        logger.info(
            f"Model discovery complete: {len(discovered)} backends, "
            f"{total_models} models total"
        )
        if total_skipped:
            logger.info(f"Skipped {total_skipped} disabled model(s)")

        return discovered

    def _discover_backend_models_sync(self, backend: BackendInfo, retries: int = 3) -> Optional[List[str]]:
        import time

        for attempt in range(retries):
            try:
                response = _sync_client.get(
                    f"{backend.base_url}/models",
                    headers={"Accept": "application/json"},
                    timeout=5.0
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
                if attempt < retries - 1:
                    wait_time = 2 ** attempt
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
        logger.info("Refreshing model registry (synchronous)...")
        discovered = {}
        total_skipped = 0

        for backend_name, backend_info in self.BACKENDS.items():
            if not backend_info.enabled:
                continue

            models = self._discover_backend_models_sync(backend_info)
            if models:
                backend_info.models = models
                backend_info.last_checked = datetime.now()
                backend_info.health_status = "healthy"
                discovered[backend_name] = backend_info

                accepted = 0
                for model_id in models:
                    if self._is_model_disabled(model_id):
                        self._log_skipped(model_id, backend_name, "disabled")
                        total_skipped += 1
                        continue
                    if backend_info.model_filter and model_id != backend_info.model_filter:
                        self._log_skipped(model_id, backend_name, f"filter: {backend_info.model_filter}")
                        continue
                    self.model_registry[model_id] = backend_name
                    accepted += 1
                    logger.debug(f"Discovered {model_id} on {backend_name}")

                if backend_info.model_filter and accepted == 0:
                    logger.warning(f"model_filter '{backend_info.model_filter}' not found on {backend_name}")

        self.backend_registry = discovered

        total_models = sum(len(b.models) for b in discovered.values()) if discovered else 0
        logger.info(
            f"Model discovery complete: {len(discovered)} backends, "
            f"{total_models} models total"
        )
        if total_skipped:
            logger.info(f"Skipped {total_skipped} disabled model(s)")

        return discovered

    async def _discover_backend_models(self, backend: BackendInfo) -> Optional[List[str]]:
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
        return self.model_registry.get(model_id)

    def get_backend_url(self, backend_name: str) -> Optional[str]:
        backend = self.backend_registry.get(backend_name)
        if backend:
            return backend.base_url
        return self.BACKENDS.get(backend_name, {}).base_url if backend_name in self.BACKENDS else None

    def get_all_models(self) -> Dict[str, str]:
        return self.model_registry.copy()

    def get_backend_status(self) -> Dict[str, Dict]:
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
