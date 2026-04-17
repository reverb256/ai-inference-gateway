"""
Audit Log Plugin — JSONL request logging.

Logs every request/response pair to a JSONL file for debugging,
analytics, and compliance. Zero dependencies beyond stdlib.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException, Request

from ai_inference_gateway.middleware.plugin_base import GatewayPlugin

logger = logging.getLogger(__name__)


class AuditLogPlugin(GatewayPlugin):
    """
    Logs requests and responses as JSONL entries.

    Each log entry contains:
    - timestamp, request_id, model, backend
    - input/output token counts
    - latency_ms
    - routing decision metadata
    """

    def __init__(self, log_dir: str = "/var/log/ai-inference", enabled: bool = True):
        self._enabled = enabled
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "audit.jsonl"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        # Record request start time in context
        context["audit_start_time"] = time.time()
        return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        start_time = context.get("audit_start_time", time.time())
        latency_ms = round((time.time() - start_time) * 1000, 2)

        route_decision = context.get("route_decision")
        usage = response.get("usage", {})

        entry = {
            "timestamp": time.time(),
            "request_id": context.get("request_id", ""),
            "model": response.get("model", ""),
            "backend": route_decision.backend if route_decision else "",
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "latency_ms": latency_ms,
            "specialization": (
                route_decision.specialization.value
                if route_decision and route_decision.specialization
                else ""
            ),
            "reason": route_decision.reason if route_decision else "",
        }

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.debug(f"Audit log write failed: {e}")

        return response

    async def on_startup(self, app) -> None:
        logger.info(f"AuditLogPlugin initialized → {self.log_path}")
