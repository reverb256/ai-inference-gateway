"""
Privacy Filter ML Service Client.

HTTP client for the OpenAI Privacy Filter NER model deployed as a K8s service.
Calls POST /redact with text payloads and returns redacted results.

Features:
- Circuit breaker with automatic fallback to regex-only redaction
- Async via run_in_executor on stdlib urllib
- Batch processing support
- Connection pooling awareness via singleton pattern
"""

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from ai_inference_gateway.config import PrivacyFilterConfig

logger = logging.getLogger(__name__)


class CircuitState:
    """Simplified circuit breaker state for the privacy filter client."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.state = self.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = 0.0

    def record_success(self) -> None:
        """Record a successful call, resetting the circuit."""
        if self.state == self.HALF_OPEN:
            logger.info("Privacy filter circuit breaker recovered, closing circuit")
        self.failure_count = 0
        self.state = self.CLOSED

    def record_failure(self) -> None:
        """Record a failed call, potentially opening the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.monotonic()

        if self.state == self.HALF_OPEN:
            logger.warning("Privacy filter circuit breaker reopening after half-open failure")
            self.state = self.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(
                "Privacy filter circuit breaker opening after %d failures",
                self.failure_count,
            )
            self.state = self.OPEN

    def allow_request(self) -> bool:
        """Check if a request should be attempted."""
        if self.state == self.CLOSED:
            return True

        if self.state == self.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.info("Privacy filter circuit breaker transitioning to half-open")
                self.state = self.HALF_OPEN
                return True
            return False

        # HALF_OPEN: allow one probe request
        return True


class PrivacyFilterClient:
    """
    HTTP client for the Privacy Filter ML service.

    Uses stdlib urllib.request wrapped in asyncio executor. Falls back to
    regex-only PIIRedactor when the ML service is unavailable.

    Usage:
        client = PrivacyFilterClient(config)
        redacted = await client.redact("some text with PII")
    """

    _instance: Optional["PrivacyFilterClient"] = None

    def __init__(self, config: PrivacyFilterConfig):
        """
        Initialize the privacy filter client.

        Args:
            config: Privacy filter service configuration.
        """
        self.config = config
        self._base_url = config.url.rstrip("/")
        self._circuit = CircuitState()
        self._redaction_count = 0

        logger.info(
            "PrivacyFilterClient initialized: url=%s, timeout=%.1fs, mode=%s",
            self._base_url,
            config.timeout,
            config.mode,
        )

    @classmethod
    def get_instance(cls, config: PrivacyFilterConfig) -> "PrivacyFilterClient":
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def _make_request(self, text: str, mode: str) -> Dict[str, Any]:
        """
        Synchronous HTTP POST to the privacy filter service.

        Args:
            text: Text to redact.
            mode: Redaction mode (redact, detect, hash).

        Returns:
            Response dict with 'redacted_text' and 'entities_found'.

        Raises:
            urllib.error.URLError: On connection failure.
            urllib.error.HTTPError: On HTTP error responses.
        """
        payload = json.dumps({"text": text, "mode": mode}).encode("utf-8")
        url = f"{self._base_url}/redact"

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout = self.config.timeout
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)

    async def redact(self, text: str, mode: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Redact PII from text using the ML privacy filter service.

        Args:
            text: Input text to redact.
            mode: Override redaction mode. Defaults to config mode.

        Returns:
            Tuple of (redacted_text, entities_found).
            On service failure, returns (original_text, []) to let caller
            fall back to regex redaction.
        """
        if not text:
            return text, []

        if not self._circuit.allow_request():
            logger.debug("Privacy filter circuit open, skipping ML redaction")
            return text, []

        effective_mode = mode or self.config.mode

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self._make_request, text, effective_mode)

            self._circuit.record_success()

            redacted_text = result.get("redacted_text", text)
            entities = result.get("entities_found", [])

            if redacted_text != text:
                self._redaction_count += 1
                logger.info(
                    "Privacy filter redacted %d entities (total: %d)",
                    len(entities),
                    self._redaction_count,
                )

            return redacted_text, entities

        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            self._circuit.record_failure()
            logger.warning(
                "Privacy filter service unavailable, falling back to regex: %s",
                exc,
            )
            return text, []
        except json.JSONDecodeError as exc:
            self._circuit.record_failure()
            logger.warning("Privacy filter returned invalid JSON: %s", exc)
            return text, []
        except Exception as exc:
            self._circuit.record_failure()
            logger.warning("Privacy filter unexpected error: %s", exc)
            return text, []

    async def redact_batch(
        self, texts: List[str], mode: Optional[str] = None
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """
        Redact PII from a batch of texts.

        Processes texts concurrently using asyncio.gather.

        Args:
            texts: List of texts to redact.
            mode: Override redaction mode.

        Returns:
            List of (redacted_text, entities_found) tuples, one per input.
        """
        if not texts:
            return []

        tasks = [self.redact(text, mode) for text in texts]
        return await asyncio.gather(*tasks)

    @property
    def is_available(self) -> bool:
        """Check if the ML service is likely available (circuit not open)."""
        return self._circuit.state != CircuitState.OPEN

    @property
    def circuit_state(self) -> str:
        """Get current circuit breaker state."""
        return self._circuit.state

    @property
    def redaction_count(self) -> int:
        """Total number of redactions performed."""
        return self._redaction_count
