"""
Circuit Breaker pattern for Knowledge Fabric source resilience.

Prevents cascading failures by temporarily skipping sources that
experience repeated failures, while allowing automatic recovery
after a cooldown period.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Optional, Callable, Any, TYPE_CHECKING
from collections import defaultdict

logger = logging.getLogger(__name__)

# Type hint for metrics to avoid circular import
if TYPE_CHECKING:
    from .metrics import KnowledgeFabricMetrics


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Source failing, block requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    # Thresholds
    failure_threshold: int = 5  # Consecutive failures to open circuit
    success_threshold: int = 2  # Consecutive successes to close circuit

    # Timing
    timeout: float = 60.0  # Seconds to stay open before half-open
    half_open_max_calls: int = 1  # Max calls in half-open state

    # Monitoring
    call_timeout: float = 30.0  # Max seconds to wait for source response

    # State tracking (not configured, but tracked)
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None


@dataclass
class CircuitBreakerState:
    """Runtime state of a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_state_change: datetime = field(default_factory=datetime.now)

    # Metrics
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls blocked due to open circuit


class CircuitBreaker:
    """
    Circuit breaker for a single knowledge source.

    Tracks failures and automatically opens/closes the circuit
    to prevent cascading failures while allowing recovery.
    """

    def __init__(
        self,
        source_name: str,
        config: Optional[CircuitBreakerConfig] = None,
        metrics: Optional["KnowledgeFabricMetrics"] = None
    ):
        """
        Initialize circuit breaker for a source.

        Args:
            source_name: Name of the protected source
            config: Circuit breaker configuration
            metrics: Optional metrics instance for recording state changes
        """
        self.source_name = source_name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState()
        self._metrics = metrics

    def is_call_allowed(self) -> bool:
        """
        Check if a call should be allowed through the circuit.

        Returns:
            True if call allowed, False if circuit is open
        """
        self.state.total_calls += 1

        if self.state.state == CircuitState.OPEN:
            # Check if timeout has passed for half-open attempt
            if self._should_attempt_half_open():
                self._transition_to_half_open()
                return True
            else:
                self.state.rejected_calls += 1
                logger.debug(
                    f"Circuit OPEN for {self.source_name}, rejecting call "
                    f"(open for {(datetime.now() - self.state.last_state_change).total_seconds():.0f}s)"
                )
                return False

        return True

    def record_success(self):
        """Record a successful call from the source."""
        self.state.successful_calls += 1

        if self.state.state == CircuitState.HALF_OPEN:
            self.state.success_count += 1
            logger.info(
                f"Success in HALF_OPEN for {self.source_name} "
                f"({self.state.success_count}/{self.config.success_threshold})"
            )

            if self.state.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state.state == CircuitState.CLOSED:
            # Reset failure count on successful call in closed state
            self.state.failure_count = 0
            self.state.last_success_time = datetime.now()

    def record_failure(self, error: Exception):
        """Record a failed call from the source."""
        self.state.failed_calls += 1

        if self.state.state == CircuitState.CLOSED:
            self.state.failure_count += 1
            logger.warning(
                f"Failure in CLOSED for {self.source_name} "
                f"({self.state.failure_count}/{self.config.failure_threshold})"
            )

            if self.state.failure_count >= self.config.failure_threshold:
                self._transition_to_open(error)

        elif self.state.state == CircuitState.HALF_OPEN:
            # Failed in half-open, go back to open
            logger.warning(
                f"Failure in HALF_OPEN for {self.source_name}, returning to OPEN"
            )
            self._transition_to_open(error)

        self.state.last_failure_time = datetime.now()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "source": self.source_name,
            "state": self.state.state.value,
            "failure_count": self.state.failure_count,
            "success_count": self.state.success_count,
            "total_calls": self.state.total_calls,
            "successful_calls": self.state.successful_calls,
            "failed_calls": self.state.failed_calls,
            "rejected_calls": self.state.rejected_calls,
            "last_failure_time": self.state.last_failure_time.isoformat() if self.state.last_failure_time else None,
            "last_state_change": self.state.last_state_change.isoformat(),
            "success_rate": (
                self.state.successful_calls / self.state.total_calls
                if self.state.total_calls > 0 else 0.0
            ),
        }

    def _should_attempt_half_open(self) -> bool:
        """Check if enough time has passed to attempt half-open."""
        if self.state.last_state_change is None:
            return True

        elapsed = (datetime.now() - self.state.last_state_change).total_seconds()
        return elapsed >= self.config.timeout

    def _transition_to_open(self, error: Exception):
        """Transition circuit to OPEN state."""
        from_state = self.state.state.value
        self.state.state = CircuitState.OPEN
        self.state.last_state_change = datetime.now()
        
        # Record state change in metrics
        if self._metrics:
            self._metrics.record_circuit_state_change(
                self.source_name, from_state, CircuitState.OPEN.value
            )
        
        logger.error(
            f"Circuit OPENED for {self.source_name} after "
            f"{self.state.failure_count} failures. Last error: {error}"
        )

    def _transition_to_half_open(self):
        """Transition circuit to HALF_OPEN state."""
        from_state = self.state.state.value
        self.state.state = CircuitState.HALF_OPEN
        self.state.last_state_change = datetime.now()
        self.state.success_count = 0
        
        # Record state change in metrics
        if self._metrics:
            self._metrics.record_circuit_state_change(
                self.source_name, from_state, CircuitState.HALF_OPEN.value
            )
        
        logger.info(f"Circuit HALF_OPEN for {self.source_name}, attempting recovery")

    def _transition_to_closed(self):
        """Transition circuit to CLOSED (normal) state."""
        from_state = self.state.state.value
        self.state.state = CircuitState.CLOSED
        self.state.last_state_change = datetime.now()
        self.state.failure_count = 0
        
        # Record state change in metrics
        if self._metrics:
            self._metrics.record_circuit_state_change(
                self.source_name, from_state, CircuitState.CLOSED.value
            )
        
        logger.info(f"Circuit CLOSED for {self.source_name}, recovered")


class CircuitBreakerRegistry:
    """
    Registry managing circuit breakers for all knowledge sources.

    Provides circuit breaker protection for source calls and
    aggregates metrics for observability.
    """

    def __init__(
        self,
        default_config: Optional[CircuitBreakerConfig] = None,
        metrics: Optional["KnowledgeFabricMetrics"] = None
    ):
        """
        Initialize circuit breaker registry.

        Args:
            default_config: Default configuration for new breakers
            metrics: Optional metrics instance for observability
        """
        self._breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = default_config or CircuitBreakerConfig()
        self._metrics = metrics

    def get_breaker(
        self,
        source_name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker for a source."""
        if source_name not in self._breakers:
            breaker = CircuitBreaker(
                source_name=source_name,
                config=config or self.default_config,
                metrics=self._metrics
            )
            self._breakers[source_name] = breaker
            logger.debug(f"Created circuit breaker for {source_name}")

        return self._breakers[source_name]

    def is_call_allowed(self, source_name: str) -> bool:
        """Check if call is allowed through circuit breaker."""
        breaker = self.get_breaker(source_name)
        return breaker.is_call_allowed()

    def record_success(self, source_name: str):
        """Record successful call for source."""
        breaker = self.get_breaker(source_name)
        breaker.record_success()

    def record_failure(self, source_name: str, error: Exception):
        """Record failed call for source."""
        breaker = self.get_breaker(source_name)
        breaker.record_failure(error)

    def get_breaker_state(self, source_name: str) -> CircuitState:
        """Get current circuit state for a source."""
        if source_name in self._breakers:
            return self._breakers[source_name].state.state
        return CircuitState.CLOSED

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {
            name: breaker.get_metrics()
            for name, breaker in self._breakers.items()
        }

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics across all breakers."""
        metrics = list(self.get_all_metrics().values())

        if not metrics:
            return {
                "total_sources": 0,
                "open_circuits": 0,
                "half_open_circuits": 0,
                "closed_circuits": 0,
            }

        states = [m["state"] for m in metrics]

        return {
            "total_sources": len(metrics),
            "open_circuits": states.count(CircuitState.OPEN.value),
            "half_open_circuits": states.count(CircuitState.HALF_OPEN.value),
            "closed_circuits": states.count(CircuitState.CLOSED.value),
            "total_calls": sum(m["total_calls"] for m in metrics),
            "successful_calls": sum(m["successful_calls"] for m in metrics),
            "failed_calls": sum(m["failed_calls"] for m in metrics),
            "rejected_calls": sum(m["rejected_calls"] for m in metrics),
            "overall_success_rate": (
                sum(m["successful_calls"] for m in metrics) /
                sum(m["total_calls"] for m in metrics)
                if sum(m["total_calls"] for m in metrics) > 0 else 0.0
            ),
        }


async def execute_with_circuit_breaker(
    registry: CircuitBreakerRegistry,
    source_name: str,
    callable_func: Callable,
    *args,
    **kwargs
) -> Any:
    """
    Execute a source call with circuit breaker protection.

    Args:
        registry: Circuit breaker registry
        source_name: Name of the source being called
        callable_func: The async function to execute
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Result from callable_func

    Raises:
        Exception: If call fails and circuit allows propagation
    """
    # Check if call is allowed
    if not registry.is_call_allowed(source_name):
        logger.warning(
            f"Circuit breaker blocking call to {source_name}"
        )
        # Return empty KnowledgeResult to indicate blocking
        from .core import KnowledgeResult
        return KnowledgeResult(
            source_name=source_name,
            chunks=[],
            query=kwargs.get("query", ""),
            retrieval_time=0,
            metadata={
                "circuit_breaker": "blocked",
                "reason": f"Circuit open after {registry.get_breaker(source_name).state.failure_count} failures"
            }
        )

    # Execute with timeout protection
    try:
        # Use asyncio.wait_for to implement call timeout
        result = await asyncio.wait_for(
            callable_func(*args, **kwargs),
            timeout=registry.get_breaker(source_name).config.call_timeout
        )

        # Record success
        registry.record_success(source_name)
        return result

    except asyncio.TimeoutError:
        error = TimeoutError(f"Source {source_name} timed out")
        registry.record_failure(source_name, error)
        raise

    except Exception as e:
        registry.record_failure(source_name, e)
        raise


def create_circuit_breaker_registry(
    failure_threshold: int = 5,
    timeout: float = 60.0,
    success_threshold: int = 2,
    metrics: Optional["KnowledgeFabricMetrics"] = None
) -> CircuitBreakerRegistry:
    """Factory function to create circuit breaker registry."""
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        timeout=timeout,
        success_threshold=success_threshold
    )
    return CircuitBreakerRegistry(default_config=config, metrics=metrics)
