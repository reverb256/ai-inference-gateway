"""
Prometheus metrics for Knowledge Fabric observability.

Provides comprehensive instrumentation for:
- Circuit breaker state transitions
- Query routing and classification
- Source retrieval latency and success rates
- Fusion operations
- System-wide context generation
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Generator
from datetime import datetime

logger = logging.getLogger(__name__)

# Gracefully handle missing prometheus_client
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Enum,
        Info,
        CollectorRegistry,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Metrics will be no-ops. "
        "Install with: pip install prometheus-client"
    )

# Metric namespace
NAMESPACE = "knowledge_fabric"


class NoOpMetric:
    """No-op metric for when prometheus_client is not available."""

    def __init__(self, *args, **kwargs):
        pass

    def labels(self, **kwargs):
        return self

    def inc(self, amount=1):
        return self

    def dec(self, amount=1):
        return self

    def set(self, value):
        return self

    def observe(self, amount):
        return self

    def state(self, state):
        return self

    def info(self, info):
        return self

    def remove(self, *args, **kwargs):
        return self

    def clear(self):
        return self


# Select metric class based on availability
_MetricClass = Counter if PROMETHEUS_AVAILABLE else NoOpMetric
_GaugeClass = Gauge if PROMETHEUS_AVAILABLE else NoOpMetric
_HistogramClass = Histogram if PROMETHEUS_AVAILABLE else NoOpMetric
_EnumClass = Enum if PROMETHEUS_AVAILABLE else NoOpMetric
_InfoClass = Info if PROMETHEUS_AVAILABLE else NoOpMetric


@dataclass
class KnowledgeFabricMetrics:
    """
    Prometheus metrics for Knowledge Fabric operations.

    All metrics use the 'knowledge_fabric_' prefix to avoid collisions.
    Metrics become no-ops if prometheus_client is not installed.

    Args:
        registry: Optional CollectorRegistry for registering metrics.
                    Uses default REGISTRY if not provided.
                    Tests should pass a custom CollectorRegistry.
    """

    # Registry for metrics (for testing, use a custom CollectorRegistry)
    registry: Any = None

    # All metric fields - initialized in __post_init__
    circuit_breaker_state: Any = field(init=False)
    circuit_breaker_state_changes: Any = field(init=False)
    circuit_breaker_rejected_calls: Any = field(init=False)
    router_classification_total: Any = field(init=False)
    router_sources_selected: Any = field(init=False)
    source_retrieval_latency: Any = field(init=False)
    source_retrieval_total: Any = field(init=False)
    source_chunks_retrieved: Any = field(init=False)
    fusion_operations_total: Any = field(init=False)
    fusion_chunks_before: Any = field(init=False)
    fusion_chunks_after: Any = field(init=False)
    fusion_latency: Any = field(init=False)
    reranking_operations_total: Any = field(init=False)
    reranking_latency: Any = field(init=False)
    query_total: Any = field(init=False)
    query_latency: Any = field(init=False)
    query_skipped_total: Any = field(init=False)
    context_generated_total: Any = field(init=False)
    context_chars: Any = field(init=False)
    context_sources_used: Any = field(init=False)
    active_sources: Any = field(init=False)

    def __post_init__(self):
        """Initialize all metrics with the appropriate registry."""
        registry = self.registry if self.registry is not None else (REGISTRY if PROMETHEUS_AVAILABLE else None)

        if PROMETHEUS_AVAILABLE:
            # Circuit breaker metrics
            self.circuit_breaker_state = _EnumClass(
                f"{NAMESPACE}_circuit_breaker_state",
                "Circuit breaker state for each knowledge source",
                ["source_name", "state"],
                states=["closed", "open", "half_open"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.circuit_breaker_state_changes = _MetricClass(
                f"{NAMESPACE}_circuit_breaker_state_changes_total",
                "Total circuit breaker state transitions",
                ["source_name", "from_state", "to_state"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.circuit_breaker_rejected_calls = _MetricClass(
                f"{NAMESPACE}_circuit_breaker_rejected_calls_total",
                "Calls rejected due to open circuit",
                ["source_name"],
                namespace=NAMESPACE,
                registry=registry,
            )

            # Router metrics
            self.router_classification_total = _MetricClass(
                f"{NAMESPACE}_router_classification_total",
                "Total query classifications",
                ["intent", "confidence_tier"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.router_sources_selected = _HistogramClass(
                f"{NAMESPACE}_router_sources_selected",
                "Number of sources selected per query",
                namespace=NAMESPACE,
                buckets=[0, 1, 2, 3, 4, 5],
                registry=registry,
            )

            # Source retrieval metrics
            self.source_retrieval_latency = _HistogramClass(
                f"{NAMESPACE}_source_retrieval_latency_seconds",
                "Source retrieval latency in seconds",
                ["source_name", "status"],
                namespace=NAMESPACE,
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=registry,
            )
            self.source_retrieval_total = _MetricClass(
                f"{NAMESPACE}_source_retrieval_total",
                "Total source retrieval attempts",
                ["source_name", "status"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.source_chunks_retrieved = _HistogramClass(
                f"{NAMESPACE}_source_chunks_retrieved",
                "Number of chunks retrieved per source",
                ["source_name"],
                namespace=NAMESPACE,
                buckets=[0, 1, 2, 5, 10, 20, 50, 100],
                registry=registry,
            )

            # Fusion metrics
            self.fusion_operations_total = _MetricClass(
                f"{NAMESPACE}_fusion_operations_total",
                "Total fusion operations",
                ["status"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.fusion_chunks_before = _HistogramClass(
                f"{NAMESPACE}_fusion_chunks_before",
                "Number of chunks before fusion",
                namespace=NAMESPACE,
                buckets=[0, 1, 2, 5, 10, 20, 50, 100, 200],
                registry=registry,
            )
            self.fusion_chunks_after = _HistogramClass(
                f"{NAMESPACE}_fusion_chunks_after",
                "Number of chunks after fusion",
                namespace=NAMESPACE,
                buckets=[0, 1, 2, 5, 10, 20, 50, 100, 200],
                registry=registry,
            )
            self.fusion_latency = _HistogramClass(
                f"{NAMESPACE}_fusion_latency_seconds",
                "Fusion operation latency in seconds",
                namespace=NAMESPACE,
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                registry=registry,
            )

            # Reranking metrics
            self.reranking_operations_total = _MetricClass(
                f"{NAMESPACE}_reranking_operations_total",
                "Total reranking operations",
                ["status"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.reranking_latency = _HistogramClass(
                f"{NAMESPACE}_reranking_latency_seconds",
                "Reranking latency in seconds",
                namespace=NAMESPACE,
                buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
                registry=registry,
            )

            # Query metrics
            self.query_total = _MetricClass(
                f"{NAMESPACE}_query_total",
                "Total queries processed",
                ["status"],
                namespace=NAMESPACE,
                registry=registry,
            )
            self.query_latency = _HistogramClass(
                f"{NAMESPACE}_query_latency_seconds",
                "End-to-end query latency",
                namespace=NAMESPACE,
                buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                registry=registry,
            )
            self.query_skipped_total = _MetricClass(
                f"{NAMESPACE}_query_skipped_total",
                "Total queries skipped (too short, etc.)",
                ["reason"],
                namespace=NAMESPACE,
                registry=registry,
            )

            # Context generation metrics
            self.context_generated_total = _MetricClass(
                f"{NAMESPACE}_context_generated_total",
                "Total knowledge contexts generated",
                namespace=NAMESPACE,
                registry=registry,
            )
            self.context_chars = _HistogramClass(
                f"{NAMESPACE}_context_chars",
                "Character count of generated contexts",
                namespace=NAMESPACE,
                buckets=[100, 500, 1000, 2500, 5000, 10000, 25000, 50000],
                registry=registry,
            )
            self.context_sources_used = _HistogramClass(
                f"{NAMESPACE}_context_sources_used",
                "Number of sources used in context",
                namespace=NAMESPACE,
                buckets=[0, 1, 2, 3, 4, 5],
                registry=registry,
            )

            # Active sources gauge
            self.active_sources = _GaugeClass(
                f"{NAMESPACE}_active_sources",
                "Number of active knowledge sources",
                namespace=NAMESPACE,
                registry=registry,
            )
        else:
            # No-op metrics when prometheus_client is not available
            self.circuit_breaker_state = NoOpMetric()
            self.circuit_breaker_state_changes = NoOpMetric()
            self.circuit_breaker_rejected_calls = NoOpMetric()
            self.router_classification_total = NoOpMetric()
            self.router_sources_selected = NoOpMetric()
            self.source_retrieval_latency = NoOpMetric()
            self.source_retrieval_total = NoOpMetric()
            self.source_chunks_retrieved = NoOpMetric()
            self.fusion_operations_total = NoOpMetric()
            self.fusion_chunks_before = NoOpMetric()
            self.fusion_chunks_after = NoOpMetric()
            self.fusion_latency = NoOpMetric()
            self.reranking_operations_total = NoOpMetric()
            self.reranking_latency = NoOpMetric()
            self.query_total = NoOpMetric()
            self.query_latency = NoOpMetric()
            self.query_skipped_total = NoOpMetric()
            self.context_generated_total = NoOpMetric()
            self.context_chars = NoOpMetric()
            self.context_sources_used = NoOpMetric()
            self.active_sources = NoOpMetric()

    def set_active_sources(self, count: int):
        """Update the active sources gauge."""
        self.active_sources.set(count)

    def record_circuit_state_change(
        self,
        source_name: str,
        from_state: str,
        to_state: str
    ):
        """Record a circuit breaker state transition."""
        self.circuit_breaker_state.labels(
            source_name=source_name,
            state=to_state
        ).state(to_state)

        self.circuit_breaker_state_changes.labels(
            source_name=source_name,
            from_state=from_state,
            to_state=to_state
        ).inc()

    def record_rejected_call(self, source_name: str):
        """Record a rejected call due to open circuit."""
        self.circuit_breaker_rejected_calls.labels(
            source_name=source_name
        ).inc()

    def record_classification(self, intent: str, confidence: float):
        """Record a query classification."""
        # Tier confidence by ranges
        if confidence >= 0.8:
            tier = "high"
        elif confidence >= 0.5:
            tier = "medium"
        else:
            tier = "low"

        self.router_classification_total.labels(
            intent=intent,
            confidence_tier=tier
        ).inc()

    def record_sources_selected(self, count: int):
        """Record number of sources selected."""
        self.router_sources_selected.observe(count)

    def record_source_retrieval(
        self,
        source_name: str,
        status: str,
        latency_seconds: float,
        chunk_count: int
    ):
        """Record a source retrieval attempt."""
        self.source_retrieval_latency.labels(
            source_name=source_name,
            status=status
        ).observe(latency_seconds)

        self.source_retrieval_total.labels(
            source_name=source_name,
            status=status
        ).inc()

        if status == "success":
            self.source_chunks_retrieved.labels(
                source_name=source_name
            ).observe(chunk_count)

    def record_fusion_operation(
        self,
        status: str,
        chunks_before: int,
        chunks_after: int,
        latency_seconds: float
    ):
        """Record a fusion operation."""
        self.fusion_operations_total.labels(status=status).inc()
        self.fusion_chunks_before.observe(chunks_before)
        self.fusion_chunks_after.observe(chunks_after)
        self.fusion_latency.observe(latency_seconds)

    def record_reranking(self, status: str, latency_seconds: float):
        """Record a reranking operation."""
        self.reranking_operations_total.labels(status=status).inc()
        self.reranking_latency.observe(latency_seconds)

    def record_query(self, status: str, latency_seconds: float):
        """Record a completed query."""
        self.query_total.labels(status=status).inc()
        self.query_latency.observe(latency_seconds)

    def record_query_skipped(self, reason: str):
        """Record a skipped query."""
        self.query_skipped_total.labels(reason=reason).inc()

    def record_context_generation(self, char_count: int, sources_used: int):
        """Record context generation."""
        self.context_generated_total.inc()
        self.context_chars.observe(char_count)
        self.context_sources_used.observe(sources_used)

    @contextmanager
    def time_source_retrieval(
        self,
        source_name: str
    ) -> Generator["_SourceLatencyContext", None, None]:
        """
        Context manager for timing source retrieval.

        Usage:
            with metrics.time_source_retrieval("rag") as ctx:
                result = await source.retrieve(query)
                ctx.complete(success=True, chunks=len(result.chunks))
        """
        ctx = _SourceLatencyContext(self, source_name, time.time())
        yield ctx
        # __exit__ handles completion automatically

    @contextmanager
    def time_query(self) -> Generator["_QueryLatencyContext", None, None]:
        """
        Context manager for timing full queries.

        Usage:
            with metrics.time_query() as ctx:
                # ... query processing ...
                ctx.complete(status="success")
        """
        ctx = _QueryLatencyContext(self, time.time())
        yield ctx
        # __exit__ handles completion automatically

    @contextmanager
    def time_fusion(self) -> Generator["_FusionLatencyContext", None, None]:
        """
        Context manager for timing fusion operations.

        Usage:
            with metrics.time_fusion() as ctx:
                result = fuse(results, context)
                ctx.complete(chunks_before=10, chunks_after=5)
        """
        ctx = _FusionLatencyContext(self, time.time())
        yield ctx
        # __exit__ handles completion automatically

    @contextmanager
    def time_reranking(self) -> Generator["_RerankingLatencyContext", None, None]:
        """
        Context manager for timing reranking operations.

        Usage:
            with metrics.time_reranking() as ctx:
                result = reranker.rank(chunks)
                ctx.complete(status="success")
        """
        ctx = _RerankingLatencyContext(self, time.time())
        yield ctx
        # __exit__ handles completion automatically


@dataclass
class _SourceLatencyContext:
    """Context for tracking source retrieval latency."""
    metrics: KnowledgeFabricMetrics
    source_name: str
    start_time: float
    _completed: bool = False

    def complete(
        self,
        status: str = "success",
        chunks: int = 0,
        error: Optional[Exception] = None
    ):
        """Complete the context and record metrics."""
        if self._completed:
            return

        latency = time.time() - self.start_time

        if error:
            status = "error"

        self.metrics.record_source_retrieval(
            source_name=self.source_name,
            status=status,
            latency_seconds=latency,
            chunk_count=chunks
        )

        self._completed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            if exc_type is not None:
                self.complete(status="error")
            else:
                self.complete(status="success")
        return False


@dataclass
class _QueryLatencyContext:
    """Context for tracking full query latency."""
    metrics: KnowledgeFabricMetrics
    start_time: float
    _completed: bool = False

    def complete(self, status: str = "success"):
        """Complete the context and record metrics."""
        if self._completed:
            return

        latency = time.time() - self.start_time
        self.metrics.record_query(status=status, latency_seconds=latency)
        self._completed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            self.complete(status="error" if exc_type else "success")
        return False


@dataclass
class _FusionLatencyContext:
    """Context for tracking fusion operation latency."""
    metrics: KnowledgeFabricMetrics
    start_time: float
    _completed: bool = False

    def complete(self, status: str = "success", chunks_before: int = 0, chunks_after: int = 0):
        """Complete the context and record metrics."""
        if self._completed:
            return

        latency = time.time() - self.start_time
        self.metrics.record_fusion_operation(
            status=status,
            chunks_before=chunks_before,
            chunks_after=chunks_after,
            latency_seconds=latency
        )
        self._completed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            self.complete(status="error" if exc_type else "success")
        return False


@dataclass
class _RerankingLatencyContext:
    """Context for tracking reranking latency."""
    metrics: KnowledgeFabricMetrics
    start_time: float
    _completed: bool = False

    def complete(self, status: str = "success"):
        """Complete the context and record metrics."""
        if self._completed:
            return

        latency = time.time() - self.start_time
        self.metrics.record_reranking(status=status, latency_seconds=latency)
        self._completed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._completed:
            self.complete(status="error" if exc_type else "success")
        return False


# Global metrics instance
_metrics_instance: Optional[KnowledgeFabricMetrics] = None
_metrics_registry: Any = None  # Isolated registry for singleton metrics


def create_metrics(registry: Any = None) -> KnowledgeFabricMetrics:
    """
    Create a new KnowledgeFabricMetrics instance.

    Args:
        registry: Optional CollectorRegistry for registering metrics.
                    Uses default REGISTRY if not provided.
                    Tests should pass a custom CollectorRegistry.

    Returns:
        KnowledgeFabricMetrics instance
    """
    return KnowledgeFabricMetrics(registry=registry)


def get_metrics() -> KnowledgeFabricMetrics:
    """
    Get the global metrics instance, creating it if necessary.

    Returns:
        KnowledgeFabricMetrics singleton instance
    """
    global _metrics_instance, _metrics_registry

    if _metrics_instance is None:
        # Create isolated registry for singleton metrics
        # This prevents conflicts with default REGISTRY used in tests
        if PROMETHEUS_AVAILABLE:
            _metrics_registry = CollectorRegistry()
            _metrics_instance = create_metrics(registry=_metrics_registry)
        else:
            _metrics_instance = create_metrics()

    return _metrics_instance
