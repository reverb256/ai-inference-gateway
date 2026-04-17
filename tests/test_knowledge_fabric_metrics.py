"""
Tests for Knowledge Fabric Prometheus metrics.

Tests that the metrics module correctly tracks:
- Circuit breaker state transitions
- Source retrieval latency and success/failure
- Router classification and confidence
- Fusion operations
- System-level query metrics
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Try importing prometheus_client, skip tests if not available
pytest.importorskip("prometheus_client")

from prometheus_client import CollectorRegistry

from ai_inference_gateway.middleware.knowledge_fabric.metrics import (
    KnowledgeFabricMetrics,
    create_metrics,
    get_metrics,
)
from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent
from ai_inference_gateway.middleware.knowledge_fabric.core import (
    KnowledgeChunk,
    KnowledgeResult,
    FabricContext,
)


class TestKnowledgeFabricMetrics:
    """Test KnowledgeFabricMetrics class."""

    def test_metrics_initialization(self):
        """Test that metrics initializes with all expected metric types."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        # Check circuit breaker metrics exist
        assert hasattr(metrics, 'circuit_breaker_state')
        assert hasattr(metrics, 'circuit_breaker_state_changes')
        assert hasattr(metrics, 'circuit_breaker_rejected_calls')

        # Check source metrics exist
        assert hasattr(metrics, 'source_retrieval_latency')
        assert hasattr(metrics, 'source_retrieval_total')
        assert hasattr(metrics, 'source_chunks_retrieved')

        # Check router metrics exist
        assert hasattr(metrics, 'router_classification_total')
        assert hasattr(metrics, 'router_sources_selected')

        # Check fusion metrics exist
        assert hasattr(metrics, 'fusion_operations_total')
        assert hasattr(metrics, 'fusion_chunks_before')
        assert hasattr(metrics, 'fusion_chunks_after')
        assert hasattr(metrics, 'fusion_latency')

        # Check query metrics exist
        assert hasattr(metrics, 'query_total')
        assert hasattr(metrics, 'query_latency')

        # Check context metrics exist
        assert hasattr(metrics, 'context_generated_total')
        assert hasattr(metrics, 'context_chars')
        assert hasattr(metrics, 'context_sources_used')

    def test_circuit_state_change(self):
        """Test recording circuit breaker state changes."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        # Record state transition - should not raise
        metrics.record_circuit_state_change(
            source_name="test_source",
            from_state="closed",
            to_state="open"
        )

    def test_record_classification(self):
        """Test recording router classification."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        # Record high confidence classification
        metrics.record_classification(intent="CODE", confidence=0.95)

        # Record medium confidence
        metrics.record_classification(intent="FACTUAL", confidence=0.6)

        # Record low confidence
        metrics.record_classification(intent="PROCEDURAL", confidence=0.3)

    def test_record_sources_selected(self):
        """Test recording number of sources selected."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)
        metrics.record_sources_selected(3)

    def test_record_source_retrieval(self):
        """Test recording source retrieval metrics."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        # Record success
        metrics.record_source_retrieval(
            source_name="test_source",
            status="success",
            latency_seconds=0.5,
            chunk_count=5
        )

        # Record error
        metrics.record_source_retrieval(
            source_name="test_source",
            status="error",
            latency_seconds=1.0,
            chunk_count=0
        )

    def test_fusion_operation(self):
        """Test recording fusion metrics."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        metrics.record_fusion_operation(
            status="success",
            chunks_before=10,
            chunks_after=7,
            latency_seconds=0.2
        )

    def test_reranking(self):
        """Test recording reranking metrics."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        metrics.record_reranking(
            status="success",
            latency_seconds=0.15
        )

    def test_query_metrics(self):
        """Test query-level metrics."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        metrics.record_query(status="success", latency_seconds=1.5)
        metrics.record_query_skipped(reason="too_short")

    def test_context_generation(self):
        """Test context generation metrics."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        metrics.record_context_generation(
            char_count=1500,
            sources_used=3
        )

    def test_set_active_sources(self):
        """Test setting active sources gauge."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)
        metrics.set_active_sources(5)

    def test_record_rejected_call(self):
        """Test recording rejected call due to open circuit."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)
        metrics.record_rejected_call(source_name="test_source")


@pytest.mark.asyncio
class TestSourceLatencyContext:
    """Test source retrieval latency context manager."""

    async def test_successful_retrieval(self):
        """Test latency recording on successful retrieval."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_source_retrieval("test_source") as ctx:
            await asyncio.sleep(0.01)
            ctx.complete(status="success", chunks=5)

    async def test_failed_retrieval(self):
        """Test latency recording on failed retrieval."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_source_retrieval("test_source") as ctx:
            await asyncio.sleep(0.01)
            ctx.complete(status="error", chunks=0)


@pytest.mark.asyncio
class TestQueryLatencyContext:
    """Test query latency context manager."""

    async def test_full_query(self):
        """Test end-to-end query timing."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_query() as ctx:
            await asyncio.sleep(0.01)
            ctx.complete(status="success")

    async def test_query_with_exception(self):
        """Test query timing when exception occurs."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_query() as ctx:
            await asyncio.sleep(0.01)
            # Simulate exception
            try:
                raise ValueError("Test error")
            except ValueError:
                pass
        # Should record as error automatically


@pytest.mark.asyncio
class TestFusionLatencyContext:
    """Test fusion latency context manager."""

    async def test_fusion_operation(self):
        """Test fusion timing context."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_fusion() as ctx:
            await asyncio.sleep(0.01)
            ctx.complete(
                status="success",
                chunks_before=15,
                chunks_after=8
            )


@pytest.mark.asyncio
class TestRerankingLatencyContext:
    """Test reranking latency context manager."""

    async def test_reranking(self):
        """Test reranking timing context."""
        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)

        with metrics.time_reranking() as ctx:
            await asyncio.sleep(0.01)
            ctx.complete(status="success")


class TestMetricsFactory:
    """Test factory functions and global getter."""

    def test_create_metrics(self):
        """Test create_metrics factory function."""
        registry = CollectorRegistry()
        metrics = create_metrics()
        assert isinstance(metrics, KnowledgeFabricMetrics)
        # Also test with custom registry
        metrics_with_registry = KnowledgeFabricMetrics(registry=registry)
        assert isinstance(metrics_with_registry, KnowledgeFabricMetrics)

    def test_get_metrics_returns_singleton(self):
        """Test that get_metrics returns a singleton instance."""
        # Clear any existing singleton by resetting to None
        # Note: We set to None rather than using delattr() because deleting
        # the module attribute breaks the 'global _metrics_instance' declaration
        # in get_metrics(), causing NameError.
        import ai_inference_gateway.middleware.knowledge_fabric.metrics as metrics_module
        if hasattr(metrics_module, '_metrics_instance'):
            metrics_module._metrics_instance = None

        # First call creates instance
        metrics1 = get_metrics()
        assert isinstance(metrics1, KnowledgeFabricMetrics)

        # Second call returns same instance
        metrics2 = get_metrics()
        assert metrics1 is metrics2


@pytest.mark.asyncio
class TestMetricsIntegration:
    """Integration tests for metrics with Knowledge Fabric components."""

    async def test_circuit_breaker_with_metrics(self):
        """Test circuit breaker records metrics through state transitions."""
        from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        registry = CollectorRegistry()
        metrics = KnowledgeFabricMetrics(registry=registry)
        breaker = CircuitBreaker(
            source_name="test_source",
            config=CircuitBreakerConfig(failure_threshold=2),
            metrics=metrics
        )

        # Simulate failures to trigger state change
        breaker.record_failure(Exception("Test 1"))
        breaker.record_failure(Exception("Test 2"))

        # Should be OPEN now
        assert breaker.state.state == CircuitState.OPEN

    async def test_fabric_initializes_metrics(self):
        """Test KnowledgeFabricMiddleware initializes metrics."""
        from ai_inference_gateway.middleware.knowledge_fabric.fabric import (
            KnowledgeFabricMiddleware,
        )

        middleware = KnowledgeFabricMiddleware(enabled=True)

        # Metrics should be initialized
        assert middleware.metrics is not None
        assert isinstance(middleware.metrics, KnowledgeFabricMetrics)

        # Circuit registry should have metrics
        assert middleware.circuit_registry is not None
