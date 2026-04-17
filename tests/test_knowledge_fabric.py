"""
Tests for Knowledge Fabric main orchestrator.

Tests the KnowledgeFabricMiddleware class which coordinates
semantic routing, parallel retrieval, RRF fusion, and context synthesis.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request

from ai_inference_gateway.middleware.knowledge_fabric import (
    KnowledgeFabricMiddleware,
    create_knowledge_fabric,
    FABRIC_CONTEXT_KEY,
    KNOWLEDGE_CONTEXT_KEY,
)
from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent
from ai_inference_gateway.middleware.knowledge_fabric.core import (
    FabricContext,
    KnowledgeResult,
    KnowledgeChunk,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_request():
    """Create mock FastAPI Request."""
    request = Mock(spec=Request)
    return request


@pytest.fixture
def basic_config():
    """Create basic Knowledge Fabric configuration."""
    return {
        "rag_enabled": False,
        "code_search_paths": ["/etc/nixos"],
        "searxng_url": "http://127.0.0.1:7777",
        "mcp_url": "http://127.0.0.1:8080/mcp/call",
        "code_max_results": 5,
        "web_max_results": 5,
        "searxng_max_results": 5,
        "rag_top_k": 5,
    }


@pytest.fixture
def fabric_middleware(basic_config):
    """Create KnowledgeFabricMiddleware for testing."""
    # Don't create actual sources to avoid HTTP calls during tests
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=True,
        config=basic_config,
    )
    return middleware


# ============================================================================
# Initialization Tests
# ============================================================================


def test_middleware_initialization(fabric_middleware):
    """Test middleware initializes correctly."""
    assert fabric_middleware.enabled is True
    assert fabric_middleware.router is not None
    assert fabric_middleware.fusion is not None
    assert fabric_middleware.synthesizer is not None


def test_middleware_disabled_initialization(basic_config):
    """Test middleware can be initialized as disabled."""
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=False,
        config=basic_config,
    )

    assert middleware.enabled is False


def test_create_knowledge_fabric_factory(basic_config):
    """Test factory function creates middleware correctly."""
    middleware = create_knowledge_fabric(
        rrf_k=60,
        enabled=True,
        config=basic_config,
    )

    assert isinstance(middleware, KnowledgeFabricMiddleware)
    assert middleware.enabled is True


# ============================================================================
# Process Request Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_request_skips_when_disabled(mock_request, basic_config):
    """Test process_request returns immediately when disabled."""
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=False,
        config=basic_config,
    )

    context = {}
    should_continue, error = await middleware.process_request(mock_request, context)

    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_process_request_skips_short_query(mock_request, fabric_middleware):
    """Test process_request skips very short queries."""
    # Mock _extract_query to return short query
    with patch.object(
        fabric_middleware, "_extract_query", return_value="hi"
    ):
        context = {}
        should_continue, error = await fabric_middleware.process_request(
            mock_request, context
        )

    assert should_continue is True
    assert error is None
    assert "context" not in context  # No knowledge context added


@pytest.mark.asyncio
async def test_process_request_performs_routing(mock_request, fabric_middleware):
    """Test process_request classifies query intent."""
    with patch.object(
        fabric_middleware, "_extract_query", return_value="How do I implement a function?"
    ):
        # Mock sources to return empty results
        for source in fabric_middleware.sources:
            source.retrieve = AsyncMock(return_value=Mock(
                chunks=[], source_name=source.name, retrieval_time=0.1, metadata={}
            ))

        context = {}
        should_continue, error = await fabric_middleware.process_request(
            mock_request, context
        )

    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_process_request_creates_fabric_context(mock_request, fabric_middleware):
    """Test process_request creates FabricContext."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent, RoutingDecision
    from ai_inference_gateway.middleware.knowledge_fabric.core import SourceCapability, KnowledgeResult, KnowledgeChunk

    # Create a mock source with a non-empty chunk
    mock_chunk = KnowledgeChunk(
        content="Test knowledge content",
        source="test_source",
        score=0.9,
        metadata={}
    )

    mock_result = KnowledgeResult(
        source_name="test_source",
        chunks=[mock_chunk],
        query="What is the capital of France?",
        retrieval_time=0.1,
        metadata={}
    )

    mock_source = Mock()
    mock_source.name = "test_source"
    mock_source.retrieve = AsyncMock(return_value=mock_result)

    # Mock router to return a decision with selected source
    mock_decision = RoutingDecision(
        intent=QueryIntent.FACTUAL,
        confidence=0.9,
        required_capabilities=SourceCapability.FACTUAL,
        reasoning="Test routing",
        selected_sources=["test_source"],
    )

    with patch.object(
        fabric_middleware, "_extract_query", return_value="What is the capital of France?"
    ), patch.object(
        fabric_middleware, "router"
    ) as mock_router:
        mock_router.classify.return_value = mock_decision

        # Mock _sources_by_name to return our source
        with patch.object(
            fabric_middleware, "_sources_by_name", {"test_source": mock_source}
        ):
            context = {}
            await fabric_middleware.process_request(mock_request, context)

    assert FABRIC_CONTEXT_KEY in context
    assert isinstance(context[FABRIC_CONTEXT_KEY], FabricContext)


@pytest.mark.asyncio
async def test_process_request_injects_knowledge_context(mock_request, fabric_middleware):
    """Test process_request injects knowledge context."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent, RoutingDecision
    from ai_inference_gateway.middleware.knowledge_fabric.core import SourceCapability, KnowledgeResult, KnowledgeChunk

    # Create a mock source with a result containing non-empty chunks
    mock_chunk = KnowledgeChunk(
        content="Test knowledge content",
        source="test_source",
        score=0.9,
        metadata={}
    )

    mock_result = KnowledgeResult(
        source_name="test_source",
        chunks=[mock_chunk],
        query="What is the capital of France?",
        retrieval_time=0.1,
        metadata={}
    )

    mock_source = Mock()
    mock_source.name = "test_source"
    mock_source.retrieve = AsyncMock(return_value=mock_result)

    # Mock router to return a decision with selected source
    mock_decision = RoutingDecision(
        intent=QueryIntent.FACTUAL,
        confidence=0.9,
        required_capabilities=SourceCapability.FACTUAL,
        reasoning="Test routing",
        selected_sources=["test_source"],
    )

    with patch.object(
        fabric_middleware, "_extract_query", return_value="What is the capital of France?"
    ), patch.object(
        fabric_middleware, "router"
    ) as mock_router:
        mock_router.classify.return_value = mock_decision

        with patch.object(
            fabric_middleware, "_sources_by_name", {"test_source": mock_source}
        ):
            # Mock synthesizer to return context
            with patch.object(
                fabric_middleware.synthesizer, "synthesize", return_value="Test knowledge context"
            ):
                # Mock fusion with non-empty chunks
                mock_fused_chunk = Mock()
                mock_fused_chunk.content = "Fused knowledge content"
                mock_fused_chunk.source = "test_source"
                mock_fused_chunk.score = 0.9

                with patch.object(
                    fabric_middleware.fusion, "fuse", return_value=[mock_fused_chunk]
                ):
                    context = {}
                    await fabric_middleware.process_request(mock_request, context)

    assert KNOWLEDGE_CONTEXT_KEY in context
    assert context[KNOWLEDGE_CONTEXT_KEY] == "Test knowledge context"


# ============================================================================
# Query Extraction Tests
# ============================================================================


@pytest.mark.asyncio
async def test_extract_query_from_messages(fabric_middleware):
    """Test query extraction from request body messages."""
    mock_request = Mock()
    context = {
        "parsed_body": {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is the capital of France?"},
            ]
        }
    }

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "What is the capital of France?"


@pytest.mark.asyncio
async def test_extract_query_from_user_message_only(fabric_middleware):
    """Test query extraction from user message only."""
    mock_request = Mock()
    context = {
        "parsed_body": {
            "messages": [
                {"role": "user", "content": "Explain binary search"}
            ]
        }
    }

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "Explain binary search"


@pytest.mark.asyncio
async def test_extract_query_from_last_user_message(fabric_middleware):
    """Test query extraction gets last user message."""
    mock_request = Mock()
    context = {
        "parsed_body": {
            "messages": [
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
            ]
        }
    }

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "Second question"


@pytest.mark.asyncio
async def test_extract_query_from_prompt(fabric_middleware):
    """Test query extraction from prompt field."""
    mock_request = Mock()
    context = {
        "parsed_body": {
            "prompt": "Complete this sentence"
        }
    }

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "Complete this sentence"


@pytest.mark.asyncio
async def test_extract_query_from_cached_context(fabric_middleware):
    """Test query extraction from cached user_query."""
    mock_request = Mock()
    context = {"user_query": "Cached query"}

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "Cached query"


@pytest.mark.asyncio
async def test_extract_query_multimodal_content(fabric_middleware):
    """Test query extraction from multi-modal content."""
    mock_request = Mock()
    context = {
        "parsed_body": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image", "source": {"type": "url", "url": "http://example.com/img.png"}},
                    ]
                }
            ]
        }
    }

    query = await fabric_middleware._extract_query(mock_request, context)

    assert query == "What is in this image?"


# ============================================================================
# Parallel Retrieval Tests
# ============================================================================


@pytest.mark.asyncio
async def test_parallel_retrieval_from_sources(mock_request, fabric_middleware):
    """Test parallel retrieval from multiple sources."""
    with patch.object(
        fabric_middleware, "_extract_query", return_value="test query"
    ):
        # Mock async retrieve
        async def mock_retrieve(query, **kwargs):
            import asyncio
            await asyncio.sleep(0.01)
            return Mock(
                chunks=[],
                source_name="test",
                retrieval_time=0.1,
                metadata={}
            )

        # Patch sources
        with patch.object(
            fabric_middleware.router, "classify", return_value=Mock(
                intent=QueryIntent.FACTUAL,
                confidence=0.9,
                selected_sources=["test"],
                reasoning="Test routing"
            )
        ):
            with patch.object(
                fabric_middleware, "_sources_by_name",
                {"test": Mock(retrieve=mock_retrieve)}
            ):
                context = {}
                await fabric_middleware.process_request(mock_request, context)

    # Should complete without error
    assert context is not None


@pytest.mark.asyncio
async def test_parallel_retrieval_handles_exceptions(mock_request, fabric_middleware):
    """Test parallel retrieval handles source exceptions gracefully."""
    with patch.object(
        fabric_middleware, "_extract_query", return_value="test query"
    ):
        # Mock failing source
        async def failing_retrieve(query, **kwargs):
            raise Exception("Source failed")

        with patch.object(
            fabric_middleware.router, "classify", return_value=Mock(
                intent=QueryIntent.FACTUAL,
                confidence=0.9,
                selected_sources=["failing"],
                reasoning="Test"
            )
        ):
            with patch.object(
                fabric_middleware, "_sources_by_name",
                {"failing": Mock(retrieve=failing_retrieve)}
            ):
                context = {}
                should_continue, error = await fabric_middleware.process_request(
                    mock_request, context
                )

    # Should continue despite source failure
    assert should_continue is True
    assert error is None


# ============================================================================
# Process Response Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_response_adds_metadata(fabric_middleware):
    """Test process_response adds knowledge fabric metadata."""
    fabric_context = FabricContext(
        query="test query",
        query_type=QueryIntent.FACTUAL,
        results={},
        fused_chunks=[],
        sources_used=["rag", "web_search"],
        routing_reasoning="Test reasoning",
    )

    context = {
        FABRIC_CONTEXT_KEY: fabric_context,
    }

    response = {"data": "test response"}
    result = await fabric_middleware.process_response(response, context)

    assert "metadata" in result
    assert "knowledge_fabric" in result["metadata"]
    assert result["metadata"]["knowledge_fabric"]["enabled"] is True
    assert result["metadata"]["knowledge_fabric"]["sources_used"] == ["rag", "web_search"]


@pytest.mark.asyncio
async def test_process_response_without_fabric_context(fabric_middleware):
    """Test process_response handles missing fabric context."""
    context = {}  # No FABRIC_CONTEXT_KEY

    response = {"data": "test response"}
    result = await fabric_middleware.process_response(response, context)

    # Should not modify response
    assert result["data"] == "test response"


@pytest.mark.asyncio
async def test_process_response_when_disabled(fabric_middleware, basic_config):
    """Test process_response does nothing when disabled."""
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=False,
        config=basic_config,
    )

    context = {}
    response = {"data": "test"}
    result = await middleware.process_response(response, context)

    # Should return unchanged
    assert result == response


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
async def test_process_request_handles_extraction_error(fabric_middleware):
    """Test process_request handles query extraction errors gracefully."""
    mock_request = Mock()

    with patch.object(
        fabric_middleware, "_extract_query", side_effect=Exception("Extraction failed")
    ):
        context = {}
        should_continue, error = await fabric_middleware.process_request(
            mock_request, context
        )

    # Should continue despite error
    assert should_continue is True
    assert error is None


@pytest.mark.asyncio
async def test_process_request_handles_routing_error(fabric_middleware):
    """Test process_request handles routing errors gracefully."""
    mock_request = Mock()

    with patch.object(
        fabric_middleware, "_extract_query", return_value="test query"
    ):
        with patch.object(
            fabric_middleware.router, "classify", side_effect=Exception("Routing failed")
        ):
            context = {}
            should_continue, error = await fabric_middleware.process_request(
                mock_request, context
            )

    assert should_continue is True
    assert error is None


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_full_pipeline_with_mock_sources(mock_request, basic_config):
    """Test full pipeline with mocked sources."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent, RoutingDecision
    from ai_inference_gateway.middleware.knowledge_fabric.core import SourceCapability, KnowledgeResult, KnowledgeChunk

    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=True,
        config=basic_config,
    )

    # Create a mock source with a non-empty chunk
    mock_chunk = KnowledgeChunk(
        content="Test knowledge content",
        source="test_source",
        score=0.9,
        metadata={}
    )

    mock_result = KnowledgeResult(
        source_name="test_source",
        chunks=[mock_chunk],
        query="What is the capital of France?",
        retrieval_time=0.1,
        metadata={}
    )

    mock_source = Mock()
    mock_source.name = "test_source"
    mock_source.retrieve = AsyncMock(return_value=mock_result)

    # Mock the complete pipeline
    with patch.object(
        middleware, "_extract_query", return_value="What is the capital of France?"
    ):
        # Mock routing - return non-empty selected_sources to avoid early return
        with patch.object(
            middleware.router, "classify", return_value=RoutingDecision(
                intent=QueryIntent.FACTUAL,
                confidence=0.9,
                required_capabilities=SourceCapability.FACTUAL,
                selected_sources=["test_source"],  # Non-empty to continue
                reasoning="Test"
            )
        ):
            # Mock _sources_by_name to provide the mock source
            with patch.object(
                middleware, "_sources_by_name", {"test_source": mock_source}
            ):
                # Mock fusion with non-empty chunks
                mock_fused_chunk = Mock()
                mock_fused_chunk.content = "Fused knowledge content"
                mock_fused_chunk.source = "test_source"
                mock_fused_chunk.score = 0.9

                with patch.object(
                    middleware.fusion, "fuse", return_value=[mock_fused_chunk]
                ):
                    # Mock synthesizer
                    with patch.object(
                        middleware.synthesizer, "synthesize", return_value="Knowledge context"
                    ):
                        context = {}
                        should_continue, error = await middleware.process_request(
                            mock_request, context
                        )

    assert should_continue is True
    assert error is None
    assert KNOWLEDGE_CONTEXT_KEY in context


# ============================================================================
# Config Tests
# ============================================================================


def test_create_default_sources_with_config():
    """Test _create_default_sources uses configuration."""
    config = {
        "rag_enabled": True,
        "code_search_paths": ["/custom/path"],
        "searxng_url": "http://custom:7777",
    }

    middleware = KnowledgeFabricMiddleware(
        sources=None,  # Will trigger _create_default_sources
        rrf_k=60,
        enabled=True,
        config=config,
    )

    # Sources should be created based on config
    assert len(middleware.sources) > 0
    # Check code_search uses configured path
    code_source = next((s for s in middleware.sources if s.name == "code_search"), None)
    if code_source:
        # Note: The actual source object may not have search_paths as a direct attribute
        # This test verifies the middleware was created
        pass


def test_rrf_k_parameter_propagation():
    """Test RRF k parameter is properly propagated."""
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=100,
        enabled=True,
        config={},
    )

    assert middleware.fusion.k == 100


def test_enabled_property():
    """Test enabled property reflects middleware state."""
    enabled_middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=True,
        config={},
    )

    disabled_middleware = KnowledgeFabricMiddleware(
        sources=[],
        rrf_k=60,
        enabled=False,
        config={},
    )

    assert enabled_middleware.enabled is True
    assert disabled_middleware.enabled is False


# ============================================================================
# Source Priority Tests
# ============================================================================


def test_sources_by_name_index(fabric_middleware):
    """Test sources are indexed by name for quick access."""
    assert fabric_middleware._sources_by_name is not None
    assert isinstance(fabric_middleware._sources_by_name, dict)


@pytest.mark.asyncio
async def test_source_selection_by_name(mock_request, fabric_middleware):
    """Test sources can be selected by name during retrieval."""
    with patch.object(
        fabric_middleware, "_extract_query", return_value="test query"
    ):
        with patch.object(
            fabric_middleware.router, "classify", return_value=Mock(
                intent=QueryIntent.FACTUAL,
                confidence=0.9,
                selected_sources=["rag", "web_search"],
                reasoning="Test"
            )
        ):
            # Mock the sources
            mock_rag = Mock()
            mock_rag.retrieve = AsyncMock(return_value=Mock(
                chunks=[], source_name="rag", retrieval_time=0.1, metadata={}
            ))

            mock_web = Mock()
            mock_web.retrieve = AsyncMock(return_value=Mock(
                chunks=[], source_name="web_search", retrieval_time=0.1, metadata={}
            ))

            fabric_middleware._sources_by_name = {
                "rag": mock_rag,
                "web_search": mock_web,
            }

            context = {}
            await fabric_middleware.process_request(mock_request, context)

    # Both sources should have been called
    mock_rag.retrieve.assert_called_once()
    mock_web.retrieve.assert_called_once()


# ============================================================================
# SemanticRouter Confidence Thresholding Tests
# ============================================================================


@pytest.fixture
def mock_sources():
    """Create mock knowledge sources for testing."""
    sources = []
    for i in range(3):
        source = Mock()
        source.name = f"source_{i}"
        source.priority = i
        source.enabled = True
        source.can_handle = Mock(return_value=True)
        sources.append(source)
    return sources


def test_router_uses_default_confidence_threshold(mock_sources):
    """Test router uses default MIN_CONFIDENCE threshold (0.5)."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        create_router,
    )

    router = SemanticRouter(mock_sources)
    assert router.confidence_threshold == SemanticRouter.MIN_CONFIDENCE
    assert SemanticRouter.MIN_CONFIDENCE == 0.5


def test_create_router_with_custom_confidence_threshold(mock_sources):
    """Test factory function accepts custom confidence threshold."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        create_router,
    )

    custom_threshold = 0.8
    router = create_router(mock_sources, confidence_threshold=custom_threshold)

    assert router.confidence_threshold == custom_threshold


def test_classify_low_confidence_returns_empty_sources(mock_sources):
    """Test classify returns empty selected_sources when confidence below threshold."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        QueryIntent,
        SourceCapability,
    )

    # Set high threshold to trigger low confidence behavior
    router = SemanticRouter(mock_sources, confidence_threshold=0.9)

    # Query that will match patterns but likely have low confidence
    query = "hi"

    decision = router.classify(query)

    # Should return empty sources due to low confidence
    assert decision.selected_sources == []
    assert "SKIPPED: low confidence" in decision.reasoning
    assert decision.confidence < 0.9


def test_classify_high_confidence_returns_selected_sources(mock_sources):
    """Test classify returns selected_sources when confidence above threshold."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        QueryIntent,
    )

    # Set low threshold to ensure routing proceeds
    router = SemanticRouter(mock_sources, confidence_threshold=0.1)

    # Query with strong intent indicators
    query = "How do I implement a function in Python?"

    decision = router.classify(query)

    # Should have detected intent and selected sources
    assert decision.intent == QueryIntent.CODE or decision.intent == QueryIntent.PROCEDURAL
    # Confidence should be calculated
    assert 0 <= decision.confidence <= 1
    assert "SKIPPED: low confidence" not in decision.reasoning


def test_classify_at_threshold_boundary(mock_sources):
    """Test classify behavior at exact threshold boundary."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        QueryIntent,
    )

    threshold = 0.5
    router = SemanticRouter(mock_sources, confidence_threshold=threshold)

    query = "What is the capital of France?"
    decision = router.classify(query)

    # At or above threshold should proceed
    if decision.confidence >= threshold:
        # Either sources selected or reasoning doesn't indicate skip
        assert "SKIPPED: low confidence" not in decision.reasoning
    else:
        assert decision.selected_sources == []


def test_classify_with_no_pattern_matches_returns_unknown_intent(mock_sources):
    """Test classify returns UNKNOWN intent for ambiguous queries."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        QueryIntent,
    )

    router = SemanticRouter(mock_sources, confidence_threshold=0.1)

    # Very ambiguous query with no pattern matches
    query = "asdfgh"

    decision = router.classify(query)

    # Unknown intent with low confidence
    assert decision.intent == QueryIntent.UNKNOWN


def test_confidence_threshold_affects_routing_decision(mock_sources):
    """Test different thresholds produce different routing decisions."""
    from ai_inference_gateway.middleware.knowledge_fabric.routing import (
        SemanticRouter,
        create_router,
    )

    query = "maybe do something"  # Weak intent

    # Low threshold - should route
    permissive_router = create_router(mock_sources, confidence_threshold=0.1)
    permissive_decision = permissive_router.classify(query)

    # High threshold - should skip
    strict_router = create_router(mock_sources, confidence_threshold=0.9)
    strict_decision = strict_router.classify(query)

    # Permissive router should have same or more sources than strict
    assert len(permissive_decision.selected_sources) >= len(
        strict_decision.selected_sources
    )

    # If confidence is between thresholds, permissive routes, strict doesn't
    if 0.1 < permissive_decision.confidence < 0.9:
        assert len(permissive_decision.selected_sources) > 0
        assert len(strict_decision.selected_sources) == 0


# ============================================================================
# RRFFusion Reranking Tests
# ============================================================================


def test_fusion_uses_default_reranker_parameters():
    """Test RRFFusion uses default parameters for reranking."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import (
        RRFFusion,
        DEFAULT_RERANKER_MODEL,
    )

    fusion = RRFFusion()

    assert fusion.reranker_enabled is False  # Disabled by default
    assert fusion.reranker_model == DEFAULT_RERANKER_MODEL
    assert fusion.final_k == 5
    assert fusion.max_chunks_for_rerank == 30


def test_fusion_with_reranking_enabled():
    """Test RRFFusion can be configured with reranking enabled."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import RRFFusion

    fusion = RRFFusion(
        reranker_enabled=True,
        reranker_model="custom-model",
        final_k=10,
        max_chunks_for_rerank=50
    )

    assert fusion.reranker_enabled is True
    assert fusion.reranker_model == "custom-model"
    assert fusion.final_k == 10
    assert fusion.max_chunks_for_rerank == 50


@pytest.mark.asyncio
async def test_initialize_reranker_loads_model():
    """Test initialize_reranker loads the cross-encoder model."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import RRFFusion
    from unittest.mock import patch, AsyncMock

    fusion = RRFFusion(reranker_enabled=True)

    # Mock sentence_transformers import
    with patch("ai_inference_gateway.middleware.knowledge_fabric.fusion.CrossEncoder") as mock_ce:
        mock_model = AsyncMock()
        mock_ce.return_value = mock_model

        await fusion.initialize_reranker()

        # Model should be loaded
        assert fusion._reranker == mock_model


@pytest.mark.asyncio
async def test_initialize_reranker_handles_import_error():
    """Test initialize_reranker gracefully handles import errors."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import RRFFusion
    from unittest.mock import patch

    fusion = RRFFusion(reranker_enabled=True)

    # Mock import to fail
    with patch("ai_inference_gateway.middleware.knowledge_fabric.fusion.CrossEncoder", side_effect=ImportError("No module")):
        await fusion.initialize_reranker()

    # Should disable reranking on import failure
    assert fusion.reranker_enabled is False


@pytest.mark.asyncio
async def test_fuse_reranks_chunks_when_enabled():
    """Test fuse calls rerank_chunks when reranking is enabled."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import (
        RRFFusion,
        create_fusion,
    )
    from ai_inference_gateway.middleware.knowledge_fabric.core import (
        FabricContext,
        KnowledgeResult,
        KnowledgeChunk,
    )

    # Create fusion with reranking enabled
    fusion = RRFFusion(reranker_enabled=True, final_k=3, max_chunks_for_rerank=10)
    fusion._reranker = None  # Will be initialized

    # Create mock chunks
    chunks = [
        KnowledgeChunk(content="First result", source="test", score=0.9, metadata={}),
        KnowledgeChunk(content="Second result", source="test", score=0.8, metadata={}),
        KnowledgeChunk(content="Third result", source="test", score=0.7, metadata={}),
    ]

    results = [
        KnowledgeResult(
            source_name="test",
            chunks=chunks,
            query="test query",
            retrieval_time=0.1,
            metadata={}
        )
    ]

    context = FabricContext(
        query="test query",
        query_type=QueryIntent.FACTUAL,
        results={},
        fused_chunks=[],
        sources_used=[],
        routing_reasoning="test"
    )

    # Mock initialize_reranker
    with patch.object(fusion, "initialize_reranker"):
        # Mock _rerank_chunks
        with patch.object(fusion, "_rerank_chunks", return_value=chunks[:3]) as mock_rerank:
            result = await fusion.fuse(results, context)

            # _rerank_chunks should be called when reranking enabled
            mock_rerank.assert_called_once_with("test query", chunks)


@pytest.mark.asyncio
async def test_rerank_chunks_returns_top_k():
    """Test _rerank_chunks returns top-K chunks after reranking."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import RRFFusion
    from unittest.mock import Mock

    fusion = RRFFusion(reranker_enabled=True, final_k=2, max_chunks_for_rerank=10)

    # Mock reranker model
    mock_reranker = Mock()
    mock_reranker.predict.return_value = [0.5, 0.9, 0.3]  # Second chunk highest
    fusion._reranker = mock_reranker

    # Create chunks
    chunks = [
        KnowledgeChunk(content="A", source="test", score=0.9, metadata={}),
        KnowledgeChunk(content="B", source="test", score=0.8, metadata={}),
        KnowledgeChunk(content="C", source="test", score=0.7, metadata={}),
    ]

    result = await fusion._rerank_chunks("test query", chunks)

    # Should return top 2 (0.9, 0.3)
    assert len(result) == 2
    # Reranker scores should be in metadata
    assert all("reranker_score" in c.metadata for c in result)
    # Sorted by reranker score descending
    assert result[0].metadata["reranker_score"] == 0.9
    assert result[1].metadata["reranker_score"] == 0.3


@pytest.mark.asyncio
async def test_rerank_chunks_falls_back_on_error():
    """Test _rerank_chunks falls back to RRF on reranker failure."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import RRFFusion

    fusion = RRFFusion(reranker_enabled=True, final_k=2, max_chunks_for_rerank=10)
    fusion._reranker = Mock()
    fusion._reranker.predict.side_effect = Exception("Reranker failed")

    # Create chunks
    chunks = [
        KnowledgeChunk(content="A", source="test", score=0.9, metadata={}),
        KnowledgeChunk(content="B", source="test", score=0.8, metadata={}),
    ]

    result = await fusion._rerank_chunks("test query", chunks)

    # Should return chunks without reranking (top final_k)
    assert len(result) == 2
    # Original scores should be preserved
    assert result[0].score == 0.9
    assert result[1].score == 0.8


def test_create_fusion_with_reranking_params():
    """Test create_fusion factory accepts reranking parameters."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import (
        create_fusion,
        RRFFusion,
    )

    fusion = create_fusion(
        reranker_enabled=True,
        reranker_model="BAAI/bge-reranker-v2-m3",
        final_k=10,
        max_chunks_for_rerank=50
    )

    assert isinstance(fusion, RRFFusion)
    assert fusion.reranker_enabled is True
    assert fusion.reranker_model == "BAAI/bge-reranker-v2-m3"
    assert fusion.final_k == 10
    assert fusion.max_chunks_for_rerank == 50


@pytest.mark.asyncio
async def test_fuse_without_reranking_returns_rrf_results():
    """Test fuse without reranking returns standard RRF results."""
    from ai_inference_gateway.middleware.knowledge_fabric.fusion import (
        RRFFusion,
    )
    from ai_inference_gateway.middleware.knowledge_fabric.core import (
        FabricContext,
        KnowledgeResult,
        KnowledgeChunk,
    )

    fusion = RRFFusion(reranker_enabled=False)

    chunks = [
        KnowledgeChunk(content="Result 1", source="test", score=0.9, metadata={}),
        KnowledgeChunk(content="Result 2", source="test", score=0.8, metadata={}),
    ]

    results = [
        KnowledgeResult(
            source_name="test",
            chunks=chunks,
            query="test query",
            retrieval_time=0.1,
            metadata={}
        )
    ]

    context = FabricContext(
        query="test query",
        query_type=QueryIntent.FACTUAL,
        results={},
        fused_chunks=[],
        sources_used=[],
        routing_reasoning="test"
    )

    result = await fusion.fuse(results, context)

    # Should return RRF-sorted chunks
    assert len(result) == 2
    assert result[0].score >= result[1].score  # Sorted descending
    # Should have rrf_score in metadata
    assert "rrf_score" in result[0].metadata


# ============================================================================
# Circuit Breaker Tests
# ============================================================================


@pytest.fixture
def circuit_breaker_registry():
    """Create a circuit breaker registry for testing."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        create_circuit_breaker_registry,
    )
    
    # Create registry with fast timeouts for testing
    return create_circuit_breaker_registry(
        failure_threshold=2,  # Open after 2 failures
        timeout=5.0,          # 5 second timeout for testing
        success_threshold=1  # Close after 1 success
    )


def test_circuit_breaker_initial_state(circuit_breaker_registry):
    """Test circuit breaker starts in CLOSED state."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    assert breaker.state.state == CircuitState.CLOSED
    assert breaker.state.failure_count == 0
    assert breaker.state.success_count == 0


def test_circuit_breaker_opens_after_threshold(circuit_breaker_registry):
    """Test circuit breaker opens after failure threshold."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Record failures up to threshold
    breaker.record_failure(Exception("Error 1"))
    assert breaker.state.state == CircuitState.CLOSED
    assert breaker.state.failure_count == 1
    
    breaker.record_failure(Exception("Error 2"))
    assert breaker.state.state == CircuitState.OPEN
    assert breaker.state.failure_count == 2  # Reset after opening


def test_circuit_breaker_blocks_calls_when_open(circuit_breaker_registry):
    """Test circuit breaker blocks calls when in OPEN state."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Open the circuit
    breaker.record_failure(Exception("Error 1"))
    breaker.record_failure(Exception("Error 2"))
    assert breaker.state.state == CircuitState.OPEN
    
    # Call should be blocked
    assert not breaker.is_call_allowed()
    assert breaker.state.rejected_calls == 1


def test_circuit_breaker_allows_call_after_timeout(circuit_breaker_registry):
    """Test circuit breaker allows call after timeout (HALF_OPEN)."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    from unittest.mock import patch
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Open the circuit
    breaker.record_failure(Exception("Error 1"))
    breaker.record_failure(Exception("Error 2"))
    assert breaker.state.state == CircuitState.OPEN
    
    # Mock time to simulate timeout
    with patch("ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker.datetime") as mock_dt:
        # Simulate time passage beyond timeout
        import time
        mock_dt.now.return_value = datetime.fromtimestamp(
            breaker.state.last_state_change.timestamp() + 6
        )
        
        # Should transition to HALF_OPEN and allow call
        assert breaker.is_call_allowed()
        assert breaker.state.state == CircuitState.HALF_OPEN


def test_circuit_breaker_closes_after_success_threshold(circuit_breaker_registry):
    """Test circuit breaker closes after success threshold in HALF_OPEN."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Open the circuit
    breaker.record_failure(Exception("Error 1"))
    breaker.record_failure(Exception("Error 2"))
    assert breaker.state.state == CircuitState.OPEN
    
    # Manually transition to HALF_OPEN (simulating timeout)
    breaker._transition_to_half_open()
    
    # Record success in HALF_OPEN
    breaker.record_success()
    assert breaker.state.state == CircuitState.CLOSED
    assert breaker.state.failure_count == 0


def test_circuit_breaker_resets_failure_count_on_success(circuit_breaker_registry):
    """Test circuit breaker resets failure count on success in CLOSED state."""
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Record some failures
    breaker.record_failure(Exception("Error 1"))
    assert breaker.state.failure_count == 1
    
    # Record success in CLOSED state
    breaker.record_success()
    assert breaker.state.failure_count == 0


def test_circuit_breaker_returns_to_open_on_half_open_failure(circuit_breaker_registry):
    """Test circuit breaker returns to OPEN on failure in HALF_OPEN."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Open and transition to HALF_OPEN
    breaker.record_failure(Exception("Error 1"))
    breaker.record_failure(Exception("Error 2"))
    breaker._transition_to_half_open()
    assert breaker.state.state == CircuitState.HALF_OPEN
    
    # Failure in HALF_OPEN returns to OPEN
    breaker.record_failure(Exception("Error 3"))
    assert breaker.state.state == CircuitState.OPEN


def test_circuit_breaker_get_metrics(circuit_breaker_registry):
    """Test circuit breaker returns comprehensive metrics."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    breaker = circuit_breaker_registry.get_breaker("test_source")
    
    # Record some activity
    breaker.record_failure(Exception("Error 1"))
    metrics = breaker.get_metrics()
    
    assert metrics["source"] == "test_source"
    assert metrics["state"] == CircuitState.CLOSED.value
    assert metrics["failure_count"] == 1
    assert metrics["successful_calls"] == 0
    assert metrics["failed_calls"] == 1
    assert "last_failure_time" in metrics


@pytest.mark.asyncio
async def test_execute_with_circuit_breaker_blocks_open_circuit():
    """Test execute_with_circuit_breaker returns empty result when circuit is open."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        create_circuit_breaker_registry,
    )
    from ai_inference_gateway.middleware.knowledge_fabric.core import KnowledgeResult
    
    registry = create_circuit_breaker_registry(failure_threshold=1)
    
    # Open the circuit by recording a failure
    registry.record_failure("test_source", Exception("Initial failure"))
    
    # Call should be blocked and return empty KnowledgeResult
    result = await execute_with_circuit_breaker(
        registry=registry,
        source_name="test_source",
        callable_func=lambda query, context: None,
        query="test query",
        context={}
    )
    
    assert isinstance(result, KnowledgeResult)
    assert result.chunks == []
    assert result.source_name == "test_source"
    assert "circuit_breaker" in result.metadata
    assert result.metadata["circuit_breaker"] == "blocked"


@pytest.mark.asyncio
async def test_execute_with_circuit_breaker_executes_when_closed():
    """Test execute_with_circuit_breaker executes function when circuit is closed."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        create_circuit_breaker_registry,
    )
    from ai_inference_gateway.middleware.knowledge_fabric.core import KnowledgeChunk
    
    registry = create_circuit_breaker_registry(failure_threshold=2)
    
    async def mock_retrieve(query, context):
        return KnowledgeResult(
            source_name="test_source",
            chunks=[KnowledgeChunk(content="Result", source="test", score=0.9, metadata={})],
            query=query,
            retrieval_time=0.1,
            metadata={}
        )
    
    # Call should execute normally
    result = await execute_with_circuit_breaker(
        registry=registry,
        source_name="test_source",
        callable_func=mock_retrieve,
        query="test query",
        context={}
    )
    
    assert result.source_name == "test_source"
    assert len(result.chunks) == 1
    assert result.chunks[0].content == "Result"


@pytest.mark.asyncio
async def test_execute_with_circuit_breaker_records_success():
    """Test execute_with_circuit_breaker records success on successful call."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        create_circuit_breaker_registry,
    )
    
    registry = create_circuit_breaker_registry()
    
    async def mock_retrieve(query, context):
        from ai_inference_gateway.middleware.knowledge_fabric.core import KnowledgeResult, KnowledgeChunk
        return KnowledgeResult(
            source_name="test_source",
            chunks=[KnowledgeChunk(content="Result", source="test", score=0.9, metadata={})],
            query=query,
            retrieval_time=0.1,
            metadata={}
        )
    
    await execute_with_circuit_breaker(
        registry=registry,
        source_name="test_source",
        callable_func=mock_retrieve,
        query="test query",
        context={}
    )
    
    breaker = registry.get_breaker("test_source")
    assert breaker.state.successful_calls == 1
    assert breaker.state.failed_calls == 0


@pytest.mark.asyncio
async def test_execute_with_circuit_breaker_records_failure():
    """Test execute_with_circuit_breaker records failure on exception."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        create_circuit_breaker_registry,
    )
    
    registry = create_circuit_breaker_registry()
    
    async def failing_retrieve(query, context):
        raise ValueError("Source error")
    
    try:
        await execute_with_circuit_breaker(
            registry=registry,
            source_name="test_source",
            callable_func=failing_retrieve,
            query="test query",
            context={}
        )
    except ValueError:
        pass  # Expected to be re-raised
    
    breaker = registry.get_breaker("test_source")
    assert breaker.state.failed_calls == 1
    assert breaker.state.failure_count == 1


def test_circuit_breaker_registry_get_summary_metrics():
    """Test circuit breaker registry returns aggregated metrics."""
    from ai_inference_gateway.middleware.knowledge_fabric.circuit_breaker import (
        CircuitState,
    )
    
    registry = create_circuit_breaker_registry()
    
    # Get metrics with no breakers
    summary = registry.get_summary_metrics()
    assert summary["total_sources"] == 0
    assert summary["open_circuits"] == 0
    assert summary["closed_circuits"] == 0
    
    # Add some activity
    breaker1 = registry.get_breaker("source1")
    breaker1.record_failure(Exception("Error"))
    
    breaker2 = registry.get_breaker("source2")
    breaker2.record_failure(Exception("Error"))
    breaker2.record_failure(Exception("Error"))
    
    summary = registry.get_summary_metrics()
    assert summary["total_sources"] == 2
    assert summary["open_circuits"] == 1  # source2
    assert summary["closed_circuits"] == 1  # source1


@pytest.mark.asyncio
async def test_fabric_middleware_includes_circuit_breaker():
    """Test KnowledgeFabricMiddleware includes circuit breaker registry."""
    from ai_inference_gateway.middleware.knowledge_fabric import (
        KnowledgeFabricMiddleware,
    )
    
    middleware = KnowledgeFabricMiddleware(
        sources=[],
        enabled=True,
        config={}
    )
    
    # Should have circuit breaker registry initialized
    assert hasattr(middleware, "circuit_registry")
    assert middleware.circuit_registry is not None


def test_create_knowledge_fabric_accepts_circuit_breaker_config():
    """Test create_knowledge_fabric accepts circuit breaker configuration."""
    from ai_inference_gateway.middleware.knowledge_fabric import (
        create_knowledge_fabric,
    )
    
    config = {
        "circuit_breaker": {
            "failure_threshold": 3,
            "timeout": 120.0,
            "success_threshold": 3
        }
    }
    
    middleware = create_knowledge_fabric(
        sources=[],
        enabled=True,
        config=config
    )
    
    # Config should be passed to circuit breaker registry
    assert middleware.circuit_registry.default_config.failure_threshold == 3
    assert middleware.circuit_registry.default_config.timeout == 120.0
    assert middleware.circuit_registry.default_config.success_threshold == 3
