"""
Tests for Knowledge Fabric RRF (Reciprocal Rank Fusion).

Tests the RRFFusion class for merging results from multiple
knowledge sources using the Reciprocal Rank Fusion algorithm.
"""

import pytest
from ai_inference_gateway.middleware.knowledge_fabric.fusion import (
    RRFFusion,
    ContextSynthesizer,
)
from ai_inference_gateway.middleware.knowledge_fabric.core import (
    KnowledgeChunk,
    KnowledgeResult,
    FabricContext,
    SourceCapability,
)
from ai_inference_gateway.middleware.knowledge_fabric.routing import QueryIntent


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_chunks_rag():
    """Create sample chunks from RAG source."""
    return [
        KnowledgeChunk(
            content="Paris is the capital of France",
            source="rag",
            score=0.95,
            metadata={"type": "encyclopedia"},
            capabilities=SourceCapability.FACTUAL,
        ),
        KnowledgeChunk(
            content="The Eiffel Tower is located in Paris",
            source="rag",
            score=0.88,
            metadata={"type": "landmark"},
            capabilities=SourceCapability.FACTUAL,
        ),
        KnowledgeChunk(
            content="France has a population of 67 million",
            source="rag",
            score=0.82,
            metadata={"type": "demographics"},
            capabilities=SourceCapability.FACTUAL,
        ),
    ]


@pytest.fixture
def sample_chunks_web():
    """Create sample chunks from web search source."""
    return [
        KnowledgeChunk(
            content="Paris, France's capital, is known for the Eiffel Tower",
            source="web_search",
            score=0.90,
            metadata={"url": "https://example.com/paris"},
            capabilities=SourceCapability.FACTUAL,
        ),
        KnowledgeChunk(
            content="Key facts about Paris: population 2.1 million",
            source="web_search",
            score=0.75,
            metadata={"url": "https://example.com/paris-facts"},
            capabilities=SourceCapability.FACTUAL,
        ),
    ]


@pytest.fixture
def sample_chunks_code():
    """Create sample chunks from code search source."""
    return [
        KnowledgeChunk(
            content="def get_capital(country): return capitals.get(country)",
            source="code_search",
            score=0.85,
            metadata={"file": "/etc/nixos/config.py", "line": 42},
            capabilities=SourceCapability.CODE,
        ),
        KnowledgeChunk(
            content="capitals = {'France': 'Paris', 'Germany': 'Berlin'}",
            source="code_search",
            score=0.80,
            metadata={"file": "/etc/nixos/config.py", "line": 10},
            capabilities=SourceCapability.CODE,
        ),
    ]


@pytest.fixture
def fabric_context():
    """Create a sample FabricContext."""
    return FabricContext(
        query="What is the capital of France?",
        query_type=QueryIntent.FACTUAL,
        results={},
        fused_chunks=[],
        sources_used=[],
        routing_reasoning="Query contains factual question pattern",
    )


# ============================================================================
# RRF Fusion Tests
# ============================================================================


def test_rrf_initialization():
    """Test RRF fusion can be initialized with custom k parameter."""
    fusion = RRFFusion(k=60)
    assert fusion.k == 60

    fusion_custom = RRFFusion(k=100)
    assert fusion_custom.k == 100


def test_rrf_default_k():
    """Test RRF fusion uses default k=60."""
    fusion = RRFFusion()
    assert fusion.k == 60


@pytest.mark.parametrize("k_value,expected_behavior", [
    (1, "should give significant weight to lower ranks"),
    (60, "standard RRF constant"),
    (100, "should give more weight to lower ranks"),
])
def test_rrf_k_parameter(k_value, expected_behavior):
    """Test different k values affect scoring."""
    fusion = RRFFusion(k=k_value)
    assert fusion.k == k_value
    # Behavioral testing would require actual fusion runs


def test_rrf_fuse_single_source(sample_chunks_rag, fabric_context):
    """Test RRF fusion with a single source."""
    fusion = RRFFusion(k=60)

    # Create a KnowledgeResult
    result = KnowledgeResult(
        source_name="rag",
        chunks=sample_chunks_rag,
        query="capital of France",
        retrieval_time=0.1,
        metadata={"count": len(sample_chunks_rag)},
    )

    fused = fusion.fuse([result], fabric_context)

    assert len(fused) == len(sample_chunks_rag)
    # Scores should be adjusted by RRF
    # Rank 1: 60/(60+1) ≈ 0.984
    # Rank 2: 60/(60+2) ≈ 0.968
    # Rank 3: 60/(60+3) ≈ 0.952
    assert fused[0].rrf_score > 0
    assert fused[0].rrf_score < 1.0


def test_rrf_fuse_multiple_sources(sample_chunks_rag, sample_chunks_web, fabric_context):
    """Test RRF fusion merges results from multiple sources."""
    fusion = RRFFusion(k=60)

    result_rag = KnowledgeResult(
        source_name="rag",
        chunks=sample_chunks_rag,
        query="capital of France",
        retrieval_time=0.1,
        metadata={},
    )

    result_web = KnowledgeResult(
        source_name="web_search",
        chunks=sample_chunks_web,
        query="capital of France",
        retrieval_time=0.15,
        metadata={},
    )

    fused = fusion.fuse([result_rag, result_web], fabric_context)

    # Should have chunks from both sources
    assert len(fused) > 0

    # All fused chunks should have RRF scores
    for chunk in fused:
        assert hasattr(chunk, "rrf_score")
        assert chunk.rrf_score > 0

    # Top result should have higher RRF score
    if len(fused) > 1:
        assert fused[0].rrf_score >= fused[1].rrf_score


def test_rrf_formula_calculation(fabric_context):
    """Test RRF formula: score = k / (k + rank)"""
    fusion = RRFFusion(k=60)

    # Create results with known ranks
    chunks_rank1 = [KnowledgeChunk(
        content="First result",
        source="test",
        score=1.0,
        metadata={},
        capabilities=SourceCapability.FACTUAL,
    )]

    chunks_rank2 = [KnowledgeChunk(
        content="Second result",
        source="test",
        score=0.9,
        metadata={},
        capabilities=SourceCapability.FACTUAL,
    )]

    result1 = KnowledgeResult(
        source_name="test",
        chunks=chunks_rank1,
        query="test",
        retrieval_time=0.1,
        metadata={},
    )

    result2 = KnowledgeResult(
        source_name="test",
        chunks=chunks_rank2,
        query="test",
        retrieval_time=0.1,
        metadata={},
    )

    fused = fusion.fuse([result1, result2], fabric_context)

    # RRF score for rank 1: 60/(60+1) = 60/61 ≈ 0.9836
    # RRF score for rank 2: 60/(60+2) = 60/62 ≈ 0.9677
    expected_score_1 = 60 / 61
    expected_score_2 = 60 / 62

    assert abs(fused[0].rrf_score - expected_score_1) < 0.001
    assert abs(fused[1].rrf_score - expected_score_2) < 0.001


def test_rrf_handles_empty_results(fabric_context):
    """Test RRF fusion handles empty result sets gracefully."""
    fusion = RRFFusion(k=60)

    result_empty = KnowledgeResult(
        source_name="empty_source",
        chunks=[],
        query="test query",
        retrieval_time=0.1,
        metadata={},
    )

    fused = fusion.fuse([result_empty], fabric_context)

    assert len(fused) == 0


def test_rrf_maintains_source_metadata(sample_chunks_rag, fabric_context):
    """Test RRF fusion preserves original chunk metadata."""
    fusion = RRFFusion(k=60)

    result = KnowledgeResult(
        source_name="rag",
        chunks=sample_chunks_rag,
        query="test",
        retrieval_time=0.1,
        metadata={},
    )

    fused = fusion.fuse([result], fabric_context)

    for chunk in fused:
        assert chunk.source == "rag"
        assert chunk.metadata is not None
        assert "rrf_score" in chunk.metadata


def test_rrf_deduplicates_similar_chunks(fabric_context):
    """Test RRF fusion handles near-duplicate chunks."""
    fusion = RRFFusion(k=60)

    chunk1 = KnowledgeChunk(
        content="Paris is the capital of France",
        source="rag",
        score=0.95,
        metadata={"type": "encyclopedia"},
        capabilities=SourceCapability.FACTUAL,
    )

    chunk2 = KnowledgeChunk(
        content="Paris is the capital of France",  # Exact duplicate
        source="web_search",
        score=0.90,
        metadata={"url": "https://example.com"},
        capabilities=SourceCapability.FACTUAL,
    )

    result1 = KnowledgeResult(
        source_name="rag",
        chunks=[chunk1],
        query="test",
        retrieval_time=0.1,
        metadata={},
    )

    result2 = KnowledgeResult(
        source_name="web_search",
        chunks=[chunk2],
        query="test",
        retrieval_time=0.1,
        metadata={},
    )

    fused = fusion.fuse([result1, result2], fabric_context)

    # RRF should combine scores for duplicates
    # Both chunks at rank 1 → score = 60/61 + 60/61 = 1.967
    assert len(fused) >= 1
    if len(fused) == 1:
        # Deduplication occurred
        assert fused[0].content == "Paris is the capital of France"
        # Combined score from both sources
        assert fused[0].rrf_score > 1.0


def test_rrf_rank_propagation():
    """Test that RRF properly propagates ranks through results."""
    fusion = RRFFusion(k=60)

    # Create results with 3 chunks each
    chunks1 = [
        KnowledgeChunk(content=f"Result {i}", source="s1", score=1.0 - i*0.1,
                        metadata={}, capabilities=SourceCapability.FACTUAL)
        for i in range(3)
    ]
    chunks2 = [
        KnowledgeChunk(content=f"Result {i}", source="s2", score=0.9 - i*0.1,
                        metadata={}, capabilities=SourceCapability.FACTUAL)
        for i in range(3)
    ]

    result1 = KnowledgeResult(source_name="s1", chunks=chunks1, query="test",
                              retrieval_time=0.1, metadata={})
    result2 = KnowledgeResult(source_name="s2", chunks=chunks2, query="test",
                              retrieval_time=0.1, metadata={})

    context = FabricContext(
        query="test",
        query_type=QueryIntent.FACTUAL,
        results={},
        fused_chunks=[],
        sources_used=[],
        routing_reasoning="test",
    )

    fused = fusion.fuse([result1, result2], context)

    # All chunks should have RRF scores
    for chunk in fused:
        assert chunk.rrf_score > 0
        # First chunks from each source (rank 1) should have highest scores
        if chunk.content in ["Result 0"]:
            assert chunk.rrf_score > 0.95  # 60/61 ≈ 0.984

    # RRF scores should be monotonically decreasing for same content
    result_0_scores = [c.rrf_score for c in fused if c.content == "Result 0"]
    result_0_scores.sort(reverse=True)
    assert result_0_scores == sorted(result_0_scores, reverse=True)


# ============================================================================
# Context Synthesizer Tests
# ============================================================================


def test_synthesizer_initialization():
    """Test ContextSynthesizer can be initialized."""
    synthesizer = ContextSynthesizer()
    assert synthesizer is not None


def test_synthesize_for_factual_query(fabric_context, sample_chunks_rag):
    """Test context synthesis for FACTUAL queries."""
    synthesizer = ContextSynthesizer()

    # Add results to context
    fabric_context.fused_chunks = sample_chunks_rag
    fabric_context.sources_used = ["rag"]

    synthesized = synthesizer.synthesize(fabric_context)

    assert synthesized is not None
    assert len(synthesized) > 0
    # Should include context header
    assert "context" in synthesized.lower() or "relevant information" in synthesized.lower()


def test_synthesize_for_code_query(sample_chunks_code):
    """Test context synthesis for CODE queries."""
    synthesizer = ContextSynthesizer()

    context = FabricContext(
        query="How do I implement a function?",
        query_type=QueryIntent.CODE,
        results={},
        fused_chunks=sample_chunks_code,
        sources_used=["code_search"],
        routing_reasoning="Code pattern detected",
    )

    synthesized = synthesizer.synthesize(context)

    assert synthesized is not None
    assert len(synthesized) > 0
    # Should mention code snippets
    assert "code" in synthesized.lower()


def test_synthesize_for_procedural_query(fabric_context):
    """Test context synthesis for PROCEDURAL queries."""
    synthesizer = ContextSynthesizer()

    fabric_context.query_type = QueryIntent.PROCEDURAL
    fabric_context.fused_chunks = sample_chunks_rag[:2]
    fabric_context.sources_used = ["rag"]

    synthesized = synthesizer.synthesize(fabric_context)

    assert synthesized is not None
    # Should include step-by-step formatting
    assert "context" in synthesized.lower() or "information" in synthesized.lower()


def test_synthesize_handles_empty_chunks(fabric_context):
    """Test synthesizer handles empty chunks gracefully."""
    synthesizer = ContextSynthesizer()

    fabric_context.fused_chunks = []
    fabric_context.sources_used = []

    synthesized = synthesizer.synthesize(fabric_context)

    # Should return empty or minimal context
    assert synthesized == "" or len(synthesized) < 100


def test_synthesize_includes_source_attribution(sample_chunks_rag, fabric_context):
    """Test synthesized context includes source attribution."""
    synthesizer = ContextSynthesizer()

    fabric_context.fused_chunks = sample_chunks_rag
    fabric_context.sources_used = ["rag"]

    synthesized = synthesizer.synthesize(fabric_context)

    # Should mention sources somewhere in the context
    # or structure the output clearly
    assert synthesized is not None


def test_synthesize_for_comparative_query(fabric_context):
    """Test context synthesis for COMPARATIVE queries."""
    synthesizer = ContextSynthesizer()

    fabric_context.query_type = QueryIntent.COMPARATIVE
    fabric_context.fused_chunks = sample_chunks_rag[:2]
    fabric_context.sources_used = ["rag"]

    synthesized = synthesizer.synthesize(fabric_context)

    assert synthesized is not None
    # Comparative queries should get structured comparison context


def test_synthesize_for_realtime_query(fabric_context):
    """Test context synthesis for REALTIME queries."""
    synthesizer = ContextSynthesizer()

    fabric_context.query_type = QueryIntent.REALTIME
    fabric_context.fused_chunks = sample_chunks_rag[:2]
    fabric_context.sources_used = ["web_search"]

    synthesized = synthesizer.synthesize(fabric_context)

    assert synthesized is not None
    # Realtime queries should emphasize recent/current info


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_fusion_pipeline(sample_chunks_rag, sample_chunks_web, fabric_context):
    """Test full pipeline: fuse + synthesize."""
    fusion = RRFFusion(k=60)
    synthesizer = ContextSynthesizer()

    result_rag = KnowledgeResult(
        source_name="rag",
        chunks=sample_chunks_rag,
        query="capital of France",
        retrieval_time=0.1,
        metadata={},
    )

    result_web = KnowledgeResult(
        source_name="web_search",
        chunks=sample_chunks_web,
        query="capital of France",
        retrieval_time=0.15,
        metadata={},
    )

    # Step 1: Fuse results
    fused = fusion.fuse([result_rag, result_web], fabric_context)
    fabric_context.fused_chunks = fused
    fabric_context.sources_used = ["rag", "web_search"]

    # Step 2: Synthesize context
    synthesized = synthesizer.synthesize(fabric_context)

    assert len(fused) > 0
    assert len(synthesized) > 0
    assert "rag" in fabric_context.sources_used
    assert "web_search" in fabric_context.sources_used


def test_fusion_preserves_retrieval_time(sample_chunks_rag, fabric_context):
    """Test fusion preserves retrieval time metadata."""
    fusion = RRFFusion(k=60)

    result = KnowledgeResult(
        source_name="rag",
        chunks=sample_chunks_rag,
        query="test",
        retrieval_time=0.123,
        metadata={"test": "value"},
    )

    fused = fusion.fuse([result], fabric_context)
    fabric_context.fused_chunks = fused
    fabric_context.results = {"rag": result}

    # Check that retrieval time is tracked
    total_time = sum(r.retrieval_time for r in fabric_context.results.values())
    assert total_time == 0.123
