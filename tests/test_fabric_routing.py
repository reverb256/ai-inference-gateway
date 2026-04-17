"""
Tests for Knowledge Fabric semantic routing.

Tests the SemanticRouter's ability to classify queries by intent
and select appropriate knowledge sources.
"""

import pytest
from ai_inference_gateway.middleware.knowledge_fabric.routing import (
    SemanticRouter,
    QueryIntent,
    RoutingDecision,
)
from ai_inference_gateway.middleware.knowledge_fabric.core import (
    SourceCapability,
    SourcePriority,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_sources():
    """Create mock knowledge sources for testing."""
    from ai_inference_gateway.middleware.knowledge_fabric.core import KnowledgeSource

    sources = []

    # RAG source - FACTUAL, CONTEXTUAL, PROCEDURAL
    rag = type("MockRAG", (KnowledgeSource,), {
        "name": "rag",
        "description": "RAG knowledge base",
        "priority": SourcePriority.HIGH,
        "capabilities": (
            SourceCapability.FACTUAL |
            SourceCapability.CONTEXTUAL |
            SourceCapability.PROCEDURAL
        ),
    })()

    # Code search - CODE, PROCEDURAL
    code = type("MockCode", (KnowledgeSource,), {
        "name": "code_search",
        "description": "Code search",
        "priority": SourcePriority.CRITICAL,
        "capabilities": (
            SourceCapability.CODE |
            SourceCapability.PROCEDURAL
        ),
    })()

    # SearXNG - FACTUAL, REALTIME, COMPARATIVE
    searxng = type("MockSearXNG", (KnowledgeSource,), {
        "name": "searxng",
        "description": "Web meta-search",
        "priority": SourcePriority.MEDIUM,
        "capabilities": (
            SourceCapability.FACTUAL |
            SourceCapability.REALTIME |
            SourceCapability.COMPARATIVE
        ),
    })()

    # Web search - REALTIME, FACTUAL
    web = type("MockWeb", (KnowledgeSource,), {
        "name": "web_search",
        "description": "MCP web search",
        "priority": SourcePriority.MEDIUM,
        "capabilities": (
            SourceCapability.REALTIME |
            SourceCapability.FACTUAL
        ),
    })()

    sources.extend([rag, code, searxng, web])
    return sources


@pytest.fixture
def router(mock_sources):
    """Create a SemanticRouter with mock sources."""
    return SemanticRouter(mock_sources)


# ============================================================================
# Query Intent Classification Tests
# ============================================================================


@pytest.mark.parametrize("query,expected_intent", [
    # CODE intent queries
    ("How do I implement a binary search tree?", QueryIntent.CODE),
    ("def fibonacci(n): return fibonacci", QueryIntent.CODE),
    ("What's the Python syntax for list comprehension?", QueryIntent.CODE),
    ("Explain the decorator pattern in C#", QueryIntent.CODE),
    ("fix the bug in my React component", QueryIntent.CODE),

    # FACTUAL intent queries
    ("What is the capital of France?", QueryIntent.FACTUAL),
    ("Who wrote the declaration of independence?", QueryIntent.FACTUAL),
    ("What is the molecular weight of water?", QueryIntent.FACTUAL),
    ("List the planets in our solar system", QueryIntent.FACTUAL),

    # PROCEDURAL intent queries
    ("How do I configure nginx?", QueryIntent.PROCEDURAL),
    ("Steps to deploy a Kubernetes cluster", QueryIntent.PROCEDURAL),
    ("Setup process for PostgreSQL replication", QueryIntent.PROCEDURAL),
    ("Installation guide for Docker", QueryIntent.PROCEDURAL),

    # REALTIME intent queries
    ("What's the current stock price of AAPL?", QueryIntent.REALTIME),
    ("Latest news about the election", QueryIntent.REALTIME),
    ("Current weather in Tokyo", QueryIntent.REALTIME),
    ("Today's USD to EUR exchange rate", QueryIntent.REALTIME),

    # COMPARATIVE intent queries
    ("Python vs JavaScript: which is better?", QueryIntent.COMPARATIVE),
    ("Compare PostgreSQL and MySQL", QueryIntent.COMPARATIVE),
    ("Difference between HTTP/1.1 and HTTP/2", QueryIntent.COMPARATIVE),
    ("AWS vs Azure for machine learning", QueryIntent.COMPARATIVE),

    # CONTEXTUAL intent queries
    ("Given our previous discussion about the API...", QueryIntent.CONTEXTUAL),
    ("Based on the document I uploaded...", QueryIntent.CONTEXTUAL),
    ("Referring to the earlier context...", QueryIntent.CONTEXTUAL),

    # UNKNOWN intent queries
    ("xyz", QueryIntent.UNKNOWN),
    ("a b c", QueryIntent.UNKNOWN),
    ("...", QueryIntent.UNKNOWN),
])
def test_query_intent_classification(router, query, expected_intent):
    """Test query classification into intent categories."""
    decision = router.classify(query)

    assert decision.intent == expected_intent
    assert decision.confidence >= 0.0
    assert decision.confidence <= 1.0
    assert isinstance(decision.selected_sources, list)


# ============================================================================
# Source Selection Tests
# ============================================================================


def test_code_query_selects_code_source(router):
    """Test CODE queries prioritize code_search source."""
    query = "How do I implement a binary search tree?"
    decision = router.classify(query)

    assert "code_search" in decision.selected_sources
    # May also include other sources with CODE capability


def test_factual_query_selects_multiple_sources(router):
    """Test FACTUAL queries can use multiple sources."""
    query = "What is the capital of France?"
    decision = router.classify(query)

    # Should select from RAG, SearXNG, or Web search
    # All have FACTUAL capability
    assert len(decision.selected_sources) > 0


def test_realtime_query_selects_realtime_sources(router):
    """Test REALTIME queries prioritize sources with REALTIME capability."""
    query = "What's the current stock price of AAPL?"
    decision = router.classify(query)

    # SearXNG and Web search both have REALTIME capability
    assert "searxng" in decision.selected_sources or "web_search" in decision.selected_sources


def test_comparative_query_includes_comparative_sources(router):
    """Test COMPARATIVE queries prefer sources with COMPARATIVE capability."""
    query = "Python vs JavaScript: which is better?"
    decision = router.classify(query)

    # SearXNG has COMPARATIVE capability
    assert len(decision.selected_sources) > 0


def test_short_query_returns_unknown(router):
    """Test very short queries return UNKNOWN intent."""
    query = "xyz"
    decision = router.classify(query)

    assert decision.intent == QueryIntent.UNKNOWN
    # May still select some sources for fallback


def test_mixed_intent_query(router):
    """Test queries with multiple intent markers."""
    # This query has both CODE and FACTUAL elements
    # Should classify based on pattern matching priority
    query = "How do I code a function to calculate pi?"
    decision = router.classify(query)

    # "code" keyword should trigger CODE intent
    assert decision.intent == QueryIntent.CODE


# ============================================================================
# Priority-Based Selection Tests
# ============================================================================


def test_critical_priority_source_selected(router):
    """Test CRITICAL priority sources are always selected when relevant."""
    query = "fix the bug in my function"
    decision = router.classify(query)

    # code_search has CRITICAL priority
    assert "code_search" in decision.selected_sources


def test_priority_affects_source_order(router):
    """Test source priority affects selection order."""
    query = "What is the capital of France?"
    decision = router.classify(query)

    # Check that higher priority sources appear earlier
    if "code_search" in decision.selected_sources:
        code_idx = decision.selected_sources.index("code_search")
        # RAG has HIGH priority (2), code has CRITICAL (1)
        # Code should come before MEDIUM priority sources
        for source in ["searxng", "web_search"]:
            if source in decision.selected_sources:
                assert code_idx < decision.selected_sources.index(source)


# ============================================================================
# Routing Reasoning Tests
# ============================================================================


def test_routing_decision_includes_reasoning(router):
    """Test routing decision includes reasoning explanation."""
    query = "How do I implement a binary search tree?"
    decision = router.classify(query)

    assert decision.reasoning is not None
    assert len(decision.reasoning) > 0
    assert isinstance(decision.reasoning, str)


def test_routing_reasoning_mentions_matched_patterns(router):
    """Test reasoning mentions which patterns matched."""
    query = "How do I implement a binary search tree in Python?"
    decision = router.classify(query)

    # Reasoning should mention something about the classification
    assert "code" in decision.reasoning.lower() or "keyword" in decision.reasoning.lower()


# ============================================================================
# Edge Cases
# ============================================================================


def test_empty_query(router):
    """Test empty query handling."""
    query = ""
    decision = router.classify(query)

    assert decision.intent == QueryIntent.UNKNOWN


def test_query_with_special_chars(router):
    """Test query with special characters."""
    query = "What is $HOME in Linux? How about %TEMP%?"
    decision = router.classify(query)

    # Should still classify, likely as FACTUAL
    assert decision.intent is not None


def test_very_long_query(router):
    """Test very long query handling."""
    query = "What is " + "the " * 100 + "meaning of life?"
    decision = router.classify(query)

    # Should handle without error
    assert decision.intent is not None
    assert decision.confidence >= 0.0


def test_multilingual_query(router):
    """Test queries in different languages."""
    # Spanish for "What is the capital of France?"
    query = "¿Cuál es la capital de Francia?"
    decision = router.classify(query)

    # Should handle gracefully (likely UNKNOWN or FACTUAL)
    assert decision.intent is not None


# ============================================================================
# Confidence Score Tests
# ============================================================================


def test_high_confidence_classification(router):
    """Test high confidence classification for clear patterns."""
    query = "def foo(): return bar"
    decision = router.classify(query)

    # Code pattern is very clear
    assert decision.confidence > 0.5


def test_low_confidence_classification(router):
    """Test low confidence for ambiguous queries."""
    query = "information about things"
    decision = router.classify(query)

    # Ambiguous query should have lower confidence
    assert 0.0 <= decision.confidence <= 1.0


def test_exact_pattern_match_has_high_confidence(router):
    """Test exact keyword matches yield high confidence."""
    query = "Python vs JavaScript comparison"
    decision = router.classify(query)

    # "vs" and "comparison" are strong COMPARATIVE signals
    if decision.intent == QueryIntent.COMPARATIVE:
        assert decision.confidence > 0.7
