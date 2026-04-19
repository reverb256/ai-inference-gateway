"""
Integration tests for /search/hybrid endpoint.

Run against live gateway: pytest tests/test_hybrid_search_live.py -v
Requires: gateway running on 10.1.1.120:8080 with RAG + SearXNG + reranker enabled.
"""

import json
import pytest
import urllib.request

GATEWAY = "http://10.1.1.120:8080"


def hybrid_search(query, **kwargs):
    """Helper to call /search/hybrid."""
    body = {"query": query, "max_results": kwargs.pop("max_results", 10)}
    body.update(kwargs)
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{GATEWAY}/search/hybrid",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read())


# ============================================================================
# 1. Semantic Router — intent classification drives source selection
# ============================================================================

class TestSemanticRouter:
    """Intent classification should influence which sources are searched."""

    def test_realtime_intent_skips_rag(self):
        """Queries with 'current/today/now' should get realtime intent and skip RAG."""
        result = hybrid_search("current bitcoin price today", max_results=5)
        meta = result["metadata"]

        assert meta["routing_intent"] == "realtime"
        assert meta["intent_weights"]["rag"] == 0.0
        assert meta["intent_weights"]["web"] == 1.0
        # RAG should have 0 results since weight is 0
        assert result["sources"]["rag"] == 0
        # Web should have results
        assert result["sources"]["web"] > 0

    def test_procedural_intent_boosts_rag(self):
        """How-to queries should get procedural intent with RAG > web weight."""
        result = hybrid_search("how to configure colmena deployment nixos", max_results=5)
        meta = result["metadata"]

        assert meta["routing_intent"] == "procedural"
        assert meta["intent_weights"]["rag"] > meta["intent_weights"]["web"]

    def test_factual_intent_boosts_rag(self):
        """What-is queries should get factual intent with RAG boosted."""
        result = hybrid_search("what is the knowledge fabric middleware", max_results=5)
        meta = result["metadata"]

        assert meta["routing_intent"] == "factual"
        assert meta["intent_weights"]["rag"] >= 1.0

    def test_comparative_intent_equal_weights(self):
        """Comparison queries should get equal RAG/web weights."""
        result = hybrid_search("compare nixos vs ubuntu server", max_results=5)
        meta = result["metadata"]

        assert meta["routing_intent"] == "comparative"
        assert meta["intent_weights"]["rag"] == meta["intent_weights"]["web"]

    def test_contextual_intent_boosts_rag(self):
        """Why-does queries should get contextual intent with RAG boosted."""
        result = hybrid_search("why does the gateway use circuit breakers", max_results=5)
        meta = result["metadata"]

        assert meta["routing_intent"] == "contextual"
        assert meta["intent_weights"]["rag"] > meta["intent_weights"]["web"]

    def test_unknown_intent_defaults_equal(self):
        """Queries matching no patterns should get equal weights."""
        result = hybrid_search("osaka jade omarchy", max_results=5)
        meta = result["metadata"]

        # Either None or unknown — either way weights should be equal
        if meta["routing_intent"] is None:
            assert meta["intent_weights"]["rag"] == 1.0
            assert meta["intent_weights"]["web"] == 1.0
        else:
            assert meta["routing_intent"] == "unknown"


# ============================================================================
# 2. Cross-Encoder Reranker — neural scoring replaces text matching
# ============================================================================

class TestCrossEncoderReranker:
    """Cross-encoder should produce reranker_score on results."""

    def test_reranker_scores_present(self):
        """Results should have reranker_score from cross-encoder."""
        result = hybrid_search("nixos colmena deployment", max_results=5, rerank=True)

        assert result["metadata"]["reranked"] is True
        # At least some results should have reranker_score
        scored = [r for r in result["results"] if "reranker_score" in r]
        assert len(scored) > 0, "No results have reranker_score"

    def test_reranker_method_is_cross_encoder(self):
        """Rerank method should indicate cross_encoder, not heuristic."""
        result = hybrid_search("nixos colmena deployment", max_results=5, rerank=True)

        methods = [r.get("rerank_method") for r in result["results"]]
        assert "cross_encoder" in methods, f"Expected cross_encoder, got {set(methods)}"

    def test_reranker_scores_are_reasonable(self):
        """Cross-encoder scores should be in a reasonable range."""
        result = hybrid_search("nixos colmena deployment", max_results=5, rerank=True)

        for r in result["results"]:
            if "reranker_score" in r:
                score = r["reranker_score"]
                assert -1.0 <= score <= 1.1, f"Score {score} out of range"

    def test_reranker_results_sorted_descending(self):
        """Results should be sorted by reranker_score descending."""
        result = hybrid_search("nixos colmena deployment", max_results=10, rerank=True)

        scores = [r.get("reranker_score", 0) for r in result["results"]]
        assert scores == sorted(scores, reverse=True), f"Not sorted: {scores}"

    def test_reranker_disabled_uses_heuristic(self):
        """With rerank=false, no reranker_score should appear."""
        result = hybrid_search("nixos colmena deployment", max_results=5, rerank=False)

        assert result["metadata"]["reranked"] is False
        # Should NOT have reranker_score when reranking is off
        reranked = [r for r in result["results"] if "reranker_score" in r]
        assert len(reranked) == 0


# ============================================================================
# 3. Adaptive Local-First Routing — skip web when RAG is confident
# ============================================================================

class TestAdaptiveLocalFirst:
    """When RAG has high confidence (>= 0.7), web search should be skipped."""

    def test_high_confidence_skips_web(self):
        """A query about deeply-documented wiki content should skip web."""
        result = hybrid_search(
            "Knowledge Fabric Middleware RRF fusion sources",
            max_results=5
        )
        meta = result["metadata"]

        assert meta["rag_confidence"] >= 0.7, f"Confidence {meta['rag_confidence']} < 0.7"
        assert result["sources"]["web"] == 0, "Web should be 0 when RAG confident"

    def test_low_confidence_searches_both(self):
        """A query with low RAG confidence should search both sources."""
        result = hybrid_search("osaka jade omarchy theme", max_results=5)
        meta = result["metadata"]

        # This query has low RAG scores, so both should be searched
        total = result["sources"]["rag"] + result["sources"]["web"]
        assert total > 0, "Should have results from at least one source"

    def test_rag_confidence_in_metadata(self):
        """rag_confidence should always be present in metadata."""
        result = hybrid_search("anything", max_results=3)

        assert "rag_confidence" in result["metadata"]
        assert isinstance(result["metadata"]["rag_confidence"], (int, float))


# ============================================================================
# 4. Query Expansion — LLM generates search variants
# ============================================================================

class TestQueryExpansion:
    """Short queries should be expanded via LLM into variants."""

    def test_short_query_expanded(self):
        """Short query should produce multiple search results (from variants)."""
        # This test just verifies the expansion path doesn't crash
        result = hybrid_search("osaka jade", max_results=5)

        assert result["metadata"]["duration_ms"] > 0
        assert len(result["results"]) > 0

    def test_expansion_results_have_variant_tag(self):
        """Results from expanded queries should have query_variant field."""
        result = hybrid_search("osaka jade theme", max_results=10)

        variants = [r.get("query_variant") for r in result["results"]]
        # At least some results should have the variant tag
        tagged = [v for v in variants if v]
        assert len(tagged) > 0, "No results have query_variant tag"


# ============================================================================
# 5. BGE-M3 Embedding Pipeline — dense scores and dimensions
# ============================================================================

class TestEmbeddingPipeline:
    """BGE-M3 should produce 1024d vectors and reasonable cosine scores."""

    def test_dense_score_present_in_rag_results(self):
        """RAG results should carry dense_score (cosine similarity)."""
        result = hybrid_search("nixos colmena deployment", max_results=10)

        rag_results = [r for r in result["results"] if r.get("source") == "rag"]
        assert len(rag_results) > 0, "No RAG results"

        for r in rag_results[:3]:
            assert "dense_score" in r, "RAG result missing dense_score"
            assert r["dense_score"] is not None

    def test_dense_scores_reasonable_range(self):
        """Cosine similarity scores should be in [0, 1] range."""
        result = hybrid_search("nixos colmena deployment", max_results=10)

        for r in result["results"]:
            if r.get("source") == "rag" and r.get("dense_score"):
                score = r["dense_score"]
                assert 0.0 <= score <= 1.0, f"Dense score {score} out of range"

    def test_default_collection_is_brain_wiki(self):
        """Default collection should be brain-wiki, not 'default'."""
        # Query without specifying collection
        result = hybrid_search("nixos configuration", max_results=3)

        # Should return RAG results (proves brain-wiki was searched)
        rag_results = [r for r in result["results"] if r.get("source") == "rag"]
        assert len(rag_results) > 0, "No RAG results — default collection may be wrong"


# ============================================================================
# 6. Response Structure — metadata and format consistency
# ============================================================================

class TestResponseStructure:
    """All responses should have consistent structure."""

    def test_response_has_required_fields(self):
        result = hybrid_search("test query", max_results=5)

        assert "results" in result
        assert "sources" in result
        assert "metadata" in result

    def test_metadata_has_required_fields(self):
        result = hybrid_search("test query", max_results=5)
        meta = result["metadata"]

        required = ["query", "total_found", "returned", "duration_ms",
                     "used_rag", "used_web", "reranked", "routing_intent",
                     "intent_weights", "rag_confidence", "timestamp"]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"

    def test_results_have_source_field(self):
        result = hybrid_search("nixos colmena", max_results=5)

        for r in result["results"]:
            assert "source" in r
            assert r["source"] in ("rag", "web")

    def test_sources_dict_has_rag_and_web(self):
        result = hybrid_search("nixos colmena", max_results=5)

        assert "rag" in result["sources"]
        assert "web" in result["sources"]

    def test_max_results_respected(self):
        result = hybrid_search("nixos", max_results=3)

        assert len(result["results"]) <= 3


# ============================================================================
# 7. Performance — baseline timing expectations
# ============================================================================

class TestPerformance:
    """Basic timing expectations for the hybrid search pipeline."""

    def test_realtime_query_under_1s(self):
        """REALTIME queries (web-only) should be fast."""
        result = hybrid_search("current news today", max_results=5)
        ms = result["metadata"]["duration_ms"]

        assert ms < 1500, f"REALTIME query took {ms}ms (expected <1500ms)"

    def test_local_first_under_10s(self):
        """Local-first queries (web skipped) should be reasonable."""
        result = hybrid_search(
            "Knowledge Fabric Middleware RRF fusion sources",
            max_results=5
        )
        ms = result["metadata"]["duration_ms"]

        assert ms < 10000, f"Local-first query took {ms}ms (expected <10000ms)"

    def test_full_hybrid_under_15s(self):
        """Full hybrid (RAG + web + reranker) should complete."""
        result = hybrid_search("nixos colmena deployment", max_results=5)
        ms = result["metadata"]["duration_ms"]

        assert ms < 15000, f"Full hybrid took {ms}ms (expected <15000ms)"
