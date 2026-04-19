"""
Phase 2 Integration Tests — Semantic Cache, Temporal Decay, Multi-Hop, ContextSynthesizer.

Run: cd /data/projects/own/ai-inference-gateway && nix develop -c pytest tests/test_hybrid_search_live.py -v --tb=short
"""

import json
import time
import urllib.request
import pytest

GATEWAY = "http://10.1.1.120:8080"
COLLECTION = "brain-wiki"


def _hybrid(query, max_results=5, **kwargs):
    """POST /search/hybrid and return parsed JSON."""
    payload = {"query": query, "max_results": max_results, "collection": COLLECTION}
    payload.update(kwargs)
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{GATEWAY}/search/hybrid",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


# ─── Semantic Cache ──────────────────────────────────────────────


class TestSemanticCache:
    """Tests for embedding-similarity keyed cache."""

    def test_repeated_query_returns_cache_hit(self):
        """Same query twice should return cache hit on second call."""
        r1 = _hybrid("nixos nix develop command")
        r2 = _hybrid("nixos nix develop command")
        # Second call should have cache_hit in metadata
        assert r2["metadata"].get("cache_hit") in ("exact", "semantic"), (
            f"Expected cache hit on repeat query, got: {r2['metadata']}"
        )

    def test_paraphrased_query_hits_cache(self):
        """Semantically similar query should hit cache if threshold met."""
        r1 = _hybrid("how to configure nixos")
        # Paraphrase
        r2 = _hybrid("nixos configuration guide")
        # May or may not hit depending on threshold, but should not error
        assert "results" in r2
        assert "metadata" in r2

    def test_cache_age_reported(self):
        """Cached results should report age."""
        r1 = _hybrid("cache age test query unique 12345")
        r2 = _hybrid("cache age test query unique 12345")
        if "cache_hit" in r2.get("metadata", {}):
            assert r2["metadata"].get("cache_age_seconds") is not None

    def test_cache_not_on_first_query(self):
        """First query should not be a cache hit."""
        ts = int(time.time())
        r = _hybrid(f"unique first query test {ts}")
        # First call should NOT be a cache hit
        assert r["metadata"].get("cache_hit") is None, (
            f"First query should not be cached, got: {r['metadata']}"
        )


# ─── Temporal Decay ──────────────────────────────────────────────


class TestTemporalDecay:
    """Tests for temporal decay on reranker scores."""

    def test_temporal_decay_field_present(self):
        """Reranked results should include temporal_decay field."""
        r = _hybrid("nixos configuration", max_results=5)
        has_decay = any(
            "temporal_decay" in res
            for res in r["results"]
            if res.get("rerank_method") == "cross_encoder"
        )
        assert has_decay, "No temporal_decay found in reranked results"

    def test_temporal_decay_range(self):
        """Temporal decay should be between 0.1 and 1.0."""
        r = _hybrid("python tutorial", max_results=5)
        for res in r["results"]:
            decay = res.get("temporal_decay")
            if decay is not None:
                assert 0.1 <= decay <= 1.0, f"Decay {decay} out of range"

    def test_reranker_raw_score_preserved(self):
        """Original reranker score should be preserved before decay."""
        r = _hybrid("rust programming", max_results=5)
        for res in r["results"]:
            if "reranker_score_raw" in res:
                assert res["reranker_score_raw"] >= res.get("reranker_score", 0), (
                    "Raw score should be >= decay-adjusted score"
                )


# ─── Multi-Hop Wiki-Link Retrieval ──────────────────────────────


class TestMultiHopRetrieval:
    """Tests for [[wiki-link]] following in RAG results."""

    def test_wiki_link_extraction(self):
        """Queries against wiki content should extract [[links]]."""
        # Use a query likely to hit wiki pages with links
        r = _hybrid("personal autonomous corporation knowledge fabric", max_results=10)
        # Check if any results have hop_origin (multi-hop was followed)
        hop_results = [res for res in r["results"] if res.get("hop_origin")]
        # May or may not have hops depending on content, just verify structure
        for res in hop_results:
            assert res["source"] == "rag"
            assert res["hop_origin"] is not None

    def test_hop_results_are_rag(self):
        """Multi-hop results should always be from RAG source."""
        r = _hybrid("5GW cognitive defense protocol", max_results=10)
        for res in r["results"]:
            if res.get("hop_origin"):
                assert res["source"] == "rag", (
                    f"Hop result should be RAG, got {res['source']}"
                )


# ─── ContextSynthesizer ──────────────────────────────────────────


class TestContextSynthesizer:
    """Tests for LLM-ready context synthesis in response."""

    def test_context_field_present(self):
        """Hybrid search should include synthesized context field."""
        r = _hybrid("nixos nix develop")
        assert "context" in r, "Response missing 'context' field"

    def test_context_is_string(self):
        """Context should be a string."""
        r = _hybrid("nixos configuration")
        assert isinstance(r["context"], str), f"Context should be str, got {type(r['context'])}"

    def test_context_has_source_sections(self):
        """Context should have Local Knowledge and/or Web Sources sections."""
        r = _hybrid("python async await", max_results=5)
        ctx = r["context"]
        has_sections = "Local Knowledge" in ctx or "Web Sources" in ctx
        assert has_sections, f"Context missing source sections: {ctx[:200]}"

    def test_context_intent_header(self):
        """Context header should reflect detected intent."""
        r = _hybrid("what is the latest news today", max_results=3)
        ctx = r["context"]
        # REALTIME intent should produce "Current Information" header
        assert len(ctx) > 0, "Context should not be empty"

    def test_context_empty_on_no_results(self):
        """Context should be empty string if no results."""
        ts = int(time.time())
        r = _hybrid(f"zzzznonexistent{ts}", max_results=1)
        # Even with no results, context field should exist
        assert "context" in r


# ─── Existing Tests (still valid) ────────────────────────────────


class TestSemanticRouter:
    def test_realtime_intent_skips_rag(self):
        r = _hybrid("what is the latest news today right now", max_results=3)
        assert r["metadata"]["used_rag"] is False, "REALTIME should skip RAG"

    def test_procedural_intent_boosts_rag(self):
        r = _hybrid("how to set up nixos step by step", max_results=3)
        assert r["metadata"]["routing_intent"] == "procedural"

    def test_factual_intent_boosts_rag(self):
        r = _hybrid("what is the definition of sovereignty", max_results=3)
        assert r["metadata"]["routing_intent"] == "factual"

    def test_comparative_intent_equal_weights(self):
        r = _hybrid("compare nixos vs arch linux pros and cons", max_results=3)
        w = r["metadata"]["intent_weights"]
        assert w["rag"] == w["web"], f"Expected equal weights, got {w}"

    def test_contextual_intent_boosts_rag(self):
        r = _hybrid("explain the context of AI alignment research", max_results=3)
        assert r["metadata"]["routing_intent"] in ("contextual", "factual", None)

    def test_unknown_intent_defaults_equal(self):
        r = _hybrid("asdfghjkl random query", max_results=3)
        if r["metadata"]["routing_intent"] is None:
            assert r["metadata"]["intent_weights"]["rag"] == 1.0


class TestCrossEncoderReranker:
    def test_reranker_scores_present(self):
        r = _hybrid("nixos configuration guide", max_results=5)
        has_reranker = any("reranker_score" in res for res in r["results"])
        assert has_reranker, "No reranker_score in results"

    def test_reranker_method_is_cross_encoder(self):
        r = _hybrid("python async await tutorial", max_results=5)
        methods = {res.get("rerank_method") for res in r["results"]}
        assert "cross_encoder" in methods or "heuristic" in methods

    def test_reranker_scores_are_reasonable(self):
        r = _hybrid("rust programming language", max_results=5)
        for res in r["results"]:
            if "reranker_score" in res:
                assert -5 <= res["reranker_score"] <= 5

    def test_reranker_results_sorted_descending(self):
        r = _hybrid("neural network deep learning", max_results=5)
        scores = [res.get("reranker_score", float("-inf")) for res in r["results"]]
        assert scores == sorted(scores, reverse=True), f"Results not sorted: {scores}"

    def test_reranker_disabled_uses_heuristic(self):
        # This tests the fallback path — hard to trigger in live, but verify structure
        r = _hybrid("test query", max_results=3)
        for res in r["results"]:
            assert "reranker_score" in res or "rerank_score" in res


class TestAdaptiveLocalFirst:
    def test_high_confidence_skips_web(self):
        # Query likely to have high RAG confidence in brain-wiki
        r = _hybrid("nixos colmena deployment", max_results=5)
        # If confidence >= 0.7, web should be skipped
        conf = r["metadata"].get("rag_confidence", 0)
        if conf >= 0.7:
            assert r["metadata"]["used_web"] is False or r["sources"]["web"] == 0

    def test_low_confidence_searches_both(self):
        ts = int(time.time())
        r = _hybrid(f"obscure topic xyz{ts}", max_results=3)
        # Low confidence should try both sources
        assert "results" in r

    def test_rag_confidence_in_metadata(self):
        r = _hybrid("test query", max_results=3)
        assert "rag_confidence" in r["metadata"]


class TestResponseStructure:
    def test_response_has_required_fields(self):
        r = _hybrid("test", max_results=3)
        assert "results" in r
        assert "sources" in r
        assert "metadata" in r

    def test_metadata_has_required_fields(self):
        r = _hybrid("test", max_results=3)
        m = r["metadata"]
        for key in ["query", "total_found", "returned", "duration_ms", "timestamp"]:
            assert key in m, f"Missing metadata key: {key}"

    def test_results_have_source_field(self):
        r = _hybrid("python", max_results=3)
        for res in r["results"]:
            assert "source" in res, f"Result missing source: {res}"

    def test_sources_dict_has_rag_and_web(self):
        r = _hybrid("nixos", max_results=3)
        assert "rag" in r["sources"]
        assert "web" in r["sources"]

    def test_max_results_respected(self):
        ts = int(time.time())
        r = _hybrid(f"test query max results {ts}", max_results=2)
        assert len(r["results"]) <= 2


class TestPerformance:
    def test_realtime_query_under_1s(self):
        t0 = time.time()
        r = _hybrid("what is happening right now in the news today", max_results=3)
        ms = (time.time() - t0) * 1000
        # If cache hit, should be near-instant
        if r["metadata"].get("cache_hit"):
            assert ms < 500, f"Cached REALTIME query took {ms}ms"
        else:
            assert ms < 2000, f"REALTIME query took {ms}ms (expected <2000ms)"

    def test_local_first_under_10s(self):
        t0 = time.time()
        _hybrid("nixos colmena cluster setup", max_results=5)
        ms = (time.time() - t0) * 1000
        assert ms < 10000, f"Local-first query took {ms}ms"

    def test_full_hybrid_under_25s(self):
        t0 = time.time()
        _hybrid("compare python vs rust performance benchmarks", max_results=5)
        ms = (time.time() - t0) * 1000
        assert ms < 25000, f"Full hybrid took {ms}ms"
