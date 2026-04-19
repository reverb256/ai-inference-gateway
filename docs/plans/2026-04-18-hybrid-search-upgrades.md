# Hybrid Search Upgrade Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Upgrade the `/search/hybrid` endpoint from simple parallel search to a full intelligent retrieval pipeline with semantic routing, cross-encoder reranking, query expansion, adaptive local-first routing, semantic caching, temporal decay, and multi-hop wiki-link retrieval.

**Architecture:** The gateway already has a sophisticated Knowledge Fabric middleware (`middleware/knowledge_fabric/`) with semantic routing, cross-encoder reranking, context synthesis, circuit breakers, and metrics. The `/search/hybrid` endpoint (`src/hybrid_search.py`) is a simpler parallel system that doesn't use any of it. We bridge them by upgrading the hybrid endpoint to leverage the Knowledge Fabric components, then add new capabilities on top.

**Tech Stack:** Python/FastAPI, Qdrant, BGE-M3 embeddings, BGE-reranker-v2-base cross-encoder, SearXNG, local Qwen3.6 for query expansion.

**Key Files:**
- `src/hybrid_search.py` — Current hybrid endpoint engine (upgraded)
- `src/middleware/knowledge_fabric/routing.py` — Semantic router (wire in)
- `src/middleware/knowledge_fabric/fusion.py` — RRF + cross-encoder reranker (enable)
- `src/middleware/knowledge_fabric/fabric.py` — Main orchestrator (reference)
- `src/main.py` — FastAPI endpoints (modify defaults)
- `src/rag/qdrant_client.py` — Qdrant client (add temporal decay, wiki-link hop)
- `src/rag/search.py` — RAG search service (add dense_score passthrough - DONE)
- `nix/options.nix` — NixOS module options (add new config options)
- `nix/gateway.nix` — NixOS service config (wire new env vars)

---

## Phase 1: Wire Existing Dormant Systems (Low Effort, High Impact)

### Task 1.1: Wire Semantic Router into HybridSearchEngine

**Objective:** Route queries to optimize source selection based on intent classification.

**Files:**
- Modify: `src/hybrid_search.py:25-43` (HybridSearchEngine.__init__)
- Modify: `src/hybrid_search.py:44-143` (HybridSearchEngine.search)

**Implementation:**
- Import `SemanticRouter`, `QueryIntent` from `middleware.knowledge_fabric.routing`
- In `__init__`, create a `SemanticRouter` with source metadata (no need for full KnowledgeSource objects — just use the routing patterns)
- In `search()`, call `router.classify(query)` to get intent
- Apply source weights based on intent:
  - `REALTIME` → `use_rag=False, use_web=True` (skip stale local docs)
  - `CODE` / `FACTUAL` → boost RAG weight 1.5x
  - `COMPARATIVE` → equal weight, increase max_results
  - `CONTEXTUAL` → boost RAG weight 1.3x
  - `UNKNOWN` → default (both sources, equal)
- Log the routing decision for debugging

**Verification:** `curl /search/hybrid -d '{"query":"current price of bitcoin"}'` should show web-only results. `curl /search/hybrid -d '{"query":"how to configure colmena deployment"}'` should prioritize RAG.

### Task 1.2: Enable Cross-Encoder Reranker

**Objective:** Replace text-matching heuristic with neural cross-encoder scoring.

**Files:**
- Modify: `src/hybrid_search.py:25-43` (add reranker init)
- Modify: `src/hybrid_search.py:222-290` (replace `_rerank_results`)
- Modify: `nix/options.nix:809-821` (ensure reranker options exist)
- Modify: `nix/gateway.nix:184` (wire RERANKER_ENABLED)

**Implementation:**
- In `HybridSearchEngine.__init__`, add optional `CrossEncoder` loading (lazy, gated on `RERANKER_ENABLED` env var)
- Replace `_rerank_results` to use cross-encoder when available, fall back to heuristic
- Cross-encoder flow: take top-30 fused results → score (query, content) pairs → return top-K
- Model: `BAAI/bge-reranker-v2-base` (~1.2GB, runs on CPU, ~200ms for 30 pairs)
- Set `RERANKER_ENABLED=true` in the nix config

**Verification:** Check logs for "Loading reranker model" on startup. Test that `/search/hybrid` results show `reranker_score` in metadata.

### Task 1.3: Wire ContextSynthesizer for LLM-Ready Output

**Objective:** Format fused results into structured prompts tailored by query type.

**Files:**
- Modify: `src/hybrid_search.py:130-143` (add synthesis to return value)
- Modify: `src/main.py:2729-2782` (add `synthesized` field to response)

**Implementation:**
- Import `ContextSynthesizer` from `middleware.knowledge_fabric.fusion`
- After fusing and ranking, call `synthesizer.synthesize(fabric_context)` to get LLM-ready text
- Add `"synthesized_context"` field to the hybrid search response (optional, opt-in via request param `include_context=true`)
- The synthesized output includes source attribution and query-type-specific formatting

**Verification:** `curl /search/hybrid -d '{"query":"...", "include_context":true}'` should return a `synthesized_context` field with formatted text.

---

## Phase 2: Intelligence Upgrades (Medium Effort, Very High Impact)

### Task 2.1: LLM Query Expansion

**Objective:** Generate 2-3 reformulated query variants before searching to improve recall.

**Files:**
- Create: `src/query_expansion.py` (new module)
- Modify: `src/hybrid_search.py:44-143` (integrate expansion into search flow)

**Implementation:**
- New `QueryExpander` class that calls the local LLM (Qwen3.6 via gateway's own `/v1/chat/completions`) with a prompt like: "Given the query '{q}', generate 2 alternative search queries that would find the same information using different terms. Return as JSON array."
- In `search()`, if query length < 50 chars (short queries benefit most), expand first
- Search with original + expanded queries in parallel
- Merge all results before reranking
- Cache expansions (same query = same expansions for 1 hour)
- Fallback gracefully if LLM is unavailable

**Verification:** Test with "osaka jade" — should generate variants like "omarchy desktop theme jade green", "dhh omarchy osaka jade theme" and find better results.

### Task 2.2: Adaptive Local-First Routing

**Objective:** Skip SearXNG when local brain has high-confidence results.

**Files:**
- Modify: `src/hybrid_search.py:76-113` (add early-exit logic)

**Implementation:**
- After RAG search, check if top result has `dense_score >= 0.7`
- If yes AND query intent is not REALTIME, skip SearXNG entirely
- This saves ~800ms per query on the majority of searches
- If RAG confidence is 0.5-0.7, still do web search but weight RAG heavier
- If RAG confidence < 0.5, equal weight or web-heavy
- Log the routing decision: "Local-first: RAG confidence 0.72, skipping web" vs "Low confidence 0.31, searching both"

**Verification:** Query for "nixos colmena deployment" (which has high RAG scores ~0.63) should be noticeably faster. Query for "current bitcoin price" should still hit web.

### Task 2.3: Semantic Caching

**Objective:** Cache query→results mappings keyed by embedding similarity, not exact string match.

**Files:**
- Create: `src/semantic_cache.py` (new module)
- Modify: `src/hybrid_search.py:44-143` (check cache before search, populate after)

**Implementation:**
- `SemanticCache` class using an in-memory dict of `{query_embedding: results}`
- On search: embed query → check cache for any entry with cosine similarity > 0.95 → return cached
- On miss: perform full search → cache results with query embedding
- LRU eviction with max 1000 entries (covers common query patterns)
- TTL of 30 minutes (stale enough for knowledge that changes slowly)
- Use the existing BGE-M3 embedder (gateway endpoint or local)
- Skip cache for REALTIME queries (news/prices need fresh data)

**Verification:** Second identical query should return in <50ms from cache. Slightly rephrased query ("how to deploy colmena" vs "colmena deployment steps") should also hit cache at >0.95 similarity.

---

## Phase 3: Knowledge Quality (Medium Effort, Medium-High Impact)

### Task 3.1: Temporal Decay on Qdrant Points

**Objective:** Prefer recent knowledge over stale docs, unless specifically queried.

**Files:**
- Modify: `src/rag/qdrant_client.py:100-112` (add decay to search_dense)
- Modify: `src/rag/search.py:67-142` (pass decay config)

**Implementation:**
- Add `last_modified` timestamp to Qdrant point metadata during ingestion
- During search, apply decay: `adjusted_score = cosine_score * exp(-lambda * days_since_modified)` where lambda=0.005 (half-life ~140 days)
- Only apply when query intent is not HISTORICAL (don't decay queries about old stuff)
- Make decay configurable via `TEMPORAL_DECAY_ENABLED` env var and `TEMPORAL_DECAY_LAMBDA`

**Verification:** Query for "nixos configuration" should prefer recent wiki pages. Old pages about deprecated configs should rank lower.

### Task 3.2: Multi-Hop Wiki-Link Retrieval

**Objective:** Follow `[[wiki-links]]` in retrieved chunks to pull in related context.

**Files:**
- Create: `src/wiki_link_hop.py` (new module)
- Modify: `src/hybrid_search.py:76-113` (add hop step after RAG results)

**Implementation:**
- After RAG search returns top-K chunks, scan for `[[wiki-link]]` patterns
- Extract unique link targets (max 5 to prevent explosion)
- For each target, look up the wiki page in Qdrant by title/metadata
- Add those chunks to the result set with `"source": "rag_hop"` and lower weight
- This captures the deeply cross-linked nature of the brain wiki (289 pages with heavy interlinking)
- Configurable via `WIKI_HOP_ENABLED` env var

**Verification:** Query about "Colmena deployment" should also pull in linked pages about "NixOS modules" and "agent architecture" even if they didn't match the original query embedding.

---

## Phase 4: NixOS Declarative Config (Low Effort, Required)

### Task 4.1: Add New Env Vars to NixOS Module

**Objective:** Make all new features configurable through NixOS declarations.

**Files:**
- Modify: `nix/options.nix` (add new options)
- Modify: `nix/gateway.nix` (wire new env vars)
- Modify: `/etc/nixos/kubernetes/modules/ai-inference.nix` (host config)

**New Options:**
```
gateway.reranker.enable (already exists, default true)
gateway.hybridSearch.semanticRouting.enable (default true)
gateway.hybridSearch.queryExpansion.enable (default false)
gateway.hybridSearch.queryExpansion.llmUrl (default gateway's own URL)
gateway.hybridSearch.adaptiveRouting.enable (default true)
gateway.hybridSearch.adaptiveRouting.confidenceThreshold (default 0.7)
gateway.hybridSearch.semanticCache.enable (default true)
gateway.hybridSearch.semanticCache.maxEntries (default 1000)
gateway.hybridSearch.semanticCache.ttlMinutes (default 30)
gateway.rag.temporalDecay.enable (default false)
gateway.rag.temporalDecay.lambda (default 0.005)
gateway.hybridSearch.wikiHop.enable (default true)
gateway.hybridSearch.wikiHop.maxHops (default 5)
```

**Verification:** `colmena build --on nexus` succeeds. All env vars present in service config.

---

## Execution Order

1. **Task 1.1** — Wire Semantic Router (existing code, low risk)
2. **Task 1.2** — Enable Cross-Encoder Reranker (existing code, high impact)
3. **Task 2.2** — Adaptive Local-First Routing (small change, big perf win)
4. **Task 2.1** — LLM Query Expansion (new code, biggest recall boost)
5. **Task 2.3** — Semantic Caching (new code, perf optimization)
6. **Task 3.1** — Temporal Decay (small change, quality improvement)
7. **Task 3.2** — Multi-Hop Wiki-Link (new code, leverages wiki structure)
8. **Task 1.3** — Context Synthesizer (existing code, output formatting)
9. **Task 4.1** — NixOS Declarative Config (wire everything declaratively)

Deploy after each task via: scp patched files → ssh nexus systemctl restart → curl test.

## Rollback

Each task is independently deployable and revertable. If any task causes issues:
1. `ssh nexus 'sudo systemctl restart ai-inference-gateway'` (reverts to last working state)
2. `git revert` the specific commit in the gateway repo
3. The imperative PYTHONPATH override means we can hot-fix without Colmena

## Dependencies

- **Task 1.2** requires `sentence_transformers` (already installed in the nix env)
- **Task 2.1** requires the gateway's own LLM endpoint (already running)
- **Task 2.3** requires the embedding endpoint (already running, BGE-M3)
- **Task 3.2** requires wiki pages with `[[link]]` syntax (already present in brain wiki)
- **All tasks** require the Colmena deploy to complete (or the imperative PYTHONPATH override to remain active)
