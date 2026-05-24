# AGENTS.md — AI Inference Gateway

## Project Overview

Python FastAPI gateway providing OpenAI/Anthropic/Ollama-compatible API endpoints with intelligent routing, circuit breaker failover, RAG (Qdrant), MCP brokerage, security proxy, and multi-backend support. Designed for sovereign Canadian AI stack on NixOS/K3s.

## Permanent Principles

### Nix-native
NixOS modules in `nix/` are the **single source of truth** for all deployment config. Do not edit k8s YAML, wrapper scripts, or other derivative files directly — CI/CD generates them from Nix. If you need to change how the gateway is deployed, change the module.

### Clean long-term solutions
Agents **must** fix root causes, not apply workarounds. Every change should be the cleanest solution that lasts — no TODOs, no half-measures, no "fix it later" debt.

### Gateway routing
All AI backend traffic routes through the gateway — circuit breakers, rate limiting, observability (Prometheus), and MCP brokerage depend on it. Never route a client directly to a backend (NIM, llama-cpp, vLLM, etc.).

## Model Routing

### Per-Request Routing (`"model": "auto"`)

Clients should send `"model": "auto"` in chat completion requests. The gateway handles per-request model selection:

| Request content | Routed to | Why |
|----------------|-----------|-----|
| Contains image or audio | Nemotron Omni 30B | Vision/multimodal support |
| Code, agentic patterns | Nemotron Super 120B | Complex reasoning |
| General queries | Nemotron Omni 30B | Efficient & capable |

Routing flow: `detect_specialization()` → `_generate_candidates()` → `_rank_candidates()` → `RouteDecision`

**Do not hardcode model names in client configs** — use `"model": "auto"` to let the gateway optimize routing.

### Kelos/TaskSpawner Routing (`model-routing-controller`)

For Kelos agent tasks, models are assigned per-task-type via the `kelos-model-routing` ConfigMap (namespace: `kelos-system`):

| Task Type | Primary Model | Fallback | Reliability |
|-----------|--------------|----------|-------------|
| default | Nemotron Nano 30B | Qwen 3 Coder 480B | gold |
| coding | Nemotron Super 120B | Qwen 3 Coder 480B | silver |
| analysis | Nemotron Nano 30B | Llama 3.3 70B | gold |
| reasoning | Nemotron Omni 30B | Super 120B | silver |
| batch | Nemotron Nano 30B | Qwen 3 Coder 480B | gold |
| urgent | Nemotron Super 120B | Nano 30B | silver |
| vision | Nemotron Omni 30B | Super 120B | bronze |
| documentation | Qwen 3 Coder 480B | Nano 30B | silver |
| ultra_context | Best 1M+ model | — | varies |
| emergency | vLLM qwen3.5-2b-awq (local) | llama.cpp 4B (local) | no-cloud |

**Components:**

| Component | What it does | Schedule |
|-----------|-------------|----------|
| `kelos-model-routing` ConfigMap | Routing rules per task type | Updated by controller + benchmark |
| `model-routing-controller` CronJob | Reads ConfigMap, patches TaskSpawner `taskTemplate.model` | Every 15min |
| Circuit breaker | Exponential backoff (15m/1h/6h/24h per model) | Persistent in ConfigMap |
| `model-benchmark-eval` CronJob | Benchmarks all 174 gateway models, auto-expands routing | Every 6h |

**Graceful degradation (7 layers):**

```
L1 Gateway:    DISCOVERY_BACKENDS + cloud fallback + overload detection
L2 Controller: Health-check models before assigning, degrade if missing
L3 Controller: Circuit breaker with exponential backoff
L4 Pod boot:   Verify MODEL exists in gateway, try fallback chain
L5 opencode:   All routing models in provider for auto-fallback
L6 env chain:  KELOS_MODEL → KELOS_MODEL_FALLBACKS → hardcoded default
L7 tiers:      gold > silver > bronze > degraded > emergency
```

**Circuit breaker states:**

| State | Meaning | Action |
|-------|---------|--------|
| CLOSED | Model healthy | Normal routing |
| HALF-OPEN | Cooldown expired | Try model once more |
| OPEN | Model failing repeatedly | Use fallback, exponential backoff |

Backoff: 1-2 failures → 15min, 3-5 → 1hr, 6-9 → 6hr, 10+ → 24hr

### Error Classification (benchmark v2)

The benchmark classifies each model failure:

| Type | Meaning | Circuit Breaker |
|------|---------|----------------|
| quota | 429/503 rate limited | Retry 1hr (resets) |
| auth | 401/403 invalid key | Retry 24hr (credential) |
| unavailable | 404 not found | Retry 24hr (removed) |
| timeout | >25s no response | Retry 1hr (transient) |
| error | 400/500/connection | Retry 6hr |

### Benchmark Data

Per-model results stored in `kelos-model-routing` ConfigMap `__benchmarks__`:

```json
{
  "model_id": {
    "avg_tok_s": 24.7,
    "std_tok_s": 4.4,
    "avg_lat_s": 4.0,
    "avg_ttft_ms": 350,
    "health": 61.9,
    "ctx": 262144
  }
}
```

Access: `kubectl get cm kelos-model-routing -n kelos-system -o json | jq '.__benchmarks__'`

### Anthropic API Support

The gateway has a `/v1/messages` endpoint (Anthropic Messages API format) with:
- Model mapping: Claude model names → gateway model IDs
- Thinking effort levels: low(5K), medium(15K), high(50K) budget_tokens
- PII/injection sanitization via MLSEC
- Routing through the same intelligent router

Current Claude → model mapping (hardcoded in `router.py`):

| Claude model | Routes to |
|-------------|-----------|
| claude-haiku-4 | qwen3.5-0.8b-distilled (local) |
| claude-sonnet-4 | qwen3.5-9b-distilled (local) |
| claude-opus-4 | qwen3.5-35b-a3b (local) |

To route Claude Code through NIM cloud models, update `claude_model_mapping` in `router.py`.

## Tech Stack

- **Language:** Python 3.11+ (developed on 3.13)
- **Framework:** FastAPI + Uvicorn
- **Package Manager:** Nix flake (primary), pyproject.toml (secondary)
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **Linting:** ruff (configured in pyproject.toml)

## Key Directories

| Path | Purpose |
|------|---------|
| `src/` | Python package (ai_inference_gateway) |
| `src/main.py` | FastAPI app + entry point |
| `src/middleware/` | Middleware pipeline (security, PII, rate-limit, knowledge fabric) |
| `src/rag/` | RAG engine (Qdrant, embeddings, hybrid search, semantic cache) |
| `src/mcp_servers/` | MCP server implementations |
| `src/services/` | Backend integrations (llama-cpp, vLLM, NIM, ZAI, etc.) |
| `tests/` | Test suite |
| `nix/` | NixOS module files — **source of truth for deployment** |
| `docs/` | Documentation |

## Backend Architecture

```
Client ("model": "auto") → Gateway (src/main.py)
  ├── detect_specialization(messages)
  │   ├── detect_vision_content() → Omni 30B
  │   ├── detect_code_patterns() → Super 120B
  │   └── default → Omni 30B
  ├── Middleware pipeline (security → PII → rate-limit → knowledge fabric)
  ├── MCP Broker (src/mcp_broker.py)
  ├── RAG Engine (Qdrant + Redis cache)
  └── Backend adapters (src/services/)
      ├── llama-cpp / vLLM (local GPU)
      ├── NIM (NVIDIA cloud)
      ├── Z.AI / Kilo (cloud fallback)
      └── Ollama (local CPU/GPU)
```

All traffic flows through circuit breakers and observability. **Never bypass the gateway.**

## Kelos Integration

The gateway's model routing and benchmark data feed into the Kelos task orchestration pipeline:

```
Kelos Issues → TaskSpawner (#38 routing) → Agent Pod → Gateway → Backend
                                                    ↓
                                              PR Created
                                                    ↓
                                        model-benchmark-eval (Phase 1)
                                        tok/s + TTFT + health scoring
                                                    ↓
                                        LLM-as-Judge eval (Phase 2, planned)
                                        PR quality scoring
                                                    ↓
                                        Routing optimizer (Phase 3, planned)
                                        Auto-adjust routing based on quality
```

See issue #60 (Code quality evaluation pipeline) for the full roadmap.

## Running

```bash
nix develop
PYTHONPATH=src python -m uvicorn ai_inference_gateway.main:app --port 8080
pytest tests/ -v
nix build .#container
```

## MCP Integration

The gateway runs an MCP broker managing connections to upstream MCP servers.
Currently connected: `searxng` (search), `maplespike` (AI Ask + Engine brief).

### Health Checks
```bash
curl http://localhost:8080/mcp/health/searxng
curl http://localhost:8080/mcp/health/maplespike
```

## Environment Variables

Core config: `BACKEND_URL`, `BACKEND_TYPE`, `GATEWAY_HOST`, `GATEWAY_PORT`, `RAG_ENABLED`, `QDRANT_URL`.
Full list in `src/config.py` and `nix/options.nix`.

## Related Issues

| # | Title | Status |
|---|-------|--------|
| #38 | Dynamic model routing for Kelos | ✅ Implemented |
| #60 | Code quality evaluation pipeline | Phase 1 deployed |
| #41 | EPIC: NIM Model Optimization Loop | In progress |

## Related Repositories

- [reverb256/maplespike](https://github.com/reverb256/maplespike) — Consumer for AI Ask and Engine briefs
- [reverb256/nixos-config](https://github.com/reverb256/nixos-config) — NixOS cluster config (kelos.nix module)
