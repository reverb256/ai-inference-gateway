# AGENTS.md — AI Inference Gateway

# Kelos test - 2026-05-18 - pipeline verified

## Project Overview

Python FastAPI gateway providing OpenAI/Anthropic/Ollama-compatible API endpoints with intelligent routing, circuit breaker failover, RAG (Qdrant), MCP brokerage, security proxy, and multi-backend support. Designed for sovereign Canadian AI stack on NixOS/K3s.

## Permanent Principles

### Nix-native
NixOS modules in `nix/` are the **single source of truth** for all deployment config. Do not edit k8s YAML, wrapper scripts, or other derivative files directly — CI/CD generates them from Nix. If you need to change how the gateway is deployed, change the module.

### Clean long-term solutions
Agents **must** fix root causes, not apply workarounds. Every change should be the cleanest solution that lasts — no TODOs, no half-measures, no "fix it later" debt. If a task seems to require a workaround, stop and fix the underlying problem instead.

### Gateway routing
All AI backend traffic routes through the gateway — circuit breakers, rate limiting, observability (Prometheus), and MCP brokerage depend on it. Never route a client directly to a backend (NIM, llama-cpp, vLLM, etc.). If a backend format is incompatible (e.g. NIM tool-call message format), fix the gateway's request transformation layer, not the routing.

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
| `src/mcp_servers/` | MCP server implementations (SearXNG, MapleSpike) |
| `src/services/` | Backend integrations (llama-cpp, vLLM, NIM, ZAI, etc.) |
| `tests/` | Test suite |
| `nix/` | NixOS module files — **source of truth for deployment** |
| `kubernetes/` | Generated output — do not edit directly |
| `docs/` | Documentation |

## Backend Architecture

```
Agent/Client → Gateway (src/main.py)
  ├── Middleware pipeline (security → PII → rate-limit → knowledge fabric)
  ├── Router (src/router.py) — model selection + cost awareness
  ├── MCP Broker (src/mcp_broker.py)
  │   ├── SearXNG MCP (search)
  │   └── MapleSpike MCP (AI Ask, engine briefs)
  ├── RAG Engine (Qdrant + Redis cache)
  └── Backend adapters (src/services/)
      ├── llama-cpp / vLLM / SGLang (local GPU)
      ├── NIM (NVIDIA cloud)
      ├── ZAI / Pollinations (cloud fallback)
      └── Ollama (local CPU/GPU)
```

All traffic flows through the gateway's circuit breakers and observability pipeline. **Never bypass the gateway.**

## Running

```bash
# Dev shell
nix develop

# Direct (no Nix)
PYTHONPATH=src python -m uvicorn ai_inference_gateway.main:app --port 8080

# Tests
pytest tests/ -v

# Build container
nix build .#container
```

## NixOS Integration

The `nix/` directory contains NixOS module files. Import via `flake.nix`:

```nix
inputs.ai-inference-gateway.url = "github:reverb256/ai-inference-gateway";
```

Key modules:
- `nix/options.nix` — All config options (MCP servers, backends, security, RAG)
- `nix/gateway.nix` — Systemd service definition
- `nix/qdrant.nix` — Qdrant vector DB module
- `nix/config-assertions.nix` — Validation of option combinations

## Common Tasks for Agents

### Adding a new MCP server
1. Add the server module to `src/mcp_servers/`
2. Register it in `nix/options.nix` under `mcp.servers` default
3. No wrapper scripts — use `type = "remote"` if connecting via HTTP
4. Update `mcp_servers/__init__.py` to export the main function

### Adding a new backend
1. Create backend adapter in `src/services/`
2. Add model config in `nix/options.nix` under the provider's option
3. If the API format differs from OpenAI, add request transformation in the adapter
4. Update AGENTS.md with the new backend's requirements

### Fixing a backend compatibility issue
- The gateway is the transformation layer — fix message formatting here, not in clients
- Example: NIM expects `tool` role content as `[{type: "text", text: "..."}]` arrays, some clients send strings. Convert in the NIM service adapter, not by routing around it.

## Environment Variables

Core config via env vars: `BACKEND_URL`, `BACKEND_TYPE`, `GATEWAY_HOST`, `GATEWAY_PORT`, `RAG_ENABLED`, `QDRANT_URL`. Full list in `src/config.py` and `nix/options.nix`.

## MCP Integration

The gateway runs an MCP broker that manages connections to upstream MCP servers.
Currently connected servers:
- `searxng` (local): Web search via SearXNG metasearch
- `maplespike` (remote): MapleSpike AI Ask + Engine brief + pipeline status

### Health Checks

```bash
curl http://localhost:8080/mcp/health/searxng
curl http://localhost:8080/mcp/health/maplespike
```

## Test Markers

- `test_security_filter.py` — Security middleware tests (PII, injection)
- `test_mlsec_phase2.py` — ML security Phase 2 (scorer, validation)
- `test_concurrent` — Concurrent request tests
- `requires_redis` / `requires_qdrant` — Integration tests requiring external services

## Related Repositories

- [reverb256/maplespike](https://github.com/reverb256/maplespike) — Consumer of this gateway for AI Ask and Engine briefs
