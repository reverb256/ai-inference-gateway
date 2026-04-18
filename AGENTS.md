# AGENTS.md — AI Inference Gateway

## Project Overview

Python FastAPI gateway providing OpenAI/Anthropic/Ollama-compatible API endpoints with intelligent routing, circuit breaker, RAG, MCP brokerage, and multi-backend support.

Extracted from `/etc/nixos/modules/services/ai-inference/` into a standalone project.

## Tech Stack

- **Language:** Python 3.11+ (developed on 3.13)
- **Framework:** FastAPI + Uvicorn
- **Package Manager:** pip/setuptools (pyproject.toml) or Nix (flake.nix)
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **Linting:** ruff

## Key Directories

| Path | Purpose |
|------|---------|
| `src/` | Python package `ai_inference_gateway` |
| `src/main.py` | FastAPI app + entry point |
| `src/middleware/` | Middleware pipeline components |
| `src/middleware/knowledge_fabric/` | Multi-source knowledge synthesis |
| `src/rag/` | RAG engine (Qdrant, embeddings, hybrid search) |
| `src/routes/` | API route handlers (admin, virtual keys) |
| `src/services/` | Business logic (Anthropic service, cost tracker, virtual keys) |
| `src/mcp_servers/` | MCP server implementations (SearXNG) |
| `tests/` | Test suite |
| `nix/` | NixOS module files (systemd services, options) |

## Running

```bash
# Dev shell
nix develop

# Direct
PYTHONPATH=src python -m uvicorn ai_inference_gateway.main:app --port 8080

# Tests
pytest tests/
```

## Environment Variables

Core config via env vars: `BACKEND_URL`, `BACKEND_TYPE`, `GATEWAY_HOST`, `GATEWAY_PORT`, `RAG_ENABLED`, `QDRANT_URL`.

Full list in `src/config.py` and `nix/options.nix`.

## NixOS Integration

The `nix/` directory contains NixOS module files for deploying as a systemd service. Import via `flake.nix`:

```nix
inputs.ai-inference-gateway.url = "path:/data/projects/own/ai-inference-gateway";
# Then use: ai-inference-gateway.nixosModules.default
```

## Important Notes

- **Extracted** from `/etc/nixos/modules/services/ai-inference/` into standalone project. Wired as `ai-gateway` flake input in `/etc/nixos/flake.nix`.
- **112 Python files** across middleware, routing, RAG, services, and MCP.
- **Docker container** buildable via `nix build .#container`.
- Tests require Redis and Qdrant for integration tests (marked `requires_redis`, `requires_qdrant`).
