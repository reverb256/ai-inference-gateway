# AGENTS.md — AI Agent Guidelines for ai-inference-gateway

## Project Context

OpenAI-compatible API gateway with:
- Circuit breakers and latency-aware load balancing
- RAG via Qdrant + semantic cache (Redis)
- MCP brokerage (connects to external MCP servers like SearXNG, MapleSpike)
- Security middleware (PII redaction, prompt injection scoring, rate limiting)
- NixOS module + K8s deployment
- Multiple backend support (llama-cpp, vLLM, SGLang, NIM, ZAI, Pollinations)

## Architecture

```
src/
├── ai_inference_gateway/
│   ├── main.py                          # FastAPI app + HTTP routes
│   ├── router.py                        # Intelligent model routing
│   ├── pipeline.py                      # Observability + circuit breaker
│   ├── config.py                        # Configuration management
│   ├── middleware/                      # Security, rate-limit, PII
│   ├── rag/                            # Qdrant hybrid search
│   ├── mcp_servers/                    # Local MCP server modules
│   │   ├── searxng_server.py           # SearXNG search MCP
│   │   └── maplespike_server.py        # MapleSpike AI Ask MCP
│   └── services/                       # Backend integrations
nix/                                    # NixOS module (options, gateway)
kubernetes/                             # K8s deployment manifests
tests/                                  # pytest suite
```

## Kelos Pipeline Verification

- Kelos test — 2026-05-18 — pipeline verified
- Kelos test — 2026-05-18 — pipeline verified (via PR #8)

## MCP Integration

The gateway runs an MCP broker that manages connections to upstream MCP servers.
Currently connected servers:
- `searxng` (local): Web search via SearXNG metasearch
- `maplespike` (remote): MapleSpike AI Ask + Engine brief + pipeline status

## Test Markers

- `test_security_filter.py`: Security middleware tests (PII, injection)
- `test_mlsec_phase2.py`: ML security Phase 2 tests (scorer, validation)
- `test_security_filter_concurrent_injection_detection`: Concurrent injection detection

## Quick Commands

```bash
# Run tests
pytest -v tests/

# Start dev server
uv run python -m ai_inference_gateway.main

# Check MCP server health
curl http://localhost:8080/mcp/health/searxng
curl http://localhost:8080/mcp/health/maplespike
```
