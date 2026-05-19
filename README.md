# AI Inference Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Nix Flake](https://img.shields.io/badge/Nix-Flake-5277C3?logo=nixos&logoColor=white)](flake.nix)
[![Built with Nix](https://img.shields.io/badge/Built%20with-Nix-5277C3?logo=nixos)](https://nixos.org)

OpenAI-compatible API gateway with intelligent routing, circuit breaker failover, security proxy, RAG, and MCP brokerage. Designed for sovereign Canadian AI stack on NixOS/K3s.

## Architecture

```
Client ("model": "auto") → Gateway (Port 8080)
  ├── detect_specialization(messages)
  │   ├── detect_vision_content() → Nemotron Omni 30B
  │   ├── detect_code_patterns() → Nemotron Super 120B
  │   └── default → Nemotron Omni 30B
  ├── Security Layer (rate limit, PII, injection scoring)
  ├── Intelligent Router (model ranking, cost awareness)
  ├── MCP Broker (SearXNG, MapleSpike)
  ├── RAG Engine (Qdrant + Redis cache)
  └── Backend Pool (NIM, llama-cpp, vLLM, ZAI, Pollinations)
```

All AI traffic flows through circuit breakers and observability. **Never bypass the gateway.**

## Intelligent Model Routing

Clients send `"model": "auto"` and the gateway selects the best backend per-request.

| Request content | Routed to | Why |
|----------------|-----------|-----|
| Contains images/audio | Nemotron Omni 30B | Vision/multimodal |
| Code, agentic patterns | Nemotron Super 120B | Complex reasoning |
| General queries | Nemotron Omni 30B | Efficient & capable |

**Do not hardcode model names in client configs** — use `"model": "auto"` to let the gateway optimize routing.

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **OpenAI-Compatible API** | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` | ✅ |
| **Anthropic API** | `/v1/messages` with Claude model mapping | ✅ |
| **Ollama-Compatible** | `/api/chat` for Spacebot integration | ✅ |
| **Intelligent Router** | Model specialization, latency-aware routing | ✅ |
| **Auto Model Selection** | `"model": "auto"` routes by content type | ✅ |
| **Circuit Breaker** | Prevents cascading failures, auto-recovery | ✅ |
| **Load Balancer** | Weighted round-robin backend selection | ✅ |
| **Security Filter** | Rate limiting, PII redaction, injection scoring | ✅ |
| **Semantic Caching** | Redis + Qdrant vector cache for deduplication | ✅ |
| **RAG** | Qdrant vector DB with hybrid search (vector + BM25) | ✅ |
| **MCP Broker** | Tool aggregation from multiple MCP servers | ✅ |
| **MCP Servers** | SearXNG search, MapleSpike AI Ask & briefs | ✅ |
| **Prometheus Metrics** | Full observability with Grafana dashboards | ✅ |
| **Content Moderation** | Jailbreak, violence, self-harm detection | ✅ |
| **JSON Schema Mode** | OpenAI JSON mode compatibility | ✅ |
| **Container Image** | Nix-built Docker container | ✅ |
| **NixOS Module** | Full NixOS service configuration | ✅ |

## Permanent Principles

### Nix-native
NixOS modules in `nix/` are the single source of truth for all deployment config. Do not edit k8s YAML, wrapper scripts, or other derivative files directly — CI/CD generates them from Nix.

### Gateway routing
All AI backend traffic routes through the gateway — circuit breakers, rate limiting, observability, and MCP brokerage depend on it. If a backend format is incompatible, fix the gateway's request transformation layer, not the routing.

### Clean long-term solutions
Fix root causes, not symptoms. No workarounds, no TODOs, no "fix it later" debt.

## Quick Start

### Run with Python
```bash
pip install -e ".[dev]"
python -m uvicorn ai_inference_gateway.main:app --host 0.0.0.0 --port 8080
```

### Run with Nix
```bash
nix develop   # Dev shell with all dependencies
nix build     # Build the package
nix build .#container  # Build container image
```

### Run as NixOS Service
```nix
{
  inputs.ai-inference-gateway.url = "github:reverb256/ai-inference-gateway";

  outputs = { nixpkgs, ai-inference-gateway, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        ai-inference-gateway.nixosModules.default
        {
          services.ai-inference = {
            enable = true;
            backend.url = "http://127.0.0.1:1234";
            backend.type = "llama-cpp";
            gateway = { enable = true; host = "127.0.0.1"; port = 8080; };
          };
        }
      ];
    };
  };
}
```

## API Endpoints

### Chat Completions (use `"model": "auto"` for intelligent routing)
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Anthropic Messages API
```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Explain NixOS."}]
  }'
```

### Models & Health
```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/metrics
```

### MCP Broker
```bash
curl http://127.0.0.1:8080/mcp/servers
curl http://127.0.0.1:8080/mcp/tools
curl -X POST http://127.0.0.1:8080/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"server": "searxng", "tool": "search", "arguments": {"query": "test"}}'
```

## Configuration

Core config via environment variables. Full options in `nix/options.nix`.

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://127.0.0.1:1234` | Primary backend API URL |
| `BACKEND_TYPE` | `llama-cpp` | Backend type |
| `GATEWAY_HOST` | `127.0.0.1` | Listen address |
| `GATEWAY_PORT` | `8080` | Listen port |
| `RAG_ENABLED` | `false` | Enable RAG with Qdrant |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant URL |

## Python Client
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "Explain NixOS in one sentence."}]
)
print(response.choices[0].message.content)
```

## Testing
```bash
pytest                              # All tests
pytest --cov --cov-report=term-missing  # With coverage
pytest tests/test_circuit_breaker.py -v  # Specific test
```

## Project Structure
```
├── src/                    # Python source
│   ├── main.py             # FastAPI entry point
│   ├── router.py           # Intelligent routing (model: auto)
│   ├── pipeline.py         # Middleware pipeline
│   ├── middleware/         # Security, rate-limit, PII, knowledge fabric
│   ├── routes/             # API route handlers
│   ├── rag/                # Qdrant + hybrid search
│   ├── mcp_servers/        # SearXNG, MapleSpike MCP servers
│   └── services/           # Backend adapters (NIM, vLLM, ZAI, etc.)
├── tests/                  # Test suite
├── nix/                    # NixOS module — source of truth
│   ├── options.nix         # All config options
│   ├── gateway.nix         # Systemd service
│   └── qdrant.nix          # Qdrant service
├── pyproject.toml
├── flake.nix
└── AGENTS.md               # Agent guidelines & routing docs
```

## Related Repositories
- [reverb256/maplespike](https://github.com/reverb256/maplespike) — AI Ask & Engine briefs
- [reverb256/nixos-config](https://github.com/reverb256/nixos-config) — Cluster deployment

## License
MIT
