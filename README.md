# AI Inference Gateway

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Nix Flake](https://img.shields.io/badge/Nix-Flake-5277C3?logo=nixos&logoColor=white)](flake.nix)
[![Built with Nix](https://img.shields.io/badge/Built%20with-Nix-5277C3?logo=nixos)](https://nixos.org)


OpenAI-compatible API gateway with intelligent routing, circuit breaker failover, security proxy, RAG, and MCP brokerage.

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────┐
│   Client    │────▶│         AI Gateway v2 (Port 8080)        │
└─────────────┘     │  ┌─────────────────────────────────────┐ │
                    │  │  Security Layer                     │ │
                    │  │  - Rate limiting                    │ │
                    │  │  - Input sanitization               │ │
                    │  │  - PII redaction                    │ │
                    │  └──────────────┬──────────────────────┘ │
                    │                 │                         │
                    │  ┌──────────────▼──────────────────────┐ │
                    │  │  Intelligent Router                 │ │
                    │  │  - Model specialization             │ │
                    │  │  - Latency-aware routing            │ │
                    │  │  - Claude model mapping             │ │
                    │  └──────────────┬──────────────────────┘ │
                    │                 │                         │
                    │  ┌──────────────▼──────────────────────┐ │
                    │  │  Middleware Pipeline                 │ │
                    │  │  - Observability                    │ │
                    │  │  - Circuit breaker                  │ │
                    │  │  - Load balancer                    │ │
                    │  │  - Knowledge fabric                 │ │
                    │  └──────────────┬──────────────────────┘ │
                    │                 │                         │
                    │  ┌──────────────▼──────────────────────┐ │
                    │  │  Backend Pool                       │ │
                    │  │  - LM Studio / llama-cpp (local)    │ │
                    │  │  - ZAI (Zhipu AI cloud)             │ │
                    │  │  - NVIDIA NIM                       │ │
                    │  │  - Pollinations AI                  │ │
                    │  └─────────────────────────────────────┘ │
                    └──────────────────────────────────────────┘
```

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **OpenAI-Compatible API** | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` | ✅ |
| **Anthropic API** | `/v1/messages` with Claude model mapping | ✅ |
| **Ollama-Compatible** | `/api/chat` for Spacebot integration | ✅ |
| **Intelligent Router** | Model specialization, latency-aware routing | ✅ |
| **Circuit Breaker** | Prevents cascading failures, auto-recovery | ✅ |
| **Load Balancer** | Weighted round-robin backend selection | ✅ |
| **Security Filter** | Rate limiting, PII redaction, input sanitization | ✅ |
| **Semantic Caching** | Redis + Qdrant vector cache for deduplication | ✅ |
| **RAG** | Qdrant vector DB with hybrid search (vector + BM25) | ✅ |
| **MCP Broker** | Tool aggregation from multiple MCP servers | ✅ |
| **Knowledge Fabric** | Multi-source knowledge synthesis middleware | ✅ |
| **Prometheus Metrics** | Full observability with Grafana dashboards | ✅ |
| **Content Moderation** | Jailbreak, violence, self-harm detection | ✅ |
| **JSON Schema Mode** | OpenAI JSON mode compatibility | ✅ |
| **Container Image** | Nix-built Docker container | ✅ |
| **NixOS Module** | Full NixOS service configuration | ✅ |

## Quick Start

### Run with Python

```bash
# Install dependencies
pip install -e ".[dev]"

# Run the gateway
python -m uvicorn ai_inference_gateway.main:app --host 0.0.0.0 --port 8080
```

### Run with Nix

```bash
# Dev shell with all dependencies
nix develop

# Build the package
nix build

# Build container image
nix build .#container
```

### Run as NixOS Service

```nix
{
  inputs.ai-inference-gateway.url = "path:/data/projects/own/ai-inference-gateway";

  outputs = { nixpkgs, ai-inference-gateway, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [
        ai-inference-gateway.nixosModules.default
        {
          services.ai-inference = {
            enable = true;
            backend = {
              url = "http://127.0.0.1:1234";
              type = "llama-cpp";
            };
            gateway = {
              enable = true;
              host = "127.0.0.1";
              port = 8080;
            };
          };
        }
      ];
    };
  };
}
```

## API Endpoints

### Chat Completions (OpenAI-compatible)
```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b",
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

Configuration is via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URL` | `http://127.0.0.1:1234` | Primary backend API URL |
| `BACKEND_TYPE` | `llama-cpp` | Backend type (llama-cpp, vllm, sglang, zai, pollinations) |
| `GATEWAY_HOST` | `127.0.0.1` | Listen address |
| `GATEWAY_PORT` | `8080` | Listen port |
| `RAG_ENABLED` | `false` | Enable RAG with Qdrant |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant URL |
| `RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `CIRCUIT_BREAKER_ENABLED` | `true` | Enable circuit breaker |
| `SECURITY_ENABLED` | `true` | Enable security filter |
| `PII_REDACTION` | `true` | Enable PII redaction |

For full NixOS configuration options, see `nix/options.nix`.

## Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="dummy",
)

response = client.chat.completions.create(
    model="qwen3.5-4b",
    messages=[{"role": "user", "content": "Explain NixOS in one sentence."}]
)
print(response.choices[0].message.content)
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ai_inference_gateway --cov-report=term-missing

# Run specific test
pytest tests/test_circuit_breaker.py -v

# Run only unit tests
pytest -m unit
```

## Project Structure

```
├── src/                          # Python source (ai_inference_gateway package)
│   ├── main.py                   # FastAPI app entry point
│   ├── config.py                 # Configuration management
│   ├── router.py                 # Intelligent model routing
│   ├── pipeline.py               # Middleware pipeline
│   ├── middleware/                # Middleware components
│   │   ├── circuit_breaker.py
│   │   ├── load_balancer.py
│   │   ├── rate_limiter.py
│   │   ├── security_filter.py
│   │   ├── observability.py
│   │   └── knowledge_fabric/     # Multi-source knowledge synthesis
│   ├── routes/                   # API route handlers
│   ├── rag/                      # RAG engine (Qdrant, embeddings, search)
│   ├── services/                 # Business logic (Anthropic, virtual keys, cost tracking)
│   ├── mcp_servers/              # MCP server implementations
│   └── utils/                    # Shared utilities
├── tests/                        # Test suite
├── nix/                          # NixOS module files
│   ├── default.nix               # Module entry point
│   ├── options.nix               # Configuration options
│   ├── gateway.nix               # Gateway systemd service
│   ├── router.nix                # Token estimator
│   ├── qdrant.nix                # Qdrant service
│   ├── redis-cache.nix           # Redis cache
│   └── auth/                     # Authentication (Tailscale, API keys)
├── pyproject.toml                # Python package metadata
├── flake.nix                     # Nix flake (package, NixOS module, dev shell)
└── pytest.ini                    # Test configuration
```

## License

MIT
