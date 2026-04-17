# AI Inference Gateway — Load Testing

## Quick Start

```bash
# Install locust
pip install locust

# Run with Web UI (recommended)
locust -f locustfile.py --host=http://localhost:8080

# Run headless (CI/benchmarking)
locust -f locustfile.py --host=http://localhost:8080 \
  --headless -u 50 -r 10 -t 60s --csv=results

# Run against K8s gateway
locust -f locustfile.py --host=http://ai-inference-gateway.ai-inference.svc.cluster.local:8080
```

## Test Scenarios

| User Type | Weight | Endpoint | Description |
|-----------|--------|----------|-------------|
| ChatCompletionsUser | 50% | /v1/chat/completions | OpenAI API (simple/coding/reasoning/streaming) |
| AnthropicMessagesUser | 30% | /v1/messages | Anthropic API (haiku/sonnet/opus) |
| HealthCheckUser | 10% | /health, /v1/models | Monitoring probes |

## Benchmark Matrix

Run each scenario and record results:

| Scenario | Concurrency | RPS | p50 Latency | p99 Latency | Notes |
|----------|-------------|-----|-------------|-------------|-------|
| Passthrough (no middleware) | 1 | - | - | - | Baseline |
| Full middleware | 10 | - | - | - | |
| Knowledge Fabric | 10 | - | - | - | |
| Streaming SSE | 10 | - | - | - | TTFB + tokens/s |
| Saturation test | 100 | - | - | - | Find ceiling |
