# Autonomous Model Selection System

> **Status: SUPERSEDED for per-request routing**
>
> The benchmark/ranking infrastructure in this document is still valid for cost comparison and model ranking.
> However, **per-request model selection is now handled by `"model": "auto"`** via `detect_specialization()` —
> the gateway routes based on request content (vision, code, general) rather than per-task labels.
>
> See `AGENTS.md` → "Intelligent Model Routing" for the current approach.
> Issue #78 and PR #81 were closed in favor of this simpler, more powerful solution.

## Overview

The AI Gateway has an **autonomous model selection system** that tests, ranks, and automatically selects the best model for each request based on real performance data - not advertised specs.

## What It Does

### 1. Discovers Real Capabilities
Unlike advertised specs, the system tests actual performance:
- **TTFT** (Time To First Token) - How fast until first token appears
- **Throughput** - Tokens per second (including prefill)
- **Context Window** - Actual limit (found via binary search)
- **Concurrency** - Max parallel requests before errors
- **Rate Limits** - Requests per minute before 429s

### 2. Ranks Models
Scores models 0-100 based on:
- Speed (TTFT + throughput)
- Context window size
- Concurrency capacity
- Cost tier (prefer free)

### 3. Autonomous Selection
Automatically picks the best model for each request considering:
- Required context size
- Speed requirements
- Concurrency needs
- Cost constraints
- Task specialization

## API Endpoints

### Get Benchmark Results
```bash
curl http://localhost:8080/models/benchmark
```

Returns benchmark data for all tested models:
```json
{
  "total_models": 5,
  "results": {
    "qwen3.5-2b-awq": {
      "backend": "vllm-local",
      "success": true,
      "timestamp": "2026-05-01T15:30:00",
      "metrics": {
        "ttft_ms": 15.2,
        "throughput_tps": 564.0,
        "context_window": 32768,
        "concurrency_limit": 32,
        "rate_limit_rpm": 0
      }
    }
  }
}
```

### Get Ranked Models
```bash
curl "http://localhost:8080/models/rankings?min_context=10000&max_ttft_ms=100"
```

Filter by requirements:
- `min_context`: Minimum context window
- `max_ttft_ms`: Maximum acceptable TTFT
- `min_throughput_tps`: Minimum throughput

Returns ranked list:
```json
{
  "total_ranked": 5,
  "requirements": {"min_context": 10000, "max_ttft_ms": 100},
  "rankings": [
    {
      "model_id": "qwen3.5-2b-awq",
      "rank": 1,
      "score": 95.2,
      "strengths": ["Ultra-fast TTFT", "High throughput", "High concurrency"],
      "weaknesses": [],
      "best_for": ["fast-chat", "realtime", "batch", "high-volume"],
      "avg_ttft_ms": 15.2,
      "avg_throughput_tps": 564.0,
      "actual_context_window": 32768
    }
  ]
}
```

### Get Recommendations
```bash
curl http://localhost:8080/models/recommendations/coding
```

### Start Auto-Benchmark
```bash
curl -X POST http://localhost:8080/admin/benchmark/start \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      {
        "model_id": "qwen3.5-2b-awq",
        "backend": "vllm-local",
        "url": "http://10.1.1.110:8040/v1",
        "api_key": null
      }
    ]
  }'
```

## How It Works

### 1. Baseline Performance Test
Sends a simple prompt, measures TTFT, throughput, and total time.

### 2. Context Window Discovery
Binary search to find actual limit (32K, 64K, 128K...).

### 3. Concurrency Testing
Parallel requests to find concurrency limit.

### 4. Rate Limit Discovery
Rapid requests to find RPM limit.

## Ranking Algorithm

Base score: 50. Bonuses for TTFT < 100ms (+20), throughput > 100 tps (+20), context > 100K (+15), concurrency > 16 (+15).

## Integration with Router

The router uses benchmark data in `_generate_candidates()` and `_rank_candidates()` to make routing decisions when `"model": "auto"` triggers specialization detection.

## Future Enhancements
1. Quality Scoring
2. Cost Optimization
3. Adaptive Selection
4. Continuous Benchmarking
5. A/B Testing
6. ML-based Ranking

## Metrics Reference

| Metric | Description | Good | Excellent |
|--------|-------------|------|-----------|
| TTFT_MS | Time to first token | < 500ms | < 100ms |
| THROUGHPUT_TPS | Tokens per second | > 50 | > 100 |
| CONTEXT_WINDOW | Actual context limit | > 32K | > 100K |
| CONCURRENCY_LIMIT | Max parallel requests | > 4 | > 16 |

---

**Created:** 2026-05-01
**Updated:** 2026-05-19 — Added supersession note for `"model": "auto"` routing
