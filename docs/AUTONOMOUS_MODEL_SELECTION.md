# Autonomous Model Selection System

## Overview

The AI Gateway now has an **autonomous model selection system** that tests, ranks, and automatically selects the best model for each request based on real performance data - not advertised specs.

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

Task types: `fast-chat`, `coding`, `long-context`, `rag`, `analysis`, `batch`, `high-volume`

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
Sends a simple prompt, measures:
- Time to first token (TTFT)
- Total tokens generated
- Total time
- Calculates throughput

### 2. Context Window Discovery
Binary search to find actual limit:
- Start with small prompt (1000 tokens)
- Progressive doubling: 32K, 64K, 128K...
- Stop when error occurs
- Returns largest successful size

### 3. Concurrency Testing
Parallel requests to find limit:
- Start with 1 concurrent request
- Increase by 4: 1, 5, 9, 13, 17...
- Stop when >20% fail
- Returns concurrency limit

### 4. Rate Limit Discovery
Rapid requests for 30 seconds:
- Send requests as fast as possible
- Count until 429 appears
- Calculate RPM

## Ranking Algorithm

Base score: 50

**TTFT Scoring:**
- < 100ms: +20
- < 500ms: +10
- < 1000ms: +5

**Throughput Scoring:**
- > 100 tps: +20
- > 50 tps: +10
- > 20 tps: +5

**Context Scoring:**
- > 100K: +15
- > 32K: +10
- > 8K: +5

**Concurrency Scoring:**
- > 16: +15
- > 8: +10
- > 4: +5

**Requirements Filter:**
- Below minimum context: Score = 0 (disqualified)
- Above max TTFT: Score × 0.5 (penalty)

## Storage

Benchmark results cached to: `/tmp/model_benchmarks.json`

Persists across gateway restarts. Re-run benchmarks to update.

## Usage Example

```python
from ai_inference_gateway.router import Router
from ai_inference_gateway.model_benchmark import get_benchmark, get_selector

# Get benchmark instance
benchmark = get_benchmark()

# Get selector
selector = get_selector()

# Select best model for requirements
requirements = {
    "estimated_input_tokens": 5000,
    "estimated_output_tokens": 1000,
    "max_ttft_ms": 500,
    "min_throughput_tps": 50,
}

best_model = await selector.select_best_model(
    requirements=requirements,
    available_models=list(router.models.keys())
)
```

## Model Categories by Performance

### Ultra-Fast (TTFT < 100ms, >100 tps)
- Best for: Chat, real-time applications
- Examples: qwen3.5-2b-awq (564 tps)

### Fast (TTFT < 500ms, >50 tps)
- Best for: Coding, general tasks
- Examples: qwen3.5-4b, qwen3.5-9b

### Balanced (TTFT < 1000ms, >20 tps)
- Best for: Analysis, document processing
- Examples: qwen3.5-27b

### Slow (TTFT > 1000ms, <20 tps)
- Best for: Complex reasoning, large context
- Examples: qwen3.5-35b-a3b

## Integration with Router

The router now has autonomous selection methods:

```python
# Get rankings
rankings = router.get_model_rankings(
    requirements={"min_context": 10000}
)

# Get recommendations
recommendations = router.get_model_recommendations("coding")

# Select by benchmark
best_model = await router.select_model_by_benchmark(
    requirements={
        "estimated_input_tokens": 10000,
        "max_ttft_ms": 200,
    }
)
```

## Future Enhancements

1. **Quality Scoring** - Compare outputs to ground truth
2. **Cost Optimization** - Pick cheapest suitable model
3. **Adaptive Selection** - Learn from user feedback
4. **Continuous Benchmarking** - Periodic re-benchmarking
5. **A/B Testing** - Compare models on real requests
6. **ML-based Ranking** - Train model on performance data

## Metrics Reference

| Metric | Description | Good | Excellent |
|--------|-------------|------|-----------|
| TTFT_MS | Time to first token | < 500ms | < 100ms |
| THROUGHPUT_TPS | Tokens per second | > 50 | > 100 |
| CONTEXT_WINDOW | Actual context limit | > 32K | > 100K |
| CONCURRENCY_LIMIT | Max parallel requests | > 4 | > 16 |
| RATE_LIMIT_RPM | Requests per minute | > 60 | > 300 |
| QUALITY_SCORE | Output quality (0-100) | > 70 | > 90 |
| ERROR_RATE | Failure rate (0-1) | < 0.05 | < 0.01 |

---

**Created:** 2026-05-01  
**Status:** ALPHA - Testing phase  
**Next:** Run benchmarks on all configured models
