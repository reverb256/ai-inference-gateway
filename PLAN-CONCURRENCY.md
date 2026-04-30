# GPU Concurrency Control Plan

**Problem:** 6 Hermes instances fire requests at 3 GPU backends. When 2+ requests land on the same GPU, one queues (llama-server `--parallel 1`) and the instance falls behind. No round-robin or "skip busy" logic exists.

**Goal:** Each GPU processes exactly 1 request at a time. When busy, the gateway picks the next idle GPU. If all GPUs busy, return 503 immediately (no queueing).

---

## Current State

### 3 Physical Backends
| Backend | Model | GPU | K8s DNS | Port |
|---------|-------|-----|---------|------|
| llama-3060ti | Qwen3.5-9B IQ3_M | RTX 3060 Ti 8GB | llama-server-zephyr-3060ti.ai-inference.svc.cluster.local | 1236 |
| llama-3090 | Qwen3.6-35B-A3B IQ3_S | RTX 3090 24GB | llama-server-zephyr-3090-moe.ai-inference.svc.cluster.local | 1237 |
| llama-sentry | Qwen3.5-4B Q4_K_M | RX 5600 XT 8GB | llama-server-sentry.ai-inference.svc.cluster.local | 1235 |

### Existing Code
- `ModelDiscovery` (model_discovery.py) — Already knows about llama-3060ti and llama-sentry backends via K8s DNS. **Missing: llama-3090** (was removed when scaled down). Maps model_id → backend_name.
- `Router` (router.py) — Has `active_requests` dict tracking `{request_id: {model, backend, stream, start_time}}` and `max_concurrent_streams = 1`. But capacity check only looks at generic `"llama-cpp"` backend, not per-instance.
- `LoadBalancerMiddleware` (load_balancer.py) — Full weighted round-robin with per-backend `max_concurrent_requests`, health checks, connection tracking. **Disabled** (`enabled=False`).
- `forward_request_with_fallback()` (main.py) — Linear fallback chain: primary → fallback URLs. No concurrency awareness.

### Research Findings
1. **llama-server `--parallel 1`** — Already set (1 decode slot). Second request queues internally. This is correct for per-GPU isolation.
2. **LiteLLM** — Has `max_parallel_requests` per deployment. Uses a counter per deployment; rejects/cooldowns when at limit. Well-tested pattern.
3. **asyncio.Semaphore** — Python-native way to limit concurrency per backend. Simple, no external deps.
4. **Weighted round-robin** — Nginx-style smooth weighted RR already implemented in LoadBalancerMiddleware.

---

## Plan

### Phase 1: Per-Backend Concurrency Semaphore (core fix)

Add an `asyncio.Semaphore(1)` per physical backend instance in the Router.

**File: `src/router.py`**

1. Replace the generic `active_requests` tracking with per-backend semaphores:
   ```python
   self.backend_semaphores: Dict[str, asyncio.Semaphore] = {}
   # Populated from ModelDiscovery backends + static config
   for name in ["llama-3060ti", "llama-3090", "llama-sentry"]:
       self.backend_semaphores[name] = asyncio.Semaphore(1)
   ```

2. Add `async def acquire_backend(backend_name) -> bool`:
   - Try `semaphore.acquire()` with zero timeout (`asyncio.wait_for` with timeout=0)
   - Returns True if acquired, False if busy
   - Never blocks — instant accept or reject

3. Add `def release_backend(backend_name)`:
   - `semaphore.release()`

4. Update routing logic in `route_request()`:
   - When routing to a local backend, check `acquire_backend()` first
   - If busy, try next local backend (round-robin)
   - If all local backends busy, fall through to cloud (ZAI/NIM) or return 503

### Phase 2: Add 3090 Backend to ModelDiscovery

**File: `src/model_discovery.py`**

Add the missing backend:
```python
"llama-3090": BackendInfo(
    name="llama-3090",
    base_url="http://llama-server-zephyr-3090-moe.ai-inference.svc.cluster.local:1237/v1",
    priority=11,  # Highest — 35B model on 24GB GPU
),
```

### Phase 3: Round-Robin Among Same-Tier Backends

**File: `src/router.py`**

1. Add a `_rr_index: int` counter for round-robin selection
2. When model is unspecified or "any local", select from idle backends in round-robin order:
   ```
   candidates = [b for b in backends if semaphore_available(b)]
   if candidates:
       selected = candidates[_rr_index % len(candidates)]
       _rr_index += 1
   ```

3. When model IS specified, check if that specific backend is free:
   - Free → acquire, route
   - Busy → return 503 with `Retry-After` header (client should retry or use cloud fallback)

### Phase 4: Wire Into Request Lifecycle

**File: `src/main.py`**

In the chat completions handler:
1. **Before** forwarding: `await router.acquire_backend(backend_name)`
2. **After** response completes (success or error): `router.release_backend(backend_name)`
3. Use `try/finally` to guarantee release even on exceptions
4. For streaming: release in the `finally` block of the `StreamingResponse` generator

### Phase 5: 503 Response + Client Retry

1. When all local backends busy, return HTTP 503 with:
   ```json
   {"error": "all_local_backends_busy", "retry_after": 2}
   ```
   Plus `Retry-After: 2` header.

2. Hermes instances already handle provider fallback — 503 triggers fallback to cloud (ZAI/NIM). This is the desired behavior: saturate local GPUs first, overflow to cloud.

### Phase 6: Admin Endpoint

Add `/admin/backends` endpoint returning:
```json
{
  "llama-3060ti": {"url": "...", "active": false, "model": "Qwen3.5-9B-IQ3_M.gguf"},
  "llama-3090": {"url": "...", "active": true, "model": "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"},
  "llama-sentry": {"url": "...", "active": false, "model": "Qwen3.5-4B-Q4_K_M.gguf"}
}
```

---

## What NOT To Do

- **Don't enable LoadBalancerMiddleware** — It operates at the middleware layer, not the routing layer. It would conflict with the existing Router logic. The semaphore approach in Router is simpler and hooks into the existing request tracking.
- **Don't use `limit_conn` nginx sidecar** — Adds complexity, doesn't integrate with gateway's model-aware routing.
- **Don't increase `--parallel` on llama-server** — That allows batched concurrent inference, which shares VRAM and slows both requests. The whole point is 1 GPU = 1 request.

---

## Implementation Order

1. Add llama-3090 to `ModelDiscovery.BACKENDS` (1 line)
2. Add `backend_semaphores` dict to `Router.__init__()` (5 lines)
3. Add `acquire_backend()` / `release_backend()` methods (15 lines)
4. Wire acquire/release into `main.py` chat completions handler (20 lines)
5. Update routing logic for round-robin + busy-skip (30 lines)
6. Add `/admin/backends` endpoint (15 lines)

**Estimated: ~85 lines of changes across 3 files. No new dependencies.**

---

## Testing

1. Send 3 concurrent requests → each goes to a different backend
2. Send 4th request → instant 503
3. Wait for 1 to finish → 4th succeeds on the freed backend
4. Check `/admin/backends` shows correct active/idle state
5. Verify streaming responses release the semaphore on completion
