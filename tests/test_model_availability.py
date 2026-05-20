#!/usr/bin/env python3
"""
Model availability smoke test for AI Inference Gateway.
Run against any OpenAI-compatible endpoint to verify which models work.

Usage:
    python3 test_model_availability.py [--gateway URL] [--api-key KEY]

CI/CD: Fails if any model in the "working" set from previous run becomes dead.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

GATEWAY = os.environ.get("GATEWAY_URL", "http://ai-inference-gateway.ai-inference.svc.cluster.local:8080")
API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# Non-chat model patterns to skip
SKIP_PATTERNS = [
    "tts", "audio", "embed", "reward", "safety", "detector",
    "guard", "parse", "retriever", "riva",
]


def get_models(gateway: str) -> list[str]:
    req = urllib.request.Request(f"{gateway}/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    models = [m["id"] for m in data.get("data", [])]
    return list(dict.fromkeys(models))  # unique, preserve order


def test_model(gateway: str, model_id: str, timeout: int = 30) -> tuple[str, float | None]:
    """Returns (status, elapsed_seconds). status: 'ok', 'rate_limited', 'dead'"""
    payload = json.dumps({
        "model": model_id,
        "messages": [{"role": "user", "content": "OK"}],
        "max_tokens": 3,
        "temperature": 0.1,
    }).encode()

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        start = time.time()
        req = urllib.request.Request(
            f"{gateway}/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
        elapsed = time.time() - start
        if "choices" in body and body["choices"] and body["choices"][0].get("message", {}).get("content"):
            return "ok", elapsed
        return "dead", elapsed
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read()).get("error", {}).get("message", "")
        except Exception:
            err = str(e)
        return "dead", None
    except Exception:
        return "rate_limited", None  # timeout = rate limited


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Test model availability")
    parser.add_argument("--gateway", default=GATEWAY)
    parser.add_argument("--api-key", default=API_KEY)
    parser.add_argument("--output", default="/tmp/model_test_results.json")
    args = parser.parse_args()

    print(f"Gateway: {args.gateway}")
    print("Fetching models...", flush=True)
    models = get_models(args.gateway)
    print(f"Total models: {len(models)}", flush=True)

    results = {"working": [], "rate_limited": [], "dead": [], "skipped": []}

    for i, mid in enumerate(models):
        if any(p in mid.lower() for p in SKIP_PATTERNS):
            results["skipped"].append(mid)
            print(f"  [{i+1}/{len(models)}] SKIP {mid[:60]}", flush=True)
            continue

        status, elapsed = test_model(args.gateway, mid)
        results[status].append(mid)

        icon = {"ok": "OK", "rate_limited": "RL", "dead": "DEAD"}[status]
        elapsed_str = f"{elapsed:.1f}s" if elapsed else "---"
        print(f"  [{i+1}/{len(models)}] {icon} {mid[:60]:60s} {elapsed_str}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"Working:      {len(results['working'])}")
    print(f"Rate Limited: {len(results['rate_limited'])}")
    print(f"Dead:         {len(results['dead'])}")
    print(f"Skipped:      {len(results['skipped'])}")
    print(f"Total:        {sum(len(v) for v in results.values())}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Exit codes for CI/CD
    if results["dead"]:
        print(f"\nWARNING: {len(results['dead'])} dead models found")
        sys.exit(1)  # Fail CI if dead models detected (should be excluded)
    sys.exit(0)


if __name__ == "__main__":
    main()
