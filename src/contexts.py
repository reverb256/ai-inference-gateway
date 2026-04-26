"""
Single source of truth for ALL model context windows and max output tokens.

RULE: This file is the ONLY place where context_length and max_tokens live.
      Every consumer (gateway router, metrics, model_defaults, OMP, external
      tools) reads from HERE — either directly or via /v1/models/context.

      If you change -c in a llama.cpp NixOS config, change it HERE too.
      If you add a model to the router, add it HERE too.
      NEVER hardcode these numbers anywhere else.
"""


# ── Local llama.cpp servers ──────────────────────────────────────────────────
# Keys MUST match the ModelInfo.id values in router.py.
# Values MUST match the -c flags in NixOS configs.

LLAMA_SERVER_CONTEXT = {
    # zephyr:1237 — RTX 3090 (24GB), Qwen3.6-35B-A3B IQ3_M
    # NixOS: hosts/zephyr/llama-3090.nix  ctxSize = 262144
    "qwen3.6-35b": 262144,

    # zephyr:1236 — RTX 3060Ti (8GB), SuperGemma4 E4B Q5_K_M
    # NixOS: hosts/zephyr/llama-3060ti.nix  ctxSize = 131072
    "supergemma4": 131072,

    # sentry:1235 — RX 5600 XT (8GB ROCm), Qwen3.5-4B Q4_K_M
    # K8s: kubernetes/modules/ai-inference.nix  -c 262144
    "qwen3.5-4b": 262144,

    # K8s: kubernetes/modules/ai-inference.nix  --ctx-size=16384
    "qwen3.5-0.8b": 16384,
}


# ── Cloud providers ──────────────────────────────────────────────────────────
# Native context windows from vendor docs.

CLOUD_MODEL_CONTEXT = {
    # Z.AI (expires May 8 2026)
    "glm-5.1":       200000,   # 200K native
    "glm-5":         200000,
    "glm-5-turbo":   200000,
    "glm-4.7":       200000,
    "glm-4.7-flash": 131072,   # 128K
    "glm-4.7-flashx":131072,   # 128K
    "glm-4.6":       131072,   # 128K
    "glm-4.6v":      131072,   # 128K (vision)
    "glm-4.5":       131072,   # 128K
    "glm-4.5-flash": 131072,   # 128K
    "glm-4.5-air":   132000,   # 132K native
    "glm-4-flash":   128000,   # 128K

    # NVIDIA NIM (free tier)
    "qwen/qwen3-coder-480b-a35b-instruct":     262144,  # 256K (NIM free tier)
    "z-ai/glm-5.1":                             202752,  # ~200K
    "meta/llama-3.1-405b-instruct":             131072,  # 128K
    "nvidia/llama-3.3-nemotron-super-49b-v1":   131072,  # 128K
    "nvidia/nemotron-3-super-120b-a12b":        131072,
    "nvidia/nemotron-mini-4b-instruct":          131072,
    "qwen/qwen3.5-397b-a17b":                    131072,
    "qwen/qwen3.5-122b-a10b":                    131072,
    "qwen/qwen3-next-80b-a3b-instruct":          131072,
    "deepseek-ai/deepseek-v4-flash":              1048576,
    "meta/llama-3.3-70b-instruct":                131072,
    "meta/llama-4-maverick-17b-128e-instruct":    131072,
    "meta/llama-3.2-90b-vision-instruct":         131072,
    "moonshotai/kimi-k2-instruct":                131072,
    "moonshotai/kimi-k2-thinking":                131072,
    "mistralai/mistral-large-3-675b-instruct-2512": 131072,
    "mistralai/devstral-2-123b-instruct-2512":     131072,
    "mistralai/mistral-small-4-119b-2603":         131072,
    "mistralai/magistral-small-2506":              32768,
    "mistralai/mixtral-8x22b-instruct-v0.1":       65536,
    "minimaxai/minimax-m2.5":                       131072,
    "minimaxai/minimax-m2.7":                       131072,
    "openai/gpt-oss-120b":                          131072,
    "z-ai/glm4.7":                                  131072,
    "google/gemma-3-27b-it":                        131072,
    "microsoft/phi-4-multimodal-instruct":          131072,
    "stepfun-ai/step-3.5-flash":                    131072,
    "bytedance/seed-oss-36b-instruct":              131072,
    "stockmark/stockmark-2-100b-instruct":          131072,

    # Google Gemini (free tier)
    "gemini-2.5-flash":      1048576,  # 1M
    "gemini-2.5-pro":        1048576,  # 1M
    "gemini-2.5-flash-lite": 1048576,  # 1M
    "gemini-2.0-flash":      1048576,  # 1M
    "gemini-2.0-flash-lite": 1048576,  # 1M

    # Pollinations (free fallback)
    "openai/gpt-oss-120b":   131072,   # 128K
}

# ── Qwen family groupings ────────────────────────────────────────────────────
# Used by router for family-based routing decisions.

QWEN_FAMILY_CONTEXT = {
    "qwen3.5": 262144,
    "qwen3.6": 262144,
}


# ── Max output tokens per model ──────────────────────────────────────────────
# Conservative caps. Cloud models often accept higher but these are safe.

MAX_OUTPUT_TOKENS = {
    # Local
    "qwen3.6-35b":  32768,
    "supergemma4":  32768,
    "qwen3.5-4b":   32768,
    "qwen3.5-0.8b":  8192,

    # Z.AI
    "glm-5.1":       131072,
    "glm-5":         131072,
    "glm-5-turbo":   131072,
    "glm-4.7":       131072,
    "glm-4.7-flash":   8192,
    "glm-4.7-flashx":  8192,
    "glm-4.6":         8192,
    "glm-4.6v":       32768,
    "glm-4.5":         8192,
    "glm-4.5-flash":   8192,
    "glm-4.5-air":     98304,
    "glm-4-flash":     8192,

    # NVIDIA NIM
    "qwen/qwen3-coder-480b-a35b-instruct":     65536,
    "z-ai/glm-5.1":                           131072,
    "meta/llama-3.1-405b-instruct":            65536,
    "nvidia/llama-3.3-nemotron-super-49b-v1":  32768,
    "nvidia/nemotron-3-super-120b-a12b":       32768,
    "nvidia/nemotron-mini-4b-instruct":         8192,
    "qwen/qwen3.5-397b-a17b":                   65536,
    "qwen/qwen3.5-122b-a10b":                   32768,
    "qwen/qwen3-next-80b-a3b-instruct":         32768,
    "deepseek-ai/deepseek-v4-flash":             65536,
    "meta/llama-3.3-70b-instruct":               32768,
    "meta/llama-4-maverick-17b-128e-instruct":   32768,
    "meta/llama-3.2-90b-vision-instruct":        32768,
    "moonshotai/kimi-k2-instruct":               32768,
    "moonshotai/kimi-k2-thinking":               65536,
    "mistralai/mistral-large-3-675b-instruct-2512": 32768,
    "mistralai/devstral-2-123b-instruct-2512":     32768,
    "mistralai/mistral-small-4-119b-2603":         32768,
    "mistralai/magistral-small-2506":              8192,
    "mistralai/mixtral-8x22b-instruct-v0.1":       32768,
    "minimaxai/minimax-m2.5":                       32768,
    "minimaxai/minimax-m2.7":                       32768,
    "openai/gpt-oss-120b":                          32768,
    "z-ai/glm4.7":                                  32768,
    "google/gemma-3-27b-it":                        32768,
    "microsoft/phi-4-multimodal-instruct":          16384,
    "stepfun-ai/step-3.5-flash":                    8192,
    "bytedance/seed-oss-36b-instruct":              16384,
    "stockmark/stockmark-2-100b-instruct":          16384,

    # Google Gemini
    "gemini-2.5-flash":      65536,
    "gemini-2.5-pro":        65536,
    "gemini-2.5-flash-lite":  8192,
    "gemini-2.0-flash":       8192,
    "gemini-2.0-flash-lite":  8192,

    # Pollinations
    "openai/gpt-oss-120b":  32768,
}


# ── Lookup helpers ───────────────────────────────────────────────────────────

def get_context_length(model_id: str) -> int:
    """Get actual serving context for any model."""
    norm = model_id.lower().replace(".gguf", "").strip()
    # Exact or substring match on local servers first
    for k, v in LLAMA_SERVER_CONTEXT.items():
        if k == norm or k in norm or norm in k:
            return v
    # Then cloud
    for k, v in CLOUD_MODEL_CONTEXT.items():
        if k == model_id or k == norm or k in norm:
            return v
    # Qwen family default
    if "qwen" in norm:
        return 262144
    return 131072  # safe default


def get_max_tokens(model_id: str) -> int:
    """Get max output tokens for a model."""
    norm = model_id.lower().replace(".gguf", "").strip()
    for k, t in MAX_OUTPUT_TOKENS.items():
        if k in norm or norm in k:
            return t
    return 16384


def get_all_models_info() -> dict:
    """Return all model info for client config generation."""
    result = {}
    for mid, ctx in {**LLAMA_SERVER_CONTEXT, **CLOUD_MODEL_CONTEXT}.items():
        result[mid] = {
            "context_window": ctx,
            "max_tokens": get_max_tokens(mid),
        }
    return result
