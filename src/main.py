# modules/services/ai-inference/ai_inference_gateway/main.py
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Initialize logger early (needed for import error handling)
logger = logging.getLogger(__name__)

from ai_inference_gateway.config import GatewayConfig
from ai_inference_gateway.mcp_broker import create_mcp_broker_from_config
from ai_inference_gateway.metrics import ModelMetricsTracker
from ai_inference_gateway.openai_client import OpenAIBackendError, create_openai_client
from ai_inference_gateway.pipeline import MiddlewarePipeline
from ai_inference_gateway.response_format import transform_request
from ai_inference_gateway.router import (
    RouteDecision,
    create_default_router,
    get_optimal_qwen_params,
)
from ai_inference_gateway.utils.redis_client import RedisClient
from ai_inference_gateway.utils.tool_utils import (
    create_tool_result_openai,
    extract_tool_calls_openai,
)

# Import TTS handler
try:
    from ai_inference_gateway.tts_handler import (
        POLLINATIONS_TTS_URL,
        QWEN3_TTS_LANGUAGES,
        QWEN3_TTS_MODELS,
        QWEN3_TTS_SPEAKERS,
        TTSRequest,
        TTSResponse,
        close_tts_handler,
        get_audio_extension,
        get_content_type,
        get_tts_handler,
    )

    TTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TTS handler not available: {e}")
    TTS_AVAILABLE = False
    TTSRequest = None
    TTSResponse = None
    QWEN3_TTS_MODELS = {}
    QWEN3_TTS_SPEAKERS = {}
    QWEN3_TTS_LANGUAGES = []
    get_audio_extension = None
    POLLINATIONS_TTS_URL = None

# Import Audio handler (STT - Speech-to-Text)
try:
    from ai_inference_gateway.audio_handler import (
        QWEN3_AUDIO_MODELS,
        SUPPORTED_AUDIO_FORMATS,
        TranscriptionRequest,
        TranscriptionResponse,
        TranslationRequest,
        close_audio_handler,
        get_audio_handler,
        read_audio_file,
    )

    AUDIO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Audio handler not available: {e}")
    AUDIO_AVAILABLE = False
    TranscriptionRequest = None
    TranscriptionResponse = None
    TranslationRequest = None
    QWEN3_AUDIO_MODELS = {}
    read_audio_file = None
    SUPPORTED_AUDIO_FORMATS = []

# Import Vision handler (Qwen3-VL for image understanding)
try:
    from ai_inference_gateway.vision_handler import (
        QWEN3_VISION_MODELS,
        ImageContent,
        VisionMessage,
        VisionRequest,
        VisionResponse,
        close_vision_handler,
        encode_image_to_base64,
        get_vision_handler,
        read_image_from_url,
    )

    VISION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vision handler not available: {e}")
    VISION_AVAILABLE = False
    VisionRequest = None
    VisionResponse = None
    VisionMessage = None
    ImageContent = None
    QWEN3_VISION_MODELS = {}
    read_image_from_url = None
    encode_image_to_base64 = None

# Import semantic cache
try:
    from ai_inference_gateway.semantic_cache import (
        CacheConfig,
        SemanticCache,
    )

    SEMANTIC_CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Semantic cache not available: {e}")
    SEMANTIC_CACHE_AVAILABLE = False
    SemanticCache = None
    CacheConfig = None

# Import SearXNG integration
try:
    from ai_inference_gateway.searxng_integration import (
        SearxngIntegration,
        get_searxng,
    )

    SEARXNG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SearXNG integration not available: {e}")
    SEARXNG_AVAILABLE = False
    SearxngIntegration = None

# Import agent search enhancements
try:
    from ai_inference_gateway.agent_search import (
        AgentSearchEngine,
        SearchIntent,
        get_agent_search_engine,
    )

    AGENT_SEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent search enhancements not available: {e}")
    AGENT_SEARCH_AVAILABLE = False
    AgentSearchEngine = None
    SearchIntent = None

# Import Self-Improvement System (meta-learning from all interactions)
try:
    from ai_inference_gateway.hermes_integration import get_hermes_bridge
    from ai_inference_gateway.self_improvement import get_self_improvement_engine
    from ai_inference_gateway.self_improvement_api import create_self_improvement_router

    SELF_IMPROVEMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Self-improvement system not available: {e}")
    SELF_IMPROVEMENT_AVAILABLE = False
    create_self_improvement_router = None

# Import HTTP-MCP bridge
try:
    from ai_inference_gateway.mcp_http_bridge import (
        HTTPMCPBridge,
        get_http_mcp_bridge,
    )

    HTTP_MCP_BRIDGE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"HTTP-MCP bridge not available: {e}")
    HTTP_MCP_BRIDGE_AVAILABLE = False
    HTTPMCPBridge = None

# Import hybrid search (RAG + SearXNG)
try:
    from ai_inference_gateway.hybrid_search import (
        HybridSearchEngine,
        get_hybrid_search,
    )

    HYBRID_SEARCH_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Hybrid search not available: {e}")
    HYBRID_SEARCH_AVAILABLE = False
    HybridSearchEngine = None

# Import RAG ingestion
try:
    from ai_inference_gateway.rag.ingestion import (
        IngestionConfig,
        IngestionSource,
        URLIngestionService,
        create_ingestion_service,
    )

    RAG_INGESTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG ingestion not available: {e}")
    RAG_INGESTION_AVAILABLE = False
    URLIngestionService = None
    IngestionConfig = None
    IngestionSource = None

# Import PII redactor
try:
    from ai_inference_gateway.pii_redactor import (
        PIIRedactor,
        RedactionMode,
        get_default_redactor,
    )

    PII_REDACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PII redactor not available: {e}")
    PII_REDACTOR_AVAILABLE = False
    PIIRedactor = None
    RedactionMode = None

# Import content moderation
try:
    from ai_inference_gateway.moderation import (
        ContentModerator,
        ModerationCategory,
        ModerationResult,
        get_default_moderator,
    )

    MODERATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Content moderation not available: {e}")
    MODERATION_AVAILABLE = False
    ContentModerator = None
    ModerationResult = None
    ModerationCategory = None


# GPU scheduler integration (for signaling workload state)
from ai_inference_gateway import gpu_scheduler

# RAG imports
try:
    from ai_inference_gateway.rag import RAGConfig
    from ai_inference_gateway.rag.embeddings import create_embedding_service
    from ai_inference_gateway.rag.qdrant_client import get_qdrant_manager
    from ai_inference_gateway.rag.search import create_search_service

    RAG_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RAG module not available: {e}")
    RAG_AVAILABLE = False
    RAGConfig = None
    get_qdrant_manager = None

# Import files module (Garage S3 storage)
try:
    from ai_inference_gateway.files import (
        FileMetadata,
        FileNotFoundError,
        FileStorageError,
        FileUploadError,
        GarageS3Client,
        generate_file_id,
        get_garage_client,
        get_mime_type,
    )

    FILES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Files module not available: {e}")
    FILES_AVAILABLE = False
    GarageS3Client = None
    get_garage_client = None

# Import middleware (placed here after conditional imports)
from ai_inference_gateway.middleware.circuit_breaker import CircuitBreaker  # noqa: E402
from ai_inference_gateway.middleware.concurrency_limiter import ConcurrencyLimiter  # noqa: E402
from ai_inference_gateway.middleware.observability import ObservabilityMiddleware  # noqa: E402
from ai_inference_gateway.middleware.rate_limiter import RateLimiterMiddleware  # noqa: E402
from ai_inference_gateway.middleware.security_filter import SecurityFilterMiddleware  # noqa: E402

# Try to import prometheus_client for metrics endpoint
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    generate_latest = None
    CONTENT_TYPE_LATEST = None


GATEWAY_VERSION = "2.0.0"


async def check_backend_health(
    url: str,
    timeout: float = 5.0,
    api_key: Optional[str] = None,
    backend_type: str = "unknown",
) -> bool:
    """
    Check if backend is healthy by querying the models endpoint.

    Args:
        url: Backend URL
        timeout: Request timeout in seconds
        api_key: Optional API key for authentication
        backend_type: Type of backend (llama-cpp, zai, pollinations, etc.)

    Returns:
        True if backend is healthy, False otherwise
    """
    try:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{url}/v1/models", headers=headers)
            return response.status_code == 200
    except Exception:
        return False


class GatewayState:
    """
    Gateway application state.

    Stores shared state across the application including config,
    Redis client, middleware pipeline, OpenAI client wrapper, and router.
    """

    def __init__(
        self,
        config: GatewayConfig,
        redis_client: Optional[RedisClient] = None,
        pipeline: Optional[MiddlewarePipeline] = None,
        openai_client=None,
        router=None,
    ):
        self.config = config
        self.redis_client = redis_client
        self.pipeline = pipeline
        self.openai_client = openai_client
        self.router = router
        # Backend health cache to avoid checking on every request
        self.backend_health_cache = {
            "healthy": True,
            "last_check": 0,
            "ttl": 30,  # Cache health status for 30 seconds
        }
        # RAG service (initialized if enabled)
        self.rag_search = None
        self.rag_config = None
        # Semantic cache (initialized if enabled)
        self.semantic_cache = None
        # MCP broker (initialized if enabled)
        self.mcp_broker = None
        # RAG ingestion service (initialized if enabled)
        self.rag_ingestion = None
        # SearXNG integration (initialized if enabled)
        self.searxng = None


def build_backend_headers(config: GatewayConfig, request_headers: dict) -> dict:
    """
    Build backend headers including authentication.

    Args:
        config: Gateway configuration
        request_headers: Original request headers

    Returns:
        Headers dictionary for backend request
    """
    # Headers to exclude from forwarding (they'll be regenerated)
    excluded_headers = {
        "host",
        "content-length",
        "content-encoding",
        "transfer-encoding",
    }

    # Start with client headers (excluding problematic headers)
    headers = {
        k: v for k, v in request_headers.items() if k.lower() not in excluded_headers
    }

    # Only add backend authentication if client didn't provide one
    if "authorization" not in {k.lower() for k in headers.keys()}:
        if config.backend_type == "zai":
            api_key = config.get_zai_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        elif config.backend_type == "pollinations":
            api_key = config.get_pollinations_api_key()
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

    return headers


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan.

    Handles startup (initialize connections) and shutdown (cleanup).
    """
    # Get gateway state from app state
    state: GatewayState = app.state.gateway

    logger.info("Starting AI Inference Gateway v%s", GATEWAY_VERSION)

    # Initialize Sentry if enabled
    if state.config.sentry.enabled:
        sentry_dsn = state.config.sentry.get_dsn()
        if sentry_dsn:
            try:
                import sentry_sdk

                sentry_sdk.init(
                    dsn=sentry_dsn,
                    environment=state.config.sentry.environment,
                    traces_sample_rate=state.config.sentry.traces_sample_rate,
                    # FastAPI integration
                    integrations=[
                        sentry_sdk.integrations.fastapi.FastApiIntegration(),
                        sentry_sdk.integrations.httpx.HttpxIntegration(),
                    ],
                    # Filter out common errors
                    ignore_errors=[
                        "KeyboardInterrupt",
                        "httpx.ConnectError",
                    ],
                    # Send PII data (disabled by default, enable if needed)
                    send_default_pii=False,
                )
                logger.info(
                    "Sentry initialized (environment=%s, traces_sample_rate=%.2f)",
                    state.config.sentry.environment,
                    state.config.sentry.traces_sample_rate,
                )
            except ImportError:
                logger.warning(
                    "sentry-sdk not available, skipping Sentry initialization"
                )
            except Exception as e:
                logger.warning(f"Sentry initialization failed: {e}")
        else:
            logger.info("Sentry enabled but no DSN configured")

    # Initialize Redis client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    state.redis_client = RedisClient(redis_url=redis_url)
    redis_connected = await state.redis_client.connect()

    if redis_connected:
        logger.info("Connected to Redis")
    else:
        logger.warning("Redis unavailable, using in-memory fallback")

    # Build middleware pipeline
    state.pipeline = build_middleware_pipeline(state.config, state.redis_client)
    logger.info(
        "Middleware pipeline initialized with %d middleware", state.pipeline.count
    )

    # Initialize router (no API key needed for llama-cpp)
    try:
        state.router = create_default_router()
        logger.info("Router initialized with %d models", len(state.router.models))
    except Exception as e:
        logger.warning(f"Router initialization failed: {e}")
        state.router = None

    # Initialize MCP broker if enabled
    state.mcp_broker = None
    try:
        state.mcp_broker = await create_mcp_broker_from_config(state.config)
        if state.mcp_broker:
            logger.info("MCP broker initialized")
        else:
            logger.warning("MCP broker creation returned None")
    except Exception as e:
        logger.error(f"MCP broker initialization failed: {e}", exc_info=True)

    # Initialize RAG if enabled
    state.rag_search = None
    state.rag_config = None

    # Check if RAG is enabled via environment variable
    rag_enabled = os.getenv("RAG_ENABLED", "false").lower() == "true"

    if RAG_AVAILABLE and rag_enabled:
        try:
            logger.info("Initializing RAG service...")

            # Build RAG config from environment variables
            from ai_inference_gateway.rag.config import (
                ChunkingConfig,
                EmbeddingConfig,
                RAGConfig,
                RerankerConfig,
                SearchConfig,
            )
            from ai_inference_gateway.rag.embeddings import create_embedding_service
            from ai_inference_gateway.rag.qdrant_client import get_qdrant_manager
            from ai_inference_gateway.rag.search import create_search_service

            # Get environment variables
            qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
            embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
            chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
            chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
            top_k = int(os.getenv("RAG_TOP_K", "5"))
            hybrid_search = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
            reranker_enabled = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
            reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

            state.rag_config = RAGConfig(
                enable=True,
                qdrant_url=qdrant_url,
                embedding=EmbeddingConfig(
                    model=embedding_model,
                    device=os.getenv("EMBEDDING_DEVICE", "cpu"),  # CPU by default (GPU often mining)
                ),
                chunking=ChunkingConfig(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap
                ),
                search=SearchConfig(default_top_k=top_k, hybrid_search=hybrid_search),
                reranker=RerankerConfig(enable=reranker_enabled, model=reranker_model),
            )

            # Initialize components
            embedder = await create_embedding_service(state.rag_config.embedding)
            qdrant = await get_qdrant_manager(state.rag_config)  # noqa: F823
            state.rag_search = await create_search_service(
                state.rag_config, embedder, qdrant
            )

            logger.info("RAG service initialized successfully")

            # Initialize RAG ingestion service if enabled
            rag_ingestion_enabled = (
                os.getenv("RAG_INGESTION_ENABLED", "false").lower() == "true"
            )

            if RAG_INGESTION_AVAILABLE and rag_ingestion_enabled:
                try:
                    logger.info("Initializing RAG ingestion service...")

                    # Get environment variables
                    allowed_domains_str = os.getenv("RAG_ALLOWED_DOMAINS", "")
                    blocked_domains_str = os.getenv("RAG_BLOCKED_DOMAINS", "")

                    allowed_domains = [
                        d.strip() for d in allowed_domains_str.split(",") if d.strip()
                    ]
                    blocked_domains = [
                        d.strip() for d in blocked_domains_str.split(",") if d.strip()
                    ]

                    # Get RAG components
                    from ai_inference_gateway.rag.chunker import DocumentChunker
                    from ai_inference_gateway.rag.qdrant_client import (
                        get_qdrant_manager,
                    )

                    chunker = DocumentChunker(state.rag_config.chunking)
                    qdrant_manager = get_qdrant_manager(state.rag_config.qdrant_url)

                    # Create ingestion service
                    state.rag_ingestion = create_ingestion_service(
                        rag_config=state.rag_config,
                        embedder=embedder,
                        chunker=chunker,
                        qdrant=qdrant_manager,
                        mcp_broker=state.mcp_broker,
                        allowed_domains=allowed_domains,
                        blocked_domains=blocked_domains,
                    )

                    logger.info(
                        f"RAG ingestion service initialized: "
                        f"allowed_domains={len(allowed_domains)}, "
                        f"blocked_domains={len(blocked_domains)}"
                    )
                except Exception as e:
                    logger.warning(f"RAG ingestion service initialization failed: {e}")
                    state.rag_ingestion = None
            else:
                logger.info(
                    "RAG ingestion service disabled (set RAG_INGESTION_ENABLED=true to enable)"
                )
        except Exception as e:
            logger.error(f"RAG initialization failed: {e}")
            import traceback

            traceback.print_exc()
            state.rag_search = None

    # Initialize semantic cache if enabled
    if SEMANTIC_CACHE_AVAILABLE:
        try:
            # Check if semantic cache is enabled via environment variable
            semantic_cache_enabled = (
                os.getenv("SEMANTIC_CACHE_ENABLED", "false").lower() == "true"
            )

            if semantic_cache_enabled:
                logger.info("Initializing semantic cache...")

                # Get environment variables
                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
                qdrant_url = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")
                similarity_threshold = float(
                    os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.85")
                )
                exact_ttl = int(os.getenv("EXACT_CACHE_TTL_SECONDS", "3600"))
                semantic_ttl = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))

                cache_config = CacheConfig(
                    redis_url=redis_url,
                    qdrant_url=qdrant_url,
                    similarity_threshold=similarity_threshold,
                    exact_ttl_seconds=exact_ttl,
                    semantic_ttl_seconds=semantic_ttl,
                    enable_exact_cache=True,
                    enable_semantic_cache=True,
                )

                state.semantic_cache = SemanticCache(config=cache_config)
                logger.info("Semantic cache initialized (Redis + Qdrant)")
            else:
                logger.info(
                    "Semantic cache disabled (set SEMANTIC_CACHE_ENABLED=true to enable)"
                )
        except Exception as e:
            logger.warning(f"Semantic cache initialization failed: {e}")
            state.semantic_cache = None
    else:
        logger.info("Semantic cache not available (install redis, qdrant-client)")

    # Initialize SearXNG integration if enabled
    if SEARXNG_AVAILABLE:
        try:
            searxng_enabled = os.getenv("SEARXNG_ENABLED", "true").lower() == "true"
            cache_ttl = int(os.getenv("SEARXNG_CACHE_TTL", "300"))

            if searxng_enabled:
                logger.info("Initializing SearXNG integration...")
                state.searxng = get_searxng(cache_ttl=cache_ttl)
                logger.info(
                    "SearXNG integration initialized (auto-improving features enabled)"
                )
            else:
                logger.info(
                    "SearXNG integration disabled (set SEARXNG_ENABLED=true to enable)"
                )
        except Exception as e:
            logger.warning(f"SearXNG initialization failed: {e}")
            state.searxng = None

    # Initialize GPU scheduler communication
    try:
        gpu_scheduler.init_scheduler_comms()
        logger.info("GPU scheduler communication initialized")
    except Exception as e:
        logger.warning(f"GPU scheduler initialization failed: {e}")

    # Startup health validation — check backend reachability
    backend_healthy = await check_backend_health(
        state.config.backend_url,
        timeout=10.0,  # Longer timeout for startup
        backend_type=state.config.backend_type,
    )
    state.backend_health_cache = {
        "healthy": backend_healthy,
        "last_check": __import__("time").time(),
        "ttl": 30,
    }
    if backend_healthy:
        logger.info(
            "Backend health check PASSED: %s (%s)",
            state.config.backend_url,
            state.config.backend_type,
        )
    else:
        logger.warning(
            "Backend health check FAILED: %s (%s) — entering degraded mode",
            state.config.backend_url,
            state.config.backend_type,
        )

    # Initialize cost tracker (always available — SQLite, zero deps)
    try:
        from ai_inference_gateway.services.cost_tracker import CostTracker
        cost_tracker = CostTracker()
        app.state.cost_tracker = cost_tracker
        logger.info("Cost tracker initialized (SQLite)")
    except Exception as e:
        logger.warning(f"Cost tracker initialization failed: {e}")
        app.state.cost_tracker = None

    # Initialize virtual key manager
    try:
        from ai_inference_gateway.services.virtual_keys import VirtualKeyManager
        app.state.virtual_key_manager = VirtualKeyManager()
        logger.info("Virtual key manager initialized")
    except Exception as e:
        logger.warning(f"Virtual key manager initialization failed: {e}")
        app.state.virtual_key_manager = None

    # Startup complete
    logger.info("Gateway startup complete")

    yield

    # Shutdown cleanup
    logger.info("Shutting down gateway")

    # Flush self-improvement memory buffers
    if SELF_IMPROVEMENT_AVAILABLE:
        try:
            from ai_inference_gateway.self_improvement import shutdown_self_improvement

            await shutdown_self_improvement()
            logger.info("Self-improvement engine shutdown complete")
        except Exception as e:
            logger.warning(f"Failed to shutdown self-improvement engine: {e}")

    if state.redis_client:
        await state.redis_client.close()
        logger.info("Redis connection closed")

    if state.semantic_cache:
        await state.semantic_cache.close()
        logger.info("Semantic cache connections closed")

    if state.searxng:
        await state.searxng.close()
        logger.info("SearXNG integration closed")

    if state.rag_ingestion:
        await state.rag_ingestion.close()
        logger.info("RAG ingestion service closed")

    if state.openai_client:
        await state.openai_client.close()
        logger.info("OpenAI clients closed")

    # Close TTS handler
    if TTS_AVAILABLE:
        try:
            await close_tts_handler()
            logger.info("TTS handler closed")
        except Exception as e:
            logger.warning(f"Failed to close TTS handler: {e}")

    # Close Audio handler
    if AUDIO_AVAILABLE:
        try:
            from ai_inference_gateway.audio_handler import close_audio_handler

            await close_audio_handler()
            logger.info("Audio handler closed")
        except Exception as e:
            logger.warning(f"Failed to close audio handler: {e}")

    # Close Vision handler
    if VISION_AVAILABLE:
        try:
            from ai_inference_gateway.vision_handler import close_vision_handler

            await close_vision_handler()
            logger.info("Vision handler closed")
        except Exception as e:
            logger.warning(f"Failed to close vision handler: {e}")

    logger.info("Gateway shutdown complete")


def build_middleware_pipeline(
    config: GatewayConfig, redis_client: Optional[RedisClient]
) -> MiddlewarePipeline:
    """
    Build the middleware pipeline from configuration.

    Args:
        config: Gateway configuration
        redis_client: Optional Redis client

    Returns:
        Configured middleware pipeline
    """
    pipeline = MiddlewarePipeline()

    # DEBUG: Log knowledge fabric config
    import os

    env_enabled = os.environ.get("MIDDLEWARE__KNOWLEDGE_FABRIC__ENABLED", "NOT_SET")
    # Add observability middleware (should always be first)
    if config.middleware.observability.enabled:
        pipeline.add(ObservabilityMiddleware(config.middleware.observability))
        logger.info("Added ObservabilityMiddleware")

    # Add RAG injector middleware (injects knowledge context automatically)
    # Must come AFTER observability but BEFORE security filter
    # so we can track RAG usage metrics
    if config.middleware.RAG_ENABLED:
        from ai_inference_gateway.middleware.rag_injector import RAGInjectorMiddleware

        # Create RAG middleware with config (search_service injected later via context)
        rag_middleware = RAGInjectorMiddleware(
            search_service=None,  # Will be injected from state.rag_search during request
            enabled=config.middleware.RAG_ENABLED,
            collection="knowledge-base",
            min_confidence=0.5,
            max_chunks=config.middleware.RAG_TOP_K,
        )
        pipeline.add(rag_middleware)
        logger.info("Added RAGInjectorMiddleware (automatic knowledge injection)")

    # Add Knowledge Fabric middleware (unified multi-source retrieval)
    if config.middleware.knowledge_fabric.enabled:
        from ai_inference_gateway.middleware.knowledge_fabric import (
            create_knowledge_fabric,
        )

        # Prepare knowledge fabric config
        fabric_config = {
            "rag_enabled": config.middleware.knowledge_fabric.rag_enabled,
            "rag_top_k": config.middleware.knowledge_fabric.rag_top_k,
            "code_search_paths": config.middleware.knowledge_fabric.code_search_paths,
            "searxng_url": config.middleware.knowledge_fabric.searxng_url,
            "mcp_url": config.middleware.knowledge_fabric.mcp_url,
            "web_max_results": config.middleware.knowledge_fabric.web_max_results,
            "searxng_max_results": config.middleware.knowledge_fabric.searxng_max_results,
            "code_max_results": config.middleware.knowledge_fabric.code_max_results,
            "brain_wiki_enabled": config.middleware.knowledge_fabric.brain_wiki_enabled,
            "brain_wiki_path": config.middleware.knowledge_fabric.brain_wiki_path,
            "brain_wiki_max_results": config.middleware.knowledge_fabric.brain_wiki_max_results,
            "brain_wiki_max_chunk_chars": config.middleware.knowledge_fabric.brain_wiki_max_chunk_chars,
        }

        fabric_middleware = create_knowledge_fabric(
            rrf_k=config.middleware.knowledge_fabric.rrf_k,
            enabled=config.middleware.knowledge_fabric.enabled,
            config=fabric_config,
        )
        pipeline.add(fabric_middleware)
        logger.info("Added KnowledgeFabricMiddleware (multi-source unified retrieval)")

    # Add security filter
    if config.middleware.security.enabled:
        pipeline.add(SecurityFilterMiddleware(config.middleware.security))
        logger.info("Added SecurityFilterMiddleware")

    # Add rate limiter
    if config.middleware.rate_limiting.enabled:
        rate_limiter = RateLimiterMiddleware(
            config=config.middleware.rate_limiting, redis_client=redis_client
        )
        pipeline.add(rate_limiter)
        logger.info("Added RateLimiterMiddleware")

    # Add concurrency limiter
    if config.middleware.concurrency_limiter.enabled:
        concurrency_limiter = ConcurrencyLimiter(
            max_concurrency=config.middleware.concurrency_limiter.max_concurrency
        )
        pipeline.add(concurrency_limiter)
        logger.info(
            f"Added ConcurrencyLimiter (max_concurrency={config.middleware.concurrency_limiter.max_concurrency})"
        )

    # Add circuit breaker
    if config.middleware.circuit_breaker.enabled:
        circuit_breaker = CircuitBreaker(
            service_id="backend",
            config=config.middleware.circuit_breaker,
            redis_client=redis_client,
        )
        pipeline.add(circuit_breaker)
        logger.info("Added CircuitBreaker")

    return pipeline


def is_reasoning_model(model_id: str) -> bool:
    """Check if a model is a reasoning model that uses reasoning_content field.

    These models use OpenAI-compatible format (/v1/chat/completions)
    and require response translation.
    """
    reasoning_indicators = [
        "claude-4.6-opus-reasoning-distilled",
        "claude-4.6-opus-distilled",
        "claude-opus-reasoning",
        "claude-opus-distilled",
        "reasoning",
        "deepseek-r1",
    ]
    model_lower = model_id.lower()
    return any(indicator in model_lower for indicator in reasoning_indicators)


def translate_openai_to_anthropic(openai_response: dict, original_model: str) -> dict:
    """
    Translate OpenAI chat/completions response to Anthropic messages format.

    Handles reasoning_content field which is used by reasoning models.
    Maps OpenAI's separate reasoning_content + content to Anthropic's content blocks.
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})

    # Extract content from OpenAI response
    reasoning_content = message.get("reasoning_content", "")
    content_text = message.get("content", "")

    # Build Anthropic content blocks
    anthropic_content = []

    # Add thinking block if reasoning_content exists
    if reasoning_content:
        anthropic_content.append({"type": "thinking", "thinking": reasoning_content})

    # Add text block if content exists
    if content_text:
        anthropic_content.append({"type": "text", "text": content_text})

    # If both are empty but we have output_tokens, something went wrong
    # Put a placeholder text
    if not anthropic_content:
        usage = openai_response.get("usage", {})
        if usage.get("completion_tokens", 0) > 0:
            logger.warning(
                f"Model {original_model} generated tokens but no content returned"
            )
            anthropic_content.append({"type": "text", "text": ""})

    return {
        "id": openai_response.get("id", f"msg_{openai_response.get('created', '')}"),
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": original_model,  # Use the originally requested Claude model ID
        "stop_reason": choice.get("finish_reason", "stop"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
            "cache_creation_input_tokens": openai_response.get("usage", {}).get(
                "cache_creation_tokens", 0
            ),
            "cache_read_input_tokens": openai_response.get("usage", {}).get(
                "cache_read_tokens", 0
            ),
        },
    }


def create_app(config: Optional[GatewayConfig] = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        config: Optional gateway configuration. If not provided, loads from environment.

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = GatewayConfig()
    # Initialize gateway state with OpenAI client wrapper
    openai_client = create_openai_client(config)
    gateway_state = GatewayState(
        config=config,
        openai_client=openai_client,
    )
    # Create FastAPI app
    app = FastAPI(
        title="AI Inference Gateway",
        description="Advanced gateway for AI inference backends with middleware",
        version=GATEWAY_VERSION,
        lifespan=lifespan,
    )
    # Store gateway state in app
    app.state.gateway = gateway_state

    # Register modular route handlers
    try:
        from ai_inference_gateway.routes.admin import router as admin_router
        app.include_router(admin_router)
    except ImportError as e:
        logger.warning(f"Admin routes not available: {e}")

    try:
        from ai_inference_gateway.routes.virtual_keys import router as vk_router
        app.include_router(vk_router)
    except ImportError as e:
        logger.warning(f"Virtual keys routes not available: {e}")

    # Add health endpoint
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint with actual backend health status.

        Returns comprehensive health information including:
        - Gateway status
        - Backend health (with cached status)
        - Circuit breaker state
        - Qdrant status (if RAG is enabled)
        - Redis status (if semantic cache is enabled)
        """
        import time

        state: GatewayState = app.state.gateway

        # Check if cached health status is still valid
        now = time.time()
        cache_age = now - state.backend_health_cache["last_check"]

        if cache_age > state.backend_health_cache["ttl"]:
            # Cache expired, check actual backend health
            api_key = None  # llama-cpp doesn't require authentication
            is_healthy = await check_backend_health(
                state.config.backend_url,
                api_key=api_key,
                backend_type=state.config.backend_type,
            )
            state.backend_health_cache = {
                "healthy": is_healthy,
                "last_check": now,
                "ttl": 30,
            }
            logger.info(f"Backend health check: {is_healthy}")
            # Recalculate cache_age after updating
            cache_age = 0

        backend_healthy = state.backend_health_cache["healthy"]

        # Build health response
        health_response = {
            "status": "healthy" if backend_healthy else "degraded",
            "gateway": {
                "version": GATEWAY_VERSION,
                "host": config.gateway_host,
                "port": config.gateway_port,
            },
            "backend": {
                "url": config.backend_url,
                "type": config.backend_type,
                "healthy": backend_healthy,
                "cache_age_seconds": int(cache_age),
            },
        }

        # Add circuit breaker state if enabled
        if config.middleware.circuit_breaker.enabled:
            try:
                # Get circuit breaker from middleware pipeline
                for middleware in state.pipeline.middleware:
                    if hasattr(middleware, "_state"):
                        state_name = (
                            middleware._state.name
                            if hasattr(middleware._state, "name")
                            else str(middleware._state)
                        )
                        health_response["circuit_breaker"] = {
                            "state": state_name,
                            "service_id": getattr(middleware, "service_id", "backend"),
                        }
                        break
            except Exception as e:
                logger.warning(f"Failed to get circuit breaker state: {e}")

        # Add Qdrant status if RAG is enabled
        if SEMANTIC_CACHE_AVAILABLE and state.semantic_cache:
            try:
                # Check Qdrant connection
                qdrant_healthy = await state.semantic_cache._check_qdrant_health()
                health_response["qdrant"] = {
                    "healthy": qdrant_healthy,
                    "url": state.semantic_cache.config.qdrant_url,
                    "collection": state.semantic_cache.config.qdrant_collection,
                }
            except Exception as e:
                logger.warning(f"Failed to check Qdrant health: {e}")
                health_response["qdrant"] = {"healthy": False, "error": str(e)}

        # Add Redis status if semantic cache is enabled
        if SEMANTIC_CACHE_AVAILABLE and state.semantic_cache:
            try:
                # Check Redis connection
                redis_healthy = await state.semantic_cache._check_redis_health()
                health_response["redis"] = {
                    "healthy": redis_healthy,
                    "url": state.semantic_cache.config.redis_url,
                }
            except Exception as e:
                logger.warning(f"Failed to check Redis health: {e}")
                health_response["redis"] = {"healthy": False, "error": str(e)}

        # Add RAG ingestion service status if enabled
        if RAG_INGESTION_AVAILABLE and state.rag_ingestion:
            try:
                health_response["rag_ingestion"] = {"healthy": True, "enabled": True}
            except Exception as e:
                logger.warning(f"Failed to check RAG ingestion health: {e}")
                health_response["rag_ingestion"] = {"healthy": False, "error": str(e)}

        return health_response

    # Add models endpoint
    @app.get("/v1/models")
    async def list_models(request: Request):
        """List available models from backend with automatic failover."""
        state: GatewayState = app.state.gateway

        try:
            # Use OpenAI SDK to list models with automatic failover
            models = await state.openai_client.primary_client.models.list()

            # Update model availability metrics
            try:
                from ai_inference_gateway.metrics import update_model_availability

                model_ids = [m.id for m in models.data]
                update_model_availability(model_ids)
            except Exception as metrics_error:
                logger.warning(
                    f"Failed to update model availability metrics: {metrics_error}"
                )

            # Convert to dict for JSON response
            models_dict = models.model_dump()

            # Add TTS models if TTS is available
            if TTS_AVAILABLE and QWEN3_TTS_MODELS:
                tts_models = []
                for model_key, config in QWEN3_TTS_MODELS.items():
                    backend = config.get("backend", "local")
                    if backend == "pollinations":
                        model_id = model_key  # Use clean model names for Pollinations
                        owned_by = "pollinations"
                    else:
                        model_id = f"qwen3-tts-{model_key}"
                        owned_by = "qwen"

                    tts_models.append(
                        {
                            "id": model_id,
                            "object": "model",
                            "created": int(datetime.now().timestamp()),
                            "owned_by": owned_by,
                            "permission": [
                                {
                                    "id": "model",
                                    "object": "model_permission",
                                    "created": int(datetime.now().timestamp()),
                                }
                            ],
                            "root": model_id,
                            "parent": None,
                            # TTS-specific metadata
                            "capabilities": {
                                "type": "tts",
                                "audio_formats": ["mp3", "wav", "opus", "aac", "flac"],
                                "sample_rate": config["sample_rate"],
                                "languages": config.get("language", "en"),
                                "quality": config["quality"],
                                "description": config["description"],
                                "backend": backend,
                            },
                        }
                    )

                # Add TTS models to the response
                if "data" in models_dict:
                    models_dict["data"].extend(tts_models)

            # Add Qwen3-Audio (STT) models if available
            if AUDIO_AVAILABLE and QWEN3_AUDIO_MODELS:
                stt_models = []
                for model_key, config in QWEN3_AUDIO_MODELS.items():
                    stt_models.append(
                        {
                            "id": model_key,
                            "object": "model",
                            "created": int(datetime.now().timestamp()),
                            "owned_by": "qwen",
                            "permission": [
                                {
                                    "id": "model",
                                    "object": "model_permission",
                                    "created": int(datetime.now().timestamp()),
                                }
                            ],
                            "root": model_key,
                            "parent": None,
                            # STT-specific metadata
                            "capabilities": {
                                "type": "stt",
                                "sample_rate": config["sample_rate"],
                                "max_duration": config["max_duration"],
                                "languages": config.get("languages", ["en"]),
                                "supports_translation": config.get(
                                    "supports_translation", False
                                ),
                                "supports_timestamps": config.get(
                                    "supports_timestamps", False
                                ),
                                "description": config["description"],
                            },
                        }
                    )

                # Add STT models to the response
                if "data" in models_dict:
                    models_dict["data"].extend(stt_models)

            # Add Qwen3-Vision models if available
            if VISION_AVAILABLE and QWEN3_VISION_MODELS:
                vision_models = []
                for model_key, config in QWEN3_VISION_MODELS.items():
                    vision_models.append(
                        {
                            "id": model_key,
                            "object": "model",
                            "created": int(datetime.now().timestamp()),
                            "owned_by": "qwen",
                            "permission": [
                                {
                                    "id": "model",
                                    "object": "model_permission",
                                    "created": int(datetime.now().timestamp()),
                                }
                            ],
                            "root": model_key,
                            "parent": None,
                            # Vision-specific metadata
                            "capabilities": {
                                "type": "vision",
                                "max_tokens": config["max_tokens"],
                                "max_images": config.get("max_images", 1),
                                "supports_video": config.get("supports_video", False),
                                "description": config["description"],
                            },
                        }
                    )

                # Add vision models to the response
                if "data" in models_dict:
                    models_dict["data"].extend(vision_models)

            return JSONResponse(content=models_dict)

        except OpenAIBackendError as e:
            logger.warning(f"Backend unavailable for model listing: {e}")
            return JSONResponse(content={"object": "list", "data": []})
        except Exception as e:
            logger.warning(f"Error fetching models, returning empty list: {e}")
            return JSONResponse(content={"object": "list", "data": []})

    # Add system prompts endpoint
    @app.get("/system-prompts")
    async def get_system_prompts(request: Request):
        """
        Get current system prompts configuration.

        Returns the configured system prompts for different request types.
        """
        state: GatewayState = app.state.gateway
        cfg = state.config.system_prompts

        return JSONResponse(
            content={
                "enabled": cfg.enabled,
                "default": cfg.default,
                "coding": cfg.coding,
                "reasoning": cfg.reasoning,
                "analysis": cfg.analysis,
                "agentic": cfg.agentic,
                "fast": cfg.fast,
                "custom": cfg.custom,
            }
        )

    @app.post("/system-prompts")
    async def update_system_prompts(request: Request):
        """
        Update system prompts configuration.

        Accepts JSON with system prompts for different categories.
        """
        state: GatewayState = app.state.gateway
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        # Update configuration
        cfg = state.config.system_prompts
        if "enabled" in body:
            cfg.enabled = body["enabled"]
        if "default" in body:
            cfg.default = body["default"]
        if "coding" in body:
            cfg.coding = body["coding"]
        if "reasoning" in body:
            cfg.reasoning = body["reasoning"]
        if "analysis" in body:
            cfg.analysis = body["analysis"]
        if "agentic" in body:
            cfg.agentic = body["agentic"]
        if "fast" in body:
            cfg.fast = body["fast"]
        if "custom" in body:
            cfg.custom = body["custom"]

        logger.info("System prompts configuration updated")
        return JSONResponse(content={"status": "updated"})

    # Add chat completions endpoint
    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """
        Chat completions endpoint with middleware processing and intelligent routing.

        Supports both streaming and non-streaming requests.
        Uses router for intelligent model selection based on request analysis.
        """
        import time

        _request_start = time.time()  # noqa: F841

        # Signal GPU scheduler that AI workload is starting
        gpu_scheduler.notify_ai_starting()

        state: GatewayState = app.state.gateway

        # Read request body
        body = await request.json()

        # Transform response_format to backend instructions
        # (OpenAI JSON mode -> system prompts)
        if "response_format" in body:
            body = await transform_request(body)
            logger.debug(
                f"Transformed response_format request for model: {body.get('model')}"
            )

        # Check if streaming is requested
        stream = body.get("stream", False)

        # Get messages for routing
        messages = body.get("messages", [])

        # Extract headers and query params for category-based routing (oh-my-opencode style)
        headers = dict(request.headers)
        query_params = dict(request.query_params)

        # Use router to select best model
        requested_model = body.get("model", None)
        route_decision: RouteDecision = await state.router.route(
            messages=messages,
            requested_model=requested_model,
            urgency="normal",  # Could be made configurable
            headers=headers,
            query_params=query_params,
        )

        # Update model in body based on routing decision
        body["model"] = route_decision.model

        # Log routing decision to self-improvement system (non-blocking)
        if SELF_IMPROVEMENT_AVAILABLE:
            try:
                import asyncio

                engine = get_self_improvement_engine()
                # Don't await - log in background
                asyncio.create_task(
                    engine.log_routing_decision(
                        model_requested=requested_model or "auto",
                        model_routed=route_decision.model,
                        routing_reason=route_decision.reason,
                        token_count=route_decision.estimated_tokens,
                        task_type=route_decision.specialization.value
                        if route_decision.specialization
                        else "general",
                        latency_ms=0,  # Will be updated after request completes
                        backend_used=route_decision.backend,
                    )
                )
            except Exception as log_error:
                logger.warning(f"Failed to log routing decision: {log_error}")

        # Detect if this is a vision request
        is_vision_request = False
        try:
            from ai_inference_gateway.vision import detect_vision_content

            is_vision_request = detect_vision_content(messages)
        except ImportError:
            pass  # Vision module not available, continue without detection

        # Apply model-specific defaults for optimal parameters
        try:
            from ai_inference_gateway.model_defaults import apply_model_defaults

            body = apply_model_defaults(
                model_id=route_decision.model,
                request_params=body,
                override=False,  # Only fill missing values, don't override user params
                is_vision_request=is_vision_request,
            )
        except Exception as defaults_error:
            logger.warning(f"Failed to apply model defaults: {defaults_error}")
            # Continue without defaults - not critical

        # Apply Qwen3.5 optimal parameters automatically
        # This enhances Qwen models with proper temperature, top_p, etc.
        if "qwen" in route_decision.model.lower():
            try:
                # Determine if thinking is enabled (check for thinking params in request)
                thinking_enabled = False
                if "thinking" in body:
                    thinking = body.get("thinking", {})
                    if isinstance(thinking, dict):
                        thinking_enabled = (
                            thinking.get("type", "disabled") != "disabled"
                        )
                    elif isinstance(thinking, bool):
                        thinking_enabled = thinking

                # Detect task type from messages for better param selection
                task_type = "general"
                messages_text = " ".join([m.get("content", "") for m in messages])
                if any(
                    keyword in messages_text.lower()
                    for keyword in ["code", "function", "debug", "fix"]
                ):
                    task_type = "coding"
                elif any(
                    keyword in messages_text.lower()
                    for keyword in ["tool", "search", "call", "execute"]
                ):
                    task_type = "agentic"
                elif len(messages_text) > 10000:  # Long conversation, prioritize speed
                    task_type = "fast"

                # Get optimal Qwen parameters
                qwen_params = get_optimal_qwen_params(
                    model_id=route_decision.model,
                    thinking_enabled=thinking_enabled,
                    task_type=task_type,
                )

                # Apply optimal params only if not explicitly set by user
                for param, value in qwen_params.items():
                    if param not in body:
                        body[param] = value
                        logger.debug(
                            f"Applied Qwen optimal param: {param}={value} "
                            f"(model={route_decision.model}, task={task_type})"
                        )
            except Exception as qwen_error:
                logger.warning(f"Failed to apply Qwen optimal params: {qwen_error}")
                # Continue without Qwen params - not critical

        # Thinking mode control — OFF by default, ON only when explicitly requested.
        # This prevents reasoning tokens from eating output budget for simple tasks
        # (structured output, JSON parsing, Vane search, etc).
        #
        # Per-model API:
        #   llama.cpp (Qwen3.5, Gemma 4): chat_template_kwargs: {"enable_thinking": bool}
        #   ZAI GLM-5.1:                   enable_thinking: bool  (top-level)
        #
        # Enable thinking by passing in the request body:
        #   {"chat_template_kwargs": {"enable_thinking": true}, ...}   (for llama.cpp)
        #   {"enable_thinking": true, ...}                              (for ZAI/GLM)
        #   {"thinking": {"type": "enabled"}, ...}                      (OpenAI reasoning style)
        try:
            model_lower = route_decision.model.lower()
            is_thinking_model = (
                "qwen" in model_lower
                or "gemma" in model_lower
                or "glm" in model_lower
            )

            if is_thinking_model:
                # Check if caller explicitly wants thinking ON
                explicit_thinking = False

                # Check chat_template_kwargs.enable_thinking (llama.cpp standard)
                ctk = body.get("chat_template_kwargs", {})
                if isinstance(ctk, dict) and "enable_thinking" in ctk:
                    explicit_thinking = bool(ctk["enable_thinking"])

                # Check top-level enable_thinking (ZAI/Alibaba Cloud standard)
                if "enable_thinking" in body:
                    explicit_thinking = explicit_thinking or bool(body["enable_thinking"])

                # Check OpenAI reasoning style: {"thinking": {"type": "enabled"}}
                thinking_cfg = body.get("thinking")
                if isinstance(thinking_cfg, dict):
                    if thinking_cfg.get("type") == "enabled" or thinking_cfg.get("enable_thinking"):
                        explicit_thinking = True
                elif isinstance(thinking_cfg, bool):
                    explicit_thinking = explicit_thinking or thinking_cfg

                # Apply thinking OFF for llama.cpp backends (Qwen, Gemma)
                if "qwen" in model_lower or "gemma" in model_lower:
                    if "chat_template_kwargs" not in body:
                        body["chat_template_kwargs"] = {}
                    if "enable_thinking" not in body["chat_template_kwargs"]:
                        body["chat_template_kwargs"]["enable_thinking"] = explicit_thinking

                # Apply thinking OFF for ZAI GLM backends
                # Only inject if not already set by caller
                if "glm" in model_lower and "enable_thinking" not in body:
                    body["enable_thinking"] = explicit_thinking

                logger.info(
                    f"Thinking control: enable_thinking={explicit_thinking} "
                    f"(model={route_decision.model}, backend={route_decision.backend})"
                )
        except Exception as think_err:
            logger.warning(f"Failed to apply thinking control: {think_err}")

        # Track request start for smart load balancing
        import uuid

        request_id = str(uuid.uuid4())
        state.router.track_request_start(
            request_id=request_id,
            model=route_decision.model,
            backend=route_decision.backend,
            stream=stream,
        )

        logger.info(
            f"Routed request to model: {route_decision.model} "
            f"(backend: {route_decision.backend}, "
            f"specialization: {route_decision.specialization})"
        )

        # Create metrics tracker for this request
        metrics_tracker = ModelMetricsTracker(
            model=route_decision.model,
            backend=route_decision.backend,
            requested_model=requested_model,
        )

        # Record routing decision metadata
        metrics_tracker.record_routing_decision(
            confidence=route_decision.confidence,
            reason=route_decision.reason,
            specialization=(
                route_decision.specialization.value
                if route_decision.specialization
                else None
            ),
        )

        # Create context for middleware
        context = {
            "request_id": request_id,
            "start_time": _request_start,  # Track request start for observability
            "request_body": body,
            "request_headers": dict(request.headers),
            "client_ip": request.client.host if request.client else "unknown",
            "model": route_decision.model,  # Use routed model for concurrency limiter
            "route_decision": route_decision,  # Store routing decision
            "metrics_tracker": metrics_tracker,  # Metrics tracker
            "cost_tracker": getattr(app.state, "cost_tracker", None),
        }

        # Inject RAG search service if available (for RAGInjectorMiddleware)
        if state.rag_search:
            context["rag_search_service"] = state.rag_search
            logger.debug("RAG search service available for automatic injection")

        # Process request through middleware pipeline
        should_continue, error = await state.pipeline.process_request(request, context)

        if not should_continue:
            # Middleware blocked the request
            if error:
                raise error
            raise HTTPException(status_code=403, detail="Request blocked by middleware")

        # Check if RAG middleware enhanced the request with knowledge context
        if "rag_enhanced_body" in context:
            body = context["rag_enhanced_body"]
            logger.info("Using RAG-enhanced request body with injected context")

        # Check if tools are requested for agentic workflow
        has_tools = "tools" in body and body.get("tools")

        # Forward to backend using OpenAI SDK
        if stream:
            # Handle streaming response
            # Use tool-enabled streaming if tools are present
            stream_func = (
                stream_backend_response_with_tools
                if has_tools and state.mcp_broker
                else stream_backend_response
            )

            # Prepare kwargs for streaming function
            stream_kwargs = {
                "openai_client": state.openai_client,
                "body": body,
                "pipeline": state.pipeline,
                "context": context,
                "config": state.config,
                "router": state.router,
                "request_id": request_id,
                "metrics_tracker": metrics_tracker,
            }

            # Add mcp_broker for tool-enabled streaming
            if has_tools and state.mcp_broker:
                stream_kwargs["mcp_broker"] = state.mcp_broker

            return StreamingResponse(
                stream_func(**stream_kwargs),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Handle non-streaming response
            try:
                response = await handle_non_streaming_request(
                    state.openai_client,
                    body,
                    state.pipeline,
                    context,
                    state.config,
                    metrics_tracker,
                )
                return response
            finally:
                # Always clean up request tracking
                state.router.track_request_end(request_id)
                # Signal GPU scheduler that AI workload is stopping
                gpu_scheduler.notify_ai_stopping()

    @app.post("/v1/messages")
    async def messages(request: Request):
        """
        Anthropic Messages API endpoint - proxies to llama.cpp backend.

        This endpoint adds:
        - Model selection by Claude model ID (haiku, sonnet, opus variants)
        - Thinking effort levels (low/medium/high) that map to budget_tokens
        - ZAI fallback when llama.cpp unavailable
        - Extended thinking support

        Model mapping (5 Claude options → 3 underlying local models):
        - claude-haiku-4 → qwen3.5-0.8b-claude-4.6-opus-reasoning-distilled
        - claude-sonnet-4-20250514 → qwen3.5-9b-claude-4.6-opus-reasoning-distilled
        - claude-sonnet-4-20250514-1m → qwen3.5-9b-claude-4.6-opus-reasoning-distilled (extended)
        - claude-opus-4-20250514 → qwen3.5-35b-a3b
        - claude-opus-4-20250514-1m → qwen3.5-35b-a3b (extended)

        Thinking effort levels (map to budget_tokens):
        - low → 5,000 tokens (quick responses)
        - medium → 15,000 tokens (balanced reasoning)
        - high → 50,000 tokens (deep analysis)
        """
        import time
        import uuid

        # Signal GPU scheduler that AI workload is starting
        gpu_scheduler.notify_ai_starting()

        state: GatewayState = app.state.gateway
        _request_start = time.time()

        # Read request body
        body = await request.json()

        # Extract parameters
        model = body.get("model", "")
        max_tokens = body.get("max_tokens", 4096)
        messages = body.get("messages", [])
        system = body.get("system", None)
        stream = body.get("stream", False)

        # Apply configured system prompts if no explicit system prompt provided
        if not system and state.config.system_prompts.enabled:
            # Determine which system prompt to use based on request type
            # Check for coding patterns
            is_coding = any(
                keyword in str(messages).lower()
                for keyword in [
                    "def ",
                    "class ",
                    "function",
                    "import ",
                    "code",
                    "function(",
                    "return ",
                ]
            )
            # Check for reasoning patterns
            is_reasoning = any(
                keyword in str(messages).lower()
                for keyword in ["reason", "think", "step", "explain", "why", "how"]
            )
            # Check for agentic patterns
            is_agentic = any(
                keyword in str(messages).lower()
                for keyword in ["agent", "workflow", "multi-step", "plan", "execute"]
            )
            # Check for fast response request
            is_fast = any(
                keyword in str(messages).lower()
                for keyword in ["quickly", "asap", "fast", "brief", "short"]
            )

            if is_coding:
                system = state.config.system_prompts.get_prompt("coding")
            elif is_agentic:
                system = state.config.system_prompts.get_prompt("agentic")
            elif is_reasoning:
                system = state.config.system_prompts.get_prompt("reasoning")
            elif is_fast:
                system = state.config.system_prompts.get_prompt("fast")
            else:
                # Use default system prompt
                system = state.config.system_prompts.get_prompt("default")

        # Extended thinking / thinking intensity parameters
        # Effort levels (low/medium/high) map to budget_tokens, NOT model selection
        thinking_budget = None
        thinking_intensity = None
        thinking_type = None  # Backend expects: "enabled" | "disabled" | "adaptive"

        if "thinking" in body:
            thinking = body["thinking"]
            if isinstance(thinking, dict):
                thinking_intensity = thinking.get("intensity", None)
                thinking_budget = thinking.get("budget_tokens", None)
                thinking_type = thinking.get("type", "enabled")
            elif isinstance(thinking, str):
                # String form like "low", "medium", "high" maps to intensity
                thinking_intensity = thinking
                thinking_type = "enabled"  # Default to enabled for string form
        elif "thinking_intensity" in body:
            thinking_intensity = body["thinking_intensity"]
            thinking_type = "enabled"

        # Map effort levels to budget_tokens if not explicitly set
        if thinking_intensity and not thinking_budget:
            effort_budget_map = {
                "low": 5000,  # Quick responses, minimal reasoning
                "medium": 15000,  # Balanced reasoning
                "high": 50000,  # Deep analysis, extensive reasoning
                "auto": None,  # Let backend decide
            }
            thinking_budget = effort_budget_map.get(thinking_intensity)
            logger.info(
                f"Thinking intensity '{thinking_intensity}' → budget_tokens={thinking_budget}"
            )

        # Build/update thinking dict in body for backend compatibility
        # Format: {"type": "enabled"|"disabled"|"adaptive", "budget_tokens": int}
        if thinking_budget is not None or thinking_type:
            if "thinking" not in body or not isinstance(body["thinking"], dict):
                body["thinking"] = {}
            if thinking_type:
                body["thinking"]["type"] = thinking_type
            if thinking_budget is not None:
                body["thinking"]["budget_tokens"] = thinking_budget
            # Store original intensity for logging/metadata
            if thinking_intensity:
                body["thinking"]["intensity"] = thinking_intensity

        # Use router to determine the best model (based on model name only, not intensity)
        # Extract headers and query params for category-based routing (oh-my-opencode style)
        headers = dict(request.headers)
        query_params = dict(request.query_params)

        route_decision: RouteDecision = await state.router.route(
            messages=messages,
            requested_model=model,
            urgency="normal",
            headers=headers,
            query_params=query_params,
        )

        # Apply prefill optimization limits based on model variant
        # Base models get aggressive limits for faster TTFT, extended models get full context
        trimmed_messages = state.router.apply_prefill_limits(messages, model)
        body["messages"] = trimmed_messages

        if len(trimmed_messages) != len(messages):
            logger.info(
                f"Prefill optimization: {len(messages)} → {len(trimmed_messages)} messages "
                f"for model {model}"
            )

        # Update model in request
        body["model"] = route_decision.model

        # Create request ID for tracking
        request_id = str(uuid.uuid4())
        state.router.track_request_start(
            request_id=request_id,
            model=route_decision.model,
            backend=route_decision.backend,
            stream=stream,
        )

        logger.info(
            f"Anthropic API request: original={model} → {route_decision.model} "
            f"(intensity={thinking_intensity}, budget={thinking_budget}, backend={route_decision.backend})"
        )

        # Create metrics tracker
        metrics_tracker = ModelMetricsTracker(
            model=route_decision.model,
            backend=route_decision.backend,
            requested_model=model,
        )

        # Record routing decision
        metrics_tracker.record_routing_decision(
            confidence=route_decision.confidence,
            reason=route_decision.reason,
            specialization=(
                route_decision.specialization.value
                if route_decision.specialization
                else None
            ),
        )

        # Build headers with authentication
        backend_headers = build_backend_headers(state.config, dict(request.headers))

        # Create context for middleware
        context = {
            "request_id": request_id,
            "start_time": _request_start,
            "request_body": body,
            "request_headers": dict(request.headers),
            "model": route_decision.model,
            "route_decision": route_decision,
            "metrics_tracker": metrics_tracker,
            "cost_tracker": getattr(app.state, "cost_tracker", None),
        }

        # Inject RAG search service if available
        if state.rag_search:
            context["rag_search_service"] = state.rag_search

        # Process request through middleware pipeline
        should_continue, error = await state.pipeline.process_request(request, context)
        if not should_continue:
            if error:
                raise error
            raise HTTPException(status_code=403, detail="Request blocked by middleware")

        # Check if RAG middleware enhanced the request
        if "rag_enhanced_body" in context:
            body = context["rag_enhanced_body"]

        # Dispatch to backend
        if stream:
            return StreamingResponse(
                stream_anthropic_response(
                    openai_client=state.openai_client,
                    body=body,
                    pipeline=state.pipeline,
                    context=context,
                    config=state.config,
                    original_model=model,
                    thinking_intensity=thinking_intensity,
                    request_id=request_id,
                    metrics_tracker=metrics_tracker,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            return await handle_anthropic_non_streaming(
                openai_client=state.openai_client,
                body=body,
                pipeline=state.pipeline,
                context=context,
                config=state.config,
                original_model=model,
                thinking_intensity=thinking_intensity,
                metrics_tracker=metrics_tracker,
            )

    # ============================================================================
    # MCP Broker Endpoints
    # ============================================================================
    @app.get("/mcp/servers")
    async def list_mcp_servers():
        """List all configured MCP servers."""
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            return {"servers": [], "message": "MCP broker not enabled"}

        servers = await state.mcp_broker.list_servers()
        return {"servers": servers}

    @app.get("/mcp/tools")
    async def list_mcp_tools(server: Optional[str] = None):
        """List available MCP tools from all servers or a specific server."""
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        tools = await state.mcp_broker.get_tools(server)
        return {"tools": tools}

    @app.post("/mcp/cache/invalidate")
    async def invalidate_mcp_cache(request: Request):
        """
        Invalidate cached MCP tool schemas.

        Body: {"server": "optional_server_name"}
        If server is omitted, invalidates all caches.
        """
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        body = await request.json()
        server_name = body.get("server")

        result = await state.mcp_broker.invalidate_cache(server_name)
        return result

    @app.get("/mcp/cache/metrics")
    async def get_mcp_cache_metrics():
        """Get MCP cache performance metrics."""
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        metrics = state.mcp_broker.get_cache_metrics()

        if metrics is None:
            return {"error": "Cache is not enabled"}

        return metrics

    @app.post("/mcp/cache/warmup")
    async def warmup_mcp_cache():
        """
        Trigger cache warm-up for all MCP servers.

        Pre-fetches tool schemas from all remote servers.
        """
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        result = await state.mcp_broker.warm_up_cache()
        return result

    @app.get("/retry/metrics")
    async def get_retry_metrics():
        """
        Get retry handler metrics.

        Returns retry statistics including attempts, successes,
        failures, and breakdown by failure reason.
        """
        state: GatewayState = app.state.gateway

        # Collect metrics from all router clients
        all_metrics = {}

        for model_id, model_info in state.router.models.items():
            client = getattr(model_info, "client", None)
            if client and hasattr(client, "get_retry_metrics"):
                metrics = client.get_retry_metrics()
                if metrics:
                    all_metrics[model_id] = metrics

        if not all_metrics:
            return {
                "message": "No retry metrics available (retry may be disabled)",
                "models": {},
            }

        return {
            "models": all_metrics,
            "summary": {
                "total_models_with_retries": len(all_metrics),
                "total_retry_attempts": sum(
                    m.get("total_retries", 0) for m in all_metrics.values()
                ),
                "total_success_after_retry": sum(
                    m.get("total_success", 0) for m in all_metrics.values()
                ),
                "total_failures_after_retry": sum(
                    m.get("total_failures", 0) for m in all_metrics.values()
                ),
            },
        }

    @app.post("/retry/reset-metrics")
    async def reset_retry_metrics():
        """Reset all retry metrics across all model clients."""
        state: GatewayState = app.state.gateway

        reset_count = 0
        for model_id, model_info in state.router.models.items():
            client = getattr(model_info, "client", None)
            if client and hasattr(client, "reset_retry_metrics"):
                client.reset_retry_metrics()
                reset_count += 1

        return {
            "message": f"Reset metrics for {reset_count} model clients",
            "models_reset": reset_count,
        }

    @app.get("/cache/metrics")
    async def get_cache_metrics():
        """
        Get semantic cache metrics.

        Returns cache performance statistics including
        hit rates for exact and semantic cache layers.
        """
        state: GatewayState = app.state.gateway

        if not hasattr(state, "semantic_cache") or not state.semantic_cache:
            return {
                "error": "Semantic cache not enabled",
                "message": "Enable Redis and Qdrant for semantic caching",
            }

        metrics = state.semantic_cache.get_metrics()

        if metrics is None:
            return {"error": "Cache metrics not available"}

        return metrics

    @app.post("/cache/invalidate")
    async def invalidate_cache(request: Request):
        """
        Invalidate semantic cache entries.

        Body: {"model": "optional_model_name"}
        If model is omitted, invalidates all cache entries.
        """
        state: GatewayState = app.state.gateway

        if not hasattr(state, "semantic_cache") or not state.semantic_cache:
            raise HTTPException(status_code=501, detail="Semantic cache not enabled")

        body = await request.json()
        model = body.get("model")

        count = await state.semantic_cache.invalidate(model)

        return {
            "message": f"Invalidated {count} cache entries",
            "model": model or "all",
        }

    @app.post("/cache/reset-metrics")
    async def reset_cache_metrics():
        """Reset semantic cache metrics."""
        state: GatewayState = app.state.gateway

        if not hasattr(state, "semantic_cache") or not state.semantic_cache:
            raise HTTPException(status_code=501, detail="Semantic cache not enabled")

        state.semantic_cache.reset_metrics()

        return {"message": "Cache metrics reset"}

    @app.post("/rag/ingest")
    async def ingest_rag_url(request: Request):
        """
        Ingest document from URL into RAG system.

        Body: {
            "url": "https://example.com/document",
            "collection": "default",  // optional
            "source": "mcp_web_reader" | "http_direct"  // optional
        }

        Returns ingestion result with chunks stored.
        """
        state: GatewayState = app.state.gateway

        if not hasattr(state, "rag_ingestion") or not state.rag_ingestion:
            raise HTTPException(
                status_code=501, detail="RAG ingestion service not enabled"
            )

        body = await request.json()
        url = body.get("url")
        collection = body.get("collection", "default")
        source_str = body.get("source", "mcp_web_reader")

        if not url:
            raise HTTPException(status_code=400, detail="Missing required field: url")

        # Parse source preference
        try:
            source = (
                IngestionSource(source_str)
                if RAG_INGESTION_AVAILABLE
                else IngestionSource.HTTP_DIRECT
            )
        except ValueError:
            source = IngestionSource.HTTP_DIRECT

        # Ingest URL
        result = await state.rag_ingestion.ingest_url(url, collection, source)

        return {
            "url": result.url,
            "success": result.success,
            "source": result.source.value,
            "title": result.title,
            "content_length": len(result.content),
            "chunks_count": len(result.chunks),
            "ingested_at": result.ingested_at.isoformat(),
            "error": result.error,
        }

    @app.post("/rag/ingest/batch")
    async def ingest_rag_urls(request: Request):
        """
        Ingest multiple documents from URLs into RAG system.

        Body: {
            "urls": ["https://example.com/doc1", "https://example.com/doc2"],
            "collection": "default",  // optional
            "source": "mcp_web_reader" | "http_direct"  // optional
        }

        Returns batch ingestion results.
        """
        state: GatewayState = app.state.gateway

        if not hasattr(state, "rag_ingestion") or not state.rag_ingestion:
            raise HTTPException(
                status_code=501, detail="RAG ingestion service not enabled"
            )

        body = await request.json()
        urls = body.get("urls", [])
        collection = body.get("collection", "default")
        source_str = body.get("source", "mcp_web_reader")

        if not urls:
            raise HTTPException(status_code=400, detail="Missing required field: urls")

        # Parse source preference
        try:
            source = (
                IngestionSource(source_str)
                if RAG_INGESTION_AVAILABLE
                else IngestionSource.HTTP_DIRECT
            )
        except ValueError:
            source = IngestionSource.HTTP_DIRECT

        # Ingest URLs
        results = await state.rag_ingestion.ingest_urls(urls, collection, source)

        return {
            "total": len(results),
            "successful": sum(1 for r in results if r.success),
            "failed": sum(1 for r in results if not r.success),
            "results": [
                {
                    "url": r.url,
                    "success": r.success,
                    "source": r.source.value,
                    "title": r.title,
                    "chunks_count": len(r.chunks),
                    "error": r.error,
                }
                for r in results
            ],
        }

    @app.post("/pii/redact")
    async def redact_pii(request: Request):
        """
        Redact PII from text.

        Body: {
            "text": "User input with PII",
            "mode": "redact" | "hash" | "mask" | "remove",  // optional
            "enabled_patterns": ["email", "phone", "ssn", ...]  // optional
        }

        Returns text with PII redacted.
        """
        if not PII_REDACTOR_AVAILABLE:
            raise HTTPException(status_code=501, detail="PII redactor not available")

        body = await request.json()
        text = body.get("text", "")
        mode_str = body.get("mode", "redact")
        enabled_patterns = body.get("enabled_patterns")

        # Parse mode
        try:
            mode = RedactionMode(mode_str) if mode_str else None
        except ValueError:
            mode = None

        # Create redactor with custom patterns if specified
        if enabled_patterns:
            redactor = PIIRedactor(enabled_patterns=enabled_patterns)
        else:
            redactor = get_default_redactor()

        # Redact text
        redacted_text = redactor.redact(text, mode=mode)

        return {
            "original": text,
            "redacted": redacted_text,
            "mode": mode_str,
            "patterns_used": [p["name"] for p in redactor.get_patterns()],
        }

    @app.post("/pii/detect")
    async def detect_pii(request: Request):
        """
        Detect PII in text without redacting.

        Body: {
            "text": "User input to analyze",
            "enabled_patterns": ["email", "phone", "ssn", ...]  // optional
        }

        Returns detected PII instances.
        """
        if not PII_REDACTOR_AVAILABLE:
            raise HTTPException(status_code=501, detail="PII redactor not available")

        body = await request.json()
        text = body.get("text", "")
        enabled_patterns = body.get("enabled_patterns")

        # Create redactor with custom patterns if specified
        if enabled_patterns:
            redactor = PIIRedactor(enabled_patterns=enabled_patterns)
        else:
            redactor = get_default_redactor()

        # Detect PII
        detections = redactor.detect(text)

        return {
            "text": text,
            "detections": detections,
            "total_count": sum(len(matches) for matches in detections.values()),
        }

    @app.get("/pii/patterns")
    async def get_pii_patterns():
        """Get available PII patterns."""
        if not PII_REDACTOR_AVAILABLE:
            raise HTTPException(status_code=501, detail="PII redactor not available")

        redactor = get_default_redactor()

        return {
            "patterns": redactor.get_patterns(),
            "total_count": len(redactor.get_patterns()),
        }

    @app.post("/moderation/check")
    async def check_content_moderation(request: Request):
        """
        Check content for policy violations.

        Body: {
            "text": "Content to check",
            "strictness": "low" | "medium" | "high",  // optional
            "threshold": 0.7  // optional, overrides strictness
        }

        Returns moderation results.
        """
        if not MODERATION_AVAILABLE:
            raise HTTPException(
                status_code=501, detail="Content moderation not available"
            )

        body = await request.json()
        text = body.get("text", "")
        strictness = body.get("strictness", "medium")
        threshold = body.get("threshold")

        # Create moderator with specified settings
        moderator = ContentModerator(strictness=strictness, threshold=threshold)

        # Check content
        result = moderator.moderate(text)

        return {
            "flagged": result.flagged,
            "safe": result.safe,
            "categories": [c.value for c in result.categories],
            "scores": result.scores,
            "strictness": strictness,
            "threshold": moderator.threshold,
        }

    @app.post("/moderation/check-messages")
    async def check_messages_moderation(request: Request):
        """
        Check chat messages for policy violations.

        Body: {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "strictness": "low" | "medium" | "high"  // optional
        }

        Returns filtered messages and moderation result.
        """
        if not MODERATION_AVAILABLE:
            raise HTTPException(
                status_code=501, detail="Content moderation not available"
            )

        body = await request.json()
        messages = body.get("messages", [])
        strictness = body.get("strictness", "medium")

        # Create moderator
        moderator = ContentModerator(strictness=strictness)

        # Check messages
        filtered_messages, result = moderator.moderate_messages(messages)

        return {
            "flagged": result.flagged,
            "safe": result.safe,
            "categories": [c.value for c in result.categories],
            "scores": result.scores,
            "original_count": len(messages),
            "filtered_count": len(filtered_messages),
            "messages": filtered_messages,
            "strictness": strictness,
        }

    @app.get("/moderation/categories")
    async def get_moderation_categories():
        """Get available moderation categories."""
        if not MODERATION_AVAILABLE:
            raise HTTPException(
                status_code=501, detail="Content moderation not available"
            )

        moderator = get_default_moderator()

        return {
            "categories": moderator.get_categories(),
            "total_count": len(moderator.get_categories()),
        }

    @app.post("/mcp/call")
    async def call_mcp_tool(request: Request):
        """Call an MCP tool on a specific server."""
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        body = await request.json()
        server_name = body.get("server")
        tool_name = body.get("tool")
        arguments = body.get("arguments", {})

        if not server_name or not tool_name:
            raise HTTPException(
                status_code=400, detail="Missing required fields: server, tool"
            )

        result = await state.mcp_broker.call_tool(
            server_name=server_name, tool_name=tool_name, arguments=arguments
        )
        return result

    @app.get("/mcp/health/{server_name}")
    async def mcp_server_health(server_name: str):
        """Check MCP server health."""
        state: GatewayState = app.state.gateway

        if not state.mcp_broker:
            raise HTTPException(status_code=501, detail="MCP broker not enabled")

        is_healthy = await state.mcp_broker.health_check(server_name)
        return {
            "server": server_name,
            "healthy": is_healthy,
            "exists": server_name in state.mcp_broker.servers,
        }

    # ============================================================================
    # SEARXNG SEARCH ENDPOINTS
    # ============================================================================
    if SEARXNG_AVAILABLE:

        @app.post("/search")
        async def searxng_search(request: Request):
            """
            Perform web search using SearXNG with auto-improving features.

            Body: {
                "query": str (required),
                "category": str (optional, default="general"),
                "language": str (optional, default="all"),
                "max_results": int (optional, default=10),
                "time_range": str (optional, values: day, week, month, year),
                "use_cache": bool (optional, default=true)
            }

            Categories: general, images, videos, news, science, it, files, map, music
            """
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501,
                    detail="SearXNG integration not enabled. Set SEARXNG_ENABLED=true.",
                )

            body = await request.json()
            query = body.get("query", "").strip()

            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Extract parameters
            category = body.get("category", "general")
            language = body.get("language", "all")
            max_results = min(body.get("max_results", 10), 50)
            time_range = body.get("time_range")
            use_cache = body.get("use_cache", True)

            # Perform search
            result = await state.searxng.search(
                query=query,
                category=category,
                language=language,
                max_results=max_results,
                time_range=time_range,
                use_cache=use_cache,
                learning_enabled=True,
            )

            return result

        @app.get("/search/stats")
        async def searxng_stats():
            """Get SearXNG learning statistics and search patterns."""
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            stats = await state.searxng.get_learning_stats()
            return stats

        @app.post("/search/cache/clear")
        async def searxng_clear_cache():
            """Clear SearXNG response cache."""
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            state.searxng.clear_cache()
            return {"success": True, "message": "SearXNG cache cleared"}

        @app.get("/search/ping")
        async def searxng_ping():
            """Check if SearXNG service is accessible."""
            import httpx

            state: GatewayState = app.state.gateway
            searxng_url = state.config.middleware.knowledge_fabric.searxng_url

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        f"{searxng_url}/search", params={"q": "test"}
                    )
                    if response.status_code == 200:
                        return {
                            "status": "healthy",
                            "service": "SearXNG",
                            "url": searxng_url,
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "service": "SearXNG",
                            "code": response.status_code,
                        }
            except Exception as e:
                return {"status": "error", "service": "SearXNG", "error": str(e)}

    # ============================================================================
    # AGENT-OPTIMIZED SEARCH ENDPOINTS
    # ============================================================================
    if AGENT_SEARCH_AVAILABLE and SEARXNG_AVAILABLE:

        @app.post("/search/agent")
        async def agent_search(request: Request):
            """
            Agent-optimized search with intent detection and result summarization.

            Body: {
                "query": str (required),
                "context": str (optional, conversation context),
                "intent": str (optional, auto-detected if not provided),
                "max_results": int (optional, default=10),
                "use_cache": bool (optional, default=true)
            }

            Intents: research, code, facts, troubleshooting, discovery
            """
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            body = await request.json()
            query = body.get("query", "").strip()

            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Get agent search engine
            agent_engine = get_agent_search_engine(state.searxng)

            # Extract parameters
            context = body.get("context")
            intent_str = body.get("intent")
            max_results = min(body.get("max_results", 10), 50)
            use_cache = body.get("use_cache", True)

            # Parse intent if provided
            intent = None
            if intent_str and SearchIntent:
                intent_map = {
                    "research": SearchIntent.RESEARCH,
                    "code": SearchIntent.CODE,
                    "facts": SearchIntent.FACTS,
                    "troubleshooting": SearchIntent.TROUBLESHOOTING,
                    "discovery": SearchIntent.DISCOVERY,
                }
                intent = intent_map.get(intent_str.lower())

            # Perform agent-optimized search
            result = await agent_engine.search_with_agent_workflow(
                query=query,
                context=context,
                intent=intent,
                max_results=max_results,
                use_cache=use_cache,
            )

            return result

        @app.get("/search/agent/stats")
        async def agent_search_stats():
            """Get agent search learning statistics."""
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            agent_engine = get_agent_search_engine(state.searxng)
            stats = agent_engine.get_learning_stats()
            return stats

        @app.post("/search/agent/feedback")
        async def agent_search_feedback(request: Request):
            """Record feedback for progressive improvement."""
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            body = await request.json()
            query = body.get("query", "").strip()
            selected_results = body.get("selected_results", [])
            rating = body.get("rating")

            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            agent_engine = get_agent_search_engine(state.searxng)
            await agent_engine.feedback(query, selected_results, rating)

            return {"success": True, "message": "Feedback recorded"}

    # ============================================================================
    # HTTP-MCP BRIDGE ENDPOINTS
    # ============================================================================
    if HTTP_MCP_BRIDGE_AVAILABLE:

        @app.get("/mcp/v1/tools")
        async def list_mcp_tools(request: Request):
            """List available MCP tools via HTTP."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            server_name = request.query_params.get("server")
            refresh = request.query_params.get("refresh", "false").lower() == "true"

            bridge = get_http_mcp_bridge(state.mcp_broker)
            tools = await bridge.list_tools(
                server_name=server_name, refresh_cache=refresh
            )

            return {"tools": tools}

        @app.post("/mcp/v1/tools/{tool_name}/execute")
        async def execute_mcp_tool(tool_name: str, request: Request):
            """Execute an MCP tool via HTTP."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            body = await request.json()
            arguments = body.get("arguments", {})
            server_name = body.get("server_name")
            timeout = body.get("timeout", 30.0)

            bridge = get_http_mcp_bridge(state.mcp_broker)
            result = await bridge.call_tool(tool_name, arguments, server_name, timeout)

            return result

        @app.get("/mcp/v1/tools/{tool_name}")
        async def get_mcp_tool_info(tool_name: str, request: Request):
            """Get detailed information about a specific MCP tool."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            server_name = request.query_params.get("server")

            bridge = get_http_mcp_bridge(state.mcp_broker)
            tool_info = await bridge.get_tool_info(tool_name, server_name)

            return tool_info

        @app.get("/mcp/v1/servers")
        async def list_mcp_servers(request: Request):
            """List all available MCP servers."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            bridge = get_http_mcp_bridge(state.mcp_broker)
            servers = await bridge.list_servers()

            return {"servers": servers}

        @app.get("/mcp/v1/servers/{server_name}/health")
        async def get_mcp_server_health(server_name: str, request: Request):
            """Get health status of an MCP server."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            bridge = get_http_mcp_bridge(state.mcp_broker)
            health = await bridge.get_server_health(server_name)

            return health

        @app.post("/mcp/v1/cache/clear")
        async def clear_mcp_cache(request: Request):
            """Clear MCP tool cache."""
            state: GatewayState = app.state.gateway

            if not state.mcp_broker:
                raise HTTPException(status_code=501, detail="MCP broker not available")

            bridge = get_http_mcp_bridge(state.mcp_broker)
            bridge.clear_cache()

            return {"success": True, "message": "MCP tool cache cleared"}

    # ============================================================================
    # HYBRID SEARCH ENDPOINTS (RAG + SearXNG)
    # ============================================================================
    if HYBRID_SEARCH_AVAILABLE and SEARXNG_AVAILABLE:

        @app.post("/search/hybrid")
        async def hybrid_search(request: Request):
            """
            Hybrid search combining RAG (local knowledge) and SearXNG (web).

            Body: {
                "query": str (required),
                "max_results": int (optional, default=10),
                "use_rag": bool (optional, default=true),
                "use_web": bool (optional, default=true),
                "collection": str (optional, default="brain-wiki"),
                "rerank": bool (optional, default=true),
                "time_range": str (optional, values: day, week, month, year)
            }
            """
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            body = await request.json()
            query = body.get("query", "").strip()

            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Extract parameters
            max_results = min(body.get("max_results", 10), 50)
            use_rag = body.get("use_rag", True)
            use_web = body.get("use_web", True)
            collection = body.get("collection", "brain-wiki")
            rerank = body.get("rerank", True)
            time_range = body.get("time_range")

            # Get RAG search service if available
            rag_search = state.rag_search if RAG_AVAILABLE and use_rag else None

            # Get hybrid search engine
            hybrid_engine = get_hybrid_search(state.searxng, rag_search)

            # Perform hybrid search
            result = await hybrid_engine.search(
                query=query,
                max_results=max_results,
                use_rag=use_rag and rag_search is not None,
                use_web=use_web,
                collection=collection,
                rerank=rerank,
                time_range=time_range,
            )

            return result

        @app.post("/search/hybrid/progressive")
        async def progressive_search(request: Request):
            """
            Progressive search with automatic query refinement.

            Keeps refining the query until sufficient results are found.

            Body: {
                "query": str (required),
                "max_results": int (optional, default=10),
                "min_results": int (optional, default=5),
                "max_iterations": int (optional, default=3)
            }
            """
            state: GatewayState = app.state.gateway

            if not state.searxng:
                raise HTTPException(
                    status_code=501, detail="SearXNG integration not enabled"
                )

            body = await request.json()
            query = body.get("query", "").strip()

            if not query:
                raise HTTPException(status_code=400, detail="Query cannot be empty")

            # Extract parameters
            max_results = min(body.get("max_results", 10), 50)
            min_results = body.get("min_results", 5)
            max_iterations = body.get("max_iterations", 3)

            # Get RAG search service if available
            rag_search = state.rag_search if RAG_AVAILABLE else None

            # Get hybrid search engine
            hybrid_engine = get_hybrid_search(state.searxng, rag_search)

            # Perform progressive search
            result = await hybrid_engine.search_with_progressive_refinement(
                query=query,
                max_results=max_results,
                min_results=min_results,
                max_iterations=max_iterations,
            )

            return result

    # ============================================================================
    # RAG ENDPOINTS
    # ============================================================================
    if RAG_AVAILABLE:

        @app.post("/rag/documents")
        async def ingest_document(request: Request):
            """Ingest document into RAG knowledge base."""
            state: GatewayState = app.state.gateway

            if not state.rag_search:
                raise HTTPException(status_code=501, detail="RAG service not enabled")

            body = await request.json()
            collection = body.get("collection", "brain-wiki")
            documents = body.get("documents", [])

            if not documents:
                raise HTTPException(status_code=400, detail="No documents provided")

            # Ingest each document
            results = []
            for doc in documents:
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                document_id = doc.get("document_id")

                result = await state.rag_search.ingest_document(
                    collection=collection,
                    content=content,
                    metadata=metadata,
                    document_id=document_id,
                )
                results.append(result)

            # Return summary
            total_chunks = sum(
                r.get("chunks_created", 0) for r in results if r.get("success")
            )
            return {
                "success": True,
                "documents_ingested": len(results),
                "chunks_created": total_chunks,
                "collection": collection,
                "results": results,
            }

        @app.get("/rag/search")
        async def search_knowledge_base(request: Request):
            """Search RAG knowledge base."""
            state: GatewayState = app.state.gateway

            if not state.rag_search:
                raise HTTPException(status_code=501, detail="RAG service not enabled")

            query = request.query_params.get("query", "")
            if not query:
                raise HTTPException(
                    status_code=400, detail="Missing required parameter: query"
                )

            collection = request.query_params.get("collection", "brain-wiki")
            top_k = int(request.query_params.get("top_k", 5))
            rerank = request.query_params.get("rerank", "true").lower() == "true"

            result = await state.rag_search.search(
                query=query, collection=collection, top_k=top_k, rerank=rerank
            )

            return result

        @app.get("/rag/collections")
        async def list_collections(request: Request):
            """List all RAG collections."""
            state: GatewayState = app.state.gateway

            if not state.rag_search:
                raise HTTPException(status_code=501, detail="RAG service not enabled")

            collections = await state.rag_search.get_collections()
            return {"collections": collections}

        @app.delete("/rag/documents")
        async def delete_document(request: Request):
            """Delete document from RAG knowledge base."""
            state: GatewayState = app.state.gateway

            if not state.rag_search:
                raise HTTPException(status_code=501, detail="RAG service not enabled")

            body = await request.json()
            collection = body.get("collection", "brain-wiki")
            document_id = body.get("document_id")

            if not document_id:
                raise HTTPException(
                    status_code=400, detail="Missing required field: document_id"
                )

            result = await state.rag_search.delete_document(
                collection=collection, document_id=document_id
            )

            return result

    # Ollama-compatible API endpoints for Spacebot integration
    @app.get("/api/tags")
    async def ollama_list_models():
        """
        List available models (Ollama-compatible).

        Compatible with: GET /api/tags
        Ollama equivalent: ollama list
        """
        state: GatewayState = app.state.gateway

        # Get models from backend
        try:
            models = await state.openai_client.primary_client.models.list()
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            # Return default models list
            return {
                "models": [
                    {
                        "name": "qwen/qwen3.5-9b",
                        "modified_at": "2024-01-01T00:00:00Z",
                        "size": 0,
                        "digest": "gateway-proxy",
                    }
                ]
            }

        # Transform to Ollama format
        ollama_models = []
        for model in models.data:
            # Extract base name without organization prefix
            _model_name = model.id.split("/")[-1] if "/" in model.id else model.id  # noqa: F841

            ollama_models.append(
                {
                    "name": model.id,
                    "modified_at": "2024-01-01T00:00:00Z",
                    "size": 0,  # Not tracked
                    "digest": "gateway-proxy",
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": "gateway-proxy",
                        "families": None,
                        "parameter_size": "unknown",
                        "quantization_level": "unknown",
                    },
                }
            )

        return {"models": ollama_models}

    @app.post("/api/generate")
    async def ollama_generate(request: Request):
        """
        Generate text completion (Ollama-compatible).

        Compatible with: POST /api/generate
        Transforms to OpenAI format and forwards to backend.
        """
        state: GatewayState = app.state.gateway

        # Parse Ollama request
        body = await request.json()
        model = body.get("model", "qwen/qwen3.5-9b")
        prompt = body.get("prompt", "")
        stream = body.get("stream", False)

        # Extract options
        options = body.get("options", {})
        max_tokens = options.get("num_predict", options.get("max_tokens", 2048))
        temperature = options.get("temperature", 0.7)

        # Transform to OpenAI format
        openai_request = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            # Streaming response
            async def generate_stream():
                async for chunk in stream_backend_response(
                    state.openai_client,
                    openai_request,
                    state.pipeline,
                    {"ollama": True},
                    state.config,
                    state.router,
                    "ollama-gen",
                ):
                    # Transform SSE to Ollama format
                    try:
                        chunk_str = (
                            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                        )
                        if '"content"' in chunk_str:
                            import json

                            # Parse and transform
                            lines = chunk_str.split("\n")
                            for line in lines:
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        data = json.loads(line[6:])
                                        if (
                                            "choices" in data
                                            and len(data["choices"]) > 0
                                        ):
                                            delta = data["choices"][0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                ollama_chunk = {
                                                    "model": model,
                                                    "created_at": datetime.now().isoformat(),
                                                    "response": content,
                                                    "done": False,
                                                }
                                                yield f"data: {json.dumps(ollama_chunk)}\n\n"
                                    except Exception:
                                        pass
                    except Exception:
                        # If transformation fails, pass through as-is
                        if isinstance(chunk, bytes):
                            yield chunk
                        else:
                            yield (
                                chunk.encode("utf-8")
                                if isinstance(chunk, str)
                                else chunk
                            )

                # Send final done signal
                done_chunk = {
                    "model": model,
                    "created_at": datetime.now().isoformat(),
                    "response": "",
                    "done": True,
                    "context": [0, 1],  # Placeholder
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response
            response = await state.openai_client.chat_completion(
                messages=openai_request["messages"],
                model=openai_request["model"],
                max_tokens=openai_request["max_tokens"],
                temperature=openai_request["temperature"],
                stream=False,
            )

            # Transform to Ollama format
            content = response.choices[0].message.content

            return {
                "model": model,
                "created_at": datetime.now().isoformat(),
                "response": content,
                "done": True,
            }

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        """
        Chat completion endpoint (Ollama-compatible).

        Compatible with: POST /api/chat
        This is the main endpoint used by Spacebot.
        """
        state: GatewayState = app.state.gateway

        # Parse Ollama request
        body = await request.json()
        model = body.get("model", "qwen/qwen3.5-9b")
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        # Extract options
        options = body.get("options", {})
        max_tokens = options.get("num_predict", options.get("max_tokens", 2048))
        temperature = options.get("temperature", 0.7)

        # Transform to OpenAI format (already compatible)
        openai_request = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stream:
            # Streaming response
            async def chat_stream():
                async for chunk in stream_backend_response(
                    state.openai_client,
                    openai_request,
                    state.pipeline,
                    {"ollama": True},
                    state.config,
                    state.router,
                    "ollama-chat",
                ):
                    # Transform SSE to Ollama format
                    try:
                        chunk_str = (
                            chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                        )
                        if '"content"' in chunk_str:
                            import json

                            lines = chunk_str.split("\n")
                            for line in lines:
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    try:
                                        data = json.loads(line[6:])
                                        if (
                                            "choices" in data
                                            and len(data["choices"]) > 0
                                        ):
                                            delta = data["choices"][0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                role = delta.get("role", "assistant")
                                                ollama_chunk = {
                                                    "model": model,
                                                    "created_at": datetime.now().isoformat(),
                                                    "message": {
                                                        "role": role,
                                                        "content": content,
                                                    },
                                                    "done": False,
                                                }
                                                yield f"data: {json.dumps(ollama_chunk)}\n\n"
                                    except Exception:
                                        pass
                    except Exception:
                        # If transformation fails, pass through as-is
                        if isinstance(chunk, bytes):
                            yield chunk
                        else:
                            yield (
                                chunk.encode("utf-8")
                                if isinstance(chunk, str)
                                else chunk
                            )

                # Send final done signal
                done_chunk = {
                    "model": model,
                    "created_at": datetime.now().isoformat(),
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "context": [0, 1],  # Placeholder
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"

            return StreamingResponse(
                chat_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            # Non-streaming response - use existing endpoint logic
            response = await state.openai_client.chat_completion(
                messages=openai_request["messages"],
                model=openai_request["model"],
                max_tokens=openai_request["max_tokens"],
                temperature=openai_request["temperature"],
                stream=False,
            )

            # Transform to Ollama format
            message = response.choices[0].message

            return {
                "model": model,
                "created_at": datetime.now().isoformat(),
                "message": {"role": message.role, "content": message.content},
                "done": True,
            }

    @app.get("/api/version")
    async def ollama_version():
        """
        Get Ollama version information (gateway-compatible).

        Compatible with: GET /api/version
        """
        return {
            "version": "2.0.0-gateway",
            "details": {
                "backend": "gateway-proxy",
                "committed": True,
                "features": [
                    "ollama-api",
                    "openai-api",
                    "rag",
                    "mcp",
                    "category-routing",
                ],
            },
        }

    @app.get("/api/categories")
    async def get_categories():
        """
        Get available task categories for category-based routing (oh-my-opencode style).

        Compatible with: GET /api/categories

        Returns information about all available categories, their descriptions,
        and which models they route to.

        Categories:
        - quick: Fast, lightweight tasks
        - ultrabrain: Deep logical reasoning, architecture decisions
        - deep: Complex algorithms, business logic
        - unspecified-high: High uncertainty, high quality needed
        - unspecified-low: Medium complexity with clear requirements
        - visual-engineering: UI/UX, design (vision models)
        - artistry: Creative work
        - writing: Documentation, prose

        Usage:
            curl -H "X-Task-Category: ultrabrain" http://gateway:8080/v1/chat/completions
            curl http://gateway:8080/v1/chat/completions?category=deep
        """
        state: GatewayState = app.state.gateway

        # Get category info from router
        category_info = state.router.get_category_info()

        return {
            "categories": category_info,
            "usage": {
                "header": "X-Task-Category: <category>",
                "query_param": "?category=<category>",
                "complexity_header": "X-Task-Complexity: <low|medium|high>",
                "complexity_param": "?complexity=<low|medium|high>",
            },
            "examples": {
                "quick": "Simple config edits, quick queries",
                "ultrabrain": "Architecture decisions, strategic planning",
                "deep": "Complex algorithms, business logic",
                "unspecified-high": "Complex tasks with high uncertainty",
                "unspecified-low": "Medium complexity with clear requirements",
                "visual-engineering": "UI/UX, frontend components",
                "artistry": "Creative writing, marketing copy",
                "writing": "Documentation, technical prose",
            },
        }

    @app.post("/api/route")
    async def test_route(request: Request):
        """
        Test category-based routing without making a full inference request.

        Compatible with: POST /api/route

        Request body:
        {
            "category": "ultrabrain",           // Optional: X-Task-Category
            "complexity": "high",                // Optional: X-Task-Complexity
            "content": "my request content...",  // Optional: for auto-detection
            "messages": [...],                  // Optional: messages for routing
        }

        Returns the routing decision that would be made for the given request.
        """
        state: GatewayState = app.state.gateway

        body = await request.json()

        # Extract routing parameters
        category = body.get("category")
        complexity = body.get("complexity")
        content = body.get("content")
        messages = body.get("messages")

        # Build headers and query params
        headers = {}
        query_params = {}

        if category:
            headers["X-Task-Category"] = category
            query_params["category"] = category
        if complexity:
            headers["X-Task-Complexity"] = complexity
            query_params["complexity"] = complexity

        # Get messages for routing
        if messages:
            route_messages = messages
        elif content:
            route_messages = [{"role": "user", "content": content}]
        else:
            route_messages = []

        # Get routing decision
        route_decision = await state.router.route(
            messages=route_messages,
            requested_model=None,
            urgency="normal",
            headers=headers,
            query_params=query_params,
        )

        return {
            "decision": {
                "model": route_decision.model,
                "backend": route_decision.backend,
                "confidence": route_decision.confidence,
                "reason": route_decision.reason,
                "specialization": route_decision.specialization.value
                if route_decision.specialization
                else None,
                "expected_latency_ms": route_decision.expected_latency_ms,
            },
            "input": {
                "category": category,
                "complexity": complexity,
                "has_content": bool(content),
                "has_messages": bool(messages),
            },
        }

    @app.post("/api/embeddings")
    async def ollama_embeddings(request: Request):
        """
        Generate embeddings (Ollama-compatible).

        Compatible with: POST /api/embeddings
        Uses RAG embedding service if available.
        """
        state: GatewayState = app.state.gateway

        if not state.rag_search or not state.rag_search.embedder:
            raise HTTPException(
                status_code=501, detail="Embeddings not enabled. Set RAG_ENABLED=true"
            )

        body = await request.json()
        _model = body.get("model", "BAAI/bge-m3")  # noqa: F841
        prompt = body.get("prompt", "")

        # Generate embedding
        embedding = await state.rag_search.embedder.embed_single(prompt)

        return {"embedding": embedding}

    @app.post("/v1/embeddings")
    async def openai_embeddings(request: Request):
        """
        Generate embeddings (OpenAI-compatible).

        Compatible with: POST /v1/embeddings
        Supports both single string and array input.
        Uses RAG embedding service if available.
        """
        state: GatewayState = app.state.gateway

        if not state.rag_search or not state.rag_search.embedder:
            raise HTTPException(
                status_code=501, detail="Embeddings not enabled. Set RAG_ENABLED=true"
            )

        body = await request.json()
        model = body.get("model", "BAAI/bge-m3")
        input_data = body.get("input", "")

        # OpenAI API supports both string and array of strings
        if isinstance(input_data, str):
            texts = [input_data]
        elif isinstance(input_data, list):
            texts = input_data
        else:
            raise HTTPException(
                status_code=400, detail="input must be a string or array of strings"
            )

        # Generate embeddings
        vectors = await state.rag_search.embedder.embed_dense(texts)

        # Format as OpenAI response
        data = [
            {"object": "embedding", "index": i, "embedding": emb}
            for i, emb in enumerate(vectors)
        ]

        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
            },
        }

    @app.post("/embed")
    async def knowledge_fabric_embed(request: Request):
        """
        Knowledge Fabric compatible embed endpoint.

        Expects: {"texts": ["...", "..."]}
        Returns: {"vectors": [[...], [...]], "dim": 384}
        """
        state: GatewayState = app.state.gateway

        if not state.rag_search or not state.rag_search.embedder:
            raise HTTPException(
                status_code=501, detail="Embeddings not enabled. Set RAG_ENABLED=true"
            )

        body = await request.json()
        texts = body.get("texts", [])

        if not texts:
            return {"vectors": [], "dim": 0}

        vectors = await state.rag_search.embedder.embed_dense(texts)
        dim = len(vectors[0]) if vectors else 0

        return {"vectors": vectors, "dim": dim}

    @app.get("/embed/health")
    async def embed_health():
        """Health check for the embed endpoint."""
        state: GatewayState = app.state.gateway
        if state.rag_search and state.rag_search.embedder:
            return {"status": "ok", "model": state.rag_config.embedding.model}
        return {"status": "unavailable"}

    # TTS (Text-to-Speech) API endpoints
    if TTS_AVAILABLE:

        @app.post("/v1/audio/speech")
        async def create_speech(request: Request):
            """
            Generate audio from text using TTS.

            Supports:
            - Pollinations.ai (free, cloud-based) - Use model: "tts-1", "tts-1-hd", or "pollinations-tts"
            - OpenAI-compatible voices: alloy, echo, fable, onyx, nova, shimmer

            Compatible with: POST /v1/audio/speech
            OpenAI-compatible TTS endpoint.

            Request body:
            {
                "model": "tts-1",
                "input": "Hello, world!",
                "voice": "alloy",
                "response_format": "mp3",
                "speed": 1.0
            }

            Returns: Audio file or JSON with base64 encoded audio
            """
            state: GatewayState = app.state.gateway

            # Parse request
            body = await request.json()
            try:
                tts_request = TTSRequest(**body)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid TTS request: {e}")

            # TTS uses Pollinations by default
            backend_url = POLLINATIONS_TTS_URL

            try:
                tts_handler = get_tts_handler(backend_url)

                # Generate speech
                (
                    audio_data,
                    content_type,
                    sample_rate,
                ) = await tts_handler.generate_speech(
                    text=tts_request.input,
                    model=tts_request.model,
                    voice=tts_request.voice,
                    speed=tts_request.speed,
                    response_format=tts_request.response_format,
                )

                # Return audio file
                ext = (
                    get_audio_extension(tts_request.response_format)
                    if get_audio_extension
                    else ".mp3"
                )
                return Response(
                    content=audio_data,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="speech_{tts_request.model[:20]}{ext}"',
                        "X-Sample-Rate": str(sample_rate),
                    },
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"TTS generation failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Speech generation failed: {str(e)}"
                )

    # STT (Speech-to-Text) API endpoints
    if AUDIO_AVAILABLE:

        @app.post("/v1/audio/transcriptions")
        async def create_transcription(request: Request):
            """
            Transcribe audio using Qwen3-Audio models.

            Compatible with: POST /v1/audio/transcriptions
            OpenAI-compatible audio transcription endpoint.

            Request body (multipart/form-data):
            - file: Audio file (mp3, mp4, mpeg, mpga, m4a, wav, webm)
            - model: Model to use (e.g., "qwen2-audio-7b-instruct")
            - language: Optional language code
            - prompt: Optional context to guide transcription
            - response_format: Response format (json, text, verbose_json, srt)
            - temperature: Sampling temperature (0-1)
            - timestamp_granularities: Optional ["word"] for word-level timestamps

            Returns:
                Transcription with text, language, duration, and timestamps
            """
            state: GatewayState = app.state.gateway

            # Parse multipart form data
            form = await request.form()
            file_field = form.get("file")
            model = form.get("model", "qwen2-audio-7b-instruct")
            language = form.get("language")  # Optional
            prompt = form.get("prompt")  # Optional
            response_format = form.get("response_format", "json")  # Optional
            temperature = float(form.get("temperature", "0.0"))  # Optional
            timestamp_granularities = form.get("timestamp_granularities")  # Optional

            if not file_field:
                raise HTTPException(status_code=400, detail="Audio file is required")

            # Convert UploadFile to our expected format
            class SimpleUploadFile:
                def __init__(self, filename, content, content_type):
                    self.filename = filename
                    self._content = content
                    self.content_type = content_type

                async def read(self):
                    return self._content

            audio_handler = get_audio_handler("https://api.openai.com")

            try:
                # Read audio file
                audio_data, audio_format = await read_audio_file(file_field)

                # Perform transcription
                result = await audio_handler.transcribe_audio(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    model=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format,
                    timestamp_granularities=[timestamp_granularities]
                    if timestamp_granularities
                    else None,
                )

                # Return based on response format
                if response_format == "text":
                    return Response(content=result["text"], media_type="text/plain")
                elif response_format == "srt":
                    return Response(
                        content=_generate_srt(result), media_type="text/plain"
                    )
                elif response_format == "verbose_json":
                    return JSONResponse(content=result)
                else:  # json
                    return JSONResponse(content={"text": result["text"]})

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Audio transcription failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Transcription failed: {str(e)}"
                )

        @app.post("/v1/audio/translations")
        async def create_translation(request: Request):
            """
            Translate audio to English text using Qwen3-Audio models.

            Compatible with: POST /v1/audio/translations
            OpenAI-compatible audio translation endpoint.

            Request body (multipart/form-data):
            - file: Audio file
            - model: Model to use
            - prompt: Optional context
            - response_format: Response format
            - temperature: Sampling temperature

            Returns:
                Translated text in English
            """
            state: GatewayState = app.state.gateway

            # Parse multipart form data
            form = await request.form()
            file_field = form.get("file")
            model = form.get("model", "qwen2-audio-7b-instruct")
            prompt = form.get("prompt")  # Optional
            response_format = form.get("response_format", "json")  # Optional
            temperature = float(form.get("temperature", "0.0"))  # Optional

            if not file_field:
                raise HTTPException(status_code=400, detail="Audio file is required")

            audio_handler = get_audio_handler("https://api.openai.com")

            try:
                # Read audio file
                audio_data, audio_format = await read_audio_file(file_field)

                # Perform transcription (Qwen3-Audio handles translation internally)
                result = await audio_handler.transcribe_audio(
                    audio_data=audio_data,
                    audio_format=audio_format,
                    model=model,
                    language="en",  # Target language for translation
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format,
                )

                # Translation result
                if response_format == "text":
                    return Response(content=result["text"], media_type="text/plain")
                else:
                    return JSONResponse(content={"text": result["text"]})

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Audio translation failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Translation failed: {str(e)}"
                )

        def _generate_srt(transcription: dict) -> str:
            """Generate SRT subtitle format from transcription."""
            if "words" not in transcription:
                # Simple SRT with full text
                return f"1\n00:00:00,000 --> 00:00:05,000\n{transcription['text']}\n\n"

            srt_lines = []
            duration = transcription.get("duration", 0)
            words = transcription["words"]

            # Group words into subtitle chunks (e.g., 10 words per subtitle)
            chunk_size = 10
            for i in range(0, len(words), chunk_size):
                chunk = words[i : i + chunk_size]
                start_time = chunk[0]["start"]
                end_time = chunk[-1]["end"]
                text = " ".join(w["word"] for w in chunk)

                srt_lines.append(f"{i // chunk_size + 1}")
                srt_lines.append(
                    f"{_format_srt_time(start_time)} --> {_format_srt_time(end_time)}"
                )
                srt_lines.append(text)
                srt_lines.append("")

            return "\n".join(srt_lines)

        def _format_srt_time(seconds: float) -> str:
            """Format seconds to SRT time format (HH:MM:SS,mmm)."""
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    # Vision API endpoints (Qwen3-VL image understanding)
    if VISION_AVAILABLE:

        @app.post("/v1/vision/chat")
        async def vision_chat(request: Request):
            """
            Analyze images using Qwen3-VL models.

            Compatible with OpenAI Vision API format.

            Request body:
            {
                "model": "qwen2-vl-7b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "https://..."}}
                        ]
                    }
                ],
                "max_tokens": 2048,
                "temperature": 0.7
            }

            Returns:
                Chat completion with image analysis
            """
            state: GatewayState = app.state.gateway

            body = await request.json()
            try:
                vision_request = VisionRequest(**body)
            except Exception as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid vision request: {e}"
                )

            vision_handler = get_vision_handler(
                state.config.backend_url or "https://api.openai.com"
            )

            try:
                # Process messages to extract images and text
                user_message = None
                images = []

                for msg in vision_request.messages:
                    if msg.role == "user":
                        user_message = msg
                        break

                if not user_message:
                    raise HTTPException(status_code=400, detail="No user message found")

                # Extract images and text from content
                content = user_message.content
                if isinstance(content, str):
                    text_prompt = content
                else:
                    text_prompt = ""
                    for item in content:
                        if isinstance(item, str):
                            text_prompt += item
                        elif hasattr(item, "type"):
                            if item.type == "text":
                                text_prompt += getattr(item, "text", "")
                            elif item.type == "image_url":
                                image_url_obj = getattr(item, "image_url", None)
                                if image_url_obj:
                                    url = getattr(image_url_obj, "url", "")
                                    if url:
                                        # Read image from URL
                                        (
                                            image_data,
                                            image_format,
                                        ) = await read_image_from_url(url)
                                        images.append((image_data, image_format))

                if not images:
                    raise HTTPException(
                        status_code=400, detail="No images found in request"
                    )

                # Use the first image for now (Qwen3-VL can handle multiple)
                image_data, image_format = images[0]

                # Analyze image
                result = await vision_handler.analyze_image(
                    image_data=image_data,
                    image_format=image_format,
                    prompt=text_prompt or "Describe this image in detail.",
                    model=vision_request.model,
                    max_tokens=vision_request.max_tokens,
                    temperature=vision_request.temperature,
                )

                return JSONResponse(content=result)

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Vision analysis failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Vision analysis failed: {str(e)}"
                )

    # Files API endpoints (Garage S3 storage)
    if FILES_AVAILABLE:

        @app.post("/v1/files")
        async def upload_file(request: Request):
            """
            Upload a file to Garage S3 storage.

            Compatible with Anthropic/OpenAI Files API.
            Accepts multipart/form-data with file and optional purpose field.

            Returns:
                File metadata including id, filename, bytes, created_at
            """
            state: GatewayState = app.state.gateway

            # Parse multipart form data
            form = await request.form()
            file_obj = request._form.get("file")

            if not file_obj:
                raise HTTPException(
                    status_code=400,
                    detail="No file provided. Use multipart/form-data with 'file' field.",
                )

            # Read file content
            content = await file_obj.read()

            # Get optional purpose field
            purpose = form.get("purpose")

            # Generate file ID
            file_id = generate_file_id()

            # Detect MIME type
            filename = file_obj.filename
            mime_type = get_mime_type(filename) or "application/octet-stream"

            # Get Garage client from state (create if not exists)
            if not hasattr(state, "garage_client") or state.garage_client is None:
                state.garage_client = get_garage_client()

            try:
                # Upload to Garage S3 with metadata
                result = await state.garage_client.put_file(
                    bucket="ai-gateway-files",
                    key=file_id,
                    content=content,
                    content_type=mime_type,
                    metadata={
                        "filename": filename,
                        "purpose": purpose or "",
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    },
                )

                logger.info(
                    f"File uploaded: {file_id} ({len(content)} bytes, {filename})"
                )

                return {
                    "id": result["id"],
                    "filename": result["filename"],
                    "bytes": result["bytes"],
                    "created_at": result["created_at"],
                    "object": "file",
                    "purpose": purpose,
                }

            except FileUploadError as e:
                logger.error(f"File upload failed: {e.message}")
                raise HTTPException(status_code=e.status_code or 500, detail=e.message)
            except Exception as e:
                logger.error(f"Unexpected error during file upload: {e}")
                raise HTTPException(
                    status_code=500, detail=f"File upload failed: {str(e)}"
                )

        @app.get("/v1/files/{file_id}")
        async def get_file(file_id: str):
            """
            Retrieve file metadata and content by ID.

            Compatible with Anthropic/OpenAI Files API.
            Returns file content with appropriate Content-Type header.

            Args:
                file_id: The ID of the file to retrieve

            Returns:
                File content with metadata headers
            """
            state: GatewayState = app.state.gateway

            # Get Garage client from state
            if not hasattr(state, "garage_client") or state.garage_client is None:
                state.garage_client = get_garage_client()

            try:
                # Get file content and metadata from Garage S3
                content, metadata = await state.garage_client.get_file(
                    bucket="ai-gateway-files", key=file_id
                )

                # Extract filename and mime type from metadata
                filename = metadata.get("filename", file_id)
                mime_type = metadata.get("mime_type", "application/octet-stream")

                logger.info(
                    f"File retrieved: {file_id} ({len(content)} bytes, {filename})"
                )

                # Return file with appropriate headers
                return Response(
                    content=content,
                    media_type=mime_type,
                    headers={
                        "Content-Disposition": f'attachment; filename="{filename}"',
                    },
                )

            except FileNotFoundError:
                logger.warning(f"File not found: {file_id}")
                raise HTTPException(
                    status_code=404, detail=f"File not found: {file_id}"
                )
            except FileStorageError as e:
                logger.error(f"File retrieval failed: {e.message}")
                raise HTTPException(status_code=e.status_code or 500, detail=e.message)
            except Exception as e:
                logger.error(f"Unexpected error during file retrieval: {e}")
                raise HTTPException(
                    status_code=500, detail=f"File retrieval failed: {str(e)}"
                )

        @app.delete("/v1/files/{file_id}")
        async def delete_file(file_id: str):
            """
            Delete a file by ID.

            Compatible with Anthropic/OpenAI Files API.

            Args:
                file_id: The ID of the file to delete

            Returns:
                Deletion confirmation
            """
            state: GatewayState = app.state.gateway

            # Get Garage client from state
            if not hasattr(state, "garage_client") or state.garage_client is None:
                state.garage_client = get_garage_client()

            try:
                # Delete file from Garage S3
                await state.garage_client.delete_file(
                    bucket="ai-gateway-files", key=file_id
                )

                logger.info(f"File deleted: {file_id}")

                return {
                    "id": file_id,
                    "object": "file",
                    "deleted": True,
                }

            except FileNotFoundError:
                # File doesn't exist - consider it already deleted
                logger.info(f"File already deleted (not found): {file_id}")
                return {
                    "id": file_id,
                    "object": "file",
                    "deleted": True,
                }
            except FileStorageError as e:
                logger.error(f"File deletion failed: {e.message}")
                raise HTTPException(status_code=e.status_code or 500, detail=e.message)
            except Exception as e:
                logger.error(f"Unexpected error during file deletion: {e}")
                raise HTTPException(
                    status_code=500, detail=f"File deletion failed: {str(e)}"
                )

    else:

        @app.post("/v1/files")
        async def upload_file_disabled():
            """File upload endpoint not available."""
            raise HTTPException(
                status_code=501,
                detail="Files API not available. Ensure files module is installed.",
            )

        @app.get("/v1/files/{file_id}")
        async def get_file_disabled():
            """File retrieval endpoint not available."""
            raise HTTPException(
                status_code=501,
                detail="Files API not available. Ensure files module is installed.",
            )

        @app.delete("/v1/files/{file_id}")
        async def delete_file_disabled():
            """File deletion endpoint not available."""
            raise HTTPException(
                status_code=501,
                detail="Files API not available. Ensure files module is installed.",
            )

    # Add metrics endpoint for Prometheus
    if PROMETHEUS_AVAILABLE:

        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint."""
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    else:

        @app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint (not available)."""
            raise HTTPException(
                status_code=501,
                detail="Prometheus metrics not available. Install prometheus-client package.",
            )

    # Add self-improvement system endpoints
    if SELF_IMPROVEMENT_AVAILABLE:
        try:
            self_improvement_router = create_self_improvement_router(
                enabled=True,
                auto_extract_patterns=True,
                min_confidence_for_update=0.7,
            )
            # Include all routes from self-improvement router
            for route in self_improvement_router.routes:
                app.routes.append(route)
            logger.info("Self-improvement endpoints registered")
        except Exception as e:
            import traceback

            logger.warning(f"Failed to initialize self-improvement system: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
    return app


async def stream_backend_response(
    openai_client,
    body: dict,
    pipeline: MiddlewarePipeline,
    context: dict,
    config: GatewayConfig,
    router,
    request_id: str,
    metrics_tracker: ModelMetricsTracker,
):
    """
    Stream backend response using OpenAI SDK with automatic failover.

    Args:
        openai_client: OpenAI client wrapper
        body: Request body
        pipeline: Middleware pipeline
        context: Request context
        config: Gateway configuration
        router: Router instance for tracking requests
        request_id: Request ID for tracking
        metrics_tracker: Metrics tracker for this request

    Yields:
        SSE formatted response chunks
    """
    try:
        # Extract parameters from request body
        messages = body.get("messages", [])
        model = body.get("model", "default")
        extra_params = {
            k: v
            for k, v in body.items()
            if k not in ["messages", "model", "stream", "backend", "response_format"]
        }

        # Get backend from route decision if available
        route_decision = context.get("route_decision")
        backend = route_decision.backend if route_decision else None

        # Create streaming chat completion with automatic failover
        stream = await openai_client.chat_completion(
            messages=messages,
            model=model,
            stream=True,
            backend=backend,
            **extra_params,
        )

        # Stream response chunks
        input_tokens = 0
        output_tokens = 0
        first_chunk = True
        async for chunk in stream:
            # Record first token time
            if first_chunk:
                metrics_tracker.record_first_token()
                first_chunk = False

            # Track tokens if usage info available
            if hasattr(chunk, "usage") and chunk.usage:
                if chunk.usage.prompt_tokens:
                    input_tokens = max(input_tokens, chunk.usage.prompt_tokens)
                if chunk.usage.completion_tokens:
                    output_tokens = max(output_tokens, chunk.usage.completion_tokens)

            # Format as SSE
            chunk_str = chunk.model_dump_json()
            yield f"data: {chunk_str}\n\n"

        # Record success metrics (streaming complete)
        total_tokens = input_tokens + output_tokens
        if total_tokens > 0:
            # Calculate latency from the tracker's start time
            import time

            latency_ms = (time.time() - metrics_tracker.start_time) * 1000
            metrics_tracker.record_success(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )

        # Notify circuit breaker of success
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_success()

    except OpenAIBackendError as e:
        logger.error(f"Backend error in streaming request: {e}")

        # Record error metrics
        metrics_tracker.record_error("backend_error")

        # Notify circuit breaker of failure
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        # Yield error as SSE event (proper JSON with escaped single quotes)
        yield f"data: {json.dumps({'error': str(e).replace("'", '&#39;')})}\n\n"
    except Exception as e:
        logger.error(f"Unexpected error in streaming request: {e}")

        # Record error metrics
        metrics_tracker.record_error("unexpected_error")

        # Notify circuit breaker of failure
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        yield f"data: {json.dumps({'error': f'Unexpected error: {str(e)}'})}\n\n"
    finally:
        # Always clean up request tracking
        router.track_request_end(request_id)
        # Signal GPU scheduler that AI workload is stopping
        gpu_scheduler.notify_ai_stopping()


async def stream_backend_response_with_tools(
    openai_client,
    body: dict,
    pipeline: MiddlewarePipeline,
    context: dict,
    config: GatewayConfig,
    router,
    request_id: str,
    metrics_tracker: ModelMetricsTracker,
    mcp_broker,
):
    """
    Stream backend response with tool calling support using OpenAI SDK.

    Implements an agentic loop that:
    1. Streams the model response
    2. Detects when the model requests tool calls
    3. Executes tools via MCP broker
    4. Feeds tool results back to the model
    5. Continues streaming until stop_reason == "end_turn"

    Args:
        openai_client: OpenAI client wrapper
        body: Request body
        pipeline: Middleware pipeline
        context: Request context (must contain GatewayState with mcp_broker)
        config: Gateway configuration
        router: Router instance for tracking requests
        request_id: Request ID for tracking
        metrics_tracker: Metrics tracker for this request
        mcp_broker: MCP broker for tool execution

    Yields:
        SSE formatted response chunks
    """
    try:
        messages = body.get("messages", [])
        model = body.get("model", "default")
        tools = body.get("tools", [])
        extra_params = {
            k: v
            for k, v in body.items()
            if k not in ["messages", "model", "stream", "tools", "backend"]
        }

        # Get backend from route decision if available
        route_decision = context.get("route_decision")
        backend = route_decision.backend if route_decision else None

        # Agentic loop: continue until model finishes
        max_iterations = 10  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Create streaming chat completion
            stream = await openai_client.chat_completion(
                messages=messages,
                model=model,
                stream=True,
                backend=backend,
                tools=tools if tools else None,
                **extra_params,
            )

            # Accumulate chunks to detect tool calls
            chunks_buffer = []
            has_tool_calls = False
            tool_calls_chunk = None
            first_chunk = True

            async for chunk in stream:
                # Record first token time
                if first_chunk:
                    metrics_tracker.record_first_token()
                    first_chunk = False

                # Buffer chunks for tool call detection
                chunks_buffer.append(chunk)

                # Check for tool calls in this chunk
                if hasattr(chunk, "choices") and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, "delta") and choice.delta:
                        delta = choice.delta
                        # Check if tool_calls are present in delta
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            has_tool_calls = True
                            tool_calls_chunk = chunk

                # Stream chunk to client immediately for low latency
                chunk_str = chunk.model_dump_json()
                yield f"data: {chunk_str}\n\n"

            # After streaming completes, check if we need to execute tools
            if not has_tool_calls:
                # No tool calls, we're done
                break

            # Extract tool calls from the accumulated response
            # Build a complete response-like object from chunks
            accumulated_response = {
                "choices": [{"message": {"tool_calls": []}}],
                "finish_reason": "tool_calls",
            }

            # Extract tool calls from the chunk that had them
            if tool_calls_chunk and hasattr(tool_calls_chunk, "choices"):
                choice = tool_calls_chunk.choices[0]
                if hasattr(choice, "delta") and hasattr(choice.delta, "tool_calls"):
                    accumulated_response["choices"][0]["message"]["tool_calls"] = [
                        tc.model_dump() for tc in choice.delta.tool_calls
                    ]

            # Extract tool calls using our utility
            extracted_calls = extract_tool_calls_openai(accumulated_response)

            if not extracted_calls:
                # Model signaled tool_calls but we couldn't extract them
                logger.warning("Tool calls detected but extraction failed")
                break

            logger.info(
                f"Executing {len(extracted_calls)} tool calls (iteration {iteration})"
            )

            # Execute each tool call
            tool_results = []
            for tool_call in extracted_calls:
                tool_id = tool_call.get("id")
                tool_name = tool_call.get("name")
                tool_args_str = tool_call.get("arguments", "{}")

                # Parse arguments
                try:
                    import json

                    tool_args = (
                        json.loads(tool_args_str)
                        if isinstance(tool_args_str, str)
                        else tool_args
                    )
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse tool arguments: {tool_args_str}")
                    tool_args = {}

                # Discover which MCP server provides this tool via broker
                server_name = _infer_mcp_server_for_tool(tool_name, mcp_broker)

                if not server_name:
                    # Try to execute the tool directly via MCP broker
                    error_msg = f"Tool '{tool_name}' not found in any MCP server"
                    logger.warning(error_msg)
                    tool_results.append(
                        create_tool_result_openai(
                            tool_call_id=tool_id, result=error_msg, tool_type="error"
                        )
                    )
                    continue

                try:
                    # Execute tool via MCP broker
                    result = await mcp_broker.call_tool(
                        server_name=server_name,
                        tool_name=tool_name,
                        arguments=tool_args,
                    )

                    # Format result as string
                    if isinstance(result, dict):
                        result_str = json.dumps(result)
                    else:
                        result_str = str(result)

                    logger.info(
                        f"Tool {tool_name} executed successfully, result length: {len(result_str)}"
                    )

                    tool_results.append(
                        create_tool_result_openai(
                            tool_call_id=tool_id, result=result_str
                        )
                    )

                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    tool_results.append(
                        create_tool_result_openai(
                            tool_call_id=tool_id, result=f"Error: {str(e)}"
                        )
                    )

            # Append tool results as a new user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

                # Continue the loop to get the model's final response
                continue
            else:
                # No tool results (all failed), break
                break

        # Record success metrics (agentic loop complete)
        latency_ms = (time.time() - metrics_tracker.start_time) * 1000
        metrics_tracker.record_success(
            input_tokens=0,  # Token tracking in agentic loops needs more work
            output_tokens=0,
            total_tokens=0,
            latency_ms=latency_ms,
        )

        # Notify circuit breaker of success
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_success()

    except OpenAIBackendError as e:
        logger.error(f"Backend error in agentic streaming request: {e}")
        metrics_tracker.record_error("backend_error")
        yield f"data: {json.dumps({'error': str(e).replace("'", '&#39;')})}\n\n"
    except Exception as e:
        logger.error(f"Unexpected error in agentic streaming request: {e}")
        metrics_tracker.record_error("unexpected_error")
        yield f"data: {json.dumps({'error': f'Unexpected error: {str(e)}'})}\n\n"
    finally:
        router.track_request_end(request_id)
        gpu_scheduler.notify_ai_stopping()


def _infer_mcp_server_for_tool(tool_name: str, mcp_broker) -> Optional[str]:
    """
    Infer which MCP server can handle the given tool.

    Uses the MCP broker's list_tools to find the server that provides
    the tool. Returns None if tool not found.

    Args:
        tool_name: Name of the tool to locate
        mcp_broker: MCP broker instance

    Returns:
        Server name that can execute the tool, or None
    """
    try:
        # Get all available servers and their tools
        import asyncio

        async def find_server():
            servers = await mcp_broker.list_servers()
            for server_name in servers.get("servers", {}):
                try:
                    tools_response = await mcp_broker.get_tools(server_name)
                    tools = tools_response.get("tools", [])
                    for tool in tools:
                        if tool.get("name") == tool_name:
                            return server_name
                except Exception:
                    continue
            return None

        # Run the async find function
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(find_server())

    except Exception as e:
        logger.warning(f"Failed to infer MCP server for tool {tool_name}: {e}")
        return None


async def try_backends_with_failover(
    config: GatewayConfig,
    request_headers: dict,
    endpoint: str,
    method: str = "POST",
    content: dict = None,
    timeout: float = 300.0,
) -> tuple[httpx.Response, str]:
    """
    Try primary and fallback backends in order until one succeeds.

    Args:
        config: Gateway configuration
        request_headers: Original request headers
        endpoint: API endpoint (e.g., "/v1/chat/completions")
        method: HTTP method
        content: Request body for POST requests
        timeout: Request timeout in seconds

    Returns:
        Tuple of (response, backend_url_used)

    Raises:
        httpx.HTTPError: If all backends fail
    """
    # Build list of backends to try
    backends_to_try = [("primary", config.backend_url, config.backend_type)]

    # Add fallback backends (assuming they're ZAI for now)
    for i, fallback_url in enumerate(config.get_backend_fallback_urls()):
        backends_to_try.append(("fallback", fallback_url, "zai"))

    last_error = None

    for backend_type_name, backend_url, backend_api_type in backends_to_try:
        try:
            logger.info(f"Attempting {backend_type_name} backend: {backend_url}")

            # Build headers for this backend, preserving User-Agent
            headers = {
                k: v
                for k, v in request_headers.items()
                if k.lower()
                not in {
                    "host",
                    "content-length",
                    "content-encoding",
                    "transfer-encoding",
                }
            }

            # Log User-Agent for debugging (only at DEBUG level)
            if "user-agent" in {k.lower(): k for k in headers.keys()}:
                ua_key = next(k for k in headers.keys() if k.lower() == "user-agent")
                logger.debug(f"Forwarding User-Agent: {headers[ua_key][:100]}")

            # Add authentication for this backend
            if "authorization" not in {k.lower() for k in headers.keys()}:
                if backend_api_type == "zai":
                    api_key = config.get_zai_api_key()
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                elif backend_api_type == "zai":
                    api_key = config.get_zai_api_key()
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    else:
                        logger.warning("ZAI API key not found for fallback backend")
                elif backend_api_type == "pollinations":
                    api_key = config.get_pollinations_api_key()
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    else:
                        logger.warning(
                            "Pollinations API key not found for fallback backend"
                        )

            logger.info(
                f"Request headers for {backend_type_name} backend: Authorization={'Bearer ' + (headers.get('Authorization', 'NO-AUTH')[:20] + '...' if 'Authorization' in headers else 'NOT SET')}"
            )

            async with httpx.AsyncClient(timeout=timeout) as client:
                # For ZAI, convert OpenAI-style endpoints and clean request body
                if backend_api_type == "zai":
                    # ZAI uses /chat/completions instead of /v1/chat/completions
                    zai_endpoint = (
                        endpoint.replace("/v1/", "/")
                        if endpoint.startswith("/v1/")
                        else endpoint
                    )
                    url = f"{backend_url}{zai_endpoint}"

                    # ZAI doesn't understand chat_template_kwargs — strip it
                    # and ensure enable_thinking is set as top-level param
                    if isinstance(content, dict):
                        content.pop("chat_template_kwargs", None)
                        # If enable_thinking not already set, default to false
                        if "enable_thinking" not in content:
                            content["enable_thinking"] = False
                else:
                    url = f"{backend_url}{endpoint}"

                # Debug logging for ZAI (only at DEBUG level)
                if backend_api_type == "zai" and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"ZAI URL: {url}")
                    logger.debug(
                        f"ZAI Headers: Authorization={headers.get('Authorization', 'MISSING')[:30]}..."
                    )
                    logger.debug(f"ZAI Body model: {content.get('model', 'NO_MODEL')}")

                # Debug logging for Pollinations (only at DEBUG level)
                if backend_api_type == "pollinations" and logger.isEnabledFor(
                    logging.DEBUG
                ):
                    logger.debug(f"Pollinations URL: {url}")
                    logger.debug(
                        f"Pollinations Headers: Authorization={headers.get('Authorization', 'MISSING')[:30]}..."
                    )
                    logger.debug(
                        f"Pollinations Body model: {content.get('model', 'NO_MODEL')}"
                    )

                if method.upper() == "POST":
                    response = await client.post(
                        url,
                        json=content,
                        headers=headers,
                    )
                else:  # GET
                    response = await client.get(
                        url,
                        headers=headers,
                    )

                # Log response status
                logger.info(
                    f"{backend_type_name} backend response: HTTP {response.status_code}"
                )

                # Debug logging for ZAI responses (only at DEBUG level)
                if backend_api_type == "zai" and logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"ZAI Response status: {response.status_code}")
                    if response.status_code != 200:
                        try:
                            logger.debug(f"ZAI Response body: {response.text[:500]}")
                        except Exception:
                            pass

                # Debug logging for Pollinations responses (only at DEBUG level)
                if backend_api_type == "pollinations" and logger.isEnabledFor(
                    logging.DEBUG
                ):
                    logger.debug(
                        f"Pollinations Response status: {response.status_code}"
                    )
                    if response.status_code != 200:
                        try:
                            logger.debug(
                                f"Pollinations Response body: {response.text[:500]}"
                            )
                        except Exception:
                            pass

                # If we got here, the request succeeded (connected, even if 4xx/5xx)
                return response, backend_url

        except (httpx.ConnectError, httpx.TimeoutException, httpx.ConnectTimeout) as e:
            logger.warning(
                f"{backend_type_name} backend {backend_url} failed: {str(e)}"
            )
            last_error = e
            continue
        except Exception as e:
            logger.warning(
                f"{backend_type_name} backend {backend_url} failed with unexpected error: {str(e)}"
            )
            last_error = e
            continue

    # All backends failed
    logger.error(f"All backends failed. Last error: {last_error}")
    raise last_error or httpx.ConnectError("All backends unavailable")


async def handle_non_streaming_request(
    openai_client,
    body: dict,
    pipeline: MiddlewarePipeline,
    context: dict,
    config: GatewayConfig,
    metrics_tracker: ModelMetricsTracker,
):
    """
    Handle non-streaming request using OpenAI SDK with automatic failover.

    Args:
        openai_client: OpenAI client wrapper
        body: Request body
        pipeline: Middleware pipeline
        context: Request context
        config: Gateway configuration
        metrics_tracker: Metrics tracker for this request

    Returns:
        JSON response
    """
    import time

    start_time = time.time()

    try:
        # Extract parameters from request body
        messages = body.get("messages", [])
        model = body.get("model", "default")
        extra_params = {
            k: v
            for k, v in body.items()
            if k not in ["messages", "model", "stream", "backend", "response_format"]
        }

        # Get backend from route decision if available
        route_decision = context.get("route_decision")
        backend = route_decision.backend if route_decision else None

        # Create chat completion with automatic failover
        response = await openai_client.chat_completion(
            messages=messages,
            model=model,
            stream=False,
            backend=backend,
            **extra_params,
        )

        # Convert OpenAI response object to dict for JSON serialization
        response_data = response.model_dump()
        # Sanitize markdown fences from ALL non-streaming responses
        # Models (Qwen, GLM, Gemma) frequently wrap JSON in ```json ... ``` fences
        # even without response_format, breaking consumers like Vane/Perplexica
        from ai_inference_gateway.utils import strip_markdown_json_fences
        for choice in response_data.get("choices", []):
            content = choice.get("message", {}).get("content")
            if content:
                choice["message"]["content"] = strip_markdown_json_fences(content)
                # TEMP: Debug log content for Vane source extraction diagnosis
                logger.info(f"GATEWAY RESPONSE content preview: {repr(choice['message']['content'][:200])}")
                logger.info(f"GATEWAY RESPONSE reasoning: {'YES' if choice['message'].get('reasoning_content') else 'none'}")

        # Calculate actual processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Record success metrics
        # Extract token usage from response
        usage = response_data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        metrics_tracker.record_success(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            latency_ms=processing_time_ms,
        )

        # Record cost tracking (non-blocking)
        try:
            cost_tracker = context.get("cost_tracker")
            if cost_tracker and total_tokens > 0:
                import asyncio
                route_decision_ctx = context.get("route_decision")
                asyncio.create_task(cost_tracker.record(
                    model=model,
                    agent_key=context.get("request_headers", {}).get("x-api-key", ""),
                    backend=route_decision_ctx.backend if route_decision_ctx else "",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ))
        except Exception as cost_err:
            logger.debug(f"Cost tracking failed: {cost_err}")

        # Add gateway metadata including routing information
        route_decision = context.get("route_decision")
        if route_decision:
            response_data["gateway_metadata"] = {
                "processing_time_ms": round(processing_time_ms, 2),
                "router": {
                    "model": route_decision.model,
                    "backend": route_decision.backend,
                    "reason": route_decision.reason,
                    "specialization": (
                        route_decision.specialization.value
                        if route_decision.specialization
                        else None
                    ),
                    "estimated_tokens": route_decision.estimated_tokens,
                    "expected_latency_ms": route_decision.expected_latency_ms,
                },
            }
            # Remove route_decision from context to avoid serialization issues
            context.pop("route_decision", None)

        # Process response through middleware pipeline (reverse order)
        response_data = await pipeline.process_response(response_data, context)

        # Notify circuit breaker of success
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_success()

        return JSONResponse(content=response_data, status_code=200)

    except OpenAIBackendError as e:
        logger.error(f"Backend error: {e}")

        # Record error metrics
        metrics_tracker.record_error("backend_error")

        # Notify circuit breaker of failure
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        # Extract status code from error message if possible
        error_str = str(e)
        status_code = 503
        if "401" in error_str:
            status_code = 401
        elif "429" in error_str:
            status_code = 429

        raise HTTPException(status_code=status_code, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")

        # Record error metrics
        metrics_tracker.record_error("unexpected_error")

        # Notify circuit breaker of failure
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


async def stream_anthropic_response(
    openai_client,
    body: dict,
    pipeline: MiddlewarePipeline,
    context: dict,
    config: GatewayConfig,
    original_model: str,
    thinking_intensity: Optional[str],
    request_id: str,
    metrics_tracker: ModelMetricsTracker,
):
    """
    Handle streaming Anthropic API response.

    Converts OpenAI streaming chunks to Anthropic format.
    """
    import json
    import time

    start_time = time.time()
    first_chunk_sent = False

    try:
        # Extract parameters
        messages = body.get("messages", [])
        model = body.get("model", "default")
        extra_params = {
            k: v
            for k, v in body.items()
            if k not in ["messages", "model", "stream", "backend", "response_format"]
        }

        route_decision = context.get("route_decision")
        backend = route_decision.backend if route_decision else None

        # Get the streaming response from OpenAI client
        async for chunk in await openai_client.stream_chat_completion(
            messages=messages,
            model=model,
            backend=backend,
            **extra_params,
        ):
            if not first_chunk_sent:
                first_chunk_sent = True
                # Send initial event with request metadata
                event_data = {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{request_id[:8]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": original_model,
                        "stop_reason": None,
                    },
                }
                yield f"event: message_start\ndata: {json.dumps(event_data)}\n\n"

            # Convert OpenAI chunk to Anthropic format
            if chunk.get("choices"):
                choice = chunk["choices"][0]
                delta = choice.get("message", {})

                # Content block
                if "content" in delta and delta["content"]:
                    content_event = {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text", "text": delta["content"]},
                    }
                    yield f"event: content_block_delta\ndata: {json.dumps(content_event)}\n\n"

                # Tool calls
                if "tool_calls" in delta:
                    for tool_call in delta["tool_calls"]:
                        tool_event = {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {
                                "type": "tool_use",
                                "id": tool_call.get("id", ""),
                                "name": tool_call.get("function", {}).get("name", ""),
                                "input": tool_call.get("function", {}).get(
                                    "arguments", "{}"
                                ),
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(tool_event)}\n\n"

                # Finish reason
                if "finish_reason" in delta:
                    stop_event = {
                        "type": "content_block_stop",
                        "index": 0,
                    }
                    yield f"event: content_block_stop\ndata: {json.dumps(stop_event)}\n\n"

                    # Send message_stop event
                    final_event = {
                        "type": "message_stop",
                        "message": {
                            "id": f"msg_{request_id[:8]}",
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": original_model,
                            "stop_reason": delta["finish_reason"],
                        },
                    }
                    yield f"event: message_stop\ndata: {json.dumps(final_event)}\n\n"

        # Record metrics
        processing_time_ms = (time.time() - start_time) * 1000
        metrics_tracker.record_success(
            input_tokens=0,  # Would need to accumulate from chunks
            output_tokens=0,
            total_tokens=0,
            latency_ms=processing_time_ms,
        )

    except Exception as e:
        logger.error(f"Error in Anthropic streaming: {e}")
        metrics_tracker.record_error("streaming_error")

        # Send error event
        error_event = {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": str(e),
            },
        }
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"


async def handle_anthropic_non_streaming(
    openai_client,
    body: dict,
    pipeline: MiddlewarePipeline,
    context: dict,
    config: GatewayConfig,
    original_model: str,
    thinking_intensity: Optional[str],
    metrics_tracker: ModelMetricsTracker,
):
    """
    Handle non-streaming Anthropic API request.

    Converts OpenAI response to Anthropic format with thinking support.
    """
    import time

    start_time = time.time()

    try:
        # Extract parameters
        messages = body.get("messages", [])
        model = body.get("model", "default")
        extra_params = {
            k: v
            for k, v in body.items()
            if k not in ["messages", "model", "stream", "backend", "response_format"]
        }

        route_decision = context.get("route_decision")
        backend = route_decision.backend if route_decision else None

        # Get response from OpenAI client
        response = await openai_client.chat_completion(
            messages=messages,
            model=model,
            stream=False,
            backend=backend,
            **extra_params,
        )

        # Convert to dict for processing
        response_data = response.model_dump()

        # Process through middleware pipeline
        response_data = await pipeline.process_response(response_data, context)

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Extract usage for metrics
        usage = response_data.get("usage", {})
        metrics_tracker.record_success(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=processing_time_ms,
        )

        # Convert to Anthropic format
        choice = response_data.get("choices", [{}])[0]
        message = choice.get("message", {})

        # Build content blocks array
        content_blocks = []

        # Main text content
        text_content = message.get("content", "")

        # Handle reasoning_content - some models use this instead of content
        if not text_content:
            text_content = message.get("reasoning_content", "")

        if text_content:
            content_blocks.append({"type": "text", "text": text_content})

        # Tool calls
        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": tool_call.get("function", {}).get("arguments", "{}"),
                    }
                )

        # Handle extended thinking metadata (for Anthropic compatibility)
        thinking_content = None
        reasoning_text = message.get("reasoning_content", "")
        if reasoning_text:
            thinking_content = {
                "thinking": reasoning_text,
                "tokens": response_data.get(
                    "reasoning_tokens", len(reasoning_text) // 4
                ),  # Rough estimate
            }

        # Build Anthropic response
        anthropic_response = {
            "id": response_data.get("id", f"msg_{time.time()}"),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": original_model,
            "stop_reason": choice.get("finish_reason", "stop"),
            "stop_sequence": None,
            "usage": {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            },
            "gateway_metadata": {
                "processing_time_ms": round(processing_time_ms, 2),
                "router": {
                    "model": route_decision.model if route_decision else model,
                    "backend": route_decision.backend if route_decision else "unknown",
                    "reason": route_decision.reason
                    if route_decision
                    else "Anthropic API",
                    "specialization": (
                        route_decision.specialization.value
                        if route_decision and route_decision.specialization
                        else None
                    ),
                    "estimated_tokens": route_decision.estimated_tokens
                    if route_decision
                    else 0,
                    "expected_latency_ms": route_decision.expected_latency_ms
                    if route_decision
                    else 0,
                },
                "thinking": {
                    "intensity": thinking_intensity,
                    "budget": context.get("thinking_budget"),
                }
                if thinking_intensity
                else None,
            },
        }

        # Add extended thinking if present
        if thinking_content:
            anthropic_response["extended_thinking"] = thinking_content

        # Notify circuit breaker of success
        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_success()

        return JSONResponse(content=anthropic_response, status_code=200)

    except OpenAIBackendError as e:
        logger.error(f"Backend error in Anthropic request: {e}")
        metrics_tracker.record_error("backend_error")

        # Log error episode to self-improvement system
        if SELF_IMPROVEMENT_AVAILABLE:
            try:
                engine = get_self_improvement_engine()
                import asyncio

                asyncio.create_task(
                    engine.log_error(
                        error_type="backend_error",
                        error_message=str(e),
                        context={
                            "endpoint": "/v1/messages",
                            "model": body.get("model"),
                        },
                        resolution="circuit_breaker_triggered",
                    )
                )
            except Exception:
                pass  # Don't fail logging

        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error in Anthropic request: {e}")
        metrics_tracker.record_error("unexpected_error")

        # Log error episode to self-improvement system
        if SELF_IMPROVEMENT_AVAILABLE:
            try:
                engine = get_self_improvement_engine()
                import asyncio

                asyncio.create_task(
                    engine.log_error(
                        error_type="unexpected_error",
                        error_message=str(e),
                        context={
                            "endpoint": "/v1/messages",
                            "model": body.get("model"),
                        },
                    )
                )
            except Exception:
                pass  # Don't fail logging

        if config.middleware.circuit_breaker.enabled:
            for middleware in pipeline.middleware:
                if isinstance(middleware, CircuitBreaker):
                    await middleware.on_failure()

        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


def main():
    """Main entry point for running the gateway."""
    import uvicorn

    config = GatewayConfig()
    app = create_app(config)

    uvicorn.run(
        app, host=config.gateway_host, port=config.gateway_port, log_level="info"
    )


if __name__ == "__main__":
    main()

# Create app for uvicorn when imported as module
# This is needed when uvicorn imports with: ai_inference_gateway.main:app
try:
    app = create_app()
    if app is None:
        raise RuntimeError("Failed to create FastAPI app - check logs for errors")
except Exception as e:
    logger.error(f"Failed to create app: {e}")
    import traceback

    traceback.print_exc()
    raise
