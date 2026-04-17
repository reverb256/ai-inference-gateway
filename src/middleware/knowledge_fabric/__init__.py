"""
Knowledge Fabric - Unified Knowledge Retrieval System

Combines RAG, MCP web search, SearXNG, and code search into one
coherent system with semantic routing, parallel retrieval, and RRF fusion.

Architecture:
- Core: Data structures and base abstractions
- Routing: Semantic query classification and source selection
- Sources: Adapters for heterogeneous knowledge systems
- Fusion: Reciprocal Rank Fusion (RRF) for multi-source merging
- Fabric: Main middleware orchestrator
"""

from .core import (
    SourcePriority,
    SourceCapability,
    KnowledgeChunk,
    KnowledgeResult,
    FabricContext,
    RetrieveFunc,
    KnowledgeSource,
)

from .routing import SemanticRouter, QueryIntent, RoutingDecision
from .fusion import (
    RRFFusion,
    ContextSynthesizer,
    create_fusion,
    create_synthesizer,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
    execute_with_circuit_breaker,
    create_circuit_breaker_registry,
)
from .metrics import (
    KnowledgeFabricMetrics,
    create_metrics,
    get_metrics,
)
from .fabric import (
    KnowledgeFabricMiddleware,
    create_knowledge_fabric,
    FABRIC_CONTEXT_KEY,
    KNOWLEDGE_CONTEXT_KEY,
)
from .sources import (
    RAGKnowledgeSource,
    create_rag_source,
    WebSearchKnowledgeSource,
    create_web_search_source,
    SearXNGKnowledgeSource,
    create_searxng_source,
    CodeSearchKnowledgeSource,
    create_code_search_source,
)

__all__ = [
    # Core
    "SourcePriority",
    "SourceCapability",
    "KnowledgeChunk",
    "KnowledgeResult",
    "FabricContext",
    "RetrieveFunc",
    "KnowledgeSource",
    # Routing
    "SemanticRouter",
    "QueryIntent",
    "RoutingDecision",
    # Fusion
    "RRFFusion",
    "ContextSynthesizer",
    "create_fusion",
    "create_synthesizer",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitState",
    "execute_with_circuit_breaker",
    "create_circuit_breaker_registry",
    # Metrics
    "KnowledgeFabricMetrics",
    "create_metrics",
    "get_metrics",
    # Fabric (Main Orchestrator)
    "KnowledgeFabricMiddleware",
    "create_knowledge_fabric",
    "FABRIC_CONTEXT_KEY",
    "KNOWLEDGE_CONTEXT_KEY",
    # Source Adapters
    "RAGKnowledgeSource",
    "create_rag_source",
    "WebSearchKnowledgeSource",
    "create_web_search_source",
    "SearXNGKnowledgeSource",
    "create_searxng_source",
    "CodeSearchKnowledgeSource",
    "create_code_search_source",
]
