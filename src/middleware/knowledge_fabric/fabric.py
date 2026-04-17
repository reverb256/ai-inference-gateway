"""
Knowledge Fabric - Main Orchestrator

Coordinates semantic routing, parallel retrieval, and RRF fusion
to provide unified knowledge access across multiple sources.

This is the main entry point for the Knowledge Fabric system.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from fastapi import Request, HTTPException

from ..base import Middleware
from .core import (
    KnowledgeSource,
    FabricContext,
    KnowledgeResult,
)
from .routing import SemanticRouter
from .fusion import RRFFusion, ContextSynthesizer
from .circuit_breaker import (
    execute_with_circuit_breaker,
    create_circuit_breaker_registry,
)
from .metrics import (
    create_metrics,
)
from .sources import (
    RAGKnowledgeSource,
    WebSearchKnowledgeSource,
    SearXNGKnowledgeSource,
    CodeSearchKnowledgeSource,
    BrainWikiSource,
)

try:
    from prometheus_client import CollectorRegistry

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = None

logger = logging.getLogger(__name__)

# Context key for storing fabric context
FABRIC_CONTEXT_KEY = "knowledge_fabric_context"
KNOWLEDGE_CONTEXT_KEY = "knowledge_context"


class KnowledgeFabricMiddleware(Middleware):
    """
    Main orchestrator for the Knowledge Fabric system.

    Coordinates:
    1. Semantic query classification (intent detection)
    2. Source selection based on capabilities and priorities
    3. Parallel retrieval from multiple sources
    4. RRF (Reciprocal Rank Fusion) for result merging
    5. Context synthesis for LLM consumption

    This middleware extends the Middleware ABC and integrates
    with the gateway's pipeline architecture.
    """

    def __init__(
        self,
        sources: Optional[List[KnowledgeSource]] = None,
        rrf_k: int = 60,
        enabled: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Knowledge Fabric middleware.

        Args:
            sources: List of knowledge sources (auto-initialized if None)
            rrf_k: RRF constant (higher = more weight to lower ranks)
            enabled: Whether the middleware is active
            config: Optional configuration dictionary
        """
        self._enabled = enabled
        self.config = config or {}

        # Initialize knowledge sources
        if sources is None:
            sources = self._create_default_sources()

        self.sources = sources

        # Initialize components
        self.router = SemanticRouter(sources)
        self.fusion = RRFFusion(k=rrf_k)
        self.synthesizer = ContextSynthesizer()

        # Initialize Prometheus metrics with isolated registry
        # This prevents duplicate registration errors when multiple instances exist
        if PROMETHEUS_AVAILABLE and CollectorRegistry is not None:
            metrics_registry = CollectorRegistry()
            self.metrics = create_metrics(registry=metrics_registry)
        else:
            self.metrics = create_metrics()

        # Initialize circuit breaker for source resilience
        circuit_config = self.config.get("circuit_breaker", {})
        self.circuit_registry = create_circuit_breaker_registry(
            failure_threshold=circuit_config.get("failure_threshold", 5),
            timeout=circuit_config.get("timeout", 60.0),
            success_threshold=circuit_config.get("success_threshold", 2),
            metrics=self.metrics,
        )

        # Store sources by name for quick access
        self._sources_by_name = {s.name: s for s in sources}

        # Record initial active sources count
        self.metrics.set_active_sources(len(sources))

        logger.info(
            f"KnowledgeFabric initialized with {len(sources)} sources: "
            f"{[s.name for s in sources]}"
        )

    def _create_default_sources(self) -> List[KnowledgeSource]:
        """
        Create default knowledge sources from configuration.

        Sources are created in priority order (CRITICAL first).
        """
        sources = []

        # RAG source (HIGH priority - internal knowledge base)
        if self.config.get("rag_enabled", False):
            # Import here to avoid circular imports

            # search_service will be injected via context during request
            rag_source = RAGKnowledgeSource(
                search_service=None,  # Injected at runtime
                collection="knowledge-base",
                max_chunks=self.config.get("rag_top_k", 5),
            )
            sources.append(rag_source)
            logger.info("Added RAGKnowledgeSource")

        # Code search source (CRITICAL priority - internal codebase)
        code_search = CodeSearchKnowledgeSource(
            search_paths=self.config.get(
                "code_search_paths",
                ["/etc/nixos"],  # Default to NixOS config
            ),
            max_results=self.config.get("code_max_results", 5),
        )
        sources.append(code_search)
        logger.info("Added CodeSearchKnowledgeSource")

        # SearXNG source (MEDIUM priority - meta-search)
        searxng = SearXNGKnowledgeSource(
            searxng_url=self.config.get("searxng_url", "http://127.0.0.1:30080"),
            max_results=self.config.get("searxng_max_results", 5),
        )
        sources.append(searxng)
        logger.info("Added SearXNGKnowledgeSource")

        # Brain wiki source (HIGH priority - local wiki knowledge base)
        # Keyword-overlap search over ~/brain/wiki/ markdown pages.
        # No embeddings required — pure token matching with recency boost.
        if self.config.get("brain_wiki_enabled", True):
            brain_source = BrainWikiSource(
                brain_wiki_path=self.config.get(
                    "brain_wiki_path",
                    str(__import__("pathlib").Path.home() / "brain" / "wiki"),
                ),
                max_results=self.config.get("brain_wiki_max_results", 5),
                max_chunk_chars=self.config.get("brain_wiki_max_chunk_chars", 2000),
            )
            sources.append(brain_source)
            logger.info("Added BrainWikiSource")

        # Web search source (MEDIUM priority - MCP web_search_prime)
        web_search = WebSearchKnowledgeSource(
            mcp_url=self.config.get("mcp_url", "http://127.0.0.1:8080/mcp/call"),
            max_results=self.config.get("web_max_results", 5),
        )
        sources.append(web_search)
        logger.info("Added WebSearchKnowledgeSource")

        return sources

    @property
    def enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self._enabled

    async def process_request(
        self,
        request: Request,
        context: dict,
    ) -> tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request to add knowledge context.

        This method:
        1. Extracts the user query from the request
        2. Classifies the query intent (CODE, FACTUAL, etc.)
        3. Selects appropriate knowledge sources
        4. Executes parallel retrieval
        5. Fuses results using RRF
        6. Synthesizes context for the LLM
        7. Injects context into the request context

        Args:
            request: The FastAPI Request object
            context: Request context dict for passing state

        Returns:
            Tuple of (should_continue, optional_error)
        """
        import sys

        print(
            f"[DEBUG] KnowledgeFabricMiddleware.process_request called, enabled={self.enabled}",
            file=sys.stderr,
            flush=True,
        )
        if not self.enabled:
            return True, None

        try:
            # Extract query from request body (will be populated by parsing middleware)
            query = await self._extract_query(request, context)
            if not query:
                return True, None

            # Start query timing
            query_timer = self.metrics.time_query()
            query_timer.__enter__()

            # Skip for very short queries (likely greetings)
            if len(query.strip()) < 10:
                logger.debug(f"Query too short for knowledge retrieval: {query[:50]}")
                self.metrics.record_query_skipped(reason="query_too_short")
                query_timer.__exit__(None, None, None)
                return True, None

            logger.info(f"Processing knowledge query: {query[:100]}")

            # Step 1: Classify query intent
            routing_decision = self.router.classify(query)
            logger.info(
                f"Routing decision: {routing_decision.intent.value} "
                f"(confidence: {routing_decision.confidence:.2f})"
            )

            # Record classification metrics
            self.metrics.record_classification(
                intent=routing_decision.intent.value,
                confidence=routing_decision.confidence,
            )
            self.metrics.record_sources_selected(len(routing_decision.selected_sources))

            # Step 2: Create fabric context
            fabric_context = FabricContext(
                query=query,
                query_type=routing_decision.intent,
                results={},
                fused_chunks=[],
                sources_used=[],
                routing_reasoning=routing_decision.reasoning,
            )

            # Step 3: Execute parallel retrieval
            selected_sources = routing_decision.selected_sources
            if not selected_sources:
                logger.info("No sources selected for this query")
                query_timer.__exit__(None, None, None)
                return True, None

            logger.info(f"Querying {len(selected_sources)} sources: {selected_sources}")

            # Prepare retrieval tasks with circuit breaker protection
            retrieval_tasks = []
            task_source_names: list[str] = []
            for source_name in selected_sources:
                if source_name not in self._sources_by_name:
                    continue
                source = self._sources_by_name[source_name]
                # Inject search_service if needed (for RAG)
                if (
                    hasattr(source, "search_service")
                    and getattr(source, "search_service", None) is None
                ):
                    # Try to get from state (injected by app)
                    search_service = context.get("rag_search_service")
                    if search_service:
                        setattr(source, "search_service", search_service)
                    else:
                        logger.warning(
                            f"RAG search service not available for {source_name}"
                        )
                        continue

                    # Wrap retrieve with circuit breaker protection
                    protected_retrieve = execute_with_circuit_breaker(
                        registry=self.circuit_registry,
                        source_name=source_name,
                        callable_func=source.retrieve,
                        query=query,
                        context=context,
                    )
                    retrieval_tasks.append(protected_retrieve)

            # Execute parallel retrieval
            if retrieval_tasks:
                results = await asyncio.gather(*retrieval_tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(
                            f"Source {task_source_names[i] if i < len(task_source_names) else 'unknown'} retrieval failed: {result}"
                        )
                        continue

                    if isinstance(result, KnowledgeResult):
                        fabric_context.results[result.source_name] = result
                        fabric_context.sources_used.append(result.source_name)

                logger.info(
                    f"Retrieved {sum(len(r.chunks) for r in fabric_context.results.values())} "
                    f"chunks from {len(fabric_context.results)} sources"
                )

                # Step 4: Fuse results using RRF
                if fabric_context.results:
                    result_list = list(fabric_context.results.values())
                    chunks_before = sum(len(r.chunks) for r in result_list)

                    fused_chunks = await self.fusion.fuse(result_list, fabric_context)
                    fabric_context.fused_chunks = fused_chunks

                    # Record fusion metrics
                    self.metrics.record_fusion_operation(
                        status="success",
                        chunks_before=chunks_before,
                        chunks_after=len(fused_chunks),
                        latency_seconds=0.0,
                    )

                    logger.info(
                        f"RRF fused into {len(fused_chunks)} chunks "
                        f"from {len(result_list)} sources"
                    )

                    # Step 5: Synthesize context for LLM
                    knowledge_context = self.synthesizer.synthesize(fabric_context)

                    if knowledge_context:
                        # Store in context for later use
                        context[FABRIC_CONTEXT_KEY] = fabric_context
                        context[KNOWLEDGE_CONTEXT_KEY] = knowledge_context

                        # Record context generation metrics
                        self.metrics.record_context_generation(
                            char_count=len(knowledge_context),
                            sources_used=len(fabric_context.sources_used),
                        )

                        logger.info(
                            f"Injected {len(knowledge_context)} chars of knowledge context"
                        )

            # Close query timing
            query_timer.__exit__(None, None, None)

        except Exception as e:
            logger.exception(f"Error in knowledge fabric process_request: {e}")
            # Don't fail the request on knowledge errors
            # Log and continue

        return True, None

    async def _extract_query(self, request: Request, context: dict) -> Optional[str]:
        """
        Extract user query from request.

        The query may be in:
        - Request body (for /v1/chat/completions or /v1/messages)
        - Already parsed in context

        Args:
            request: FastAPI Request object
            context: Request context dict

        Returns:
            Extracted query string or None
        """
        # First check if already extracted
        if "user_query" in context:
            return context["user_query"]

        # Import shared utility
        from ai_inference_gateway.utils.message_utils import (  # type: ignore[import-not-found]
            extract_user_query_from_request_body,
            parse_request_body_safely,
        )

        # Try to get from cached request body (populated by other middleware)
        if "parsed_body" in context:
            body = context["parsed_body"]
        else:
            # Parse the body ourselves using shared utility
            body = await parse_request_body_safely(request)
            if body is None:
                return None

        # Extract query using shared utility
        return extract_user_query_from_request_body(body)

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process response to add knowledge usage metadata.

        Adds metadata about which knowledge sources were used,
        retrieval times, and RRF scores to the response.

        Args:
            response: The response dict to modify
            context: State from request processing

        Returns:
            Modified response dict with added metadata
        """
        if not self.enabled:
            return response

        # Check if knowledge fabric was used
        fabric_context = context.get(FABRIC_CONTEXT_KEY)
        if not fabric_context:
            return response

        try:
            # Add usage metadata to response
            # This can be used by monitoring/observability systems

            usage_metadata = {
                "knowledge_fabric": {
                    "enabled": True,
                    "query_type": fabric_context.query_type.value,
                    "sources_used": fabric_context.sources_used,
                    "chunks_retrieved": sum(
                        len(r.chunks) for r in fabric_context.results.values()
                    ),
                    "chunks_fused": len(fabric_context.fused_chunks),
                    "routing_reasoning": fabric_context.routing_reasoning,
                    "total_retrieval_time": sum(
                        r.retrieval_time for r in fabric_context.results.values()
                    ),
                    "circuit_breakers": self.circuit_registry.get_all_metrics(),
                }
            }

            # Add to response metadata (if supported)
            if "metadata" not in response:
                response["metadata"] = {}

            # Merge knowledge fabric metadata
            response["metadata"]["knowledge_fabric"] = usage_metadata[
                "knowledge_fabric"
            ]

            logger.debug(
                f"Added knowledge fabric metadata: {fabric_context.sources_used}"
            )

        except Exception as e:
            logger.exception(f"Error in knowledge fabric process_response: {e}")

        return response


def create_knowledge_fabric(
    sources: Optional[List[KnowledgeSource]] = None,
    rrf_k: int = 60,
    enabled: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> KnowledgeFabricMiddleware:
    """
    Factory function to create Knowledge Fabric middleware.

    Args:
        sources: Optional list of knowledge sources
        rrf_k: RRF constant for fusion (default 60)
        enabled: Whether the middleware is active
        config: Optional configuration dictionary

    Returns:
        Configured KnowledgeFabricMiddleware instance

    Example:
        ```python
        from ai_inference_gateway.middleware.knowledge_fabric import (
            create_knowledge_fabric
        )

        fabric_middleware = create_knowledge_fabric(
            enabled=True,
            config={
                "rag_enabled": True,
                "rag_top_k": 5,
                "code_search_paths": ["/etc/nixos"],
            }
        )
        ```
    """
    return KnowledgeFabricMiddleware(
        sources=sources,
        rrf_k=rrf_k,
        enabled=enabled,
        config=config,
    )
