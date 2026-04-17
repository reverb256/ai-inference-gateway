"""
RAG Context Injection Middleware

Automatically retrieves relevant knowledge and injects it into requests.
This makes RAG a natural, embedded part of inference rather than requiring explicit API calls.

Strategies:
1. Query Classification: Detect knowledge-seeking queries
2. Context Retrieval: Search knowledge base via RAG
3. Seamless Injection: Inject as system context, transparent to client
4. Fallback Graceful: Degrade gracefully if RAG unavailable

Usage:
    Just add to middleware pipeline - no client changes needed!
"""

import asyncio
import logging
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, HTTPException
from ai_inference_gateway.middleware.base import Middleware

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for RAG decisions."""
    FACTUAL = "factual"           # Specific facts, definitions, "what is X"
    HOW_TO = "how_to"            # Procedures, "how do I", "how to"
    COMPARISON = "comparison"     # Comparisons, "X vs Y", difference
    TROUBLESHOOTING = "troubleshoot"  # Debugging, "why doesn't X work"
    CREATIVE = "creative"         # Generation, writing, coding from scratch
    CONVERSATION = "conversation"  # Chat, opinions, casual
    UNKNOWN = "unknown"


@dataclass
class RAGContext:
    """Retrieved context from knowledge base."""
    chunks: List[Dict[str, Any]]
    collection: str
    query: str
    confidence: float
    sources: List[str]


class QueryClassifier:
    """
    Classifies queries to determine RAG applicability.

    Uses pattern matching and heuristics to identify:
    - Knowledge-seeking queries (benefit from RAG)
    - Creative/conversational queries (don't need RAG)
    """

    # Patterns that indicate knowledge-seeking queries
    KNOWLEDGE_PATTERNS = [
        # Factual questions
        r"\bwhat is\b",
        r"\bdefine\b",
        r"\bexplain\b",
        r"\bdescribe\b",
        r"\bwhich\b",
        r"\bwho (is|are|was|were)\b",
        r"\bwhen (did|do|does|is|are)\b",
        r"\bwhere (is|are|was|were)\b",

        # How-to questions
        r"\bhow (do|does|did|to|can|should|would)\b",
        r"\bstep(s)? by step\b",
        r"\bguide\b",
        r"\btutorial\b",
        r"\bsetup\b",
        r"\bconfigure\b",
        r"\binstall\b",

        # Comparisons
        r"\bvs\b",
        r"\bversus\b",
        r"\bdifference\b",
        r"\bbetter\b",
        r"\bcompare\b",

        # Troubleshooting
        r"\bwhy (doesn't|don't|does not|do not)\b",
        r"\berror\b",
        r"\bproblem\b",
        r"\bfix\b",
        r"\bdebug\b",
        r"\btroubleshoot\b",
        r"\bissue\b",
    ]

    # Patterns that indicate creative/generative queries
    CREATIVE_PATTERNS = [
        r"\bwrite\b.*(code|poem|story|essay|article|blog)",
        r"\bgenerate\b",
        r"\bcreate\b.*(content|image|design|idea)",
        r"\bdraft\b",
        r"\bcompose\b",
        r"\binvent\b",
        r"\bimagine\b",
    ]

    # Patterns that indicate conversational queries
    CONVERSATIONAL_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|bye|goodbye)",
        r"\bhow are you\b",
        r"\bwhat do you think\b",
        r"\byour opinion\b",
        r"\bdo you (like|love|prefer)\b",
    ]

    def __init__(self):
        # Compile patterns for efficiency
        self.knowledge_regex = re.compile(
            "|".join(self.KNOWLEDGE_PATTERNS),
            re.IGNORECASE
        )
        self.creative_regex = re.compile(
            "|".join(self.CREATIVE_PATTERNS),
            re.IGNORECASE
        )
        self.conversational_regex = re.compile(
            "|".join(self.CONVERSATIONAL_PATTERNS),
            re.IGNORECASE
        )

    def classify(
        self,
        query: str,
        history: Optional[List[Dict]] = None
    ) -> Tuple[QueryType, float]:
        """
        Classify a query with confidence score.

        Args:
            query: The user's query text
            history: Optional conversation history for context

        Returns:
            Tuple of (QueryType, confidence 0-1)
        """
        query_lower = query.lower().strip()

        # Check for conversational patterns (highest priority)
        if self.conversational_regex.search(query):
            return QueryType.CONVERSATION, 0.9

        # Check for creative patterns
        if self.creative_regex.search(query):
            return QueryType.CREATIVE, 0.85

        # Check for knowledge-seeking patterns
        if self.knowledge_regex.search(query):
            # Further classify the type of knowledge query
            if "how" in query_lower and ("do" in query_lower or "to" in query_lower):
                return QueryType.HOW_TO, 0.8
            if "error" in query_lower or "doesn't" in query_lower or "fix" in query_lower:
                return QueryType.TROUBLESHOOTING, 0.8
            if "vs" in query_lower or "versus" in query_lower or "difference" in query_lower:
                return QueryType.COMPARISON, 0.8
            return QueryType.FACTUAL, 0.75

        # Check for technical/programming keywords (likely knowledge-seeking)
        tech_keywords = [
            "api", "function", "class", "method", "nixos", "kubernetes",
            "docker", "python", "rust", "algorithm", "data structure",
            "configuration", "deployment", "service"
        ]
        if any(kw in query_lower for kw in tech_keywords):
            return QueryType.FACTUAL, 0.6

        # Check if query is short and casual (likely conversational)
        if len(query.split()) < 4:
            return QueryType.CONVERSATION, 0.5

        return QueryType.UNKNOWN, 0.3


class RAGInjectorMiddleware(Middleware):
    """
    Middleware that automatically injects RAG context into requests.

    This makes RAG completely transparent to API users - they just ask
    questions and the system automatically provides relevant context from
    the knowledge base.

    Flow:
    1. Extract user's last message
    2. Classify query type
    3. If knowledge-seeking: search RAG
    4. Inject relevant context as system message
    5. Let LLM use context naturally
    """

    # Query types that benefit from RAG
    RAG_ENABLED_TYPES = {
        QueryType.FACTUAL,
        QueryType.HOW_TO,
        QueryType.COMPARISON,
        QueryType.TROUBLESHOOTING,
    }

    # Minimum confidence to apply RAG
    MIN_RAG_CONFIDENCE = 0.5

    # Maximum context chunks to inject
    MAX_CONTEXT_CHUNKS = 5

    # Maximum context tokens (rough estimate)
    MAX_CONTEXT_TOKENS = 2000

    def __init__(
        self,
        search_service=None,
        classifier: Optional[QueryClassifier] = None,
        enabled: bool = True,
        collection: str = "knowledge-base",
        min_confidence: float = 0.5,
        max_chunks: int = 5,
    ):
        """
        Initialize RAG injector middleware.

        Args:
            search_service: RAG search service (HybridSearchService)
            classifier: Query classifier (defaults to QueryClassifier())
            enabled: Whether middleware is active
            collection: Default collection to search
            min_confidence: Minimum confidence to apply RAG
            max_chunks: Maximum context chunks to inject
        """
        self.search_service = search_service
        self.classifier = classifier or QueryClassifier()
        self._enabled = enabled
        self.collection = collection
        self.min_confidence = min_confidence
        self.max_chunks = max_chunks

    @property
    def enabled(self) -> bool:
        """Check if middleware is enabled."""
        return self._enabled and self.search_service is not None

    async def process_request(
        self, request: Request, context: dict
    ) -> Tuple[bool, Optional[HTTPException]]:
        """
        Process incoming request - inject RAG context if applicable.

        Modifies the request body in-place to add context before routing.
        """
        # Get search service from context (injected by gateway during request)
        search_service = context.get("rag_search_service") if context else None
        if not search_service:
            # RAG not available, skip injection
            return True, None

        if not self.enabled:
            return True, None

        try:
            # Import utility function
            from ai_inference_gateway.utils.message_utils import (
                extract_last_user_message,
                extract_message_content,
            )

            # Read request body
            body = await request.json()

            # Extract messages
            messages = body.get("messages", [])
            if not messages:
                return True, None

            # Get the last user message using shared utility
            last_message = extract_last_user_message(messages)

            if not last_message:
                return True, None

            query = extract_message_content(last_message)
            if not query or len(query) < 10:  # Skip very short queries
                return True, None

            # Classify the query
            query_type, confidence = self.classifier.classify(query, messages)
            logger.debug(
                f"RAG classifier: {query_type.value} (confidence: {confidence:.2f}) "
                f"for query: {query[:50]}..."
            )

            # Only apply RAG for knowledge-seeking queries with sufficient confidence
            if (
                query_type not in self.RAG_ENABLED_TYPES
                or confidence < self.min_confidence
            ):
                context["rag_classification"] = {
                    "type": query_type.value,
                    "confidence": confidence,
                    "rag_applied": False,
                    "reason": f"Query type {query_type.value} or confidence too low",
                }
                return True, None

            # Search knowledge base
            search_result = await search_service.search(
                query=query,
                collection=self.collection,
                top_k=self.max_chunks,
            )

            if not search_result.get("results"):
                logger.debug(f"No RAG results found for query: {query[:50]}...")
                context["rag_classification"] = {
                    "type": query_type.value,
                    "confidence": confidence,
                    "rag_applied": False,
                    "reason": "No relevant results found",
                }
                return True, None

            # Build context injection
            rag_context = self._build_context_injection(
                search_result["results"][: self.max_chunks],
                query_type,
            )

            # Inject context as system message
            # Insert after any existing system messages
            messages_with_rag = []
            system_inserted = False

            for msg in messages:
                if msg.get("role") == "system" and not system_inserted:
                    # Insert RAG context after first system message
                    messages_with_rag.append(msg)
                    messages_with_rag.append({
                        "role": "system",
                        "content": rag_context,
                    })
                    system_inserted = True
                elif msg.get("role") == "system" and system_inserted:
                    # Already inserted, just append
                    messages_with_rag.append(msg)
                else:
                    messages_with_rag.append(msg)

            # If no system message existed, insert at beginning
            if not system_inserted:
                messages_with_rag.insert(0, {
                    "role": "system",
                    "content": rag_context,
                })

            # Update the body with RAG-enhanced messages
            body["messages"] = messages_with_rag

            # Store RAG info in context for metrics/logging
            context["rag_classification"] = {
                "type": query_type.value,
                "confidence": confidence,
                "rag_applied": True,
                "chunks_retrieved": len(search_result['results']),
                "collection": self.collection,
            }

            # Update request body (this will be used by the chat handler)
            # Note: We can't modify request.json() directly, so we store in context
            # The chat handler needs to check for this
            context["rag_enhanced_body"] = body
            logger.info(
                f"RAG context injected: {len(search_result['results'])} chunks "
                f"for {query_type.value} query"
            )

            return True, None

        except Exception as e:
            logger.error(f"Error in RAG injection: {e}", exc_info=True)
            # Don't block requests on RAG errors
            return True, None

    async def process_response(self, response: dict, context: dict) -> dict:
        """
        Process response - add metadata about RAG usage.

        This can be used by clients to know when context was injected.
        """
        if "rag_classification" in context:
            rag_info = context["rag_classification"]

            # Add RAG metadata to response
            if "usage" not in response:
                response["usage"] = {}

            response["usage"]["rag"] = {
                "enabled": rag_info.get("rag_applied", False),
                "query_type": rag_info.get("type"),
                "confidence": rag_info.get("confidence"),
                "chunks_retrieved": rag_info.get("chunks_retrieved", 0),
            }

        return response

    def _build_context_injection(
        self, results: List[Dict], query_type: QueryType
    ) -> str:
        """
        Build the context injection message.

        Format varies by query type for optimal LLM usage.
        """
        sources = set()
        context_parts = []

        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            # Track sources
            source = metadata.get("source", metadata.get("title", "Unknown"))
            sources.add(source)

            # Format based on query type
            if query_type == QueryType.HOW_TO:
                # For how-to queries, show steps clearly
                context_parts.append(
                    f"Step {i} (from {source}):\n{content}"
                )
            elif query_type == QueryType.COMPARISON:
                # For comparisons, show different perspectives
                context_parts.append(
                    f"Perspective {i} (from {source}):\n{content}"
                )
            else:
                # Default: just number and show content
                context_parts.append(
                    f"[{i}] From {source}:\n{content}"
                )

        context_text = "\n\n".join(context_parts)

        # Build the system message
        header = f"""The following relevant context was found from the knowledge base to help answer the user's {query_type.value} query:

---

{context_text}

---

Use this context to provide an accurate, helpful response. If the context doesn't fully address the query, you can supplement with your general knowledge, but prioritize the provided information."""

        if sources:
            header += f"\n\nSources: {', '.join(sorted(sources))}"

        return header


def create_rag_middleware(
    search_service=None,
    enabled: bool = True,
    **config
) -> RAGInjectorMiddleware:
    """
    Factory function to create RAG injector middleware.

    Args:
        search_service: HybridSearchService instance
        enabled: Whether middleware is active
        **config: Additional config passed to middleware

    Returns:
        Configured RAGInjectorMiddleware instance
    """
    return RAGInjectorMiddleware(
        search_service=search_service,
        enabled=enabled,
        **config
    )
