"""
Core data structures for Knowledge Fabric.

Defines the unified interfaces that all knowledge sources implement,
enabling semantic routing and parallel retrieval.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Flag
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
)
from datetime import datetime


class SourcePriority(IntEnum):
    """Priority levels for knowledge sources (lower = higher priority)."""
    CRITICAL = 1  # Internal codebase, authoritative docs
    HIGH = 2      # RAG knowledge base, structured data
    MEDIUM = 3    # Web search, SearXNG
    LOW = 4       # General web, unstructured sources


class SourceCapability(Flag):
    """Capabilities a knowledge source provides (flags can combine)."""
    FACTUAL = 1        # Specific facts, definitions, entities
    PROCEDURAL = 2     # How-to, tutorials, step-by-step
    CONTEXTUAL = 4     # Context-rich, explanations
    REALTIME = 8       # Current data, news, live info
    CODE = 16          # Code snippets, examples, implementations
    COMPARATIVE = 32   # Comparisons, alternatives, analysis


@dataclass
class KnowledgeChunk:
    """
    A single piece of retrieved knowledge.

    Attributes:
        content: The actual text/content
        source: Which source provided this (e.g., "rag", "web_search")
        score: Relevance score from source (0-1 or rank-based)
        metadata: Additional info (URL, file path, confidence, etc.)
        capabilities: What type of knowledge this represents
    """
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    capabilities: Optional[SourceCapability] = None

    def __post_init__(self):
        if self.capabilities is None:
            # Infer from metadata if available
            self.capabilities = self.metadata.get("capabilities", SourceCapability.FACTUAL)


@dataclass
class KnowledgeResult:
    """
    Complete result from a single knowledge source.

    Attributes:
        source_name: Which source this came from
        chunks: List of retrieved knowledge chunks
        query: The original query that produced this
        retrieval_time: How long the retrieval took (seconds)
        metadata: Source-specific metadata (total_results, pagination, etc.)
    """
    source_name: str
    chunks: List[KnowledgeChunk]
    query: str
    retrieval_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def has_results(self) -> bool:
        return self.total_chunks > 0


@dataclass
class FabricContext:
    """
    Complete context for a knowledge fabric query.

    Holds all state from query classification through fusion,
    enabling synthesis and tracing.

    Attributes:
        query: The user's original query
        query_type: Classified intent (e.g., CODE, FACTUAL)
        results: Raw results from each source (by source name)
        fused_chunks: Merged/ranked chunks after RRF fusion
        sources_used: Which sources were actually queried
        total_retrieval_time: Cumulative time across all sources
        synthesis: Optional synthesized context for LLM
        timestamp: When this query was executed
        routing_reasoning: Explanation of routing decision
    """
    query: str
    query_type: "QueryIntent"
    results: Dict[str, KnowledgeResult] = field(default_factory=dict)
    fused_chunks: List[KnowledgeChunk] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    total_retrieval_time: float = 0.0
    synthesis: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    routing_reasoning: Optional[str] = None

    @property
    def total_sources_queried(self) -> int:
        return len(self.sources_used)

    @property
    def total_chunks_retrieved(self) -> int:
        return sum(r.total_chunks for r in self.results.values())


# Type alias for retrieve functions
RetrieveFunc = Callable[[str, Dict[str, Any]], Awaitable[KnowledgeResult]]


@dataclass
class KnowledgeSource:
    """
    A knowledge source that can be queried for information.

    This is the core abstraction - all sources (RAG, web search, code search)
    implement this interface, enabling unified routing and retrieval.

    Attributes:
        name: Unique identifier for this source
        description: Human-readable description
        priority: SourcePriority level (affects routing order)
        capabilities: What types of queries this source can handle
        retrieve_func: Async function that executes the query
        config: Source-specific configuration options
        enabled: Whether this source is active
    """
    name: str
    description: str
    priority: SourcePriority
    capabilities: SourceCapability
    retrieve_func: RetrieveFunc
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    async def retrieve(
        self,
        query: str,
        **kwargs
    ) -> KnowledgeResult:
        """
        Execute a query against this knowledge source.

        Args:
            query: The user's question/search terms
            **kwargs: Additional source-specific parameters

        Returns:
            KnowledgeResult with chunks and metadata
        """
        import time
        start = time.time()
        try:
            result = await self.retrieve_func(query, {**self.config, **kwargs})
            result.retrieval_time = time.time() - start
            return result
        except Exception as e:
            # Return empty result on error rather than raising
            return KnowledgeResult(
                source_name=self.name,
                chunks=[],
                query=query,
                retrieval_time=time.time() - start,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )

    def can_handle(self, capabilities: SourceCapability) -> bool:
        """Check if this source has the required capabilities."""
        return bool(self.capabilities & capabilities)

    def __str__(self) -> str:
        caps = "+".join(c.name for c in SourceCapability if c in self.capabilities)
        return f"{self.name} ({caps})"

    def __repr__(self) -> str:
        return f"KnowledgeSource(name={self.name!r}, priority={self.priority.name}, enabled={self.enabled})"
