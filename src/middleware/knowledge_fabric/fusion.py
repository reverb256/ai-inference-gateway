"""
Knowledge fusion using Reciprocal Rank Fusion (RRF).

Combines results from multiple knowledge sources into a single
ranked list, handling score normalization and duplicate detection.
Supports optional cross-encoder reranking for result refinement.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict

from .core import KnowledgeChunk, KnowledgeResult, FabricContext
from .routing import QueryIntent

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion for multi-source result merging.

    RRF Formula: score(d) = Σ (k / (k + rank_i(d)))

    Where:
    - k is a constant (typically 60)
    - rank_i(d) is the rank of document d in source i
    - Sum is over all sources that contain d

    RRF is robust to score scale differences and works well
    when combining ranked lists from heterogeneous sources.

    Supports optional cross-encoder reranking for result refinement.
    """

    DEFAULT_K = 60  # RRF constant (higher = more weight to lower ranks)
    DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-base"

    def __init__(
        self,
        k: int = DEFAULT_K,
        reranker_enabled: bool = False,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        final_k: int = 5,
        max_chunks_for_rerank: int = 30
    ):
        """
        Initialize RRF fusion.

        Args:
            k: RRF constant (default 60 is standard)
            reranker_enabled: Enable cross-encoder reranking after RRF
            reranker_model: Model name for cross-encoder reranker
            final_k: Number of results to return after reranking
            max_chunks_for_rerank: Max chunks to feed to reranker (recall)
        """
        self.k = k
        self.reranker_enabled = reranker_enabled
        self.reranker_model = reranker_model
        self.final_k = final_k
        self.max_chunks_for_rerank = max_chunks_for_rerank
        self._reranker = None

    async def initialize_reranker(self) -> None:
        """Initialize the reranker model (lazy loading)."""
        if self.reranker_enabled and self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker model: {self.reranker_model}")
                self._reranker = CrossEncoder(self.reranker_model)
                logger.info("Reranker model loaded successfully")
            except ImportError as e:
                logger.error(f"sentence_transformers not available: {e}")
                self.reranker_enabled = False
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self.reranker_enabled = False

    async def fuse(
        self,
        results: List[KnowledgeResult],
        context: FabricContext
    ) -> List[KnowledgeChunk]:
        """
        Fuse multiple knowledge results using RRF.

        Args:
            results: List of KnowledgeResult from different sources
            context: The FabricContext for this query

        Returns:
            List of KnowledgeChunks sorted by fused RRF score
        """
        # Ensure reranker is initialized if enabled
        if self.reranker_enabled and self._reranker is None:
            await self.initialize_reranker()

        # Track RRF scores for each unique chunk
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_data: Dict[str, KnowledgeChunk] = {}

        # Process each source's results
        for result in results:
            if not result.chunks:
                continue

            for rank, chunk in enumerate(result.chunks, start=1):
                # Create a unique key for deduplication
                # Use content hash or truncated content as key
                key = self._chunk_key(chunk)

                # Add RRF contribution: k / (k + rank)
                rrf_scores[key] += self.k / (self.k + rank)

                # Store chunk data on first occurrence
                if key not in chunk_data:
                    chunk_data[key] = chunk
                    # Preserve source metadata
                    if "sources" not in chunk.metadata:
                        chunk.metadata["sources"] = []
                    chunk.metadata["sources"].append(result.source_name)
                else:
                    # Track additional sources
                    if result.source_name not in chunk_data[key].metadata.get("sources", []):
                        chunk_data[key].metadata["sources"].append(result.source_name)

        # Build final sorted list
        fused_chunks = []
        for key, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
            chunk = chunk_data[key]
            chunk.score = rrf_score  # Replace original score with RRF score
            chunk.metadata["rrf_score"] = rrf_score
            fused_chunks.append(chunk)

        logger.debug(f"RRF fused {len(results)} sources into {len(fused_chunks)} chunks")

        # Apply reranking if enabled
        if self.reranker_enabled and self._reranker is not None and fused_chunks:
            return await self._rerank_chunks(context.query, fused_chunks)

        return fused_chunks[:self._max_chunks(context)]

    def _chunk_key(self, chunk: KnowledgeChunk) -> str:
        """Create a unique key for deduplication."""
        # Use first 100 chars of content as simple dedup key
        content_preview = chunk.content[:100].strip()
        # Could also use URL, file path, or actual hash
        if "url" in chunk.metadata:
            return chunk.metadata["url"]
        if "file_path" in chunk.metadata:
            return chunk.metadata["file_path"]
        return content_preview

    async def _rerank_chunks(
        self,
        query: str,
        chunks: List[KnowledgeChunk]
    ) -> List[KnowledgeChunk]:
        """
        Rerank chunks using cross-encoder model.

        Implements recall-then-rerank pattern:
        1. Take top-N chunks from RRF (recall phase)
        2. Score query-document pairs with cross-encoder
        3. Return top-K after reranking

        Args:
            query: Original user query
            chunks: Fused chunks from RRF (sorted by RRF score)

        Returns:
            Reranked chunks (top-K by cross-encoder score)
        """
        if not self._reranker:
            logger.warning("Reranking enabled but model not loaded, skipping")
            return chunks[:self.final_k]

        # Limit chunks for reranking (recall phase)
        recall_chunks = chunks[:self.max_chunks_for_rerank]

        # Prepare query-document pairs
        pairs = [(query, chunk.content) for chunk in recall_chunks]

        try:
            # Run cross-encoder in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, lambda: self._reranker.predict(pairs)
            )

            # Update chunks with reranker scores
            for chunk, score in zip(recall_chunks, scores):
                chunk.metadata["reranker_score"] = float(score)
                chunk.score = float(score)

            # Sort by reranker score and return top-K
            recall_chunks.sort(key=lambda c: c.score, reverse=True)
            result = recall_chunks[:self.final_k]

            logger.debug(
                f"Reranked {len(recall_chunks)} chunks, returning top {len(result)}"
            )
            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}, falling back to RRF results")
            return chunks[:self.final_k]

    def _max_chunks(self, context: FabricContext) -> int:
        """Determine max chunks to return based on query type."""
        # Code queries need fewer, more precise results
        if context.query_type == QueryIntent.CODE:
            return 5
        # Factual queries benefit from more context
        if context.query_type == QueryIntent.FACTUAL:
            return 10
        # Procedural often needs step-by-step detail
        if context.query_type == QueryIntent.PROCEDURAL:
            return 8
        # Default
        return 7


class ContextSynthesizer:
    """
    Synthesizes retrieved knowledge into LLM-ready context.

    Formats fused chunks into a coherent prompt that the LLM
    can use effectively, with source attribution and structure.
    """

    # Templates for different query types
    TEMPLATES = {
        QueryIntent.CODE: """The following code examples and implementations were found to help answer the query:

{chunks}

Use these code examples to provide an accurate implementation. If examples differ, explain the trade-offs.""",

        QueryIntent.PROCEDURAL: """The following step-by-step guides were found to help answer the query:

{chunks}

Follow these steps to provide clear instructions. Mention any prerequisites or warnings.""",

        QueryIntent.COMPARATIVE: """The following comparisons and alternatives were found:

{chunks}

Present a balanced comparison, highlighting key differences and trade-offs.""",

        QueryIntent.FACTUAL: """The following factual information was retrieved:

{chunks}

Use these facts to provide an accurate answer. Prioritize verified information.""",

        QueryIntent.REALTIME: """The following current information was retrieved:

{chunks}

Note the retrieval times and provide context about data freshness.""",

        QueryIntent.CONTEXTUAL: """The following contextual information was retrieved:

{chunks}

Synthesize this information to provide a comprehensive explanation.""",

        QueryIntent.UNKNOWN: """The following information was retrieved:

{chunks}

Use this information to help answer the query, supplementing with general knowledge as needed.""",
    }

    def synthesize(self, context: FabricContext) -> str:
        """
        Synthesize fused chunks into LLM context.

        Args:
            context: FabricContext with fused_chunks populated

        Returns:
            Formatted context string for LLM
        """
        if not context.fused_chunks:
            return ""

        # Format chunks based on query type
        formatted_chunks = self._format_chunks(context)

        # Get template
        template = self.TEMPLATES.get(
            context.query_type,
            self.TEMPLATES[QueryIntent.UNKNOWN]
        )

        # Build final context
        llm_context = template.format(chunks=formatted_chunks)

        # Add source attribution
        sources = set()
        for chunk in context.fused_chunks:
            sources.update(chunk.metadata.get("sources", [chunk.source]))

        if sources:
            llm_context += f"\n\nSources: {', '.join(sorted(sources))}"

        return llm_context

    def _format_chunks(self, context: FabricContext) -> str:
        """Format chunks based on query type."""
        parts = []

        for i, chunk in enumerate(context.fused_chunks, 1):
            source_str = chunk.metadata.get("sources", [chunk.source])[0]
            # Prefer reranker_score if available, otherwise rrf_score or base score
            if "reranker_score" in chunk.metadata:
                score = chunk.metadata["reranker_score"]
                score_type = "reranker"
            elif "rrf_score" in chunk.metadata:
                score = chunk.metadata["rrf_score"]
                score_type = "rrf"
            else:
                score = chunk.score
                score_type = "score"

            if context.query_type == QueryIntent.CODE:
                parts.append(
                    f"Example {i} (from {source_str}, {score_type}={score:.2f}):\n"
                    f"```\n{chunk.content}\n```"
                )
            elif context.query_type == QueryIntent.PROCEDURAL:
                parts.append(
                    f"Step {i} (from {source_str}):\n{chunk.content}"
                )
            elif context.query_type == QueryIntent.COMPARATIVE:
                parts.append(
                    f"Option {i} (from {source_str}):\n{chunk.content}"
                )
            else:
                # Default formatting
                parts.append(
                    f"[{i}] From {source_str} ({score_type}={score:.2f}):\n{chunk.content}"
                )

        return "\n\n".join(parts)


def create_fusion(
    k: int = 60,
    reranker_enabled: bool = False,
    reranker_model: str = RRFFusion.DEFAULT_RERANKER_MODEL,
    final_k: int = 5,
    max_chunks_for_rerank: int = 30
) -> RRFFusion:
    """Factory function to create RRF fusion.

    Args:
        k: RRF constant (default 60 is standard)
        reranker_enabled: Enable cross-encoder reranking after RRF
        reranker_model: Model name for cross-encoder reranker
        final_k: Number of results to return after reranking
        max_chunks_for_rerank: Max chunks to feed to reranker (recall)

    Returns:
        Configured RRFFusion instance
    """
    return RRFFusion(
        k=k,
        reranker_enabled=reranker_enabled,
        reranker_model=reranker_model,
        final_k=final_k,
        max_chunks_for_rerank=max_chunks_for_rerank
    )


def create_synthesizer() -> ContextSynthesizer:
    """Factory function to create context synthesizer."""
    return ContextSynthesizer()
