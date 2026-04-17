"""
SearXNG RAG-Optimized Search

Integrates SearXNG search results with the RAG system for persistent
knowledge storage and retrieval.

Features:
- Automatic result chunking and embedding
- Qdrant storage for future retrieval
- Quality-based filtering before indexing
- Deduplication of indexed content
"""

import asyncio
import logging
import hashlib
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SearXNGRAGIndexer:
    """
    RAG-optimized indexer for SearXNG search results.

    Automatically chunks, embeds, and stores high-quality search results
    in Qdrant for future semantic retrieval.
    """

    def __init__(
        self,
        search_service: Any,  # HybridSearchService
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_quality_score: float = 0.3,
        default_collection: str = "searxng-results",
    ):
        """
        Initialize SearXNG RAG indexer.

        Args:
            search_service: HybridSearchService for embeddings/storage
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
            min_quality_score: Minimum quality to index a result
            default_collection: Default Qdrant collection name
        """
        self.search_service = search_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_quality_score = min_quality_score
        self.default_collection = default_collection

        # Track indexed URLs for deduplication
        self._indexed_urls: Dict[str, float] = {}  # url -> timestamp

    def _generate_result_id(self, url: str, title: str) -> str:
        """Generate stable ID for a search result."""
        content = f"{url}|{title}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _chunk_result(self, result: Dict[str, Any]) -> List[str]:
        """
        Chunk a search result into manageable pieces.

        Args:
            result: Search result with title, content, url

        Returns:
            List of text chunks
        """
        title = result.get("title", "")
        content = result.get("content", result.get("snippet", ""))
        url = result.get("url", "")

        # Combine title and content
        full_text = f"{title}\n\n{content}".strip()

        # If content is short, return as-is
        if len(full_text) <= self.chunk_size:
            return [full_text]

        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = full_text.split("\n\n")

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding would exceed chunk size
            if len(current_chunk) + len(para) + 2 > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk)
                current_chunk = overlap_text + para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from previous chunk."""
        if len(text) <= self.chunk_overlap:
            return text

        # Find sentence boundary
        for sep in [". ", "! ", "? ", "\n"]:
            last_sep = text[: self.chunk_overlap].rfind(sep)
            if last_sep > self.chunk_overlap // 2:
                return text[last_sep + len(sep) :]

        return text[-self.chunk_overlap :]

    def _should_index(self, result: Dict[str, Any]) -> bool:
        """
        Determine if a result should be indexed.

        Args:
            result: Search result with quality_score

        Returns:
            True if result meets quality threshold
        """
        # Check quality score
        quality = result.get("quality_score", 0.0)
        if quality < self.min_quality_score:
            return False

        # Check for minimum content length
        content = result.get("content", result.get("snippet", ""))
        if len(content) < 100:
            return False

        # Check for duplicate URL
        url = result.get("url", "")
        if url in self._indexed_urls:
            return False

        return True

    async def search_and_index(
        self,
        query: str,
        searxng_client: Any,
        collection: Optional[str] = None,
        max_results: int = 10,
        domain: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search SearXNG and index results in Qdrant.

        Workflow:
        1. Execute SearXNG search
        2. Filter results by quality
        3. Chunk results into manageable pieces
        4. Generate embeddings via EmbeddingService
        5. Store in Qdrant via HybridSearchService

        Args:
            query: Search query
            searxng_client: SearxngIntegration instance
            collection: Qdrant collection name
            max_results: Maximum results to index
            domain: Optional domain hint for routing

        Returns:
            Dict with search results and indexing status
        """
        collection = collection or self.default_collection

        # Execute search with domain routing
        search_result = await searxng_client.search_with_domain_routing(
            query=query,
            domain=domain,
            max_results=max_results * 2,  # Fetch more for quality filtering
            use_cache=True,
        )

        results = search_result.get("results", [])
        if not results:
            return {
                "query": query,
                "results": [],
                "indexed": 0,
                "chunks_stored": 0,
                "collection": collection,
                "message": "No results found",
            }

        # Filter by quality and deduplicate
        filtered_results = [r for r in results if self._should_index(r)]
        filtered_results = filtered_results[:max_results]

        if not filtered_results:
            return {
                "query": query,
                "results": results[:max_results],
                "indexed": 0,
                "chunks_stored": 0,
                "collection": collection,
                "message": "No results met quality threshold",
            }

        # Prepare documents for indexing
        documents_to_index = []

        for result in filtered_results:
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", result.get("snippet", ""))

            # Create full document text
            doc_text = f"{title}\n\n{content}".strip()

            # Prepare metadata
            metadata = {
                "url": url,
                "title": title,
                "source": "searxng",
                "domain": domain or "general",
                "quality_score": result.get("quality_score", 0.0),
                "indexed_at": datetime.now().isoformat(),
                "query": query,
            }

            documents_to_index.append(
                {"content": doc_text, "metadata": metadata, "url": url}
            )

            # Track as indexed
            self._indexed_urls[url] = datetime.now().timestamp()

        # Index documents via search service
        total_chunks = 0

        for doc in documents_to_index:
            try:
                result = await self.search_service.ingest_document(
                    collection=collection,
                    content=doc["content"],
                    metadata=doc["metadata"],
                    document_id=self._generate_result_id(doc["url"], doc["metadata"]["title"]),
                )

                if result.get("success"):
                    total_chunks += result.get("chunks_created", 0)
                else:
                    logger.warning(f"Failed to index {doc['url']}: {result.get('error')}")

            except Exception as e:
                logger.error(f"Error indexing document: {e}")

        return {
            "query": query,
            "results": filtered_results,
            "indexed": len(filtered_results),
            "chunks_stored": total_chunks,
            "collection": collection,
            "routing": search_result.get("routing", {}),
        }

    async def search_similar_to_url(
        self,
        url: str,
        searxng_client: Any,
        ingestion_service: Any,
        collection: Optional[str] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Find content similar to a given URL.

        Workflow:
        1. Fetch content from URL (via URLIngestionService)
        2. Generate embedding
        3. Search Qdrant for similar vectors
        4. Optionally fetch fresh SearXNG results

        Args:
            url: URL to find similar content for
            searxng_client: SearxngIntegration instance
            ingestion_service: URLIngestionService for fetching
            collection: Qdrant collection name
            max_results: Maximum similar results
            similarity_threshold: Minimum similarity score

        Returns:
            Similar content from Qdrant + fresh search results
        """
        collection = collection or self.default_collection

        # Fetch and ingest the URL
        doc = await ingestion_service.ingest_url(url, collection=collection)

        if not doc.success:
            return {
                "url": url,
                "error": doc.error,
                "similar_results": [],
                "fresh_results": [],
            }

        # Search for similar content in Qdrant
        try:
            # Use the search service to find similar content
            # This requires a similarity search method
            search_results = await self.search_service.search(
                query=doc.title or doc.content[:200],
                collection=collection,
                top_k=max_results,
            )

            similar_results = []
            for result in search_results.get("results", []):
                # Filter out the source document
                result_url = result.get("metadata", {}).get("url")
                if result_url != url:
                    score = result.get("score", 0.0)
                    if score >= similarity_threshold:
                        similar_results.append(
                            {
                                "content": result.get("content"),
                                "score": score,
                                "metadata": result.get("metadata", {}),
                            }
                        )

        except Exception as e:
            logger.error(f"Error searching for similar content: {e}")
            similar_results = []

        # Optionally fetch fresh SearXNG results based on title
        fresh_results = []
        if doc.title:
            query = doc.title
            try:
                fresh_search = await searxng_client.search_with_domain_routing(
                    query=query, max_results=5, use_cache=False
                )
                fresh_results = fresh_search.get("results", [])
            except Exception as e:
                logger.error(f"Error fetching fresh results: {e}")

        return {
            "url": url,
            "title": doc.title,
            "similar_results": similar_results,
            "fresh_results": fresh_results,
            "collection": collection,
        }

    async def search_similar_to_text(
        self,
        text: str,
        collection: Optional[str] = None,
        max_results: int = 10,
        similarity_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        """
        Find content similar to a given text.

        Args:
            text: Text to find similar content for
            collection: Qdrant collection name
            max_results: Maximum similar results
            similarity_threshold: Minimum similarity score

        Returns:
            Similar content from Qdrant
        """
        collection = collection or self.default_collection

        try:
            search_results = await self.search_service.search(
                query=text,
                collection=collection,
                top_k=max_results,
            )

            similar_results = []
            for result in search_results.get("results", []):
                score = result.get("score", 0.0)
                if score >= similarity_threshold:
                    similar_results.append(
                        {
                            "content": result.get("content"),
                            "score": score,
                            "metadata": result.get("metadata", {}),
                        }
                    )

            return {
                "text": text[:200] + "..." if len(text) > 200 else text,
                "similar_results": similar_results,
                "collection": collection,
                "total_found": len(similar_results),
            }

        except Exception as e:
            logger.error(f"Error searching for similar text: {e}")
            return {
                "text": text[:200],
                "error": str(e),
                "similar_results": [],
            }

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed content."""
        return {
            "total_urls_indexed": len(self._indexed_urls),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_quality_score": self.min_quality_score,
            "default_collection": self.default_collection,
        }


def create_rag_indexer(
    search_service: Any,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_quality_score: float = 0.3,
    default_collection: str = "searxng-results",
) -> SearXNGRAGIndexer:
    """
    Create SearXNG RAG indexer.

    Args:
        search_service: HybridSearchService instance
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        min_quality_score: Minimum quality for indexing
        default_collection: Default Qdrant collection

    Returns:
        Configured SearXNGRAGIndexer
    """
    return SearXNGRAGIndexer(
        search_service=search_service,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_quality_score=min_quality_score,
        default_collection=default_collection,
    )
