"""
RAG URL Ingestion Service

Fetches and ingests documents from URLs into the RAG system.
Supports both direct HTTP fetching and MCP web-reader integration.

Features:
- HTTP client for direct URL fetching
- MCP web-reader integration for enhanced fetching
- Domain whitelist for security
- Batch ingestion support
- Content extraction and chunking
- Automatic embedding and storage
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
from enum import Enum

import httpx
from bs4 import BeautifulSoup

try:
    from ai_inference_gateway.mcp_broker import MCPBroker

    MCP_BROKER_AVAILABLE = True
except ImportError:
    MCP_BROKER_AVAILABLE = False
    MCPBroker = None

from .config import RAGConfig
from .embeddings import EmbeddingService
from .chunker import DocumentChunker
from .qdrant_client import QdrantManager

logger = logging.getLogger(__name__)


class IngestionSource(Enum):
    """Document ingestion source types."""

    HTTP_DIRECT = "http_direct"  # Direct HTTP fetch
    MCP_WEB_READER = "mcp_web_reader"  # MCP web-reader tool
    FILE = "file"  # Local file upload


@dataclass
class IngestionConfig:
    """
    URL ingestion configuration.

    Attributes:
        allowed_domains: Whitelist of allowed domains (empty = all allowed)
        blocked_domains: Blacklist of blocked domains
        max_file_size_bytes: Maximum file size to fetch (default: 10MB)
        timeout_seconds: HTTP timeout (default: 30s)
        user_agent: User agent string for HTTP requests
        enable_mcp_web_reader: Use MCP web-reader if available
        batch_size: Number of URLs to process in parallel (default: 5)
    """

    allowed_domains: Set[str] = field(default_factory=set)
    blocked_domains: Set[str] = field(default_factory=set)
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    timeout_seconds: int = 30
    user_agent: str = "Mozilla/5.0 (compatible; AIInferenceGateway/1.0)"
    enable_mcp_web_reader: bool = True
    batch_size: int = 5

    def is_domain_allowed(self, domain: str) -> bool:
        """Check if domain is allowed based on whitelist/blacklist."""
        # Check blacklist first
        if domain in self.blocked_domains:
            return False

        # If whitelist is empty, allow all domains
        if not self.allowed_domains:
            return True

        # Check whitelist
        return domain in self.allowed_domains


@dataclass
class IngestedDocument:
    """Result of document ingestion."""

    url: str
    source: IngestionSource
    title: Optional[str] = None
    content: str = ""
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if ingestion was successful."""
        return self.error is None and len(self.content) > 0


class URLIngestionService:
    """
    Service for ingesting documents from URLs into RAG system.

    Supports both direct HTTP fetching and MCP web-reader integration.
    """

    def __init__(
        self,
        config: IngestionConfig,
        rag_config: RAGConfig,
        embedder: EmbeddingService,
        chunker: DocumentChunker,
        qdrant: QdrantManager,
        mcp_broker: Optional[MCPBroker] = None,
    ):
        """
        Initialize URL ingestion service.

        Args:
            config: Ingestion configuration
            rag_config: RAG configuration
            embedder: Embedding service
            chunker: Document chunker
            qdrant: Qdrant manager
            mcp_broker: Optional MCP broker for web-reader tool
        """
        self.config = config
        self.rag_config = rag_config
        self.embedder = embedder
        self.chunker = chunker
        self.qdrant = qdrant
        self.mcp_broker = mcp_broker

        # HTTP client for direct fetching
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            f"URLIngestionService initialized: "
            f"allowed_domains={len(config.allowed_domains)}, "
            f"blocked_domains={len(config.blocked_domains)}, "
            f"mcp_web_reader={config.enable_mcp_web_reader}"
        )

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                headers={"User-Agent": self.config.user_agent},
                follow_redirects=True,
            )
        return self._http_client

    async def ingest_url(
        self,
        url: str,
        collection: str = "default",
        source_preference: IngestionSource = IngestionSource.MCP_WEB_READER,
    ) -> IngestedDocument:
        """
        Ingest document from URL.

        Args:
            url: URL to fetch
            collection: Target Qdrant collection
            source_preference: Preferred ingestion source

        Returns:
            IngestedDocument with results
        """
        # Validate URL
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            if not self.config.is_domain_allowed(domain):
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.HTTP_DIRECT,
                    error=f"Domain not allowed: {domain}",
                )

            if parsed.scheme not in ("http", "https"):
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.HTTP_DIRECT,
                    error=f"Unsupported scheme: {parsed.scheme}",
                )

        except Exception as e:
            return IngestedDocument(
                url=url, source=IngestionSource.HTTP_DIRECT, error=f"Invalid URL: {e}"
            )

        # Try preferred source first
        if source_preference == IngestionSource.MCP_WEB_READER:
            if self.config.enable_mcp_web_reader and self.mcp_broker:
                doc = await self._ingest_via_mcp(url)
                if doc.success:
                    # Store in Qdrant
                    await self._store_document(doc, collection)
                    return doc

            # Fallback to HTTP direct
            doc = await self._ingest_via_http(url)
            if doc.success:
                await self._store_document(doc, collection)
            return doc

        else:  # HTTP_DIRECT
            doc = await self._ingest_via_http(url)
            if doc.success:
                await self._store_document(doc, collection)
            return doc

    async def ingest_urls(
        self,
        urls: List[str],
        collection: str = "default",
        source_preference: IngestionSource = IngestionSource.MCP_WEB_READER,
    ) -> List[IngestedDocument]:
        """
        Ingest multiple URLs in batches.

        Args:
            urls: List of URLs to ingest
            collection: Target Qdrant collection
            source_preference: Preferred ingestion source

        Returns:
            List of IngestedDocument results
        """
        results = []

        # Process in batches
        for i in range(0, len(urls), self.config.batch_size):
            batch = urls[i : i + self.config.batch_size]

            # Process batch in parallel
            tasks = [
                self.ingest_url(url, collection, source_preference) for url in batch
            ]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error ingesting URL: {result}")
                    # Create error document
                    results.append(
                        IngestedDocument(
                            url="unknown",
                            source=IngestionSource.HTTP_DIRECT,
                            error=str(result),
                        )
                    )
                else:
                    results.append(result)

        return results

    async def _ingest_via_mcp(self, url: str) -> IngestedDocument:
        """Ingest URL via MCP web-reader tool."""
        if not self.mcp_broker:
            return IngestedDocument(
                url=url,
                source=IngestionSource.MCP_WEB_READER,
                error="MCP broker not available",
            )

        try:
            # Call web-reader MCP tool
            result = await self.mcp_broker.call_tool(
                server_name="web-reader", tool_name="webReader", arguments={"url": url}
            )

            if "error" in result:
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.MCP_WEB_READER,
                    error=result.get("error"),
                )

            # Extract content from result
            content = result.get("content", "")
            if not content:
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.MCP_WEB_READER,
                    error="No content returned from web-reader",
                )

            # Parse title if available
            title = self._extract_title_from_content(content)

            return IngestedDocument(
                url=url,
                source=IngestionSource.MCP_WEB_READER,
                title=title,
                content=content,
                metadata={"ingestion_method": "mcp_web_reader"},
            )

        except Exception as e:
            logger.error(f"Error ingesting via MCP: {e}")
            return IngestedDocument(
                url=url, source=IngestionSource.MCP_WEB_READER, error=f"MCP error: {e}"
            )

    async def _ingest_via_http(self, url: str) -> IngestedDocument:
        """Ingest URL via direct HTTP fetch."""
        try:
            client = await self._get_http_client()

            # Fetch URL
            response = await client.get(url)

            # Check file size
            content_length = response.headers.get("content-length")
            if content_length:
                size = int(content_length)
                if size > self.config.max_file_size_bytes:
                    return IngestedDocument(
                        url=url,
                        source=IngestionSource.HTTP_DIRECT,
                        error=f"File too large: {size} bytes (max: {self.config.max_file_size_bytes})",
                    )

            # Check response status
            if response.status_code != 200:
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.HTTP_DIRECT,
                    error=f"HTTP {response.status_code}",
                )

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()

            # Extract main content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Get text content
            content = soup.get_text(separator="\n", strip=True)

            # Clean up content
            content = self._clean_content(content)

            if not content:
                return IngestedDocument(
                    url=url,
                    source=IngestionSource.HTTP_DIRECT,
                    error="No content extracted",
                )

            return IngestedDocument(
                url=url,
                source=IngestionSource.HTTP_DIRECT,
                title=title,
                content=content,
                metadata={
                    "content_type": response.headers.get("content-type"),
                    "content_length": len(response.content),
                },
            )

        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching {url}: {e}")
            return IngestedDocument(
                url=url, source=IngestionSource.HTTP_DIRECT, error=f"HTTP error: {e}"
            )
        except Exception as e:
            logger.error(f"Error ingesting via HTTP: {e}")
            return IngestedDocument(
                url=url, source=IngestionSource.HTTP_DIRECT, error=f"Error: {e}"
            )

    async def _store_document(self, doc: IngestedDocument, collection: str):
        """Store ingested document in Qdrant."""
        try:
            # Chunk document
            chunks = await self.chunker.chunk_text(doc.content)

            # Generate embeddings
            embeddings = await self.embedder.embed_texts(chunks)

            # Store in Qdrant
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = f"{doc.url}_{i}".encode().hex()
                points.append(
                    {
                        "id": point_id,
                        "vector": embedding,
                        "payload": {
                            "text": chunk,
                            "url": doc.url,
                            "title": doc.title,
                            "chunk_index": i,
                            "source": doc.source.value,
                            "ingested_at": doc.ingested_at.isoformat(),
                        },
                    }
                )

            # Upsert to Qdrant
            await self.qdrant.upsert_points(collection, points)

            doc.chunks = chunks
            logger.info(f"Stored {len(chunks)} chunks from {doc.url}")

        except Exception as e:
            logger.error(f"Error storing document: {e}")
            doc.error = f"Storage error: {e}"

    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        # Remove excessive whitespace
        content = re.sub(r"\n\s*\n", "\n\n", content)
        content = re.sub(r" +", " ", content)

        # Remove common boilerplate
        boilerplate_patterns = [
            r"Cookies help us deliver.*?privacy policy",
            r"By using our site.*?agree to our",
            r"Skip to main content",
            r"Subscribe to our newsletter",
        ]

        for pattern in boilerplate_patterns:
            content = re.sub(pattern, "", content, flags=re.IGNORECASE | re.DOTALL)

        return content.strip()

    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title from markdown content."""
        # Look for first heading
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Fallback to first line
        lines = content.split("\n")
        for line in lines[:3]:
            line = line.strip()
            if line and not line.startswith("#"):
                return line[:100]  # Truncate long titles

        return None

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None


def create_ingestion_service(
    rag_config: RAGConfig,
    embedder: EmbeddingService,
    chunker: DocumentChunker,
    qdrant: QdrantManager,
    mcp_broker: Optional[MCPBroker] = None,
    allowed_domains: Optional[List[str]] = None,
    blocked_domains: Optional[List[str]] = None,
) -> URLIngestionService:
    """
    Create URL ingestion service.

    Args:
        rag_config: RAG configuration
        embedder: Embedding service
        chunker: Document chunker
        qdrant: Qdrant manager
        mcp_broker: Optional MCP broker
        allowed_domains: Whitelist of allowed domains
        blocked_domains: Blacklist of blocked domains

    Returns:
        Configured URLIngestionService
    """
    config = IngestionConfig(
        allowed_domains=set(allowed_domains or []),
        blocked_domains=set(blocked_domains or []),
    )

    return URLIngestionService(
        config=config,
        rag_config=rag_config,
        embedder=embedder,
        chunker=chunker,
        qdrant=qdrant,
        mcp_broker=mcp_broker,
    )
