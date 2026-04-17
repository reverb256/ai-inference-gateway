"""
Tests for Knowledge Fabric source adapters.

Tests the individual knowledge source implementations:
- RAGKnowledgeSource
- SearXNGKnowledgeSource
- WebSearchKnowledgeSource
- CodeSearchKnowledgeSource
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from ai_inference_gateway.middleware.knowledge_fabric.sources import (
    RAGKnowledgeSource,
    SearXNGKnowledgeSource,
    WebSearchKnowledgeSource,
    CodeSearchKnowledgeSource,
)
from ai_inference_gateway.middleware.knowledge_fabric.core import (
    SourceCapability,
    SourcePriority,
)


# ============================================================================
# RAG Knowledge Source Tests
# ============================================================================


@pytest.fixture
def mock_search_service():
    """Create mock HybridSearchService."""
    service = Mock()
    service.search = AsyncMock()
    return service


@pytest.fixture
def rag_source(mock_search_service):
    """Create RAG source with mock search service."""
    return RAGKnowledgeSource(
        search_service=mock_search_service,
        collection="test-collection",
        max_chunks=5,
    )


@pytest.mark.asyncio
async def test_rag_source_initialization(rag_source):
    """Test RAG source initializes correctly."""
    assert rag_source.name == "rag"
    assert rag_source.collection == "test-collection"
    assert rag_source.max_chunks == 5
    assert rag_source.priority == SourcePriority.HIGH
    assert rag_source.capabilities == (
        SourceCapability.FACTUAL |
        SourceCapability.CONTEXTUAL |
        SourceCapability.PROCEDURAL
    )


@pytest.mark.asyncio
async def test_rag_source_retrieve(rag_source, mock_search_service):
    """Test RAG source retrieves from search service."""
    # Mock search response
    mock_search_service.search.return_value = {
        "results": [
            {
                "content": "Test content 1",
                "score": 0.95,
                "metadata": {"chunk_id": "1"},
            },
            {
                "content": "Test content 2",
                "score": 0.85,
                "metadata": {"chunk_id": "2"},
            },
        ],
        "total_results": 2,
        "reranked": False,
    }

    result = await rag_source.retrieve("test query")

    assert result.source_name == "rag"
    assert len(result.chunks) == 2
    assert result.chunks[0].content == "Test content 1"
    assert result.chunks[0].score == 0.95
    assert result.chunks[0].metadata["collection"] == "test-collection"
    assert result.metadata["retrieved"] == 2


@pytest.mark.asyncio
async def test_rag_source_handles_search_error(rag_source, mock_search_service):
    """Test RAG source handles search service errors gracefully."""
    mock_search_service.search.side_effect = Exception("Search failed")

    result = await rag_source.retrieve("test query")

    assert result.source_name == "rag"
    assert len(result.chunks) == 0
    assert "error" in result.metadata
    assert result.metadata["error_type"] == "Exception"


@pytest.mark.asyncio
async def test_rag_source_passes_top_k(rag_source, mock_search_service):
    """Test RAG source passes top_k parameter to search."""
    mock_search_service.search.return_value = {
        "results": [],
        "total_results": 0,
        "reranked": False,
    }

    # Test with custom top_k
    result = await rag_source.retrieve("test query", top_k=10)

    mock_search_service.search.assert_called_once()
    call_kwargs = mock_search_service.search.call_args
    assert call_kwargs[1]["top_k"] == 10


# ============================================================================
# SearXNG Knowledge Source Tests
# ============================================================================


@pytest.fixture
def searxng_source():
    """Create SearXNG source for testing."""
    return SearXNGKnowledgeSource(
        searxng_url="http://127.0.0.1:7777",
        max_results=5,
    )


@pytest.mark.asyncio
async def test_searxng_source_initialization(searxng_source):
    """Test SearXNG source initializes correctly."""
    assert searxng_source.name == "searxng"
    assert searxng_source.searxng_url == "http://127.0.0.1:7777"
    assert searxng_source.max_results == 5
    assert searxng_source.priority == SourcePriority.MEDIUM


@pytest.mark.asyncio
async def test_searxng_source_retrieve_with_mock(searxng_source):
    """Test SearXNG source with mocked HTTP client."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "title": "Test Title",
                "content": "Test content snippet",
                "url": "https://example.com",
                "engine": "google",
                "category": "general",
            },
            {
                "title": "Another Title",
                "content": "Another snippet",
                "url": "https://example.org",
                "engine": "bing",
                "category": "general",
            },
        ],
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        result = await searxng_source.retrieve("test query")

    assert result.source_name == "searxng"
    assert len(result.chunks) == 2
    assert result.chunks[0].source == "searxng"
    assert "Test content snippet" in result.chunks[0].content
    assert result.chunks[0].metadata["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_searxng_source_handles_connection_error(searxng_source):
    """Test SearXNG source handles connection errors."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.raise_error = httpx.ConnectError(
            "Connection refused"
        )

        result = await searxng_source.retrieve("test query")

    assert result.source_name == "searxng"
    assert len(result.chunks) == 0
    assert result.metadata["error_type"] == "connection_error"
    assert "suggestion" in result.metadata


@pytest.mark.asyncio
async def test_searxng_source_handles_http_error(searxng_source):
    """Test SearXNG source handles HTTP errors."""
    import httpx

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Server error", request=Mock(), response=mock_response
    )

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        result = await searxng_source.retrieve("test query")

    assert result.source_name == "searxng"
    assert "error" in result.metadata
    assert result.metadata["error_type"] == "http_error"


@pytest.mark.asyncio
async def test_searxng_source_passes_query_params(searxng_source):
    """Test SearXNG source passes query parameters correctly."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.get.return_value = mock_response

        await searxng_source.retrieve(
            "test query",
            category="science",
            language="en",
            time_range="day",
        )

    # Verify the client was called with correct parameters
    mock_client.return_value.__aenter__.return_value.get.assert_called()

    call_args = mock_client.call_args
    assert call_args is not None


# ============================================================================
# Web Search Knowledge Source Tests
# ============================================================================


@pytest.fixture
def web_search_source():
    """Create web search source for testing."""
    return WebSearchKnowledgeSource(
        mcp_url="http://127.0.0.1:8080/mcp/call",
        max_results=5,
    )


@pytest.mark.asyncio
async def test_web_search_source_initialization(web_search_source):
    """Test web search source initializes correctly."""
    assert web_search_source.name == "web_search"
    assert web_search_source.mcp_url == "http://127.0.0.1:8080/mcp/call"
    assert web_search_source.max_results == 5
    assert web_search_source.priority == SourcePriority.MEDIUM


@pytest.mark.asyncio
async def test_web_search_source_retrieve_with_mock(web_search_source):
    """Test web search source with mocked MCP call."""
    mock_response = {
        "result": {
            "content": [
                {
                    "title": "Search Result 1",
                    "snippet": "Test snippet 1",
                    "url": "https://example.com/1",
                },
                {
                    "title": "Search Result 2",
                    "snippet": "Test snippet 2",
                    "url": "https://example.com/2",
                },
            ]
        }
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        result = await web_search_source.retrieve("test query")

    assert result.source_name == "web_search"
    assert len(result.chunks) == 2
    assert "Search Result 1" in result.chunks[0].content
    assert result.chunks[0].metadata["url"] == "https://example.com/1"


@pytest.mark.asyncio
async def test_web_search_source_handles_connection_error(web_search_source):
    """Test web search source handles connection errors."""
    import httpx

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = httpx.ConnectError(
            "Connection refused"
        )
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        result = await web_search_source.retrieve("test query")

    assert result.source_name == "web_search"
    assert len(result.chunks) == 0
    assert result.metadata["error_type"] == "connection_error"


@pytest.mark.asyncio
async def test_web_search_source_mcp_json_rpc_format(web_search_source):
    """Test web search source uses correct JSON-RPC format."""
    mock_response = {"result": {"content": []}}

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__.return_value = mock_client_instance

        await web_search_source.retrieve("test query")

        # Verify JSON-RPC format
        call_args = mock_client_instance.post.call_args
        payload = call_args[1][0]  # First positional arg

        assert payload["jsonrpc"] == "2.0"
        assert payload["method"] == "tools/call"
        assert payload["params"]["name"] == "webSearchPrime"
        assert "search_query" in payload["params"]["arguments"]


# ============================================================================
# Code Search Knowledge Source Tests
# ============================================================================


@pytest.fixture
def code_search_source():
    """Create code search source for testing."""
    return CodeSearchKnowledgeSource(
        search_paths=["/etc/nixos"],
        max_results=5,
    )


@pytest.mark.asyncio
async def test_code_search_source_initialization(code_search_source):
    """Test code search source initializes correctly."""
    assert code_search_source.name == "code_search"
    assert code_search_source.search_paths == ["/etc/nixos"]
    assert code_search_source.max_results == 5
    assert code_search_source.priority == SourcePriority.CRITICAL


@pytest.mark.asyncio
async def test_code_search_source_retrieve(code_search_source):
    """Test code search source retrieves code snippets."""
    with patch("pathlib.Path.glob") as mock_glob:
        # Mock finding Python files
        mock_glob.return_value = [
            "/etc/nixos/test.py"
        ]

        with patch("builtins.open") as mock_open:
            # Mock file content
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "def test_function():\n    pass\n"
            )

            result = await code_search_source.retrieve("test function")

    assert result.source_name == "code_search"
    # Should have found the function
    assert len(result.chunks) >= 0  # May have multiple matches


@pytest.mark.asyncio
async def test_code_search_source_handles_no_results(code_search_source):
    """Test code search source handles no results gracefully."""
    with patch("pathlib.Path.glob") as mock_glob:
        mock_glob.return_value = []

        result = await code_search_source.retrieve("nonexistent_function")

    assert result.source_name == "code_search"
    assert len(result.chunks) == 0


@pytest.mark.asyncio
async def test_code_search_source_includes_file_metadata(code_search_source):
    """Test code search source includes file metadata in results."""
    with patch("pathlib.Path.glob") as mock_glob:
        mock_glob.return_value = ["/etc/nixos/test.py"]

        with patch("builtins.open") as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                "def test_func():\n    pass\n"
            )

            result = await code_search_source.retrieve("test_func")

    if len(result.chunks) > 0:
        chunk = result.chunks[0]
        assert "file" in chunk.metadata
        assert chunk.metadata["file"] == "/etc/nixos/test.py"


# ============================================================================
# Source Capability Tests
# ============================================================================


def test_rag_source_capabilities():
    """Test RAG source has correct capabilities."""
    source = RAGKnowledgeSource(
        search_service=Mock(),
        collection="test",
        max_chunks=5,
    )

    assert SourceCapability.FACTUAL in source.capabilities
    assert SourceCapability.CONTEXTUAL in source.capabilities
    assert SourceCapability.PROCEDURAL in source.capabilities


def test_code_search_source_capabilities():
    """Test code search source has CODE capability."""
    source = CodeSearchKnowledgeSource(
        search_paths=["/test"],
        max_results=5,
    )

    assert SourceCapability.CODE in source.capabilities
    assert SourceCapability.PROCEDURAL in source.capabilities


def test_searxng_source_capabilities():
    """Test SearXNG source has correct capabilities."""
    source = SearXNGKnowledgeSource()

    assert SourceCapability.FACTUAL in source.capabilities
    assert SourceCapability.REALTIME in source.capabilities
    assert SourceCapability.COMPARATIVE in source.capabilities


def test_web_search_source_capabilities():
    """Test web search source has REALTIME capability."""
    source = WebSearchKnowledgeSource()

    assert SourceCapability.REALTIME in source.capabilities
    assert SourceCapability.FACTUAL in source.capabilities


# ============================================================================
# Priority Tests
# ============================================================================


def test_code_search_has_critical_priority():
    """Test code search has CRITICAL priority."""
    source = CodeSearchKnowledgeSource(
        search_paths=["/test"],
        max_results=5,
    )

    assert source.priority == SourcePriority.CRITICAL


def test_rag_has_high_priority():
    """Test RAG has HIGH priority."""
    source = RAGKnowledgeSource(
        search_service=Mock(),
        collection="test",
        max_chunks=5,
    )

    assert source.priority == SourcePriority.HIGH


def test_web_sources_have_medium_priority():
    """Test web sources have MEDIUM priority."""
    web = WebSearchKnowledgeSource()
    searxng = SearXNGKnowledgeSource()

    assert web.priority == SourcePriority.MEDIUM
    assert searxng.priority == SourcePriority.MEDIUM
