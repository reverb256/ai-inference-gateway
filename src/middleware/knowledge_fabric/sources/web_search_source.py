"""
Web Search Knowledge Source Adapter for Knowledge Fabric

Provides MCP-based web search integration using the web_search_prime tool.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List
import httpx

from ..core import (
    KnowledgeChunk,
    KnowledgeResult,
    SourceCapability,
    SourcePriority,
)

logger = logging.getLogger(__name__)


@dataclass
class WebSearchKnowledgeSource:
    """
    Web search knowledge source via MCP web_search_prime integration.

    Provides real-time web search capabilities through the MCP broker,
    aggregating results from multiple search engines.
    """
    mcp_url: str = "http://127.0.0.1:8080/mcp/call"
    max_results: int = 5
    timeout: float = 30.0
    name: str = "web_search"
    description: str = "Web search via MCP web_search_prime"
    priority: SourcePriority = SourcePriority.MEDIUM
    capabilities: SourceCapability = SourceCapability.REALTIME | SourceCapability.FACTUAL
    enabled: bool = True

    def can_handle(self, capabilities: "SourceCapability") -> bool:
        """Check if this source has the required capabilities."""
        return bool(self.capabilities & capabilities)

    async def retrieve(self, query: str, **kwargs) -> KnowledgeResult:
        """
        Execute web search via MCP web_search_prime tool.

        Uses JSON-RPC 2.0 over HTTP with SSE response handling.
        """
        import time
        start = time.time()

        # Sanitize query
        sanitized_query = query[:500]

        chunks = []
        metadata = {
            "tool": "webSearchPrime",
            "mcp_server": "web-search-prime",
        }

        try:
            # Prepare MCP JSON-RPC request
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "webSearchPrime",
                    "arguments": {
                        "search_query": sanitized_query,
                    }
                }
            }

            # Critical header for MCP (from AGENTS.md)
            headers = {
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.mcp_url,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 200:
                    # Parse JSON response
                    data = response.json()

                    # Handle JSON-RPC response
                    if data.get("result"):
                        result_data = data["result"]

                        # Extract search results
                        if "content" in result_data:
                            content = result_data["content"]
                            if isinstance(content, list):
                                items = content
                            elif isinstance(content, str):
                                items = [{"title": "Search Result", "snippet": content}]
                            else:
                                items = []

                            for idx, item in enumerate(items[:self.max_results]):
                                title = item.get("title", "")
                                snippet = item.get("snippet", item.get("text", ""))
                                url = item.get("url", "")

                                content_text = f"{title}\n{snippet}"

                                chunk = KnowledgeChunk(
                                    content=content_text,
                                    source=self.name,
                                    score=1.0 - (idx * 0.1),
                                    metadata={
                                        "url": url,
                                        "title": title,
                                    },
                                    capabilities=self.capabilities,
                                )
                                chunks.append(chunk)

                        metadata["total_results"] = len(chunks)
                    else:
                        # Error in MCP response
                        error_msg = data.get("error", {}).get("message", "Unknown MCP error")
                        metadata["error"] = error_msg
                        logger.warning(f"MCP error: {error_msg}")
                else:
                    metadata["error"] = f"HTTP {response.status_code}"
                    logger.warning(f"MCP returned status {response.status_code}")

        except httpx.ConnectError as e:
            metadata["error"] = "Cannot connect to MCP broker"
            metadata["error_type"] = "connection_error"
            metadata["suggestion"] = "Check if AI gateway is running"
            logger.error(f"Web search connection error: {e}")

        except httpx.TimeoutException:
            metadata["error"] = "Request timeout"
            metadata["error_type"] = "timeout"
            logger.error(f"Web search timeout after {self.timeout}s")

        except Exception as e:
            metadata["error"] = str(e)
            metadata["error_type"] = type(e).__name__
            logger.exception(f"Web search unexpected error: {e}")

        retrieval_time = time.time() - start

        return KnowledgeResult(
            source_name=self.name,
            chunks=chunks,
            query=query,
            retrieval_time=retrieval_time,
            metadata=metadata,
        )


def create_web_search_source(
    mcp_url: str = "http://127.0.0.1:8080/mcp/call",
    max_results: int = 5,
) -> WebSearchKnowledgeSource:
    """Factory function to create web search knowledge source."""
    return WebSearchKnowledgeSource(
        mcp_url=mcp_url,
        max_results=max_results,
    )
