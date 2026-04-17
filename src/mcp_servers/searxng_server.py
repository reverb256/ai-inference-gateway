#!/usr/bin/env python3
"""
SearXNG MCP Server

Exposes SearXNG metasearch as MCP tools for AI agents.
Based on the official MCP server pattern from modelcontextprotocol/servers.

Usage:
    python -m ai_inference_gateway.mcp_servers.searxng_server

Environment Variables:
    SEARXNG_URL: SearXNG instance URL (default: http://10.4.98.141:7777)
    SEARXNG_CACHE_TTL: Cache TTL in seconds (default: 300)

Configuration for Claude.app/Cursor:
    {
      "mcpServers": {
        "searxng": {
          "command": "python",
          "args": ["-m", "ai_inference_gateway.mcp_servers.searxng_server"],
          "env": {
            "SEARXNG_URL": "http://10.4.98.141:7777"
          }
        }
      }
    }
"""

import asyncio
import json
import logging
import os
from typing import Annotated

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions
from pydantic import BaseModel, Field

# Import SearXNG integration
try:
    from ai_inference_gateway.searxng_integration import SearxngIntegration
    SEARXNG_AVAILABLE = True
except ImportError:
    SEARXNG_AVAILABLE = False
    SearxngIntegration = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server metadata
SERVER_NAME = "mcp-searxng"
SERVER_VERSION = "1.0.0"

# SearXNG configuration
# Default: Kubernetes service DNS
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://10.4.98.141:7777")
SEARXNG_CACHE_TTL = int(os.getenv("SEARXNG_CACHE_TTL", "300"))


# ============================================================================
# INPUT SCHEMAS (Pydantic models for validation)
# ============================================================================

class WebSearchParams(BaseModel):
    """Parameters for web search."""

    query: Annotated[str, Field(description="Search query string")]
    category: Annotated[
        str,
        Field(
            description="Search category",
            default="general",
        ),
    ]
    max_results: Annotated[
        int,
        Field(
            default=10,
            description="Maximum number of results to return",
            ge=1,
            le=50,
        ),
    ]
    language: Annotated[
        str,
        Field(
            default="all",
            description="Language filter (e.g., 'en', 'fr', 'de', 'all')",
        ),
    ]
    time_range: Annotated[
        str | None,
        Field(
            default=None,
            description="Time range filter (day, week, month, year)",
        ),
    ]
    use_cache: Annotated[
        bool,
        Field(
            default=True,
            description="Use cached results if available",
        ),
    ]


class EmptyParams(BaseModel):
    """Empty parameters for stats/cache operations."""
    pass


class SiteSearchParams(BaseModel):
    """Parameters for site-specific search."""

    query: Annotated[str, Field(description="Search query string")]
    max_results: Annotated[
        int,
        Field(
            default=10,
            description="Maximum number of results to return",
            ge=1,
            le=50,
        ),
    ]
    use_cache: Annotated[
        bool,
        Field(
            default=True,
            description="Use cached results if available",
        ),
    ]


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

TOOLS: list[Tool] = [
    # ========================================================================
    # GENERAL SEARCH TOOLS
    # ========================================================================
    Tool(
        name="web_search",
        description=(
            "Perform web search using SearXNG metasearch engine. "
            "Aggregates results from multiple search engines (Google, Bing, DuckDuckGo, etc.). "
            "Returns relevant results with auto-improving features like query pattern learning "
            "and adaptive engine selection. Supports different categories: general, images, videos, "
            "news, science, IT, files, music, map."
        ),
        inputSchema=WebSearchParams.model_json_schema(),
    ),

    # ========================================================================
    # DOMAIN-SPECIFIC SEARCH TOOLS (AI-Optimized)
    # ========================================================================
    Tool(
        name="search_code",
        description=(
            "Search for code examples, libraries, and implementations with AI optimization. "
            "Intelligently routes queries to GitHub, StackOverflow, GitLab, and developer docs. "
            "Results are quality-scored for relevance. Use for: finding implementations, "
            "code examples, API usage, library documentation, troubleshooting code issues."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_research",
        description=(
            "Search academic papers, research, and technical documentation. "
            "Routes queries to Google Scholar, ArXiv, Semantic Scholar, and academic sources. "
            "Results are quality-scored for academic relevance. Use for: finding papers, "
            "researching algorithms, scholarly articles, technical documentation, state-of-the-art methods."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_devops",
        description=(
            "Search DevOps, infrastructure, and deployment content. "
            "Routes queries to Docker Hub, GitLab, StackExchange, and infrastructure docs. "
            "Results are quality-scored for operations relevance. Use for: Kubernetes deployments, "
            "Docker configurations, CI/CD pipelines, infrastructure automation, monitoring."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_data",
        description=(
            "Search data science, ML, and AI-related content. "
            "Routes queries to HuggingFace, Kaggle, ArXiv ML papers, and AI repositories. "
            "Results are quality-scored for data science relevance. Use for: finding datasets, "
            "ML models, training techniques, AI research, data engineering, LLM fine-tuning."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),

    # ========================================================================
    # SITE-SPECIFIC SEARCH TOOLS
    # ========================================================================
    Tool(
        name="search_github",
        description=(
            "Search GitHub for code, repositories, and developers. "
            "Finds relevant GitHub repositories, code snippets, and discussions. "
            "Useful for finding open source libraries, examples, and implementations."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_nixos_options",
        description=(
            "Search NixOS configuration options documentation. "
            "Directly searches the official NixOS options manual for configuration settings. "
            "Useful for finding the correct NixOS configuration options for services and settings."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_mdn",
        description=(
            "Search MDN Web Docs for web development documentation. "
            "Searches Mozilla Developer Network for HTML, CSS, JavaScript, and web API references. "
            "Useful for finding accurate web development documentation and examples."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_stackoverflow",
        description=(
            "Search Stack Overflow for programming Q&A. "
            "Finds solutions to common programming problems and errors. "
            "Useful for troubleshooting and finding practical code solutions."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),
    Tool(
        name="search_reddit",
        description=(
            "Search Reddit for community discussions. "
            "Finds discussions on subreddits for programming, Linux, self-hosting, and more. "
            "Useful for finding community opinions and real-world experiences."
        ),
        inputSchema=SiteSearchParams.model_json_schema(),
    ),

    # ========================================================================
    # UTILITY TOOLS
    # ========================================================================
    Tool(
        name="search_stats",
        description=(
            "Get SearXNG learning statistics including query patterns, "
            "engine performance metrics, cache size, and top cached queries. "
            "Useful for understanding what the system has learned from previous searches."
        ),
        inputSchema=EmptyParams.model_json_schema(),
    ),
    Tool(
        name="clear_search_cache",
        description=(
            "Clear the SearXNG response cache. Useful when you want fresh results "
            "or to free up memory. Cached results are automatically refreshed after 5 minutes."
        ),
        inputSchema=EmptyParams.model_json_schema(),
    ),
    Tool(
        name="ping_searxng",
        description=(
            "Check if SearXNG service is accessible and healthy. "
            "Returns status and connection information."
        ),
        inputSchema=EmptyParams.model_json_schema(),
    ),
]


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for SearXNG MCP server."""
    if not SEARXNG_AVAILABLE:
        logger.error("SearXNG integration not available. Exiting.")
        logger.error("Ensure searxng_integration.py is in the Python path.")
        return

    logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
    logger.info(f"SearXNG URL: {SEARXNG_URL}")
    logger.info(f"Cache TTL: {SEARXNG_CACHE_TTL}s")

    # Initialize SearXNG integration
    searxng = SearxngIntegration(cache_ttl=SEARXNG_CACHE_TTL)
    logger.info(f"SearXNG MCP Server initialized (url={SEARXNG_URL}, cache_ttl={SEARXNG_CACHE_TTL}s)")

    # Create MCP server
    server = Server(SERVER_NAME)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Handle tool calls."""

        try:
            if name == "web_search":
                params = WebSearchParams(**arguments)
                result = await searxng.search(
                    query=params.query,
                    category=params.category,
                    max_results=params.max_results,
                    language=params.language,
                    time_range=params.time_range,
                    use_cache=params.use_cache,
                    learning_enabled=True,
                )

                # Format results for AI consumption
                if "error" in result:
                    return [TextContent(
                        type="text",
                        text=f"Search Error: {result['error']}"
                    )]

                if not result.get("results"):
                    return [TextContent(
                        type="text",
                        text=f"No results found for query: '{params.query}'"
                    )]

                # Build formatted response
                lines = []
                lines.append(f"# Search Results for: {params.query}")
                lines.append(f"**Category:** {result.get('category', 'general')}")
                lines.append(f"**Cached:** {result.get('cached', False)}")
                lines.append(f"**Engines:** {', '.join(result.get('engines_used', []))}")
                lines.append("")

                for i, item in enumerate(result.get("results", [])[:params.max_results], 1):
                    lines.append(f"## {i}. {item.get('title', 'Untitled')}")
                    lines.append(f"- **URL:** {item.get('url', 'N/A')}")
                    lines.append(f"- **Engine:** {item.get('engine', 'unknown')}")

                    if item.get('content'):
                        # Truncate long content
                        content = item['content'][:300]
                        if len(item['content']) > 300:
                            content += "..."
                        lines.append(f"- **Snippet:** {content}")
                    lines.append("")

                # Include suggestions if available
                if result.get("suggestions"):
                    lines.append("## Suggestions")
                    for suggestion in result["suggestions"]:
                        lines.append(f"- **{suggestion.get('suggestion', '')}:** {suggestion.get('reason', '')}")

                return [TextContent(
                    type="text",
                    text="\n".join(lines)
                )]

            # ========================================================================
            # DOMAIN-SPECIFIC SEARCH HANDLERS (AI-Optimized)
            # ========================================================================
            elif name in ("search_code", "search_research", "search_devops", "search_data"):
                params = SiteSearchParams(**arguments)

                # Map tool names to domains
                domain_mapping = {
                    "search_code": "code",
                    "search_research": "research",
                    "search_devops": "devops",
                    "search_data": "data",
                }

                domain = domain_mapping.get(name, "general")

                # Use domain-aware search routing
                result = await searxng.search_with_domain_routing(
                    query=params.query,
                    domain=domain,
                    max_results=params.max_results,
                    use_cache=params.use_cache,
                )

                # Format results for AI consumption
                if "error" in result:
                    return [TextContent(
                        type="text",
                        text=f"Search Error ({domain}): {result['error']}"
                    )]

                if not result.get("results"):
                    return [TextContent(
                        type="text",
                        text=f"No results found for {domain} query: '{params.query}'"
                    )]

                # Build formatted response with quality scores
                lines = []
                domain_display = domain.capitalize()
                lines.append(f"# {domain_display} Search Results for: {params.query}")
                lines.append(f"**Domain:** {domain}")

                # Add routing metadata if available
                if result.get("routing"):
                    routing = result["routing"]
                    lines.append(f"**Detected Domain:** {routing.get('detected_domain', domain)}")
                    lines.append(f"**Engines Used:** {', '.join(routing.get('engines_selected', []))}")
                    lines.append(f"**Quality Scoring:** Enabled")

                lines.append(f"**Cached:** {result.get('cached', False)}")
                lines.append("")

                for i, item in enumerate(result.get("results", [])[:params.max_results], 1):
                    lines.append(f"## {i}. {item.get('title', 'Untitled')}")
                    lines.append(f"- **URL:** {item.get('url', 'N/A')}")
                    lines.append(f"- **Engine:** {item.get('engine', 'unknown')}")
                    lines.append(f"- **Quality Score:** {item.get('quality_score', 0):.2f}")

                    if item.get('content'):
                        content = item['content'][:400]
                        if len(item['content']) > 400:
                            content += "..."
                        lines.append(f"- **Snippet:** {content}")
                    lines.append("")

                return [TextContent(
                    type="text",
                    text="\n".join(lines)
                )]

            # Site-specific search handlers
            elif name in ("search_github", "search_nixos_options", "search_mdn",
                          "search_stackoverflow", "search_reddit"):
                params = SiteSearchParams(**arguments)

                # Map tool names to their search sites
                site_mappings = {
                    "search_github": ("github.com", "GitHub"),
                    "search_nixos_options": ("search.nixos.org/options", "NixOS Options"),
                    "search_mdn": ("developer.mozilla.org", "MDN Web Docs"),
                    "search_stackoverflow": ("stackoverflow.com", "Stack Overflow"),
                    "search_reddit": ("reddit.com", "Reddit"),
                }

                site, site_name = site_mappings.get(name, ("", ""))

                # Build site-specific query
                site_query = f"site:{site} {params.query}"

                result = await searxng.search(
                    query=site_query,
                    category="general",
                    max_results=params.max_results,
                    language="all",
                    time_range=None,
                    use_cache=params.use_cache,
                    learning_enabled=True,
                )

                # Format results for AI consumption
                if "error" in result:
                    return [TextContent(
                        type="text",
                        text=f"Search Error on {site_name}: {result['error']}"
                    )]

                if not result.get("results"):
                    return [TextContent(
                        type="text",
                        text=f"No results found on {site_name} for query: '{params.query}'"
                    )]

                # Build formatted response
                lines = []
                lines.append(f"# {site_name} Search Results for: {params.query}")
                lines.append(f"**Site:** {site}")
                lines.append(f"**Cached:** {result.get('cached', False)}")
                lines.append(f"**Engines:** {', '.join(result.get('engines_used', []))}")
                lines.append("")

                for i, item in enumerate(result.get("results", [])[:params.max_results], 1):
                    lines.append(f"## {i}. {item.get('title', 'Untitled')}")
                    lines.append(f"- **URL:** {item.get('url', 'N/A')}")
                    lines.append(f"- **Engine:** {item.get('engine', 'unknown')}")

                    if item.get('content'):
                        content = item['content'][:400]
                        if len(item['content']) > 400:
                            content += "..."
                        lines.append(f"- **Snippet:** {content}")
                    lines.append("")

                return [TextContent(
                    type="text",
                    text="\n".join(lines)
                )]

            elif name == "search_stats":
                stats = await searxng.get_learning_stats()
                return [TextContent(
                    type="text",
                    text=json.dumps(stats, indent=2)
                )]

            elif name == "clear_search_cache":
                searxng.clear_cache()
                return [TextContent(
                    type="text",
                    text="SearXNG cache cleared successfully"
                )]

            elif name == "ping_searxng":
                try:
                    # Use the SearxngIntegration instance to test connectivity
                    result = await searxng.search(
                        query="test",
                        max_results=1,
                        use_cache=False
                    )

                    if result.get("results") is not None:
                        return [TextContent(
                            type="text",
                            text=json.dumps({
                                "status": "healthy",
                                "service": "SearXNG",
                                "url": SEARXNG_URL,
                                "cache_ttl": SEARXNG_CACHE_TTL,
                                "test_query": "test"
                            }, indent=2)
                        )]
                    else:
                        return [TextContent(
                            type="text",
                            text=f"SearXNG returned unexpected response: {result}"
                        )]
                except Exception as e:
                    import traceback
                    logger.error(f"[ping_searxng] Exception: {type(e).__name__}: {e}")
                    logger.error(f"[ping_searxng] Traceback:\n{traceback.format_exc()}")
                    return [TextContent(
                        type="text",
                        text=f"SearXNG unreachable: {type(e).__name__}: {str(e)}"
                    )]

            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

        except Exception as e:
            logger.exception(f"Error in tool call ({name}): {e}")
            return [TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    # Run the server with stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mcp-searxng",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={}
                )
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
