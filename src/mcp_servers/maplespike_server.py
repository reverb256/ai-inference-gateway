#!/usr/bin/env python3
"""
MapleSpike MCP Server

Exposes MapleSpike tools as MCP tools for AI agents.
This allows the AI Inference Gateway to route MCP tool calls to MapleSpike's
existing functionality (AI Ask, Engine briefs, data pipeline, etc.).

Usage:
    python -m mcp_servers.maplespike_server

Environment Variables:
    MAPLESPIKE_URL: Base URL for MapleSpike instance
                    (default: http://maplespike.maplespike.svc.cluster.local:3000)
    MAPLESPIKE_API_KEY: API key for authenticated endpoints (optional)

Configuration for Claude.app/Cursor:
    {
      "mcpServers": {
        "maplespike": {
          "command": "python",
          "args": ["-m", "mcp_servers.maplespike_server"],
          "env": {
            "MAPLESPIKE_URL": "http://maplespike.maplespike.svc.cluster.local:3000"
          }
        }
      }
    }
"""

import asyncio
import json
import logging
import os
from typing import Annotated, Optional

import httpx

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from mcp.server.models import InitializationOptions
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Server metadata
SERVER_NAME = "mcp-maplespike"
SERVER_VERSION = "1.0.0"

# MapleSpike configuration
# Default: Kubernetes service DNS (maplespike namespace)
MAPLESPIKE_URL = os.getenv(
    "MAPLESPIKE_URL",
    "http://maplespike.maplespike.svc.cluster.local:3000",
)
MAPLESPIKE_API_KEY = os.getenv("MAPLESPIKE_API_KEY", "")

# ============================================================================
# INPUT SCHEMAS (Pydantic models for validation)
# ============================================================================


class AIAskParams(BaseModel):
    """Parameters for AI Ask natural language query."""

    query: Annotated[str, Field(description="Natural language question to ask")]
    context: Annotated[
        Optional[str],
        Field(description="Additional context for the query", default=None),
    ]
    max_sources: Annotated[
        int,
        Field(description="Maximum number of sources to consult", default=5),
    ]


class EngineBriefParams(BaseModel):
    """Parameters for generating an Engine brief."""

    topic: Annotated[str, Field(description="Topic for the brief")]
    depth: Annotated[
        str,
        Field(
            description="Depth of analysis: 'quick', 'standard', 'deep'",
            default="standard",
        ),
    ]
    include_sources: Annotated[
        bool,
        Field(description="Whether to include source citations", default=True),
    ]


class PipelineStatusParams(BaseModel):
    """Parameters for checking data pipeline status."""

    pipeline_name: Annotated[
        Optional[str],
        Field(description="Specific pipeline to check", default=None),
    ]
    include_metrics: Annotated[
        bool,
        Field(description="Whether to include performance metrics", default=True),
    ]


# ============================================================================
# MAPLESPIKE HTTP CLIENT
# ============================================================================


class MapleSpikeClient:
    """HTTP client for communicating with MapleSpike instance."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def __aenter__(self):
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request to MapleSpike."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        url = f"{self.base_url}{path}"
        headers = {**self.headers, **kwargs.pop("headers", {})}

        response = await self._client.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()

    async def ai_ask(
        self,
        query: str,
        context: Optional[str] = None,
        max_sources: int = 5,
    ) -> dict:
        """Submit a natural language query to MapleSpike's AI Ask."""
        payload = {
            "query": query,
            "context": context,
            "max_sources": max_sources,
        }
        return await self._request("POST", "/api/ai-ask", json=payload)

    async def engine_brief(
        self,
        topic: str,
        depth: str = "standard",
        include_sources: bool = True,
    ) -> dict:
        """Generate an Engine brief on a topic."""
        payload = {
            "topic": topic,
            "depth": depth,
            "include_sources": include_sources,
        }
        return await self._request("POST", "/api/engine-brief", json=payload)

    async def pipeline_status(
        self,
        pipeline_name: Optional[str] = None,
        include_metrics: bool = True,
    ) -> dict:
        """Check status of data pipelines."""
        params = {}
        if pipeline_name:
            params["pipeline_name"] = pipeline_name
        params["include_metrics"] = str(include_metrics).lower()

        return await self._request("GET", "/api/pipeline/status", params=params)

    async def health(self) -> dict:
        """Check if MapleSpike is reachable."""
        try:
            return await self._request("GET", "/api/health")
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}


# ============================================================================
# MCP SERVER SETUP
# ============================================================================

server = Server(SERVER_NAME)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MapleSpike tools."""
    return [
        Tool(
            name="ai_ask",
            description=(
                "Ask a natural language question and get an answer with sources "
                "from MapleSpike's sovereign Canadian knowledge base"
            ),
            inputSchema=AIAskParams.model_json_schema(),
        ),
        Tool(
            name="engine_brief",
            description=(
                "Generate a comprehensive intelligence brief on a topic with "
                "deep analysis and curated sources"
            ),
            inputSchema=EngineBriefParams.model_json_schema(),
        ),
        Tool(
            name="pipeline_status",
            description=(
                "Check the status and health of MapleSpike's data ingestion "
                "and processing pipelines"
            ),
            inputSchema=PipelineStatusParams.model_json_schema(),
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool execution requests."""
    async with MapleSpikeClient(MAPLESPIKE_URL, MAPLESPIKE_API_KEY) as client:
        try:

            if name == "ai_ask":
                params = AIAskParams(**arguments)
                result = await client.ai_ask(
                    query=params.query,
                    context=params.context,
                    max_sources=params.max_sources,
                )

            elif name == "engine_brief":
                params = EngineBriefParams(**arguments)
                result = await client.engine_brief(
                    topic=params.topic,
                    depth=params.depth,
                    include_sources=params.include_sources,
                )

            elif name == "pipeline_status":
                params = PipelineStatusParams(**arguments)
                result = await client.pipeline_status(
                    pipeline_name=params.pipeline_name,
                    include_metrics=params.include_metrics,
                )

            else:
                raise ValueError(f"Unknown tool: {name}")

            return [
                TextContent(
                    type="text",
                    text=json.dumps(result, indent=2),
                )
            ]

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"error": str(type(e).__name__), "detail": str(e)},
                        indent=2,
                    ),
                )
            ]


async def main():
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name=SERVER_NAME,
                server_version=SERVER_VERSION,
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
