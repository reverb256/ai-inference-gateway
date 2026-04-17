"""
HTTP-MCP Bridge

Exposes MCP (Model Context Protocol) tools as HTTP endpoints for web-based agents.

This bridge allows:
- HTTP-based tool discovery and execution
- Integration with web agents that can't use stdio MCP transport
- Unified API surface combining OpenAI, MCP, and custom tools
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ai_inference_gateway.mcp_broker import MCPBroker

logger = logging.getLogger(__name__)


class HTTPMCPBridge:
    """
    HTTP bridge for MCP tools.

    Exposes MCP server tools via REST API for easy integration with web agents.
    """

    def __init__(self, mcp_broker: MCPBroker):
        self.mcp_broker = mcp_broker
        self.tool_cache: Dict[str, Any] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl = 60  # Cache tool list for 60 seconds

    async def list_tools(
        self,
        server_name: Optional[str] = None,
        refresh_cache: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List available MCP tools, optionally filtered by server.

        Args:
            server_name: Optional MCP server name to filter by
            refresh_cache: Force refresh of tool cache

        Returns:
            List of tool definitions with schemas
        """
        # Check cache
        if (not refresh_cache and
            self.tool_cache and
            self.cache_timestamp and
            (datetime.now() - self.cache_timestamp).seconds < self.cache_ttl):
            cached_tools = self.tool_cache.get("tools", [])
            if server_name:
                return [t for t in cached_tools if t.get("server") == server_name]
            return cached_tools

        # Fetch from MCP broker
        all_tools = []

        # Get all servers
        servers = await self.mcp_broker.list_servers()
        for server_info in servers:
            server_name = server_info.get("name", {}).get("name") if isinstance(server_info.get("name"), dict) else server_info.get("name")
            try:
                tools = await self.mcp_broker.get_tools(server_name)
                for tool in tools:
                    tool_info = {
                        "name": tool.get("name"),
                        "description": tool.get("description"),
                        "server": server_name,
                        "input_schema": tool.get("inputSchema"),
                    }
                    all_tools.append(tool_info)
            except Exception as e:
                logger.warning(f"Failed to list tools from {server_name}: {e}")

        # Update cache
        self.tool_cache = {"tools": all_tools}
        self.cache_timestamp = datetime.now()

        # Filter by server if requested
        if server_name:
            return [t for t in all_tools if t.get("server") == server_name]

        return all_tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_name: Optional[str] = None,
        timeout: float = 30.0
    ) -> Dict[str, Any]:
        """
        Execute an MCP tool via HTTP.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments (must match input schema)
            server_name: Optional MCP server name (auto-detected if None)
            timeout: Execution timeout in seconds

        Returns:
            Tool execution result
        """
        # Find the tool if server not specified
        if server_name is None:
            tools = await self.list_tools()
            matching = [t for t in tools if t.get("name") == tool_name]
            if not matching:
                raise ValueError(f"Tool '{tool_name}' not found")
            if len(matching) > 1:
                raise ValueError(
                    f"Tool '{tool_name}' found in multiple servers. "
                    f"Please specify server_name: {', '.join(set(t.get('server') for t in matching))}"
                )
            server_name = matching[0]["server"]

        # Execute tool via MCP broker
        try:
            result = await asyncio.wait_for(
                self.mcp_broker.call_tool(server_name, tool_name, arguments),
                timeout=timeout
            )

            return {
                "success": True,
                "server": server_name,
                "tool": tool_name,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }
        except asyncio.TimeoutError:
            return {
                "success": False,
                "server": server_name,
                "tool": tool_name,
                "error": f"Tool execution timed out after {timeout}s",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.exception(f"Tool execution failed: {tool_name}")
            return {
                "success": False,
                "server": server_name,
                "tool": tool_name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_tool_info(
        self,
        tool_name: str,
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed information about a specific tool.

        Args:
            tool_name: Name of the tool
            server_name: Optional server name (auto-detected if None)

        Returns:
            Detailed tool information with examples
        """
        tools = await self.list_tools(refresh_cache=False)
        matching = [t for t in tools if t.get("name") == tool_name]

        if server_name:
            matching = [t for t in matching if t.get("server") == server_name]

        if not matching:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Return first match
        tool_info = matching[0].copy()

        # Add usage examples
        tool_info["examples"] = self._generate_tool_examples(tool_info)

        return tool_info

    def _generate_tool_examples(self, tool_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate usage examples for a tool."""
        examples = []

        tool_name = tool_info.get("name", "")
        schema = tool_info.get("input_schema", {})

        # Generate basic example
        if "properties" in schema:
            example_args = {}
            for prop_name, prop_schema in schema["properties"].items():
                # Generate example based on type
                prop_type = prop_schema.get("type", "string")
                if prop_type == "string":
                    default_val = prop_schema.get("default", "example")
                    example_args[prop_name] = default_val
                elif prop_type == "integer":
                    example_args[prop_name] = prop_schema.get("default", 1)
                elif prop_type == "boolean":
                    example_args[prop_name] = prop_schema.get("default", True)
                elif prop_type == "array":
                    example_args[prop_name] = prop_schema.get("default", [])

            examples.append({
                "description": f"Basic usage of {tool_name}",
                "arguments": example_args,
            })

        return examples

    async def list_servers(self) -> List[Dict[str, Any]]:
        """
        List all available MCP servers.

        Returns:
            List of server information
        """
        servers = await self.mcp_broker.list_servers()

        server_info = []
        for server in servers:
            try:
                # Extract server name from dict
                server_name = server.get("name") if isinstance(server, dict) else server

                # Get tools for this server
                tools = await self.mcp_broker.get_tools(server_name)
                server_info.append({
                    "name": server_name,
                    "tool_count": len(tools),
                    "status": "available",
                })
            except Exception as e:
                server_info.append({
                    "name": server,
                    "tool_count": 0,
                    "status": f"error: {e}",
                })

        return server_info

    async def get_server_health(self, server_name: str) -> Dict[str, Any]:
        """
        Get health status of a specific MCP server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Health status information
        """
        try:
            # Try to list tools as a health check
            tools = await self.mcp_broker.get_tools(server_name)
            return {
                "server": server_name,
                "status": "healthy",
                "tool_count": len(tools),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "server": server_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def clear_cache(self):
        """Clear the tool cache."""
        self.tool_cache = {}
        self.cache_timestamp = None


# Global instance
_http_mcp_bridge: Optional[HTTPMCPBridge] = None


def get_http_mcp_bridge(mcp_broker: MCPBroker) -> HTTPMCPBridge:
    """Get or create global HTTP-MCP bridge."""
    global _http_mcp_bridge
    if _http_mcp_bridge is None:
        _http_mcp_bridge = HTTPMCPBridge(mcp_broker)
    return _http_mcp_bridge
