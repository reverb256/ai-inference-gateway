"""
MCP Broker for AI Inference Gateway.

Proxies requests to configured MCP servers, both local (stdio)
and remote (SSE/HTTP). Provides unified API for all MCP tools.

This is a foundational implementation that can be extended with:
- Full MCP protocol support (stdio, SSE, HTTP transports)
- Tool schema caching and validation
- Request/response logging
- Health monitoring and circuit breaking
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import httpx

logger = logging.getLogger(__name__)

# Import tool schema caching
try:
    from ai_inference_gateway.mcp_cache import get_cache  # noqa: F401

    CACHE_AVAILABLE = True
except ImportError:
    logger.warning("MCP cache module not available")
    CACHE_AVAILABLE = False


class MCPServerType(Enum):
    """MCP server types."""

    LOCAL = "local"  # stdio-based servers
    REMOTE = "remote"  # HTTP/SSE servers


@dataclass
class MCPServer:
    """MCP server configuration."""

    name: str
    type: MCPServerType
    command: Optional[List[str]] = None
    url: Optional[str] = None
    headers: Dict[str, str] = None
    environment: Dict[str, str] = None
    process: Optional[asyncio.subprocess.Process] = None


class MCPBroker:
    """
    Manages MCP server connections and tool calls.

    Provides unified access to MCP tools from multiple servers.
    """

    def __init__(
        self,
        servers: List[MCPServer],
        cache_ttl_seconds: int = 300,
        enable_cache: bool = True,
    ):
        """
        Initialize MCP broker.

        Args:
            servers: List of MCP server configurations
            cache_ttl_seconds: TTL for cached tool schemas (default: 300s = 5 min)
            enable_cache: Enable tool schema caching
        """
        self.servers = {server.name: server for server in servers}
        self.active_connections: Dict[str, Any] = {}
        # Track MCP server initialization state (MCP protocol requires initialize before tools/call)
        self._initialized_servers: Dict[str, bool] = {}

        # Initialize tool schema cache
        self.enable_cache = enable_cache and CACHE_AVAILABLE
        if self.enable_cache:
            self.tool_cache = get_cache(ttl_seconds=cache_ttl_seconds)
            logger.info(
                f"MCP Broker initialized with {len(servers)} servers "
                f"(cache enabled, TTL={cache_ttl_seconds}s)"
            )
        else:
            self.tool_cache = None
            logger.info(
                f"MCP Broker initialized with {len(servers)} servers (cache disabled)"
            )

    async def _ensure_initialized(self, server: MCPServer) -> bool:
        """
        Ensure MCP server is initialized (required by MCP protocol).

        Many MCP servers require an initialize handshake before
        accepting tools/call requests. This method calls initialize if not already done.

        Args:
            server: MCP server configuration

        Returns:
            True if initialization succeeded, False otherwise
        """
        # Skip initialization for local servers (stdio manages its own handshake)
        if server.type != MCPServerType.REMOTE or not server.url:
            return True

        # Check if already initialized
        if server.name in self._initialized_servers:
            return self._initialized_servers[server.name]

        try:
            headers = {"Content-Type": "application/json"}
            if server.headers:
                # Process headers - handle file paths for API keys
                for header_name, header_value in server.headers.items():
                    if isinstance(header_value, str) and (
                        "-key" in header_name.lower()
                        or "_key" in header_name.lower()
                        or "token" in header_name.lower()
                    ):
                        try:
                            if header_value.startswith("Bearer "):
                                file_path = header_value.split(" ", 1)[1].strip()
                                use_bearer = True
                            else:
                                file_path = header_value
                                use_bearer = False

                            with open(file_path, "r") as f:
                                api_key = f.read().strip()
                                if use_bearer:
                                    headers[header_name] = f"Bearer {api_key}"
                                else:
                                    headers[header_name] = api_key
                        except Exception as e:
                            logger.warning(f"Failed to read API key for init: {e}")
                            headers[header_name] = header_value
                    else:
                        headers[header_name] = header_value

            # MCP initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "ai-inference-gateway",
                        "version": "2.0.0"
                    }
                }
            }

            headers["Accept"] = "application/json, text/event-stream"

            logger.warning(f"Initializing MCP server: {server.name}")

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    server.url, json=init_request, headers=headers
                )

                if response.status_code == 200:
                    # Parse SSE response
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" in content_type:
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                try:
                                    data = json.loads(line[5:].strip())
                                    if "result" in data:
                                        logger.warning(
                                            f"Successfully initialized {server.name}: {data['result'].get('serverInfo', {})}"
                                        )
                                        self._initialized_servers[server.name] = True
                                        return True
                                except json.JSONDecodeError:
                                    continue
                    else:
                        # JSON response
                        result = response.json()
                        if "result" in result:
                            logger.warning(
                                f"Successfully initialized {server.name}: {result['result'].get('serverInfo', {})}"
                            )
                            self._initialized_servers[server.name] = True
                            return True

                logger.warning(f"Failed to initialize {server.name}: HTTP {response.status_code}")
                self._initialized_servers[server.name] = False
                return False

        except Exception as e:
            logger.error(f"Error initializing {server.name}: {e}")
            self._initialized_servers[server.name] = False
            return False

    async def list_servers(self) -> List[Dict]:
        """
        List all configured MCP servers.

        Returns:
            List of server information dicts
        """
        servers_list = []
        for name, server in self.servers.items():
            servers_list.append(
                {
                    "name": server.name,
                    "type": server.type.value,
                    "url": server.url if server.type == MCPServerType.REMOTE else None,
                    "healthy": await self.health_check(name),
                }
            )
        return servers_list

    async def get_tools(self, server_name: Optional[str] = None) -> List[Dict]:
        """
        Get available tools from server(s) with caching support.

        Args:
            server_name: Optional server name. If None, returns tools from all servers.

        Returns:
            List of available tools
        """
        if server_name:
            servers_to_check = [server_name]
        else:
            servers_to_check = list(self.servers.keys())

        all_tools = []
        for name in servers_to_check:
            if name not in self.servers:
                logger.warning(f"Server {name} not found")
                continue

            server = self.servers[name]

            # Use cache if enabled for remote servers
            if self.enable_cache and server.type == MCPServerType.REMOTE:
                # Define fetch function for this server
                async def fetch_func():
                    return await self._fetch_tools_from_server(name, server)

                # Use cache to get tools
                tools = await self.tool_cache.get_tools(
                    server_name=name, fetch_func=fetch_func, force_refresh=False
                )

                if tools:
                    all_tools.extend(tools)
            else:
                # No cache - fetch directly
                tools = await self._fetch_tools_from_server(name, server)
                if tools:
                    all_tools.extend(tools)

        return all_tools

    async def _fetch_tools_from_server(
        self, name: str, server: MCPServer
    ) -> Optional[List[Dict]]:
        """
        Fetch tools from a specific MCP server.

        Args:
            name: Server name
            server: Server configuration

        Returns:
            List of tool definitions, or None if fetch fails
        """
        # For remote servers, use MCP protocol to list tools
        if server.type == MCPServerType.REMOTE and server.url:
            try:
                headers = {"Content-Type": "application/json"}
                if server.headers:
                    headers.update(server.headers)

                # Use MCP JSON-RPC protocol to list tools
                mcp_request = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}

                # MCP servers require SSE-capable Accept header
                headers["Accept"] = "application/json, text/event-stream"

                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        server.url, json=mcp_request, headers=headers
                    )

                    if response.status_code == 200:
                        # Check if response is SSE format
                        content_type = response.headers.get("content-type", "")
                        if "text/event-stream" in content_type:
                            # Parse SSE response to get tools list
                            tools = await self._parse_sse_tools_response(response, name)
                        else:
                            # Parse JSON response
                            tools = await self._parse_json_tools_response(
                                response, name
                            )

                        if tools:
                            return tools
                        else:
                            logger.warning(f"No tools found in response from {name}")

                    else:
                        logger.warning(
                            f"Failed to list tools from {name}: HTTP {response.status_code}"
                        )
                        return None

            except Exception as e:
                logger.error(f"Error listing tools from {name}: {e}")
                return None

        else:
            # Local server - use stdio communication
            return await self._fetch_tools_from_local_server(server)

        return None

    async def _parse_sse_tools_response(
        self, response: httpx.Response, server_name: str
    ) -> Optional[List[Dict]]:
        """Parse SSE (Server-Sent Events) response for tools."""
        tools = []

        try:
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    try:
                        data = json.loads(line[5:].strip())
                        if "result" in data and "tools" in data["result"]:
                            for tool in data["result"]["tools"]:
                                tools.append(
                                    {
                                        "server": server_name,
                                        "name": tool.get("name"),
                                        "description": tool.get("description", ""),
                                        "type": "remote",
                                        "inputSchema": tool.get("inputSchema", {}),
                                    }
                                )
                            break  # Got the tools, stop parsing
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error parsing SSE response from {server_name}: {e}")

        return tools if tools else None

    async def _parse_json_tools_response(
        self, response: httpx.Response, server_name: str
    ) -> Optional[List[Dict]]:
        """Parse JSON response for tools."""
        try:
            result = response.json()

            # Check for JSON-RPC error
            if "error" in result:
                logger.warning(
                    f"MCP server {server_name} returned error: {result['error']}"
                )
                return None

            # Extract tools from MCP response
            if "result" in result and "tools" in result["result"]:
                tools = []
                for tool in result["result"]["tools"]:
                    tools.append(
                        {
                            "server": server_name,
                            "name": tool.get("name"),
                            "description": tool.get("description", ""),
                            "type": "remote",
                            "inputSchema": tool.get("inputSchema", {}),
                        }
                    )
                return tools

            logger.warning(f"Unexpected response from MCP server {server_name}")
            return None

        except Exception as e:
            logger.error(f"Error parsing JSON response from {server_name}: {e}")
            return None

    async def invalidate_cache(
        self, server_name: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Invalidate cached tool schemas.

        Args:
            server_name: Specific server to invalidate, or None for all servers

        Returns:
            Dict of {server_name: success_status}
        """
        if not self.enable_cache:
            return {"error": "Cache is not enabled"}

        if server_name:
            success = await self.tool_cache.invalidate(server_name)
            return {server_name: success}
        else:
            count = await self.tool_cache.invalidate_all()
            logger.info(f"Invalidated all {count} cached servers")
            return {"all_servers": True, "count": count}

    def get_cache_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get cache performance metrics.

        Returns:
            Cache metrics dict, or None if cache disabled
        """
        if not self.enable_cache:
            return None

        return self.tool_cache.get_metrics()

    async def warm_up_cache(self) -> Dict[str, bool]:
        """
        Warm up cache by fetching tools from all servers.

        Returns:
            Dict of {server_name: success_status}
        """
        if not self.enable_cache:
            return {"error": "Cache is not enabled"}

        # Build fetch functions for all remote servers
        servers = {}
        for name, server in self.servers.items():
            if server.type == MCPServerType.REMOTE and server.url:
                servers[name] = lambda n=name, s=server: self._fetch_tools_from_server(
                    n, s
                )

        if servers:
            return await self.tool_cache.warm_up(servers, max_concurrency=5)

        return {}

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Dict
    ) -> Dict:
        """
        Call a tool on a specific MCP server.

        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        if server_name not in self.servers:
            return {
                "error": f"Server {server_name} not found",
                "available_servers": list(self.servers.keys()),
            }

        server = self.servers[server_name]

        if server.type == MCPServerType.REMOTE and server.url:
            # For remote servers, make HTTP call
            return await self._call_remote_tool(server, tool_name, arguments)
        else:
            # For local servers, use stdio communication
            return await self._call_local_tool(server, tool_name, arguments)

    async def _call_remote_tool(
        self, server: MCPServer, tool_name: str, arguments: Dict
    ) -> Dict:
        """
        Call a tool on a remote MCP server via direct MCP JSON-RPC protocol.

        Args:
            server: MCP server configuration
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Call direct MCP protocol
        return await self._call_direct_mcp(server, tool_name, arguments)

    async def _call_direct_mcp(
        self, server: MCPServer, tool_name: str, arguments: Dict
    ) -> Dict:
        """
        Call a tool on a remote MCP server via direct MCP JSON-RPC protocol.

        Args:
            server: MCP server configuration
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # CRITICAL: Ensure MCP server is initialized before making tool calls
        # Many MCP servers require initialize handshake before tools/call
        if not await self._ensure_initialized(server):
            logger.warning(
                f"Failed to initialize {server.name}, tool call may fail"
            )

        logger.info(f"=== _call_direct_mcp called for {server.name}.{tool_name} ===")
        logger.info(f"server.headers = {server.headers}")

        try:
            # Build headers
            headers = {"Content-Type": "application/json"}
            logger.info(
                f"Processing MCP call for {server.name}, server.headers: {server.headers}"
            )

            if server.headers:
                # Read API key from file if needed
                for header_name, header_value in server.headers.items():
                    logger.info(
                        f"Processing header {header_name}: {str(header_value)[:50]}..."
                    )

                    if (
                        isinstance(header_value, str)
                        and "/run/" in header_value
                        and "-key" in header_value
                    ):
                        try:
                            logger.info(f"Detected API key file path in {header_name}")
                            if header_value.startswith("Bearer "):
                                file_path = header_value.split(" ", 1)[1].strip()
                                use_bearer = True
                            else:
                                file_path = header_value
                                use_bearer = False

                            logger.info(f"Reading API key from {file_path}")
                            with open(file_path, "r") as f:
                                api_key = f.read().strip()
                                if use_bearer:
                                    headers[header_name] = f"Bearer {api_key}"
                                else:
                                    headers[header_name] = api_key
                            logger.info(
                                f"Successfully loaded API key from {file_path} for {server.name}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to read API key from {file_path}: {e}"
                            )
                            # Fall back to raw value (preserves error context)
                            logger.warning(
                                f"Using raw header value as fallback for {server.name}"
                            )
                            headers[header_name] = header_value
                    else:
                        # Not a file path - use value directly
                        logger.debug(f"Using direct header value for {header_name}")
                        headers[header_name] = header_value

            # Use standard MCP JSON-RPC protocol
            # https://modelcontextprotocol.io/beta/docs/specification/
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }

            # MCP servers require SSE-capable Accept header
            headers["Accept"] = "application/json, text/event-stream"

            # CRITICAL DEBUG: Log the actual headers being sent
            logger.warning(f"=== FINAL HEADERS for {server.name}.{tool_name} ===")
            for k, v in headers.items():
                if k.lower() == "authorization":
                    logger.warning(f"  {k}: {v[:30]}...")
                else:
                    logger.warning(f"  {k}: {v}")

            # Debug logging
            logger.debug(f"Calling MCP tool: {server.name}.{tool_name}")
            logger.debug(f"Request URL: {server.url}")
            logger.debug(f"Request headers: {headers}")
            logger.debug(f"Request body: {mcp_request}")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    server.url, json=mcp_request, headers=headers
                )
                logger.debug(f"Response status: {response.status_code}")

                # Handle SSE response from MCP servers
                if response.status_code == 200:
                    # Check if response is SSE format
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" in content_type:
                        # Parse SSE response
                        async for line in response.aiter_lines():
                            if line.startswith("data:"):
                                try:
                                    data = json.loads(line[5:].strip())
                                    logger.debug(
                                        f"Parsed SSE data: {json.dumps(data)[:500]}"
                                    )

                                    # Check for JSON-RPC response
                                    if "result" in data:
                                        result = data["result"]

                                        # CRITICAL: Check for MCP error format BEFORE processing content
                                        # Some MCP servers return errors as: {"result": {"content": [{"text": "MCP error -401..."}], "isError": true}}
                                        is_error = result.get("isError", False)
                                        error_message = None

                                        # Extract content from MCP result format
                                        # MCP returns: {"content": [{"type": "text", "text": "..."}], "isError": false}
                                        if (
                                            "content" in result
                                            and len(result["content"]) > 0
                                        ):
                                            content_item = result["content"][0]

                                            # Handle text content
                                            if (
                                                content_item.get("type") == "text"
                                                and "text" in content_item
                                            ):
                                                text_content = content_item["text"]

                                                # Check if text content contains an error message
                                                if "MCP error" in text_content or "Api key not found" in text_content:
                                                    is_error = True
                                                    error_message = text_content
                                                    logger.warning(
                                                        f"MCP server returned error in content: {text_content[:200]}"
                                                    )

                                                # If this is an error response, return it with error key
                                                if is_error:
                                                    return {
                                                        "error": error_message or "MCP server returned an error",
                                                        "server": server.name,
                                                        "tool": tool_name,
                                                        "routed_via": "direct_mcp",
                                                    }

                                                # Try to parse nested JSON (some tools wrap results in JSON strings)
                                                try:
                                                    nested = json.loads(text_content)
                                                    return {
                                                        "result": nested,
                                                        "server": server.name,
                                                        "tool": tool_name,
                                                        "routed_via": "direct_mcp",
                                                    }
                                                except json.JSONDecodeError:
                                                    # Return as plain text
                                                    return {
                                                        "result": text_content,
                                                        "server": server.name,
                                                        "tool": tool_name,
                                                        "routed_via": "direct_mcp",
                                                    }

                                            # Handle other content types
                                            return {
                                                "result": content_item,
                                                "server": server.name,
                                                "tool": tool_name,
                                                "routed_via": "direct_mcp",
                                            }

                                        # If isError flag was set but no content, return error
                                        if is_error:
                                            return {
                                                "error": error_message or "MCP server returned isError=true",
                                                "server": server.name,
                                                "tool": tool_name,
                                                "routed_via": "direct_mcp",
                                            }

                                        # Return raw result if no content field
                                        return {
                                            "result": result,
                                            "server": server.name,
                                            "tool": tool_name,
                                            "routed_via": "direct_mcp",
                                        }

                                    elif "error" in data:
                                        return {
                                            "error": data["error"],
                                            "server": server.name,
                                            "tool": tool_name,
                                        }
                                except json.JSONDecodeError:
                                    continue

                        return {
                            "error": "No valid data in SSE response",
                            "server": server.name,
                            "tool": tool_name,
                        }
                    else:
                        # Regular JSON response
                        result = response.json()
                        # Check for JSON-RPC error response
                        if "error" in result:
                            return {
                                "error": result["error"],
                                "server": server.name,
                                "tool": tool_name,
                            }
                        # Check for MCP result with isError flag
                        if "result" in result:
                            mcp_result = result["result"]
                            # Check if this is an error response from MCP server
                            if isinstance(mcp_result, dict):
                                is_error = mcp_result.get("isError", False)
                                error_msg = None

                                # Check content for error messages
                                if "content" in mcp_result and len(mcp_result["content"]) > 0:
                                    content_item = mcp_result["content"][0]
                                    if isinstance(content_item, dict):
                                        text_content = content_item.get("text", "")
                                        if "MCP error" in text_content or "Api key not found" in text_content:
                                            is_error = True
                                            error_msg = text_content

                                if is_error:
                                    return {
                                        "error": error_msg or "MCP server returned isError=true",
                                        "server": server.name,
                                        "tool": tool_name,
                                        "routed_via": "direct_mcp",
                                    }

                        # Return the result part of JSON-RPC response
                        return result.get("result", result)
                else:
                    return {
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                        "server": server.name,
                        "tool": tool_name,
                    }

        except httpx.HTTPError as e:
            logger.error(
                f"HTTP error calling remote tool {tool_name} on {server.name}: {e}"
            )
            return {"error": str(e), "server": server.name, "tool": tool_name}
        except Exception as e:
            logger.error(f"Unexpected error calling remote tool: {e}")
            return {
                "error": f"Unexpected error: {str(e)}",
                "server": server.name,
                "tool": tool_name,
            }

    async def health_check(self, server_name: str) -> bool:
        """
        Check if MCP server is responsive.

        Args:
            server_name: Name of the server to check

        Returns:
            True if server is healthy, False otherwise
        """
        if server_name not in self.servers:
            return False

        server = self.servers[server_name]

        if server.type == MCPServerType.REMOTE and server.url:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                }
                if server.headers:
                    headers.update(server.headers)

                # Try to ping the server using MCP initialize method
                mcp_request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "ai-inference-gateway",
                            "version": "2.0.0",
                        },
                    },
                }

                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(
                        server.url, json=mcp_request, headers=headers
                    )
                    # Accept any 2xx response as healthy
                    return 200 <= response.status_code < 300
            except Exception as e:
                logger.debug(f"Health check failed for {server_name}: {e}")
                return False
        else:
            # For local servers, check if the process is running
            return await self._check_local_server_health(server)

    # ========================================================================
    # LOCAL MCP SERVER METHODS (stdio-based communication)
    # ========================================================================

    async def _spawn_local_server(self, server: MCPServer) -> bool:
        """
        Spawn a local MCP server subprocess.

        Args:
            server: MCP server configuration

        Returns:
            True if spawn succeeded, False otherwise
        """
        if server.process and not server.process.returncode:
            # Process already running
            return True

        try:
            # Prepare environment variables
            env = None
            if server.environment:
                import os

                env = os.environ.copy()
                # Process environment variables - read from file if needed
                for env_name, env_value in server.environment.items():
                    # Check if this is an API key file reference (ends with _FILE)
                    if (
                        env_name.endswith("_FILE")
                        and isinstance(env_value, str)
                        and env_value.startswith("/")
                    ):
                        try:
                            with open(env_value, "r") as f:
                                file_content = f.read().strip()
                            # Set the base env name (without _FILE) to the file content
                            base_env_name = env_name[:-5]  # Remove _FILE suffix
                            env[base_env_name] = file_content
                            logger.debug(
                                f"Loaded {base_env_name} from file {env_value}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to read {env_name} from {env_value}: {e}"
                            )
                            # Fall back to original value
                            env[env_name] = env_value
                    else:
                        env[env_name] = env_value

            # For module-style commands (using -m), add PYTHONPATH to find ai_inference_gateway
            if server.command and len(server.command) >= 2 and server.command[1] == "-m":
                # Try to find the gateway package in the system closure
                print(f"[DEBUG] Spawning MCP server {server.name}, command: {server.command}")
                try:
                    import sys
                    import os
                    import subprocess

                    # Method 1: Try to find gateway package via Python import system (works in containers)
                    gateway_pkg = None
                    gateway_python = None

                    try:
                        import ai_inference_gateway
                        gateway_pkg_path = os.path.dirname(ai_inference_gateway.__file__)
                        # Get the parent directory (should be the site-packages root)
                        gateway_pkg = os.path.dirname(gateway_pkg_path)
                        print(f"[DEBUG] [MCP {server.name}] Found gateway_pkg via import: {gateway_pkg}")
                    except ImportError:
                        print(f"[DEBUG] [MCP {server.name}] Could not import ai_inference_gateway")

                    # Method 2: Find Python interpreter with mcp installed
                    # Use sys.executable (current Python) or search for mcp in site-packages
                    try:
                        import mcp
                        # Current Python has mcp, use it
                        gateway_python = os.path.dirname(sys.executable)
                        print(f"[DEBUG] [MCP {server.name}] Found gateway_python via sys.executable: {gateway_python}")
                    except ImportError:
                        # Try to find mcp in current Python's site-packages
                        mcp_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "mcp")
                        if os.path.exists(mcp_path):
                            gateway_python = sys.prefix
                            print(f"[DEBUG] [MCP {server.name}] Found gateway_python via sys.prefix: {gateway_python}")

                    print(f"[DEBUG] [MCP {server.name}] Final: gateway_pkg={gateway_pkg}, gateway_python={gateway_python}")
                    logger.info(f"[MCP {server.name}] gateway_pkg={gateway_pkg}, gateway_python={gateway_python}")

                    # Build PYTHONPATH with both paths
                    pythonpath_parts = []
                    if gateway_pkg:
                        pythonpath_parts.append(gateway_pkg)
                    if gateway_python:
                        pythonpath_parts.append(f"{gateway_python}/lib/python3.13/site-packages")

                    # Build the modified command with gateway Python
                    modified_command = list(server.command)
                    if gateway_python and modified_command[0] == "python3":
                        # Use the gateway Python interpreter which has all the dependencies
                        modified_command[0] = f"{gateway_python}/bin/python3"
                        print(f"[DEBUG] [MCP {server.name}] Using gateway Python: {modified_command[0]}")

                    if pythonpath_parts:
                        existing_pythonpath = env.get("PYTHONPATH", "")
                        if existing_pythonpath:
                            pythonpath_parts.append(existing_pythonpath)
                        env["PYTHONPATH"] = ":".join(pythonpath_parts)
                        logger.info(f"[MCP {server.name}] Set PYTHONPATH: {env['PYTHONPATH'][:300]}...")
                        print(f"[DEBUG] PYTHONPATH for {server.name}: {env['PYTHONPATH'][:300]}...")
                    else:
                        logger.warning(f"[MCP {server.name}] No gateway paths found, PYTHONPATH not set")
                        print(f"[DEBUG] No gateway paths found for {server.name}")
                except Exception as e:
                    logger.warning(f"Failed to add PYTHONPATH for {server.name}: {e}")

            # Determine the command to use (possibly modified with gateway Python)
            command_to_use = modified_command if 'modified_command' in locals() else server.command

            # Spawn the subprocess
            server.process = await asyncio.create_subprocess_exec(
                *command_to_use,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Initialize the MCP server
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ai-inference-gateway", "version": "2.0.0"},
                },
            }

            init_response = await self._send_local_request(server, init_request)
            if init_response and "error" not in init_response:
                logger.info(f"Successfully initialized local MCP server {server.name}")
                return True
            else:
                logger.warning(
                    f"Failed to initialize local MCP server {server.name}: {init_response}"
                )
                # Terminate the process if initialization failed
                if server.process:
                    server.process.terminate()
                    server.process = None
                return False

        except Exception as e:
            import traceback
            logger.error(f"Failed to spawn local MCP server {server.name}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Clean up process if it was created
            if server.process:
                try:
                    server.process.terminate()
                except Exception:
                    pass
                server.process = None
            return False

    async def _send_local_request(
        self, server: MCPServer, request: Dict
    ) -> Optional[Dict]:
        """
        Send a JSON-RPC request to a local MCP server via stdio.

        Args:
            server: MCP server configuration
            request: JSON-RPC request object

        Returns:
            JSON-RPC response or None if communication failed
        """
        # Ensure the server is running
        if not server.process or server.process.returncode is not None:
            if not await self._spawn_local_server(server):
                return None

        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            server.process.stdin.write(request_json.encode())
            await server.process.stdin.drain()

            # Read response line by line
            response_line = await asyncio.wait_for(
                server.process.stdout.readline(), timeout=30.0
            )

            if not response_line:
                logger.warning(f"No response from local MCP server {server.name}")
                return None

            response = json.loads(response_line.decode().strip())

            # Check for JSON-RPC error
            if "error" in response:
                logger.warning(
                    f"Local MCP server {server.name} returned error: {response['error']}"
                )

            return response

        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for response from local MCP server {server.name}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Error communicating with local MCP server {server.name}: {e}"
            )
            # Process may have died, mark it as None so we respawn next time
            server.process = None
            return None

    async def _fetch_tools_from_local_server(
        self, server: MCPServer
    ) -> Optional[List[Dict]]:
        """
        Fetch tools from a local MCP server.

        Args:
            server: MCP server configuration

        Returns:
            List of tool definitions or None if fetch fails
        """
        # Ensure the server is running
        if not server.process or server.process.returncode is not None:
            if not await self._spawn_local_server(server):
                return None

        # Request tools list
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}

        response = await self._send_local_request(server, request)
        if not response:
            return None

        if "error" in response:
            logger.warning(
                f"Local MCP server {server.name} returned error: {response['error']}"
            )
            return None

        # Extract tools from response
        if "result" in response and "tools" in response["result"]:
            tools = []
            for tool in response["result"]["tools"]:
                tools.append(
                    {
                        "server": server.name,
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "type": "local",
                        "inputSchema": tool.get("inputSchema", {}),
                    }
                )
            return tools

        logger.warning(f"Unexpected response from local MCP server {server.name}")
        return None

    async def _call_local_tool(
        self, server: MCPServer, tool_name: str, arguments: Dict
    ) -> Dict:
        """
        Call a tool on a local MCP server.

        Args:
            server: MCP server configuration
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool execution result
        """
        # Request tool call
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        response = await self._send_local_request(server, request)
        if not response:
            return {
                "error": f"Failed to communicate with local MCP server {server.name}",
                "server": server.name,
                "tool": tool_name,
            }

        if "error" in response:
            return {
                "error": response["error"],
                "server": server.name,
                "tool": tool_name,
            }

        # Return the result
        if "result" in response:
            # Handle different response formats
            result = response["result"]
            if isinstance(result, list):
                # Format content blocks
                content = []
                for item in result:
                    if item.get("type") == "text":
                        content.append(item.get("text", ""))
                    else:
                        content.append(str(item))
                return {
                    "server": server.name,
                    "tool": tool_name,
                    "result": "\n".join(content),
                }
            elif isinstance(result, dict):
                if result.get("type") == "text":
                    return {
                        "server": server.name,
                        "tool": tool_name,
                        "result": result.get("text", ""),
                    }
                else:
                    return {
                        "server": server.name,
                        "tool": tool_name,
                        "result": json.dumps(result),
                    }
            else:
                return {"server": server.name, "tool": tool_name, "result": str(result)}

        return {
            "error": "Unexpected response from local MCP server",
            "server": server.name,
            "tool": tool_name,
        }

    async def _check_local_server_health(self, server: MCPServer) -> bool:
        """
        Check if a local MCP server process is healthy.

        Args:
            server: MCP server configuration

        Returns:
            True if server is healthy, False otherwise
        """
        if not server.process:
            # Try to spawn the server
            return await self._spawn_local_server(server)

        # Check if process is still running
        if server.process.returncode is not None:
            # Process died, try to respawn
            logger.warning(
                f"Local MCP server {server.name} process died (exit code {server.process.returncode})"
            )
            server.process = None
            return await self._spawn_local_server(server)

        return True

    async def close_local_servers(self):
        """Close all local MCP server processes."""
        for server in self.servers.values():
            if server.type == MCPServerType.LOCAL and server.process:
                try:
                    server.process.terminate()
                    await asyncio.wait_for(server.process.wait(), timeout=5.0)
                    logger.info(f"Closed local MCP server {server.name}")
                except asyncio.TimeoutError:
                    server.process.kill()
                    await server.process.wait()
                    logger.warning(f"Force killed local MCP server {server.name}")
                except Exception as e:
                    logger.error(f"Error closing local MCP server {server.name}: {e}")
                finally:
                    server.process = None


async def create_mcp_broker_from_config(config) -> Optional[MCPBroker]:
    """
    Create MCP broker from gateway configuration.

    Args:
        config: Gateway configuration

    Returns:
        MCPBroker instance or None if MCP is disabled
    """
    import os
    import json

    # Check if MCP broker is enabled via config
    mcp_enabled = False
    if hasattr(config, "middleware") and hasattr(config.middleware, "mcp") and hasattr(config.middleware.mcp, "enabled"):
        mcp_enabled = config.middleware.mcp.enabled

    # Also check environment variable for Nix-based configuration
    if os.getenv("MCP_ENABLED"):
        mcp_enabled = os.getenv("MCP_ENABLED").lower() == "true"

    if not mcp_enabled:
        logger.info("MCP broker disabled in configuration")
        return None

    servers = []

    # Try to load from environment variable (Nix format)
    mcp_servers_json = os.getenv("MCP_SERVERS")
    if mcp_servers_json:
        try:
            mcp_servers_dict = json.loads(mcp_servers_json)
            logger.info(
                f"Loading MCP servers from environment: {list(mcp_servers_dict.keys())}"
            )

            for server_name, server_config in mcp_servers_dict.items():
                # Check if server is enabled
                if not server_config.get("enabled", True):
                    continue

                # Determine server type (defaults to remote for backward compatibility)
                server_type_str = server_config.get("type", "remote")
                server_type = (
                    MCPServerType.LOCAL
                    if server_type_str == "local"
                    else MCPServerType.REMOTE
                )

                if server_type == MCPServerType.LOCAL:
                    # Local server configuration
                    command = server_config.get("command")
                    environment = server_config.get("environment", {})

                    if not command:
                        logger.warning(
                            f"Local MCP server {server_name} missing command, skipping"
                        )
                        continue

                    server = MCPServer(
                        name=server_name,
                        type=MCPServerType.LOCAL,
                        command=command,
                        environment=environment,
                        url=None,
                        headers=None,
                    )
                    servers.append(server)
                    logger.info(
                        f"Added MCP server: {server.name} (local, command={' '.join(command)})"
                    )
                else:
                    # Remote server configuration
                    url = server_config.get("url")
                    if not url:
                        logger.warning(
                            f"Remote MCP server {server_name} missing url, skipping"
                        )
                        continue

                    # Read API key from file if specified in headers
                    headers = dict(server_config.get("headers", {}))
                    for header_name, header_value in list(headers.items()):
                        if (
                            isinstance(header_value, str)
                            and "/run/" in header_value
                            and "-key" in header_value
                        ):
                            try:
                                # Extract file path from "Bearer /path/to/key" or just "/path/to/key"
                                if header_value.startswith("Bearer "):
                                    file_path = header_value.split(" ", 1)[1].strip()
                                    use_bearer = True
                                else:
                                    file_path = header_value
                                    use_bearer = False

                                with open(file_path, "r") as f:
                                    api_key = f.read().strip()
                                    if use_bearer:
                                        headers[header_name] = f"Bearer {api_key}"
                                    else:
                                        headers[header_name] = api_key
                                logger.info(
                                    f"Loaded API key from {file_path} for {server_name}/{header_name}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"Failed to read API key from {header_value}: {e}"
                                )

                    server = MCPServer(
                        name=server_name,
                        type=MCPServerType.REMOTE,
                        url=url,
                        headers=headers,
                        command=None,
                        environment=None,
                    )
                    servers.append(server)
                    logger.info(f"Added MCP server: {server.name} (remote)")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse MCP_SERVERS environment variable: {e}")
            return None
    elif hasattr(config, "middleware") and hasattr(config.middleware, "mcp") and hasattr(config.middleware.mcp, "servers"):
        # Fallback to config object (Python-based configuration)
        for server_config in config.middleware.mcp.servers:
            server_type = (
                MCPServerType.LOCAL
                if server_config.type == "local"
                else MCPServerType.REMOTE
            )

            server = MCPServer(
                name=server_config.name,
                type=server_type,
                command=server_config.command,
                url=server_config.url,
                headers=server_config.headers,
                environment=server_config.environment,
            )
            servers.append(server)
            logger.info(f"Added MCP server: {server.name} ({server_type.value})")

    if not servers:
        logger.warning("MCP broker enabled but no servers configured")
        return None

    return MCPBroker(servers=servers)
