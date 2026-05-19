"""MCP Servers for AI Inference Gateway."""

from mcp_servers.searxng_server import main as searxng_main
from mcp_servers.maplespike_server import main as maplespike_main

__all__ = ["searxng_main", "maplespike_main"]
