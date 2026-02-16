# GAAP Tools

try:
    from .mcp_client import MCPClient, MCPTool, MCPToolRegistry

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPClient = None  # type: ignore
    MCPTool = None  # type: ignore
    MCPToolRegistry = None  # type: ignore

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPToolRegistry",
    "MCP_AVAILABLE",
]
