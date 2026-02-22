"""
GAAP Tools Module
=================

Tool integration and synthesis capabilities:

Native Tools:
    - NativeToolCaller: Built-in tools for file I/O, commands
    - ToolSchema: JSON schema definitions
    - ToolCall/ToolResult: Request/response types

MCP Integration:
    - MCPClient: Model Context Protocol client
    - MCPToolRegistry: Tool discovery and management
    - MCPServerConfig: Server configuration

Tool Synthesis (JIT):
    - ToolSynthesizer: Generate tools on-demand
    - SynthesizedTool: Hot-loaded custom tools
    - Axiom validation for generated code

Usage:
    from gaap.tools import NativeToolCaller, MCPClient

    caller = NativeToolCaller()
    result = caller.execute_call(ToolCall(...))
"""

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .mcp_client import MCPClient, MCPTool, MCPToolRegistry
    from .synthesizer import SynthesizedTool, ToolSynthesizer
else:

    class MCPTool(Protocol):
        name: str
        description: str
        input_schema: dict[str, Any]

    class MCPClient(Protocol):
        async def call_tool(self, name: str, args: dict) -> Any: ...
        async def list_tools(self) -> list[MCPTool]: ...

    class MCPToolRegistry(Protocol):
        async def get_tool(self, name: str) -> MCPTool | None: ...

    class SynthesizedTool(Protocol):
        name: str
        code: str

    class ToolSynthesizer(Protocol):
        async def synthesize(self, spec: str) -> SynthesizedTool: ...


try:
    from .mcp_client import MCPClient, MCPTool, MCPToolRegistry

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

from .native_caller import (
    BUILTIN_TOOLS,
    NativeToolCaller,
    ToolCall,
    ToolResult,
    ToolSchema,
    create_tool_caller,
)

try:
    from .synthesizer import SynthesizedTool, ToolSynthesizer
except ImportError:
    pass

__all__ = [
    "MCPClient",
    "MCPTool",
    "MCPToolRegistry",
    "MCP_AVAILABLE",
    "NativeToolCaller",
    "ToolSchema",
    "ToolCall",
    "ToolResult",
    "BUILTIN_TOOLS",
    "create_tool_caller",
    "ToolSynthesizer",
    "SynthesizedTool",
]
