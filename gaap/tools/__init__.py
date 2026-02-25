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
    from .mcp_client import MCPClient, MCPResource, MCPServerConfig, MCPTool, MCPToolRegistry
    from .synthesizer import SynthesizedTool, ToolSynthesizer
else:

    class MCPTool(Protocol):
        name: str
        description: str
        input_schema: dict[str, Any]

    class MCPResource(Protocol):
        uri: str
        name: str

    class MCPClient(Protocol):
        async def call_tool(self, name: str, args: dict) -> Any: ...
        async def list_tools(self) -> list[MCPTool]: ...

    class MCPServerConfig(Protocol):
        name: str
        transport: str

    class MCPToolRegistry(Protocol):
        async def get_tool(self, name: str) -> MCPTool | None: ...

    class SynthesizedTool(Protocol):
        name: str
        code: str

    class ToolSynthesizer(Protocol):
        async def synthesize(self, spec: str) -> SynthesizedTool: ...


try:
    from .mcp_client import MCPClient, MCPResource, MCPServerConfig, MCPTool, MCPToolRegistry

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

try:
    from .skill_cache import SkillCache, SkillCacheStats, SkillMetadata, get_skill_cache
except ImportError:
    pass

try:
    from .library_discoverer import LibraryDiscoverer, LibraryInfo, SearchResult
except ImportError:
    pass

try:
    from .code_synthesizer import (
        CodeSynthesizer,
        CodeSynthesizerConfig,
        SynthesisRequest,
        SynthesisResult,
    )
except ImportError:
    pass

try:
    from .tool_registry import (
        DynamicToolWatcher,
        MCPToolAdapter,
        SecurityValidator,
        ToolRegistry,
        create_tool_registry,
    )
except ImportError:
    pass

try:
    from .interpreter_tool import (
        ExecutionResult,
        ExecutionStatus,
        InterpreterTool,
        TestCase,
        TestResult,
        create_interpreter,
    )
except ImportError:
    pass

try:
    from .search_tool import (
        APICategory,
        APISearchTool,
        APIInfo,
        DeprecationStatus,
        EndpointInfo,
        create_api_search_tool,
    )
except ImportError:
    pass

__all__ = [
    "APICategory",
    "APIInfo",
    "APISearchTool",
    "BUILTIN_TOOLS",
    "CodeSynthesizer",
    "CodeSynthesizerConfig",
    "DeprecationStatus",
    "DynamicToolWatcher",
    "EndpointInfo",
    "ExecutionResult",
    "ExecutionStatus",
    "InterpreterTool",
    "LibraryDiscoverer",
    "LibraryInfo",
    "MCPClient",
    "MCPResource",
    "MCPServerConfig",
    "MCPTool",
    "MCPToolAdapter",
    "MCPToolRegistry",
    "MCP_AVAILABLE",
    "NativeToolCaller",
    "SearchResult",
    "SecurityValidator",
    "SkillCache",
    "SkillCacheStats",
    "SkillMetadata",
    "SynthesizedTool",
    "SynthesisRequest",
    "SynthesisResult",
    "TestCase",
    "TestResult",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "ToolSchema",
    "ToolSynthesizer",
    "create_api_search_tool",
    "create_tool_caller",
    "create_tool_registry",
    "create_interpreter",
    "get_skill_cache",
]
