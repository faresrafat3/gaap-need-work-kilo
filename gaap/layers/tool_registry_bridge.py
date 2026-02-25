"""
Tool Registry Bridge
====================

Bridges the new ToolRegistry (from providers.tool_calling) with Layer3 execution.
Allows Layer3 to use structured tool calling with JSON Schema validation.

Usage:
    from gaap.layers.tool_registry_bridge import ToolRegistryBridge

    bridge = ToolRegistryBridge()
    bridge.register_native_tools(native_caller)

    # Get tools for LLM
    tools = bridge.get_tools_for_provider("openai")

    # Execute tool call
    result = await bridge.execute(tool_call)
"""

import asyncio
import logging
from typing import Any

from gaap.providers.tool_calling import (
    ToolRegistry,
    ToolDefinition,
    ToolCall,
    ToolResult,
    ParameterSchema,
    create_tool_from_function,
)

logger = logging.getLogger("gaap.layers.tool_bridge")


class ToolRegistryBridge:
    """
    Bridges ToolRegistry with Layer3 execution.

    Features:
    - Converts NativeToolCaller tools to ToolDefinition format
    - Provides provider-specific tool schemas
    - Validates tool arguments
    - Executes tools with proper error handling
    """

    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self._native_caller = None
        self._mcp_registry = None
        self._logger = logger

    def register_native_tools(self, native_caller: Any) -> None:
        """Register tools from NativeToolCaller."""
        self._native_caller = native_caller

        if hasattr(native_caller, "_tools"):
            for name, tool in native_caller._tools.items():
                self._convert_and_register(name, tool)

    def register_mcp_tools(self, mcp_registry: Any) -> None:
        """Register tools from MCP registry."""
        self._mcp_registry = mcp_registry

    def _convert_and_register(self, name: str, tool: Any) -> None:
        """Convert native tool to ToolDefinition and register."""
        description = getattr(tool, "description", f"Tool: {name}")

        parameters = {}
        if hasattr(tool, "parameters"):
            for param_name, param_info in tool.parameters.items():
                if isinstance(param_info, dict):
                    parameters[param_name] = ParameterSchema(
                        type=param_info.get("type", "string"),
                        description=param_info.get("description", ""),
                        required=param_info.get("required", True),
                    )
                else:
                    parameters[param_name] = ParameterSchema(
                        type="string",
                        description=str(param_info),
                    )

        handler = None
        if hasattr(tool, "func"):
            handler = tool.func
        elif hasattr(tool, "execute"):
            handler = tool.execute
        elif callable(tool):
            handler = tool

        tool_def = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

        self._registry.register(tool_def)

    def register_function(
        self,
        func: Any,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Register a Python function as a tool."""
        tool_def = create_tool_from_function(func, name=name, description=description)
        self._registry.register(tool_def)

    def get_tools_for_provider(self, provider: str) -> list[dict[str, Any]]:
        """Get tools in provider-specific format."""
        return self._registry.to_provider_format(provider)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible tool schemas."""
        return self._registry.to_openai_tools()

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            ToolResult with execution result
        """
        is_valid, errors = self._registry.validate_arguments(tool_call)
        if not is_valid:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Validation error: {', '.join(errors)}",
                is_error=True,
            )

        if self._native_caller and tool_call.name in self._native_caller._tools:
            try:
                from gaap.tools import ToolCall as NativeToolCall

                native_call = NativeToolCall(name=tool_call.name, arguments=tool_call.arguments)
                result = self._native_caller.execute_call(native_call)
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=result,
                )
            except Exception as e:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    result=f"Execution error: {e}",
                    is_error=True,
                )

        return await self._registry.execute(tool_call)

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return self._registry.list_tools()

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._registry.get(name)


def create_tool_bridge(
    native_caller: Any = None,
    mcp_registry: Any = None,
) -> ToolRegistryBridge:
    """Create a configured ToolRegistryBridge."""
    bridge = ToolRegistryBridge()

    if native_caller:
        bridge.register_native_tools(native_caller)

    if mcp_registry:
        bridge.register_mcp_tools(mcp_registry)

    return bridge
