"""
Structured Tool Calling Module
==============================

Native tool/function calling for providers:
- OpenAI: tools parameter with JSON Schema
- Anthropic: tools with input_schema
- Gemini: function_declarations
- Groq: OpenAI-compatible tools

Usage:
    from gaap.providers.tool_calling import ToolRegistry, ToolDefinition

    registry = ToolRegistry()
    registry.register(my_tool)
    tools = registry.to_openai_tools()
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("gaap.providers.tool_calling")


class ToolType(Enum):
    """Tool types."""

    FUNCTION = "function"
    CODE_INTERPRETER = "code_interpreter"
    RETRIEVAL = "retrieval"


@dataclass
class ParameterSchema:
    """JSON Schema for a parameter."""

    type: str = "string"
    description: str = ""
    enum: list[str] | None = None
    default: Any = None
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            result["enum"] = self.enum
        if self.default is not None:
            result["default"] = self.default
        return result


@dataclass
class ToolDefinition:
    """Definition of a callable tool."""

    name: str
    description: str
    parameters: dict[str, ParameterSchema] = field(default_factory=dict)
    type: ToolType = ToolType.FUNCTION
    handler: Callable | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI tool schema."""
        properties = {}
        required = []

        for name, param in self.parameters.items():
            properties[name] = param.to_dict()
            if param.required:
                required.append(name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool schema."""
        properties = {}
        required = []

        for name, param in self.parameters.items():
            properties[name] = param.to_dict()
            if param.required:
                required.append(name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def to_gemini_schema(self) -> dict[str, Any]:
        """Convert to Gemini function declaration."""
        properties = {}
        required = []

        for name, param in self.parameters.items():
            properties[name] = param.to_dict()
            if param.required:
                required.append(name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolCall:
    """A tool call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    type: ToolType = ToolType.FUNCTION

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "ToolCall":
        """Parse from OpenAI format."""
        function = data.get("function", {})
        return cls(
            id=data.get("id", ""),
            name=function.get("name", ""),
            arguments=json.loads(function.get("arguments", "{}")),
            type=ToolType(data.get("type", "function")),
        )

    @classmethod
    def from_anthropic(cls, data: dict[str, Any]) -> "ToolCall":
        """Parse from Anthropic format."""
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("input", {}),
            type=ToolType.FUNCTION,
        )

    @classmethod
    def from_gemini(cls, data: dict[str, Any]) -> "ToolCall":
        """Parse from Gemini format."""
        args = data.get("args", {})
        if isinstance(args, str):
            args = json.loads(args)

        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=args,
            type=ToolType.FUNCTION,
        )


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_call_id: str
    name: str
    result: Any
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_message(self) -> dict[str, Any]:
        """Convert to OpenAI tool message."""
        content = str(self.result) if not isinstance(self.result, str) else self.result
        return {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": content,
        }

    def to_anthropic_message(self) -> dict[str, Any]:
        """Convert to Anthropic tool result."""
        content = str(self.result) if not isinstance(self.result, str) else self.result
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": self.tool_call_id,
                    "content": content,
                }
            ],
        }


class ToolRegistry:
    _tools: dict[str, ToolDefinition]

    def __init__(self) -> None:
        self._tools = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tools."""
        return list(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Export all tools in OpenAI format."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """Export all tools in Anthropic format."""
        return [tool.to_anthropic_schema() for tool in self._tools.values()]

    def to_gemini_tools(self) -> list[dict[str, Any]]:
        """Export all tools in Gemini format."""
        return [tool.to_gemini_schema() for tool in self._tools.values()]

    def to_provider_format(self, provider: str) -> list[dict[str, Any]]:
        """Export tools in provider-specific format."""
        provider_lower = provider.lower()

        if provider_lower in ("openai", "groq", "deepseek"):
            return self.to_openai_tools()
        elif provider_lower == "anthropic":
            return self.to_anthropic_tools()
        elif provider_lower in ("gemini", "google"):
            return self.to_gemini_tools()
        else:
            return self.to_openai_tools()

    async def execute(
        self,
        tool_call: ToolCall,
        context: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute
            context: Optional execution context

        Returns:
            ToolResult with execution result
        """
        tool = self._tools.get(tool_call.name)

        if not tool:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error: Tool '{tool_call.name}' not found",
                is_error=True,
            )

        if tool.handler is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error: Tool '{tool_call.name}' has no handler",
                is_error=True,
            )

        try:
            import asyncio

            ctx = context or {}
            ctx.update(tool_call.arguments)

            result = tool.handler(**ctx)

            if asyncio.iscoroutine(result):
                result = await result

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=result,
            )

        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                result=f"Error: {str(e)}",
                is_error=True,
            )

    def validate_arguments(
        self,
        tool_call: ToolCall,
    ) -> tuple[bool, list[str]]:
        """
        Validate tool arguments against schema.

        Args:
            tool_call: The tool call to validate

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        tool = self._tools.get(tool_call.name)

        if not tool:
            return False, [f"Tool '{tool_call.name}' not found"]

        errors = []

        for name, param in tool.parameters.items():
            if param.required and name not in tool_call.arguments:
                errors.append(f"Missing required parameter: {name}")

        for name, value in tool_call.arguments.items():
            if name not in tool.parameters:
                continue

            param = tool.parameters[name]
            expected_type = param.type

            type_mapping = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict,
            }

            expected = type_mapping.get(expected_type)
            if expected and not isinstance(value, expected):  # type: ignore[arg-type]
                if expected_type == "integer" and isinstance(value, float) and value.is_integer():
                    continue
                if expected_type == "number" and isinstance(value, (int, float)):
                    continue
                errors.append(
                    f"Parameter '{name}' expected type {expected_type}, got {type(value).__name__}"
                )

            if param.enum and value not in param.enum:
                errors.append(f"Parameter '{name}' must be one of {param.enum}")

        return len(errors) == 0, errors


def create_tool_from_function(
    func: Callable,
    name: str | None = None,
    description: str | None = None,
) -> ToolDefinition:
    """
    Create a ToolDefinition from a function.

    Uses function signature and docstring for schema.

    Args:
        func: The function to convert
        name: Optional tool name (uses function name if not provided)
        description: Optional description (uses docstring if not provided)

    Returns:
        ToolDefinition
    """
    import inspect

    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or ""

    tool_name = name or func.__name__
    tool_description = description or (doc.split("\n")[0] if doc else "")

    parameters = {}
    for param_name, param in sig.parameters.items():
        param_type = "string"

        if param.annotation != inspect.Parameter.empty:
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                list: "array",
                dict: "object",
            }
            param_type = type_map.get(param.annotation, "string")

        parameters[param_name] = ParameterSchema(
            type=param_type,
            required=param.default == inspect.Parameter.empty,
        )

    return ToolDefinition(
        name=tool_name,
        description=tool_description,
        parameters=parameters,
        handler=func,
    )


def tool(
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable], ToolDefinition]:
    """
    Decorator to create a tool from a function.

    Usage:
        >>> @tool(description="Get weather for a location")
        ... def get_weather(location: str) -> str:
        ...     return f"Weather in {location}: Sunny"
    """

    def decorator(func: Callable) -> ToolDefinition:
        return create_tool_from_function(func, name=name, description=description)

    return decorator
