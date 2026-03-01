"""
Native Function Caller - Structured Tool Calling
================================================

Evolution 2026: Native function calling with fallback support.

Priority:
1. Provider's native tools API (OpenAI/Gemini/Anthropic)
2. Structured JSON Schema output
3. Legacy regex (fallback only)

Key Features:
- Multi-provider support
- JSON Schema generation
- Structured output parsing
- Fallback mechanisms
"""

import json
import re
import time
from typing import Any, Literal

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message, MessageRole
from gaap.layers.execution_schema import (
    StructuredOutput,
    StructuredToolCall,
    StructuredToolRegistry,
    ToolCallStatus,
    ToolDefinition,
    ToolResult,
)
from gaap.layers.layer3_config import Layer3Config

logger = get_logger("gaap.layer3.native_caller")


class NativeFunctionCaller:
    """
    Native function calling with intelligent fallback.

    Supports:
    - OpenAI function calling API
    - Anthropic tool use
    - Gemini function declarations
    - Structured JSON output fallback
    - Legacy regex parsing (last resort)
    """

    PROVIDER_SUPPORT = {
        "openai": "native",
        "anthropic": "native",
        "gemini": "native",
        "kimi": "native",
        "mistral": "native",
        "ollama": "structured",
        "grok": "structured",
        "default": "legacy",
    }

    def __init__(
        self,
        config: Layer3Config | None = None,
        tool_registry: StructuredToolRegistry | None = None,
    ):
        self._config = config or Layer3Config()
        self._registry = tool_registry or StructuredToolRegistry()
        self._logger = logger

        self._native_calls = 0
        self._structured_calls = 0
        self._legacy_calls = 0

    def register_tool(self, tool: ToolDefinition) -> None:
        self._registry.register(tool)

    def get_tools_schema(
        self,
        provider: str,
        tools: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tools schema in provider-specific format"""

        format_type = self._get_format_for_provider(provider)

        if tools:
            schemas = []
            for tool_name in tools:
                tool = self._registry.get(tool_name)
                if tool:
                    schemas.append(self._format_tool(tool, format_type))
            return schemas
        else:
            return self._registry.get_all_schemas(format_type)

    def _get_format_for_provider(self, provider: str) -> Literal["openai", "anthropic", "gemini"]:
        provider_lower = provider.lower()

        if provider_lower in ("openai", "kimi", "deepseek"):
            return "openai"
        elif provider_lower == "anthropic":
            return "anthropic"
        elif provider_lower in ("gemini", "google"):
            return "gemini"
        else:
            return "openai"

    def _format_tool(
        self,
        tool: ToolDefinition,
        format_type: Literal["openai", "anthropic", "gemini"],
    ) -> dict[str, Any]:
        if format_type == "openai":
            return tool.to_openai_schema()
        elif format_type == "anthropic":
            return tool.to_anthropic_schema()
        else:
            return tool.to_gemini_schema()

    async def call_tool(
        self,
        provider: Any,
        tool_call: StructuredToolCall,
        executor: Any = None,
    ) -> ToolResult:
        """Execute a tool call"""

        start_time = time.time()
        tool_call.status = ToolCallStatus.EXECUTING

        try:
            if executor and hasattr(executor, "execute_tool"):
                output = await executor.execute_tool(
                    tool_name=tool_call.tool_name,
                    args=tool_call.arguments,
                )
            elif hasattr(provider, "execute_tool"):
                output = await provider.execute_tool(
                    tool_name=tool_call.tool_name,
                    args=tool_call.arguments,
                )
            else:
                output = await self._default_tool_executor(tool_call)

            tool_call.status = ToolCallStatus.SUCCESS
            tool_call.result = output

            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output=output,
                success=True,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            tool_call.status = ToolCallStatus.FAILED
            tool_call.error = str(e)

            return ToolResult(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                output="",
                success=False,
                error=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _default_tool_executor(self, tool_call: StructuredToolCall) -> str:
        """Default tool executor for basic operations"""

        tool_name = tool_call.tool_name.lower()
        args = tool_call.arguments

        if tool_name in ("read_file", "read"):
            path = args.get("path", args.get("file_path", ""))
            if path:
                try:
                    with open(path, "r") as f:
                        return f.read()[:10000]
                except Exception as e:
                    return f"Error reading file: {e}"
            return "Error: No path specified"

        elif tool_name in ("write_file", "write"):
            path = args.get("path", args.get("file_path", ""))
            content = args.get("content", "")
            if path:
                try:
                    with open(path, "w") as f:
                        f.write(content)
                    return f"Successfully wrote {len(content)} characters to {path}"
                except Exception as e:
                    return f"Error writing file: {e}"
            return "Error: No path specified"

        elif tool_name in ("execute", "run_code", "python"):
            code = args.get("code", args.get("script", ""))
            if code:
                try:
                    from gaap.security.sandbox import get_sandbox

                    sandbox = get_sandbox(use_docker=True)
                    result = await sandbox.execute(code, language="python")
                    return f"STDOUT:\n{result.output}\nSTDERR:\n{result.error}"
                except Exception as e:
                    return f"Error executing code: {e}"
            return "Error: No code specified"

        else:
            return (
                f"Unknown tool: {tool_name}. Available tools: {list(self._registry.tools.keys())}"
            )

    async def execute_with_tools(
        self,
        provider: Any,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str = "",
        max_iterations: int = 10,
        tool_executor: Any = None,
    ) -> StructuredOutput:
        """
        Execute with tool calling loop.

        Handles the full tool calling loop until completion.
        """

        provider_name = getattr(provider, "provider_name", "default")
        support_level = self.PROVIDER_SUPPORT.get(provider_name, "legacy")

        iteration = 0
        all_tool_calls: list[StructuredToolCall] = []
        total_tokens = 0
        current_messages = messages.copy()

        while iteration < max_iterations:
            iteration += 1

            if support_level == "native":
                output = await self._execute_native(
                    provider, current_messages, tools, model, provider_name
                )
                self._native_calls += 1
            elif support_level == "structured":
                output = await self._execute_structured(provider, current_messages, tools, model)
                self._structured_calls += 1
            else:
                output = await self._execute_legacy(provider, current_messages, tools, model)
                self._legacy_calls += 1

            total_tokens += output.tokens_used

            if not output.has_tool_calls():
                output.tool_calls = all_tool_calls
                output.tokens_used = total_tokens
                return output

            for tool_call in output.tool_calls:
                result = await self.call_tool(provider, tool_call, tool_executor)
                all_tool_calls.append(tool_call)

                current_messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=output.content,
                        tool_calls=[tc.to_openai_format() for tc in output.tool_calls],
                    )
                )

                current_messages.append(
                    Message(
                        role=MessageRole.TOOL,
                        content=result.output if result.success else f"Error: {result.error}",
                        tool_call_id=result.call_id,
                    )
                )

        return StructuredOutput(
            content="Max iterations reached",
            tool_calls=all_tool_calls,
            tokens_used=total_tokens,
            model=model,
        )

    async def _execute_native(
        self,
        provider: Any,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
        provider_name: str,
    ) -> StructuredOutput:
        """Execute using provider's native tool calling"""

        format_type = self._get_format_for_provider(provider_name)
        tools_schema = [t.to_openai_schema() for t in tools]

        try:
            if provider_name in ("openai", "kimi", "deepseek"):
                response = await provider.chat_completion(
                    messages=messages,
                    model=model,
                    tools=tools_schema,
                    tool_choice="auto",
                )
                return StructuredOutput.from_openai_response(response, model, provider_name)

            elif provider_name == "anthropic":
                anthropic_tools = [t.to_anthropic_schema() for t in tools]
                response = await provider.chat_completion(
                    messages=messages,
                    model=model,
                    tools=anthropic_tools,
                )
                return StructuredOutput.from_anthropic_response(response, model, provider_name)

            elif provider_name in ("gemini", "google"):
                response = await provider.chat_completion(
                    messages=messages,
                    model=model,
                    tools=tools_schema,
                )
                return StructuredOutput.from_openai_response(response, model, provider_name)

            else:
                return await self._execute_structured(provider, messages, tools, model)

        except Exception as e:
            self._logger.warning(f"Native tool calling failed: {e}, falling back to structured")
            return await self._execute_structured(provider, messages, tools, model)

    async def _execute_structured(
        self,
        provider: Any,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
    ) -> StructuredOutput:
        """Execute using structured JSON output"""

        schema = self._build_tool_selection_schema(tools)

        system_prompt = self._build_structured_system_prompt(tools)

        enhanced_messages = [Message(role=MessageRole.SYSTEM, content=system_prompt)]
        enhanced_messages.extend(messages)

        try:
            response = await provider.chat_completion(
                messages=enhanced_messages,
                model=model,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content if response.choices else ""

            return self._parse_structured_response(content, response, model)

        except Exception as e:
            self._logger.warning(f"Structured output failed: {e}, falling back to legacy")
            return await self._execute_legacy(provider, messages, tools, model)

    def _build_tool_selection_schema(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        tool_names = [t.name for t in tools]

        return {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "Your text response to the user",
                },
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "enum": tool_names,
                            },
                            "arguments": {
                                "type": "object",
                            },
                        },
                        "required": ["tool_name", "arguments"],
                    },
                },
            },
            "required": ["response"],
        }

    def _build_structured_system_prompt(self, tools: list[ToolDefinition]) -> str:
        tools_desc = []
        for tool in tools:
            params_desc = []
            for param in tool.parameters:
                params_desc.append(f"  - {param.name} ({param.type}): {param.description}")
            tools_desc.append(f"- {tool.name}: {tool.description}\n" + "\n".join(params_desc))

        return f"""You are a helpful assistant with access to tools.

AVAILABLE TOOLS:
{chr(10).join(tools_desc)}

RESPONSE FORMAT:
Respond with a JSON object with this structure:
{{
  "response": "Your text response explaining what you're doing",
  "tool_calls": [
    {{
      "tool_name": "name_of_tool",
      "arguments": {{"arg1": "value1", "arg2": "value2"}}
    }}
  ]
}}

If you don't need to call any tools, omit the "tool_calls" field.
Only include tool calls that are actually needed.
"""

    def _parse_structured_response(
        self,
        content: str,
        response: Any,
        model: str,
    ) -> StructuredOutput:
        """Parse structured JSON response"""

        tool_calls: list[StructuredToolCall] = []
        text_content = content

        try:
            data = json.loads(content)

            text_content = data.get("response", content)

            for tc_data in data.get("tool_calls", []):
                tool_calls.append(
                    StructuredToolCall(
                        call_id=f"call_{len(tool_calls)}_{int(time.time())}",
                        tool_name=tc_data.get("tool_name", ""),
                        arguments=tc_data.get("arguments", {}),
                    )
                )

        except json.JSONDecodeError:
            pass

        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens

        return StructuredOutput(
            content=text_content,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            model=model,
            provider="structured",
        )

    async def _execute_legacy(
        self,
        provider: Any,
        messages: list[Message],
        tools: list[ToolDefinition],
        model: str,
    ) -> StructuredOutput:
        """Execute using legacy regex parsing"""

        tools_desc = []
        for tool in tools:
            params = ", ".join(p.name for p in tool.parameters if p.required)
            tools_desc.append(f"- {tool.name}({params}): {tool.description}")

        tool_instructions = f"""
AVAILABLE TOOLS (call with: CALL: tool_name(param1='value1', param2='value2')):
{chr(10).join(tools_desc)}
"""

        enhanced_messages = list(messages)
        enhanced_messages.insert(0, Message(role=MessageRole.SYSTEM, content=tool_instructions))

        response = await provider.chat_completion(
            messages=enhanced_messages,
            model=model,
        )

        content = response.choices[0].message.content if response.choices else ""

        tool_calls = self._parse_legacy_tool_calls(content)

        tokens_used = 0
        if hasattr(response, "usage") and response.usage:
            tokens_used = response.usage.total_tokens

        return StructuredOutput(
            content=content,
            tool_calls=tool_calls,
            tokens_used=tokens_used,
            model=model,
            provider="legacy",
        )

    def _parse_legacy_tool_calls(self, content: str) -> list[StructuredToolCall]:
        """Parse tool calls from legacy format: CALL: tool_name(param='value')"""

        tool_calls = []

        pattern = r"CALL:\s*(\w+)\((.*?)\)"
        matches = re.finditer(pattern, content, re.DOTALL)

        for i, match in enumerate(matches):
            tool_name = match.group(1)
            args_str = match.group(2)

            args = self._parse_legacy_args(args_str)

            tool_calls.append(
                StructuredToolCall(
                    call_id=f"legacy_{i}_{int(time.time())}",
                    tool_name=tool_name,
                    arguments=args,
                )
            )

        return tool_calls

    def _parse_legacy_args(self, args_str: str) -> dict[str, str]:
        """Parse arguments from legacy format: param1='value1', param2='value2'"""

        result: dict[str, str] = {}
        i = 0

        while i < len(args_str):
            if args_str[i].isalpha() or args_str[i] == "_":
                key_start = i
                while i < len(args_str) and (args_str[i].isalnum() or args_str[i] == "_"):
                    i += 1
                key = args_str[key_start:i]

                while i < len(args_str) and args_str[i] in " \t\n":
                    i += 1

                if i < len(args_str) and args_str[i] == "=":
                    i += 1

                    while i < len(args_str) and args_str[i] in " \t\n":
                        i += 1

                    if i < len(args_str) and args_str[i] in "\"'":
                        quote = args_str[i]
                        i += 1
                        value_start = i

                        while i < len(args_str):
                            if args_str[i] == "\\" and i + 1 < len(args_str):
                                i += 2
                            elif args_str[i] == quote:
                                break
                            else:
                                i += 1

                        value = args_str[value_start:i]
                        if i < len(args_str):
                            i += 1

                        result[key] = value
            else:
                i += 1

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get caller statistics"""

        total = self._native_calls + self._structured_calls + self._legacy_calls

        return {
            "total_calls": total,
            "native_calls": self._native_calls,
            "structured_calls": self._structured_calls,
            "legacy_calls": self._legacy_calls,
            "registered_tools": len(self._registry.tools),
        }


def create_native_caller(
    config: Layer3Config | None = None,
) -> NativeFunctionCaller:
    """Factory function to create NativeFunctionCaller"""

    return NativeFunctionCaller(config=config)
