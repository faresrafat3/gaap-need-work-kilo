"""
Execution Schema - Structured Output Definitions
================================================

Evolution 2026: Structured schemas for tool calls and outputs.

Key Features:
- Structured tool calls (no regex parsing)
- JSON schema definitions
- Execution plans
- Tool result schemas

These schemas enable native function calling with providers
that support it, and structured JSON output for those that don't.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal
import json


class FinishReason(Enum):
    """Why the LLM finished generating"""

    STOP = auto()
    TOOL_CALL = auto()
    MAX_TOKENS = auto()
    ERROR = auto()
    TIMEOUT = auto()


class ToolCallStatus(Enum):
    """Status of a tool call"""

    PENDING = auto()
    EXECUTING = auto()
    SUCCESS = auto()
    FAILED = auto()
    TIMEOUT = auto()


@dataclass
class ToolParameter:
    """Parameter definition for a tool"""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum:
            schema["enum"] = self.enum
        if self.default is not None:
            schema["default"] = self.default
        return schema


@dataclass
class ToolDefinition:
    """Complete tool definition with JSON schema"""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)

    def to_openai_schema(self) -> dict[str, Any]:
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

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
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

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
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)

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
class StructuredToolCall:
    """Structured tool call (no regex parsing needed)"""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]

    status: ToolCallStatus = ToolCallStatus.PENDING
    result: str | None = None
    error: str | None = None
    execution_time_ms: float = 0.0

    def __post_init__(self) -> None:
        if not self.call_id:
            self.call_id = f"call_{uuid.uuid4().hex[:8]}"

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.tool_name,
                "arguments": json.dumps(self.arguments),
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "type": "tool_use",
            "id": self.call_id,
            "name": self.tool_name,
            "input": self.arguments,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.name,
            "result": self.result,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredToolCall":
        return cls(
            call_id=data.get("call_id", ""),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            status=ToolCallStatus[data.get("status", "PENDING")],
            result=data.get("result"),
            error=data.get("error"),
            execution_time_ms=data.get("execution_time_ms", 0.0),
        )

    @classmethod
    def from_openai(cls, data: dict[str, Any]) -> "StructuredToolCall":
        return cls(
            call_id=data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            tool_name=data.get("function", {}).get("name", ""),
            arguments=json.loads(data.get("function", {}).get("arguments", "{}")),
        )

    @classmethod
    def from_anthropic(cls, data: dict[str, Any]) -> "StructuredToolCall":
        return cls(
            call_id=data.get("id", f"call_{uuid.uuid4().hex[:8]}"),
            tool_name=data.get("name", ""),
            arguments=data.get("input", {}),
        )


@dataclass
class ToolResult:
    """Result from tool execution"""

    call_id: str
    tool_name: str
    output: str
    success: bool = True
    error: str | None = None
    execution_time_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.call_id,
            "role": "tool",
            "content": self.output if self.success else f"Error: {self.error}",
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "type": "tool_result",
            "tool_use_id": self.call_id,
            "content": self.output if self.success else f"Error: {self.error}",
            "is_error": not self.success,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "success": self.success,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class StructuredOutput:
    """
    Structured LLM output with tool calls.

    This replaces regex parsing of responses.
    """

    content: str
    tool_calls: list[StructuredToolCall] = field(default_factory=list)
    finish_reason: FinishReason = FinishReason.STOP

    raw_response: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    model: str = ""
    provider: str = ""

    created_at: datetime = field(default_factory=datetime.now)

    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def get_tool_call_ids(self) -> list[str]:
        return [tc.call_id for tc in self.tool_calls]

    def to_dict(self) -> dict[str, Any]:
        return {
            "content": self.content,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "finish_reason": self.finish_reason.name,
            "tokens_used": self.tokens_used,
            "model": self.model,
            "provider": self.provider,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StructuredOutput":
        return cls(
            content=data.get("content", ""),
            tool_calls=[StructuredToolCall.from_dict(tc) for tc in data.get("tool_calls", [])],
            finish_reason=FinishReason[data.get("finish_reason", "STOP")],
            tokens_used=data.get("tokens_used", 0),
            model=data.get("model", ""),
            provider=data.get("provider", ""),
        )

    @classmethod
    def from_openai_response(
        cls,
        response: Any,
        model: str = "",
        provider: str = "openai",
    ) -> "StructuredOutput":
        content = ""
        tool_calls = []
        finish_reason = FinishReason.STOP

        if response.choices:
            choice = response.choices[0]
            content = choice.message.content or ""

            if choice.finish_reason == "tool_calls":
                finish_reason = FinishReason.TOOL_CALL
            elif choice.finish_reason == "length":
                finish_reason = FinishReason.MAX_TOKENS
            elif choice.finish_reason == "stop":
                finish_reason = FinishReason.STOP

            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append(StructuredToolCall.from_openai(tc))

        tokens_used = 0
        if response.usage:
            tokens_used = response.usage.total_tokens

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            model=model,
            provider=provider,
        )

    @classmethod
    def from_anthropic_response(
        cls,
        response: Any,
        model: str = "",
        provider: str = "anthropic",
    ) -> "StructuredOutput":
        content = ""
        tool_calls = []
        finish_reason = FinishReason.STOP

        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "type"):
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append(StructuredToolCall.from_anthropic(block))

            if tool_calls:
                finish_reason = FinishReason.TOOL_CALL

        tokens_used = 0
        if hasattr(response, "usage"):
            tokens_used = getattr(response.usage, "input_tokens", 0) + getattr(
                response.usage, "output_tokens", 0
            )

        return cls(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            tokens_used=tokens_used,
            model=model,
            provider=provider,
        )


@dataclass
class ExecutionStep:
    """A single step in an execution plan"""

    step_id: str
    description: str
    action: Literal["llm_call", "tool_call", "code_exec", "validation"]

    input_data: dict[str, Any] = field(default_factory=dict)
    expected_output: dict[str, Any] = field(default_factory=dict)

    dependencies: list[str] = field(default_factory=list)
    timeout_seconds: int = 30

    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "action": self.action,
            "input_data": self.input_data,
            "expected_output": self.expected_output,
            "dependencies": self.dependencies,
            "timeout_seconds": self.timeout_seconds,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }


@dataclass
class ExecutionPlan:
    """
    Complete execution plan for a task.

    Breaking down complex tasks into steps.
    """

    plan_id: str
    task_id: str
    steps: list[ExecutionStep] = field(default_factory=list)

    estimated_tokens: int = 1000
    estimated_time_seconds: int = 60
    risk_level: str = "MEDIUM"

    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = f"plan_{uuid.uuid4().hex[:8]}"

    def get_step(self, step_id: str) -> ExecutionStep | None:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_ready_steps(self, completed: set[str]) -> list[ExecutionStep]:
        ready = []
        for step in self.steps:
            if step.step_id in completed:
                continue
            if all(dep in completed for dep in step.dependencies):
                ready.append(step)
        return ready

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "task_id": self.task_id,
            "steps": [s.to_dict() for s in self.steps],
            "estimated_tokens": self.estimated_tokens,
            "estimated_time_seconds": self.estimated_time_seconds,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class StructuredToolRegistry:
    """Registry of tools with JSON schemas"""

    tools: dict[str, ToolDefinition] = field(default_factory=dict)

    def register(self, tool: ToolDefinition) -> None:
        self.tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self.tools.get(name)

    def get_all_schemas(
        self,
        format: Literal["openai", "anthropic", "gemini"] = "openai",
    ) -> list[dict[str, Any]]:
        schemas = []
        for tool in self.tools.values():
            if format == "openai":
                schemas.append(tool.to_openai_schema())
            elif format == "anthropic":
                schemas.append(tool.to_anthropic_schema())
            elif format == "gemini":
                schemas.append(tool.to_gemini_schema())
        return schemas

    def to_dict(self) -> dict[str, Any]:
        return {
            "tools": list(self.tools.keys()),
        }
