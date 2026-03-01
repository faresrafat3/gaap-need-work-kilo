from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolSchema:
    """JSON Schema for a tool with domain-specific tags"""

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    func: Any = None

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


@dataclass
class ToolCall:
    """Represents a parsed tool call"""

    name: str
    arguments: dict[str, Any]
    call_id: str = ""
    raw_output: str = ""


@dataclass
class ToolResult:
    """Result of tool execution"""

    call_id: str
    name: str
    output: str
    success: bool = True
    error: str | None = None


BUILTIN_TOOLS: dict[str, ToolSchema] = {
    "read_file": ToolSchema(
        name="read_file",
        description="Read contents of a file (Restricted to project paths)",
        parameters={"path": {"type": "string"}},
        required=["path"],
        tags=["coding", "diagnostic"],
    ),
    "write_file": ToolSchema(
        name="write_file",
        description="Write content to a file safely",
        parameters={"path": {"type": "string"}, "content": {"type": "string"}},
        required=["path", "content"],
        tags=["coding"],
    ),
    "execute_python": ToolSchema(
        name="execute_python",
        description="Execute Python in a secure sandbox",
        parameters={"code": {"type": "string"}},
        required=["code"],
        tags=["coding", "analysis"],
    ),
    "google_search": ToolSchema(
        name="google_search",
        description="Search web via Google",
        parameters={"query": {"type": "string"}},
        required=["query"],
        tags=["research"],
    ),
    "arxiv_search": ToolSchema(
        name="arxiv_search",
        description="Search ArXiv papers",
        parameters={"query": {"type": "string"}},
        required=["query"],
        tags=["research"],
    ),
}

from gaap.core.logging import get_standard_logger as get_logger


class NativeToolCaller:
    """Native Function Calling Handler (Sovereign Edition)"""

    TOOL_CALL_PROMPT = "You have access to these tools: {tools}\nUse JSON format for calls."

    def __init__(self, tools: dict[str, ToolSchema] | None = None):
        self._tools = tools or BUILTIN_TOOLS.copy()
        self._logger = get_logger("gaap.tools.native")

    def __repr__(self) -> str:
        return f"NativeToolCaller(tools={len(self._tools)})"

    def get_tools_by_tags(self, tags: list[str]) -> list[ToolSchema]:
        return [t for t in self._tools.values() if any(tag in t.tags for tag in tags)]

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Any = None,
        tags: list[str] | None = None,
    ) -> None:
        """Register a new tool."""
        self._tools[name] = ToolSchema(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
            tags=tags or ["custom"],
        )

    def get_instructions(self, tools: list[ToolSchema] | None = None) -> str:
        target = tools if tools is not None else list(self._tools.values())
        desc = [f"- {t.name}: {t.description}" for t in target]
        return self.TOOL_CALL_PROMPT.format(tools="\n".join(desc))

    def execute_call(self, call: ToolCall) -> ToolResult:
        if call.name not in self._tools:
            return ToolResult(
                call_id=call.call_id,
                name=call.name,
                output="",
                success=False,
                error=f"Unknown tool: {call.name}",
            )

        try:
            res = self._default_execute(call.name, call.arguments)
            return ToolResult(call_id=call.call_id, name=call.name, output=str(res))
        except Exception as e:
            return ToolResult(
                call_id=call.call_id, name=call.name, output="", success=False, error=str(e)
            )

    def _default_execute(self, name: str, args: dict[str, Any]) -> str:
        import os

        safe_root = os.getcwd()
        path = args.get("path", "")
        if path:
            abs_path = os.path.abspath(path)
            if not abs_path.startswith(safe_root) or ".env" in path:
                return "SECURITY ERROR: Unauthorized path."

        if name == "read_file":
            with open(path, "r") as f:
                return f.read()
        elif name == "write_file":
            with open(path, "w") as f:
                f.write(args.get("content", ""))
                return "SUCCESS"
        return f"Tool {name} executed simulation mode."


def create_tool_caller(tools: dict[str, ToolSchema] | None = None) -> NativeToolCaller:
    return NativeToolCaller(tools=tools)
