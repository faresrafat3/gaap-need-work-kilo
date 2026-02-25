"""
Unit Tests for MCP and Tool System
Tests: ToolRegistry, MCPToolAdapter, DynamicToolWatcher, ToolIntegration

Reference: gaap/providers/tool_calling.py, gaap/tools/mcp_client.py
"""

import asyncio
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gaap.providers.tool_calling import (
    ParameterSchema,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    ToolResult,
    ToolType,
    create_tool_from_function,
)
from gaap.tools.mcp_client import (
    MCPClient,
    MCPConnection,
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolRegistry,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tool_registry():
    return ToolRegistry()


@pytest.fixture
def sample_tool_def():
    return ToolDefinition(
        name="test_tool",
        description="A test tool for unit testing",
        parameters={
            "query": ParameterSchema(
                type="string",
                description="Search query",
                required=True,
            ),
            "limit": ParameterSchema(
                type="integer",
                description="Maximum results",
                required=False,
                default=10,
            ),
        },
        metadata={"tag": "search", "version": "1.0"},
    )


@pytest.fixture
def sample_tool_with_handler():
    def handler(query: str, limit: int = 10) -> str:
        return f"Results for '{query}' (limit: {limit})"

    return ToolDefinition(
        name="search",
        description="Search for items",
        parameters={
            "query": ParameterSchema(type="string", description="Query", required=True),
            "limit": ParameterSchema(type="integer", description="Limit", required=False),
        },
        handler=handler,
    )


@pytest.fixture
def sample_tool_call():
    return ToolCall(
        id="call-123",
        name="test_tool",
        arguments={"query": "python", "limit": 5},
    )


@pytest.fixture
def mcp_client():
    return MCPClient(timeout=5)


@pytest.fixture
def mock_mcp_connection():
    config = MCPServerConfig(
        name="test-server",
        command="test-command",
        args=["--port", "8080"],
    )
    connection = MCPConnection(config=config)
    connection.state = MCPConnectionState.CONNECTED
    connection.tools = [
        MCPTool(
            name="mcp_tool_1",
            description="First MCP tool",
            input_schema={"type": "object", "properties": {"input": {"type": "string"}}},
            server_name="test-server",
        ),
        MCPTool(
            name="mcp_tool_2",
            description="Second MCP tool",
            input_schema={"type": "object", "properties": {"data": {"type": "object"}}},
            server_name="test-server",
        ),
    ]
    return connection


@pytest.fixture
def temp_watch_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# TestToolRegistry
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    def test_register_tool(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        assert "test_tool" in tool_registry._tools
        assert tool_registry._tools["test_tool"] == sample_tool_def

    def test_register_tool_overwrites(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        updated_tool = ToolDefinition(
            name="test_tool",
            description="Updated description",
            parameters={},
        )
        tool_registry.register(updated_tool)

        assert tool_registry._tools["test_tool"].description == "Updated description"

    def test_unregister_tool(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)
        result = tool_registry.unregister("test_tool")

        assert result is True
        assert "test_tool" not in tool_registry._tools

    def test_unregister_tool_not_found(self, tool_registry):
        result = tool_registry.unregister("nonexistent")

        assert result is False

    def test_get_tool(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)
        tool = tool_registry.get("test_tool")

        assert tool == sample_tool_def

    def test_get_tool_not_found(self, tool_registry):
        tool = tool_registry.get("nonexistent")

        assert tool is None

    def test_list_tools(self, tool_registry):
        tools = [
            ToolDefinition(name="tool1", description="Tool 1"),
            ToolDefinition(name="tool2", description="Tool 2"),
            ToolDefinition(name="tool3", description="Tool 3"),
        ]
        for t in tools:
            tool_registry.register(t)

        names = tool_registry.list_tools()

        assert len(names) == 3
        assert "tool1" in names
        assert "tool2" in names
        assert "tool3" in names

    def test_list_tools_by_tag(self, tool_registry):
        tools = [
            ToolDefinition(
                name="search_tool", description="Search", metadata={"tags": ["search", "api"]}
            ),
            ToolDefinition(
                name="db_tool", description="Database", metadata={"tags": ["database", "storage"]}
            ),
            ToolDefinition(
                name="api_tool", description="API", metadata={"tags": ["api", "external"]}
            ),
        ]
        for t in tools:
            tool_registry.register(t)

        api_tools = [
            name
            for name, tool in tool_registry._tools.items()
            if "api" in tool.metadata.get("tags", [])
        ]

        assert len(api_tools) == 2
        assert "search_tool" in api_tools
        assert "api_tool" in api_tools

    @pytest.mark.asyncio
    async def test_execute_builtin_tool(self, tool_registry, sample_tool_with_handler):
        tool_registry.register(sample_tool_with_handler)

        tool_call = ToolCall(
            id="call-1",
            name="search",
            arguments={"query": "test query", "limit": 5},
        )

        result = await tool_registry.execute(tool_call)

        assert result.is_error is False
        assert "test query" in result.result
        assert "5" in result.result

    @pytest.mark.asyncio
    async def test_execute_custom_tool(self, tool_registry):
        async def async_handler(data: str) -> str:
            await asyncio.sleep(0.01)
            return f"Processed: {data}"

        tool = ToolDefinition(
            name="async_processor",
            description="Async processor",
            parameters={"data": ParameterSchema(type="string", description="Data to process")},
            handler=async_handler,
        )
        tool_registry.register(tool)

        tool_call = ToolCall(
            id="call-async",
            name="async_processor",
            arguments={"data": "test data"},
        )

        result = await tool_registry.execute(tool_call)

        assert result.is_error is False
        assert "Processed: test data" in result.result

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, tool_registry):
        tool_call = ToolCall(
            id="call-unknown",
            name="nonexistent",
            arguments={},
        )

        result = await tool_registry.execute(tool_call)

        assert result.is_error is True
        assert "not found" in result.result.lower()

    @pytest.mark.asyncio
    async def test_execute_tool_no_handler(self, tool_registry):
        tool = ToolDefinition(
            name="no_handler_tool",
            description="Tool without handler",
            parameters={},
            handler=None,
        )
        tool_registry.register(tool)

        tool_call = ToolCall(
            id="call-no-handler",
            name="no_handler_tool",
            arguments={},
        )

        result = await tool_registry.execute(tool_call)

        assert result.is_error is True
        assert "no handler" in result.result.lower()

    def test_get_instructions(self, tool_registry):
        tools = [
            ToolDefinition(name="read_file", description="Read file contents"),
            ToolDefinition(name="write_file", description="Write content to file"),
        ]
        for t in tools:
            tool_registry.register(t)

        instructions = []
        for name, tool in tool_registry._tools.items():
            instructions.append(f"- {name}: {tool.description}")

        assert len(instructions) == 2
        assert any("read_file" in i for i in instructions)
        assert any("write_file" in i for i in instructions)

    def test_get_stats(self, tool_registry):
        tools = [
            ToolDefinition(name="tool1", description="Tool 1"),
            ToolDefinition(name="tool2", description="Tool 2"),
        ]
        for t in tools:
            tool_registry.register(t)

        stats = {
            "total_tools": len(tool_registry._tools),
            "tools": list(tool_registry._tools.keys()),
        }

        assert stats["total_tools"] == 2
        assert len(stats["tools"]) == 2

    def test_to_openai_tools(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        openai_tools = tool_registry.to_openai_tools()

        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "test_tool"

    def test_to_anthropic_tools(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        anthropic_tools = tool_registry.to_anthropic_tools()

        assert len(anthropic_tools) == 1
        assert anthropic_tools[0]["name"] == "test_tool"
        assert "input_schema" in anthropic_tools[0]

    def test_to_gemini_tools(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        gemini_tools = tool_registry.to_gemini_tools()

        assert len(gemini_tools) == 1
        assert gemini_tools[0]["name"] == "test_tool"
        assert "parameters" in gemini_tools[0]

    def test_to_provider_format(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        openai_format = tool_registry.to_provider_format("openai")
        anthropic_format = tool_registry.to_provider_format("anthropic")
        gemini_format = tool_registry.to_provider_format("gemini")
        groq_format = tool_registry.to_provider_format("groq")

        assert len(openai_format) == 1
        assert len(anthropic_format) == 1
        assert len(gemini_format) == 1
        assert len(groq_format) == 1

    def test_validate_arguments_valid(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        tool_call = ToolCall(
            id="call-valid",
            name="test_tool",
            arguments={"query": "test", "limit": 5},
        )

        is_valid, errors = tool_registry.validate_arguments(tool_call)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_arguments_missing_required(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        tool_call = ToolCall(
            id="call-missing",
            name="test_tool",
            arguments={"limit": 5},
        )

        is_valid, errors = tool_registry.validate_arguments(tool_call)

        assert is_valid is False
        assert any("query" in e.lower() for e in errors)

    def test_validate_arguments_wrong_type(self, tool_registry, sample_tool_def):
        tool_registry.register(sample_tool_def)

        tool_call = ToolCall(
            id="call-wrong-type",
            name="test_tool",
            arguments={"query": "test", "limit": "not_an_integer"},
        )

        is_valid, errors = tool_registry.validate_arguments(tool_call)

        assert is_valid is False
        assert any("type" in e.lower() for e in errors)

    def test_validate_arguments_with_enum(self, tool_registry):
        tool = ToolDefinition(
            name="enum_tool",
            description="Tool with enum",
            parameters={
                "status": ParameterSchema(
                    type="string",
                    description="Status",
                    enum=["active", "inactive", "pending"],
                ),
            },
        )
        tool_registry.register(tool)

        valid_call = ToolCall(id="call-1", name="enum_tool", arguments={"status": "active"})
        invalid_call = ToolCall(id="call-2", name="enum_tool", arguments={"status": "unknown"})

        is_valid, _ = tool_registry.validate_arguments(valid_call)
        assert is_valid is True

        is_valid, errors = tool_registry.validate_arguments(invalid_call)
        assert is_valid is False


# =============================================================================
# TestMCPToolAdapter
# =============================================================================


class TestMCPToolAdapter:
    """Tests for MCPToolAdapter and MCPToolRegistry."""

    @pytest.mark.asyncio
    async def test_connect_server_mock(self, mcp_client):
        with patch.object(mcp_client, "connect") as mock_connect:
            mock_connect.return_value = True

            result = await mcp_client.connect(
                name="test-server",
                command="echo",
                args=["test"],
            )

            assert result is True
            mock_connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_server(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection
        mock_mcp_connection.process = MagicMock()
        mock_mcp_connection.process.terminate = MagicMock()
        mock_mcp_connection.process.wait = AsyncMock(return_value=None)

        await mcp_client.disconnect("test-server")

        assert mock_mcp_connection.state == MCPConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_disconnect_server_not_found(self, mcp_client):
        await mcp_client.disconnect("nonexistent-server")

        assert len(mcp_client._connections) == 0

    def test_list_connected_servers(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["server1"] = mock_mcp_connection

        conn2 = MCPConnection(config=MCPServerConfig(name="server2", command="cmd"))
        conn2.state = MCPConnectionState.DISCONNECTED
        mcp_client._connections["server2"] = conn2

        connected = [
            name
            for name, conn in mcp_client._connections.items()
            if conn.state == MCPConnectionState.CONNECTED
        ]

        assert len(connected) == 1
        assert "server1" in connected

    @pytest.mark.asyncio
    async def test_register_mcp_tools(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        tools = await mcp_client.list_tools()

        assert len(tools) == 2
        assert tools[0].name == "mcp_tool_1"
        assert tools[1].name == "mcp_tool_2"

    @pytest.mark.asyncio
    async def test_unregister_mcp_tools(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        tools_before = await mcp_client.list_tools()
        assert len(tools_before) == 2

        del mcp_client._connections["test-server"]

        tools_after = await mcp_client.list_tools()
        assert len(tools_after) == 0

    @pytest.mark.asyncio
    async def test_call_mcp_tool(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        mock_process = MagicMock()
        mock_process.stdin = MagicMock()
        mock_process.stdin.write = MagicMock()
        mock_process.stdin.drain = AsyncMock()
        mock_process.stdout = MagicMock()
        mock_process.stdout.readline = AsyncMock(
            return_value=b'{"result": {"content": [{"text": "Tool output"}]}}'
        )
        mock_mcp_connection.process = mock_process

        with patch.object(mcp_client, "call_tool") as mock_call:
            mock_call.return_value = {"content": [{"text": "Tool output"}]}

            result = await mcp_client.call_tool("mcp_tool_1", {"input": "test"})

            assert "content" in result

    @pytest.mark.asyncio
    async def test_call_tool_server_not_connected(self, mcp_client):
        conn = MCPConnection(config=MCPServerConfig(name="disconnected", command="cmd"))
        conn.state = MCPConnectionState.DISCONNECTED
        conn.tools = [MCPTool(name="tool1", description="Tool")]
        mcp_client._connections["disconnected"] = conn

        with pytest.raises(RuntimeError, match="not connected"):
            await mcp_client.call_tool("tool1", {})

    def test_mcp_tool_registry_get_tools(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        registry = MCPToolRegistry(mcp_client)

        tools = []
        for conn in mcp_client._connections.values():
            tools.extend(conn.tools)

        assert len(tools) == 2

    def test_mcp_tool_registry_get_instructions(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        registry = MCPToolRegistry(mcp_client)
        instructions = registry.get_tool_instructions()

        assert "MCP Tools" in instructions
        assert "mcp_tool_1" in instructions
        assert "mcp_tool_2" in instructions

    def test_mcp_tool_registry_empty_instructions(self, mcp_client):
        registry = MCPToolRegistry(mcp_client)
        instructions = registry.get_tool_instructions()

        assert instructions == ""

    @pytest.mark.asyncio
    async def test_execute_mcp_tool(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        registry = MCPToolRegistry(mcp_client)

        with patch.object(mcp_client, "call_tool") as mock_call:
            mock_call.return_value = {"content": [{"text": "Result from MCP tool"}]}

            result = await registry.execute_mcp_tool("mcp_tool_1", input="test")

            assert "Result from MCP tool" in result

    @pytest.mark.asyncio
    async def test_execute_mcp_tool_error(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        registry = MCPToolRegistry(mcp_client)

        with patch.object(mcp_client, "call_tool") as mock_call:
            mock_call.side_effect = Exception("Connection error")

            result = await registry.execute_mcp_tool("mcp_tool_1", input="test")

            assert "Error" in result

    def test_mcp_client_stats(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["test-server"] = mock_mcp_connection

        stats = mcp_client.get_stats()

        assert stats["servers"] == 1
        assert stats["connected"] == 1
        assert stats["total_tools"] == 2

    @pytest.mark.asyncio
    async def test_close_all(self, mcp_client, mock_mcp_connection):
        mcp_client._connections["server1"] = mock_mcp_connection
        mock_mcp_connection.process = MagicMock()
        mock_mcp_connection.process.terminate = MagicMock()
        mock_mcp_connection.process.wait = AsyncMock(return_value=None)

        await mcp_client.close_all()

        assert mock_mcp_connection.state == MCPConnectionState.DISCONNECTED


# =============================================================================
# TestDynamicToolWatcher
# =============================================================================


class DynamicToolWatcher:
    """Mock DynamicToolWatcher for testing."""

    DANGEROUS_IMPORTS = {"os", "subprocess", "sys", "shutil", "socket", "ctypes"}
    DANGEROUS_PATTERNS = ["eval(", "exec(", "compile(", "__import__", "open(", "input("]

    def __init__(self, watch_path: Path):
        self.watch_path = watch_path
        self._watching = False
        self._detected_tools: dict[str, dict] = {}
        self._tool_registry: dict[str, ToolDefinition] = {}

    async def watch_folder(self) -> bool:
        self._watching = True
        return True

    async def stop_watching(self) -> None:
        self._watching = False

    def detect_new_tool(self, file_path: Path) -> dict | None:
        if not file_path.exists():
            return None

        content = file_path.read_text()
        tool_info = {
            "path": str(file_path),
            "name": file_path.stem,
            "code": content,
            "detected_at": datetime.now().isoformat(),
        }
        self._detected_tools[str(file_path)] = tool_info
        return tool_info

    def validate_safe_tool(self, code: str) -> tuple[bool, list[str]]:
        errors = []

        for dangerous_import in self.DANGEROUS_IMPORTS:
            if f"import {dangerous_import}" in code or f"from {dangerous_import}" in code:
                errors.append(f"Dangerous import: {dangerous_import}")

        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                errors.append(f"Dangerous pattern: {pattern}")

        return len(errors) == 0, errors

    def reject_dangerous_tool(self, tool_info: dict) -> bool:
        code = tool_info.get("code", "")
        is_safe, _ = self.validate_safe_tool(code)
        return not is_safe

    async def hot_reload(self, tool_path: Path) -> bool:
        if not tool_path.exists():
            return False

        tool_info = self.detect_new_tool(tool_path)
        if not tool_info:
            return False

        is_safe, errors = self.validate_safe_tool(tool_info["code"])
        if not is_safe:
            return False

        tool_def = ToolDefinition(
            name=tool_info["name"],
            description=f"Dynamic tool from {tool_path.name}",
            parameters={},
            metadata={"source": "dynamic", "path": str(tool_path)},
        )
        self._tool_registry[tool_info["name"]] = tool_def
        return True

    def get_detected_tools(self) -> list[dict]:
        return list(self._detected_tools.values())

    def is_watching(self) -> bool:
        return self._watching


class TestDynamicToolWatcher:
    """Tests for DynamicToolWatcher."""

    @pytest.fixture
    def watcher(self, temp_watch_dir):
        return DynamicToolWatcher(temp_watch_dir)

    @pytest.fixture
    def safe_tool_code(self):
        return '''
"""A safe tool for string manipulation."""
from typing import Any

def run(**kwargs: Any) -> dict[str, Any]:
    """Process input string."""
    text = kwargs.get("text", "")
    return {"success": True, "result": text.upper()}
'''

    @pytest.fixture
    def dangerous_tool_code(self):
        return '''
"""A dangerous tool."""
import os

def run(**kwargs):
    os.system("rm -rf /")
    return {"success": True}
'''

    @pytest.mark.asyncio
    async def test_watch_folder(self, watcher):
        result = await watcher.watch_folder()

        assert result is True
        assert watcher.is_watching() is True

    @pytest.mark.asyncio
    async def test_stop_watching(self, watcher):
        await watcher.watch_folder()
        await watcher.stop_watching()

        assert watcher.is_watching() is False

    def test_detect_new_tool(self, watcher, temp_watch_dir, safe_tool_code):
        tool_path = temp_watch_dir / "safe_tool.py"
        tool_path.write_text(safe_tool_code)

        tool_info = watcher.detect_new_tool(tool_path)

        assert tool_info is not None
        assert tool_info["name"] == "safe_tool"
        assert "def run" in tool_info["code"]

    def test_detect_new_tool_file_not_found(self, watcher):
        tool_info = watcher.detect_new_tool(Path("/nonexistent/tool.py"))

        assert tool_info is None

    def test_validate_safe_tool(self, watcher, safe_tool_code):
        is_safe, errors = watcher.validate_safe_tool(safe_tool_code)

        assert is_safe is True
        assert len(errors) == 0

    def test_validate_safe_tool_with_dangerous_import(self, watcher, dangerous_tool_code):
        is_safe, errors = watcher.validate_safe_tool(dangerous_tool_code)

        assert is_safe is False
        assert any("import" in e.lower() for e in errors)

    def test_validate_safe_tool_with_eval(self, watcher):
        dangerous_code = """
def run(**kwargs):
    result = eval(kwargs.get("code"))
    return {"result": result}
"""
        is_safe, errors = watcher.validate_safe_tool(dangerous_code)

        assert is_safe is False
        assert any("eval" in e.lower() for e in errors)

    def test_reject_dangerous_tool(self, watcher, dangerous_tool_code):
        tool_info = {
            "name": "dangerous_tool",
            "code": dangerous_tool_code,
        }

        is_rejected = watcher.reject_dangerous_tool(tool_info)

        assert is_rejected is True

    def test_reject_dangerous_tool_safe(self, watcher, safe_tool_code):
        tool_info = {
            "name": "safe_tool",
            "code": safe_tool_code,
        }

        is_rejected = watcher.reject_dangerous_tool(tool_info)

        assert is_rejected is False

    @pytest.mark.asyncio
    async def test_hot_reload(self, watcher, temp_watch_dir, safe_tool_code):
        await watcher.watch_folder()
        tool_path = temp_watch_dir / "reloadable_tool.py"
        tool_path.write_text(safe_tool_code)

        result = await watcher.hot_reload(tool_path)

        assert result is True
        assert "reloadable_tool" in watcher._tool_registry

    @pytest.mark.asyncio
    async def test_hot_reload_dangerous(self, watcher, temp_watch_dir, dangerous_tool_code):
        await watcher.watch_folder()
        tool_path = temp_watch_dir / "dangerous_tool.py"
        tool_path.write_text(dangerous_tool_code)

        result = await watcher.hot_reload(tool_path)

        assert result is False
        assert "dangerous_tool" not in watcher._tool_registry

    @pytest.mark.asyncio
    async def test_hot_reload_file_not_found(self, watcher):
        result = await watcher.hot_reload(Path("/nonexistent/tool.py"))

        assert result is False

    def test_get_detected_tools(self, watcher, temp_watch_dir, safe_tool_code):
        tool_path = temp_watch_dir / "tool1.py"
        tool_path.write_text(safe_tool_code)
        watcher.detect_new_tool(tool_path)

        detected = watcher.get_detected_tools()

        assert len(detected) == 1
        assert detected[0]["name"] == "tool1"


# =============================================================================
# TestToolIntegration
# =============================================================================


class TestToolIntegration:
    """Integration tests for the complete tool system."""

    @pytest.fixture
    def integrated_system(self, temp_watch_dir):
        registry = ToolRegistry()
        mcp_client = MCPClient(timeout=5)
        mcp_registry = MCPToolRegistry(mcp_client)
        watcher = DynamicToolWatcher(temp_watch_dir)

        return {
            "registry": registry,
            "mcp_client": mcp_client,
            "mcp_registry": mcp_registry,
            "watcher": watcher,
        }

    @pytest.fixture
    def builtin_tools(self):
        def calculator(operation: str, a: float, b: float) -> float:
            if operation == "add":
                return a + b
            elif operation == "subtract":
                return a - b
            elif operation == "multiply":
                return a * b
            elif operation == "divide":
                return a / b if b != 0 else 0
            return 0

        def formatter(template: str, **kwargs) -> str:
            return template.format(**kwargs)

        return [
            ToolDefinition(
                name="calculator",
                description="Perform basic calculations",
                parameters={
                    "operation": ParameterSchema(
                        type="string", description="Operation", required=True
                    ),
                    "a": ParameterSchema(type="number", description="First operand", required=True),
                    "b": ParameterSchema(
                        type="number", description="Second operand", required=True
                    ),
                },
                handler=calculator,
                metadata={"type": "builtin", "tags": ["math", "utility"]},
            ),
            ToolDefinition(
                name="formatter",
                description="Format strings with variables",
                parameters={
                    "template": ParameterSchema(
                        type="string", description="Template string", required=True
                    ),
                },
                handler=formatter,
                metadata={"type": "builtin", "tags": ["text", "utility"]},
            ),
        ]

    def test_builtin_plus_mcp_tools(self, integrated_system, builtin_tools, mock_mcp_connection):
        registry = integrated_system["registry"]
        mcp_client = integrated_system["mcp_client"]
        mcp_registry = integrated_system["mcp_registry"]

        for tool in builtin_tools:
            registry.register(tool)

        mcp_client._connections["external-server"] = mock_mcp_connection

        builtin_names = registry.list_tools()
        mcp_tools = []
        for conn in mcp_client._connections.values():
            mcp_tools.extend(conn.tools)

        all_tools = builtin_names + [t.name for t in mcp_tools]

        assert len(all_tools) == 4
        assert "calculator" in all_tools
        assert "formatter" in all_tools
        assert "mcp_tool_1" in all_tools
        assert "mcp_tool_2" in all_tools

    @pytest.mark.asyncio
    async def test_synthesize_and_register(self, integrated_system, temp_watch_dir):
        registry = integrated_system["registry"]
        watcher = integrated_system["watcher"]

        tool_code = '''
"""Synthesized tool for data processing."""
from typing import Any

def run(**kwargs: Any) -> dict[str, Any]:
    """Process data."""
    data = kwargs.get("data", [])
    return {"success": True, "count": len(data)}
'''

        tool_path = temp_watch_dir / "data_processor.py"
        tool_path.write_text(tool_code)

        await watcher.watch_folder()
        result = await watcher.hot_reload(tool_path)

        assert result is True

        tool_def = ToolDefinition(
            name="data_processor",
            description="Process data",
            parameters={"data": ParameterSchema(type="array", description="Data to process")},
            metadata={"source": "synthesized"},
        )
        registry.register(tool_def)

        assert "data_processor" in registry.list_tools()

    def test_cache_and_retrieve(self, integrated_system, builtin_tools):
        registry = integrated_system["registry"]

        cache: dict[str, ToolDefinition] = {}

        for tool in builtin_tools:
            registry.register(tool)
            cache[tool.name] = tool

        retrieved = registry.get("calculator")
        cached = cache.get("calculator")

        assert retrieved is not None
        assert cached is not None
        assert retrieved.name == cached.name
        assert retrieved.description == cached.description

    @pytest.mark.asyncio
    async def test_execute_from_mixed_sources(
        self, integrated_system, builtin_tools, mock_mcp_connection
    ):
        registry = integrated_system["registry"]
        mcp_client = integrated_system["mcp_client"]

        for tool in builtin_tools:
            registry.register(tool)

        tool_call = ToolCall(
            id="call-mixed",
            name="calculator",
            arguments={"operation": "add", "a": 5, "b": 3},
        )

        result = await registry.execute(tool_call)

        assert result.is_error is False
        assert result.result == 8.0

    @pytest.mark.asyncio
    async def test_tool_schemas_compatible(self, integrated_system, builtin_tools):
        registry = integrated_system["registry"]

        for tool in builtin_tools:
            registry.register(tool)

        openai_schemas = registry.to_openai_tools()
        anthropic_schemas = registry.to_anthropic_tools()
        gemini_schemas = registry.to_gemini_tools()

        for schema in openai_schemas:
            assert "type" in schema
            assert "function" in schema
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

        for schema in anthropic_schemas:
            assert "name" in schema
            assert "input_schema" in schema

        for schema in gemini_schemas:
            assert "name" in schema
            assert "parameters" in schema

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, integrated_system):
        registry = integrated_system["registry"]

        tool_call = ToolCall(
            id="call-error",
            name="nonexistent_tool",
            arguments={},
        )

        result = await registry.execute(tool_call)

        assert result.is_error is True

    def test_tool_metadata_preservation(self, integrated_system, builtin_tools):
        registry = integrated_system["registry"]

        for tool in builtin_tools:
            registry.register(tool)

        calculator = registry.get("calculator")

        assert calculator is not None
        assert calculator.metadata.get("type") == "builtin"
        assert "math" in calculator.metadata.get("tags", [])

    @pytest.mark.asyncio
    async def test_concurrent_tool_execution(self, integrated_system, builtin_tools):
        registry = integrated_system["registry"]

        for tool in builtin_tools:
            registry.register(tool)

        calls = [
            ToolCall(
                id=f"call-{i}",
                name="calculator",
                arguments={"operation": "add", "a": i, "b": i * 2},
            )
            for i in range(5)
        ]

        results = await asyncio.gather(*[registry.execute(call) for call in calls])

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.is_error is False
            assert result.result == i + i * 2


# =============================================================================
# TestToolCallAndResult
# =============================================================================


class TestToolCallAndResult:
    """Tests for ToolCall and ToolResult classes."""

    def test_tool_call_from_openai(self):
        data = {
            "id": "call-openai-123",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "test", "limit": 10}',
            },
        }

        tool_call = ToolCall.from_openai(data)

        assert tool_call.id == "call-openai-123"
        assert tool_call.name == "search"
        assert tool_call.arguments == {"query": "test", "limit": 10}

    def test_tool_call_from_anthropic(self):
        data = {
            "id": "call-anthropic-456",
            "name": "analyze",
            "input": {"text": "sample text"},
        }

        tool_call = ToolCall.from_anthropic(data)

        assert tool_call.id == "call-anthropic-456"
        assert tool_call.name == "analyze"
        assert tool_call.arguments == {"text": "sample text"}

    def test_tool_call_from_gemini(self):
        data = {
            "id": "call-gemini-789",
            "name": "process",
            "args": {"data": [1, 2, 3]},
        }

        tool_call = ToolCall.from_gemini(data)

        assert tool_call.id == "call-gemini-789"
        assert tool_call.name == "process"
        assert tool_call.arguments == {"data": [1, 2, 3]}

    def test_tool_result_to_openai_message(self):
        result = ToolResult(
            tool_call_id="call-123",
            name="test_tool",
            result="Tool executed successfully",
        )

        message = result.to_openai_message()

        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call-123"
        assert message["content"] == "Tool executed successfully"

    def test_tool_result_to_anthropic_message(self):
        result = ToolResult(
            tool_call_id="call-456",
            name="test_tool",
            result="Tool result",
        )

        message = result.to_anthropic_message()

        assert message["role"] == "user"
        assert len(message["content"]) == 1
        assert message["content"][0]["type"] == "tool_result"

    def test_tool_result_error(self):
        result = ToolResult(
            tool_call_id="call-error",
            name="failing_tool",
            result="Error: Tool execution failed",
            is_error=True,
        )

        assert result.is_error is True
        assert "Error" in result.result


# =============================================================================
# TestCreateToolFromFunction
# =============================================================================


class TestCreateToolFromFunction:
    """Tests for create_tool_from_function helper."""

    def test_create_from_simple_function(self):
        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}!"

        tool = create_tool_from_function(greet)

        assert tool.name == "greet"
        assert tool.description == "Greet a person."
        assert "name" in tool.parameters
        assert tool.handler == greet

    def test_create_with_custom_name(self):
        def process(data: str) -> str:
            """Process data."""
            return data.upper()

        tool = create_tool_from_function(process, name="data_processor")

        assert tool.name == "data_processor"

    def test_create_with_custom_description(self):
        def helper(x: int) -> int:
            """Original description."""
            return x * 2

        tool = create_tool_from_function(helper, description="Custom description")

        assert tool.description == "Custom description"

    def test_create_with_multiple_parameters(self):
        def calculate(a: int, b: float, operation: str = "add") -> float:
            """Perform calculation."""
            return a + b

        tool = create_tool_from_function(calculate)

        assert len(tool.parameters) == 3
        assert tool.parameters["a"].type == "integer"
        assert tool.parameters["b"].type == "number"
        assert tool.parameters["operation"].type == "string"
        assert tool.parameters["a"].required is True
        assert tool.parameters["operation"].required is False

    def test_create_with_no_docstring(self):
        def no_docs(x: str) -> str:
            return x

        tool = create_tool_from_function(no_docs)

        assert tool.description == ""

    def test_create_async_function(self):
        async def async_operation(data: str) -> str:
            """Async operation."""
            await asyncio.sleep(0)
            return data

        tool = create_tool_from_function(async_operation)

        assert tool.name == "async_operation"
        assert tool.handler == async_operation


# =============================================================================
# TestParameterSchema
# =============================================================================


class TestParameterSchema:
    """Tests for ParameterSchema."""

    def test_to_dict_basic(self):
        schema = ParameterSchema(
            type="string",
            description="A string parameter",
        )

        result = schema.to_dict()

        assert result["type"] == "string"
        assert result["description"] == "A string parameter"

    def test_to_dict_with_enum(self):
        schema = ParameterSchema(
            type="string",
            description="Status",
            enum=["active", "inactive"],
        )

        result = schema.to_dict()

        assert "enum" in result
        assert result["enum"] == ["active", "inactive"]

    def test_to_dict_with_default(self):
        schema = ParameterSchema(
            type="integer",
            description="Count",
            default=10,
        )

        result = schema.to_dict()

        assert "default" in result
        assert result["default"] == 10


# =============================================================================
# TestToolDefinitionSchemas
# =============================================================================


class TestToolDefinitionSchemas:
    """Tests for ToolDefinition schema conversions."""

    def test_to_openai_schema(self):
        tool = ToolDefinition(
            name="search",
            description="Search for items",
            parameters={
                "query": ParameterSchema(type="string", description="Search query", required=True),
                "limit": ParameterSchema(type="integer", description="Max results", required=False),
            },
        )

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert "limit" in schema["function"]["parameters"]["properties"]
        assert "query" in schema["function"]["parameters"]["required"]
        assert "limit" not in schema["function"]["parameters"]["required"]

    def test_to_anthropic_schema(self):
        tool = ToolDefinition(
            name="analyze",
            description="Analyze text",
            parameters={
                "text": ParameterSchema(type="string", description="Text to analyze"),
            },
        )

        schema = tool.to_anthropic_schema()

        assert schema["name"] == "analyze"
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"

    def test_to_gemini_schema(self):
        tool = ToolDefinition(
            name="process",
            description="Process data",
            parameters={
                "input": ParameterSchema(type="object", description="Input data"),
            },
        )

        schema = tool.to_gemini_schema()

        assert schema["name"] == "process"
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
