"""
Tool Registry - Unified Tool Management
========================================

Central registry that integrates:
- Built-in tools (native_caller)
- MCP tools (external servers)
- Synthesized tools (just-in-time)
- Dynamic tools (hot-loaded from .gaap/dynamic_tools/)

Usage:
    registry = create_tool_registry()
    await registry.discover_mcp_tools()
    result = await registry.execute_tool("read_file", {"path": "test.txt"})
"""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    FileSystemEvent = None
    FileSystemEventHandler = object
    Observer = None

from gaap.core.logging import get_standard_logger
from gaap.tools.mcp_client import MCPClient
from gaap.tools.native_caller import BUILTIN_TOOLS, ToolResult, ToolSchema
from gaap.tools.skill_cache import SkillCache
from gaap.tools.synthesizer import ToolSynthesizer

logger = get_standard_logger("gaap.tools.registry")

DANGEROUS_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"__import__\s*\(",
    r"compile\s*\(",
    r"open\s*\(['\"]\/etc\/",
    r"subprocess\.(call|run|Popen)",
    r"os\.system\s*\(",
    r"pty\.spawn",
    r"commands\.getoutput",
    r"pickle\.loads?\s*\(",
    r"marshal\.loads?\s*\(",
    r"shelve\.open",
]


@dataclass
class RegistryStats:
    total_tools: int = 0
    builtin_tools: int = 0
    mcp_tools: int = 0
    synthesized_tools: int = 0
    dynamic_tools: int = 0
    servers_connected: int = 0
    cache_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tools": self.total_tools,
            "builtin_tools": self.builtin_tools,
            "mcp_tools": self.mcp_tools,
            "synthesized_tools": self.synthesized_tools,
            "dynamic_tools": self.dynamic_tools,
            "servers_connected": self.servers_connected,
            "cache_size": self.cache_size,
        }


class SecurityValidator:
    """Validates tool code for security concerns."""

    @staticmethod
    def check_dangerous_patterns(code: str) -> list[str]:
        """Check for dangerous code patterns."""
        found = []
        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                found.append(pattern)
        return found

    @staticmethod
    def validate_file_path(path: str, allowed_roots: list[str] | None = None) -> bool:
        """Validate a file path is safe."""
        if not path:
            return False

        abs_path = os.path.abspath(path)

        dangerous_paths = ["/etc/passwd", "/etc/shadow", "/root/.ssh", "~/.ssh"]
        for dangerous in dangerous_paths:
            if dangerous in abs_path:
                return False

        if abs_path.endswith((".env", ".pem", ".key", "credentials", "secrets")):
            return False

        if allowed_roots:
            return any(abs_path.startswith(root) for root in allowed_roots)

        return True

    @staticmethod
    def validate_ast(code: str) -> tuple[bool, str]:
        """Validate code can be parsed as valid Python AST."""
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            return False, str(e)

    @staticmethod
    def sandbox_check(code: str) -> tuple[bool, list[str]]:
        """Run all security checks and return result."""
        issues = SecurityValidator.check_dangerous_patterns(code)
        is_valid_ast, ast_error = SecurityValidator.validate_ast(code)

        if ast_error:
            issues.append(f"AST error: {ast_error}")

        return len(issues) == 0 and is_valid_ast, issues


class ToolRegistry:
    """
    Unified registry for all tool types.

    Manages:
    - Built-in tools
    - MCP tools from external servers
    - Synthesized tools (just-in-time)
    - Dynamic tools (hot-loaded)
    """

    def __init__(
        self,
        mcp_client: MCPClient | None = None,
        synthesizer: ToolSynthesizer | None = None,
        skill_cache: SkillCache | None = None,
        workspace_path: str = ".gaap",
    ):
        self._tools: dict[str, ToolSchema] = {}
        self._tool_funcs: dict[str, Callable[..., Any]] = {}
        self._tool_sources: dict[str, str] = {}

        self._mcp_client = mcp_client
        self._synthesizer = synthesizer
        self._skill_cache = skill_cache
        self._workspace_path = Path(workspace_path)

        self._dynamic_watcher: DynamicToolWatcher | None = None
        self._lock = threading.RLock()
        self._logger = logger

        self._init_builtin_tools()

    def _init_builtin_tools(self) -> None:
        """Initialize built-in tools."""
        for name, schema in BUILTIN_TOOLS.items():
            self._tools[name] = schema
            self._tool_sources[name] = "builtin"
        self._logger.info(f"Loaded {len(BUILTIN_TOOLS)} built-in tools")

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={len(self._tools)}, sources={len(set(self._tool_sources.values()))})"

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        func: Callable[..., Any] | None = None,
        tags: list[str] | None = None,
        source: str = "custom",
    ) -> bool:
        """
        Register a new tool.

        Args:
            name: Tool name (unique identifier)
            description: Tool description for LLM
            parameters: JSON schema of parameters
            func: Optional callable to execute
            tags: Optional tags for categorization
            source: Source type (builtin, mcp, synthesized, dynamic, custom)

        Returns:
            True if registered successfully
        """
        with self._lock:
            if name in self._tools:
                self._logger.warning(f"Tool '{name}' already exists, overwriting")

            schema = ToolSchema(
                name=name,
                description=description,
                parameters=parameters,
                required=list(parameters.keys()),
                tags=tags or [],
                func=func,
            )

            self._tools[name] = schema
            if func:
                self._tool_funcs[name] = func
            self._tool_sources[name] = source

            self._logger.debug(f"Registered tool '{name}' from source '{source}'")
            return True

    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool.

        Args:
            name: Tool name to unregister

        Returns:
            True if unregistered, False if not found
        """
        with self._lock:
            if name not in self._tools:
                return False

            del self._tools[name]
            self._tool_funcs.pop(name, None)
            self._tool_sources.pop(name, None)

            self._logger.debug(f"Unregistered tool '{name}'")
            return True

    def get_tool(self, name: str) -> ToolSchema | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            ToolSchema or None if not found
        """
        return self._tools.get(name)

    def list_tools(self) -> list[ToolSchema]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_tools_by_tag(self, tag: str) -> list[ToolSchema]:
        """
        List tools by tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of matching ToolSchema
        """
        return [t for t in self._tools.values() if tag in t.tags]

    def list_tools_by_source(self, source: str) -> list[ToolSchema]:
        """
        List tools by source.

        Args:
            source: Source type (builtin, mcp, synthesized, dynamic, custom)

        Returns:
            List of matching ToolSchema
        """
        return [self._tools[name] for name in self._tools if self._tool_sources.get(name) == source]

    async def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            ToolResult with output or error
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error=f"Tool '{name}' not found",
            )

        source = self._tool_sources.get(name, "unknown")

        try:
            if source == "mcp" and self._mcp_client:
                return await self._execute_mcp_tool(name, arguments)

            if source == "synthesized":
                return await self._execute_synthesized_tool(name, arguments)

            if source == "dynamic":
                return await self._execute_dynamic_tool(name, arguments)

            if name in self._tool_funcs:
                result = self._tool_funcs[name](**arguments)
                return ToolResult(
                    call_id="",
                    name=name,
                    output=str(result),
                    success=True,
                )

            return await self._execute_builtin_tool(name, arguments)

        except Exception as e:
            self._logger.error(f"Tool execution failed for '{name}': {e}")
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error=str(e),
            )

    async def _execute_builtin_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a built-in tool with security checks."""
        if name == "read_file":
            path = arguments.get("path", "")
            if not SecurityValidator.validate_file_path(path):
                return ToolResult(
                    call_id="",
                    name=name,
                    output="",
                    success=False,
                    error="Security: Invalid or unauthorized file path",
                )
            try:
                with open(path, "r") as f:
                    content = f.read()
                return ToolResult(call_id="", name=name, output=content, success=True)
            except Exception as e:
                return ToolResult(call_id="", name=name, output="", success=False, error=str(e))

        elif name == "write_file":
            path = arguments.get("path", "")
            content = arguments.get("content", "")
            if not SecurityValidator.validate_file_path(path):
                return ToolResult(
                    call_id="",
                    name=name,
                    output="",
                    success=False,
                    error="Security: Invalid or unauthorized file path",
                )
            try:
                with open(path, "w") as f:
                    f.write(content)
                return ToolResult(call_id="", name=name, output="SUCCESS", success=True)
            except Exception as e:
                return ToolResult(call_id="", name=name, output="", success=False, error=str(e))

        return ToolResult(
            call_id="",
            name=name,
            output="",
            success=False,
            error=f"Built-in tool '{name}' not implemented",
        )

    async def _execute_mcp_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute an MCP tool."""
        if not self._mcp_client:
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error="MCP client not configured",
            )

        try:
            result = await self._mcp_client.call_tool(name, arguments)
            output = self._format_mcp_result(result)
            return ToolResult(call_id="", name=name, output=output, success=True)
        except Exception as e:
            return ToolResult(call_id="", name=name, output="", success=False, error=str(e))

    def _format_mcp_result(self, result: dict[str, Any]) -> str:
        """Format MCP tool result as string."""
        import json

        if isinstance(result, dict):
            if "content" in result:
                content = result["content"]
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                    return "\n".join(texts)
            return json.dumps(result, indent=2)
        return str(result)

    async def _execute_synthesized_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a synthesized tool."""
        if not self._synthesizer:
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error="Synthesizer not configured",
            )

        tool = self._synthesizer.skill_cache.retrieve(name.replace("synth_", ""))
        if not tool or not tool.module:
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error=f"Synthesized tool '{name}' not found",
            )

        try:
            if hasattr(tool.module, "run"):
                result = tool.module.run(**arguments)
            elif hasattr(tool.module, "execute"):
                result = tool.module.execute(**arguments)
            else:
                return ToolResult(
                    call_id="",
                    name=name,
                    output="",
                    success=False,
                    error="Tool has no run() or execute() function",
                )

            return ToolResult(
                call_id="",
                name=name,
                output=str(result),
                success=True,
            )
        except Exception as e:
            return ToolResult(call_id="", name=name, output="", success=False, error=str(e))

    async def _execute_dynamic_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a dynamically loaded tool."""
        func = self._tool_funcs.get(name)
        if not func:
            return ToolResult(
                call_id="",
                name=name,
                output="",
                success=False,
                error=f"Dynamic tool '{name}' function not found",
            )

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            return ToolResult(call_id="", name=name, output=str(result), success=True)
        except Exception as e:
            return ToolResult(call_id="", name=name, output="", success=False, error=str(e))

    async def discover_mcp_tools(self) -> int:
        """
        Discover and register tools from connected MCP servers.

        Returns:
            Count of discovered tools
        """
        if not self._mcp_client:
            self._logger.warning("MCP client not configured")
            return 0

        tools = await self._mcp_client.list_tools()
        count = 0

        for mcp_tool in tools:
            schema = ToolSchema(
                name=mcp_tool.name,
                description=mcp_tool.description,
                parameters=mcp_tool.input_schema.get("properties", {}),
                required=mcp_tool.input_schema.get("required", []),
                tags=["mcp", mcp_tool.server_name],
            )

            with self._lock:
                self._tools[mcp_tool.name] = schema
                self._tool_sources[mcp_tool.name] = "mcp"

            count += 1
            self._logger.debug(f"Discovered MCP tool: {mcp_tool.name}")

        self._logger.info(f"Discovered {count} MCP tools")
        return count

    async def synthesize_tool(self, intent: str) -> str | None:
        """
        Synthesize a new tool for the given intent.

        Args:
            intent: Description of what the tool should do

        Returns:
            Tool ID if successful, None otherwise
        """
        if not self._synthesizer:
            self._logger.warning("Synthesizer not configured")
            return None

        tool = await self._synthesizer.synthesize_with_discovery(intent)
        if not tool:
            return None

        tool_name = f"synth_{tool.id}"

        params = {}
        if hasattr(tool.module, "__annotations__"):
            for param_name, param_type in tool.module.__annotations__.items():
                if param_name != "return":
                    params[param_name] = {"type": "string"}

        self.register_tool(
            name=tool_name,
            description=tool.description,
            parameters=params,
            tags=["synthesized", tool.category],
            source="synthesized",
        )

        if tool.module and hasattr(tool.module, "run"):
            self._tool_funcs[tool_name] = tool.module.run
        elif tool.module and hasattr(tool.module, "execute"):
            self._tool_funcs[tool_name] = tool.module.execute

        self._logger.info(f"Synthesized and registered tool: {tool_name}")
        return tool.id

    def load_cached_tool(self, tool_id: str) -> bool:
        """
        Load a tool from the skill cache.

        Args:
            tool_id: The skill ID to load

        Returns:
            True if loaded successfully
        """
        if not self._skill_cache:
            self._logger.warning("Skill cache not configured")
            return False

        tool = self._skill_cache.retrieve(tool_id)
        if not tool:
            return False

        tool_name = f"cached_{tool_id}"

        params = {}
        if tool.module and hasattr(tool.module, "__annotations__"):
            for param_name, param_type in tool.module.__annotations__.items():
                if param_name != "return":
                    params[param_name] = {"type": "string"}

        self.register_tool(
            name=tool_name,
            description=tool.description,
            parameters=params,
            tags=["cached", tool.category] if hasattr(tool, "category") else ["cached"],
            source="synthesized",
        )

        if tool.module:
            if hasattr(tool.module, "run"):
                self._tool_funcs[tool_name] = tool.module.run
            elif hasattr(tool.module, "execute"):
                self._tool_funcs[tool_name] = tool.module.execute

        self._logger.info(f"Loaded cached tool: {tool_name}")
        return True

    def start_dynamic_watcher(self) -> None:
        """Start watching for dynamic tool changes."""
        if self._dynamic_watcher:
            return

        dynamic_path = self._workspace_path / "dynamic_tools"
        dynamic_path.mkdir(parents=True, exist_ok=True)

        self._dynamic_watcher = DynamicToolWatcher(self, str(dynamic_path))
        self._dynamic_watcher.start()

        self._load_existing_dynamic_tools(dynamic_path)

    def stop_dynamic_watcher(self) -> None:
        """Stop the dynamic tool watcher."""
        if self._dynamic_watcher:
            self._dynamic_watcher.stop()
            self._dynamic_watcher = None

    def _load_existing_dynamic_tools(self, path: Path) -> None:
        """Load existing dynamic tools from directory."""
        for py_file in path.glob("*.py"):
            self._load_dynamic_tool_file(py_file)

    def _load_dynamic_tool_file(self, file_path: Path) -> bool:
        """
        Load a dynamic tool from file with security validation.

        Args:
            file_path: Path to the Python file

        Returns:
            True if loaded successfully
        """
        try:
            with open(file_path, "r") as f:
                code = f.read()

            is_safe, issues = SecurityValidator.sandbox_check(code)
            if not is_safe:
                self._logger.error(f"Security check failed for {file_path}: {issues}")
                return False

            tool_name = file_path.stem
            module_name = f"dynamic_{tool_name}"

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if not spec or not spec.loader:
                self._logger.error(f"Failed to create module spec for {file_path}")
                return False

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            if not (hasattr(module, "run") or hasattr(module, "execute")):
                self._logger.warning(f"Dynamic tool {tool_name} has no run() or execute() function")
                return False

            description = getattr(module, "DESCRIPTION", f"Dynamic tool: {tool_name}")
            params = getattr(module, "PARAMETERS", {})
            tags = getattr(module, "TAGS", ["dynamic"])

            func = module.run if hasattr(module, "run") else module.execute

            self.register_tool(
                name=tool_name,
                description=description,
                parameters=params,
                func=func,
                tags=tags,
                source="dynamic",
            )

            self._logger.info(f"Loaded dynamic tool: {tool_name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load dynamic tool {file_path}: {e}")
            return False

    def get_instructions(self) -> str:
        """
        Get tool instructions for LLM prompt.

        Returns:
            Formatted string of available tools
        """
        lines = ["# Available Tools\n"]
        lines.append("You have access to the following tools:\n")

        by_source: dict[str, list[ToolSchema]] = {}
        for name, schema in self._tools.items():
            source = self._tool_sources.get(name, "other")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(schema)

        source_labels = {
            "builtin": "Built-in Tools",
            "mcp": "MCP Tools (External)",
            "synthesized": "Synthesized Tools",
            "dynamic": "Dynamic Tools",
            "custom": "Custom Tools",
        }

        for source in ["builtin", "mcp", "synthesized", "dynamic", "custom"]:
            if source not in by_source:
                continue

            tools = by_source[source]
            label = source_labels.get(source, source.title())
            lines.append(f"\n## {label}\n")

            for tool in sorted(tools, key=lambda t: t.name):
                lines.append(f"### {tool.name}")
                lines.append(f"{tool.description}\n")

                if tool.parameters:
                    lines.append("**Parameters:**")
                    for param, schema in tool.parameters.items():
                        param_type = schema.get("type", "any")
                        required = param in tool.required
                        req_str = " (required)" if required else ""
                        lines.append(f"- `{param}`: {param_type}{req_str}")
                    lines.append("")

                if tool.tags:
                    lines.append(f"**Tags:** {', '.join(tool.tags)}\n")

        return "\n".join(lines)

    def get_stats(self) -> dict[str, Any]:
        """
        Get registry statistics.

        Returns:
            Dict with tool counts and other stats
        """
        stats = RegistryStats()

        stats.total_tools = len(self._tools)
        stats.builtin_tools = sum(1 for s in self._tool_sources.values() if s == "builtin")
        stats.mcp_tools = sum(1 for s in self._tool_sources.values() if s == "mcp")
        stats.synthesized_tools = sum(1 for s in self._tool_sources.values() if s == "synthesized")
        stats.dynamic_tools = sum(1 for s in self._tool_sources.values() if s == "dynamic")

        if self._mcp_client:
            mcp_stats = self._mcp_client.get_stats()
            stats.servers_connected = mcp_stats.get("connected", 0)

        if self._skill_cache:
            cache_stats = self._skill_cache.get_stats()
            stats.cache_size = cache_stats.total_skills

        return stats.to_dict()

    async def close(self) -> None:
        """Clean up resources."""
        self.stop_dynamic_watcher()

        if self._mcp_client:
            await self._mcp_client.close_all()

        if self._synthesizer:
            await self._synthesizer.close()


class MCPToolAdapter:
    """
    Adapter for managing MCP server connections and tool registration.

    Usage:
        adapter = MCPToolAdapter(registry, mcp_client)
        await adapter.connect_server("github", "npx", ["-y", "@github/mcp-server"])
        await adapter.register_mcp_tools("github")
    """

    def __init__(self, registry: ToolRegistry, mcp_client: MCPClient):
        self._registry = registry
        self._mcp_client = mcp_client
        self._logger = logger

    async def connect_server(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Server name
            command: Command to run
            args: Command arguments
            env: Environment variables

        Returns:
            True if connected successfully
        """
        try:
            connected = await self._mcp_client.connect(
                name=name,
                command=command,
                args=args,
                env=env,
            )
            if connected:
                self._logger.info(f"Connected to MCP server: {name}")
            return connected
        except Exception as e:
            self._logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False

    async def disconnect_server(self, name: str) -> bool:
        """
        Disconnect from an MCP server.

        Args:
            name: Server name

        Returns:
            True if disconnected
        """
        try:
            await self._mcp_client.disconnect(name)
            await self.unregister_mcp_tools(name)
            self._logger.info(f"Disconnected from MCP server: {name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to disconnect from MCP server '{name}': {e}")
            return False

    def list_connected_servers(self) -> list[str]:
        """List all connected MCP servers."""
        stats = self._mcp_client.get_stats()
        servers = []
        for name, info in stats.get("servers_detail", {}).items():
            if info.get("state") == "connected":
                servers.append(name)
        return servers

    async def register_mcp_tools(self, server_name: str) -> int:
        """
        Register all tools from an MCP server.

        Args:
            server_name: Server name

        Returns:
            Count of registered tools
        """
        tools = await self._mcp_client.list_tools(server_name=server_name)
        count = 0

        for mcp_tool in tools:
            schema = ToolSchema(
                name=mcp_tool.name,
                description=mcp_tool.description,
                parameters=mcp_tool.input_schema.get("properties", {}),
                required=mcp_tool.input_schema.get("required", []),
                tags=["mcp", server_name],
            )

            self._registry._tools[mcp_tool.name] = schema
            self._registry._tool_sources[mcp_tool.name] = "mcp"
            count += 1

        self._logger.info(f"Registered {count} tools from MCP server '{server_name}'")
        return count

    async def unregister_mcp_tools(self, server_name: str) -> int:
        """
        Unregister all tools from an MCP server.

        Args:
            server_name: Server name

        Returns:
            Count of unregistered tools
        """
        count = 0
        to_remove = []

        for name, source in self._registry._tool_sources.items():
            if source == "mcp":
                tool = self._registry._tools.get(name)
                if tool and server_name in tool.tags:
                    to_remove.append(name)

        for name in to_remove:
            self._registry.unregister_tool(name)
            count += 1

        self._logger.info(f"Unregistered {count} tools from MCP server '{server_name}'")
        return count


class DynamicToolWatcher:
    """
    Watches .gaap/dynamic_tools/ for new tools and hot-reloads on changes.
    """

    def __init__(self, registry: ToolRegistry, watch_path: str):
        self._registry = registry
        self._watch_path = Path(watch_path)
        self._observer: Any = None
        self._logger = logger
        self._loaded_files: dict[str, float] = {}
        self._handler: Any = None

    def start(self) -> None:
        """Start watching for file changes."""
        if not WATCHDOG_AVAILABLE:
            self._logger.warning("watchdog not installed, dynamic tool watching disabled")
            return

        self._watch_path.mkdir(parents=True, exist_ok=True)

        watcher_self = self

        class _EventHandler(FileSystemEventHandler):  # type: ignore
            def on_created(self, event: Any) -> None:
                watcher_self.on_created(event)

            def on_modified(self, event: Any) -> None:
                watcher_self.on_modified(event)

            def on_deleted(self, event: Any) -> None:
                watcher_self.on_deleted(event)

        self._handler = _EventHandler()
        self._observer = Observer()  # type: ignore
        self._observer.schedule(self._handler, str(self._watch_path), recursive=False)
        self._observer.start()
        self._logger.info(f"Started watching {self._watch_path}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
        self._logger.info("Stopped dynamic tool watcher")

    def on_created(self, event: Any) -> None:
        """Handle file creation."""
        if getattr(event, "is_directory", False) or not str(
            getattr(event, "src_path", "")
        ).endswith(".py"):
            return

        self._logger.info(f"New dynamic tool detected: {event.src_path}")
        self._load_tool(Path(event.src_path))

    def on_modified(self, event: Any) -> None:
        """Handle file modification."""
        if getattr(event, "is_directory", False) or not str(
            getattr(event, "src_path", "")
        ).endswith(".py"):
            return

        self._logger.info(f"Dynamic tool modified: {event.src_path}")
        self._reload_tool(Path(event.src_path))

    def on_deleted(self, event: Any) -> None:
        """Handle file deletion."""
        if getattr(event, "is_directory", False) or not str(
            getattr(event, "src_path", "")
        ).endswith(".py"):
            return

        self._logger.info(f"Dynamic tool deleted: {event.src_path}")
        self._unload_tool(Path(event.src_path))

    def _load_tool(self, file_path: Path) -> bool:
        """Load a tool from file."""
        return self._registry._load_dynamic_tool_file(file_path)

    def _reload_tool(self, file_path: Path) -> bool:
        """Reload a tool after modification."""
        tool_name = file_path.stem

        self._registry.unregister_tool(tool_name)

        return self._load_tool(file_path)

    def _unload_tool(self, file_path: Path) -> None:
        """Unload a tool when file is deleted."""
        tool_name = file_path.stem
        self._registry.unregister_tool(tool_name)


def create_tool_registry(
    workspace_path: str = ".gaap",
    mcp_timeout: int = 30,
    enable_mcp: bool = True,
    enable_synthesizer: bool = True,
) -> ToolRegistry:
    """
    Factory function to create a configured ToolRegistry.

    Args:
        workspace_path: Path for dynamic tools and cache
        mcp_timeout: Timeout for MCP connections
        enable_mcp: Whether to enable MCP client
        enable_synthesizer: Whether to enable tool synthesizer

    Returns:
        Configured ToolRegistry instance
    """
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    mcp_client = MCPClient(timeout=mcp_timeout) if enable_mcp else None

    synthesizer = None
    skill_cache = None

    if enable_synthesizer:
        skill_cache = SkillCache(cache_path=workspace / "skills")
        synthesizer = ToolSynthesizer(
            workspace_path=workspace / "custom_tools",
            cache_path=workspace / "skills",
        )

    registry = ToolRegistry(
        mcp_client=mcp_client,
        synthesizer=synthesizer,
        skill_cache=skill_cache,
        workspace_path=workspace_path,
    )

    logger.info(f"Created tool registry at {workspace_path}")
    return registry
