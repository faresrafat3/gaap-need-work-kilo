"""
MCP (Model Context Protocol) Client
====================================

Enables GAAP to connect to external MCP servers and use their tools.

MCP is a protocol for connecting AI systems to external tools and data sources.
Each MCP server provides:
- Tools: Functions the AI can call
- Resources: Data the AI can read
- Prompts: Pre-defined prompt templates

Usage:
    client = MCPClient()
    await client.connect("github-mcp-server")
    tools = await client.list_tools()
    result = await client.call_tool("search_repos", {"query": "gaap"})
"""

import asyncio
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.mcp")


class MCPConnectionState(Enum):
    """State of MCP connection"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class MCPTool:
    """Represents an MCP tool"""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    server_name: str = ""


@dataclass
class MCPResource:
    """Represents an MCP resource"""

    uri: str
    name: str
    description: str = ""
    mime_type: str = "text/plain"


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""

    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    timeout: int = 30


@dataclass
class MCPConnection:
    """Active MCP connection"""

    config: MCPServerConfig
    state: MCPConnectionState = MCPConnectionState.DISCONNECTED
    process: Any = None
    tools: list[MCPTool] = field(default_factory=list)
    resources: list[MCPResource] = field(default_factory=list)
    request_id: int = 0
    error: str = ""


class MCPClient:
    """
    Client for connecting to MCP servers.

    Supports:
    - Stdio transport (most common)
    - Tool discovery
    - Tool execution
    - Resource reading
    """

    def __init__(self, timeout: int = 30):
        self._connections: dict[str, MCPConnection] = {}
        self._timeout = timeout
        self._logger = logger

    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> bool:
        """
        Connect to an MCP server.

        Args:
            name: Unique name for this connection
            command: Command to run (e.g., "npx", "python")
            args: Arguments for the command
            env: Environment variables

        Returns:
            True if connected successfully
        """
        config = MCPServerConfig(
            name=name,
            command=command,
            args=args or [],
            env=env or {},
            timeout=self._timeout,
        )

        connection = MCPConnection(config=config)
        self._connections[name] = connection

        try:
            connection.state = MCPConnectionState.CONNECTING

            full_command = shutil.which(command) or command

            process = await asyncio.create_subprocess_exec(
                full_command,
                *config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**dict(os.environ), **config.env},
            )

            connection.process = process

            await self._send_request(
                connection,
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "gaap", "version": "1.0.0"},
                },
            )

            await self._send_request(connection, "notifications/initialized", {})

            tools_result = await self._send_request(connection, "tools/list", {})
            if tools_result and "tools" in tools_result:
                for tool_data in tools_result["tools"]:
                    connection.tools.append(
                        MCPTool(
                            name=tool_data.get("name", ""),
                            description=tool_data.get("description", ""),
                            input_schema=tool_data.get("inputSchema", {}),
                            server_name=name,
                        )
                    )

            connection.state = MCPConnectionState.CONNECTED
            self._logger.info(
                f"Connected to MCP server '{name}' with {len(connection.tools)} tools"
            )
            return True

        except Exception as e:
            connection.state = MCPConnectionState.ERROR
            connection.error = str(e)
            self._logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False

    async def disconnect(self, name: str) -> None:
        """Disconnect from an MCP server"""
        connection = self._connections.get(name)
        if connection and connection.process:
            try:
                connection.process.terminate()
                await asyncio.wait_for(connection.process.wait(), timeout=5)
            except Exception as e:
                self._logger.debug(f"Force killing MCP process: {e}")
                connection.process.kill()

            connection.state = MCPConnectionState.DISCONNECTED
            self._logger.info(f"Disconnected from MCP server '{name}'")

    async def list_tools(self, server_name: str | None = None) -> list[MCPTool]:
        """
        List available tools.

        Args:
            server_name: Specific server, or None for all

        Returns:
            List of available tools
        """
        tools = []
        for name, conn in self._connections.items():
            if server_name and name != server_name:
                continue
            if conn.state == MCPConnectionState.CONNECTED:
                tools.extend(conn.tools)
        return tools

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        server_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            server_name: Specific server (auto-detected if None)

        Returns:
            Tool result
        """
        connection = None
        if server_name:
            connection = self._connections.get(server_name)
        else:
            for conn in self._connections.values():
                if any(t.name == tool_name for t in conn.tools):
                    connection = conn
                    break

        if not connection:
            raise ValueError(f"Tool '{tool_name}' not found")

        if connection.state != MCPConnectionState.CONNECTED:
            raise RuntimeError(f"Server '{connection.config.name}' not connected")

        result = await self._send_request(
            connection,
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        return result or {}

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        """
        Read an MCP resource.

        Args:
            uri: Resource URI
            server_name: Specific server

        Returns:
            Resource content
        """
        connection = None
        if server_name:
            connection = self._connections.get(server_name)
        else:
            connection = next(
                (c for c in self._connections.values() if c.state == MCPConnectionState.CONNECTED),
                None,
            )

        if not connection:
            raise ValueError("No connected server")

        result = await self._send_request(
            connection,
            "resources/read",
            {"uri": uri},
        )

        if result and "contents" in result:
            contents = result["contents"]
            if isinstance(contents, list) and contents:
                return contents[0].get("text", str(contents[0]))

        return ""

    async def _send_request(
        self,
        connection: MCPConnection,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Send a JSON-RPC request to the MCP server"""
        if not connection.process or not connection.process.stdin:
            raise RuntimeError("Process not available")

        connection.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": connection.request_id,
            "method": method,
            "params": params,
        }

        request_str = json.dumps(request) + "\n"

        try:
            connection.process.stdin.write(request_str.encode())
            await connection.process.stdin.drain()

            if connection.process.stdout is None:
                return None

            response_line = await asyncio.wait_for(
                connection.process.stdout.readline(),
                timeout=connection.config.timeout,
            )

            response = json.loads(response_line.decode())

            if "error" in response:
                raise RuntimeError(f"MCP error: {response['error']}")

            return response.get("result")

        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout for method '{method}'")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics"""
        return {
            "servers": len(self._connections),
            "connected": sum(
                1 for c in self._connections.values() if c.state == MCPConnectionState.CONNECTED
            ),
            "total_tools": sum(len(c.tools) for c in self._connections.values()),
            "servers_detail": {
                name: {
                    "state": conn.state.value,
                    "tools": len(conn.tools),
                    "error": conn.error,
                }
                for name, conn in self._connections.items()
            },
        }

    async def close_all(self) -> None:
        """Disconnect from all servers"""
        for name in list(self._connections.keys()):
            await self.disconnect(name)


class MCPToolRegistry:
    """
    Bridge between MCP tools and GAAP's ToolRegistry.

    Allows MCP tools to be used seamlessly alongside built-in tools.
    """

    def __init__(self, mcp_client: MCPClient):
        self._mcp = mcp_client
        self._logger = logger

    async def get_mcp_tools(self) -> dict[str, MCPTool]:
        """Get all MCP tools as a dictionary"""
        tools = await self._mcp.list_tools()
        return {t.name: t for t in tools}

    async def execute_mcp_tool(self, name: str, **kwargs: Any) -> str:
        """Execute an MCP tool and return result as string"""
        try:
            result = await self._mcp.call_tool(name, kwargs)

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

        except Exception as e:
            return f"Error: {e}"

    def get_tool_instructions(self) -> str:
        """Get instructions for using MCP tools"""
        tools = []
        for conn in self._mcp._connections.values():
            tools.extend(conn.tools)

        if not tools:
            return ""

        instructions = "\n### MCP Tools (External) ###\n"
        for tool in tools:
            instructions += f"- {tool.name}: {tool.description}\n"
            if tool.input_schema:
                props = tool.input_schema.get("properties", {})
                if props:
                    params = ", ".join(f"{k}='...'" for k in props.keys())
                    instructions += f"  Usage: CALL: {tool.name}({params})\n"

        return instructions
