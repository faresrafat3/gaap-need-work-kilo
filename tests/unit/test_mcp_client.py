"""Tests for MCP Client"""

import pytest

from gaap.tools.mcp_client import (
    MCPClient,
    MCPConnection,
    MCPConnectionState,
    MCPServerConfig,
    MCPTool,
    MCPToolRegistry,
)


class TestMCPTool:
    """Tests for MCPTool"""

    def test_tool_creation(self):
        """Test creating an MCP tool"""
        tool = MCPTool(
            name="search_repos",
            description="Search GitHub repositories",
            input_schema={"query": {"type": "string"}},
            server_name="github",
        )

        assert tool.name == "search_repos"
        assert tool.server_name == "github"


class TestMCPServerConfig:
    """Tests for MCPServerConfig"""

    def test_config_creation(self):
        """Test creating server config"""
        config = MCPServerConfig(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
        )

        assert config.name == "github"
        assert config.command == "npx"
        assert len(config.args) == 2


class TestMCPConnection:
    """Tests for MCPConnection"""

    def test_connection_creation(self):
        """Test creating connection"""
        config = MCPServerConfig(name="test", command="test")
        connection = MCPConnection(config=config)

        assert connection.state == MCPConnectionState.DISCONNECTED
        assert len(connection.tools) == 0


class TestMCPClient:
    """Tests for MCPClient"""

    @pytest.fixture
    def client(self):
        """Create MCP client"""
        return MCPClient(timeout=5)

    def test_client_creation(self, client):
        """Test creating client"""
        assert len(client._connections) == 0

    def test_get_stats_empty(self, client):
        """Test stats when no connections"""
        stats = client.get_stats()

        assert stats["servers"] == 0
        assert stats["connected"] == 0
        assert stats["total_tools"] == 0

    @pytest.mark.asyncio
    async def test_list_tools_empty(self, client):
        """Test listing tools with no connections"""
        tools = await client.list_tools()

        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_connect_invalid_command(self, client):
        """Test connecting with invalid command"""
        result = await client.connect(
            name="invalid",
            command="nonexistent_command_xyz",
        )

        assert result is False

        stats = client.get_stats()
        assert stats["servers"] == 1
        assert stats["connected"] == 0

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent(self, client):
        """Test disconnecting from nonexistent server"""
        await client.disconnect("nonexistent")

    @pytest.mark.asyncio
    async def test_close_all(self, client):
        """Test closing all connections"""
        await client.close_all()

        assert len(client._connections) == 0

    @pytest.mark.asyncio
    async def test_call_tool_not_found(self, client):
        """Test calling nonexistent tool"""
        with pytest.raises(ValueError, match="not found"):
            await client.call_tool("nonexistent_tool", {})


class TestMCPToolRegistry:
    """Tests for MCPToolRegistry"""

    @pytest.fixture
    def registry(self):
        """Create registry"""
        client = MCPClient()
        return MCPToolRegistry(client)

    @pytest.mark.asyncio
    async def test_get_mcp_tools_empty(self, registry):
        """Test getting tools with no connections"""
        tools = await registry.get_mcp_tools()

        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_get_tool_instructions_empty(self, registry):
        """Test instructions with no tools"""
        instructions = registry.get_tool_instructions()

        assert instructions == ""


class TestMCPConnectionState:
    """Tests for MCPConnectionState enum"""

    def test_states_exist(self):
        """Test all states exist"""
        assert MCPConnectionState.DISCONNECTED.value == "disconnected"
        assert MCPConnectionState.CONNECTING.value == "connecting"
        assert MCPConnectionState.CONNECTED.value == "connected"
        assert MCPConnectionState.ERROR.value == "error"
