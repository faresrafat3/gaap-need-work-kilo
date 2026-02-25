# GAAP Evolution: MCP & Dynamic Tooling

**Focus:** Breaking the boundaries of hardcoded tools using the Model Context Protocol.

---

## Implementation Status: ✅ COMPLETE

**Completion Date:** February 25, 2026

### Implemented Components

| Component | File | Description |
|-----------|------|-------------|
| ToolRegistry | `gaap/tools/registry.py` | Dynamic tool registration and management |
| MCPToolAdapter | `gaap/tools/mcp_client.py` | Model Context Protocol integration |
| DynamicToolWatcher | `gaap/tools/watcher.py` | Hot-reload for custom tools |

### Statistics

| Metric | Value |
|--------|-------|
| Lines of Code | ~600 |
| Test Functions | 76 |
| MCP Server Support | Filesystem, SQLite, GitHub |

---

## 1. The Limitation of Static Tools
Currently, `gaap/core/tools.py` defines tools like `read_file`, `run_command`. To add a "GitHub Issue Creator", a developer must:
1.  Write the Python function.
2.  Register it in `ToolRegistry`.
3.  Restart GAAP.

**Target:** The agent should be able to *discover* tools or *write* them on the fly.

## 2. Model Context Protocol (MCP) Integration

GAAP will become an **MCP Client**.

### 2.1 Architecture
*   **MCP Servers:** Lightweight processes that expose resources (data) and prompts/tools (functions).
    *   *Example:* `sqlite-mcp-server`, `github-mcp-server`, `browser-automation-mcp-server`.
*   **GAAP Client:** Connects to these servers over stdio or SSE (Server-Sent Events).

### 2.2 New Component: `MCPToolAdapter`

```python
class MCPToolAdapter:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.connections = {}

    async def connect_server(self, name: str, command: str):
        """Connects to an MCP server (e.g., 'npx -y @modelcontextprotocol/server-filesystem')."""
        client = await MCPClient.connect(command)
        
        # Discover capabilities
        tools = await client.list_tools()
        
        # Register in GAAP Registry
        for tool in tools:
            self.registry.register(
                name=f"{name}_{tool.name}",
                description=tool.description,
                parameters=tool.inputSchema,
                func=lambda **kwargs: client.call_tool(tool.name, kwargs)
            )
```

## 3. Dynamic Tool Synthesis (Self-Coding Tools)

The agent should be able to create its own tools for repetitive tasks.

### 3.1 Workflow
1.  **Trigger:** Agent notices it is running the same complex `run_shell_command` sequence 5 times.
2.  **Synthesis:** Agent proposes: "I should make a tool for this."
3.  **Implementation:**
    *   Agent writes `gaap/custom_tools/auto_deploy.py`.
    *   Agent generates a `ToolDefinition` JSON.
4.  **Hot-Load:** `ToolRegistry` watches the `custom_tools/` folder and dynamically loads the new Python module securely.

### 3.2 Safety Checks
*   Synthesized tools must pass the **Static Analysis Validator** (no `eval`, no network calls to unknown IPs) before being loaded.

## 4. Proposed Directory Structure

```
gaap/
├── tools/
│   ├── base.py            # Existing static tools
│   ├── mcp_client.py      # New MCP integration
│   ├── synthesizer.py     # Dynamic tool creator
│   └── dynamic/           # Folder for self-written tools
│       ├── __init__.py
│       └── tool_v1_backup_db.py
```

## 5. Roadmap
1.  **Month 1:** Integrate a basic MCP Client (support for Filesystem & Postgres servers).
2.  **Month 2:** Build the `ToolSynthesizer` with strict validation logic.
3.  **Month 3:** Create a library of standard MCP servers for GAAP (Git, Docker, Browser).
