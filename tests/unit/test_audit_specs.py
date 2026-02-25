"""
Tests for Audit Specs (37-43)
"""

import tempfile

import pytest

NETWORKX_AVAILABLE = True
try:
    import networkx  # noqa: F401
except ImportError:
    NETWORKX_AVAILABLE = False


class TestDLPScanner:
    def test_scan_clean_text(self):
        from gaap.security.dlp import DLPScanner

        scanner = DLPScanner()
        result = scanner.scan("This is a normal text without secrets")

        assert result.is_safe is True
        assert len(result.findings) == 0

    def test_scan_email(self):
        from gaap.security.dlp import DLPScanner, LeakType

        scanner = DLPScanner()
        result = scanner.scan("Contact me at user@example.com")

        assert result.is_safe is False
        assert any(f.leak_type == LeakType.EMAIL for f in result.findings)

    def test_scan_github_token(self):
        from gaap.security.dlp import DLPScanner, LeakType

        scanner = DLPScanner()
        result = scanner.scan("My token is ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789")

        assert result.is_safe is False
        assert any(f.leak_type == LeakType.GITHUB_TOKEN for f in result.findings)

    def test_entropy_detection(self):
        from gaap.security.dlp import DLPScanner, LeakType

        scanner = DLPScanner(entropy_threshold=3.5)
        high_entropy = "aB3kL9mN2pQ7rS4tU1vW5xY8zA0bC6dEfGhIjKl"
        result = scanner.scan(f"Token: {high_entropy}")

        assert len(result.findings) > 0
        assert any(f.leak_type == LeakType.HIGH_ENTROPY for f in result.findings)


class TestAuditLogger:
    def test_log_entry(self):
        from gaap.security.audit_logger import AuditLogger, AuditLoggerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditLoggerConfig(log_dir=tmpdir, rotation_size=1000)
            logger = AuditLogger(config)

            entry = logger.log("api_call", "agent_1", "/users", "success")

            assert entry.entry_hash != ""
            assert entry.action == "api_call"

    def test_hash_chain(self):
        from gaap.security.audit_logger import AuditLogger, AuditLoggerConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = AuditLoggerConfig(log_dir=tmpdir, auto_flush=False)
            logger = AuditLogger(config)

            entry1 = logger.log("action1", "agent1", "res1", "ok")
            entry2 = logger.log("action2", "agent1", "res2", "ok")

            assert entry2.previous_hash == entry1.entry_hash


class TestPromptFirewallScanOutput:
    def test_scan_output_clean(self):
        from gaap.security.firewall import PromptFirewall

        firewall = PromptFirewall()
        result = firewall.scan_output("This is a safe response")

        assert result.is_safe is True

    def test_scan_output_with_email(self):
        from gaap.security.firewall import PromptFirewall

        firewall = PromptFirewall()
        result = firewall.scan_output("The user email is test@example.com")

        assert result.is_safe is False
        assert "[REDACTED_EMAIL]" in result.sanitized_input


class TestASTGuard:
    def test_clean_code(self):
        from gaap.validators.ast_guard import ASTGuard

        guard = ASTGuard()
        code = "def add(a, b):\n    return a + b"
        result = guard.scan(code)

        assert result.is_safe is True

    def test_dangerous_eval(self):
        from gaap.validators.ast_guard import ASTGuard

        guard = ASTGuard()
        code = "def run(user_input):\n    return eval(user_input)"
        result = guard.scan(code)

        assert result.is_safe is False
        assert any(i.issue_type.name == "DANGEROUS_FUNCTION" for i in result.issues)

    def test_shell_injection(self):
        from gaap.validators.ast_guard import ASTGuard

        guard = ASTGuard()
        code = "import subprocess\nsubprocess.run(cmd, shell=True)"
        result = guard.scan(code)

        assert any("shell=True" in i.message for i in result.issues)


class TestPerformanceValidator:
    def test_simple_function(self):
        from gaap.validators.performance import PerformanceValidator

        validator = PerformanceValidator()
        code = "def simple():\n    return 1"
        report = validator.validate(code)

        assert report.is_acceptable is True

    def test_complex_function_detection(self):
        from gaap.validators.performance import PerformanceValidator

        validator = PerformanceValidator()
        code = """
def complex_func(a, b, c, d, e):
    if a:
        if b:
            for i in range(10):
                if c:
                    while d:
                        if e:
                            return 1
    return 0
"""
        report = validator.validate(code)

        assert report.max_complexity > 5


class TestSmartChunker:
    def test_chunk_simple_function(self):
        from gaap.context.smart_chunking import SmartChunker

        chunker = SmartChunker()
        code = 'def hello():\n    print("Hello")'
        chunks = chunker.chunk(code, "test.py")

        assert len(chunks) > 0
        assert any(c.name == "hello" for c in chunks)


class TestSemanticIndex:
    @pytest.mark.asyncio
    async def test_index_code(self):
        from gaap.context.semantic_index import SemanticIndex

        index = SemanticIndex()
        entry_id = await index.index_code(
            "def hello(): pass",
            "test.py",
            "function",
            "hello",
        )

        assert entry_id != ""


class TestPricingTable:
    def test_get_pricing(self):
        from gaap.routing.pricing_table import get_pricing

        pricing = get_pricing("gpt-4o")

        assert pricing is not None
        assert pricing.provider == "openai"

    def test_calculate_cost(self):
        from gaap.routing.pricing_table import estimate_cost

        cost = estimate_cost("gpt-4o", 1000, 500)

        assert cost is not None
        assert cost > 0


class TestAsyncSessionManager:
    def test_session_creation(self):
        from gaap.providers.async_session import AsyncSessionManager

        session = AsyncSessionManager(impersonate="chrome", timeout=60.0)

        assert session.impersonate == "chrome"
        assert session.timeout == 60.0

    def test_sse_event(self):
        from gaap.providers.async_session import SSEEvent

        event = SSEEvent(event="message", data="test data")

        assert event.event == "message"
        assert event.data == "test data"
        assert event.is_close() is False

        close_event = SSEEvent(event="close", data="")
        assert close_event.is_close() is True


class TestNativeStreamer:
    def test_stream_config(self):
        from gaap.providers.streaming import StreamConfig, StreamProtocol

        config = StreamConfig(protocol=StreamProtocol.SSE)

        assert config.protocol == StreamProtocol.SSE
        assert config.max_response_bytes == 512 * 1024

    def test_token_chunk(self):
        from gaap.providers.streaming import TokenChunk

        chunk = TokenChunk(content="Hello", is_final=False, token_count=1)

        assert chunk.content == "Hello"
        assert chunk.is_final is False
        assert chunk.token_count == 1

    def test_connect_rpc_parser(self):
        from gaap.providers.streaming import ConnectRPCParser

        parser = ConnectRPCParser()
        envelope = b'\x00\x00\x00\x00\x10{"test": "data"}'
        envelopes = parser.parse_envelopes(envelope)

        assert len(envelopes) == 1


class TestPromptCache:
    def test_cache_config(self):
        from gaap.providers.prompt_caching import CacheConfig

        config = CacheConfig(enabled=True, min_tokens=1024)

        assert config.enabled is True
        assert config.min_tokens == 1024

    def test_optimize_anthropic(self):
        from gaap.providers.prompt_caching import PromptCache

        cache = PromptCache()
        messages = [
            {"role": "system", "content": "You are a helpful assistant. " * 200},
            {"role": "user", "content": "Hello"},
        ]

        optimized = cache.optimize(messages, "anthropic")

        assert len(optimized) == 2
        assert "cache_control" in optimized[0]

    def test_cache_stats(self):
        from gaap.providers.prompt_caching import PromptCache

        cache = PromptCache()
        stats = cache.get_stats()

        assert "hits" in stats
        assert "misses" in stats
        assert "tokens_saved" in stats


class TestToolRegistry:
    def test_register_tool(self):
        from gaap.providers.tool_calling import ToolRegistry, ToolDefinition, ParameterSchema

        registry = ToolRegistry()
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            parameters={
                "location": ParameterSchema(type="string", description="City name"),
            },
        )

        registry.register(tool)

        assert "get_weather" in registry.list_tools()

    def test_tool_schema_conversion(self):
        from gaap.providers.tool_calling import ToolDefinition, ParameterSchema

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            parameters={
                "input": ParameterSchema(type="string", description="Input value"),
            },
        )

        openai_schema = tool.to_openai_schema()
        anthropic_schema = tool.to_anthropic_schema()
        gemini_schema = tool.to_gemini_schema()

        assert openai_schema["type"] == "function"
        assert anthropic_schema["name"] == "test_tool"
        assert gemini_schema["name"] == "test_tool"

    def test_tool_call_parsing(self):
        from gaap.providers.tool_calling import ToolCall

        openai_data = {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "test",
                "arguments": '{"arg1": "value1"}',
            },
        }

        tool_call = ToolCall.from_openai(openai_data)

        assert tool_call.id == "call_123"
        assert tool_call.name == "test"
        assert tool_call.arguments == {"arg1": "value1"}

    def test_tool_decorator(self):
        from gaap.providers.tool_calling import tool

        @tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert "a" in add.parameters
        assert "b" in add.parameters


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
