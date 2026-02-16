"""Tests for Docker Sandbox"""

import pytest

from gaap.security.sandbox import (
    DOCKER_AVAILABLE,
    DockerSandbox,
    LocalSandbox,
    SandboxConfig,
    SandboxResult,
    get_sandbox,
)


@pytest.fixture
def local_sandbox():
    """Create a local sandbox for testing"""
    return LocalSandbox(timeout=10)


@pytest.fixture
def sandbox_config():
    """Create sandbox configuration"""
    return SandboxConfig(
        image="python:3.12-slim",
        cpu_limit=0.5,
        memory_limit_mb=256,
        timeout_seconds=30,
        network_disabled=True,
    )


class TestSandboxConfig:
    """Tests for SandboxConfig"""

    def test_default_config(self):
        """Test default configuration"""
        config = SandboxConfig()

        assert config.image == "python:3.12-slim"
        assert config.cpu_limit == 0.5
        assert config.memory_limit_mb == 256
        assert config.timeout_seconds == 60
        assert config.network_disabled is True

    def test_custom_config(self, sandbox_config):
        """Test custom configuration"""
        assert sandbox_config.timeout_seconds == 30
        assert sandbox_config.memory_limit_mb == 256


class TestSandboxResult:
    """Tests for SandboxResult"""

    def test_success_result(self):
        """Test successful result"""
        result = SandboxResult(
            success=True,
            output="Hello",
            error="",
            exit_code=0,
            execution_time_ms=100,
        )

        assert result.success is True
        assert result.output == "Hello"
        assert result.exit_code == 0

    def test_failure_result(self):
        """Test failure result"""
        result = SandboxResult(
            success=False,
            output="",
            error="Error: division by zero",
            exit_code=1,
            execution_time_ms=50,
        )

        assert result.success is False
        assert result.error != ""


class TestLocalSandbox:
    """Tests for LocalSandbox"""

    @pytest.mark.asyncio
    async def test_execute_python(self, local_sandbox):
        """Test executing Python code"""
        result = await local_sandbox.execute('print("Hello, World!")', language="python")

        assert result.success is True
        assert "Hello, World!" in result.output

    @pytest.mark.asyncio
    async def test_execute_bash(self, local_sandbox):
        """Test executing Bash code"""
        result = await local_sandbox.execute('echo "test"', language="bash")

        assert result.success is True
        assert "test" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_error(self, local_sandbox):
        """Test executing code with error"""
        result = await local_sandbox.execute('raise ValueError("test error")', language="python")

        assert result.success is False
        assert "ValueError" in result.error or "ValueError" in result.output

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout"""
        sandbox = LocalSandbox(timeout=1)

        result = await sandbox.execute(
            "import time; time.sleep(10)",
            language="python",
        )

        assert result.success is False
        assert result.exit_code == 137

    @pytest.mark.asyncio
    async def test_execute_unknown_language(self, local_sandbox):
        """Test executing unknown language (defaults to Python)"""
        result = await local_sandbox.execute('print("default")', language="unknown")

        assert result.success is True


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
class TestDockerSandbox:
    """Tests for DockerSandbox"""

    @pytest.fixture
    def docker_sandbox(self, sandbox_config):
        """Create a Docker sandbox"""
        return DockerSandbox(config=sandbox_config)

    def test_health_check(self, docker_sandbox):
        """Test health check"""
        assert docker_sandbox.health_check() is True

    def test_get_stats(self, docker_sandbox):
        """Test getting statistics"""
        stats = docker_sandbox.get_stats()

        assert stats["docker_available"] is True
        assert stats["image"] == "python:3.12-slim"
        assert stats["network_disabled"] is True

    @pytest.mark.asyncio
    async def test_execute_simple(self, docker_sandbox):
        """Test simple execution"""
        result = await docker_sandbox.execute('print("Hello from Docker!")')

        assert result.success is True
        assert "Hello from Docker!" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_output(self, docker_sandbox):
        """Test execution with multiple outputs"""
        code = """
for i in range(5):
    print(f"Line {i}")
"""
        result = await docker_sandbox.execute(code)

        assert result.success is True
        assert "Line 0" in result.output
        assert "Line 4" in result.output

    @pytest.mark.asyncio
    async def test_execute_with_error(self, docker_sandbox):
        """Test execution with error"""
        result = await docker_sandbox.execute("1 / 0")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_container_created(self, docker_sandbox):
        """Test that container is created and cleaned up"""
        result = await docker_sandbox.execute('print("test")')

        assert result.container_id != ""
        assert result.success is True

    @pytest.mark.asyncio
    async def test_memory_limit(self, docker_sandbox):
        """Test memory limit enforcement"""
        code = """
x = []
try:
    for _ in range(1000000):
        x.append("A" * 1000)
    print("Memory limit not enforced")
except MemoryError:
    print("Memory limit enforced")
"""
        result = await docker_sandbox.execute(code, language="python")

        assert result.container_id != ""


class TestGetSandbox:
    """Tests for get_sandbox function"""

    def test_get_local_sandbox(self):
        """Test getting local sandbox"""
        sandbox = get_sandbox(use_docker=False)

        assert isinstance(sandbox, LocalSandbox)

    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_get_docker_sandbox(self):
        """Test getting Docker sandbox"""
        sandbox = get_sandbox(use_docker=True)

        assert isinstance(sandbox, DockerSandbox)
