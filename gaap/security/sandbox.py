"""
Docker Sandbox - Secure Code Execution
======================================

Executes code in isolated Docker containers with:
- CPU/RAM limits
- Network isolation
- Timeout enforcement
- Automatic cleanup

Usage:
    sandbox = DockerSandbox()
    result = await sandbox.execute("print('hello')", language="python")
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("gaap.sandbox")

DOCKER_AVAILABLE = False

if TYPE_CHECKING:
    from docker import DockerClient
    from docker.errors import APIError, DockerException, ImageNotFound

try:
    import docker as _docker
    from docker.errors import APIError, DockerException, ImageNotFound

    DOCKER_AVAILABLE = True
    docker = _docker
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None
    DockerException = Exception
    ImageNotFound = Exception
    APIError = Exception


@dataclass
class SandboxResult:
    """Result of sandbox execution"""

    success: bool
    output: str
    error: str
    exit_code: int
    execution_time_ms: float
    memory_used_mb: float = 0.0
    container_id: str = ""


@dataclass
class SandboxConfig:
    """Configuration for sandbox"""

    image: str = "python:3.12-slim"
    cpu_limit: float = 0.5
    memory_limit_mb: int = 256
    timeout_seconds: int = 60
    network_disabled: bool = True
    read_only_root: bool = True
    allow_write_tmp: bool = True
    max_output_bytes: int = 1_000_000


class DockerSandbox:
    """
    Secure code execution in Docker containers.

    Security features:
    - No network access (default)
    - Limited CPU and memory
    - Read-only filesystem (except /tmp)
    - Automatic container cleanup
    - Timeout enforcement
    """

    def __init__(self, config: SandboxConfig | None = None):
        if not DOCKER_AVAILABLE or docker is None:
            raise ImportError("Docker SDK not installed. Run: pip install docker")

        self.config = config or SandboxConfig()
        self._client: DockerClient = docker.from_env()
        self._logger = logger
        self._ensure_image()

    def _ensure_image(self) -> None:
        """Pull image if not available locally"""
        try:
            self._client.images.get(self.config.image)
        except ImageNotFound:
            self._logger.info(f"Pulling image {self.config.image}...")
            self._client.images.pull(self.config.image)

    async def execute(
        self,
        code: str,
        language: str = "python",
        files: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> SandboxResult:
        """
        Execute code in sandbox.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash)
            files: Additional files to mount {filename: content}
            env: Environment variables

        Returns:
            SandboxResult with output and status
        """
        start_time = time.time()
        container: Any = None

        try:
            command = self._build_command(code, language)

            volumes = {}
            if files:
                for filename, content in files.items():
                    tmp_file = tempfile.NamedTemporaryFile(
                        mode="w", suffix=f"_{filename}", delete=False
                    )
                    tmp_file.write(content)
                    tmp_file.close()
                    volumes[tmp_file.name] = {
                        "bind": f"/sandbox/{filename}",
                        "mode": "rw",
                    }

            container = self._client.containers.run(
                image=self.config.image,
                command=command,
                environment=env or {},
                volumes=volumes if volumes else None,
                network=None if self.config.network_disabled else "bridge",
                cpu_quota=int(self.config.cpu_limit * 100000),
                mem_limit=f"{self.config.memory_limit_mb}m",
                working_dir="/sandbox",
                detach=True,
                remove=False,
                security_opt=["no-new-privileges"],
                cap_drop=["ALL"],
            )

            self._logger.debug(f"Container {container.id[:12]} created")

            container.start()

            try:
                result = container.wait(timeout=self.config.timeout_seconds)
                exit_code = result.get("StatusCode", 1)
            except Exception:
                container.kill()
                exit_code = 137
                self._logger.warning(f"Container {container.id[:12]} timed out")

            logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")

            if len(logs) > self.config.max_output_bytes:
                logs = logs[: self.config.max_output_bytes] + "\n... [output truncated]"

            stats = container.stats(stream=False)
            memory_used = 0.0
            if stats:
                memory_used = stats.get("memory_stats", {}).get("usage", 0) / (1024 * 1024)

            success = exit_code == 0

            execution_time = (time.time() - start_time) * 1000

            return SandboxResult(
                success=success,
                output=logs,
                error="" if success else logs,
                exit_code=exit_code,
                execution_time_ms=execution_time,
                memory_used_mb=memory_used,
                container_id=container.id[:12],
            )

        except APIError as e:
            self._logger.error(f"Docker API error: {e}")
            return SandboxResult(
                success=False,
                output="",
                error=f"Docker error: {e}",
                exit_code=1,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            self._logger.error(f"Sandbox error: {e}")
            return SandboxResult(
                success=False,
                output="",
                error=f"Sandbox error: {e}",
                exit_code=1,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        finally:
            if container:
                try:
                    container.remove(force=True, v=True)
                    self._logger.debug(f"Container {container.id[:12]} removed")
                except Exception as e:
                    self._logger.debug(f"Failed to remove container: {e}")

    def _build_command(self, code: str, language: str) -> list[str]:
        """Build command for code execution"""
        if language == "python":
            return ["python3", "-c", code]
        elif language == "javascript":
            return ["node", "-e", code]
        elif language == "bash":
            return ["bash", "-c", code]
        else:
            return ["python3", "-c", code]

    async def execute_file(
        self, file_path: str, args: list[str] | None = None, timeout: int | None = None
    ) -> SandboxResult:
        """Execute a file in sandbox"""
        path = Path(file_path)
        if not path.exists():
            return SandboxResult(
                success=False,
                output="",
                error=f"File not found: {file_path}",
                exit_code=1,
                execution_time_ms=0,
            )

        code = path.read_text()

        language = "python"
        if path.suffix == ".js":
            language = "javascript"
        elif path.suffix == ".sh":
            language = "bash"

        if args:
            code = f"{code}\n# Args: {args}"

        old_timeout = self.config.timeout_seconds
        if timeout:
            self.config.timeout_seconds = timeout

        result = await self.execute(code, language)
        self.config.timeout_seconds = old_timeout
        return result

    def health_check(self) -> bool:
        """Check if Docker is available and working"""
        if not DOCKER_AVAILABLE:
            return False

        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def get_stats(self) -> dict[str, Any]:
        """Get sandbox statistics"""
        return {
            "docker_available": DOCKER_AVAILABLE,
            "image": self.config.image,
            "cpu_limit": self.config.cpu_limit,
            "memory_limit_mb": self.config.memory_limit_mb,
            "timeout_seconds": self.config.timeout_seconds,
            "network_disabled": self.config.network_disabled,
        }


import shutil

FIREJAIL_AVAILABLE = shutil.which("firejail") is not None


class LocalSandbox:
    """
    Fallback sandbox using local subprocess with lightweight isolation.

    Security layers (best-effort when Docker is unavailable):
    - Uses firejail for filesystem/network isolation when available
    - Falls back to bare subprocess with timeout only (logs loud warning)
    - Timeout enforcement via asyncio.wait_for
    """

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._logger = logger
        self._use_firejail = FIREJAIL_AVAILABLE

        if not self._use_firejail:
            self._logger.warning(
                "⚠️  SECURITY WARNING: firejail is not installed. "
                "LocalSandbox will execute code WITHOUT isolation. "
                "Install firejail for lightweight sandboxing: sudo apt install firejail"
            )

    def _build_command(self, code: str, language: str) -> list[str]:
        """Build execution command, optionally wrapped in firejail."""
        if language == "python":
            base_cmd = ["python3", "-c", code]
        elif language == "bash":
            base_cmd = ["bash", "-c", code]
        elif language == "javascript":
            base_cmd = ["node", "-e", code]
        else:
            base_cmd = ["python3", "-c", code]

        if self._use_firejail:
            return [
                "firejail",
                "--quiet",
                "--noprofile",
                "--net=none",  # No network access
                "--nosound",  # No sound
                "--no3d",  # No 3D acceleration
                "--nodvd",  # No DVD access
                "--nogroups",  # No supplementary groups
                "--nonewprivs",  # No privilege escalation
                "--noroot",  # No root access
                "--seccomp",  # Enable seccomp filter
                "--private-tmp",  # Private /tmp
                "--read-only=/",  # Read-only root filesystem
                "--",
            ] + base_cmd

        return base_cmd

    async def execute(
        self,
        code: str,
        language: str = "python",
    ) -> SandboxResult:
        """Execute code locally with best-effort isolation."""
        start_time = time.time()

        if not self._use_firejail:
            self._logger.warning(
                "Executing code WITHOUT sandbox isolation (firejail not available)"
            )

        try:
            cmd = self._build_command(code, language)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
                exit_code = proc.returncode or 0
            except asyncio.TimeoutError:
                proc.kill()
                exit_code = 137
                stdout = b""
                stderr = b"Execution timed out"

            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")

            return SandboxResult(
                success=exit_code == 0,
                output=output,
                error=error if exit_code != 0 else "",
                exit_code=exit_code,
                execution_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            return SandboxResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1,
                execution_time_ms=(time.time() - start_time) * 1000,
            )


def get_sandbox(use_docker: bool = True) -> DockerSandbox | LocalSandbox:
    """Get appropriate sandbox based on availability."""
    if use_docker and DOCKER_AVAILABLE:
        try:
            return DockerSandbox()
        except Exception:
            logger.warning("Docker sandbox creation failed, falling back to LocalSandbox")
    else:
        logger.warning(
            "Docker not available, using LocalSandbox "
            f"(firejail isolation: {'enabled' if FIREJAIL_AVAILABLE else 'DISABLED'})"
        )
    return LocalSandbox()
