"""
Layer 3 Configuration - Zero-Trust Execution Settings
======================================================

Evolution 2026: Configurable execution with intelligent defaults.

Key Features:
- Execution mode selection (native, structured, legacy)
- Sandbox configuration (wasmtime, docker)
- Lesson injection settings
- Code audit configuration
- Resource limits

Usage:
    # Simple - intelligent defaults
    config = Layer3Config()

    # Secure - maximum security
    config = Layer3Config.secure()

    # Fast - minimum overhead
    config = Layer3Config.fast()
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution"""

    max_cpu_percent: float = 50.0
    max_memory_mb: int = 512
    max_execution_time_seconds: int = 30
    max_file_size_mb: int = 10
    max_network_connections: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cpu_percent": self.max_cpu_percent,
            "max_memory_mb": self.max_memory_mb,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_network_connections": self.max_network_connections,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceLimits":
        return cls(
            max_cpu_percent=data.get("max_cpu_percent", 50.0),
            max_memory_mb=data.get("max_memory_mb", 512),
            max_execution_time_seconds=data.get("max_execution_time_seconds", 30),
            max_file_size_mb=data.get("max_file_size_mb", 10),
            max_network_connections=data.get("max_network_connections", 0),
        )


@dataclass
class AuditConfig:
    """Configuration for code auditing"""

    enabled: bool = True
    tools: list[str] = field(default_factory=lambda: ["ruff", "bandit"])
    fail_on_errors: bool = True
    fail_on_warnings: bool = False
    max_issues: int = 50

    banned_imports: list[str] = field(
        default_factory=lambda: [
            "socket",
            "subprocess",
            "os.system",
            "eval",
            "exec",
            "compile",
            "__import__",
            "importlib",
            "pickle",
            "marshal",
            "shelve",
        ]
    )

    banned_functions: list[str] = field(
        default_factory=lambda: [
            "eval(",
            "exec(",
            "compile(",
            "__import__(",
            "os.system(",
            "os.popen(",
            "subprocess.call(",
            "subprocess.run(",
            "subprocess.Popen(",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tools": self.tools,
            "fail_on_errors": self.fail_on_errors,
            "fail_on_warnings": self.fail_on_warnings,
            "max_issues": self.max_issues,
            "banned_imports": self.banned_imports,
            "banned_functions": self.banned_functions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditConfig":
        return cls(
            enabled=data.get("enabled", True),
            tools=data.get("tools", ["ruff", "bandit"]),
            fail_on_errors=data.get("fail_on_errors", True),
            fail_on_warnings=data.get("fail_on_warnings", False),
            max_issues=data.get("max_issues", 50),
            banned_imports=data.get(
                "banned_imports",
                ["socket", "subprocess", "os.system", "eval", "exec"],
            ),
            banned_functions=data.get(
                "banned_functions",
                ["eval(", "exec(", "compile(", "__import__("],
            ),
        )


@dataclass
class LessonInjectionConfig:
    """Configuration for active lesson injection"""

    enabled: bool = True
    max_lessons: int = 5
    relevance_threshold: float = 0.6
    injection_position: Literal["system", "user", "prepend"] = "system"

    category_filter: list[str] = field(default_factory=list)

    include_failures_only: bool = False
    max_lesson_age_days: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_lessons": self.max_lessons,
            "relevance_threshold": self.relevance_threshold,
            "injection_position": self.injection_position,
            "category_filter": self.category_filter,
            "include_failures_only": self.include_failures_only,
            "max_lesson_age_days": self.max_lesson_age_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LessonInjectionConfig":
        return cls(
            enabled=data.get("enabled", True),
            max_lessons=data.get("max_lessons", 5),
            relevance_threshold=data.get("relevance_threshold", 0.6),
            injection_position=data.get("injection_position", "system"),
            category_filter=data.get("category_filter", []),
            include_failures_only=data.get("include_failures_only", False),
            max_lesson_age_days=data.get("max_lesson_age_days", 30),
        )


@dataclass
class Layer3Config:
    """
    Configuration for Layer 3 Execution.

    Provides intelligent defaults with full user control.

    Attributes:
        execution_mode: How tools are called
        sandbox_mode: Which sandbox to use
        network_enabled: Whether network access is allowed
        resource_limits: Resource constraints
        audit: Code audit configuration
        lesson_injection: Lesson injection settings

        quality_threshold: Minimum quality score
        enable_twin: Enable genetic twin system
        twin_for_critical_only: Only use twin for critical tasks
        max_parallel: Maximum parallel executions

        enable_tool_synthesis: Allow dynamic tool creation
        enable_sop: Enable SOP governance
        llm_temperature: Temperature for LLM calls
        fallback_enabled: Enable fallback strategies
    """

    execution_mode: Literal["auto", "native", "structured", "legacy"] = "auto"
    sandbox_mode: Literal["auto", "wasmtime", "docker", "disabled"] = "auto"
    network_enabled: bool = False

    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    audit: AuditConfig = field(default_factory=AuditConfig)
    lesson_injection: LessonInjectionConfig = field(default_factory=LessonInjectionConfig)

    quality_threshold: float = 70.0
    enable_twin: bool = True
    twin_for_critical_only: bool = True
    max_parallel: int = 10

    enable_tool_synthesis: bool = True
    enable_sop: bool = True
    llm_temperature: float = 0.3
    fallback_enabled: bool = True

    enable_streaming_audit: bool = True
    max_repetition: int = 3

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.quality_threshold <= 100.0:
            raise ValueError("quality_threshold must be between 0.0 and 100.0")

        if self.max_parallel < 1:
            raise ValueError("max_parallel must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "sandbox_mode": self.sandbox_mode,
            "network_enabled": self.network_enabled,
            "resource_limits": self.resource_limits.to_dict(),
            "audit": self.audit.to_dict(),
            "lesson_injection": self.lesson_injection.to_dict(),
            "quality_threshold": self.quality_threshold,
            "enable_twin": self.enable_twin,
            "twin_for_critical_only": self.twin_for_critical_only,
            "max_parallel": self.max_parallel,
            "enable_tool_synthesis": self.enable_tool_synthesis,
            "enable_sop": self.enable_sop,
            "llm_temperature": self.llm_temperature,
            "fallback_enabled": self.fallback_enabled,
            "enable_streaming_audit": self.enable_streaming_audit,
            "max_repetition": self.max_repetition,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Layer3Config":
        resource_limits_data = data.get("resource_limits", {})
        audit_data = data.get("audit", {})
        lesson_data = data.get("lesson_injection", {})

        return cls(
            execution_mode=data.get("execution_mode", "auto"),
            sandbox_mode=data.get("sandbox_mode", "auto"),
            network_enabled=data.get("network_enabled", False),
            resource_limits=ResourceLimits.from_dict(resource_limits_data),
            audit=AuditConfig.from_dict(audit_data),
            lesson_injection=LessonInjectionConfig.from_dict(lesson_data),
            quality_threshold=data.get("quality_threshold", 70.0),
            enable_twin=data.get("enable_twin", True),
            twin_for_critical_only=data.get("twin_for_critical_only", True),
            max_parallel=data.get("max_parallel", 10),
            enable_tool_synthesis=data.get("enable_tool_synthesis", True),
            enable_sop=data.get("enable_sop", True),
            llm_temperature=data.get("llm_temperature", 0.3),
            fallback_enabled=data.get("fallback_enabled", True),
            enable_streaming_audit=data.get("enable_streaming_audit", True),
            max_repetition=data.get("max_repetition", 3),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def secure(cls) -> "Layer3Config":
        """Preset for maximum security (strict auditing, sandboxed)"""
        return cls(
            execution_mode="native",
            sandbox_mode="docker",
            network_enabled=False,
            resource_limits=ResourceLimits(
                max_cpu_percent=30.0,
                max_memory_mb=256,
                max_execution_time_seconds=15,
                max_network_connections=0,
            ),
            audit=AuditConfig(
                enabled=True,
                tools=["ruff", "bandit", "mypy"],
                fail_on_errors=True,
                fail_on_warnings=True,
                banned_imports=[
                    "socket",
                    "subprocess",
                    "os.system",
                    "eval",
                    "exec",
                    "compile",
                    "__import__",
                    "importlib",
                    "pickle",
                    "marshal",
                    "shelve",
                    "ctypes",
                    "multiprocessing",
                ],
            ),
            lesson_injection=LessonInjectionConfig(
                enabled=True,
                max_lessons=10,
                relevance_threshold=0.5,
            ),
            enable_twin=True,
            twin_for_critical_only=False,
            enable_tool_synthesis=False,
            llm_temperature=0.1,
        )

    @classmethod
    def fast(cls) -> "Layer3Config":
        """Preset for speed (minimal overhead)"""
        return cls(
            execution_mode="auto",
            sandbox_mode="auto",
            network_enabled=False,
            resource_limits=ResourceLimits(
                max_cpu_percent=80.0,
                max_memory_mb=1024,
                max_execution_time_seconds=60,
            ),
            audit=AuditConfig(
                enabled=False,
            ),
            lesson_injection=LessonInjectionConfig(
                enabled=False,
            ),
            enable_twin=False,
            enable_tool_synthesis=True,
            llm_temperature=0.5,
        )

    @classmethod
    def balanced(cls) -> "Layer3Config":
        """Preset for balanced security and performance"""
        return cls(
            execution_mode="auto",
            sandbox_mode="auto",
            network_enabled=False,
            resource_limits=ResourceLimits(
                max_cpu_percent=50.0,
                max_memory_mb=512,
                max_execution_time_seconds=30,
            ),
            audit=AuditConfig(
                enabled=True,
                tools=["ruff", "bandit"],
                fail_on_errors=True,
                fail_on_warnings=False,
            ),
            lesson_injection=LessonInjectionConfig(
                enabled=True,
                max_lessons=5,
                relevance_threshold=0.6,
            ),
            enable_twin=True,
            twin_for_critical_only=True,
            enable_tool_synthesis=True,
            llm_temperature=0.3,
        )

    @classmethod
    def development(cls) -> "Layer3Config":
        """Preset for development (relaxed for testing)"""
        return cls(
            execution_mode="legacy",
            sandbox_mode="disabled",
            network_enabled=True,
            resource_limits=ResourceLimits(
                max_cpu_percent=100.0,
                max_memory_mb=2048,
                max_execution_time_seconds=300,
                max_network_connections=10,
            ),
            audit=AuditConfig(
                enabled=True,
                fail_on_errors=False,
                fail_on_warnings=False,
            ),
            lesson_injection=LessonInjectionConfig(
                enabled=True,
                max_lessons=3,
            ),
            enable_twin=False,
            enable_sop=False,
            llm_temperature=0.5,
        )


def create_layer3_config(
    preset: Literal["default", "secure", "fast", "balanced", "development"] = "default",
    **overrides: Any,
) -> Layer3Config:
    """
    Factory function to create Layer3Config.

    Args:
        preset: Configuration preset
        **overrides: Override specific settings

    Returns:
        Layer3Config instance

    Example:
        >>> config = create_layer3_config("secure", max_parallel=5)
    """
    if preset == "secure":
        config = Layer3Config.secure()
    elif preset == "fast":
        config = Layer3Config.fast()
    elif preset == "balanced":
        config = Layer3Config.balanced()
    elif preset == "development":
        config = Layer3Config.development()
    else:
        config = Layer3Config()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
