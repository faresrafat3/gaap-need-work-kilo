"""
GAAP Security Module
====================

Multi-layer security for safe AI execution:

Firewall (Layer 0):
    - PromptFirewall: Input validation and sanitization
    - RiskLevel: Risk assessment (LOW, MEDIUM, HIGH, CRITICAL)
    - AuditTrail: Request logging and tracking

Sandbox (Execution):
    - DockerSandbox: Containerized code execution
    - LocalSandbox: Local execution with firejail
    - SandboxConfig: Resource limits (CPU, memory, timeout)

Pre-flight Checks:
    - PreFlightCheck: Code validation before execution
    - BANNED_IMPORTS: Dangerous module detection
    - Bandit integration for security scanning

DLP (Data Loss Prevention):
    - DLPScanner: Sensitive data detection
    - Pattern matching for PII, secrets
    - Entropy-based detection

Audit Logger:
    - AuditLogger: Persistent audit trail
    - Hash chain verification
    - Automatic rotation

Usage:
    from gaap.security import PromptFirewall, DockerSandbox, DLPScanner, AuditLogger

    firewall = PromptFirewall(strictness="high")
    result = firewall.scan(user_input)

    dlp = DLPScanner()
    safe_output = dlp.scan_and_redact(output)
"""

from typing import TYPE_CHECKING, Any, Protocol

from .audit_logger import (
    AuditLogEntry,
    AuditLogger,
    AuditLoggerConfig,
    create_audit_logger,
)
from .dlp import (
    DLPFinding,
    DLPScanner,
    DLPScanResult,
    LeakType,
    create_dlp_scanner,
)
from .firewall import AuditTrail, CapabilityManager, FirewallResult, PromptFirewall, RiskLevel
from .semantic_shield import (
    SemanticShield,
    ThreatAssessment,
    ThreatCategory,
    create_semantic_shield,
)

# =============================================================================
# Type Stubs for Optional Dependencies
# =============================================================================

if TYPE_CHECKING:
    from .sandbox import (
        DOCKER_AVAILABLE,
        DockerSandbox,
        LocalSandbox,
        SandboxConfig,
        SandboxResult,
        get_sandbox,
    )
else:

    class SandboxConfig(Protocol):
        image: str
        cpu_limit: float
        memory_limit_mb: int
        timeout_seconds: int
        max_output_bytes: int
        network_enabled: bool

    class SandboxResult(Protocol):
        success: bool
        output: str
        error: str
        exit_code: int
        execution_time_ms: float
        memory_used_mb: float
        container_id: str

    class DockerSandbox(Protocol):
        async def execute(self, code: str, language: str = "python") -> SandboxResult: ...

    class LocalSandbox(Protocol):
        async def execute(self, code: str, language: str = "python") -> SandboxResult: ...


# =============================================================================
# Imports with Graceful Degradation
# =============================================================================

try:
    from .sandbox import (
        DOCKER_AVAILABLE,
        DockerSandbox,
        LocalSandbox,
        SandboxConfig,
        SandboxResult,
        get_sandbox,
    )
except ImportError:
    DOCKER_AVAILABLE = False

from .preflight import (
    BANNED_IMPORTS,
    CheckResult,
    CheckSeverity,
    PreFlightCheck,
    PreFlightReport,
    create_preflight_check,
)

__all__ = [
    "PromptFirewall",
    "AuditTrail",
    "CapabilityManager",
    "RiskLevel",
    "FirewallResult",
    "DockerSandbox",
    "LocalSandbox",
    "SandboxConfig",
    "SandboxResult",
    "DOCKER_AVAILABLE",
    "get_sandbox",
    "PreFlightCheck",
    "PreFlightReport",
    "CheckResult",
    "CheckSeverity",
    "BANNED_IMPORTS",
    "create_preflight_check",
    "DLPScanner",
    "DLPScanResult",
    "DLPFinding",
    "LeakType",
    "create_dlp_scanner",
    "SemanticShield",
    "ThreatCategory",
    "ThreatAssessment",
    "create_semantic_shield",
    "AuditLogger",
    "AuditLoggerConfig",
    "AuditLogEntry",
    "create_audit_logger",
]
