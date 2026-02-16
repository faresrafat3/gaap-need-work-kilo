# Security
from .firewall import AuditTrail, CapabilityManager, FirewallResult, PromptFirewall, RiskLevel

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
    DockerSandbox = None  # type: ignore
    LocalSandbox = None  # type: ignore
    SandboxConfig = None  # type: ignore
    SandboxResult = None  # type: ignore
    get_sandbox = None  # type: ignore

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
]
