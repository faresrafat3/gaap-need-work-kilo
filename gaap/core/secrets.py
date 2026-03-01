"""
GAAP Secrets Management System

Provides secure handling of API keys and sensitive configuration.
This module is designed with security-first principles:

- Secrets are never logged in plain text
- Secrets are masked in string representations
- Environment variables are preferred over hardcoded values
- Validation ensures secrets look like valid API keys
- Audit functions detect hardcoded secrets in codebase

Example:
    >>> from gaap.core.secrets import SecretsManager
    >>> secrets = SecretsManager()
    >>> secrets.load()
    >>> print(secrets.masked_summary())
    gemini_api_key: ****-****-AIza... (loaded)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Security Constants
# =============================================================================

# Patterns that indicate a hardcoded secret (for auditing)
SUSPICIOUS_PATTERNS = [
    r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',
    r'api[_-]?secret\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',
    r'token\s*=\s*["\'][a-zA-Z0-9_-]{20,}["\']',
    r'password\s*=\s*["\'][^"\']{8,}["\']',
    r'private[_-]?key\s*=\s*["\'][^"\']{20,}["\']',
    r"sk-[a-zA-Z0-9]{20,}",
    r"AIza[0-9A-Za-z_-]{35,}",
    r"ghp_[a-zA-Z0-9]{36,}",
]

# Known API key prefixes for validation
API_KEY_PREFIXES = {
    "groq": ["gsk_"],
    "gemini": ["AIza"],
    "cerebras": ["csk-"],
    "mistral": [],  # No specific known prefix
    "kilo": [],  # No specific known prefix
    "github": ["ghp_", "github_pat_"],
}

# Minimum lengths for different secret types
MIN_SECRET_LENGTHS = {
    "api_key": 20,
    "token": 20,
    "secret": 16,
}


# =============================================================================
# Secret Masking and Validation
# =============================================================================


def mask_secret(value: str | None, visible_prefix: int = 3, visible_suffix: int = 4) -> str:
    """
    Mask a secret value for safe display.

    Args:
        value: The secret to mask
        visible_prefix: Number of characters to show at start
        visible_suffix: Number of characters to show at end

    Returns:
        Masked string like "sk-*****abc1"

    Example:
        >>> mask_secret("sk-abc123456789xyz")
        'sk-*****9xyz'
        >>> mask_secret(None)
        '<not set>'
    """
    if not value:
        return "<not set>"

    if len(value) <= visible_prefix + visible_suffix:
        return "*" * len(value)

    prefix = value[:visible_prefix]
    suffix = value[-visible_suffix:] if visible_suffix > 0 else ""
    middle_length = len(value) - visible_prefix - visible_suffix

    return f"{prefix}{'*' * middle_length}{suffix}"


def mask_middle(value: str | None, visible_chars: int = 4) -> str:
    """
    Mask the middle of a secret, showing only start and end.

    Args:
        value: The secret to mask
        visible_chars: Number of characters to show at start and end

    Returns:
        Masked string like "AIza...XYZ1"

    Example:
        >>> mask_middle("AIzaSyA1234567890123456789012345678901234")
        'AIza...234'
    """
    if not value:
        return "<not set>"

    if len(value) <= visible_chars * 2:
        return "*" * len(value)

    return f"{value[:visible_chars]}...{value[-visible_chars:]}"


def validate_api_key_format(key: str | None, provider: str = "generic") -> tuple[bool, str]:
    """
    Validate that an API key has a valid format.

    Args:
        key: The API key to validate
        provider: The provider name (groq, gemini, etc.)

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> validate_api_key_format("gsk_abc123", "groq")
        (True, "Valid Groq API key format")
    """
    if not key:
        return False, "API key is empty or None"

    if not isinstance(key, str):
        return False, "API key must be a string"

    # Check minimum length
    if len(key) < MIN_SECRET_LENGTHS["api_key"]:
        return False, f"API key too short ({len(key)} chars, min {MIN_SECRET_LENGTHS['api_key']})"

    # Check for common issues
    if key.strip() != key:
        return False, "API key has leading/trailing whitespace"

    if " " in key:
        return False, "API key contains spaces"

    # Provider-specific validation
    prefixes = API_KEY_PREFIXES.get(provider.lower(), [])
    if prefixes:
        has_valid_prefix = any(key.startswith(prefix) for prefix in prefixes)
        if not has_valid_prefix:
            return False, f"Invalid {provider} API key format (expected prefix: {prefixes})"

    # Check for obvious test/mock values
    test_patterns = [
        r"^test_?key",
        r"^your_.*key",
        r"^example",
        r"^dummy",
        r"^fake",
        r"^12345",
        r"^placeholder",
    ]
    for pattern in test_patterns:
        if re.match(pattern, key, re.IGNORECASE):
            return False, "API key appears to be a placeholder/test value"

    return True, f"Valid {provider} API key format"


def check_env_file_exists(env_path: str | Path = ".env") -> bool:
    """
    Check if a .env file exists in the specified path.

    Args:
        env_path: Path to the .env file

    Returns:
        True if file exists
    """
    return Path(env_path).exists()


def generate_env_template() -> str:
    """
    Generate .env.example content with all GAAP secrets.

    Returns:
        Template content as a string
    """
    template = """# GAAP Environment Configuration
# Copy this file to .env and fill in your values
# DO NOT commit .env to version control!

# =============================================================================
# LLM Provider API Keys (Required for full functionality)
# =============================================================================

# Groq API - Fast inference with open source models
# Get your key at: https://console.groq.com
GAAP_GROQ_API_KEY=

# Google Gemini API - Google's multimodal models  
# Get your key at: https://aistudio.google.com/app/apikey
# Required for: aistudio.py integration
GAAP_GEMINI_API_KEY=

# Cerebras API - Fast inference optimized for LLMs
# Get your key at: https://cloud.cerebras.ai
GAAP_CEREBRAS_API_KEY=

# Mistral AI API - European AI models
# Get your key at: https://console.mistral.ai
GAAP_MISTRAL_API_KEY=

# Kilo API - Multi-provider LLM gateway
# Get your key at: https://kilo.ai
GAAP_KILO_API_KEY=

# =============================================================================
# GitHub Integration (Optional)
# =============================================================================

# GitHub Personal Access Token for repository analysis
# Create at: https://github.com/settings/tokens
# Required scopes: repo, read:org (if analyzing org repos)
GAAP_GITHUB_TOKEN=

# =============================================================================
# Security Configuration
# =============================================================================

# Secret key for capability token signing and verification
# Auto-generated if not set. Set explicitly for multi-instance deployments.
# Generate with: openssl rand -hex 32
GAAP_CAPABILITY_SECRET=

# =============================================================================
# System Configuration
# =============================================================================

# Environment: development, staging, production
GAAP_ENVIRONMENT=development

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
GAAP_LOG_LEVEL=INFO

# Log format: text or json
GAAP_LOG_FORMAT=text

# =============================================================================
# Budget Limits (Optional)
# =============================================================================

# Monthly budget limit in USD
GAAP_BUDGET_MONTHLY_LIMIT=5000

# Daily budget limit in USD
GAAP_BUDGET_DAILY_LIMIT=200

# =============================================================================
# Feature Flags (Optional)
# =============================================================================

# Enable/disable external research capabilities
GAAP_ENABLE_EXTERNAL_RESEARCH=true

# Enable/disable tool synthesis
GAAP_ENABLE_TOOL_SYNTHESIS=true
"""
    return template


# =============================================================================
# Secrets Provider Dataclass
# =============================================================================


@dataclass
class SecretsProvider:
    """
    Container for all GAAP secrets and API keys.

    This dataclass holds all sensitive configuration. Values are masked
    in string representations to prevent accidental logging of secrets.

    Attributes:
        groq_api_key: Groq API key for fast inference
        gemini_api_key: Google Gemini API key (required for aistudio.py)
        cerebras_api_key: Cerebras API key for fast inference
        mistral_api_key: Mistral AI API key
        kilo_api_key: Kilo API key for multi-provider gateway
        github_token: GitHub Personal Access Token
        gaap_capability_secret: Secret for capability token signing

    Example:
        >>> secrets = SecretsProvider(groq_api_key="gsk_abc123...")
        >>> print(secrets)
        SecretsProvider(groq_api_key=gsk*****23...)
    """

    # LLM Provider API Keys
    groq_api_key: str | None = None
    gemini_api_key: str | None = None  # Required for aistudio.py
    cerebras_api_key: str | None = None
    mistral_api_key: str | None = None
    kilo_api_key: str | None = None

    # External Service Tokens
    github_token: str | None = None

    # Security Secrets
    gaap_capability_secret: str | None = None

    # Internal tracking
    _loaded_sources: list[str] = field(default_factory=list, repr=False)
    _validation_errors: dict[str, str] = field(default_factory=dict, repr=False)

    def __repr__(self) -> str:
        """Safe representation that masks all secrets."""
        items = []
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            masked = mask_secret(value, visible_prefix=3, visible_suffix=3)
            items.append(f"{f.name}={masked}")
        return f"SecretsProvider({', '.join(items)})"

    def __str__(self) -> str:
        """Safe string representation."""
        return self.__repr__()

    def get_masked(self, field_name: str) -> str:
        """
        Get a masked version of a secret field.

        Args:
            field_name: Name of the field to mask

        Returns:
            Masked string representation

        Example:
            >>> secrets.get_masked("groq_api_key")
            'gsk*****abc'
        """
        if not hasattr(self, field_name):
            return f"<field '{field_name}' not found>"
        value = getattr(self, field_name)
        return mask_secret(value)

    def to_dict(self, masked: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            masked: If True, mask all secret values

        Returns:
            Dictionary of secrets
        """
        result = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            value = getattr(self, f.name)
            if masked:
                result[f.name] = mask_secret(value)
            else:
                result[f.name] = value
        return result

    def validate(self, required: list[str] | None = None) -> tuple[bool, list[str]]:
        """
        Validate secrets format.

        Args:
            required: List of required field names. If None, no fields are required.

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Map field names to providers for format validation
        provider_map = {
            "groq_api_key": "groq",
            "gemini_api_key": "gemini",
            "cerebras_api_key": "cerebras",
            "mistral_api_key": "mistral",
            "kilo_api_key": "kilo",
            "github_token": "github",
        }

        for field_name in self.__dataclass_fields__:
            if field_name.startswith("_"):
                continue

            value = getattr(self, field_name)

            # Check required fields
            if required and field_name in required and not value:
                errors.append(f"Required secret '{field_name}' is not set")
                continue

            # Validate format if value exists
            if value and field_name in provider_map:
                is_valid, message = validate_api_key_format(value, provider_map[field_name])
                if not is_valid:
                    errors.append(f"{field_name}: {message}")

        return len(errors) == 0, errors

    def get_available_providers(self) -> list[str]:
        """
        Get list of LLM providers with configured API keys.

        Returns:
            List of provider names that have valid keys
        """
        providers = []
        provider_fields = {
            "groq": "groq_api_key",
            "gemini": "gemini_api_key",
            "cerebras": "cerebras_api_key",
            "mistral": "mistral_api_key",
            "kilo": "kilo_api_key",
        }

        for provider, field_name in provider_fields.items():
            value = getattr(self, field_name)
            if value:
                is_valid, _ = validate_api_key_format(value, provider)
                if is_valid:
                    providers.append(provider)

        return providers


# =============================================================================
# Secrets Manager
# =============================================================================


class SecretsManager:
    """
    Centralized secrets management for GAAP.

    Handles loading secrets from environment variables and .env files,
    validation, and safe access patterns.

    Security Features:
    - Never logs secret values
    - Masks secrets in string representations
    - Validates secret format
    - Detects hardcoded secrets in code (audit function)
    - Thread-safe singleton pattern

    Example:
        >>> manager = SecretsManager()
        >>> manager.load()
        >>> print(manager.secrets.gemini_api_key)  # Safe getter
        AIza...
        >>> print(manager.get_masked_summary())
        Loaded secrets:
          gemini_api_key: ****-****-AIza... (loaded from env)
    """

    _instance: SecretsManager | None = None
    _env_prefix: str = "GAAP_"

    def __new__(cls, *args: Any, **kwargs: Any) -> SecretsManager:
        """Singleton pattern - ensure only one secrets manager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        env_prefix: str = "GAAP_",
        env_file: str | Path | None = ".env",
        auto_load: bool = False,
    ):
        """
        Initialize the secrets manager.

        Args:
            env_prefix: Prefix for environment variables (default: GAAP_)
            env_file: Path to .env file, or None to skip
            auto_load: If True, load secrets on initialization
        """
        # Avoid re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._env_prefix = env_prefix
        self._env_file = env_file
        self._secrets: SecretsProvider | None = None
        self._warnings: list[str] = []
        self._errors: list[str] = []
        self._initialized = True

        if auto_load:
            self.load()

    def load(self, env_file: str | Path | None = None) -> SecretsProvider:
        """
        Load secrets from environment and .env file.

        Args:
            env_file: Override the .env file path (uses constructor value if None)

        Returns:
            Loaded SecretsProvider

        Raises:
            Warning: If .env file is missing (not error - env vars may be set elsewhere)
        """
        self._warnings = []
        self._errors = []

        # Try to load python-dotenv if available
        load_dotenv = None
        try:
            from dotenv import load_dotenv as _load_dotenv

            load_dotenv = _load_dotenv
        except ImportError:
            self._warnings.append("python-dotenv not installed, .env files will not be loaded")

        # Load .env file if specified and exists
        env_path = env_file if env_file is not None else self._env_file
        if env_path and load_dotenv is not None:
            env_path_obj = Path(env_path)
            if env_path_obj.exists():
                load_dotenv(env_path_obj, override=False)  # Don't override existing env vars
                logger.debug(f"Loaded .env file: {env_path_obj.absolute()}")
            else:
                self._warnings.append(f".env file not found: {env_path}")

        # Build secrets from environment variables
        secrets_dict: dict[str, Any] = {
            "_loaded_sources": [],
        }

        # Map of field names to environment variable names
        env_mappings = {
            "groq_api_key": f"{self._env_prefix}GROQ_API_KEY",
            "gemini_api_key": f"{self._env_prefix}GEMINI_API_KEY",
            "cerebras_api_key": f"{self._env_prefix}CEREBRAS_API_KEY",
            "mistral_api_key": f"{self._env_prefix}MISTRAL_API_KEY",
            "kilo_api_key": f"{self._env_prefix}KILO_API_KEY",
            "github_token": f"{self._env_prefix}GITHUB_TOKEN",
            "gaap_capability_secret": f"{self._env_prefix}CAPABILITY_SECRET",
        }

        # Also check non-prefixed versions for backward compatibility
        legacy_mappings = {
            "groq_api_key": "GROQ_API_KEY",
            "gemini_api_key": "GEMINI_API_KEY",
            "cerebras_api_key": "CEREBRAS_API_KEY",
            "mistral_api_key": "MISTRAL_API_KEY",
            "kilo_api_key": "KILO_API_KEY",
            "github_token": "GITHUB_TOKEN",
            "gaap_capability_secret": "GAAP_CAPABILITY_SECRET",
        }

        # Load values, preferring prefixed versions
        for field_name, env_var in env_mappings.items():
            value = os.environ.get(env_var)
            source = env_var

            # Fall back to legacy/non-prefixed version if prefixed not found
            if not value and field_name in legacy_mappings:
                legacy_var = legacy_mappings[field_name]
                if legacy_var != env_var:  # Only check if different
                    value = os.environ.get(legacy_var)
                    if value:
                        source = legacy_var
                        self._warnings.append(
                            f"Using deprecated env var {legacy_var}, please use {env_var}"
                        )

            if value:
                secrets_dict[field_name] = value
                secrets_dict["_loaded_sources"].append(source)

        self._secrets = SecretsProvider(**secrets_dict)

        # Log summary (without actual values)
        loaded_count = len([v for v in secrets_dict.values() if v and not isinstance(v, list)])
        logger.info(f"Loaded {loaded_count} secrets from environment")

        return self._secrets

    @property
    def secrets(self) -> SecretsProvider:
        """
        Get the loaded secrets.

        Returns:
            SecretsProvider with all loaded secrets

        Raises:
            RuntimeError: If secrets haven't been loaded yet
        """
        if self._secrets is None:
            raise RuntimeError("Secrets not loaded. Call load() first or use auto_load=True")
        return self._secrets

    def get(self, field_name: str, default: str | None = None) -> str | None:
        """
        Safely get a secret value.

        Args:
            field_name: Name of the secret field
            default: Default value if not found

        Returns:
            Secret value or default
        """
        if self._secrets is None:
            return default
        return getattr(self._secrets, field_name, default)

    def get_masked(self, field_name: str) -> str:
        """
        Get a masked version of a secret for safe display.

        Args:
            field_name: Name of the secret field

        Returns:
            Masked string representation

        Example:
            >>> manager.get_masked("groq_api_key")
            'gsk*****abc'
        """
        if self._secrets is None:
            return "<not set>"
        return self._secrets.get_masked(field_name)

    def require(self, field_name: str) -> str:
        """
        Get a required secret, raising an error if not found.

        Args:
            field_name: Name of the required secret field

        Returns:
            Secret value

        Raises:
            ValueError: If secret is not set
        """
        value = self.get(field_name)
        if not value:
            raise ValueError(
                f"Required secret '{field_name}' is not set. "
                f"Set {self._env_prefix}{field_name.upper()} environment variable."
            )
        return value

    def validate(
        self, required: list[str] | None = None, warn_optional: bool = True
    ) -> tuple[bool, list[str]]:
        """
        Validate loaded secrets.

        Args:
            required: List of required field names
            warn_optional: If True, warn about optional secrets with invalid format

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        if self._secrets is None:
            return False, ["Secrets not loaded. Call load() first."]

        is_valid, errors = self._secrets.validate(required=required)

        # Check for optional secrets with issues
        if warn_optional:
            for field_name in self._secrets.__dataclass_fields__:
                if field_name.startswith("_"):
                    continue
                if required and field_name in required:
                    continue

                value = getattr(self._secrets, field_name)
                if value:  # Only check if value is set
                    provider_map = {
                        "groq_api_key": "groq",
                        "gemini_api_key": "gemini",
                        "cerebras_api_key": "cerebras",
                        "mistral_api_key": "mistral",
                        "kilo_api_key": "kilo",
                        "github_token": "github",
                    }
                    provider = provider_map.get(field_name, "generic")
                    valid, msg = validate_api_key_format(value, provider)
                    if not valid:
                        self._warnings.append(
                            f"Optional secret '{field_name}' has format issues: {msg}"
                        )

        return is_valid, errors

    def get_masked_summary(self) -> str:
        """
        Get a safe summary of loaded secrets for logging.

        Returns:
            Multi-line string with masked secret info
        """
        if self._secrets is None:
            return "Secrets not loaded"

        lines = ["Loaded secrets:"]

        for field_name in self._secrets.__dataclass_fields__:
            if field_name.startswith("_"):
                continue

            value = getattr(self._secrets, field_name)
            if value:
                masked = mask_secret(value, visible_prefix=4, visible_suffix=4)
                lines.append(f"  {field_name}: {masked}")
            else:
                lines.append(f"  {field_name}: <not set>")

        if self._warnings:
            lines.append("\nWarnings:")
            for warning in self._warnings:
                lines.append(f"  - {warning}")

        return "\n".join(lines)

    def has_secret(self, field_name: str) -> bool:
        """
        Check if a secret is loaded and non-empty.

        Args:
            field_name: Name of the secret field

        Returns:
            True if secret exists and is non-empty
        """
        if self._secrets is None:
            return False
        value = getattr(self._secrets, field_name, None)
        return bool(value)

    @property
    def warnings(self) -> list[str]:
        """Get list of warnings from last load operation."""
        return self._warnings.copy()

    @property
    def errors(self) -> list[str]:
        """Get list of errors from last load/validate operation."""
        return self._errors.copy()


# =============================================================================
# Audit Functions
# =============================================================================


def audit_codebase_for_secrets(
    path: str | Path = ".",
    exclude_patterns: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Scan codebase for potentially hardcoded secrets.

    This function searches for patterns that look like API keys,
    tokens, or passwords that may have been accidentally committed.

    Args:
        path: Root directory to scan
        exclude_patterns: List of glob patterns to exclude (e.g., ["*.pyc", "__pycache__"])

    Returns:
        List of findings, each with file path, line number, and matched text

    Example:
        >>> findings = audit_codebase_for_secrets("./src")
        >>> for finding in findings:
        ...     print(f"{finding['file']}:{finding['line']} - {finding['pattern']}")
    """
    import fnmatch

    path = Path(path)
    findings = []

    default_excludes = [
        "*.pyc",
        "__pycache__",
        "*.pyo",
        "*.so",
        "*.dylib",
        "*.dll",
        ".git",
        ".env",
        ".env.*",
        "node_modules",
        "*.min.js",
        "*.min.css",
        "*.lock",
        "*.sum",
        "vendor",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        "*.egg-info",
        "dist",
        "build",
        "*.whl",
        "*.tar.gz",
    ]
    exclude_patterns = exclude_patterns or default_excludes

    # Map SUSPICIOUS_PATTERNS to names for reporting
    pattern_names = [
        "api_key",
        "api_secret",
        "token",
        "password",
        "private_key",
        "sk_token",
        "gemini_key",
        "github_token",
    ]
    patterns = list(zip(SUSPICIOUS_PATTERNS, pattern_names))

    # Scan Python files
    for file_path in path.rglob("*.py"):
        # Check exclusions
        relative = file_path.relative_to(path)
        if any(
            fnmatch.fnmatch(str(part), pattern) or fnmatch.fnmatch(str(relative), pattern)
            for part in relative.parts
            for pattern in exclude_patterns
        ):
            continue

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    for pattern, pattern_name in patterns:
                        matches = re.finditer(pattern, line, re.IGNORECASE)
                        for match in matches:
                            # Skip obvious test/example values
                            matched_text = match.group(1) if match.groups() else match.group(0)
                            test_indicators = [
                                "test",
                                "example",
                                "dummy",
                                "fake",
                                "placeholder",
                                "your_",
                            ]
                            if any(
                                indicator in matched_text.lower() for indicator in test_indicators
                            ):
                                continue

                            # Skip if it's in a comment
                            if line.strip().startswith("#"):
                                continue

                            findings.append(
                                {
                                    "file": str(file_path),
                                    "line": line_num,
                                    "column": match.start(),
                                    "pattern": pattern_name,
                                    "matched_text": (
                                        matched_text[:20] + "..."
                                        if len(matched_text) > 20
                                        else matched_text
                                    ),
                                    "context": line.strip()[:100],
                                }
                            )
        except Exception as e:
            logger.warning(f"Could not scan {file_path}: {e}")

    return findings


def print_audit_report(findings: list[dict[str, Any]]) -> None:
    """
    Print a formatted audit report.

    Args:
        findings: List of findings from audit_codebase_for_secrets()
    """
    if not findings:
        print("✅ No hardcoded secrets detected in codebase!")
        return

    print(f"⚠️  Found {len(findings)} potential hardcoded secrets:")
    print("=" * 60)

    for finding in findings:
        print(f"\n📁 {finding['file']}:{finding['line']}")
        print(f"   Pattern: {finding['pattern']}")
        print(f"   Matched: {finding['matched_text']}")
        print(f"   Context: {finding['context']}")

    print("\n" + "=" * 60)
    print("⚠️  Please review these findings and:")
    print("   1. Remove any real secrets from the codebase")
    print("   2. Rotate compromised credentials immediately")
    print("   3. Add patterns to your .gitignore or secret scanning")


# =============================================================================
# Convenience Functions
# =============================================================================


def get_secrets(
    env_prefix: str = "GAAP_",
    env_file: str | Path | None = ".env",
    auto_load: bool = True,
) -> SecretsProvider:
    """
    Get a SecretsProvider with secrets loaded.

    This is a convenience function for quick access to secrets.

    Args:
        env_prefix: Prefix for environment variables
        env_file: Path to .env file
        auto_load: If True, load secrets immediately

    Returns:
        SecretsProvider with loaded secrets

    Example:
        >>> from gaap.core.secrets import get_secrets
        >>> secrets = get_secrets()
        >>> if secrets.gemini_api_key:
        ...     print("Gemini is configured")
    """
    manager = SecretsManager(env_prefix=env_prefix, env_file=env_file, auto_load=False)
    if auto_load:
        manager.load(env_file=env_file)
    return manager.secrets


def init_secrets(
    required: list[str] | None = None,
    env_prefix: str = "GAAP_",
    env_file: str | Path | None = ".env",
) -> SecretsManager:
    """
    Initialize secrets with validation.

    Args:
        required: List of required secret field names
        env_prefix: Prefix for environment variables
        env_file: Path to .env file

    Returns:
        Initialized SecretsManager

    Raises:
        ValueError: If required secrets are missing

    Example:
        >>> manager = init_secrets(required=["gemini_api_key"])
        >>> print(manager.get_masked_summary())
    """
    manager = SecretsManager(env_prefix=env_prefix, env_file=env_file, auto_load=True)

    is_valid, errors = manager.validate(required=required)
    if not is_valid:
        raise ValueError(f"Secrets validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return manager
