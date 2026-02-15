"""
Multi-Provider Configuration
============================

Comprehensive configuration for all available free-tier API providers.
Smart routing, fallback chains, and rate limit management.
"""

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class ProviderType(Enum):
    """Provider types"""
    CEREBRAS = "cerebras"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    MISTRAL_CODESTRAL = "mistral_codestral"
    HUGGINGFACE = "huggingface"
    GITHUB = "github"
    CLOUDFLARE = "cloudflare"
    VERCEL = "vercel"


def _find_env_file() -> Path | None:
    candidates = [
        Path.cwd() / ".gaap_env",
        Path(__file__).resolve().parents[2] / ".gaap_env",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _read_var_from_env_file(var_name: str) -> str:
    env_file = _find_env_file()
    if not env_file:
        return ""
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith(f"{var_name}="):
                return stripped.split("=", 1)[1].strip()
    except Exception:
        return ""
    return ""


def _env_keys(var_name: str) -> list[str]:
    value = os.getenv(var_name, "").strip() or _read_var_from_env_file(var_name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class ProviderLimits:
    """Rate limits for a provider"""
    requests_per_minute: int = 0
    requests_per_day: int = 0
    tokens_per_minute: int = 0
    tokens_per_day: int = 0
    concurrent_requests: int = 1

    # Cooldown window (seconds)
    cooldown_window: int = 60

    # Cost (for tracking, even if free)
    cost_per_1k_tokens: float = 0.0


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    name: str
    provider_type: ProviderType
    api_keys: list[str]
    base_url: str
    models: list[str]
    limits: ProviderLimits
    priority: int = 50  # Higher = better (0-100)
    enabled: bool = True
    notes: str = ""


# =============================================================================
# PROVIDER CONFIGURATIONS
# =============================================================================

CEREBRAS_CONFIG = ProviderConfig(
    name="Cerebras",
    provider_type=ProviderType.CEREBRAS,
    api_keys=_env_keys("CEREBRAS_API_KEYS"),
    base_url="https://api.cerebras.ai/v1",
    models=[
        "llama3.3-70b",  # Primary model - 70B, best accuracy
    ],
    limits=ProviderLimits(
        requests_per_minute=30,
        requests_per_day=14_400,
        tokens_per_minute=64_000,
        tokens_per_day=1_000_000,
        concurrent_requests=3,
    ),
    priority=95,  # Highest priority - best limits
    notes="Best free tier - 30 RPM × 7 keys = 210 RPM total"
)

OPENROUTER_CONFIG = ProviderConfig(
    name="OpenRouter",
    provider_type=ProviderType.OPENROUTER,
    api_keys=_env_keys("OPENROUTER_API_KEYS"),
    base_url="https://openrouter.ai/api/v1",
    models=[
        "meta-llama/llama-3.3-70b-instruct:free",  # Primary (HTTP 429 - needs credit)
    ],
    limits=ProviderLimits(
        requests_per_minute=20,
        requests_per_day=50,  # Limited!
        concurrent_requests=2,
    ),
    priority=75,
    notes="20 RPM × 7 keys = 140 RPM, but only 50 req/day per key"
)

GROQ_CONFIG = ProviderConfig(
    name="Groq",
    provider_type=ProviderType.GROQ,
    api_keys=_env_keys("GROQ_API_KEYS"),
    base_url="https://api.groq.com/openai/v1",
    models=[
        "llama-3.3-70b-versatile",  # Primary - FASTEST, 227ms avg latency
    ],
    limits=ProviderLimits(
        requests_per_minute=30,  # Varies by model
        requests_per_day=1_000,  # For 70B models
        tokens_per_minute=12_000,
        concurrent_requests=2,
    ),
    priority=85,
    notes="Very fast inference, 1000 req/day for 70B models × 7 keys"
)

GEMINI_CONFIG = ProviderConfig(
    name="Google Gemini",
    provider_type=ProviderType.GEMINI,
    api_keys=_env_keys("GEMINI_API_KEYS"),
    base_url="https://generativelanguage.googleapis.com/v1beta",
    models=[
        "gemini-2.5-flash",  # Primary - Latest flash model (quota exhausted currently)
    ],
    limits=ProviderLimits(
        requests_per_minute=5,  # Very limited!
        requests_per_day=20,
        tokens_per_minute=250_000,
        concurrent_requests=2,
    ),
    priority=40,  # Lower priority due to severe rate limits
    notes="Severe limits: 5 RPM, 20 RPD - use as fallback only"
)

MISTRAL_CONFIG = ProviderConfig(
    name="Mistral La Plateforme",
    provider_type=ProviderType.MISTRAL,
    api_keys=_env_keys("MISTRAL_API_KEYS"),
    base_url="https://api.mistral.ai/v1",
    models=[
        "mistral-large-latest",  # Primary - Best performance, 603ms avg latency
    ],
    limits=ProviderLimits(
        requests_per_minute=60,  # 1 req/sec
        tokens_per_minute=500_000,
        tokens_per_day=1_000_000_000,
        concurrent_requests=3,
    ),
    priority=70,
    notes="1 req/sec = 60 RPM, good token limits"
)

MISTRAL_CODESTRAL_CONFIG = ProviderConfig(
    name="Mistral Codestral",
    provider_type=ProviderType.MISTRAL_CODESTRAL,
    api_keys=_env_keys("MISTRAL_CODESTRAL_API_KEYS"),
    base_url="https://codestral.mistral.ai/v1",
    models=[
        "codestral-latest",  # Code-specific model (not currently working)
    ],
    limits=ProviderLimits(
        requests_per_minute=30,
        requests_per_day=2_000,
        concurrent_requests=2,
    ),
    priority=65,
    notes="30 RPM, 2000 RPD - good for code tasks"
)

GITHUB_CONFIG = ProviderConfig(
    name="GitHub Models",
    provider_type=ProviderType.GITHUB,
    api_keys=_env_keys("GITHUB_MODELS_TOKEN"),
    base_url="https://models.inference.ai.azure.com",
    models=[
        "gpt-4o-mini",  # Primary - Backup provider, 1498ms avg latency
    ],
    limits=ProviderLimits(
        requests_per_minute=15,  # Copilot Pro tier
        requests_per_day=150,
        tokens_per_minute=8_000,  # Input limit
        concurrent_requests=5,
    ),
    priority=60,
    notes="Copilot Pro: 15 RPM, 150 RPD, token limits per request"
)

CLOUDFLARE_CONFIG = ProviderConfig(
    name="Cloudflare Workers AI",
    provider_type=ProviderType.CLOUDFLARE,
    api_keys=_env_keys("CLOUDFLARE_API_KEYS"),
    base_url="https://api.cloudflare.com/client/v4/accounts/<account_id>/ai/run",
    models=[
        "@cf/meta/llama-3.1-8b-instruct",  # Primary - Smaller model (needs account ID)
    ],
    limits=ProviderLimits(
        requests_per_day=10_000,  # 10k neurons/day
        concurrent_requests=5,
    ),
    priority=55,
    notes="10,000 neurons/day limit (usage-based)"
)


# =============================================================================
# PROVIDER REGISTRY
# =============================================================================

# ✅ WORKING PROVIDERS (Tested Feb 13, 2026)
WORKING_PROVIDERS = [
    CEREBRAS_CONFIG,      # ✅ 210 RPM, 511ms avg, PRODUCTION READY
    GROQ_CONFIG,          # ✅ 210 RPM, 227ms avg, FASTEST
    MISTRAL_CONFIG,       # ✅ 60 RPM, 603ms avg, VERIFIED
    GITHUB_CONFIG,        # ✅ 15 RPM, 1498ms avg, BACKUP
]

# ❌ FAILED/UNAVAILABLE PROVIDERS (For reference only)
FAILED_PROVIDERS = [
    OPENROUTER_CONFIG,       # ❌ HTTP 429 - Needs $10 credit
    GEMINI_CONFIG,           # ❌ HTTP 429 - Quota exhaused
    MISTRAL_CODESTRAL_CONFIG,  # ❌ Wrong model configuration
    CLOUDFLARE_CONFIG,       # ❌ Needs CLOUDFLARE_ACCOUNT_ID
]

# All providers (for backwards compatibility)
ALL_PROVIDERS = WORKING_PROVIDERS + FAILED_PROVIDERS

# Sort by priority (highest first)
ALL_PROVIDERS.sort(key=lambda p: p.priority, reverse=True)


def get_provider_config(provider_type: ProviderType) -> ProviderConfig | None:
    """Get config for a specific provider type"""
    for config in ALL_PROVIDERS:
        if config.provider_type == provider_type:
            return config
    return None


def get_enabled_providers() -> list[ProviderConfig]:
    """Get all enabled providers sorted by priority"""
    return [p for p in ALL_PROVIDERS if p.enabled]


def get_total_capacity() -> dict[str, Any]:
    """Calculate total capacity across all providers"""
    total_rpm = 0
    total_rpd = 0
    total_keys = 0

    for provider in get_enabled_providers():
        num_keys = len(provider.api_keys)
        total_keys += num_keys

        # Calculate per-provider capacity
        provider_rpm = provider.limits.requests_per_minute * num_keys
        provider_rpd = provider.limits.requests_per_day * num_keys if provider.limits.requests_per_day > 0 else 999_999

        total_rpm += provider_rpm
        total_rpd = min(total_rpd + provider_rpd, 999_999)

    return {
        "total_requests_per_minute": total_rpm,
        "total_requests_per_day": total_rpd,
        "total_api_keys": total_keys,
        "num_providers": len(get_enabled_providers()),
        "providers": [
            {
                "name": p.name,
                "rpm": p.limits.requests_per_minute * len(p.api_keys),
                "rpd": p.limits.requests_per_day * len(p.api_keys) if p.limits.requests_per_day > 0 else "unlimited",
                "keys": len(p.api_keys),
                "priority": p.priority,
            }
            for p in get_enabled_providers()
        ]
    }


if __name__ == "__main__":
    # Print capacity summary

    capacity = get_total_capacity()
    print("=" * 80)
    print("MULTI-PROVIDER CAPACITY SUMMARY")
    print("=" * 80)
    print("\nTotal Capacity:")
    print(f"  - Requests per Minute: {capacity['total_requests_per_minute']:,}")
    print(f"  - Requests per Day: {capacity['total_requests_per_day']:,}")
    print(f"  - Total API Keys: {capacity['total_api_keys']}")
    print(f"  - Active Providers: {capacity['num_providers']}")
    print("\nProviders (by priority):")
    print(f"{'Name':<25} {'RPM':>10} {'RPD':>15} {'Keys':>6} {'Priority':>10}")
    print("-" * 80)
    for p in capacity['providers']:
        rpd_str = f"{p['rpd']:,}" if isinstance(p['rpd'], int) else p['rpd']
        print(f"{p['name']:<25} {p['rpm']:>10,} {rpd_str:>15} {p['keys']:>6} {p['priority']:>10}")
    print("=" * 80)
