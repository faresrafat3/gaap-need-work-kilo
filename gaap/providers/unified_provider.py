"""
Unified Provider - Primary + Fallback Chain
=============================================

Combines g4f (free web chat) and API providers into a single
interface with automatic fallback.

Strategy:
  Primary:   gemini-2.5-flash via g4f (free, 1M context)
  Fallback1: gemini-2.5-pro via g4f (free, 1M context, slower)
  Fallback2: gemini-2.5-flash via Gemini API (7 keys, 35 RPM, stable)

Rate Limits:
  g4f: ~5 RPM (shared across models)
  Gemini API: 5 RPM × 7 keys = 35 RPM total

All models: 1M token context window
"""

import asyncio
import base64
import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# =============================================================================
# Configuration
# =============================================================================


class BackendType(Enum):
    G4F = "g4f"  # g4f default routing
    G4F_PROVIDER = "g4f_provider"  # g4f with specific provider (LMArena, OpenaiChat, etc.)
    OPENAI_COMPAT = "openai_compatible"
    WEBCHAT = "webchat"  # Browser-authenticated web chat (GLM, Kimi, DeepSeek)


@dataclass
class ModelSlot:
    """A single model slot in the fallback chain."""

    name: str  # Display name
    model_id: str  # Model ID for the API
    backend: BackendType  # How to call it
    base_url: str = ""  # API base URL (for OpenAI-compat)
    g4f_provider: str = ""  # g4f Provider class name (e.g. "LMArena", "OpenaiChat")
    api_keys: list[str] = field(default_factory=list)
    rpm_per_key: int = 5  # Rate limit per key
    context_window: int = 1_000_000  # Max context tokens
    priority: int = 100  # Higher = preferred
    enabled: bool = True

    # Runtime state
    _current_key_idx: int = 0
    _call_count: int = 0
    _success_count: int = 0
    _error_count: int = 0
    _total_latency_ms: float = 0
    _last_call_time: float = 0
    _consecutive_errors: int = 0
    _cooldown_until: float = 0
    _last_error_type: str = ""

    @property
    def total_rpm(self) -> int:
        if self.backend == BackendType.G4F:
            return self.rpm_per_key  # g4f has global limit
        return self.rpm_per_key * max(len(self.api_keys), 1)

    @property
    def min_delay(self) -> float:
        """Minimum seconds between requests to stay within RPM."""
        return 60.0 / max(self.total_rpm, 1)

    @property
    def avg_latency_ms(self) -> float:
        if self._success_count == 0:
            return 0
        return self._total_latency_ms / self._success_count

    @property
    def success_rate(self) -> float:
        if self._call_count == 0:
            return 1.0
        return self._success_count / self._call_count

    def next_api_key(self) -> str:
        """Round-robin key selection."""
        if not self.api_keys:
            return ""
        key = self.api_keys[self._current_key_idx % len(self.api_keys)]
        self._current_key_idx += 1
        return key

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    @property
    def cooldown_remaining(self) -> int:
        if not self.in_cooldown:
            return 0
        return int(max(0, self._cooldown_until - time.time()))


# =============================================================================
# Default Fallback Chain
# =============================================================================

# Security-first default: no hardcoded keys in source code.
GEMINI_API_KEYS: list[str] = []


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


def _split_env_list(var_name: str) -> list[str]:
    value = os.getenv(var_name, "").strip()
    if not value:
        value = _read_var_from_env_file(var_name)
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _refresh_lmarena_models() -> None:
    """Force LMArena to refresh its model list from arena.ai (requires curl_cffi)."""
    try:
        from g4f.Provider import LMArena

        LMArena._models_loaded = False
        models = LMArena.get_models(timeout=30)
        if models:
            print(f"    🔄 LMArena: refreshed model list ({len(models)} models)")
    except Exception:
        pass  # curl_cffi not installed or network issue – fall back to static list


# =============================================================================
# LMArena Cache / Auth Utilities
# =============================================================================

LMARENA_CACHE_PATH: Path | None = None


def _get_lmarena_cache_path() -> Path:
    """Get the LMArena auth cache file path."""
    global LMARENA_CACHE_PATH
    if LMARENA_CACHE_PATH is not None:
        return LMARENA_CACHE_PATH
    try:
        from g4f.cookies import get_cookies_dir

        LMARENA_CACHE_PATH = Path(get_cookies_dir()) / "auth_LMArena.json"
    except ImportError:
        LMARENA_CACHE_PATH = Path.home() / ".config" / "g4f" / "cookies" / "auth_LMArena.json"
    return LMARENA_CACHE_PATH


def check_lmarena_auth() -> dict[str, Any]:
    """
    Check the LMArena auth cache status.

    Returns dict with:
      - valid: bool (token exists and not expired)
      - expires_in_sec: int (seconds until expiry, negative = expired)
      - is_anonymous: bool
      - cache_file: str (path to cache file)
      - message: str (human-readable status)
    """
    cache_file = _get_lmarena_cache_path()
    result = {
        "valid": False,
        "expires_in_sec": 0,
        "is_anonymous": True,
        "cache_file": str(cache_file),
        "message": "",
    }

    if not cache_file.exists():
        result["message"] = "No cache file found — browser auth needed"
        return result

    try:
        with cache_file.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        result["message"] = "Cache file corrupted"
        return result

    cookies = data.get("cookies", {})
    auth_token = cookies.get("arena-auth-prod-v1", "")

    if not auth_token:
        result["message"] = "No auth token in cache — browser auth needed"
        return result

    # Decode base64 token
    if auth_token.startswith("base64-"):
        raw = auth_token[7:]
        raw += "=" * (-len(raw) % 4)  # fix padding
        try:
            token_data = json.loads(base64.b64decode(raw).decode())
            expires_at = token_data.get("expires_at", 0)
            now = time.time()
            remaining = int(expires_at - now)
            result["expires_in_sec"] = remaining
            result["is_anonymous"] = token_data.get("user", {}).get("is_anonymous", True)

            if remaining > 0:
                result["valid"] = True
                mins = remaining // 60
                result["message"] = f"Valid — {mins}m remaining"
            else:
                result["message"] = f"Expired {abs(remaining)//60}m ago — needs refresh"
        except Exception:
            result["message"] = "Cannot decode auth token"
    else:
        # Non-base64 token – assume valid if present
        result["valid"] = True
        result["message"] = "Token present (format unknown)"

    return result


def warmup_lmarena_auth(force: bool = False) -> dict[str, Any]:
    """
    Warm up LMArena auth by triggering browser automation.

    If force=True or token is expired/missing, opens real Chrome browser
    so the user can help solve captcha. The browser window will be visible.

    Returns check_lmarena_auth() result after warmup.
    """
    status = check_lmarena_auth()

    if status["valid"] and not force:
        print(f"  ✅ LMArena auth is valid ({status['message']})")
        return status

    print(f"  🔐 LMArena auth: {status['message']}")
    print("  🌐 Opening browser for authentication...")
    print("  ⚠️  If captcha appears, please solve it in the browser window!")
    print("  ⏳ Waiting for auth (timeout: 2 minutes)...")

    try:
        # Trigger g4f's nodriver auth flow
        from g4f.Provider import LMArena

        # Run the async auth in a sync context
        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()

        if loop and loop.is_running():
            # Already in async context – create task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                args, grecaptcha = pool.submit(
                    lambda: asyncio.run(LMArena.get_args_from_nodriver(proxy=None, force=True))
                ).result(timeout=180)
        else:
            args, grecaptcha = asyncio.run(LMArena.get_args_from_nodriver(proxy=None, force=True))

        print("  ✅ Authentication successful!")
        status = check_lmarena_auth()
        print(f"  📋 Status: {status['message']}")
        return status

    except Exception as e:
        print(f"  ❌ Auth failed: {e}")
        status = check_lmarena_auth()
        return status


def invalidate_lmarena_cache() -> bool:
    """Delete the LMArena cache file to force re-auth."""
    cache_file = _get_lmarena_cache_path()
    if cache_file.exists():
        cache_file.unlink()
        print(f"  🗑️  Deleted: {cache_file}")
        return True
    print("  ℹ️  No cache file to delete")
    return False


def _apply_profile_priorities(chain: list[ModelSlot], profile: str) -> list[ModelSlot]:
    """Adjust priorities by profile: premium | quality | balanced | throughput."""
    profile = (profile or "quality").lower().strip()

    name_to_priority_balanced = {
        "gemini-2.5-flash (g4f)": 100,
        "gemini-2.5-pro (g4f)": 90,
        "gemini-2.5-flash (Gemini API)": 80,
        "gpt-4o-mini (GitHub Models)": 70,
        "llama3.3-70b (Cerebras)": 68,
        "llama-3.3-70b-versatile (Groq)": 67,
        "mistral-large-latest (Mistral)": 66,
        # WebChat Direct providers
        "GLM-5 (Direct)": 64,
        "Kimi K2.5 (Direct)": 63,
        "DeepSeek V3.2 (Direct)": 62,
        "llama-3.3-70b-instruct:free (OpenRouter)": 60,
        # LMArena premium - lower in balanced
        "claude-opus-4.6-thinking (LMArena)": 59,
        "claude-opus-4.6 (LMArena)": 58,
        "grok-4.1-thinking (LMArena)": 57,
        "claude-opus-4.5 (LMArena)": 56,
        "grok-4.1 (LMArena)": 55,
        "gpt-5.1-high (LMArena)": 54,
        "glm-5 (LMArena)": 53,
        "ernie-5.0 (LMArena)": 52,
        "claude-sonnet-4.5 (LMArena)": 51,
        "claude-opus-4.1 (LMArena)": 50,
        "glm-4.7 (LMArena)": 49,
        "kimi-k2.5 (LMArena)": 48,
        "gpt-5.2-high (LMArena)": 47,
        "gpt-5-high (LMArena)": 46,
        "qwen3-max (LMArena)": 45,
        "kimi-k2-thinking (LMArena)": 44,
        "deepseek-v3.2 (LMArena)": 43,
        "deepseek-v3.2-thinking (LMArena)": 42,
        "mistral-large-3 (LMArena)": 41,
    }

    name_to_priority_quality = {
        "gemini-2.5-pro (g4f)": 100,
        "gemini-2.5-flash (Gemini API)": 95,
        # WebChat Direct - high quality with native features
        "GLM-5 (Direct)": 93,
        "Kimi K2.5 (Direct)": 92,
        "DeepSeek V3.2 (Direct)": 91,
        "gemini-2.5-flash (g4f)": 90,
        # LMArena premium - high priority in quality (by arena rank)
        "claude-opus-4.6-thinking (LMArena)": 89,
        "claude-opus-4.6 (LMArena)": 88,
        "grok-4.1-thinking (LMArena)": 87,
        "claude-opus-4.5 (LMArena)": 86,
        "grok-4.1 (LMArena)": 85,
        "gpt-5.1-high (LMArena)": 84,
        "glm-5 (LMArena)": 83,
        "ernie-5.0 (LMArena)": 82,
        "claude-sonnet-4.5 (LMArena)": 81,
        "claude-opus-4.1 (LMArena)": 80,
        "glm-4.7 (LMArena)": 79,
        "kimi-k2.5 (LMArena)": 78,
        "gpt-5.2-high (LMArena)": 77,
        "gpt-5-high (LMArena)": 76,
        "qwen3-max (LMArena)": 75,
        "kimi-k2-thinking (LMArena)": 74,
        "deepseek-v3.2 (LMArena)": 73,
        "deepseek-v3.2-thinking (LMArena)": 72,
        "mistral-large-3 (LMArena)": 71,
        "gpt-4o-mini (GitHub Models)": 65,
        "mistral-large-latest (Mistral)": 63,
        "llama3.3-70b (Cerebras)": 61,
        "llama-3.3-70b-versatile (Groq)": 59,
        "llama-3.3-70b-instruct:free (OpenRouter)": 55,
    }

    name_to_priority_throughput = {
        "gemini-2.5-flash (Gemini API)": 100,
        "mistral-large-latest (Mistral)": 95,
        "llama3.3-70b (Cerebras)": 94,
        "llama-3.3-70b-versatile (Groq)": 93,
        "gpt-4o-mini (GitHub Models)": 85,
        "gemini-2.5-flash (g4f)": 70,
        "gemini-2.5-pro (g4f)": 65,
        # WebChat Direct - moderate throughput (browser-auth based)
        "GLM-5 (Direct)": 62,
        "Kimi K2.5 (Direct)": 61,
        "DeepSeek V3.2 (Direct)": 59,
        "llama-3.3-70b-instruct:free (OpenRouter)": 60,
        # LMArena premium - lowest in throughput (slow browser automation)
        "claude-opus-4.6-thinking (LMArena)": 43,
        "claude-opus-4.6 (LMArena)": 42,
        "grok-4.1-thinking (LMArena)": 41,
        "claude-opus-4.5 (LMArena)": 40,
        "grok-4.1 (LMArena)": 39,
        "gpt-5.1-high (LMArena)": 38,
        "glm-5 (LMArena)": 37,
        "ernie-5.0 (LMArena)": 36,
        "claude-sonnet-4.5 (LMArena)": 35,
        "claude-opus-4.1 (LMArena)": 34,
        "glm-4.7 (LMArena)": 33,
        "kimi-k2.5 (LMArena)": 32,
        "gpt-5.2-high (LMArena)": 31,
        "gpt-5-high (LMArena)": 30,
        "qwen3-max (LMArena)": 29,
        "kimi-k2-thinking (LMArena)": 28,
        "deepseek-v3.2 (LMArena)": 27,
        "deepseek-v3.2-thinking (LMArena)": 26,
        "mistral-large-3 (LMArena)": 25,
    }

    # Premium: LMArena top models FIRST (by arena rank), then free chain
    name_to_priority_premium = {
        # WebChat Direct - native features (search, tools)
        "GLM-5 (Direct)": 235,
        "Kimi K2.5 (Direct)": 233,
        "DeepSeek V3.2 (Direct)": 231,
        "claude-opus-4.6-thinking (LMArena)": 230,
        "claude-opus-4.6 (LMArena)": 225,
        "grok-4.1-thinking (LMArena)": 220,
        "claude-opus-4.5 (LMArena)": 215,
        "grok-4.1 (LMArena)": 210,
        "gpt-5.1-high (LMArena)": 205,
        "glm-5 (LMArena)": 200,
        "ernie-5.0 (LMArena)": 195,
        "claude-sonnet-4.5 (LMArena)": 190,
        "claude-opus-4.1 (LMArena)": 185,
        "glm-4.7 (LMArena)": 180,
        "kimi-k2.5 (LMArena)": 175,
        "gpt-5.2-high (LMArena)": 170,
        "gpt-5-high (LMArena)": 165,
        "qwen3-max (LMArena)": 160,
        "kimi-k2-thinking (LMArena)": 155,
        "deepseek-v3.2 (LMArena)": 150,
        "deepseek-v3.2-thinking (LMArena)": 145,
        "mistral-large-3 (LMArena)": 140,
        # Free chain as fallback
        "gemini-2.5-pro (g4f)": 100,
        "gemini-2.5-flash (Gemini API)": 95,
        "gemini-2.5-flash (g4f)": 90,
        "gpt-4o-mini (GitHub Models)": 75,
        "mistral-large-latest (Mistral)": 70,
        "llama3.3-70b (Cerebras)": 68,
        "llama-3.3-70b-versatile (Groq)": 67,
        "llama-3.3-70b-instruct:free (OpenRouter)": 60,
    }

    table = name_to_priority_balanced
    if profile == "quality":
        table = name_to_priority_quality
    elif profile == "throughput":
        table = name_to_priority_throughput
    elif profile == "premium":
        table = name_to_priority_premium

    for slot in chain:
        if slot.name in table:
            slot.priority = table[slot.name]

    return sorted(chain, key=lambda slot: slot.priority, reverse=True)


def build_default_chain(profile: str = "quality") -> list[ModelSlot]:
    """
    Build fallback chain dynamically from available env vars.

    Priority:
      1. Kimi K2.5 (WebChat Direct) - Primary
      2. DeepSeek V3.2 (WebChat Direct) - Fallback
      3. GLM-5 (WebChat Direct) - Fallback
      4. Gemini API (Official) - Stable Fallback
    """
    chain: list[ModelSlot] = []

    # 1. Kimi K2.5 (Primary)
    kimi_direct_enabled = (
        os.getenv("GAAP_KIMI_DIRECT", "").strip()
        or _read_var_from_env_file("GAAP_KIMI_DIRECT")
        or "1"  # Default to enabled
    )
    if kimi_direct_enabled == "1":
        chain.append(
            ModelSlot(
                name="Kimi K2.5 (Direct)",
                model_id="kimi",
                backend=BackendType.WEBCHAT,
                g4f_provider="kimi",
                rpm_per_key=60,  # WebChat supports higher RPM
                context_window=128_000,
                priority=200,  # Highest priority
            )
        )

    # 2. DeepSeek V3.2
    deepseek_direct_enabled = (
        os.getenv("GAAP_DEEPSEEK_DIRECT", "").strip()
        or _read_var_from_env_file("GAAP_DEEPSEEK_DIRECT")
        or "1"
    )
    if deepseek_direct_enabled == "1":
        chain.append(
            ModelSlot(
                name="DeepSeek V3.2 (Direct)",
                model_id="deepseek",
                backend=BackendType.WEBCHAT,
                g4f_provider="deepseek",
                rpm_per_key=60,
                context_window=128_000,
                priority=190,
            )
        )

    # 3. GLM-5
    glm_direct_enabled = (
        os.getenv("GAAP_GLM_DIRECT", "").strip()
        or _read_var_from_env_file("GAAP_GLM_DIRECT")
        or "1"
    )
    if glm_direct_enabled == "1":
        chain.append(
            ModelSlot(
                name="GLM-5 (Direct)",
                model_id="GLM-5",
                backend=BackendType.WEBCHAT,
                g4f_provider="glm",
                rpm_per_key=60,
                context_window=128_000,
                priority=180,
            )
        )

    # 4. Gemini API (Stable Fallback)
    gemini_keys = _split_env_list("GEMINI_API_KEYS") or GEMINI_API_KEYS
    if gemini_keys:
        chain.append(
            ModelSlot(
                name="gemini-2.5-flash (Gemini API)",
                model_id="gemini-2.5-flash",
                backend=BackendType.OPENAI_COMPAT,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_keys=gemini_keys,
                rpm_per_key=5,
                context_window=1_000_000,
                priority=100,
            )
        )

    return _apply_profile_priorities(chain, profile)


def get_onboarding_requirements() -> list[dict[str, str]]:
    """Return registration/setup requirements for free provider paths."""
    return [
        {
            "source": "g4f (no auth)",
            "needed": "Nothing required",
            "env": "none",
            "notes": "Works immediately, but strict shared limits (~5 RPM).",
        },
        {
            "source": "LMArena Premium (no auth!)",
            "needed": "Nothing required - zendriver + platformdirs packages",
            "env": "GAAP_ENABLE_LMARENA (default: 1)",
            "notes": "284 top models FREE: Claude Opus 4.6, Gemini 3 Pro, GPT-5.2 High, GLM-5, Grok 4.1. Browser automation (~3 RPM).",
        },
        {
            "source": "Gemini API Free",
            "needed": "Google AI Studio account + API keys",
            "env": "GEMINI_API_KEYS",
            "notes": "Recommended for 1M context and better stability than g4f (supports .gaap_env).",
        },
        {
            "source": "GitHub Models",
            "needed": "GitHub account (token)",
            "env": "GITHUB_MODELS_TOKEN",
            "notes": "Useful backup path for modern compact models.",
        },
        {
            "source": "Cerebras / Groq / Mistral",
            "needed": "Free account(s) + API key(s)",
            "env": "CEREBRAS_API_KEYS / GROQ_API_KEYS / MISTRAL_API_KEYS",
            "notes": "High RPM, good for throughput; context usually lower than Gemini.",
        },
        {
            "source": "OpenRouter free models",
            "needed": "OpenRouter account + key",
            "env": "OPENROUTER_API_KEYS",
            "notes": "Model availability varies; can require credit in some periods.",
        },
        {
            "source": "OpenaiChat (cookies)",
            "needed": "ChatGPT account + browser login",
            "env": "GAAP_OPENAI_CHAT=1",
            "notes": "GPT-5.2, GPT-5.1, GPT-5 + thinking variants. Needs browser cookies.",
        },
        {
            "source": "Grok (cookies)",
            "needed": "X/Twitter account + browser login",
            "env": "GAAP_GROK_CHAT=1",
            "notes": "Grok-4, Grok-4-heavy. Needs X account cookies.",
        },
        {
            "source": "GithubCopilot (auth)",
            "needed": "GitHub Copilot subscription",
            "env": "GAAP_GITHUB_COPILOT=1",
            "notes": "GPT-5, Claude models via Copilot. Needs GitHub CLI auth.",
        },
        {
            "source": "GLM-5 Direct (webchat)",
            "needed": "chat.z.ai account + browser login",
            "env": "GAAP_GLM_DIRECT=1",
            "notes": "GLM-5, GLM-4.7 direct via chat.z.ai. Native search/tools/thinking. Needs browser login.",
        },
        {
            "source": "Kimi K2.5 Direct (webchat)",
            "needed": "kimi.com account + browser login",
            "env": "GAAP_KIMI_DIRECT=1",
            "notes": "Kimi K2.5 Thinking direct via kimi.com. Native search/tools. Needs browser login.",
        },
        {
            "source": "DeepSeek V3.2 Direct (webchat)",
            "needed": "chat.deepseek.com account + browser login + Node.js",
            "env": "GAAP_DEEPSEEK_DIRECT=1",
            "notes": "DeepSeek V3.2 direct via chat.deepseek.com. PoW anti-bot (custom Keccak). Needs browser login + Node.js for PoW solver.",
        },
    ]


# =============================================================================
# Unified Provider
# =============================================================================


class UnifiedProvider:
    """
    Primary + Fallback provider with rate limit awareness.

    Usage:
        provider = UnifiedProvider()
        text, model, latency = provider.call("Your prompt here")

    Fallback Logic:
        1. Try primary (gemini-2.5-flash g4f)
        2. If fails → try gemini-2.5-pro g4f
        3. If fails → try gemini-2.5-flash Gemini API
        4. If all fail → raise error

    Rate Limit Logic:
        - Tracks time between calls per slot
        - Enforces minimum delay based on RPM
        - If a slot hits rate limit, auto-fallback to next
    """

    def __init__(
        self,
        chain: list[ModelSlot] = None,
        profile: str = "quality",
        min_delay: float = 0.0,  # Override minimum delay (0 = auto)
        verbose: bool = True,
        cooldowns: dict[str, int] | None = None,
    ):
        self.profile = profile
        self.chain = chain or build_default_chain(profile=profile)

        self.min_delay_override = min_delay
        self.verbose = verbose
        self._g4f_client = None
        self._openai_clients: dict[str, Any] = {}
        self.cooldowns = cooldowns or {
            "rate_limit": 120,
            "auth": 300,
            "timeout": 45,
            "network": 30,
            "provider": 60,
            "model_unavailable": 3600,  # 1 hour - model won't become available quickly
            "unknown": 20,
        }

        # Ensure sorted by priority (highest first)
        self.chain.sort(key=lambda s: s.priority, reverse=True)

    @property
    def primary(self) -> ModelSlot:
        return self.chain[0]

    @property
    def total_rpm(self) -> int:
        """Total capacity across all slots."""
        return sum(s.total_rpm for s in self.chain if s.enabled)

    def _get_g4f_client(self):
        if self._g4f_client is None:
            from g4f.client import Client as G4FClient

            self._g4f_client = G4FClient()
        return self._g4f_client

    def _resolve_g4f_provider(self, provider_name: str):
        """Resolve a g4f provider class by name."""
        import g4f.Provider as gp

        cls = getattr(gp, provider_name, None)
        if cls is None:
            raise ValueError(f"Unknown g4f provider: {provider_name}")
        return cls

    def _get_openai_client(self, slot: ModelSlot):
        key = slot.next_api_key()
        cache_key = f"{slot.base_url}:{key[:8]}"
        if cache_key not in self._openai_clients:
            import openai

            self._openai_clients[cache_key] = openai.OpenAI(api_key=key, base_url=slot.base_url)
        return self._openai_clients[cache_key]

    def _wait_rate_limit(self, slot: ModelSlot):
        """Wait to stay within rate limits."""
        min_delay = self.min_delay_override or slot.min_delay
        elapsed = time.time() - slot._last_call_time
        if elapsed < min_delay:
            wait = min_delay - elapsed
            time.sleep(wait)
        slot._last_call_time = time.time()

    def _classify_error(self, error: Exception) -> str:
        message = str(error).lower()
        if "429" in message or "rate limit" in message or "quota" in message:
            return "rate_limit"
        if (
            "api key" in message
            or "unauthorized" in message
            or "forbidden" in message
            or "401" in message
            or "403" in message
        ):
            return "auth"
        if "timeout" in message or "timed out" in message:
            return "timeout"
        if "connection" in message or "network" in message or "ssl" in message:
            return "network"
        if "provider" in message or "cookie" in message:
            return "provider"
        if (
            "private model" in message
            or "not supported" in message
            or "model" in message
            and "not found" in message
        ):
            return "model_unavailable"
        return "unknown"

    def _apply_cooldown(self, slot: ModelSlot, error: Exception):
        error_type = self._classify_error(error)
        seconds = self.cooldowns.get(error_type, self.cooldowns["unknown"])
        slot._cooldown_until = max(slot._cooldown_until, time.time() + seconds)
        slot._last_error_type = error_type
        if self.verbose:
            print(f"    🧯 Cooldown {slot.name}: {error_type} for {seconds}s")

    def _call_slot(
        self, slot: ModelSlot, prompt: str, system: str = None, timeout: int = 60
    ) -> tuple[str, float]:
        """Call a single model slot. Returns (text, latency_ms)."""
        self._wait_rate_limit(slot)

        # LMArena browser automation needs longer timeout
        effective_timeout = (
            max(timeout, 180) if slot.backend == BackendType.G4F_PROVIDER else timeout
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        if slot.backend == BackendType.G4F:
            client = self._get_g4f_client()
            response = client.chat.completions.create(
                model=slot.model_id,
                messages=messages,
                timeout=effective_timeout,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from g4f")
            content = content.strip()

        elif slot.backend == BackendType.G4F_PROVIDER:
            client = self._get_g4f_client()
            provider_cls = self._resolve_g4f_provider(slot.g4f_provider)
            response = client.chat.completions.create(
                model=slot.model_id,
                messages=messages,
                provider=provider_cls,
                timeout=effective_timeout,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError(f"Empty response from {slot.g4f_provider} (likely rate-limited)")
            content = content.strip()

        elif slot.backend == BackendType.OPENAI_COMPAT:
            client = self._get_openai_client(slot)
            response = client.chat.completions.create(
                model=slot.model_id,
                messages=messages,
                timeout=effective_timeout,
            )
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Empty response from API")
            content = content.strip()

        elif slot.backend == BackendType.WEBCHAT:
            from .webchat_providers import webchat_call

            content = webchat_call(
                provider_name=slot.g4f_provider,
                messages=messages,
                model=slot.model_id,
                timeout=effective_timeout,
            )
            if not content or not content.strip():
                raise ValueError(f"Empty response from {slot.g4f_provider} webchat")
            content = content.strip()

        latency_ms = (time.time() - start) * 1000

        # Update stats
        slot._call_count += 1
        slot._success_count += 1
        slot._total_latency_ms += latency_ms
        slot._consecutive_errors = 0

        return content, latency_ms

    def call(
        self,
        prompt: str,
        system: str = None,
        timeout: int = 60,
        max_retries_per_slot: int = 1,
    ) -> tuple[str, str, float]:
        """
        Call with automatic fallback.

        Returns: (answer_text, model_name, latency_ms)
        """
        errors = []

        for slot in self.chain:
            if not slot.enabled:
                continue

            if slot.in_cooldown:
                if self.verbose:
                    print(f"    ⏸️  Cooldown skip {slot.name} ({slot.cooldown_remaining}s left)")
                continue

            # Skip slots with too many consecutive errors
            if slot._consecutive_errors >= 5:
                if self.verbose:
                    print(f"    ⏩ Skipping {slot.name} (5+ consecutive errors)")
                continue

            for attempt in range(max_retries_per_slot + 1):
                try:
                    content, latency_ms = self._call_slot(slot, prompt, system, timeout)
                    return content, slot.name, latency_ms

                except Exception as e:
                    slot._call_count += 1
                    slot._error_count += 1
                    slot._consecutive_errors += 1
                    self._apply_cooldown(slot, e)
                    err_msg = str(e)[:60]
                    errors.append(f"{slot.name}: {err_msg}")

                    if attempt < max_retries_per_slot and not slot.in_cooldown:
                        wait = (attempt + 1) * 2
                        if self.verbose:
                            print(f"    ⚠️  {slot.name} failed ({err_msg}), retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        if self.verbose:
                            print(f"    ❌ {slot.name} failed, falling back...")
                        break

        raise Exception(f"All providers failed: {'; '.join(errors)}")

    def get_stats(self) -> str:
        """Get formatted stats for all slots."""
        lines = [
            f"{'Slot':<30} | {'Calls':>6} | {'OK':>4} | {'Err':>4} | {'Rate':>6} | {'Avg ms':>8} | {'RPM':>5} | {'LastError':<10}",
            "-" * 100,
        ]
        for slot in self.chain:
            rate = f"{slot.success_rate*100:.0f}%" if slot._call_count > 0 else "N/A"
            avg = f"{slot.avg_latency_ms:.0f}" if slot._success_count > 0 else "N/A"
            if not slot.enabled:
                status = "⏸️"
            elif slot.in_cooldown:
                status = "🧯"
            else:
                status = "✅"
            err_type = slot._last_error_type or "-"
            lines.append(
                f"{status} {slot.name:<28} | {slot._call_count:>6} | "
                f"{slot._success_count:>4} | {slot._error_count:>4} | "
                f"{rate:>6} | {avg:>8} | {slot.total_rpm:>5} | {err_type:<10}"
            )
        return "\n".join(lines)

    def get_config_summary(self) -> str:
        """Print configuration summary."""
        lines = [
            "=" * 70,
            f"🔗 Unified Provider - Fallback Chain [{self.profile}]",
            "=" * 70,
        ]
        for i, slot in enumerate(self.chain):
            role = "PRIMARY" if i == 0 else f"FALLBACK-{i}"
            status = "✅" if slot.enabled else "⏸️"
            if slot.backend == BackendType.G4F:
                backend = "g4f"
            elif slot.backend == BackendType.G4F_PROVIDER:
                backend = slot.g4f_provider[:8]
            else:
                backend = "API"
            keys = len(slot.api_keys) if slot.api_keys else 0
            lines.append(
                f"  {status} [{role:<10}] {slot.name:<35} "
                f"| {backend:<8} | {slot.total_rpm:>3} RPM | "
                f"{slot.context_window//1000}K ctx | {keys} keys"
            )
        lines.append(f"\n  📊 Total capacity: {self.total_rpm} RPM")
        lines.append(f"  ⏱️  Min delay: {self.primary.min_delay:.1f}s (primary)")
        lines.append("=" * 70)
        return "\n".join(lines)

    def get_onboarding_summary(self) -> str:
        """Human-readable checklist for required registrations/tokens."""
        reqs = get_onboarding_requirements()
        lines = [
            "=" * 70,
            "🧩 Free Access Onboarding (What you need)",
            "=" * 70,
        ]
        for item in reqs:
            configured = False
            env_key = item["env"]
            if item["source"] == "Gemini API Free":
                configured = bool(_split_env_list("GEMINI_API_KEYS"))
            elif "LMArena" in item["source"]:
                # LMArena is enabled by default (no auth needed)
                lm_flag = (
                    os.getenv("GAAP_ENABLE_LMARENA", "").strip()
                    or _read_var_from_env_file("GAAP_ENABLE_LMARENA")
                    or "1"
                )
                configured = lm_flag == "1"
            elif env_key != "none":
                first_key = env_key.split(" /")[0]
                val = os.getenv(first_key, "").strip() or _read_var_from_env_file(first_key)
                configured = bool(val)
            status = "✅" if configured or env_key == "none" else "⚪"
            lines.append(f"  {status} {item['source']}: {item['needed']} | ENV: {env_key}")
            lines.append(f"      - {item['notes']}")
        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    provider = UnifiedProvider(verbose=True, profile=os.getenv("GAAP_PROVIDER_PROFILE", "quality"))
    print(provider.get_config_summary())
    print(provider.get_onboarding_summary())

    print("\n🧪 Testing fallback chain...")
    try:
        text, model, latency = provider.call("Reply with only: OK")
        print(f"  ✅ Response: '{text}' from {model} in {latency:.0f}ms")
    except Exception as e:
        print(f"  ❌ All failed: {e}")

    print(f"\n{provider.get_stats()}")
