"""
Smart Account Manager — Multi-Account Pool with Proactive Monitoring
=====================================================================

Manages multiple accounts per AI provider, with:
  - Rate limit tracking & proactive warnings
  - Session/chat quota monitoring
  - Auto-rotation to best available account
  - Credential storage & lifecycle management
  - Health dashboard for all accounts

Architecture:
  AccountPool
    └── AccountSlot (per account)
          ├── credentials (token, cookies, etc.)
          ├── RateLimitTracker (RPM/RPD/tokens)
          ├── SessionTracker (active sessions, message counts)
          └── HealthStatus (score, warnings, cooldown)

Usage:
  pool = AccountPool("kimi")
  pool.add_account("main", credentials={...})
  pool.add_account("alt1", credentials={...})

  # Auto-selects best account
  account = pool.best_account()

  # Record usage after each call
  pool.record_call(account.label, success=True, latency_ms=1200, tokens_used=500)

  # Check health proactively
  warnings = pool.get_warnings()
  dashboard = pool.dashboard()
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("gaap.providers.account_manager")

# =============================================================================
# Storage
# =============================================================================

ACCOUNTS_DIR = Path.home() / ".config" / "gaap" / "accounts"
USAGE_DIR = Path.home() / ".config" / "gaap" / "usage"


def _ensure_dirs() -> None:
    ACCOUNTS_DIR.mkdir(parents=True, exist_ok=True)
    USAGE_DIR.mkdir(parents=True, exist_ok=True)


import re as _re

# Known provider hard rate-limit error patterns → cooldown seconds
_HARD_RATE_LIMIT_PATTERNS = [
    # Kimi: "当前模型对话次数已达上限，3小时后恢复" (daily limit reached, 3h cooldown)
    (
        _re.compile(r"(\d+)\s*小时后恢复", _re.IGNORECASE),
        lambda m: int(m.group(1)) * 3600,
        "Kimi daily limit",
    ),
    # Kimi: "X分钟后恢复" (X minutes cooldown)
    (
        _re.compile(r"(\d+)\s*分钟后恢复", _re.IGNORECASE),
        lambda m: int(m.group(1)) * 60,
        "Kimi cooldown",
    ),
    # Generic "rate limit" with hours
    (
        _re.compile(r"rate.?limit.*?(\d+)\s*hour", _re.IGNORECASE),
        lambda m: int(m.group(1)) * 3600,
        "Rate limit",
    ),
    # REASON_RATE_LIMIT_EXCEEDED (Kimi Connect-RPC error code)
    (
        _re.compile(r"REASON_RATE_LIMIT_EXCEEDED", _re.IGNORECASE),
        lambda m: 3 * 3600,
        "Kimi rate limit (3h)",
    ),
    # 已达上限 (limit reached) without specific time → assume 3h
    (_re.compile(r"已达上限", _re.IGNORECASE), lambda m: 3 * 3600, "Provider daily limit (3h)"),
]


def detect_hard_cooldown(error_msg: str) -> tuple | None:
    """
    Check if an error message indicates a provider-imposed hard rate limit.

    Returns (cooldown_seconds, reason) if detected, None otherwise.
    """
    if not error_msg:
        return None
    for pattern, calc_seconds, reason in _HARD_RATE_LIMIT_PATTERNS:
        m = pattern.search(error_msg)
        if m:
            seconds = calc_seconds(m)
            return (seconds, reason)
    return None


# =============================================================================
# Rate Limit Tracking
# =============================================================================


class RateLimitTracker:
    """
    Tracks request rate with sliding window.

    Monitors:
      - Requests per minute (RPM)
      - Requests per hour (RPH)
      - Requests per day (RPD)
      - Token usage per day
      - Consecutive errors
    """

    def __init__(
        self,
        max_rpm: int = 5,
        max_rph: int = 100,
        max_rpd: int = 1500,
        max_tokens_per_day: int = 0,  # 0 = unlimited
        warn_threshold: float = 0.8,  # warn at 80% of limit
    ):
        self.max_rpm = max_rpm
        self.max_rph = max_rph
        self.max_rpd = max_rpd
        self.max_tokens_per_day = max_tokens_per_day
        self.warn_threshold = warn_threshold

        # Sliding window timestamps
        self._call_times: list[float] = []
        self._tokens_today: int = 0
        self._tokens_day_start: float = 0.0

        # Error tracking
        self._consecutive_errors: int = 0
        self._last_error_time: float = 0.0
        self._last_error_msg: str = ""
        self._total_errors: int = 0
        self._total_calls: int = 0
        self._total_success: int = 0
        self._total_latency_ms: float = 0.0

        # Cooldown
        self._cooldown_until: float = 0.0
        self._cooldown_reason: str = ""

    def _prune_old(self) -> None:
        """Remove timestamps older than 24h."""
        cutoff = time.time() - 86400
        self._call_times = [t for t in self._call_times if t > cutoff]

    def _reset_daily_tokens(self) -> None:
        """Reset daily token counter if new day."""
        now = time.time()
        if now - self._tokens_day_start > 86400:
            self._tokens_today = 0
            self._tokens_day_start = now

    def record_call(
        self, success: bool, latency_ms: float = 0, tokens_used: int = 0, error_msg: str = ""
    ) -> None:
        """Record a call result."""
        now = time.time()
        self._call_times.append(now)
        self._total_calls += 1

        if success:
            self._total_success += 1
            self._total_latency_ms += latency_ms
            self._consecutive_errors = 0
        else:
            self._total_errors += 1
            self._consecutive_errors += 1
            self._last_error_time = now
            self._last_error_msg = error_msg

            # Auto-cooldown on consecutive errors
            if self._consecutive_errors >= 3:
                self._cooldown_until = now + min(60 * self._consecutive_errors, 600)

        if tokens_used > 0:
            self._reset_daily_tokens()
            self._tokens_today += tokens_used

        self._prune_old()

    @property
    def rpm_current(self) -> int:
        """Current requests in the last minute."""
        cutoff = time.time() - 60
        return sum(1 for t in self._call_times if t > cutoff)

    @property
    def rph_current(self) -> int:
        """Current requests in the last hour."""
        cutoff = time.time() - 3600
        return sum(1 for t in self._call_times if t > cutoff)

    @property
    def rpd_current(self) -> int:
        """Current requests in the last 24h."""
        self._prune_old()
        return len(self._call_times)

    @property
    def rpm_remaining(self) -> int:
        return max(0, self.max_rpm - self.rpm_current)

    @property
    def rph_remaining(self) -> int:
        return max(0, self.max_rph - self.rph_current)

    @property
    def rpd_remaining(self) -> int:
        return max(0, self.max_rpd - self.rpd_current)

    @property
    def tokens_remaining(self) -> int:
        if self.max_tokens_per_day <= 0:
            return 999_999_999
        self._reset_daily_tokens()
        return max(0, self.max_tokens_per_day - self._tokens_today)

    def set_hard_cooldown(self, seconds: float, reason: str = "") -> None:
        """Set an explicit cooldown (e.g., provider-imposed 3-hour ban).

        Unlike the auto-cooldown from consecutive errors, this represents
        a hard server-side rate limit with a known duration.
        """
        self._cooldown_until = time.time() + seconds
        self._cooldown_reason = reason
        # Reset consecutive errors so we don't double-penalize
        self._consecutive_errors = 0

    @property
    def in_cooldown(self) -> bool:
        return time.time() < self._cooldown_until

    @property
    def cooldown_remaining_sec(self) -> int:
        return max(0, int(self._cooldown_until - time.time()))

    @property
    def cooldown_reason(self) -> str:
        """Why this account is in cooldown."""
        if not self.in_cooldown:
            return ""
        return getattr(self, "_cooldown_reason", "auto-cooldown from errors")

    @property
    def cooldown_expires_at(self) -> float:
        """Unix timestamp when cooldown expires (0 if not in cooldown)."""
        if not self.in_cooldown:
            return 0.0
        return self._cooldown_until

    @property
    def avg_latency_ms(self) -> float:
        if self._total_success == 0:
            return 0
        return self._total_latency_ms / self._total_success

    @property
    def success_rate(self) -> float:
        if self._total_calls == 0:
            return 1.0
        return self._total_success / self._total_calls

    @property
    def seconds_until_next_allowed(self) -> float:
        """How long to wait before next call is safe."""
        if self.in_cooldown:
            return self.cooldown_remaining_sec
        if self.rpm_current >= self.max_rpm:
            # Find oldest call in last minute, wait until it expires
            cutoff = time.time() - 60
            recent = sorted(t for t in self._call_times if t > cutoff)
            if recent:
                return max(0, recent[0] + 60 - time.time())
        return 0.0

    def can_call(self) -> tuple[bool, str]:
        """Check if a call is allowed right now. Returns (allowed, reason)."""
        if self.in_cooldown:
            reason = self.cooldown_reason or "cooldown"
            remaining = self.cooldown_remaining_sec
            mins = remaining // 60
            secs = remaining % 60
            if mins >= 60:
                hrs = mins // 60
                mins = mins % 60
                time_str = f"{hrs}h{mins}m"
            elif mins > 0:
                time_str = f"{mins}m{secs}s"
            else:
                time_str = f"{secs}s"
            return False, f"{reason} ({time_str} remaining)"
        if self._consecutive_errors >= 5:
            return False, f"Too many errors ({self._consecutive_errors} consecutive)"
        if self.rpm_current >= self.max_rpm:
            return False, f"RPM limit ({self.rpm_current}/{self.max_rpm})"
        if self.rph_current >= self.max_rph:
            return False, f"RPH limit ({self.rph_current}/{self.max_rph})"
        if self.rpd_current >= self.max_rpd:
            return False, f"RPD limit ({self.rpd_current}/{self.max_rpd})"
        if self.max_tokens_per_day > 0 and self.tokens_remaining <= 0:
            return False, "Daily token limit reached"
        return True, "OK"

    def get_warnings(self) -> list[str]:
        """Get proactive warnings about approaching limits."""
        warnings = []

        # RPM warning
        rpm_ratio = self.rpm_current / max(self.max_rpm, 1)
        if rpm_ratio >= self.warn_threshold:
            warnings.append(f"⚠️ RPM at {self.rpm_current}/{self.max_rpm} ({rpm_ratio * 100:.0f}%)")

        # RPH warning
        rph_ratio = self.rph_current / max(self.max_rph, 1)
        if rph_ratio >= self.warn_threshold:
            warnings.append(f"⚠️ RPH at {self.rph_current}/{self.max_rph} ({rph_ratio * 100:.0f}%)")

        # RPD warning
        rpd_ratio = self.rpd_current / max(self.max_rpd, 1)
        if rpd_ratio >= self.warn_threshold:
            warnings.append(f"⚠️ RPD at {self.rpd_current}/{self.max_rpd} ({rpd_ratio * 100:.0f}%)")

        # Token warning
        if self.max_tokens_per_day > 0:
            tok_ratio = self._tokens_today / self.max_tokens_per_day
            if tok_ratio >= self.warn_threshold:
                warnings.append(f"⚠️ Tokens at {self._tokens_today}/{self.max_tokens_per_day}")

        # Error warning
        if self._consecutive_errors >= 2:
            warnings.append(
                f"🔴 {self._consecutive_errors} consecutive errors: {self._last_error_msg[:50]}"
            )

        return warnings

    def health_score(self) -> float:
        """0.0 (dead) to 1.0 (perfect) health score."""
        if self.in_cooldown:
            return 0.0
        if self._consecutive_errors >= 5:
            return 0.0

        score = 1.0

        # Penalize for approaching RPM limit
        rpm_ratio = self.rpm_current / max(self.max_rpm, 1)
        score -= rpm_ratio * 0.3

        # Penalize for errors
        if self._total_calls > 0:
            error_rate = self._total_errors / self._total_calls
            score -= error_rate * 0.3

        # Penalize for consecutive errors
        score -= self._consecutive_errors * 0.1

        # Penalize for approaching daily limit
        rpd_ratio = self.rpd_current / max(self.max_rpd, 1)
        score -= rpd_ratio * 0.2

        return max(0.0, min(1.0, score))

    def to_dict(self) -> dict:
        return {
            "rpm_current": self.rpm_current,
            "rph_current": self.rph_current,
            "rpd_current": self.rpd_current,
            "rpm_remaining": self.rpm_remaining,
            "tokens_today": self._tokens_today,
            "tokens_remaining": self.tokens_remaining,
            "consecutive_errors": self._consecutive_errors,
            "total_calls": self._total_calls,
            "total_success": self._total_success,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "health_score": round(self.health_score(), 3),
            "in_cooldown": self.in_cooldown,
            "cooldown_sec": self.cooldown_remaining_sec,
        }


# =============================================================================
# Session Tracker
# =============================================================================


class SessionTracker:
    """
    Tracks chat sessions per account.

    Monitors:
      - Active session count
      - Messages per session
      - Session age
      - When to rotate sessions
    """

    def __init__(self, max_sessions: int = 10, max_messages_per_session: int = 50):
        self.max_sessions = max_sessions
        self.max_messages_per_session = max_messages_per_session
        self._sessions: dict[str, dict] = {}  # session_id → {created, messages, last_used}

    def create_session(self, session_id: str) -> None:
        """Register a new session."""
        self._sessions[session_id] = {
            "created": time.time(),
            "messages": 0,
            "last_used": time.time(),
        }
        # Evict oldest if over limit
        while len(self._sessions) > self.max_sessions:
            oldest = min(self._sessions, key=lambda k: self._sessions[k]["last_used"])
            del self._sessions[oldest]

    def record_message(self, session_id: str) -> None:
        """Record a message sent in a session."""
        if session_id in self._sessions:
            self._sessions[session_id]["messages"] += 1
            self._sessions[session_id]["last_used"] = time.time()

    def should_rotate(self, session_id: str) -> bool:
        """Check if a session should be rotated (too many messages or too old)."""
        if session_id not in self._sessions:
            return True
        s = self._sessions[session_id]
        if s["messages"] >= self.max_messages_per_session:
            return True
        created: float = s["created"]
        return time.time() - created > 7200

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def total_messages(self) -> int:
        return sum(s["messages"] for s in self._sessions.values())

    def to_dict(self) -> dict:
        return {
            "active_sessions": self.active_count,
            "total_messages": self.total_messages,
            "sessions": {
                sid: {
                    "messages": s["messages"],
                    "age_min": int((time.time() - s["created"]) / 60),
                    "should_rotate": self.should_rotate(sid),
                }
                for sid, s in self._sessions.items()
            },
        }


# =============================================================================
# Account Slot
# =============================================================================


class AccountStatus(Enum):
    ACTIVE = "active"
    COOLDOWN = "cooldown"
    EXPIRED = "expired"
    DISABLED = "disabled"
    NEEDS_LOGIN = "needs_login"


@dataclass
class AccountSlot:
    """
    A single account for a provider with full state tracking.
    """

    provider: str
    label: str  # "main", "alt1", "alt2", etc.

    # Credentials
    email: str = ""
    # token/cookies stored separately in WebChatAuth

    # Configuration
    models: list[str] = field(default_factory=list)  # which models this account can access
    max_rpm: int = 5
    max_rph: int = 100
    max_rpd: int = 1500
    max_tokens_per_day: int = 0
    max_messages_per_session: int = 50

    # Metadata
    account_type: str = "free"  # "free", "plus", "pro", "student"
    notes: str = ""
    created_at: float = 0.0

    # Runtime (not serialized)
    rate_tracker: RateLimitTracker = field(init=False, repr=False)
    session_tracker: SessionTracker = field(init=False, repr=False)
    _enabled: bool = True

    def __post_init__(self) -> None:
        self.rate_tracker = RateLimitTracker(
            max_rpm=self.max_rpm,
            max_rph=self.max_rph,
            max_rpd=self.max_rpd,
            max_tokens_per_day=self.max_tokens_per_day,
        )
        self.session_tracker = SessionTracker(
            max_messages_per_session=self.max_messages_per_session,
        )
        if self.created_at <= 0:
            self.created_at = time.time()

    @property
    def status(self) -> AccountStatus:
        """Current account status."""
        if not self._enabled:
            return AccountStatus.DISABLED
        if self.rate_tracker.in_cooldown:
            return AccountStatus.COOLDOWN
        # Check auth via webchat_providers
        try:
            from .webchat_providers import load_auth

            auth = load_auth(self.provider, self.label)
            if auth is None:
                return AccountStatus.NEEDS_LOGIN
            if auth.is_expired:
                return AccountStatus.EXPIRED
        except Exception as e:
            logger.debug(f"Account operation failed: {e}")
        return AccountStatus.ACTIVE

    @property
    def health_score(self) -> float:
        """Combined health score considering rate limits and auth."""
        base = self.rate_tracker.health_score()
        status = self.status
        if status == AccountStatus.DISABLED:
            return 0.0
        if status == AccountStatus.NEEDS_LOGIN:
            return 0.0
        if status == AccountStatus.EXPIRED:
            return 0.0
        if status == AccountStatus.COOLDOWN:
            return 0.0
        return base

    def can_call(self) -> tuple[bool, str]:
        """Check if this account can make a call."""
        status = self.status
        if status == AccountStatus.DISABLED:
            return False, "Account disabled"
        if status == AccountStatus.NEEDS_LOGIN:
            return False, "Needs browser login"
        if status == AccountStatus.EXPIRED:
            return False, "Auth expired"
        return self.rate_tracker.can_call()

    def record_call(
        self,
        success: bool,
        latency_ms: float = 0,
        tokens_used: int = 0,
        error_msg: str = "",
        session_id: str = "",
    ) -> None:
        """Record a call result."""
        self.rate_tracker.record_call(success, latency_ms, tokens_used, error_msg)
        if session_id:
            self.session_tracker.record_message(session_id)

    def get_warnings(self) -> list[str]:
        """Get all warnings for this account."""
        warnings = self.rate_tracker.get_warnings()

        # Auth expiry warning
        try:
            from .webchat_providers import load_auth

            auth = load_auth(self.provider, self.label)
            if auth:
                remaining_hours = auth.remaining_sec / 3600
                if remaining_hours < 2:
                    warnings.append(f"🔐 Auth expires in {auth.remaining_sec // 60}m!")
                elif remaining_hours < 12:
                    warnings.append(f"🔐 Auth expires in {remaining_hours:.1f}h")
        except Exception as e:
            logger.debug(f"Account operation failed: {e}")

        return warnings

    def to_config_dict(self) -> dict:
        """Serialize config (not runtime state) for storage."""
        return {
            "provider": self.provider,
            "label": self.label,
            "email": self.email,
            "models": self.models,
            "max_rpm": self.max_rpm,
            "max_rph": self.max_rph,
            "max_rpd": self.max_rpd,
            "max_tokens_per_day": self.max_tokens_per_day,
            "max_messages_per_session": self.max_messages_per_session,
            "account_type": self.account_type,
            "notes": self.notes,
            "created_at": self.created_at,
            "enabled": self._enabled,
        }

    @classmethod
    def from_config_dict(cls, d: dict) -> "AccountSlot":
        slot = cls(
            provider=d["provider"],
            label=d["label"],
            email=d.get("email", ""),
            models=d.get("models", []),
            max_rpm=d.get("max_rpm", 5),
            max_rph=d.get("max_rph", 100),
            max_rpd=d.get("max_rpd", 1500),
            max_tokens_per_day=d.get("max_tokens_per_day", 0),
            max_messages_per_session=d.get("max_messages_per_session", 50),
            account_type=d.get("account_type", "free"),
            notes=d.get("notes", ""),
            created_at=d.get("created_at", 0),
        )
        slot._enabled = d.get("enabled", True)
        return slot

    def to_status_dict(self) -> dict:
        """Full status including runtime state."""
        return {
            "provider": self.provider,
            "label": self.label,
            "email": self.email,
            "account_type": self.account_type,
            "status": self.status.value,
            "health_score": round(self.health_score, 3),
            "can_call": self.can_call()[0],
            "can_call_reason": self.can_call()[1],
            "rate_limits": self.rate_tracker.to_dict(),
            "sessions": self.session_tracker.to_dict(),
            "warnings": self.get_warnings(),
        }


# =============================================================================
# Provider Limits Database
# =============================================================================

# Known limits per provider (defaults for new accounts)
PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "glm": {
        "max_rpm": 60,
        "max_rph": 1000,
        "max_rpd": 5000,
        "max_tokens_per_day": 0,
        "max_messages_per_session": 100,
        "models": ["GLM-5", "GLM-4.7", "GLM-4.6", "GLM-4.5", "GLM-4.5V", "Z1-Rumination"],
        "auth_lifetime_hours": 24,
        "max_output_words": 500,
        "max_concurrent": 5,
        "notes": "chat.z.ai - HMAC-SHA256 signed. Token valid 24h. 60+ RPM confirmed. Slow output (94s/100w). Concurrency N5:5/5.",
    },
    "kimi": {
        "max_rpm": 60,
        "max_rph": 1000,
        "max_rpd": 5000,
        "max_tokens_per_day": 0,
        "max_messages_per_session": 50,
        "models": ["kimi-k2.5-thinking", "kimi-k2", "kimi-k2-thinking", "kimi-research"],
        "model_scenarios": {
            "kimi-k2.5-thinking": "SCENARIO_K2D5",
            "kimi-k2": "SCENARIO_K2",
            "kimi-k2-thinking": "SCENARIO_K2_THINKING",
            "kimi-research": "SCENARIO_RESEARCH",
        },
        "auth_lifetime_hours": 720,  # ~30 days
        "max_output_words": 1010,
        "max_concurrent": 3,
        "notes": "kimi.com - Connect-RPC. JWT valid ~30 days. 60+ RPM confirmed. N5 drops to 3/5.",
    },
    "deepseek": {
        "max_rpm": 60,
        "max_rph": 1000,
        "max_rpd": 5000,
        "max_tokens_per_day": 0,
        "max_messages_per_session": 50,
        "models": ["deepseek", "deepseek-thinking"],
        "auth_lifetime_hours": 720,  # ~30 days
        "max_output_words": 2731,
        "max_concurrent": 3,
        "notes": "chat.deepseek.com - PoW (Keccak/Node.js). Token valid ~30 days. 60+ RPM confirmed. Best output (2731w). N5 drops to 4/5.",
    },
    "copilot": {
        "max_rpm": 10,
        "max_rph": 100,
        "max_rpd": 1000,
        "max_tokens_per_day": 0,
        "max_messages_per_session": 100,
        "models": [
            # GPT-5 Series
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1",
            "gpt-5.2",
            # Claude 4 Series
            "claude-opus-4.6",
            "claude-sonnet-4.5",
            "claude-sonnet-4",
            "claude-haiku-4.5",
            # Gemini
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            # GPT-4 Series
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
        ],
        "auth_lifetime_hours": 8760,  # 1 year (GitHub OAuth)
        "notes": "GitHub Copilot API - OAuth device flow. Pro/Student account.",
    },
    "gemini_api": {
        "max_rpm": 5,
        "max_rph": 200,
        "max_rpd": 1500,
        "max_tokens_per_day": 0,
        "max_messages_per_session": 999,
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "auth_lifetime_hours": 999999,  # API keys don't expire
        "notes": "Google AI Studio API keys. 1M context window.",
    },
}


# =============================================================================
# Account Pool
# =============================================================================


class AccountPool:
    """
    Manages multiple accounts for a single provider.

    Features:
      - Add/remove accounts dynamically
      - Auto-select best account for next call
      - Track rate limits per account
      - Proactive warnings
      - Persistent config storage
    """

    def __init__(self, provider: str):
        self.provider = provider
        self._accounts: dict[str, AccountSlot] = {}
        self._lock = threading.Lock()
        self._config_path = ACCOUNTS_DIR / f"{provider}_pool.json"

        # Load saved config
        self._load_config()

    def _load_config(self) -> None:
        """Load account pool config from disk."""
        _ensure_dirs()
        if self._config_path.exists():
            try:
                data = json.loads(self._config_path.read_text(encoding="utf-8"))
                for acct_data in data.get("accounts", []):
                    slot = AccountSlot.from_config_dict(acct_data)
                    self._accounts[slot.label] = slot
            except (OSError, json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠️ Failed to load {self.provider} pool config: {e}")

    def _save_config(self) -> None:
        """Persist account pool config to disk."""
        _ensure_dirs()
        data = {
            "provider": self.provider,
            "updated_at": time.time(),
            "accounts": [slot.to_config_dict() for slot in self._accounts.values()],
        }
        self._config_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add_account(
        self,
        label: str = "default",
        email: str = "",
        account_type: str = "free",
        models: list[str] | None = None,
        notes: str = "",
        **overrides: Any,
    ) -> AccountSlot:
        """
        Add an account to the pool.

        Uses PROVIDER_DEFAULTS for limits, with optional overrides.
        """
        defaults = PROVIDER_DEFAULTS.get(self.provider, {})

        slot = AccountSlot(
            provider=self.provider,
            label=label,
            email=email,
            account_type=account_type,
            models=models or defaults.get("models", []),
            max_rpm=overrides.get("max_rpm", defaults.get("max_rpm", 5)),
            max_rph=overrides.get("max_rph", defaults.get("max_rph", 100)),
            max_rpd=overrides.get("max_rpd", defaults.get("max_rpd", 1500)),
            max_tokens_per_day=overrides.get(
                "max_tokens_per_day", defaults.get("max_tokens_per_day", 0)
            ),
            max_messages_per_session=overrides.get(
                "max_messages_per_session", defaults.get("max_messages_per_session", 50)
            ),
            notes=notes,
        )

        with self._lock:
            self._accounts[label] = slot
            self._save_config()

        return slot

    def remove_account(self, label: str) -> bool:
        """Remove an account from the pool."""
        with self._lock:
            if label in self._accounts:
                del self._accounts[label]
                self._save_config()
                return True
        return False

    def get_account(self, label: str) -> AccountSlot | None:
        """Get a specific account by label."""
        return self._accounts.get(label)

    @property
    def accounts(self) -> list[AccountSlot]:
        """All accounts in the pool."""
        return list(self._accounts.values())

    @property
    def active_accounts(self) -> list[AccountSlot]:
        """Accounts that can currently make calls."""
        return [a for a in self._accounts.values() if a.can_call()[0]]

    def best_account(self, model: str = "") -> AccountSlot | None:
        """
        Select the best account for next call.

        Strategy:
          1. Filter to accounts that can call
          2. If model specified, filter to accounts that support it
          3. Sort by health score (highest first)
          4. Break ties by: fewer RPM used → fewer errors → lower latency

        Returns None if no account is available.
        """
        candidates = []
        for acct in self._accounts.values():
            can, _ = acct.can_call()
            if not can:
                continue
            if model and acct.models and model not in acct.models:
                continue
            candidates.append(acct)

        if not candidates:
            return None

        # Sort by health score desc, then RPM used asc, then errors asc
        candidates.sort(
            key=lambda a: (
                -a.health_score,
                a.rate_tracker.rpm_current,
                a.rate_tracker._consecutive_errors,
                a.rate_tracker.avg_latency_ms,
            )
        )

        return candidates[0]

    def record_call(
        self,
        label: str,
        success: bool,
        latency_ms: float = 0,
        tokens_used: int = 0,
        error_msg: str = "",
        session_id: str = "",
    ) -> None:
        """Record a call result for a specific account."""
        acct = self._accounts.get(label)
        if acct:
            acct.record_call(success, latency_ms, tokens_used, error_msg, session_id)

    def get_warnings(self) -> dict[str, list[str]]:
        """Get warnings for all accounts."""
        return {
            label: acct.get_warnings()
            for label, acct in self._accounts.items()
            if acct.get_warnings()
        }

    def should_call(self, label: str = "", model: str = "") -> tuple[bool, str, str | None]:
        """
        Proactive check: should we send this call?

        Returns (should_proceed, reason, recommended_account_label).
        """
        if label:
            acct = self._accounts.get(label)
            if acct is None:
                return False, f"Account '{label}' not found", None
            can, reason = acct.can_call()
            if can:
                return True, "OK", label
            # Try fallback to another account
            best = self.best_account(model)
            if best:
                return True, f"Switched from '{label}' ({reason}) to '{best.label}'", best.label
            return False, f"No available accounts: {reason}", None

        # No specific account requested
        best = self.best_account(model)
        if best:
            return True, "OK", best.label

        # Explain why none are available
        reasons = []
        for acct in self._accounts.values():
            can, reason = acct.can_call()
            if not can:
                reasons.append(f"{acct.label}: {reason}")
        return False, f"All accounts unavailable: {'; '.join(reasons)}", None

    def wait_for_availability(self, model: str = "", timeout: float = 60) -> AccountSlot | None:
        """
        Wait until an account becomes available.

        Useful for rate-limited scenarios.
        Returns the account when available, or None on timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            best = self.best_account(model)
            if best:
                return best

            # Find minimum wait time across all accounts
            min_wait = float("inf")
            for acct in self._accounts.values():
                wait = acct.rate_tracker.seconds_until_next_allowed
                if wait < min_wait:
                    min_wait = wait

            if min_wait == float("inf") or min_wait > timeout:
                return None

            time.sleep(min(min_wait + 0.1, 5.0))

        return None

    def dashboard(self) -> str:
        """Human-readable dashboard of all accounts."""
        if not self._accounts:
            return f"  📭 {self.provider}: No accounts configured"

        lines = [
            f"{'Account':<15} | {'Status':<12} | {'Health':>6} | {'RPM':>8} | {'RPH':>8} | {'RPD':>8} | {'Errs':>5} | {'Avg ms':>7} | Notes",
            "-" * 110,
        ]

        for label, acct in sorted(self._accounts.items()):
            status = acct.status.value
            health = f"{acct.health_score:.2f}"
            rpm = f"{acct.rate_tracker.rpm_current}/{acct.max_rpm}"
            rph = f"{acct.rate_tracker.rph_current}/{acct.max_rph}"
            rpd = f"{acct.rate_tracker.rpd_current}/{acct.max_rpd}"
            errs = str(acct.rate_tracker._consecutive_errors)
            avg = (
                f"{acct.rate_tracker.avg_latency_ms:.0f}"
                if acct.rate_tracker._total_success > 0
                else "-"
            )

            icon = {
                AccountStatus.ACTIVE: "✅",
                AccountStatus.COOLDOWN: "🧯",
                AccountStatus.EXPIRED: "⏰",
                AccountStatus.DISABLED: "⏸️",
                AccountStatus.NEEDS_LOGIN: "🔐",
            }.get(acct.status, "❓")

            lines.append(
                f"{icon} {label:<13} | {status:<12} | {health:>6} | {rpm:>8} | {rph:>8} | {rpd:>8} | {errs:>5} | {avg:>7} | {acct.notes[:20]}"
            )

            # Show warnings inline
            for w in acct.get_warnings():
                lines.append(f"  └── {w}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full serializable state."""
        return {
            "provider": self.provider,
            "account_count": len(self._accounts),
            "active_count": len(self.active_accounts),
            "accounts": {label: acct.to_status_dict() for label, acct in self._accounts.items()},
            "warnings": self.get_warnings(),
        }


# =============================================================================
# Global Pool Manager
# =============================================================================


class PoolManager:
    """
    Manages AccountPools for all providers.

    Singleton pattern — one instance per process.
    """

    _instance: Optional["PoolManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._pools: dict[str, AccountPool] = {}

    @classmethod
    def instance(cls) -> "PoolManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def pool(self, provider: str) -> AccountPool:
        """Get or create an AccountPool for a provider."""
        if provider not in self._pools:
            self._pools[provider] = AccountPool(provider)
        return self._pools[provider]

    def all_pools(self) -> dict[str, AccountPool]:
        """Get all pools. Auto-loads from disk."""
        _ensure_dirs()
        for f in ACCOUNTS_DIR.iterdir():
            if f.name.endswith("_pool.json"):
                provider = f.name.replace("_pool.json", "")
                if provider not in self._pools:
                    self._pools[provider] = AccountPool(provider)
        return dict(self._pools)

    def full_dashboard(self) -> str:
        """Dashboard for all providers."""
        pools = self.all_pools()
        if not pools:
            return "📭 No accounts configured for any provider."

        lines = ["=" * 110, "🎛️  GAAP Account Manager — Full Dashboard", "=" * 110]

        for provider, pool in sorted(pools.items()):
            lines.append(f"\n📦 {provider.upper()}")
            lines.append(pool.dashboard())

        # Global warnings
        all_warnings = {}
        for provider, pool in pools.items():
            warnings = pool.get_warnings()
            if warnings:
                all_warnings[provider] = warnings

        if all_warnings:
            lines.append(f"\n{'=' * 110}")
            lines.append("⚠️  WARNINGS:")
            for provider, warnings in all_warnings.items():
                for label, warns in warnings.items():
                    for w in warns:
                        lines.append(f"  [{provider}/{label}] {w}")

        lines.append("=" * 110)
        return "\n".join(lines)

    def smart_call(
        self,
        provider: str,
        messages: list[dict[str, str]],
        model: str = "",
        preferred_account: str = "",
        timeout: int = 120,
    ) -> tuple[str, str, str, float]:
        """
        Smart call with auto-account selection and tracking.

        Returns: (response_text, model_used, account_used, latency_ms)

        Raises RuntimeError if no account available.
        """
        pool = self.pool(provider)

        # Check if we should proceed
        should, reason, acct_label = pool.should_call(label=preferred_account, model=model)

        if not should:
            raise RuntimeError(f"[{provider}] Cannot call: {reason}")

        if acct_label is None:
            raise RuntimeError(f"[{provider}] No account label returned from should_call")

        acct = pool.get_account(acct_label)
        if not acct:
            raise RuntimeError(f"[{provider}] Account '{acct_label}' not found")

        # Make the actual call via webchat_providers
        from .webchat_providers import get_provider as get_webchat_provider

        webchat = get_webchat_provider(provider, acct.label)

        start = time.time()
        try:
            result = webchat.call(messages, model=model, timeout=timeout)
            latency_ms = (time.time() - start) * 1000

            # Record success
            pool.record_call(
                acct.label,
                success=True,
                latency_ms=latency_ms,
                tokens_used=len(result) // 4,  # rough estimate
            )

            return result, model or webchat.DEFAULT_MODEL, acct.label, latency_ms

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            pool.record_call(
                acct.label,
                success=False,
                latency_ms=latency_ms,
                error_msg=str(e)[:100],
            )

            # Try fallback to another account
            if preferred_account or acct_label:
                fallback = pool.best_account(model)
                if fallback and fallback.label != acct_label:
                    print(f"  🔄 Falling back from {acct_label} to {fallback.label}")
                    webchat2 = get_webchat_provider(provider, fallback.label)
                    start2 = time.time()
                    try:
                        result2 = webchat2.call(messages, model=model, timeout=timeout)
                        latency2 = (time.time() - start2) * 1000
                        pool.record_call(
                            fallback.label,
                            success=True,
                            latency_ms=latency2,
                            tokens_used=len(result2) // 4,
                        )
                        return result2, model or webchat2.DEFAULT_MODEL, fallback.label, latency2
                    except Exception as e2:
                        pool.record_call(
                            fallback.label,
                            success=False,
                            error_msg=str(e2)[:100],
                        )

            raise


# =============================================================================
# Auto-Discovery: Populate pools from existing auth cache
# =============================================================================


def auto_discover_accounts() -> dict[str, AccountPool]:
    """
    Scan ~/.config/gaap/webchat_auth/ for existing auth files
    and auto-register accounts in the pool if not already registered.

    Returns dict of {provider: pool}.
    """
    from .webchat_providers import list_accounts, load_auth

    mgr = PoolManager.instance()
    known_providers = ["glm", "kimi", "deepseek", "copilot"]
    discovered = {}

    for provider in known_providers:
        accounts = list_accounts(provider)
        if not accounts:
            continue

        pool = mgr.pool(provider)
        for label in accounts:
            # Skip if already registered
            if pool.get_account(label):
                continue

            # Load auth to get metadata
            auth = load_auth(provider, label)

            slot = pool.add_account(
                label=label,
                email="",
                account_type="free",
                notes=f"Auto-discovered. user_id={auth.user_id[:12] + '...' if auth and auth.user_id else 'N/A'}",
            )

            if auth and auth.user_id:
                slot.email = f"uid:{auth.user_id}"

            discovered[provider] = pool

    return discovered


def bootstrap_pools() -> str:
    """
    Initialize the pool system with all discovered accounts.
    Call this at startup.

    Returns summary string.
    """
    auto_discover_accounts()
    mgr = PoolManager.instance()

    lines = ["🔄 Account Pool Bootstrap:"]
    all_pools = mgr.all_pools()

    for provider, pool in sorted(all_pools.items()):
        accts = pool.accounts
        if accts:
            active = sum(1 for a in accts if a.can_call()[0])
            lines.append(f"  📦 {provider}: {len(accts)} accounts ({active} active)")
            for acct in accts:
                can, reason = acct.can_call()
                icon = "✅" if can else "❌"
                lines.append(f"     {icon} {acct.label}: {reason}")

    if not all_pools:
        lines.append("  📭 No accounts found. Use 'add' to register accounts.")

    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def _cli() -> None:
    """Command-line interface for account management."""
    import sys

    usage = """
Usage: python -m gaap.providers.account_manager <command> [args]

Commands:
  dashboard                         Show full dashboard for all providers
  bootstrap                         Auto-discover auth & initialize pools
  status <provider>                 Show status for a specific provider
  add <provider> <label> [email]    Add an account
  remove <provider> <label>         Remove an account
  list <provider>                   List accounts for a provider
  test <provider> [label] [model]   Test a call with account selection
  warnings                          Show all warnings
  models <provider>                 Show available models for a provider

Examples:
  python -m gaap.providers.account_manager dashboard
  python -m gaap.providers.account_manager bootstrap
  python -m gaap.providers.account_manager add kimi main user@email.com
  python -m gaap.providers.account_manager add kimi alt1 other@email.com
  python -m gaap.providers.account_manager test kimi "" kimi-k2.5-thinking
  python -m gaap.providers.account_manager models kimi
  python -m gaap.providers.account_manager warnings
"""

    args = sys.argv[1:]
    if not args:
        print(usage)
        return

    cmd = args[0]
    mgr = PoolManager.instance()

    if cmd == "dashboard":
        auto_discover_accounts()
        print(mgr.full_dashboard())

    elif cmd == "bootstrap":
        print(bootstrap_pools())

    elif cmd == "models":
        if len(args) < 2:
            print("Usage: models <provider>")
            print("Available providers:", ", ".join(PROVIDER_DEFAULTS.keys()))
            return
        provider = args[1]
        defaults = PROVIDER_DEFAULTS.get(provider, {})
        if not defaults:
            print(f"  ❌ Unknown provider: {provider}")
            print("  Available:", ", ".join(PROVIDER_DEFAULTS.keys()))
            return
        print(f"\n📦 {provider.upper()} Models:")
        for m in defaults.get("models", []):
            scenario = defaults.get("model_scenarios", {}).get(m, "")
            extra = f" → {scenario}" if scenario else ""
            print(f"  • {m}{extra}")
        print(
            f"\nLimits: {defaults.get('max_rpm', '?')} RPM, {defaults.get('max_rph', '?')} RPH, {defaults.get('max_rpd', '?')} RPD"
        )
        print(f"Auth: ~{defaults.get('auth_lifetime_hours', '?')}h")
        if defaults.get("notes"):
            print(f"Notes: {defaults['notes']}")

    elif cmd == "status":
        if len(args) < 2:
            print("Usage: status <provider>")
            return
        pool = mgr.pool(args[1])
        print(f"\n📦 {args[1].upper()}")
        print(pool.dashboard())

    elif cmd == "add":
        if len(args) < 3:
            print("Usage: add <provider> <label> [email] [account_type]")
            return
        provider = args[1]
        label = args[2]
        email = args[3] if len(args) > 3 else ""
        acct_type = args[4] if len(args) > 4 else "free"

        pool = mgr.pool(provider)
        slot = pool.add_account(label=label, email=email, account_type=acct_type)
        print(f"  ✅ Added {provider}/{label} ({acct_type})")
        print(f"     Models: {slot.models}")
        print(f"     Limits: {slot.max_rpm} RPM, {slot.max_rph} RPH, {slot.max_rpd} RPD")

    elif cmd == "remove":
        if len(args) < 3:
            print("Usage: remove <provider> <label>")
            return
        pool = mgr.pool(args[1])
        if pool.remove_account(args[2]):
            print(f"  🗑️  Removed {args[1]}/{args[2]}")
        else:
            print(f"  ❌ Account {args[1]}/{args[2]} not found")

    elif cmd == "list":
        if len(args) < 2:
            print("Usage: list <provider>")
            return
        pool = mgr.pool(args[1])
        for acct in pool.accounts:
            icon = "✅" if acct.can_call()[0] else "❌"
            print(
                f"  {icon} {acct.label}: {acct.email} ({acct.account_type}) — {acct.status.value}"
            )

    elif cmd == "test":
        if len(args) < 2:
            print("Usage: test <provider> [label] [model]")
            return
        provider = args[1]
        label = args[2] if len(args) > 2 and args[2] else ""
        model = args[3] if len(args) > 3 else ""

        print(
            f"  🧪 Testing {provider}"
            + (f" [{label}]" if label else "")
            + (f" model={model}" if model else "")
        )
        try:
            text, model_used, acct_used, latency = mgr.smart_call(
                provider,
                [{"role": "user", "content": "Reply with only: OK"}],
                model=model,
                preferred_account=label,
                timeout=60,
            )
            print(
                f"  ✅ Response: '{text[:80]}' via {acct_used} model={model_used} in {latency:.0f}ms"
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")

    elif cmd == "warnings":
        all_pools = mgr.all_pools()
        found = False
        for provider, pool in sorted(all_pools.items()):
            warnings = pool.get_warnings()
            if warnings:
                found = True
                for label, warns in warnings.items():
                    for w in warns:
                        print(f"  [{provider}/{label}] {w}")
        if not found:
            print("  ✅ No warnings — all accounts healthy")

    else:
        print(f"Unknown command: {cmd}")
        print(usage)


if __name__ == "__main__":
    _cli()
