"""
WebChat Authentication - Token caching and management.

Provides persistent auth storage for webchat providers:
- WebChatAuth: Dataclass for cached auth state
- save_auth/load_auth/invalidate_auth: Disk cache operations
- list_accounts: List cached accounts per provider
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("gaap.providers.webchat")

WEBCHAT_CACHE_DIR = Path.home() / ".config" / "gaap" / "webchat_auth"


@dataclass
class WebChatAuth:
    """Cached authentication state for a webchat provider."""

    provider: str
    token: str
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    user_id: str = ""
    account_label: str = "default"
    captured_at: float = 0.0
    expires_at: float = 0.0

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            return (time.time() - self.captured_at) > 43200
        return time.time() >= self.expires_at

    @property
    def remaining_sec(self) -> int:
        if self.expires_at <= 0:
            return max(0, int(43200 - (time.time() - self.captured_at)))
        return max(0, int(self.expires_at - time.time()))

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "token": self.token,
            "cookies": self.cookies,
            "headers": self.headers,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "account_label": self.account_label,
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WebChatAuth":
        return cls(
            provider=d.get("provider", ""),
            token=d.get("token", ""),
            cookies=d.get("cookies", {}),
            headers=d.get("headers", {}),
            user_agent=d.get("user_agent", ""),
            user_id=d.get("user_id", ""),
            account_label=d.get("account_label", "default"),
            captured_at=d.get("captured_at", 0.0),
            expires_at=d.get("expires_at", 0.0),
        )


def _cache_path(provider: str, account: str = "default") -> Path:
    return WEBCHAT_CACHE_DIR / f"{provider}_{account}.json"


def save_auth(auth: WebChatAuth) -> Path:
    """Save auth to disk cache."""
    WEBCHAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(auth.provider, auth.account_label)
    path.write_text(json.dumps(auth.to_dict(), indent=2), encoding="utf-8")
    return path


def load_auth(provider: str, account: str = "default") -> WebChatAuth | None:
    """Load auth from disk cache. Returns None if missing or expired."""
    path = _cache_path(provider, account)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        auth = WebChatAuth.from_dict(data)
        if auth.is_expired:
            return None
        return auth
    except (OSError, json.JSONDecodeError):
        return None


def invalidate_auth(provider: str, account: str = "default") -> bool:
    """Delete cached auth file."""
    path = _cache_path(provider, account)
    if path.exists():
        path.unlink()
        return True
    return False


def list_accounts(provider: str) -> list[str]:
    """List all cached account labels for a provider."""
    if not WEBCHAT_CACHE_DIR.exists():
        return []
    prefix = f"{provider}_"
    accounts = []
    for f in WEBCHAT_CACHE_DIR.iterdir():
        if f.name.startswith(prefix) and f.name.endswith(".json"):
            label = f.name[len(prefix) : -5]
            accounts.append(label)
    return sorted(accounts)


__all__ = [
    "WEBCHAT_CACHE_DIR",
    "WebChatAuth",
    "save_auth",
    "load_auth",
    "invalidate_auth",
    "list_accounts",
]
