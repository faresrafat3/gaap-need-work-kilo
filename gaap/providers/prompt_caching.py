"""
Prompt Caching Module for Providers
====================================

Provider-level prompt caching for:
- Anthropic: cache_control flags
- DeepSeek: context caching
- OpenAI: automatic prompt caching

Saves up to 90% on input tokens for repeated prompts.

Usage:
    from gaap.providers.prompt_caching import PromptCache, CacheConfig

    cache = PromptCache()
    cached_messages = cache.optimize(messages, provider="anthropic")
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("gaap.providers.prompt_caching")


class CacheProvider(Enum):
    """Providers supporting prompt caching."""

    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    GEMINI = "gemini"


@dataclass
class CacheConfig:
    """Prompt caching configuration."""

    enabled: bool = True
    min_tokens: int = 1024
    max_cache_size: int = 100
    ttl_seconds: int = 300
    cache_system_prompt: bool = True
    cache_tools: bool = True


@dataclass
class CacheEntry:
    """A cached prompt entry."""

    key: str
    content_hash: str
    cached_tokens: int
    created_at: float
    last_used: float
    hit_count: int = 0
    provider: str = ""

    @property
    def is_expired(self) -> bool:
        return time.time() - self.created_at > 300


class PromptCache:
    """
    Manages prompt caching across providers.

    Features:
    - Content-based deduplication
    - TTL-based expiration
    - Provider-specific optimizations
    - Cache statistics

    Usage:
        >>> cache = PromptCache()
        >>> messages = cache.optimize(messages, provider="anthropic")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._entries: dict[str, CacheEntry] = {}
        self._stats: dict[str, float] = {
            "hits": 0,
            "misses": 0,
            "tokens_saved": 0.0,
        }

    def _hash_content(self, content: str) -> str:
        """Generate content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cache_key(self, messages: list[dict], provider: str) -> str:
        """Generate cache key for messages."""
        content = json_dumps(messages, sort_keys=True)
        content_hash = self._hash_content(content)
        return f"{provider}:{content_hash}"

    def optimize(
        self,
        messages: list[dict[str, str]],
        provider: str,
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Optimize messages for caching.

        Args:
            messages: List of message dicts
            provider: Provider name
            model: Optional model name

        Returns:
            Optimized messages with cache_control flags
        """
        if not self.config.enabled:
            return messages

        provider_lower = provider.lower()

        if provider_lower == "anthropic":
            return self._optimize_anthropic(messages, model)
        elif provider_lower == "deepseek":
            return self._optimize_deepseek(messages, model)
        elif provider_lower == "openai":
            return self._optimize_openai(messages, model)
        else:
            return messages

    def _optimize_anthropic(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add cache_control flags for Anthropic.

        Anthropic supports caching:
        - System messages
        - Long context blocks
        - Tool definitions
        """
        optimized = []
        has_system = False

        for i, msg in enumerate(messages):
            new_msg: dict[str, Any] = dict(msg)

            if msg.get("role") == "system" and self.config.cache_system_prompt:
                content = msg.get("content", "")
                if len(content) >= self.config.min_tokens:
                    new_msg["cache_control"] = {"type": "ephemeral"}
                    has_system = True
                    self._record_cache(msg, "anthropic")

            elif i == len(messages) - 2:
                content = msg.get("content", "")
                if len(content) >= self.config.min_tokens and not has_system:
                    new_msg["cache_control"] = {"type": "ephemeral"}

            optimized.append(new_msg)

        return optimized

    def _optimize_deepseek(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Optimize for DeepSeek context caching.

        DeepSeek caches context automatically for:
        - Repeated system prompts
        - Large context blocks
        """
        optimized = []

        for msg in messages:
            new_msg: dict[str, Any] = dict(msg)

            if msg.get("role") == "system" and self.config.cache_system_prompt:
                content = msg.get("content", "")
                if len(content) >= self.config.min_tokens:
                    self._record_cache(msg, "deepseek")

            optimized.append(new_msg)

        return optimized

    def _optimize_openai(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        OpenAI has automatic prompt caching.

        No manual flags needed, but we track for statistics.
        """
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if len(content) >= self.config.min_tokens:
                    self._record_cache(msg, "openai")

        return messages

    def _record_cache(self, message: dict, provider: str) -> None:
        """Record a cached prompt."""
        content = message.get("content", "")
        content_hash = self._hash_content(content)
        key = f"{provider}:{content_hash}"

        if key in self._entries:
            self._entries[key].hit_count += 1
            self._entries[key].last_used = time.time()
            self._stats["hits"] += 1
            self._stats["tokens_saved"] += len(content.split()) * 1.3
        else:
            if len(self._entries) >= self.config.max_cache_size:
                self._evict_oldest()

            self._entries[key] = CacheEntry(
                key=key,
                content_hash=content_hash,
                cached_tokens=int(len(content.split()) * 1.3),
                created_at=time.time(),
                last_used=time.time(),
                hit_count=1,
                provider=provider,
            )
            self._stats["misses"] += 1

    def _evict_oldest(self) -> None:
        """Evict oldest cache entries."""
        if not self._entries:
            return

        sorted_entries = sorted(
            self._entries.items(),
            key=lambda x: x[1].last_used,
        )

        for key, _ in sorted_entries[: len(sorted_entries) // 4]:
            del self._entries[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": hit_rate,
            "tokens_saved": int(self._stats["tokens_saved"]),
            "entries": len(self._entries),
            "config": {
                "enabled": self.config.enabled,
                "min_tokens": self.config.min_tokens,
                "ttl_seconds": self.config.ttl_seconds,
            },
        }

    def clear(self) -> None:
        self._entries.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "tokens_saved": 0.0,
        }

    def check_cache_status(
        self,
        messages: list[dict[str, str]],
        provider: str,
    ) -> dict[str, Any]:
        """
        Check if messages would be cached.

        Returns:
            Cache status information
        """
        status: dict[str, Any] = {
            "cacheable": False,
            "cached_tokens": 0,
            "potential_savings": 0.0,
            "cached_messages": [],
        }

        for i, msg in enumerate(messages):
            content = msg.get("content", "")
            if len(content) >= self.config.min_tokens:
                content_hash = self._hash_content(content)
                key = f"{provider}:{content_hash}"

                if key in self._entries:
                    status["cached_messages"].append(i)
                    status["cached_tokens"] += self._entries[key].cached_tokens

        if status["cached_messages"]:
            status["cacheable"] = True
            status["potential_savings"] = status["cached_tokens"] * 0.9

        return status


def json_dumps(obj: Any, **kwargs: Any) -> str:
    import json

    return json.dumps(obj, **kwargs)


def estimate_cache_savings(
    messages: list[dict[str, str]],
    provider: str,
    pricing: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Estimate potential savings from caching.

    Args:
        messages: Message list
        provider: Provider name
        pricing: Optional pricing dict {input: $/1k tokens}

    Returns:
        Savings estimate
    """
    default_pricing = {
        "anthropic": {"input": 0.003, "cached": 0.0003},
        "deepseek": {"input": 0.0005, "cached": 0.0001},
        "openai": {"input": 0.0025, "cached": 0.00125},
    }

    pricing = pricing or default_pricing.get(provider.lower(), {"input": 0.01, "cached": 0.001})

    total_tokens: float = 0.0
    cacheable_tokens: float = 0.0

    for msg in messages:
        content = msg.get("content", "")
        tokens = len(content.split()) * 1.3

        total_tokens += tokens
        if len(content) >= 1024:
            cacheable_tokens += tokens

    normal_cost = (total_tokens / 1000) * pricing["input"]
    cached_cost = ((total_tokens - cacheable_tokens) / 1000) * pricing["input"]
    cached_cost += (cacheable_tokens / 1000) * pricing["cached"]

    return {
        "total_tokens": int(total_tokens),
        "cacheable_tokens": int(cacheable_tokens),
        "normal_cost": normal_cost,
        "cached_cost": cached_cost,
        "savings": normal_cost - cached_cost,
        "savings_percent": ((normal_cost - cached_cost) / normal_cost * 100)
        if normal_cost > 0
        else 0,
    }
