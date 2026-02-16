# mypy: ignore-errors
import hashlib
import logging
from typing import Any

from gaap.cache.disk_cache import DiskCache
from gaap.cache.memory_cache import MemoryCache


class ResponseCache:
    """تخزين مؤقت لاستجابات LLM"""

    def __init__(
        self,
        backend: str = "memory",
        ttl_seconds: int = 3600,
        max_size: int = 1000,
        storage_path: str = "./.gaap_cache",
    ):
        self.backend_name = backend

        if backend == "disk":
            self._cache = DiskCache(
                storage_path=storage_path,
                ttl_seconds=ttl_seconds,
                max_size=max_size,
            )
        else:
            self._cache = MemoryCache(
                ttl_seconds=ttl_seconds,
                max_size=max_size,
            )

        self._logger = logging.getLogger("gaap.cache.response")

    def get_cache_key(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        provider: str = "",
        **kwargs,
    ) -> str:
        """إنشاء مفتاح التخزين المؤقت"""
        key_parts = [
            prompt,
            model,
            str(temperature),
            provider,
        ]

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        provider: str = "",
        **kwargs,
    ) -> dict[str, Any] | None:
        """الحصول على استجابة مخزنة"""
        cache_key = self.get_cache_key(prompt, model, temperature, provider, **kwargs)
        cached = self._cache.get(cache_key)

        if cached:
            self._logger.debug(f"Cache hit for key: {cache_key[:16]}...")
            return cached

        self._logger.debug(f"Cache miss for key: {cache_key[:16]}...")
        return None

    def set(
        self,
        prompt: str,
        response: dict[str, Any],
        model: str,
        temperature: float = 0.7,
        provider: str = "",
        ttl_seconds: int | None = None,
        **kwargs,
    ) -> None:
        """تخزين استجابة"""
        cache_key = self.get_cache_key(prompt, model, temperature, provider, **kwargs)

        cache_entry = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "temperature": temperature,
            "provider": provider,
            "cached_at": str(self._get_timestamp()),
        }

        self._cache.set(cache_key, cache_entry, ttl_seconds)
        self._logger.debug(f"Cached response for key: {cache_key[:16]}...")

    def invalidate(self, model: str | None = None, provider: str | None = None) -> int:
        """إلغاء التخزين المؤقت"""
        if model and provider:
            pattern = f"{model}:{provider}"
        elif model:
            pattern = model
        elif provider:
            pattern = provider
        else:
            self._cache.clear()
            return 0

        count = self._cache.invalidate_pattern(pattern)
        self._logger.info(f"Invalidated {count} cache entries")
        return count

    def clear(self) -> None:
        """مسح التخزين المؤقت"""
        self._cache.clear()
        self._logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات التخزين المؤقت"""
        return {
            "backend": self.backend_name,
            "stats": self._cache.get_stats(),
        }

    def _get_timestamp(self) -> str:
        """الحصول على timestamp"""
        from datetime import datetime

        return datetime.now().isoformat()


def create_response_cache(
    backend: str = "memory",
    ttl_seconds: int = 3600,
    max_size: int = 1000,
    storage_path: str = "./.gaap_cache",
) -> ResponseCache:
    """إنشاء ResponseCache"""
    return ResponseCache(
        backend=backend,
        ttl_seconds=ttl_seconds,
        max_size=max_size,
        storage_path=storage_path,
    )
