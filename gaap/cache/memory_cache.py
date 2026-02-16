from collections import OrderedDict
from datetime import datetime
from typing import Any

from gaap.cache.base import CacheBackend, CacheEntry


class MemoryCache(CacheBackend):
    """تخزين مؤقت في الذاكرة (LRU)"""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        super().__init__(ttl_seconds=ttl_seconds, max_size=max_size)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Any | None:
        """الحصول على قيمة"""
        if key not in self._cache:
            self._record_miss()
            return None

        entry = self._cache[key]

        if entry.is_expired():
            del self._cache[key]
            self._record_miss()
            self._record_eviction()
            return None

        entry.access()
        self._cache.move_to_end(key)
        self._record_hit()
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """تخزين قيمة"""
        if key in self._cache:
            del self._cache[key]

        ttl = ttl_seconds or self.ttl_seconds
        expires_at = datetime.now().timestamp() + ttl if ttl > 0 else None

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=datetime.fromtimestamp(expires_at) if expires_at else None,
        )

        self._cache[key] = entry
        self._cache.move_to_end(key)
        self._record_set()

        if len(self._cache) > self.max_size:
            self._evict_oldest()

    def _evict_oldest(self) -> None:
        """إزالة الأقدم"""
        if self._cache:
            self._cache.popitem(last=False)
            self._record_eviction()

    def delete(self, key: str) -> bool:
        """حذف قيمة"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> None:
        """مسح جميع القيم"""
        self._cache.clear()
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    def exists(self, key: str) -> bool:
        """فحص وجود المفتاح"""
        if key not in self._cache:
            return False
        entry = self._cache[key]
        if entry.is_expired():
            del self._cache[key]
            self._record_eviction()
            return False
        return True

    def invalidate_pattern(self, pattern: str) -> int:
        """حذف جميع المفاتيح التي تطابق نمط"""
        keys_to_delete = [k for k in self._cache.keys() if pattern in k]
        for key in keys_to_delete:
            del self._cache[key]
        return len(keys_to_delete)

    def cleanup_expired(self) -> int:
        """تنظيف المنتهية صلاحيتها"""
        expired_keys = [k for k, e in self._cache.items() if e.is_expired()]
        for key in expired_keys:
            del self._cache[key]
            self._record_eviction()
        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        stats = super().get_stats()
        stats.update(
            {
                "current_size": len(self._cache),
                "max_size": self.max_size,
                "expired_entries": sum(1 for e in self._cache.values() if e.is_expired()),
            }
        )
        return stats
