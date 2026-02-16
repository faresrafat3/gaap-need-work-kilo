from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """مدخل ذاكرة مؤقتة"""

    key: str
    value: T
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    hits: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """هل انتهت صلاحية المدخل"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def access(self) -> None:
        """تسجيل وصول"""
        self.hits += 1


class CacheBackend(ABC):
    """الواجهة الأساسية للتخزين المؤقت"""

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """الحصول على قيمة"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """تخزين قيمة"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """حذف قيمة"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """مسح جميع القيم"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """فحص وجود المفتاح"""
        pass

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "hit_rate": hit_rate,
            "total_requests": total,
        }

    def _record_hit(self) -> None:
        self._stats["hits"] += 1

    def _record_miss(self) -> None:
        self._stats["misses"] += 1

    def _record_set(self) -> None:
        self._stats["sets"] += 1

    def _record_eviction(self) -> None:
        self._stats["evictions"] += 1
