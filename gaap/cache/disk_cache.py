import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.cache.base import CacheBackend, CacheEntry


class DiskCache(CacheBackend):
    """تخزين مؤقت على القرص (JSON)"""

    def __init__(
        self, storage_path: str = "./.gaap_cache", ttl_seconds: int = 3600, max_size: int = 1000
    ):
        super().__init__(ttl_seconds=ttl_seconds, max_size=max_size)
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._index_path = self.storage_path / "index.json"
        self._logger = logging.getLogger("gaap.cache.disk")
        self._load_index()

    def _load_index(self) -> None:
        """تحميل الفهرس"""
        if self._index_path.exists():
            try:
                with open(self._index_path, encoding="utf-8") as f:
                    self._index = json.load(f)
            except Exception as e:
                self._logger.warning(f"Failed to load cache index: {e}")
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """حفظ الفهرس"""
        try:
            with open(self._index_path, "w", encoding="utf-8") as f:
                json.dump(self._index, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self._logger.error(f"Failed to save cache index: {e}")

    def _get_file_path(self, key: str) -> Path:
        """الحصول على مسار الملف"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.storage_path / f"{key_hash}.json"

    def get(self, key: str) -> Any | None:
        """الحصول على قيمة"""
        if key not in self._index:
            self._record_miss()
            return None

        file_path = self._get_file_path(key)
        if not file_path.exists():
            del self._index[key]
            self._save_index()
            self._record_miss()
            return None

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            entry = CacheEntry(
                key=data["key"],
                value=data["value"],
                created_at=datetime.fromisoformat(data["created_at"]),
                expires_at=(
                    datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None
                ),
                hits=data.get("hits", 0),
                metadata=data.get("metadata", {}),
            )

            if entry.is_expired():
                self.delete(key)
                self._record_miss()
                return None

            entry.hits += 1
            self._record_hit()
            self._save_entry(entry)
            return entry.value

        except Exception as e:
            self._logger.error(f"Failed to read cache entry: {e}")
            self.delete(key)
            self._record_miss()
            return None

    def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """تخزين قيمة"""
        ttl = ttl_seconds or self.ttl_seconds
        expires_at = None
        if ttl > 0:
            from datetime import timedelta

            expires_at = datetime.now() + timedelta(seconds=ttl)

        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at,
        )

        self._save_entry(entry)
        self._index[key] = {
            "file": str(self._get_file_path(key).name),
            "created": entry.created_at.isoformat(),
            "expires": entry.expires_at.isoformat() if entry.expires_at else None,
        }

        if len(self._index) > self.max_size:
            self._evict_oldest()

        self._save_index()
        self._record_set()

    def _save_entry(self, entry: CacheEntry) -> None:
        """حفظ مدخل واحد"""
        file_path = self._get_file_path(entry.key)
        data = {
            "key": entry.key,
            "value": entry.value,
            "created_at": entry.created_at.isoformat(),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
            "hits": entry.hits,
            "metadata": entry.metadata,
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def _evict_oldest(self) -> None:
        """إزالة الأقدم"""
        if not self._index:
            return

        oldest_key = min(self._index.items(), key=lambda x: x[1].get("created", ""))[0]
        self.delete(oldest_key)
        self._record_eviction()

    def delete(self, key: str) -> bool:
        """حذف قيمة"""
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()

        if key in self._index:
            del self._index[key]
            self._save_index()
            return True
        return False

    def clear(self) -> None:
        """مسح جميع القيم"""
        for key in list(self._index.keys()):
            self.delete(key)
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "evictions": 0}

    def exists(self, key: str) -> bool:
        """فحص وجود المفتاح"""
        return self.get(key) is not None

    def invalidate_pattern(self, pattern: str) -> int:
        """حذف جميع المفاتيح التي تطابق نمط"""
        keys_to_delete = [k for k in self._index.keys() if pattern in k]
        for key in keys_to_delete:
            self.delete(key)
        return len(keys_to_delete)

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        stats = super().get_stats()
        stats.update(
            {
                "current_size": len(self._index),
                "max_size": self.max_size,
                "storage_path": str(self.storage_path),
            }
        )
        return stats
