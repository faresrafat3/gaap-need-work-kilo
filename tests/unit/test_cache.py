"""
Tests for GAAP Cache System
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from gaap.cache.base import CacheBackend, CacheEntry


class TestCacheEntry:
    def test_create_entry(self):
        entry = CacheEntry(key="test_key", value="test_value")

        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at is not None
        assert entry.expires_at is None
        assert entry.hits == 0

    def test_is_expired_no_expiry(self):
        entry = CacheEntry(key="test", value="value")

        assert entry.is_expired() is False

    def test_is_expired_future(self):
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=datetime.now() + timedelta(hours=1),
        )

        assert entry.is_expired() is False

    def test_is_expired_past(self):
        entry = CacheEntry(
            key="test",
            value="value",
            expires_at=datetime.now() - timedelta(hours=1),
        )

        assert entry.is_expired() is True

    def test_access(self):
        entry = CacheEntry(key="test", value="value")

        entry.access()
        assert entry.hits == 1

        entry.access()
        assert entry.hits == 2

    def test_metadata(self):
        entry = CacheEntry(
            key="test",
            value="value",
            metadata={"source": "api", "version": "1.0"},
        )

        assert entry.metadata["source"] == "api"
        assert entry.metadata["version"] == "1.0"


class MockCacheBackend(CacheBackend):
    def __init__(self, ttl_seconds=3600, max_size=1000):
        super().__init__(ttl_seconds, max_size)
        self._data: dict[str, CacheEntry] = {}

    def get(self, key: str):
        entry = self._data.get(key)
        if entry is None:
            self._record_miss()
            return None
        if entry.is_expired():
            self.delete(key)
            self._record_miss()
            return None
        entry.access()
        self._record_hit()
        return entry.value

    def set(self, key: str, value, ttl_seconds=None):
        ttl = ttl_seconds or self.ttl_seconds
        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=datetime.now() + timedelta(seconds=ttl),
        )
        self._data[key] = entry
        self._record_set()

    def delete(self, key: str):
        if key in self._data:
            del self._data[key]
            return True
        return False

    def clear(self):
        self._data.clear()

    def exists(self, key: str):
        return key in self._data


class TestCacheBackend:
    @pytest.fixture
    def cache(self):
        return MockCacheBackend()

    def test_set_and_get(self, cache):
        cache.set("key1", "value1")

        result = cache.get("key1")
        assert result == "value1"

    def test_get_nonexistent(self, cache):
        result = cache.get("nonexistent")
        assert result is None

    def test_delete(self, cache):
        cache.set("key1", "value1")

        result = cache.delete("key1")
        assert result is True

        result = cache.get("key1")
        assert result is None

    def test_delete_nonexistent(self, cache):
        result = cache.delete("nonexistent")
        assert result is False

    def test_clear(self, cache):
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_exists(self, cache):
        cache.set("key1", "value1")

        assert cache.exists("key1") is True
        assert cache.exists("nonexistent") is False

    def test_stats(self, cache):
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()

        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate"] == 0.5

    def test_ttl_expiry(self, cache):
        cache.set("key1", "value1", ttl_seconds=-1)

        result = cache.get("key1")
        assert result is None

    def test_custom_ttl(self, cache):
        cache.set("key1", "value1", ttl_seconds=10)

        result = cache.get("key1")
        assert result == "value1"

    def test_multiple_operations(self, cache):
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")

        for i in range(10):
            assert cache.get(f"key_{i}") == f"value_{i}"

        stats = cache.get_stats()
        assert stats["sets"] == 10
        assert stats["hits"] == 10
