"""
Tests for GAAP JSON Storage
"""

import json
import tempfile
from pathlib import Path

import pytest

from gaap.storage.json_store import JSONStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = JSONStore(base_dir=Path(tmpdir))
        yield store


class TestJSONStore:
    def test_init(self, temp_store):
        assert temp_store.base_dir.exists()

    def test_save_and_load(self, temp_store):
        data = {"key": "value", "number": 42}
        temp_store.save("test", data)

        loaded = temp_store.load("test")
        assert loaded == data

    def test_load_nonexistent(self, temp_store):
        result = temp_store.load("nonexistent", default={"default": True})
        assert result == {"default": True}

    def test_load_nonexistent_no_default(self, temp_store):
        result = temp_store.load("nonexistent")
        assert result is None

    def test_append(self, temp_store):
        item_id = temp_store.append("history", {"message": "Hello"})

        assert item_id is not None
        assert len(item_id) == 8

        data = temp_store.load("history")
        assert len(data) == 1
        assert data[0]["message"] == "Hello"
        assert data[0]["id"] == item_id
        assert "timestamp" in data[0]

    def test_get_by_id(self, temp_store):
        item_id = temp_store.append("items", {"name": "Test"})

        item = temp_store.get_by_id("items", item_id)
        assert item is not None
        assert item["name"] == "Test"

    def test_get_by_id_not_found(self, temp_store):
        item = temp_store.get_by_id("items", "nonexistent")
        assert item is None

    def test_delete_by_id(self, temp_store):
        item_id = temp_store.append("items", {"name": "Test"})

        result = temp_store.delete_by_id("items", item_id)
        assert result is True

        item = temp_store.get_by_id("items", item_id)
        assert item is None

    def test_delete_by_id_not_found(self, temp_store):
        result = temp_store.delete_by_id("items", "nonexistent")
        assert result is False

    def test_clear(self, temp_store):
        temp_store.append("items", {"name": "Test1"})
        temp_store.append("items", {"name": "Test2"})

        temp_store.clear("items")

        data = temp_store.load("items")
        assert data == []

    def test_update(self, temp_store):
        item_id = temp_store.append("items", {"name": "Test", "count": 1})

        result = temp_store.update("items", item_id, {"count": 5})
        assert result is True

        item = temp_store.get_by_id("items", item_id)
        assert item["count"] == 5
        assert "updated_at" in item

    def test_update_not_found(self, temp_store):
        result = temp_store.update("items", "nonexistent", {"count": 5})
        assert result is False

    def test_multiple_appends(self, temp_store):
        id1 = temp_store.append("history", {"msg": "Hello"})
        id2 = temp_store.append("history", {"msg": "World"})

        data = temp_store.load("history")
        assert len(data) == 2
        assert id1 != id2

    def test_complex_data(self, temp_store):
        data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "bool": True,
            "none": None,
        }
        temp_store.save("complex", data)

        loaded = temp_store.load("complex")
        assert loaded == data

    def test_get_lock(self, temp_store):
        lock1 = temp_store._get_lock("test1")
        lock2 = temp_store._get_lock("test2")
        lock1_again = temp_store._get_lock("test1")

        assert lock1 is lock1_again
        assert lock1 is not lock2

    def test_get_store(self):
        from gaap.storage.json_store import get_store

        store1 = get_store()
        store2 = get_store()
        assert store1 is store2
