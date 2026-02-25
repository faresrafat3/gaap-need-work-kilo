"""
Tests for SQLite Storage
"""

import pytest
from gaap.storage import SQLiteStore, SQLiteConfig


class TestSQLiteStore:
    @pytest.fixture
    def store(self):
        config = SQLiteConfig.in_memory()
        return SQLiteStore(config)

    def test_insert_and_get(self, store):
        item_id = store.insert("test_table", {"name": "test", "value": 42})
        assert item_id is not None

        record = store.get("test_table", item_id)
        assert record is not None
        assert record["data"]["name"] == "test"
        assert record["data"]["value"] == 42
        assert "created_at" in record

    def test_query_with_where(self, store):
        store.insert("events", {"type": "click", "target": "button"})
        store.insert("events", {"type": "scroll", "target": "window"})
        store.insert("events", {"type": "click", "target": "link"})

        results = store.query("events", where={"type": "click"})
        assert len(results) == 2

        for r in results:
            assert r["data"]["type"] == "click"

    def test_query_with_limit(self, store):
        for i in range(10):
            store.insert("items", {"index": i})

        results = store.query("items", limit=5)
        assert len(results) == 5

    def test_update(self, store):
        item_id = store.insert("items", {"name": "original"})

        success = store.update("items", item_id, {"name": "updated"})
        assert success

        record = store.get("items", item_id)
        assert record["data"]["name"] == "updated"
        assert record["updated_at"] is not None

    def test_delete(self, store):
        item_id = store.insert("items", {"name": "to_delete"})

        success = store.delete("items", item_id)
        assert success

        record = store.get("items", item_id)
        assert record is None

    def test_count(self, store):
        store.insert("items", {"a": 1})
        store.insert("items", {"a": 2})
        store.insert("items", {"a": 1})

        total = store.count("items")
        assert total == 3

        filtered = store.count("items", where={"a": 1})
        assert filtered == 2

    def test_clear(self, store):
        store.insert("items", {"a": 1})
        store.insert("items", {"a": 2})

        cleared = store.clear("items")
        assert cleared == 2

        assert store.count("items") == 0

    def test_get_stats(self, store):
        store.insert("table1", {"x": 1})
        store.insert("table2", {"y": 2})

        stats = store.get_stats()
        assert "tables" in stats
        assert "table1" in stats["tables"]
        assert "table2" in stats["tables"]

    def test_custom_id(self, store):
        custom_id = "my_custom_id_123"
        item_id = store.insert("items", {"name": "test"}, item_id=custom_id)

        assert item_id == custom_id

        record = store.get("items", custom_id)
        assert record is not None

    def test_complex_data(self, store):
        complex_data = {
            "nested": {"a": {"b": {"c": 1}}},
            "list": [1, 2, 3, {"x": "y"}],
            "mixed": [{"id": 1}, {"id": 2}],
        }

        item_id = store.insert("complex", complex_data)
        record = store.get("complex", item_id)

        assert record["data"]["nested"]["a"]["b"]["c"] == 1
        assert record["data"]["list"][3]["x"] == "y"
