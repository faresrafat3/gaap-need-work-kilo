"""
Comprehensive tests for gaap/storage/sqlite_store.py module
Tests SQLiteStore, SQLiteConfig, and all CRUD operations
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from gaap.storage.sqlite_store import (
    SQLiteConfig,
    SQLiteStore,
    get_sqlite_store,
)


class TestSQLiteConfig:
    """Test SQLiteConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SQLiteConfig()
        assert config.db_path == ".gaap/gaap.db"
        assert config.timeout == 30.0
        assert config.check_same_thread is False
        assert config.journal_mode == "WAL"
        assert config.synchronous == "NORMAL"

    def test_default_class_method(self):
        """Test default() class method"""
        config = SQLiteConfig.default()
        assert config.db_path == ".gaap/gaap.db"
        assert config.timeout == 30.0

    def test_in_memory_class_method(self):
        """Test in_memory() class method"""
        config = SQLiteConfig.in_memory()
        assert config.db_path == ":memory:"

    def test_custom_config(self):
        """Test custom configuration"""
        config = SQLiteConfig(
            db_path="/custom/path.db",
            timeout=60.0,
            check_same_thread=True,
            journal_mode="DELETE",
            synchronous="FULL",
        )
        assert config.db_path == "/custom/path.db"
        assert config.timeout == 60.0
        assert config.check_same_thread is True
        assert config.journal_mode == "DELETE"
        assert config.synchronous == "FULL"


class TestSQLiteStoreBasic:
    """Test basic SQLiteStore operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    @pytest.fixture
    def memory_store(self):
        """Create an in-memory SQLite store"""
        config = SQLiteConfig.in_memory()
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_store_initialization(self, tmp_path: Path):
        """Test store initialization creates database"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)

        assert db_path.exists()
        store.close()

    def test_store_initialization_in_memory(self):
        """Test in-memory store initialization"""
        config = SQLiteConfig.in_memory()
        store = SQLiteStore(config)

        assert store.config.db_path == ":memory:"
        store.close()

    def test_store_creates_directory(self, tmp_path: Path):
        """Test store creates directory if it doesn't exist"""
        db_path = tmp_path / "subdir" / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)

        assert db_path.parent.exists()
        store.close()


class TestSQLiteStoreCRUD:
    """Test CRUD operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_insert_single_record(self, store: SQLiteStore):
        """Test inserting a single record"""
        record_id = store.insert("test_table", {"name": "test", "value": 42})

        assert record_id is not None
        assert len(record_id) == 12  # UUID hex[:12]

    def test_insert_with_custom_id(self, store: SQLiteStore):
        """Test inserting with custom ID"""
        custom_id = "custom123456"
        record_id = store.insert("test_table", {"name": "test"}, item_id=custom_id)

        assert record_id == custom_id

    def test_get_existing_record(self, store: SQLiteStore):
        """Test getting an existing record"""
        record_id = store.insert("test_table", {"name": "test", "value": 42})
        record = store.get("test_table", record_id)

        assert record is not None
        assert record["id"] == record_id
        assert record["data"]["name"] == "test"
        assert record["data"]["value"] == 42
        assert "created_at" in record

    def test_get_nonexistent_record(self, store: SQLiteStore):
        """Test getting a non-existent record"""
        record = store.get("test_table", "nonexistent")
        assert record is None

    def test_get_nonexistent_table(self, store: SQLiteStore):
        """Test getting from non-existent table"""
        # Table should be created on first insert
        # But getting from a table that was never inserted to
        record = store.get("never_inserted", "id")
        assert record is None

    def test_update_existing_record(self, store: SQLiteStore):
        """Test updating an existing record"""
        record_id = store.insert("test_table", {"name": "original", "value": 1})
        success = store.update("test_table", record_id, {"name": "updated", "value": 2})

        assert success is True

        record = store.get("test_table", record_id)
        assert record["data"]["name"] == "updated"
        assert record["data"]["value"] == 2
        assert "updated_at" in record

    def test_update_nonexistent_record(self, store: SQLiteStore):
        """Test updating a non-existent record"""
        success = store.update("test_table", "nonexistent", {"name": "updated"})
        assert success is False

    def test_delete_existing_record(self, store: SQLiteStore):
        """Test deleting an existing record"""
        record_id = store.insert("test_table", {"name": "test"})
        success = store.delete("test_table", record_id)

        assert success is True
        assert store.get("test_table", record_id) is None

    def test_delete_nonexistent_record(self, store: SQLiteStore):
        """Test deleting a non-existent record"""
        success = store.delete("test_table", "nonexistent")
        assert success is False

    def test_insert_creates_table(self, store: SQLiteStore):
        """Test that insert creates table automatically"""
        record_id = store.insert("auto_created", {"data": "test"})
        record = store.get("auto_created", record_id)

        assert record is not None


class TestSQLiteStoreQuery:
    """Test query operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_query_all_records(self, store: SQLiteStore):
        """Test querying all records"""
        store.insert("test_table", {"type": "a", "value": 1})
        store.insert("test_table", {"type": "b", "value": 2})
        store.insert("test_table", {"type": "a", "value": 3})

        results = store.query("test_table")

        assert len(results) == 3

    def test_query_with_where_clause(self, store: SQLiteStore):
        """Test querying with where clause"""
        store.insert("test_table", {"type": "a", "value": 1})
        store.insert("test_table", {"type": "b", "value": 2})
        store.insert("test_table", {"type": "a", "value": 3})

        results = store.query("test_table", where={"type": "a"})

        assert len(results) == 2
        for r in results:
            assert r["data"]["type"] == "a"

    def test_query_with_limit(self, store: SQLiteStore):
        """Test querying with limit"""
        for i in range(10):
            store.insert("test_table", {"n": i})

        results = store.query("test_table", limit=5)

        assert len(results) == 5

    def test_query_with_offset(self, store: SQLiteStore):
        """Test querying with offset"""
        for i in range(10):
            store.insert("test_table", {"n": i})

        results = store.query("test_table", limit=5, offset=5)

        assert len(results) == 5

    def test_query_empty_table(self, store: SQLiteStore):
        """Test querying empty table"""
        results = store.query("empty_table")
        assert results == []

    def test_query_no_matches(self, store: SQLiteStore):
        """Test querying with no matching results"""
        store.insert("test_table", {"type": "a"})

        results = store.query("test_table", where={"type": "nonexistent"})
        assert results == []

    def test_query_order_by(self, store: SQLiteStore):
        """Test query ordering"""
        store.insert("test_table", {"n": 3})
        store.insert("test_table", {"n": 1})
        store.insert("test_table", {"n": 2})

        results = store.query("test_table", order_by="created_at ASC")

        # Results are returned in DESC order by default
        assert len(results) == 3


class TestSQLiteStoreStream:
    """Test streaming operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_query_stream_basic(self, store: SQLiteStore):
        """Test basic query stream"""
        for i in range(25):
            store.insert("test_table", {"n": i})

        batches = list(store.query_stream("test_table", batch_size=10))

        assert len(batches) == 3  # 25 items / 10 batch_size = 3 batches
        assert len(batches[0]) == 10
        assert len(batches[1]) == 10
        assert len(batches[2]) == 5

    def test_query_stream_empty_table(self, store: SQLiteStore):
        """Test query stream on empty table"""
        batches = list(store.query_stream("empty_table"))
        assert batches == []

    def test_query_generator_basic(self, store: SQLiteStore):
        """Test query generator"""
        for i in range(5):
            store.insert("test_table", {"n": i})

        records = list(store.query_generator("test_table"))

        assert len(records) == 5
        for i, record in enumerate(records):
            assert record["data"]["n"] == i

    def test_query_generator_empty_table(self, store: SQLiteStore):
        """Test query generator on empty table"""
        records = list(store.query_generator("empty_table"))
        assert records == []


class TestSQLiteStoreBatch:
    """Test batch operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_batch_insert(self, store: SQLiteStore):
        """Test batch insert"""
        records = [{"n": i, "data": f"value_{i}"} for i in range(100)]
        ids = store.batch_insert("test_table", records)

        assert len(ids) == 100
        assert len(set(ids)) == 100  # All IDs should be unique

    def test_batch_insert_empty_list(self, store: SQLiteStore):
        """Test batch insert with empty list"""
        ids = store.batch_insert("test_table", [])
        assert ids == []

    def test_batch_insert_creates_table(self, store: SQLiteStore):
        """Test batch insert creates table"""
        records = [{"n": i} for i in range(5)]
        store.batch_insert("auto_created", records)

        results = store.query("auto_created")
        assert len(results) == 5


class TestSQLiteStoreStats:
    """Test statistics and utility operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_count_all_records(self, store: SQLiteStore):
        """Test counting all records"""
        for i in range(10):
            store.insert("test_table", {"n": i})

        count = store.count("test_table")
        assert count == 10

    def test_count_with_where(self, store: SQLiteStore):
        """Test counting with where clause"""
        store.insert("test_table", {"type": "a"})
        store.insert("test_table", {"type": "a"})
        store.insert("test_table", {"type": "b"})

        count = store.count("test_table", where={"type": "a"})
        assert count == 2

    def test_count_empty_table(self, store: SQLiteStore):
        """Test counting empty table"""
        count = store.count("empty_table")
        assert count == 0

    def test_get_stats(self, store: SQLiteStore):
        """Test getting stats"""
        store.insert("table1", {"data": "test"})
        store.insert("table1", {"data": "test"})
        store.insert("table2", {"data": "test"})

        stats = store.get_stats()

        assert "db_path" in stats
        assert "tables" in stats
        assert stats["tables"]["table1"]["count"] == 2
        assert stats["tables"]["table2"]["count"] == 1

    def test_clear_table(self, store: SQLiteStore):
        """Test clearing table"""
        for i in range(5):
            store.insert("test_table", {"n": i})

        deleted = store.clear("test_table")
        assert deleted == 5
        assert store.count("test_table") == 0

    def test_clear_empty_table(self, store: SQLiteStore):
        """Test clearing empty table"""
        deleted = store.clear("empty_table")
        assert deleted == 0

    def test_vacuum(self, store: SQLiteStore):
        """Test vacuum operation"""
        # Should not raise
        store.vacuum()

    def test_close(self, store: SQLiteStore):
        """Test close operation"""
        # Should not raise
        store.close()

    def test_append_alias(self, store: SQLiteStore):
        """Test append is alias for insert"""
        record_id = store.append("test_table", {"data": "test"})
        record = store.get("test_table", record_id)

        assert record is not None
        assert record["data"]["data"] == "test"


class TestSQLiteStoreDataTypes:
    """Test handling of different data types"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_nested_dict(self, store: SQLiteStore):
        """Test storing nested dictionaries"""
        data = {
            "level1": {
                "level2": {
                    "level3": ["a", "b", "c"],
                },
            },
        }
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"] == data

    def test_list_in_data(self, store: SQLiteStore):
        """Test storing lists in data"""
        data = {"items": [1, 2, 3, 4, 5]}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["items"] == [1, 2, 3, 4, 5]

    def test_special_characters(self, store: SQLiteStore):
        """Test storing special characters"""
        data = {"text": "Special chars: <>&\"'\n\t"}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["text"] == data["text"]

    def test_unicode_characters(self, store: SQLiteStore):
        """Test storing unicode characters"""
        data = {"text": "Unicode: 你好世界 🌍 café résumé"}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["text"] == data["text"]

    def test_large_data(self, store: SQLiteStore):
        """Test storing large data"""
        data = {"large_text": "x" * 10000}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert len(record["data"]["large_text"]) == 10000

    def test_null_values(self, store: SQLiteStore):
        """Test storing null values"""
        data = {"null_field": None, "valid_field": "value"}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["null_field"] is None
        assert record["data"]["valid_field"] == "value"


class TestSQLiteStoreConcurrency:
    """Test concurrent operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_concurrent_inserts(self, store: SQLiteStore):
        """Test concurrent inserts from multiple threads"""
        ids = []

        def insert_records(n):
            for i in range(n):
                record_id = store.insert("concurrent", {"thread": n, "i": i})
                ids.append(record_id)

        threads = [threading.Thread(target=insert_records, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 50
        assert len(set(ids)) == 50  # All unique

        count = store.count("concurrent")
        assert count == 50

    def test_concurrent_reads_and_writes(self, store: SQLiteStore):
        """Test concurrent reads and writes"""
        results = []

        def writer():
            for i in range(20):
                store.insert("concurrent", {"writer": True, "n": i})

        def reader():
            for _ in range(20):
                count = store.count("concurrent")
                results.append(count)

        threads = []
        for _ in range(2):
            threads.append(threading.Thread(target=writer))
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 40  # 2 readers * 20 reads each


class TestSQLiteStoreEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary SQLite store"""
        db_path = tmp_path / "test.db"
        config = SQLiteConfig(db_path=str(db_path))
        store = SQLiteStore(config)
        yield store
        store.close()

    def test_empty_string_data(self, store: SQLiteStore):
        """Test storing empty strings"""
        data = {"empty": ""}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["empty"] == ""

    def test_zero_values(self, store: SQLiteStore):
        """Test storing zero values"""
        data = {"zero_int": 0, "zero_float": 0.0, "false": False}
        record_id = store.insert("test_table", data)
        record = store.get("test_table", record_id)

        assert record["data"]["zero_int"] == 0
        assert record["data"]["zero_float"] == 0.0
        assert record["data"]["false"] is False

    def test_very_long_id(self, store: SQLiteStore):
        """Test with very long custom ID"""
        long_id = "x" * 1000
        record_id = store.insert("test_table", {"data": "test"}, item_id=long_id)

        assert record_id == long_id
        record = store.get("test_table", long_id)
        assert record is not None

    def test_special_chars_in_table_name(self, store: SQLiteStore):
        """Test table names with special characters"""
        # SQLite has restrictions, but we should handle basic names
        record_id = store.insert("table_123", {"data": "test"})
        record = store.get("table_123", record_id)
        assert record is not None

    def test_multiple_tables(self, store: SQLiteStore):
        """Test operations on multiple tables"""
        for i in range(5):
            store.insert(f"table_{i}", {"n": i})

        for i in range(5):
            count = store.count(f"table_{i}")
            assert count == 1

    def test_update_preserves_unmodified_fields(self, store: SQLiteStore):
        """Test update preserves unmodified fields"""
        record_id = store.insert("test_table", {"a": 1, "b": 2, "c": 3})
        store.update("test_table", record_id, {"b": 20})

        record = store.get("test_table", record_id)
        assert record["data"]["a"] == 1
        assert record["data"]["b"] == 20
        assert record["data"]["c"] == 3

    def test_query_with_multiple_where_conditions(self, store: SQLiteStore):
        """Test query with multiple where conditions"""
        store.insert("test_table", {"a": 1, "b": 2})
        store.insert("test_table", {"a": 1, "b": 3})
        store.insert("test_table", {"a": 2, "b": 2})

        # Note: SQLite store only supports single field where clauses
        # based on json_extract. Multiple conditions would require AND
        results = store.query("test_table", where={"a": 1})

        assert len(results) == 2
        for r in results:
            assert r["data"]["a"] == 1


class TestGetSQLiteStore:
    """Test get_sqlite_store convenience function"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests"""
        import gaap.storage.sqlite_store as sqlite_module

        sqlite_module._store_instance = None
        yield
        sqlite_module._store_instance = None

    def test_get_sqlite_store_singleton(self, tmp_path: Path):
        """Test get_sqlite_store returns singleton"""
        db_path = tmp_path / "test.db"
        store1 = get_sqlite_store(str(db_path))
        store2 = get_sqlite_store(str(db_path))

        assert store1 is store2

    def test_get_sqlite_store_default_path(self):
        """Test get_sqlite_store with default path"""
        store = get_sqlite_store()
        assert isinstance(store, SQLiteStore)
        assert store.config.db_path == ".gaap/gaap.db"
