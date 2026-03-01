"""
Comprehensive tests for gaap/storage/json_store.py module
Tests JSONStore, ValidatedJSONStore, and all CRUD operations
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from gaap.storage.json_store import (
    JSONStore,
    ValidatedJSONStore,
    get_store,
)


class TestJSONStoreBasic:
    """Test basic JSONStore operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_initialization_creates_directory(self, tmp_path: Path):
        """Test that initialization creates base directory"""
        base_dir = tmp_path / "new_storage"
        store = JSONStore(base_dir=base_dir)
        assert base_dir.exists()
        assert base_dir.is_dir()

    def test_initialization_with_string_path(self, tmp_path: Path):
        """Test initialization with string path"""
        store = JSONStore(base_dir=str(tmp_path))
        assert store.base_dir == tmp_path

    def test_initialization_default_path(self):
        """Test initialization with default path"""
        store = JSONStore()
        assert store.base_dir.name == "storage"
        assert store.base_dir.parent.name == ".gaap"


class TestJSONStoreSaveLoad:
    """Test save and load operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_save_dict(self, store: JSONStore, tmp_path: Path):
        """Test saving a dictionary"""
        data = {"key": "value", "number": 42}
        result = store.save("config", data)

        assert result is True
        assert (tmp_path / "config.json").exists()

    def test_save_list(self, store: JSONStore, tmp_path: Path):
        """Test saving a list"""
        data = [{"id": 1}, {"id": 2}]
        result = store.save("items", data)

        assert result is True
        assert (tmp_path / "items.json").exists()

    def test_save_creates_valid_json(self, store: JSONStore, tmp_path: Path):
        """Test that saved data is valid JSON"""
        data = {"key": "value"}
        store.save("config", data)

        with open(tmp_path / "config.json") as f:
            loaded = json.load(f)

        assert loaded == data

    def test_load_existing_file(self, store: JSONStore):
        """Test loading an existing file"""
        data = {"key": "value"}
        store.save("config", data)

        loaded = store.load("config")
        assert loaded == data

    def test_load_nonexistent_file_returns_default(self, store: JSONStore):
        """Test loading non-existent file returns default"""
        loaded = store.load("nonexistent")
        assert loaded is None

    def test_load_with_custom_default(self, store: JSONStore):
        """Test loading with custom default"""
        default = {"default": True}
        loaded = store.load("nonexistent", default=default)
        assert loaded == default

    def test_load_with_list_default(self, store: JSONStore):
        """Test loading with list default"""
        default = []
        loaded = store.load("nonexistent", default=default)
        assert loaded == default

    def test_load_corrupted_file_returns_default(self, store: JSONStore, tmp_path: Path):
        """Test loading corrupted file returns default"""
        # Write invalid JSON
        (tmp_path / "corrupted.json").write_text("{invalid json}")

        loaded = store.load("corrupted", default={"default": True})
        assert loaded == {"default": True}

    def test_save_overwrites_existing(self, store: JSONStore):
        """Test that save overwrites existing file"""
        store.save("config", {"version": 1})
        store.save("config", {"version": 2})

        loaded = store.load("config")
        assert loaded == {"version": 2}

    def test_save_unicode_content(self, store: JSONStore):
        """Test saving unicode content"""
        data = {"text": "Hello 世界 🌍 café"}
        store.save("config", data)

        loaded = store.load("config")
        assert loaded["text"] == "Hello 世界 🌍 café"


class TestJSONStoreAppend:
    """Test append operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_append_creates_new_file(self, store: JSONStore, tmp_path: Path):
        """Test append creates new file if not exists"""
        item = {"name": "test"}
        item_id = store.append("items", item)

        assert item_id is not None
        assert len(item_id) == 8
        assert (tmp_path / "items.json").exists()

    def test_append_adds_id_and_timestamp(self, store: JSONStore):
        """Test append adds id and timestamp"""
        item = {"name": "test"}
        item_id = store.append("items", item)

        items = store.load("items", default=[])
        assert len(items) == 1
        assert items[0]["id"] == item_id
        assert "timestamp" in items[0]

    def test_append_multiple_items(self, store: JSONStore):
        """Test appending multiple items"""
        ids = []
        for i in range(5):
            item_id = store.append("items", {"index": i})
            ids.append(item_id)

        items = store.load("items", default=[])
        assert len(items) == 5
        assert len(set(ids)) == 5  # All unique

    def test_append_preserves_existing_items(self, store: JSONStore):
        """Test append preserves existing items"""
        store.append("items", {"name": "first"})
        store.append("items", {"name": "second"})

        items = store.load("items", default=[])
        assert len(items) == 2
        assert items[0]["name"] == "first"
        assert items[1]["name"] == "second"

    def test_append_returns_none_on_failure(self, store: JSONStore, tmp_path: Path):
        """Test append returns None on failure"""
        with patch("gaap.storage.json_store.atomic_write", return_value=False):
            item_id = store.append("items", {"name": "test"})
            assert item_id is None


class TestJSONStoreGetById:
    """Test get_by_id operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_get_by_id_existing(self, store: JSONStore):
        """Test getting item by existing ID"""
        item = {"name": "test", "value": 42}
        item_id = store.append("items", item)

        found = store.get_by_id("items", item_id)
        assert found is not None
        assert found["name"] == "test"
        assert found["value"] == 42

    def test_get_by_id_nonexistent(self, store: JSONStore):
        """Test getting item by non-existent ID"""
        found = store.get_by_id("items", "nonexistent")
        assert found is None

    def test_get_by_id_nonexistent_file(self, store: JSONStore):
        """Test getting by ID when file doesn't exist"""
        found = store.get_by_id("nonexistent", "id")
        assert found is None

    def test_get_by_id_file_is_dict(self, store: JSONStore):
        """Test get_by_id when file contains dict (not list)"""
        store.save("config", {"key": "value"})
        found = store.get_by_id("config", "id")
        assert found is None


class TestJSONStoreDeleteById:
    """Test delete_by_id operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_delete_by_id_existing(self, store: JSONStore):
        """Test deleting item by existing ID"""
        item_id = store.append("items", {"name": "test"})
        result = store.delete_by_id("items", item_id)

        assert result is True
        assert store.get_by_id("items", item_id) is None

    def test_delete_by_id_nonexistent(self, store: JSONStore):
        """Test deleting item by non-existent ID"""
        store.append("items", {"name": "test"})
        result = store.delete_by_id("items", "nonexistent")

        assert result is False

    def test_delete_by_id_nonexistent_file(self, store: JSONStore):
        """Test delete by ID when file doesn't exist"""
        result = store.delete_by_id("nonexistent", "id")
        assert result is False

    def test_delete_one_of_many(self, store: JSONStore):
        """Test deleting one item from many"""
        id1 = store.append("items", {"name": "first"})
        id2 = store.append("items", {"name": "second"})
        id3 = store.append("items", {"name": "third"})

        store.delete_by_id("items", id2)

        items = store.load("items", default=[])
        assert len(items) == 2
        assert store.get_by_id("items", id1) is not None
        assert store.get_by_id("items", id3) is not None


class TestJSONStoreUpdate:
    """Test update operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_update_existing_item(self, store: JSONStore):
        """Test updating existing item"""
        item_id = store.append("items", {"name": "original", "value": 1})
        result = store.update("items", item_id, {"name": "updated", "value": 2})

        assert result is True

        item = store.get_by_id("items", item_id)
        assert item["name"] == "updated"
        assert item["value"] == 2

    def test_update_nonexistent_item(self, store: JSONStore):
        """Test updating non-existent item"""
        result = store.update("items", "nonexistent", {"name": "updated"})
        assert result is False

    def test_update_nonexistent_file(self, store: JSONStore):
        """Test update when file doesn't exist"""
        result = store.update("nonexistent", "id", {"name": "updated"})
        assert result is False

    def test_update_adds_updated_at(self, store: JSONStore):
        """Test update adds updated_at timestamp"""
        item_id = store.append("items", {"name": "original"})
        store.update("items", item_id, {"name": "updated"})

        item = store.get_by_id("items", item_id)
        assert "updated_at" in item

    def test_update_partial_fields(self, store: JSONStore):
        """Test updating only some fields"""
        item_id = store.append("items", {"name": "original", "value": 1, "keep": "this"})
        store.update("items", item_id, {"value": 2})

        item = store.get_by_id("items", item_id)
        assert item["name"] == "original"
        assert item["value"] == 2
        assert item["keep"] == "this"


class TestJSONStoreClear:
    """Test clear operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_clear_existing_list(self, store: JSONStore):
        """Test clearing existing list"""
        store.append("items", {"name": "test"})
        result = store.clear("items")

        assert result is True
        items = store.load("items", default=[])
        assert items == []

    def test_clear_empty_list(self, store: JSONStore):
        """Test clearing empty list"""
        result = store.clear("items")
        assert result is True


class TestJSONStoreSearch:
    """Test search operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_search_by_single_field(self, store: JSONStore):
        """Test searching by single field"""
        store.append("items", {"type": "a", "value": 1})
        store.append("items", {"type": "b", "value": 2})
        store.append("items", {"type": "a", "value": 3})

        results = store.search("items", {"type": "a"})

        assert len(results) == 2
        for r in results:
            assert r["type"] == "a"

    def test_search_by_multiple_fields(self, store: JSONStore):
        """Test searching by multiple fields"""
        store.append("items", {"type": "a", "status": "active"})
        store.append("items", {"type": "a", "status": "inactive"})
        store.append("items", {"type": "b", "status": "active"})

        results = store.search("items", {"type": "a", "status": "active"})

        assert len(results) == 1

    def test_search_no_matches(self, store: JSONStore):
        """Test search with no matches"""
        store.append("items", {"type": "a"})

        results = store.search("items", {"type": "nonexistent"})
        assert results == []

    def test_search_with_limit(self, store: JSONStore):
        """Test search with limit"""
        for i in range(10):
            store.append("items", {"type": "a", "index": i})

        results = store.search("items", {"type": "a"}, limit=5)

        assert len(results) == 5

    def test_search_nonexistent_file(self, store: JSONStore):
        """Test search on non-existent file"""
        results = store.search("nonexistent", {"key": "value"})
        assert results == []

    def test_search_file_is_dict(self, store: JSONStore):
        """Test search when file contains dict"""
        store.save("config", {"key": "value"})
        results = store.search("config", {"key": "value"})
        assert results == []


class TestJSONStoreCount:
    """Test count operations"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_count_empty_store(self, store: JSONStore):
        """Test count on empty store"""
        count = store.count("items")
        assert count == 0

    def test_count_with_items(self, store: JSONStore):
        """Test count with items"""
        for i in range(5):
            store.append("items", {"index": i})

        count = store.count("items")
        assert count == 5

    def test_count_file_is_dict(self, store: JSONStore):
        """Test count when file contains dict"""
        store.save("config", {"key": "value"})
        count = store.count("config")
        assert count == 0


class TestJSONStoreThreadSafety:
    """Test thread safety"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_concurrent_appends(self, store: JSONStore):
        """Test concurrent appends"""
        ids = []

        def append_items(n):
            for i in range(n):
                item_id = store.append("items", {"thread": threading.current_thread().name, "i": i})
                if item_id:
                    ids.append(item_id)

        threads = [threading.Thread(target=append_items, args=(10,)) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 50
        assert len(set(ids)) == 50

    def test_lock_per_store_name(self, store: JSONStore):
        """Test that different store names have different locks"""
        lock1 = store._get_lock("store1")
        lock2 = store._get_lock("store2")

        assert lock1 is not lock2

    def test_same_lock_for_same_name(self, store: JSONStore):
        """Test that same name returns same lock"""
        lock1 = store._get_lock("store1")
        lock2 = store._get_lock("store1")

        assert lock1 is lock2


class TestGetStore:
    """Test get_store convenience function"""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton between tests"""
        import gaap.storage.json_store as json_store_module

        json_store_module._store_instance = None
        yield
        json_store_module._store_instance = None

    def test_get_store_singleton(self, tmp_path: Path):
        """Test get_store returns singleton"""
        store1 = get_store(str(tmp_path))
        store2 = get_store(str(tmp_path))

        assert store1 is store2

    def test_get_store_default_path(self):
        """Test get_store with default path"""
        store = get_store()
        assert isinstance(store, JSONStore)


class TestValidatedJSONStore:
    """Test ValidatedJSONStore"""

    class TestItem(BaseModel):
        """Test model"""

        name: str
        value: int = Field(default=0)

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary validated JSON store"""
        store = ValidatedJSONStore(base_dir=tmp_path, model=self.TestItem)
        yield store

    def test_append_validated_valid_item(self, store: ValidatedJSONStore):
        """Test appending valid item"""
        item = self.TestItem(name="test", value=42)
        item_id = store.append_validated("items", item)

        assert item_id is not None

    def test_append_validated_invalid_item(self, store: ValidatedJSONStore):
        """Test appending invalid item raises error"""
        with pytest.raises(ValueError, match="Item must be instance of"):
            store.append_validated("items", {"name": "test"})

    def test_load_validated(self, store: ValidatedJSONStore):
        """Test loading validated items"""
        item = self.TestItem(name="test", value=42)
        store.append_validated("items", item)

        loaded = store.load_validated("items")
        assert len(loaded) == 1
        assert loaded[0].name == "test"
        assert loaded[0].value == 42

    def test_load_validated_no_model(self, tmp_path: Path):
        """Test load_validated without model raises error"""
        store = ValidatedJSONStore(base_dir=tmp_path)

        with pytest.raises(ValueError, match="No model configured"):
            store.load_validated("items")

    def test_load_validated_invalid_items(self, store: ValidatedJSONStore, tmp_path: Path):
        """Test load_validated skips invalid items"""
        # Manually add invalid item
        store.append("items", {"invalid": "data"})

        loaded = store.load_validated("items")
        assert loaded == []


class TestJSONStoreEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.fixture
    def store(self, tmp_path: Path):
        """Create a temporary JSON store"""
        store = JSONStore(base_dir=tmp_path)
        yield store

    def test_empty_string_as_id(self, store: JSONStore):
        """Test using empty string as custom ID"""
        item_id = store.append("items", {"name": "test"})
        # ID is auto-generated, not empty
        assert len(item_id) == 8

    def test_very_long_content(self, store: JSONStore):
        """Test with very long content"""
        long_text = "x" * 100000
        item_id = store.append("items", {"text": long_text})

        item = store.get_by_id("items", item_id)
        assert len(item["text"]) == 100000

    def test_nested_deep_structure(self, store: JSONStore):
        """Test with deeply nested structure"""
        data = {"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}
        item_id = store.append("items", data)

        item = store.get_by_id("items", item_id)
        assert item["level1"]["level2"]["level3"]["level4"]["level5"] == "deep"

    def test_special_float_values(self, store: JSONStore):
        """Test with special float values"""
        data = {"inf": float("inf"), "neg_inf": float("-inf")}
        store.save("config", data)

        loaded = store.load("config")
        # JSON converts inf to string via default=str
        assert loaded["inf"] is not None

    def test_datetime_serialization(self, store: JSONStore):
        """Test datetime serialization"""
        from datetime import datetime

        now = datetime.now()
        data = {"timestamp": now}
        item_id = store.append("items", data)

        item = store.get_by_id("items", item_id)
        # Datetime should be serialized to string
        assert isinstance(item["timestamp"], str)

    def test_bytes_serialization(self, store: JSONStore):
        """Test bytes serialization"""
        data = {"binary": b"binary data"}
        item_id = store.append("items", data)

        item = store.get_by_id("items", item_id)
        # Bytes should be serialized via default=str
        assert isinstance(item["binary"], str)

    def test_unicode_in_keys(self, store: JSONStore):
        """Test unicode in dictionary keys"""
        data = {"ключ": "value", "键": "value2"}
        item_id = store.append("items", data)

        item = store.get_by_id("items", item_id)
        assert item["ключ"] == "value"
        assert item["键"] == "value2"
