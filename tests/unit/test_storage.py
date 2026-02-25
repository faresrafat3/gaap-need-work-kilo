"""
Tests for GAAP Storage Module (Atomic Writes)
==============================================

Implements tests for:
- docs/evolution_plan_2026/43_STORAGE_AUDIT_SPEC.md
"""

import json
import os
import tempfile
import threading
from pathlib import Path

import pytest
from pydantic import BaseModel

from gaap.storage.atomic import atomic_write, atomic_writer, AtomicWriter
from gaap.storage.json_store import JSONStore, get_store


class TestAtomicWrite:
    """Tests for atomic_write function."""

    def test_write_string_content(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        content = '{"key": "value"}'

        result = atomic_write(target, content)

        assert result is True
        assert target.exists()
        assert target.read_text() == content

    def test_write_bytes_content(self, tmp_path: Path) -> None:
        target = tmp_path / "test.bin"
        content = b"binary data"

        result = atomic_write(target, content)

        assert result is True
        assert target.exists()
        assert target.read_bytes() == content

    def test_no_temp_file_on_success(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        atomic_write(target, '{"data": 1}')

        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_no_temp_file_on_failure(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "test.json"
        os.makedirs(tmp_path / "nested", exist_ok=True)

        atomic_write(target, '{"data": 1}')

        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists()

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "deeply" / "nested" / "dir" / "test.json"

        result = atomic_write(target, '{"created": true}')

        assert result is True
        assert target.exists()

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        target.write_text('{"old": true}')

        result = atomic_write(target, '{"new": true}')

        assert result is True
        data = json.loads(target.read_text())
        assert data == {"new": True}


class TestAtomicWriter:
    """Tests for atomic_writer context manager."""

    def test_context_manager_write(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"

        with atomic_writer(target) as f:
            f.write('{"context": "manager"}')  # type: ignore[arg-type]

        assert target.exists()
        assert json.loads(target.read_text()) == {"context": "manager"}

    def test_binary_mode(self, tmp_path: Path) -> None:
        target = tmp_path / "test.bin"

        with atomic_writer(target, mode="wb") as f:
            f.write(b"binary")  # type: ignore[arg-type]

        assert target.read_bytes() == b"binary"

    def test_cleanup_on_exception(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"

        try:
            with atomic_writer(target) as f:
                f.write("partial")  # type: ignore[arg-type]
                raise ValueError("Simulated error")
        except ValueError:
            pass

        tmp_file = target.with_suffix(".json.tmp")
        assert not tmp_file.exists()


class TestAtomicWriterClass:
    """Tests for AtomicWriter class."""

    def test_write_json(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)
        data = {"key": "value", "number": 42}

        result = writer.write_json("test", data)

        assert result is True
        loaded = writer.read_json("test")
        assert loaded == data

    def test_write_creates_backup(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)
        writer.write_json("test", {"v": 1})

        result = writer.write_json("test", {"v": 2}, backup=True)

        assert result is True
        backups = list((tmp_path / "backups").glob("*_test.json"))
        assert len(backups) == 1

    def test_max_backups_limit(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path, max_backups=2)

        for i in range(5):
            writer.write_json("test", {"v": i}, backup=True)

        backups = list((tmp_path / "backups").glob("*_test.json"))
        assert len(backups) <= 2

    def test_read_nonexistent(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)

        result = writer.read_json("nonexistent")

        assert result is None

    def test_exists(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)

        assert not writer.exists("test")
        writer.write_json("test", {"data": 1})
        assert writer.exists("test")

    def test_delete(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)
        writer.write_json("test", {"data": 1})

        result = writer.delete("test")

        assert result is True
        assert not writer.exists("test")


class SampleModel(BaseModel):
    name: str
    value: int


class TestAtomicWriterPydantic:
    """Tests for Pydantic model support."""

    def test_write_model(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)
        model = SampleModel(name="test", value=42)

        result = writer.write_model("sample", model)

        assert result is True

    def test_read_model(self, tmp_path: Path) -> None:
        writer = AtomicWriter(tmp_path)
        model = SampleModel(name="test", value=42)
        writer.write_model("sample", model)

        loaded = writer.read_model("sample", SampleModel)

        assert loaded is not None
        assert loaded.name == "test"
        assert loaded.value == 42


class TestJSONStore:
    """Tests for JSONStore class."""

    def test_init(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        assert store.base_dir.exists()

    def test_save_and_load(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        data = {"key": "value", "number": 42}

        store.save("test", data)
        loaded = store.load("test")

        assert loaded == data

    def test_load_nonexistent(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        result = store.load("nonexistent", default={"default": True})

        assert result == {"default": True}

    def test_append(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        item_id = store.append("history", {"message": "Hello"})

        assert item_id is not None
        assert len(item_id) == 8

        data = store.load("history")
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["message"] == "Hello"
        assert data[0]["id"] == item_id
        assert "timestamp" in data[0]

    def test_get_by_id(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        item_id = store.append("items", {"name": "Test"})
        assert item_id is not None

        item = store.get_by_id("items", item_id)

        assert item is not None
        assert item["name"] == "Test"

    def test_get_by_id_not_found(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        item = store.get_by_id("items", "nonexistent")

        assert item is None

    def test_delete_by_id(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        item_id = store.append("items", {"name": "Test"})
        assert item_id is not None

        result = store.delete_by_id("items", item_id)

        assert result is True
        assert store.get_by_id("items", item_id) is None

    def test_update(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        item_id = store.append("items", {"name": "Test", "count": 1})
        assert item_id is not None

        result = store.update("items", item_id, {"count": 5})

        assert result is True
        item = store.get_by_id("items", item_id)
        assert item is not None
        assert item["count"] == 5
        assert "updated_at" in item

    def test_clear(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        store.append("items", {"name": "Test1"})
        store.append("items", {"name": "Test2"})

        store.clear("items")

        data = store.load("items")
        assert data == []

    def test_search(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        store.append("items", {"category": "a", "name": "Item 1"})
        store.append("items", {"category": "b", "name": "Item 2"})
        store.append("items", {"category": "a", "name": "Item 3"})

        results = store.search("items", {"category": "a"})

        assert len(results) == 2

    def test_count(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        store.append("items", {"name": "Test1"})
        store.append("items", {"name": "Test2"})

        assert store.count("items") == 2

    def test_thread_safety(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        errors: list[Exception] = []

        def writer(i: int) -> None:
            try:
                for j in range(10):
                    store.append("items", {"thread": i, "item": j})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count("items") == 50

    def test_get_lock_uniqueness(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        lock1 = store._get_lock("test1")
        lock2 = store._get_lock("test2")
        lock1_again = store._get_lock("test1")

        assert lock1 is lock1_again
        assert lock1 is not lock2


class TestGetStore:
    """Tests for singleton get_store function."""

    def test_singleton(self, tmp_path: Path) -> None:
        import gaap.storage.json_store as module

        module._store_instance = None

        store1 = get_store(tmp_path)
        store2 = get_store(tmp_path)

        assert store1 is store2

        module._store_instance = None
