"""
JSON Store with Atomic Writes
=============================

Implements: docs/evolution_plan_2026/43_STORAGE_AUDIT_SPEC.md

Features:
- Atomic writes to prevent data corruption
- Thread-safe operations with locks
- ID-based item management
- Automatic timestamping
"""

import json
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from gaap.storage.atomic import atomic_write

logger = logging.getLogger("gaap.storage.json_store")


class JSONStore:
    """
    Thread-safe JSON storage with atomic writes.

    Features:
    - Atomic file writes
    - ID-based item tracking
    - Automatic timestamps
    - Thread-safe locks

    Example:
        >>> store = JSONStore(base_dir=".gaap/data")
        >>> store.save("config", {"api_key": "secret"})
        >>> config = store.load("config")
        >>> item_id = store.append("history", {"action": "query"})
    """

    def __init__(self, base_dir: Path | str = ".gaap/storage") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

        self._logger = logger

    def _get_lock(self, name: str) -> threading.Lock:
        """Get or create a lock for a specific store."""
        with self._global_lock:
            if name not in self._locks:
                self._locks[name] = threading.Lock()
            return self._locks[name]

    def _get_path(self, name: str) -> Path:
        """Get file path for a store."""
        return self.base_dir / f"{name}.json"

    def _generate_id(self) -> str:
        """Generate unique 8-character ID."""
        return uuid.uuid4().hex[:8]

    def save(self, name: str, data: dict[str, Any] | list[Any]) -> bool:
        """
        Save data to a JSON file atomically.

        Args:
            name: Store name (without extension)
            data: Data to save

        Returns:
            True if successful
        """
        lock = self._get_lock(name)
        path = self._get_path(name)

        with lock:
            content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            return atomic_write(path, content)

    def load(
        self,
        name: str,
        default: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any] | list[Any] | None:
        """
        Load data from a JSON file.

        Args:
            name: Store name (without extension)
            default: Default value if file doesn't exist

        Returns:
            Loaded data or default
        """
        path = self._get_path(name)

        if not path.exists():
            return default

        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except Exception as e:
            self._logger.error(f"Failed to load {name}: {e}")
            return default

    def append(self, name: str, item: dict[str, Any]) -> str | None:
        """
        Append item to a list store.

        Args:
            name: Store name
            item: Item to append

        Returns:
            Generated item ID or None on failure
        """
        lock = self._get_lock(name)
        path = self._get_path(name)

        with lock:
            data = []

            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            data = loaded
                except Exception as e:
                    self._logger.error(f"Failed to load {name} for append: {e}")

            item_id = self._generate_id()
            item_with_meta = {
                **item,
                "id": item_id,
                "timestamp": datetime.now().isoformat(),
            }

            data.append(item_with_meta)

            content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            if atomic_write(path, content):
                return item_id

            return None

    def get_by_id(self, name: str, item_id: str) -> dict[str, Any] | None:
        """
        Get item by ID from a list store.

        Args:
            name: Store name
            item_id: Item ID

        Returns:
            Item dict or None if not found
        """
        data = self.load(name, default=[])

        if not isinstance(data, list):
            return None

        for item in data:
            if isinstance(item, dict) and item.get("id") == item_id:
                return item

        return None

    def delete_by_id(self, name: str, item_id: str) -> bool:
        """
        Delete item by ID from a list store.

        Args:
            name: Store name
            item_id: Item ID

        Returns:
            True if deleted
        """
        lock = self._get_lock(name)
        path = self._get_path(name)

        with lock:
            if not path.exists():
                return False

            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    return False

                original_len = len(data)
                data = [
                    item
                    for item in data
                    if not (isinstance(item, dict) and item.get("id") == item_id)
                ]

                if len(data) == original_len:
                    return False

                content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                return atomic_write(path, content)

            except Exception as e:
                self._logger.error(f"Failed to delete {item_id} from {name}: {e}")
                return False

    def update(self, name: str, item_id: str, updates: dict[str, Any]) -> bool:
        """
        Update item by ID in a list store.

        Args:
            name: Store name
            item_id: Item ID
            updates: Fields to update

        Returns:
            True if updated
        """
        lock = self._get_lock(name)
        path = self._get_path(name)

        with lock:
            if not path.exists():
                return False

            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    return False

                found = False
                for item in data:
                    if isinstance(item, dict) and item.get("id") == item_id:
                        item.update(updates)
                        item["updated_at"] = datetime.now().isoformat()
                        found = True
                        break

                if not found:
                    return False

                content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
                return atomic_write(path, content)

            except Exception as e:
                self._logger.error(f"Failed to update {item_id} in {name}: {e}")
                return False

    def clear(self, name: str) -> bool:
        """
        Clear all items from a store.

        Args:
            name: Store name

        Returns:
            True if cleared
        """
        return self.save(name, [])

    def search(
        self,
        name: str,
        query: dict[str, Any],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Search items in a list store.

        Args:
            name: Store name
            query: Key-value pairs to match
            limit: Maximum results

        Returns:
            Matching items
        """
        data = self.load(name, default=[])

        if not isinstance(data, list):
            return []

        results = []
        for item in data:
            if isinstance(item, dict):
                match = all(item.get(k) == v for k, v in query.items())
                if match:
                    results.append(item)
                    if len(results) >= limit:
                        break

        return results

    def count(self, name: str) -> int:
        """Count items in a list store."""
        data = self.load(name, default=[])
        if isinstance(data, list):
            return len(data)
        return 0


_store_instance: JSONStore | None = None
_store_lock = threading.Lock()


def get_store(base_dir: Path | str = ".gaap/storage") -> JSONStore:
    """
    Get singleton JSONStore instance.

    Args:
        base_dir: Base directory for storage

    Returns:
        JSONStore instance
    """
    global _store_instance

    with _store_lock:
        if _store_instance is None:
            _store_instance = JSONStore(base_dir=base_dir)
        return _store_instance


class ValidatedJSONStore(JSONStore):
    """
    JSONStore with Pydantic model validation.

    Example:
        >>> from pydantic import BaseModel
        >>> class Item(BaseModel):
        ...     name: str
        ...     value: int
        >>> store = ValidatedJSONStore(model=Item)
        >>> store.append_validated("items", Item(name="test", value=42))
    """

    def __init__(
        self,
        base_dir: Path | str = ".gaap/storage",
        model: type[BaseModel] | None = None,
    ) -> None:
        super().__init__(base_dir=base_dir)
        self.model = model

    def append_validated(self, name: str, item: BaseModel) -> str | None:
        """Append a validated Pydantic model."""
        if self.model and not isinstance(item, self.model):
            raise ValueError(f"Item must be instance of {self.model.__name__}")

        return self.append(name, item.model_dump())

    def load_validated(self, name: str) -> list[BaseModel] | None:
        """Load and validate items."""
        if not self.model:
            raise ValueError("No model configured")

        data = self.load(name, default=[])
        if not isinstance(data, list):
            return None

        validated = []
        for item in data:
            try:
                validated.append(self.model.model_validate(item))
            except Exception as e:
                self._logger.warning(f"Validation failed for item in {name}: {e}")

        return validated
