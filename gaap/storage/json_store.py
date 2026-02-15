"""
JSON-based Storage for GAAP

Provides simple JSON file storage for:
- Conversation history
- Configuration
- Statistics
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any


class JSONStore:
    """
    Thread-safe JSON storage

    Usage:
        store = JSONStore()
        store.save("history", {"id": "123", "message": "Hello"})
        data = store.load("history")
    """

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.home() / ".gaap"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, Lock] = {}

    def _get_lock(self, name: str) -> Lock:
        if name not in self._locks:
            self._locks[name] = Lock()
        return self._locks[name]

    def _get_path(self, name: str) -> Path:
        return self.base_dir / f"{name}.json"

    def save(self, name: str, data: Any) -> None:
        """Save data to JSON file"""
        with self._get_lock(name):
            path = self._get_path(name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load(self, name: str, default: Any = None) -> Any:
        """Load data from JSON file"""
        with self._get_lock(name):
            path = self._get_path(name)
            if not path.exists():
                return default
            with open(path, encoding="utf-8") as f:
                return json.load(f)

    def append(self, name: str, item: dict[str, Any]) -> str:
        """Append item to a list, returns item ID"""
        data = self.load(name, [])
        item_id = str(uuid.uuid4())[:8]
        item["id"] = item_id
        item["timestamp"] = datetime.now().isoformat()
        data.append(item)
        self.save(name, data)
        return item_id

    def get_by_id(self, name: str, item_id: str) -> dict[str, Any] | None:
        """Get item by ID"""
        data = self.load(name, [])
        for item in data:
            if item.get("id") == item_id:
                return item
        return None

    def delete_by_id(self, name: str, item_id: str) -> bool:
        """Delete item by ID"""
        with self._get_lock(name):
            data = self.load(name, [])
            original_len = len(data)
            data = [item for item in data if item.get("id") != item_id]
            if len(data) < original_len:
                self.save(name, data)
                return True
            return False

    def clear(self, name: str) -> None:
        """Clear all data in a file"""
        self.save(name, [])

    def update(self, name: str, item_id: str, updates: dict[str, Any]) -> bool:
        """Update item by ID"""
        with self._get_lock(name):
            data = self.load(name, [])
            for item in data:
                if item.get("id") == item_id:
                    item.update(updates)
                    item["updated_at"] = datetime.now().isoformat()
                    self.save(name, data)
                    return True
            return False

    def search(self, name: str, key: str, value: Any) -> list[dict[str, Any]]:
        """Search items by key-value"""
        data = self.load(name, [])
        return [item for item in data if item.get(key) == value]

    def count(self, name: str) -> int:
        """Count items"""
        data = self.load(name, [])
        return len(data) if isinstance(data, list) else 0


_store: JSONStore | None = None


def get_store() -> JSONStore:
    """Get singleton store instance"""
    global _store
    if _store is None:
        _store = JSONStore()
    return _store


# Convenience functions for common operations


def save_history(
    role: str,
    content: str,
    provider: str = None,
    model: str = None,
    tokens: int = None,
    cost: float = None,
) -> str:
    """Save a message to history"""
    store = get_store()
    return store.append(
        "history",
        {
            "role": role,
            "content": content,
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "cost": cost,
        },
    )


def load_history(limit: int = 100) -> list[dict[str, Any]]:
    """Load conversation history"""
    store = get_store()
    data = store.load("history", [])
    if isinstance(data, list):
        return data[-limit:] if limit else data
    return []


def clear_history() -> None:
    """Clear all history"""
    store = get_store()
    store.clear("history")


def save_config(key: str, value: Any) -> None:
    """Save a config value"""
    store = get_store()
    config = store.load("config", {})
    config[key] = value
    store.save("config", config)


def load_config() -> dict[str, Any]:
    """Load all config"""
    store = get_store()
    return store.load("config", {})


def get_config(key: str, default: Any = None) -> Any:
    """Get a config value"""
    config = load_config()
    return config.get(key, default)


def save_stats(
    requests: int = None,
    tokens: int = None,
    cost: float = None,
    errors: int = None,
) -> None:
    """Update statistics"""
    store = get_store()
    stats = store.load(
        "stats",
        {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_errors": 0,
            "by_provider": {},
            "by_model": {},
            "daily": {},
        },
    )

    if requests:
        stats["total_requests"] += requests
    if tokens:
        stats["total_tokens"] += tokens
    if cost:
        stats["total_cost"] += cost
    if errors:
        stats["total_errors"] += errors

    today = datetime.now().strftime("%Y-%m-%d")
    if today not in stats["daily"]:
        stats["daily"][today] = {"requests": 0, "tokens": 0, "cost": 0.0}

    if requests:
        stats["daily"][today]["requests"] += requests
    if tokens:
        stats["daily"][today]["tokens"] += tokens
    if cost:
        stats["daily"][today]["cost"] += cost

    store.save("stats", stats)


def load_stats() -> dict[str, Any]:
    """Load statistics"""
    store = get_store()
    return store.load(
        "stats",
        {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_errors": 0,
            "by_provider": {},
            "by_model": {},
            "daily": {},
        },
    )


def record_request(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    success: bool = True,
) -> None:
    """Record a request with full details"""
    store = get_store()
    stats = load_stats()

    stats["total_requests"] += 1
    stats["total_tokens"] += input_tokens + output_tokens
    stats["total_cost"] += cost

    if not success:
        stats["total_errors"] += 1

    if provider not in stats["by_provider"]:
        stats["by_provider"][provider] = {"requests": 0, "tokens": 0, "cost": 0.0}
    stats["by_provider"][provider]["requests"] += 1
    stats["by_provider"][provider]["tokens"] += input_tokens + output_tokens
    stats["by_provider"][provider]["cost"] += cost

    if model not in stats["by_model"]:
        stats["by_model"][model] = {"requests": 0, "tokens": 0, "cost": 0.0}
    stats["by_model"][model]["requests"] += 1
    stats["by_model"][model]["tokens"] += input_tokens + output_tokens
    stats["by_model"][model]["cost"] += cost

    today = datetime.now().strftime("%Y-%m-%d")
    if today not in stats["daily"]:
        stats["daily"][today] = {"requests": 0, "tokens": 0, "cost": 0.0, "errors": 0}
    stats["daily"][today]["requests"] += 1
    stats["daily"][today]["tokens"] += input_tokens + output_tokens
    stats["daily"][today]["cost"] += cost
    if not success:
        stats["daily"][today]["errors"] += 1

    store.save("stats", stats)
