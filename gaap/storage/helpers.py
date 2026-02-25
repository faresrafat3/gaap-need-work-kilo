"""
Storage Helper Functions
========================

High-level convenience functions for common storage operations.
"""

from typing import Any

from gaap.storage.json_store import get_store


def load_history(limit: int = 1000) -> list[dict[str, Any]]:
    """Load conversation history.

    Args:
        limit: Maximum number of items to return

    Returns:
        List of history items
    """
    store = get_store()
    data = store.load("history", default=[])
    if isinstance(data, list):
        return data[-limit:] if len(data) > limit else data
    return []


def load_stats() -> dict[str, Any]:
    """Load usage statistics.

    Returns:
        Statistics dictionary
    """
    store = get_store()
    data = store.load("stats", default={})
    if isinstance(data, dict):
        return data
    return {
        "total_requests": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
    }


def load_config() -> dict[str, Any]:
    """Load configuration.

    Returns:
        Configuration dictionary
    """
    store = get_store()
    data = store.load("config", default={})
    if isinstance(data, dict):
        return data
    return {}


def get_config(key: str) -> Any:
    """Get a specific config value.

    Args:
        key: Configuration key

    Returns:
        Configuration value or None
    """
    config = load_config()
    return config.get(key)


def save_config(key: str, value: Any) -> bool:
    """Save a configuration value.

    Args:
        key: Configuration key
        value: Value to save

    Returns:
        True if successful
    """
    store = get_store()
    config = load_config()
    config[key] = value
    return store.save("config", config)
