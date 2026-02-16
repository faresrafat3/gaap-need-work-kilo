"""
Storage Module - JSON-based data persistence
"""

from .json_store import (
    JSONStore,
    get_config,
    get_store,
    load_config,
    load_history,
    load_stats,
    save_config,
    save_history,
    save_stats,
)

__all__ = [
    "JSONStore",
    "get_config",
    "get_store",
    "save_history",
    "load_history",
    "save_config",
    "load_config",
    "save_stats",
    "load_stats",
]
