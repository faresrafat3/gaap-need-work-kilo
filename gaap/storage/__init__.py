"""
GAAP Storage Module

Implements: docs/evolution_plan_2026/43_STORAGE_AUDIT_SPEC.md
- Atomic writes for data integrity
- Pydantic validation
- SQLite hybrid for high-volume data
- Automatic backups
"""

from .atomic import atomic_write, AtomicWriter
from .json_store import JSONStore, get_store, ValidatedJSONStore
from .sqlite_store import SQLiteStore, SQLiteConfig, get_sqlite_store
from .helpers import load_history, load_stats, load_config, get_config, save_config

__all__ = [
    "atomic_write",
    "AtomicWriter",
    "JSONStore",
    "ValidatedJSONStore",
    "get_store",
    "SQLiteStore",
    "SQLiteConfig",
    "get_sqlite_store",
    "load_history",
    "load_stats",
    "load_config",
    "get_config",
    "save_config",
]
