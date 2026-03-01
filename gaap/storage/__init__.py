"""
GAAP Storage Module

Implements: docs/evolution_plan_2026/43_STORAGE_AUDIT_SPEC.md
- Atomic writes for data integrity
- Pydantic validation
- SQLite hybrid for high-volume data
- Automatic backups
"""

from .atomic import AtomicWriter, atomic_write
from .helpers import get_config, load_config, load_history, load_stats, save_config
from .json_store import JSONStore, ValidatedJSONStore, get_store
from .sqlite_store import SQLiteConfig, SQLiteStore, get_sqlite_store

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
