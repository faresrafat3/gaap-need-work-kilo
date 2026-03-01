"""
SQLite Store - High-Volume Data Storage
=======================================

Implements: docs/evolution_plan_2026/43_STORAGE_AUDIT_SPEC.md

Features:
- ACID compliance
- Efficient appends without full rewrites
- Instant lookups by ID
- Query support with indexes
- Thread-safe operations
- Streaming queries for large datasets

Usage:
    store = SQLiteStore(db_path=".gaap/gaap.db")

    # Insert
    record_id = store.insert("history", {"action": "query", "result": "success"})

    # Query
    results = store.query("history", where={"action": "query"}, limit=10)

    # Stream large results
    for batch in store.query_stream("history", batch_size=100):
        process_batch(batch)
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger("gaap.storage.sqlite")


@dataclass
class SQLiteConfig:
    db_path: str = ".gaap/gaap.db"
    timeout: float = 30.0
    check_same_thread: bool = False
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"

    @classmethod
    def default(cls) -> SQLiteConfig:
        return cls()

    @classmethod
    def in_memory(cls) -> SQLiteConfig:
        return cls(db_path=":memory:")


class SQLiteStore:
    """
    Thread-safe SQLite storage with automatic schema creation.

    Features:
    - Automatic table creation
    - JSON column support
    - Indexed queries
    - Connection pooling via thread-local connections
    - ACID compliance
    - Streaming queries for large datasets

    Example:
        >>> store = SQLiteStore()
        >>> store.insert("events", {"type": "query", "data": {"key": "value"}})
        >>> events = store.query("events", where={"type": "query"})
        >>>
        >>> # Stream large results in batches
        >>> for batch in store.query_stream("events", batch_size=100):
        ...     process_batch(batch)
    """

    def __init__(self, config: SQLiteConfig | None = None) -> None:
        self.config = config or SQLiteConfig.default()
        self._logger = logger
        self._local = threading.local()
        self._lock = threading.Lock()

        if self.config.db_path != ":memory:":
            Path(self.config.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self.config.db_path,
                timeout=self.config.timeout,
                check_same_thread=self.config.check_same_thread,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
            self._local.conn.execute(f"PRAGMA synchronous={self.config.synchronous}")

        yield self._local.conn

    def _init_db(self) -> None:
        """Initialize database with metadata table."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS _gaap_metadata (
                    table_name TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    schema_version INTEGER DEFAULT 1
                )
            """)
            conn.commit()

    def _ensure_table(self, table: str) -> None:
        """Create table if not exists with standard schema."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        data JSON NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT
                    )
                """)
                conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table}_created 
                    ON {table}(created_at)
                """)

                conn.execute(
                    """
                    INSERT OR IGNORE INTO _gaap_metadata (table_name, created_at)
                    VALUES (?, ?)
                """,
                    (table, datetime.now().isoformat()),
                )

                conn.commit()

    def _generate_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def insert(
        self,
        table: str,
        data: dict[str, Any],
        item_id: str | None = None,
    ) -> str:
        """
        Insert a record into a table.

        Args:
            table: Table name
            data: Record data
            item_id: Optional ID (auto-generated if not provided)

        Returns:
            Record ID
        """
        self._ensure_table(table)

        item_id = item_id or self._generate_id()
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute(
                f"INSERT INTO {table} (id, data, created_at) VALUES (?, ?, ?)",
                (item_id, json.dumps(data, ensure_ascii=False, default=str), now),
            )
            conn.commit()

        return item_id

    def get(self, table: str, item_id: str) -> dict[str, Any] | None:
        """Get record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT id, data, created_at, updated_at FROM {table} WHERE id = ?", (item_id,)
            )
            row = cursor.fetchone()

            if row:
                return {
                    "id": row["id"],
                    "data": json.loads(row["data"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

        return None

    def query(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str = "created_at DESC",
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """
        Query records from a table.

        Args:
            table: Table name
            where: JSON path conditions (e.g., {"type": "query"})
            order_by: ORDER BY clause
            limit: Maximum results
            offset: Result offset

        Returns:
            List of matching records
        """
        results = []

        with self._get_connection() as conn:
            sql = f"SELECT id, data, created_at, updated_at FROM {table}"
            params: list[Any] = []

            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(f"json_extract(data, '$.{key}') = ?")
                    params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                sql += " WHERE " + " AND ".join(conditions)

            sql += f" ORDER BY {order_by} LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor = conn.execute(sql, params)

            for row in cursor:
                results.append(
                    {
                        "id": row["id"],
                        "data": json.loads(row["data"]),
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                    }
                )

        return results

    def query_stream(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str = "created_at DESC",
        batch_size: int = 100,
    ) -> Iterator[list[dict[str, Any]]]:
        """
        Stream query results in batches for large datasets.

        This generator yields batches of records, allowing processing of
        large datasets without loading everything into memory.

        Args:
            table: Table name
            where: JSON path conditions
            order_by: ORDER BY clause
            batch_size: Number of records per batch

        Yields:
            Batches of records (list of dicts)

        Example:
            >>> for batch in store.query_stream("events", batch_size=100):
            ...     for record in batch:
            ...         process(record)
        """
        offset = 0

        while True:
            batch = self.query(
                table=table,
                where=where,
                order_by=order_by,
                limit=batch_size,
                offset=offset,
            )

            if not batch:
                break

            yield batch

            if len(batch) < batch_size:
                # Last batch
                break

            offset += batch_size

    def query_generator(
        self,
        table: str,
        where: dict[str, Any] | None = None,
        order_by: str = "created_at DESC",
    ) -> Iterator[dict[str, Any]]:
        """
        Stream individual records as a generator.

        This is useful when you want to process records one at a time
        without the overhead of batching.

        Args:
            table: Table name
            where: JSON path conditions
            order_by: ORDER BY clause

        Yields:
            Individual records

        Example:
            >>> for record in store.query_generator("events"):
            ...     process(record)
        """
        offset = 0
        batch_size = 100

        while True:
            batch = self.query(
                table=table,
                where=where,
                order_by=order_by,
                limit=batch_size,
                offset=offset,
            )

            if not batch:
                break

            for record in batch:
                yield record

            if len(batch) < batch_size:
                break

            offset += batch_size

    def update(
        self,
        table: str,
        item_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update record by ID."""
        record = self.get(table, item_id)
        if not record:
            return False

        data = record["data"]
        data.update(updates)

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE {table} SET data = ?, updated_at = ? WHERE id = ?",
                (
                    json.dumps(data, ensure_ascii=False, default=str),
                    datetime.now().isoformat(),
                    item_id,
                ),
            )
            conn.commit()

        return True

    def delete(self, table: str, item_id: str) -> bool:
        """Delete record by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"DELETE FROM {table} WHERE id = ?", (item_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self, table: str, where: dict[str, Any] | None = None) -> int:
        """Count records in table."""
        with self._get_connection() as conn:
            sql = f"SELECT COUNT(*) FROM {table}"
            params: list[Any] = []

            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(f"json_extract(data, '$.{key}') = ?")
                    params.append(json.dumps(value) if isinstance(value, (dict, list)) else value)
                sql += " WHERE " + " AND ".join(conditions)

            cursor = conn.execute(sql, params)
            return cursor.fetchone()[0]  # type: ignore[no-any-return]

    def append(self, table: str, data: dict[str, Any]) -> str:
        """Alias for insert()."""
        return self.insert(table, data)

    def clear(self, table: str) -> int:
        """Clear all records from table."""
        with self._get_connection() as conn:
            cursor = conn.execute(f"DELETE FROM {table}")
            conn.commit()
            return cursor.rowcount

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT table_name FROM _gaap_metadata")
            tables = [row[0] for row in cursor]

            stats = {
                "db_path": self.config.db_path,
                "tables": {},
            }

            for table in tables:
                try:
                    count = self.count(table)
                    stats["tables"][table] = {"count": count}  # type: ignore[index]
                except Exception:
                    pass

            return stats

    def close(self) -> None:
        """Close connection."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None

    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()

    def batch_insert(
        self,
        table: str,
        records: list[dict[str, Any]],
    ) -> list[str]:
        """
        Insert multiple records in a single transaction.

        More efficient than individual inserts for large datasets.

        Args:
            table: Table name
            records: List of record data dictionaries

        Returns:
            List of inserted record IDs
        """
        self._ensure_table(table)

        ids = []
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            for data in records:
                item_id = self._generate_id()
                conn.execute(
                    f"INSERT INTO {table} (id, data, created_at) VALUES (?, ?, ?)",
                    (item_id, json.dumps(data, ensure_ascii=False, default=str), now),
                )
                ids.append(item_id)

            conn.commit()

        return ids


_store_instance: SQLiteStore | None = None
_store_lock = threading.Lock()


def get_sqlite_store(db_path: str = ".gaap/gaap.db") -> SQLiteStore:
    """Get singleton SQLiteStore instance."""
    global _store_instance

    with _store_lock:
        if _store_instance is None:
            _store_instance = SQLiteStore(config=SQLiteConfig(db_path=db_path))
        return _store_instance
