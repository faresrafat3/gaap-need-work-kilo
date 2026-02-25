"""
Vector Backend Abstractions and Implementations

Provides a unified interface for different vector database backends:
- LanceDB: Primary backend (fast, embedded)
- ChromaDB: Fallback backend
- InMemoryBackend: Simple testing backend

Usage:
    from gaap.memory.vector_backends import get_backend

    backend = get_backend("lancedb")
    backend.connect(".gaap/vectors")
    backend.create_table("documents", schema)
    backend.insert("documents", vectors, metadata)
    results = backend.search("documents", query_vector, k=5)
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.vector_backends")


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding functions."""

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class VectorRecord:
    """A single vector record."""

    id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str = ""
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "metadata": self.metadata,
            "text": self.text,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SearchResult:
    """Result from vector search."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
        }


class VectorBackend(ABC):
    """
    Abstract base class for vector database backends.

    All backends must implement:
    - connect: Establish connection to database
    - create_table: Create a collection/table
    - insert: Insert vectors with metadata
    - search: Search for similar vectors
    - delete: Delete vectors by ID
    """

    @abstractmethod
    def connect(self, path: str) -> bool:
        """
        Connect to the vector database.

        Args:
            path: Path to database storage

        Returns:
            True if connection successful
        """
        ...

    @abstractmethod
    def create_table(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        dimension: int = 384,
    ) -> bool:
        """
        Create a table/collection.

        Args:
            name: Table name
            schema: Optional schema definition
            dimension: Vector dimension

        Returns:
            True if creation successful
        """
        ...

    @abstractmethod
    def insert(
        self,
        table: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[str]:
        """
        Insert vectors into table.

        Args:
            table: Table name
            vectors: List of vector embeddings
            metadata: Optional metadata for each vector
            ids: Optional IDs for each vector
            texts: Optional text for each vector

        Returns:
            List of inserted IDs
        """
        ...

    @abstractmethod
    def search(
        self,
        table: str,
        query_vector: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar vectors.

        Args:
            table: Table name
            query_vector: Query embedding
            k: Number of results
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        ...

    @abstractmethod
    def delete(self, table: str, ids: list[str]) -> bool:
        """
        Delete vectors by ID.

        Args:
            table: Table name
            ids: IDs to delete

        Returns:
            True if deletion successful
        """
        ...

    @abstractmethod
    def count(self, table: str) -> int:
        """
        Count vectors in table.

        Args:
            table: Table name

        Returns:
            Number of vectors
        """
        ...

    @abstractmethod
    def drop_table(self, table: str) -> bool:
        """
        Drop a table.

        Args:
            table: Table name

        Returns:
            True if successful
        """
        ...


class InMemoryBackend(VectorBackend):
    """
    Simple in-memory vector backend for testing.

    Stores vectors in memory with basic cosine similarity search.
    Suitable for development and testing, not production.

    Attributes:
        tables: Dictionary of table name to vector records
        dimension: Default vector dimension
    """

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.tables: dict[str, list[VectorRecord]] = {}
        self._connected = False
        self._logger = logger

    def connect(self, path: str) -> bool:
        self._connected = True
        self._logger.debug("InMemoryBackend connected (no persistence)")
        return True

    def create_table(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        dimension: int = 384,
    ) -> bool:
        if name not in self.tables:
            self.tables[name] = []
            self.dimension = dimension
        return True

    def insert(
        self,
        table: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[str]:
        if table not in self.tables:
            self.create_table(table)

        inserted_ids = []
        for i, vector in enumerate(vectors):
            record_id = ids[i] if ids and i < len(ids) else self._generate_id()
            record_metadata = metadata[i] if metadata and i < len(metadata) else {}
            record_text = texts[i] if texts and i < len(texts) else ""

            record = VectorRecord(
                id=record_id,
                vector=vector,
                metadata=record_metadata,
                text=record_text,
            )

            self.tables[table].append(record)
            inserted_ids.append(record_id)

        return inserted_ids

    def search(
        self,
        table: str,
        query_vector: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if table not in self.tables:
            return []

        candidates = self.tables[table]

        if filter_metadata:
            candidates = [
                r
                for r in candidates
                if all(r.metadata.get(k) == v for k, v in filter_metadata.items())
            ]

        scored = []
        for record in candidates:
            score = self._cosine_similarity(query_vector, record.vector)
            scored.append((record, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for record, score in scored[:k]:
            results.append(
                SearchResult(
                    id=record.id,
                    score=score,
                    text=record.text,
                    metadata=record.metadata,
                )
            )

        return results

    def delete(self, table: str, ids: list[str]) -> bool:
        if table not in self.tables:
            return False

        id_set = set(ids)
        self.tables[table] = [r for r in self.tables[table] if r.id not in id_set]
        return True

    def count(self, table: str) -> int:
        return len(self.tables.get(table, []))

    def drop_table(self, table: str) -> bool:
        if table in self.tables:
            del self.tables[table]
        return True

    def _generate_id(self) -> str:
        import uuid

        return str(uuid.uuid4())

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


LANCEDB_AVAILABLE = False
_lancedb = None
try:
    import lancedb as _lancedb_module

    _lancedb = _lancedb_module
    LANCEDB_AVAILABLE = True
except ImportError:
    pass


class LanceDBBackend(VectorBackend):
    """
    LanceDB vector backend implementation.

    LanceDB is a serverless vector database with:
    - Embedded storage (no server required)
    - Fast vector search with HNSW
    - Automatic embedding generation
    - SQLite-like simplicity

    Attributes:
        db: LanceDB connection
        path: Database path
        dimension: Default vector dimension
    """

    def __init__(self, dimension: int = 384) -> None:
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB is not installed. Install with: pip install lancedb")

        self.db = None
        self.path = ""
        self.dimension = dimension
        self._tables: dict[str, Any] = {}
        self._logger = logger

    def connect(self, path: str) -> bool:
        try:
            import os

            os.makedirs(path, exist_ok=True)
            import lancedb

            self.db = lancedb.connect(path)
            self.path = path
            self._logger.debug(f"LanceDB connected at {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect LanceDB: {e}")
            return False

    def create_table(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        dimension: int = 384,
    ) -> bool:
        if not self.db:
            self._logger.error("Not connected to database")
            return False

        try:
            if name in self.db.table_names():
                self._tables[name] = self.db.open_table(name)
                return True

            import pyarrow as pa

            if schema is None:
                schema = pa.schema(
                    [
                        pa.field("id", pa.string()),
                        pa.field("vector", pa.list_(pa.float32(), dimension)),
                        pa.field("text", pa.string()),
                        pa.field("metadata", pa.string()),
                    ]
                )

            table = self.db.create_table(name, schema=schema)
            self._tables[name] = table
            self._logger.debug(f"Created LanceDB table: {name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to create table {name}: {e}")
            return False

    def insert(
        self,
        table: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[str]:
        if table not in self._tables:
            self.create_table(table, dimension=len(vectors[0]) if vectors else self.dimension)

        tbl = self._tables.get(table)
        if not tbl:
            return []

        import json
        import uuid

        records = []
        inserted_ids = []

        for i, vector in enumerate(vectors):
            record_id = ids[i] if ids and i < len(ids) else str(uuid.uuid4())
            record_metadata = metadata[i] if metadata and i < len(metadata) else {}
            record_text = texts[i] if texts and i < len(texts) else ""

            records.append(
                {
                    "id": record_id,
                    "vector": vector,
                    "text": record_text,
                    "metadata": json.dumps(record_metadata),
                }
            )
            inserted_ids.append(record_id)

        try:
            tbl.add(records)
            return inserted_ids
        except Exception as e:
            self._logger.error(f"Failed to insert into {table}: {e}")
            return []

    def search(
        self,
        table: str,
        query_vector: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        tbl = self._tables.get(table)
        if not tbl:
            return []

        try:
            query = tbl.search(query_vector).limit(k)

            results = query.to_list()

            search_results = []
            import json

            for result in results:
                metadata_str = result.get("metadata", "{}")
                try:
                    metadata = (
                        json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                    )
                except Exception:
                    metadata = {}

                if filter_metadata:
                    if not all(metadata.get(k) == v for k, v in filter_metadata.items()):
                        continue

                search_results.append(
                    SearchResult(
                        id=result.get("id", ""),
                        score=result.get("_distance", 0.0),
                        text=result.get("text", ""),
                        metadata=metadata,
                    )
                )

            return search_results[:k]
        except Exception as e:
            self._logger.error(f"Search failed in {table}: {e}")
            return []

    def delete(self, table: str, ids: list[str]) -> bool:
        tbl = self._tables.get(table)
        if not tbl:
            return False

        try:
            tbl.delete(f"id IN ({','.join(repr(i) for i in ids)})")
            return True
        except Exception as e:
            self._logger.error(f"Delete failed in {table}: {e}")
            return False

    def count(self, table: str) -> int:
        tbl = self._tables.get(table)
        if not tbl:
            return 0

        try:
            return len(tbl.to_pandas())
        except Exception:
            return 0

    def drop_table(self, table: str) -> bool:
        if not self.db:
            return False

        try:
            self.db.drop_table(table)
            if table in self._tables:
                del self._tables[table]
            return True
        except Exception as e:
            self._logger.error(f"Failed to drop table {table}: {e}")
            return False


CHROMADB_AVAILABLE = False
try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    pass


class ChromaDBBackend(VectorBackend):
    """
    ChromaDB vector backend implementation.

    ChromaDB is an AI-native open-source embedding database:
    - Simple API
    - Built-in embedding functions
    - Persistent storage
    - Metadata filtering

    Attributes:
        client: ChromaDB client
        path: Database path
        collections: Dictionary of collection name to collection
    """

    def __init__(self, dimension: int = 384) -> None:
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")

        self.client = None
        self.path = ""
        self.dimension = dimension
        self._collections: dict[str, Any] = {}
        self._logger = logger

    def connect(self, path: str) -> bool:
        try:
            import os

            os.makedirs(path, exist_ok=True)
            import chromadb  # type: ignore
            from chromadb.config import Settings  # type: ignore

            self.client = chromadb.PersistentClient(path=path, settings=Settings(allow_reset=True))
            self.path = path
            self._logger.debug(f"ChromaDB connected at {path}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to connect ChromaDB: {e}")
            return False

    def create_table(
        self,
        name: str,
        schema: dict[str, Any] | None = None,
        dimension: int = 384,
    ) -> bool:
        if not self.client:
            self._logger.error("Not connected to database")
            return False

        try:
            collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            self._collections[name] = collection
            self._logger.debug(f"Created ChromaDB collection: {name}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to create collection {name}: {e}")
            return False

    def insert(
        self,
        table: str,
        vectors: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> list[str]:
        if table not in self._collections:
            self.create_table(table)

        collection = self._collections.get(table)
        if not collection:
            return []

        import uuid

        inserted_ids = []
        record_ids = ids or [str(uuid.uuid4()) for _ in vectors]
        record_metadata = metadata or [{} for _ in vectors]
        record_texts = texts or ["" for _ in vectors]

        try:
            collection.add(
                ids=record_ids,
                embeddings=vectors,
                metadatas=record_metadata,
                documents=record_texts,
            )
            return record_ids
        except Exception as e:
            self._logger.error(f"Failed to insert into {table}: {e}")
            return []

    def search(
        self,
        table: str,
        query_vector: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        collection = self._collections.get(table)
        if not collection:
            return []

        try:
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filter_metadata,
            )

            search_results = []
            if results["ids"] and results["ids"][0]:
                ids = results["ids"][0]
                distances = results["distances"][0] if results["distances"] else []
                documents = results["documents"][0] if results["documents"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []

                for i, doc_id in enumerate(ids):
                    score = 1.0 - distances[i] if i < len(distances) else 0.0
                    metadata = metadatas[i] if i < len(metadatas) else {}

                    search_results.append(
                        SearchResult(
                            id=doc_id,
                            score=score,
                            text=documents[i] if i < len(documents) else "",
                            metadata=metadata,
                        )
                    )

            return search_results
        except Exception as e:
            self._logger.error(f"Search failed in {table}: {e}")
            return []

    def delete(self, table: str, ids: list[str]) -> bool:
        collection = self._collections.get(table)
        if not collection:
            return False

        try:
            collection.delete(ids=ids)
            return True
        except Exception as e:
            self._logger.error(f"Delete failed in {table}: {e}")
            return False

    def count(self, table: str) -> int:
        collection = self._collections.get(table)
        if not collection:
            return 0

        try:
            return collection.count()
        except Exception:
            return 0

    def drop_table(self, table: str) -> bool:
        if not self.client:
            return False

        try:
            self.client.delete_collection(table)
            if table in self._collections:
                del self._collections[table]
            return True
        except Exception as e:
            self._logger.error(f"Failed to drop collection {table}: {e}")
            return False


def get_backend(
    backend_type: str = "auto",
    dimension: int = 384,
) -> VectorBackend:
    """
    Get a vector backend instance.

    Args:
        backend_type: Backend type ("lancedb", "chromadb", "memory", "auto")
        dimension: Vector dimension

    Returns:
        VectorBackend instance

    Note:
        With "auto", tries LanceDB first, then ChromaDB, then InMemory.
    """
    if backend_type == "auto":
        if LANCEDB_AVAILABLE:
            return LanceDBBackend(dimension=dimension)
        elif CHROMADB_AVAILABLE:
            return ChromaDBBackend(dimension=dimension)
        else:
            logger.warning("No external backend available, using InMemoryBackend")
            return InMemoryBackend(dimension=dimension)

    if backend_type == "lancedb":
        if not LANCEDB_AVAILABLE:
            raise ImportError("LanceDB not available")
        return LanceDBBackend(dimension=dimension)

    if backend_type == "chromadb":
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available")
        return ChromaDBBackend(dimension=dimension)

    if backend_type == "memory":
        return InMemoryBackend(dimension=dimension)

    raise ValueError(f"Unknown backend type: {backend_type}")


def get_available_backends() -> list[str]:
    """Get list of available backend types."""
    backends = ["memory"]
    if LANCEDB_AVAILABLE:
        backends.append("lancedb")
    if CHROMADB_AVAILABLE:
        backends.append("chromadb")
    return backends
