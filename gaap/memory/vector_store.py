"""
Vector Memory Store - Semantic Search with ChromaDB
====================================================

Provides vector-based memory storage for semantic search and retrieval.
Replaces keyword-based lookups with embedding similarity.

Usage:
    store = VectorMemoryStore()
    await store.add("Learned to use Docker for sandboxing", {"type": "lesson"})
    results = await store.search("container security", n=5)
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.memory.vector")

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore


@dataclass
class MemoryEntry:
    """Entry in vector memory"""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    importance: float = 1.0
    created_at: float = field(default_factory=time.time)
    access_count: int = 0


@dataclass
class SearchResult:
    """Result from vector search"""

    id: str
    content: str
    score: float
    metadata: dict[str, Any]


class VectorMemoryStore:
    """
    Vector-based memory store using ChromaDB.

    Features:
    - Semantic search (find similar concepts, not just keywords)
    - Persistent storage (survives restarts)
    - Automatic embedding (no external model needed)
    - Metadata filtering
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "gaap_memory",
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed. Run: pip install chromadb")

        self.persist_dir = Path(persist_dir or Path.home() / ".gaap" / "vector_memory")
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "GAAP semantic memory"},
        )

        self._logger = logger
        self._entry_count = self._collection.count()

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> str:
        """
        Add content to vector memory.

        Args:
            content: Text content to store
            metadata: Optional metadata (type, tags, source, etc.)
            entry_id: Optional ID (auto-generated if not provided)

        Returns:
            Entry ID
        """
        if not content.strip():
            raise ValueError("Content cannot be empty")

        entry_id = entry_id or self._generate_id(content)
        metadata = metadata or {}

        metadata["created_at"] = time.time()
        metadata["importance"] = metadata.get("importance", 1.0)

        self._collection.upsert(
            ids=[entry_id],
            documents=[content],
            metadatas=[metadata],
        )

        self._entry_count += 1
        self._logger.debug(f"Added memory entry: {entry_id[:8]}...")

        return entry_id

    def search(
        self,
        query: str,
        n: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar content.

        Args:
            query: Search query
            n: Number of results
            where: Optional metadata filter (e.g., {"type": "lesson"})

        Returns:
            List of search results with similarity scores
        """
        if not query.strip():
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._entry_count) if self._entry_count > 0 else 0,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        search_results = []
        for i, doc_id in enumerate(results["ids"][0]):
            content = results["documents"][0][i] if results["documents"] else ""
            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0

            score = 1.0 - min(distance, 1.0)

            search_results.append(
                SearchResult(
                    id=doc_id,
                    content=content,
                    score=score,
                    metadata=dict(metadata) if metadata else {},
                )
            )

        self._logger.debug(f"Found {len(search_results)} results for: {query[:50]}...")
        return search_results

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Get a specific entry by ID"""
        results = self._collection.get(
            ids=[entry_id],
            include=["documents", "metadatas"],
        )

        if not results["ids"]:
            return None

        content = results["documents"][0] if results["documents"] else ""
        raw_metadata = results["metadatas"][0] if results["metadatas"] else {}
        metadata = dict(raw_metadata) if raw_metadata else {}

        return MemoryEntry(
            id=entry_id,
            content=content,
            metadata=metadata,
            importance=float(metadata.get("importance", 1.0)),
            created_at=float(metadata.get("created_at", 0.0)),
            access_count=int(metadata.get("access_count", 0)),
        )

    def delete(self, entry_id: str) -> None:
        """Delete an entry by ID"""
        self._collection.delete(ids=[entry_id])
        self._entry_count = max(0, self._entry_count - 1)

    def update(
        self,
        entry_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing entry"""
        existing = self.get(entry_id)
        if not existing:
            raise KeyError(f"Entry not found: {entry_id}")

        new_content = content or existing.content
        new_metadata = {**existing.metadata, **(metadata or {})}

        self._collection.upsert(
            ids=[entry_id],
            documents=[new_content],
            metadatas=[new_metadata],
        )

    def clear(self) -> None:
        """Clear all entries"""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
        self._entry_count = 0

    def count(self) -> int:
        """Get total number of entries"""
        return self._collection.count()

    def _generate_id(self, content: str) -> str:
        """Generate unique ID from content"""
        hash_input = f"{content}{time.time()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics"""
        return {
            "total_entries": self._entry_count,
            "persist_dir": str(self.persist_dir),
            "collection_name": self._collection.name,
        }


class LessonStore(VectorMemoryStore):
    """
    Specialized store for learned lessons.

    Stores lessons learned from task execution with automatic
    categorization and retrieval.
    """

    def __init__(self, persist_dir: str | None = None):
        super().__init__(persist_dir=persist_dir, collection_name="gaap_lessons")

    def add_lesson(
        self,
        lesson: str,
        category: str = "general",
        task_type: str | None = None,
        success: bool = True,
    ) -> str:
        """
        Add a learned lesson.

        Args:
            lesson: The lesson learned
            category: Category (security, performance, debugging, etc.)
            task_type: Type of task this lesson applies to
            success: Whether this was from a successful operation

        Returns:
            Lesson ID
        """
        metadata = {
            "type": "lesson",
            "category": category,
            "success": success,
        }
        if task_type:
            metadata["task_type"] = task_type

        return self.add(lesson, metadata)

    def get_lessons_for_task(self, task_type: str, n: int = 5) -> list[SearchResult]:
        """Get relevant lessons for a task type"""
        return self.search(
            query=task_type,
            n=n,
            where={"type": "lesson"},
        )

    def get_security_lessons(self, n: int = 10) -> list[SearchResult]:
        """Get all security-related lessons"""
        return self.search(
            query="security vulnerability exploit attack",
            n=n,
            where={"type": "lesson", "category": "security"},
        )
