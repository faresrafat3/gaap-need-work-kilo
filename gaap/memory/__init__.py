# Memory Package
import logging
from typing import Any

logger = logging.getLogger("gaap.memory")

# --- Evolution 2026 Additions ---
VECTOR_MEMORY_AVAILABLE = False
try:
    from gaap.memory.vector_store import VectorStore, get_vector_store

    VECTOR_MEMORY_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Vector memory dependencies not found: {e}. Running in degraded mode.")


class LessonStore:
    """Wrapper to store 'Lessons' specifically using VectorStore if available."""

    def __init__(self) -> None:
        if VECTOR_MEMORY_AVAILABLE:
            self.store = get_vector_store()
        else:
            self.store = None

    def add_lesson(self, lesson: str, category: str, task_type: str, success: bool) -> str:
        if not self.store:
            return ""

        meta = {
            "type": "lesson",
            "category": category,
            "task_type": task_type,
            "outcome": "success" if success else "failure",
        }
        return self.store.add(content=lesson, metadata=meta)

    def get(self, entry_id: str) -> Any:
        if not self.store:
            return None
        return self.store.get(entry_id)

    def get_lessons_for_task(self, task_type: str, k: int = 10) -> list[Any]:
        if not self.store:
            return []
        results = self.store.search(task_type, n_results=k)
        return [r for r in results if r.metadata.get("task_type") == task_type]

    def get_security_lessons(self, k: int = 10) -> list[Any]:
        if not self.store:
            return []
        results = self.store.search("security", n_results=k)
        return [r for r in results if r.metadata.get("category") == "security"]

    def retrieve_lessons(self, query: str, k: int = 3) -> list[str]:
        if not self.store:
            return []

        results = self.store.search(query, n_results=k)
        return [r.content for r in results if r.metadata.get("type") == "lesson"]


# --- Legacy Imports (Preserving original structure) ---
from gaap.memory.dream_processor import DreamProcessor  # noqa: E402
from gaap.memory.hierarchical import (  # noqa: E402
    EpisodicMemory,
    EpisodicMemoryStore,
    HierarchicalMemory,
    MemoryTier,
    ProceduralMemoryStore,
    SemanticMemoryStore,
    SemanticRule,
    WorkingMemory,
)
from gaap.memory.memorag import (  # noqa: E402
    KnowledgeGraph,
    MemoRAG,
    RetrievalResult,
    get_memorag,
)

__all__ = [
    "HierarchicalMemory",
    "MemoryTier",
    "WorkingMemory",
    "EpisodicMemory",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "SemanticRule",
    "ProceduralMemoryStore",
    "DreamProcessor",
    "VECTOR_MEMORY_AVAILABLE",
    "LessonStore",
    "MemoRAG",
    "KnowledgeGraph",
    "RetrievalResult",
    "get_memorag",
]
