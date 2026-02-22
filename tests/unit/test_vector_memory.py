"""
Tests for Vector Memory Store

Tests the vector store implementation for semantic memory retrieval.
"""

import pytest

from gaap.memory import LessonStore
from gaap.memory.vector_store import VectorEntry, VectorStore


@pytest.fixture
def vector_store():
    """Create a fresh vector store for each test"""
    store = VectorStore(collection_name="test_memory")
    store.reset()
    return store


@pytest.fixture
def lesson_store():
    """Create a fresh lesson store for each test"""
    return LessonStore()


class TestVectorStore:
    """Tests for VectorStore"""

    def test_add_and_search(self, vector_store) -> None:
        """Test adding and searching content"""
        vector_store.add("Docker is used for containerization", {"topic": "docker"})
        vector_store.add("Kubernetes orchestrates containers", {"topic": "k8s"})

        results = vector_store.search("containers")

        assert len(results) >= 1
        assert hasattr(results[0], "id")
        assert hasattr(results[0], "content")

    def test_add_with_custom_id(self, vector_store) -> None:
        """Test adding with custom ID"""
        entry_id = vector_store.add(
            "Custom ID test",
            {"type": "test"},
            entry_id="custom_id_123",
        )

        assert entry_id == "custom_id_123"

    def test_delete(self, vector_store) -> None:
        """Test deleting entry"""
        entry_id = vector_store.add("To be deleted", {"temp": True})

        assert vector_store.count() >= 1

        result = vector_store.delete(entry_id)

        assert result is True

    def test_count(self, vector_store) -> None:
        """Test counting entries"""
        initial_count = vector_store.count()

        vector_store.add("Entry 1")
        vector_store.add("Entry 2")

        assert vector_store.count() == initial_count + 2

    def test_reset(self, vector_store) -> None:
        """Test resetting all entries"""
        vector_store.add("Entry 1")
        vector_store.add("Entry 2")
        vector_store.add("Entry 3")

        vector_store.reset()

        assert vector_store.count() == 0

    def test_search_with_metadata_filter(self, vector_store) -> None:
        """Test searching with metadata filter"""
        vector_store.add("Python lesson", {"type": "lesson", "category": "python"})
        vector_store.add("JavaScript lesson", {"type": "lesson", "category": "javascript"})
        vector_store.add("Python note", {"type": "note", "category": "python"})

        results = vector_store.search("Python", filter_meta={"type": "lesson"})

        assert all(r.metadata.get("type") == "lesson" for r in results)

    def test_available_property(self, vector_store) -> None:
        """Test available property"""
        assert isinstance(vector_store.available, bool)


class TestLessonStore:
    """Tests for LessonStore"""

    def test_add_lesson(self, lesson_store) -> None:
        """Test adding a lesson"""
        lesson_id = lesson_store.add_lesson(
            lesson="Always validate user input",
            category="security",
            task_type="api",
            success=True,
        )

        assert lesson_id is not None

    def test_get_lessons_for_task(self, lesson_store) -> None:
        """Test getting lessons for a task type"""
        lesson_store.add_lesson(
            "Use prepared statements", category="security", task_type="database", success=True
        )
        lesson_store.add_lesson(
            "Cache query results", category="performance", task_type="database", success=True
        )
        lesson_store.add_lesson(
            "Validate API keys", category="security", task_type="api", success=True
        )

        results = lesson_store.get_lessons_for_task("database")

        assert isinstance(results, list)

    def test_get_security_lessons(self, lesson_store) -> None:
        """Test getting security lessons"""
        lesson_store.add_lesson(
            "Sanitize SQL inputs", category="security", task_type="database", success=True
        )
        lesson_store.add_lesson("Use HTTPS", category="security", task_type="web", success=True)
        lesson_store.add_lesson(
            "Cache results", category="performance", task_type="api", success=True
        )

        results = lesson_store.get_security_lessons()

        assert isinstance(results, list)

    def test_retrieve_lessons(self, lesson_store) -> None:
        """Test retrieving lessons by query"""
        lesson_store.add_lesson(
            "Always validate user input", category="security", task_type="api", success=True
        )

        results = lesson_store.retrieve_lessons("validation")

        assert isinstance(results, list)


class TestVectorEntry:
    """Tests for VectorEntry dataclass"""

    def test_vector_entry_creation(self) -> None:
        """Test creating a vector entry"""
        entry = VectorEntry(
            id="test_id",
            content="Test content",
            metadata={"key": "value"},
        )

        assert entry.id == "test_id"
        assert entry.content == "Test content"
        assert entry.metadata == {"key": "value"}

    def test_vector_entry_with_defaults(self) -> None:
        """Test creating entry with default values"""
        entry = VectorEntry(
            id="test_id",
            content="Test content",
            metadata={},
        )

        assert entry.created_at is not None
