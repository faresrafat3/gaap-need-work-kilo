"""Tests for Vector Memory Store"""

import pytest

from gaap.memory.vector_store import LessonStore, MemoryEntry, SearchResult, VectorMemoryStore


@pytest.fixture
def vector_store():
    """Create a fresh vector store for each test"""
    store = VectorMemoryStore(collection_name="test_memory")
    store.clear()
    return store


@pytest.fixture
def lesson_store():
    """Create a fresh lesson store for each test"""
    store = LessonStore()
    store.clear()
    return store


class TestVectorMemoryStore:
    """Tests for VectorMemoryStore"""

    def test_add_and_search(self, vector_store):
        """Test adding and searching content"""
        vector_store.add("Docker is used for containerization", {"topic": "docker"})
        vector_store.add("Kubernetes orchestrates containers", {"topic": "k8s"})

        results = vector_store.search("containers")

        assert len(results) >= 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id
        assert results[0].content

    def test_add_with_custom_id(self, vector_store):
        """Test adding with custom ID"""
        entry_id = vector_store.add(
            "Custom ID test",
            {"type": "test"},
            entry_id="custom_id_123",
        )

        assert entry_id == "custom_id_123"

        entry = vector_store.get("custom_id_123")
        assert entry is not None
        assert entry.content == "Custom ID test"

    def test_get_nonexistent(self, vector_store):
        """Test getting nonexistent entry"""
        entry = vector_store.get("nonexistent_id")
        assert entry is None

    def test_delete(self, vector_store):
        """Test deleting entry"""
        entry_id = vector_store.add("To be deleted", {"temp": True})

        assert vector_store.get(entry_id) is not None

        vector_store.delete(entry_id)

        assert vector_store.get(entry_id) is None

    def test_update(self, vector_store):
        """Test updating entry"""
        entry_id = vector_store.add("Original content")

        vector_store.update(entry_id, content="Updated content")

        entry = vector_store.get(entry_id)
        assert entry is not None
        assert entry.content == "Updated content"

    def test_count(self, vector_store):
        """Test counting entries"""
        assert vector_store.count() == 0

        vector_store.add("Entry 1")
        vector_store.add("Entry 2")

        assert vector_store.count() == 2

    def test_clear(self, vector_store):
        """Test clearing all entries"""
        vector_store.add("Entry 1")
        vector_store.add("Entry 2")
        vector_store.add("Entry 3")

        vector_store.clear()

        assert vector_store.count() == 0

    def test_search_with_metadata_filter(self, vector_store):
        """Test searching with metadata filter"""
        vector_store.add("Python lesson", {"type": "lesson", "category": "python"})
        vector_store.add("JavaScript lesson", {"type": "lesson", "category": "javascript"})
        vector_store.add("Python note", {"type": "note", "category": "python"})

        results = vector_store.search("Python", where={"type": "lesson"})

        assert all(r.metadata.get("type") == "lesson" for r in results)

    def test_get_stats(self, vector_store):
        """Test getting statistics"""
        vector_store.add("Test")

        stats = vector_store.get_stats()

        assert "total_entries" in stats
        assert stats["total_entries"] >= 1
        assert "persist_dir" in stats


class TestLessonStore:
    """Tests for LessonStore"""

    def test_add_lesson(self, lesson_store):
        """Test adding a lesson"""
        lesson_id = lesson_store.add_lesson(
            lesson="Always validate user input",
            category="security",
            task_type="api",
            success=True,
        )

        assert lesson_id

        entry = lesson_store.get(lesson_id)
        assert entry is not None
        assert entry.metadata.get("type") == "lesson"
        assert entry.metadata.get("category") == "security"

    def test_get_lessons_for_task(self, lesson_store):
        """Test getting lessons for a task type"""
        lesson_store.add_lesson(
            "Use prepared statements", category="security", task_type="database"
        )
        lesson_store.add_lesson("Cache query results", category="performance", task_type="database")
        lesson_store.add_lesson("Validate API keys", category="security", task_type="api")

        results = lesson_store.get_lessons_for_task("database")

        assert len(results) >= 2

    def test_get_security_lessons(self, lesson_store):
        """Test getting security lessons"""
        lesson_store.add_lesson("Sanitize SQL inputs", category="security")
        lesson_store.add_lesson("Use HTTPS", category="security")
        lesson_store.add_lesson("Cache results", category="performance")

        results = lesson_store.get_security_lessons()

        assert all(r.metadata.get("category") == "security" for r in results)


class TestMemoryEntry:
    """Tests for MemoryEntry dataclass"""

    def test_memory_entry_creation(self):
        """Test creating a memory entry"""
        entry = MemoryEntry(
            id="test_id",
            content="Test content",
            metadata={"key": "value"},
            importance=0.8,
        )

        assert entry.id == "test_id"
        assert entry.content == "Test content"
        assert entry.importance == 0.8
        assert entry.access_count == 0


class TestSearchResult:
    """Tests for SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating a search result"""
        result = SearchResult(
            id="result_id",
            content="Result content",
            score=0.95,
            metadata={"source": "test"},
        )

        assert result.id == "result_id"
        assert result.score == 0.95
