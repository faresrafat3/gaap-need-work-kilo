"""
Tests for GAAP Memory Hierarchical Module
"""

import pytest

from gaap.memory.hierarchical import (
    EpisodicMemory,
    EpisodicMemoryStore,
    MemoryEntry,
    MemoryPriority,
    MemoryTier,
    SemanticRule,
    WorkingMemory,
)


class TestMemoryTier:
    def test_values(self):
        assert MemoryTier.WORKING.value == 1
        assert MemoryTier.EPISODIC.value == 2
        assert MemoryTier.SEMANTIC.value == 3
        assert MemoryTier.PROCEDURAL.value == 4


class TestMemoryPriority:
    def test_values(self):
        assert MemoryPriority.CRITICAL.value == 1
        assert MemoryPriority.HIGH.value == 2
        assert MemoryPriority.NORMAL.value == 3
        assert MemoryPriority.LOW.value == 4
        assert MemoryPriority.BACKGROUND.value == 5


class TestMemoryEntry:
    def test_defaults(self):
        entry = MemoryEntry(
            id="test-1",
            tier=MemoryTier.WORKING,
            content="test content",
        )
        assert entry.id == "test-1"
        assert entry.tier == MemoryTier.WORKING
        assert entry.content == "test content"
        assert entry.priority == MemoryPriority.NORMAL
        assert entry.access_count == 0

    def test_access(self):
        entry = MemoryEntry(
            id="test-1",
            tier=MemoryTier.WORKING,
            content="test",
        )
        entry.access()
        assert entry.access_count == 1
        entry.access()
        assert entry.access_count == 2

    def test_get_strength_fresh(self):
        entry = MemoryEntry(
            id="test-1",
            tier=MemoryTier.WORKING,
            content="test",
            importance=0.8,
        )
        strength = entry.get_strength()
        assert strength > 0
        assert strength <= entry.importance + 0.5

    def test_get_strength_with_access(self):
        entry = MemoryEntry(
            id="test-1",
            tier=MemoryTier.WORKING,
            content="test",
            importance=0.5,
        )
        entry.access()
        entry.access()
        entry.access()
        strength = entry.get_strength()
        assert strength > 0.5


class TestEpisodicMemory:
    def test_defaults(self):
        episode = EpisodicMemory(
            task_id="task-1",
            action="test_action",
            result="success",
            success=True,
        )
        assert episode.task_id == "task-1"
        assert episode.action == "test_action"
        assert episode.success is True
        assert episode.category == "general"
        assert episode.duration_ms == 0.0

    def test_to_dict(self):
        episode = EpisodicMemory(
            task_id="task-1",
            action="code_generation",
            result="Success",
            success=True,
            duration_ms=1000,
            tokens_used=500,
            lessons=["Use tests"],
        )
        d = episode.to_dict()
        assert d["task_id"] == "task-1"
        assert d["success"] is True
        assert d["duration_ms"] == 1000
        assert "Use tests" in d["lessons"]


class TestSemanticRule:
    def test_defaults(self):
        rule = SemanticRule(
            id="rule-1",
            condition="test condition",
            action="test action",
            confidence=0.9,
            support_count=5,
        )
        assert rule.id == "rule-1"
        assert rule.confidence == 0.9

    def test_to_dict(self):
        rule = SemanticRule(
            id="rule-1",
            condition="test condition",
            action="test action",
            confidence=0.85,
            support_count=10,
        )
        d = rule.to_dict()
        assert d["id"] == "rule-1"
        assert d["confidence"] == 0.85
        assert d["support"] == 10


class TestWorkingMemory:
    def test_defaults(self):
        memory = WorkingMemory()
        assert memory.max_size == 100
        assert memory.get_size() == 0

    def test_custom_max_size(self):
        memory = WorkingMemory(max_size=10)
        assert memory.max_size == 10

    def test_store_and_retrieve(self):
        memory = WorkingMemory()
        memory.store("key1", {"data": "value1"})
        value = memory.retrieve("key1")
        assert value == {"data": "value1"}

    def test_retrieve_nonexistent(self):
        memory = WorkingMemory()
        value = memory.retrieve("nonexistent")
        assert value is None

    def test_store_overwrites(self):
        memory = WorkingMemory()
        memory.store("key1", "value1")
        memory.store("key1", "value2")
        value = memory.retrieve("key1")
        assert value == "value2"

    def test_eviction(self):
        memory = WorkingMemory(max_size=2)
        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.store("key3", "value3")
        assert memory.get_size() == 2

    def test_clear(self):
        memory = WorkingMemory()
        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.clear()
        assert memory.get_size() == 0


class TestEpisodicMemoryStore:
    def test_defaults(self):
        store = EpisodicMemoryStore()
        assert len(store._episodes) == 0

    def test_record(self):
        store = EpisodicMemoryStore()
        episode = EpisodicMemory(
            task_id="task-1",
            action="test",
            result="success",
            success=True,
        )
        store.record(episode)
        assert len(store._episodes) == 1

    def test_get_episodes(self):
        store = EpisodicMemoryStore()
        store.record(EpisodicMemory(task_id="task-1", action="a", result="r", success=True))
        store.record(EpisodicMemory(task_id="task-1", action="b", result="r", success=False))

        episodes = store.get_episodes(task_id="task-1")
        assert len(episodes) == 2

    def test_get_episodes_success_only(self):
        store = EpisodicMemoryStore()
        store.record(EpisodicMemory(task_id="task-1", action="a", result="r", success=True))
        store.record(EpisodicMemory(task_id="task-1", action="b", result="r", success=False))

        episodes = store.get_episodes(task_id="task-1", success_only=True)
        assert len(episodes) == 1
        assert episodes[0].success is True

    def test_get_success_rate(self):
        store = EpisodicMemoryStore()
        store.record(EpisodicMemory(task_id="task-1", action="a", result="r", success=True))
        store.record(EpisodicMemory(task_id="task-2", action="b", result="r", success=True))
        store.record(EpisodicMemory(task_id="task-3", action="c", result="r", success=False))

        rate = store.get_success_rate()
        assert rate == 2 / 3

    def test_get_success_rate_empty(self):
        store = EpisodicMemoryStore()
        rate = store.get_success_rate()
        assert rate == 0.0

    def test_get_recent_lessons(self):
        store = EpisodicMemoryStore()
        store.record(
            EpisodicMemory(
                task_id="task-1", action="a", result="r", success=True, lessons=["lesson1"]
            )
        )
        store.record(
            EpisodicMemory(
                task_id="task-2", action="b", result="r", success=True, lessons=["lesson2"]
            )
        )

        lessons = store.get_recent_lessons()
        assert len(lessons) == 2
        assert "lesson1" in lessons

    def test_get_recent_lessons_limit(self):
        store = EpisodicMemoryStore()
        for i in range(5):
            store.record(
                EpisodicMemory(
                    task_id=f"task-{i}",
                    action="a",
                    result="r",
                    success=True,
                    lessons=[f"lesson{i}"],
                )
            )

        lessons = store.get_recent_lessons(limit=3)
        assert len(lessons) == 3

    def test_save_no_storage_path(self):
        store = EpisodicMemoryStore()
        result = store.save()
        assert result is False

    def test_load_no_file(self, tmp_path):
        store = EpisodicMemoryStore(storage_path=str(tmp_path / "nonexistent"))
        result = store.load()
        assert result is False

    def test_save_and_load(self, tmp_path):
        store = EpisodicMemoryStore(storage_path=str(tmp_path))

        store.record(
            EpisodicMemory(
                task_id="task-1",
                action="code_generation",
                result="Success",
                success=True,
                duration_ms=1000,
                lessons=["Use tests"],
            )
        )

        save_result = store.save()
        assert save_result is True

        new_store = EpisodicMemoryStore(storage_path=str(tmp_path))
        load_result = new_store.load()
        assert load_result is True
        assert len(new_store._episodes) == 1
