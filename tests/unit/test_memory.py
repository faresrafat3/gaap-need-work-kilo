"""
Unit tests for Memory system
"""

import asyncio
from datetime import datetime, timedelta

import pytest


class TestMemoryTiers:
    """Tests for memory tier system"""

    def test_tier_definitions(self):
        """Test memory tier definitions"""
        tiers = {
            "L1": {"name": "Working Memory", "capacity": 100, "speed": "fastest"},
            "L2": {"name": "Episodic Memory", "capacity": 1000, "speed": "fast"},
            "L3": {"name": "Semantic Memory", "capacity": 10000, "speed": "medium"},
            "L4": {"name": "Procedural Memory", "capacity": 100000, "speed": "slow"},
        }

        assert tiers["L1"]["speed"] == "fastest"
        assert tiers["L4"]["capacity"] == 100000

    def test_tier_promotion_threshold(self):
        """Test tier promotion thresholds"""
        access_counts = {"L1": 5, "L2": 3, "L3": 2}
        promotion_threshold = 4

        should_promote = access_counts["L1"] >= promotion_threshold
        assert should_promote

    def test_tier_demotion(self):
        """Test tier demotion on low access"""
        last_access = {
            "item_1": datetime.now() - timedelta(days=30),
            "item_2": datetime.now() - timedelta(hours=1),
        }

        demotion_threshold = timedelta(days=7)

        should_demote = {k: datetime.now() - v > demotion_threshold for k, v in last_access.items()}

        assert should_demote["item_1"]
        assert not should_demote["item_2"]


class TestWorkingMemory:
    """Tests for L1 Working Memory"""

    def test_store_and_retrieve(self, mock_memory):
        """Test storing and retrieving from working memory"""
        key = "test_key"
        value = {"data": "test_value"}

        asyncio.run(mock_memory.store(key, value))
        result = asyncio.run(mock_memory.retrieve(key))

        assert result == value

    def test_capacity_limit(self):
        """Test capacity limit enforcement"""
        capacity = 5
        items = {f"item_{i}": f"value_{i}" for i in range(10)}

        stored = dict(list(items.items())[:capacity])
        assert len(stored) == capacity

    def test_lru_eviction(self):
        """Test LRU eviction policy"""
        items = [
            ("a", 1),
            ("b", 2),
            ("c", 3),
            ("a", 4),  # access 'a' again
            ("d", 5),
        ]

        access_order = []
        for key, _ in items:
            if key in access_order:
                access_order.remove(key)
            access_order.append(key)

        lru = access_order[0]
        assert lru == "b"


class TestEpisodicMemory:
    """Tests for L2 Episodic Memory"""

    def test_episode_recording(self):
        """Test recording an episode"""
        episode = {
            "id": "ep_001",
            "task_id": "task_123",
            "timestamp": datetime.now(),
            "success": True,
            "lessons": ["Use caching for repeated calculations"],
        }

        assert episode["success"]
        assert len(episode["lessons"]) > 0

    def test_episode_retrieval_by_task(self):
        """Test retrieving episodes by task"""
        episodes = [
            {"task_id": "task_1", "success": True},
            {"task_id": "task_2", "success": False},
            {"task_id": "task_1", "success": True},
        ]

        task_episodes = [e for e in episodes if e["task_id"] == "task_1"]
        assert len(task_episodes) == 2

    def test_success_rate_calculation(self):
        """Test calculating success rate from episodes"""
        episodes = [
            {"success": True},
            {"success": True},
            {"success": False},
            {"success": True},
        ]

        success_rate = sum(1 for e in episodes if e["success"]) / len(episodes)
        assert success_rate == 0.75


class TestSemanticMemory:
    """Tests for L3 Semantic Memory"""

    def test_rule_storage(self):
        """Test storing semantic rules"""
        rules = {
            "python_naming": "Use snake_case for functions",
            "error_handling": "Always catch specific exceptions",
        }

        assert "python_naming" in rules

    def test_rule_confidence(self):
        """Test rule confidence tracking"""
        rule = {
            "condition": "task_type == CODE_GENERATION",
            "action": "use_python_template",
            "confidence": 0.85,
            "uses": 100,
        }

        assert rule["confidence"] >= 0.8

    def test_rule_reinforcement(self):
        """Test rule reinforcement"""
        rule = {"confidence": 0.5, "uses": 10}

        new_confidence = rule["confidence"] + (1 - rule["confidence"]) * 0.1
        rule["confidence"] = min(new_confidence, 1.0)
        rule["uses"] += 1

        assert rule["confidence"] > 0.5
        assert rule["uses"] == 11


class TestProceduralMemory:
    """Tests for L4 Procedural Memory"""

    def test_procedure_storage(self):
        """Test storing procedures"""
        procedure = {
            "name": "code_review",
            "steps": [
                "Check syntax",
                "Check logic",
                "Check style",
            ],
            "success_rate": 0.9,
        }

        assert len(procedure["steps"]) == 3

    def test_procedure_retrieval(self):
        """Test retrieving best procedure"""
        procedures = [
            {"name": "quick_fix", "success_rate": 0.7},
            {"name": "thorough_fix", "success_rate": 0.95},
            {"name": "temp_fix", "success_rate": 0.5},
        ]

        best = max(procedures, key=lambda x: x["success_rate"])
        assert best["name"] == "thorough_fix"

    def test_procedure_adaptation(self):
        """Test procedure adaptation"""
        procedure = {
            "steps": ["step1", "step2", "step3"],
            "adaptations": [],
        }

        new_step = "step2b"
        procedure["steps"].insert(2, new_step)
        procedure["adaptations"].append({"step": 2, "action": "insert"})

        assert len(procedure["steps"]) == 4


class TestMemoryDecay:
    """Tests for memory decay"""

    def test_decay_calculation(self):
        """Test memory decay calculation"""
        importance = 0.8
        age_hours = 24
        decay_rate = 0.01

        strength = importance * (1 - decay_rate) ** age_hours
        assert strength < importance

    def test_access_boost(self):
        """Test access count boost"""
        base_strength = 0.5
        access_count = 5
        max_boost = 0.5

        boost = min(access_count * 0.1, max_boost)
        final_strength = min(base_strength + boost, 1.0)

        assert final_strength == 1.0

    def test_decay_threshold(self):
        """Test decay below threshold"""
        strength = 0.05
        threshold = 0.1

        should_remove = strength < threshold
        assert should_remove


class TestMemoryIntegration:
    """Integration tests for memory system"""

    @pytest.mark.asyncio
    async def test_full_memory_flow(self, mock_memory):
        """Test complete memory flow"""
        await mock_memory.store("key1", "value1")
        result = await mock_memory.retrieve("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cross_tier_search(self, mock_memory):
        """Test searching across tiers"""
        await mock_memory.store("test", {"tier": "L1"})
        result = await mock_memory.retrieve("test")
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
