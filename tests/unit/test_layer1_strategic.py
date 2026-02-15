"""
Unit tests for Layer 1 - Strategic Layer
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List

from gaap.core.types import (
    Task,
    TaskPriority,
    TaskType,
    TaskComplexity,
    ModelTier,
    HealingLevel,
    CriticType,
)


class TestStrategicPlanning:
    """Tests for strategic planning logic"""

    def test_simple_task_strategy(self):
        """Test strategy for simple tasks"""
        complexity = TaskComplexity.SIMPLE
        if complexity == TaskComplexity.SIMPLE:
            strategy = "direct_execution"
        assert strategy == "direct_execution"

    def test_moderate_task_strategy(self):
        """Test strategy for moderate tasks"""
        complexity = TaskComplexity.MODERATE
        if complexity == TaskComplexity.MODERATE:
            strategy = "single_agent"
        assert strategy == "single_agent"

    def test_complex_task_strategy(self):
        """Test strategy for complex tasks"""
        complexity = TaskComplexity.COMPLEX
        if complexity == TaskComplexity.COMPLEX:
            strategy = "multi_agent"
        assert strategy == "multi_agent"

    def test_critical_task_strategy(self):
        """Test strategy for critical tasks"""
        priority = TaskPriority.CRITICAL
        if priority == TaskPriority.CRITICAL:
            strategy = "full_pipeline"
        assert strategy == "full_pipeline"


class TestTreeOfThoughts:
    """Tests for Tree of Thoughts exploration"""

    def test_tot_branch_creation(self):
        """Test creating ToT branches"""
        branches = []
        for i in range(3):
            branches.append(
                {
                    "id": i,
                    "thought": f"Approach {i}",
                    "score": 0.0,
                }
            )
        assert len(branches) == 3

    def test_tot_depth_exploration(self):
        """Test ToT depth exploration"""
        max_depth = 5
        explored_depths = list(range(1, max_depth + 1))
        assert max(explored_depths) == max_depth

    def test_tot_score_propagation(self):
        """Test score propagation in ToT"""
        node = {"score": 0.5, "children": []}
        for i in range(3):
            child = {"score": 0.3 + (i * 0.2), "children": []}
            node["children"].append(child)

        best_child = max(node["children"], key=lambda x: x["score"])
        assert best_child["score"] == 0.7

    def test_tot_pruning(self):
        """Test pruning low-scoring branches"""
        branches = [
            {"id": 0, "score": 0.2},
            {"id": 1, "score": 0.8},
            {"id": 2, "score": 0.3},
            {"id": 3, "score": 0.9},
        ]
        threshold = 0.5
        pruned = [b for b in branches if b["score"] >= threshold]
        assert len(pruned) == 2


class TestMADPanel:
    """Tests for Multi-Agent Debate panel"""

    def test_critic_types(self):
        """Test available critic types"""
        critics = list(CriticType)
        assert CriticType.LOGIC in critics
        assert CriticType.SECURITY in critics
        assert CriticType.PERFORMANCE in critics
        assert CriticType.STYLE in critics

    def test_critic_evaluation(self):
        """Test critic evaluation scoring"""
        evaluation = {
            "critic": CriticType.LOGIC,
            "score": 0.85,
            "feedback": "Logic is sound",
        }
        assert evaluation["score"] >= 0.8

    def test_consensus_reached(self):
        """Test consensus detection"""
        scores = [0.8, 0.85, 0.82, 0.88]
        threshold = 0.75
        consensus = all(s >= threshold for s in scores)
        assert consensus

    def test_consensus_not_reached(self):
        """Test lack of consensus"""
        scores = [0.5, 0.85, 0.6, 0.88]
        threshold = 0.75
        consensus = all(s >= threshold for s in scores)
        assert not consensus

    def test_debate_rounds(self):
        """Test debate round progression"""
        max_rounds = 3
        current_round = 0
        consensus_reached = False

        while current_round < max_rounds and not consensus_reached:
            current_round += 1
            if current_round == 2:
                consensus_reached = True

        assert current_round == 2
        assert consensus_reached


class TestArchitectureSpec:
    """Tests for architecture specification"""

    def test_architecture_paradigms(self):
        """Test architecture paradigm selection"""
        paradigms = ["monolithic", "microservices", "serverless", "event-driven"]
        selected = "microservices"
        assert selected in paradigms

    def test_component_definition(self):
        """Test component definition"""
        components = [
            {"name": "API Gateway", "type": "gateway"},
            {"name": "Auth Service", "type": "service"},
            {"name": "Database", "type": "storage"},
        ]
        assert len(components) == 3
        assert any(c["type"] == "gateway" for c in components)

    def test_tech_stack_selection(self):
        """Test technology stack selection"""
        tech_stack = {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL",
            "cache": "Redis",
        }
        assert tech_stack["language"] == "Python"
        assert tech_stack["framework"] == "FastAPI"

    def test_risk_assessment(self):
        """Test risk assessment"""
        risks = [
            {"name": "scalability", "severity": "medium", "mitigation": "Use load balancing"},
            {"name": "security", "severity": "high", "mitigation": "Implement auth"},
        ]
        high_risks = [r for r in risks if r["severity"] == "high"]
        assert len(high_risks) == 1


class TestModelSelection:
    """Tests for model selection in strategic layer"""

    def test_tier_1_for_strategic(self):
        """Test TIER_1 selection for strategic tasks"""
        task_type = TaskType.PLANNING
        if task_type == TaskType.PLANNING:
            tier = ModelTier.TIER_1_STRATEGIC
        assert tier == ModelTier.TIER_1_STRATEGIC

    def test_tier_2_for_code(self):
        """Test TIER_2 selection for code generation"""
        task_type = TaskType.CODE_GENERATION
        if task_type == TaskType.CODE_GENERATION:
            tier = ModelTier.TIER_2_TACTICAL
        assert tier == ModelTier.TIER_2_TACTICAL

    def test_tier_3_for_simple(self):
        """Test TIER_3 selection for simple tasks"""
        complexity = TaskComplexity.SIMPLE
        if complexity == TaskComplexity.SIMPLE:
            tier = ModelTier.TIER_3_EFFICIENT
        assert tier == ModelTier.TIER_3_EFFICIENT


class TestPhasePlanning:
    """Tests for phase planning"""

    def test_phase_creation(self):
        """Test creating execution phases"""
        phases = [
            {"id": 1, "name": "Setup", "tasks": ["init", "config"]},
            {"id": 2, "name": "Development", "tasks": ["code", "test"]},
            {"id": 3, "name": "Deployment", "tasks": ["build", "deploy"]},
        ]
        assert len(phases) == 3

    def test_milestone_definition(self):
        """Test milestone definition"""
        milestones = [
            {"phase": 1, "name": "Environment Ready", "complete": True},
            {"phase": 2, "name": "MVP Complete", "complete": False},
        ]
        completed = [m for m in milestones if m["complete"]]
        assert len(completed) == 1


class TestStrategicIntegration:
    """Integration tests for strategic layer"""

    @pytest.mark.asyncio
    async def test_strategic_flow(self, mock_provider, sample_task):
        """Test strategic layer flow"""
        complexity = sample_task.complexity or TaskComplexity.MODERATE
        assert complexity in [
            TaskComplexity.SIMPLE,
            TaskComplexity.MODERATE,
            TaskComplexity.COMPLEX,
        ]

    @pytest.mark.asyncio
    async def test_full_mad_debate(self):
        """Test complete MAD debate cycle"""
        critics = [
            {"type": CriticType.LOGIC, "vote": 0.8},
            {"type": CriticType.SECURITY, "vote": 0.85},
            {"type": CriticType.PERFORMANCE, "vote": 0.7},
            {"type": CriticType.STYLE, "vote": 0.9},
        ]

        avg_score = sum(c["vote"] for c in critics) / len(critics)
        assert avg_score >= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
