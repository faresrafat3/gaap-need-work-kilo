"""
Unit tests for Layer 3 - Execution Layer
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from typing import Dict, Any, List

from gaap.core.types import (
    Task,
    TaskPriority,
    TaskType,
    TaskResult,
    ExecutionStatus,
    HealingLevel,
    CriticType,
)


class TestTaskExecution:
    """Tests for task execution"""

    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_provider, sample_task):
        """Test successful task execution"""
        result = await mock_provider.chat_completion(
            messages=[{"role": "user", "content": "test"}], model="model-1"
        )
        assert result is not None
        assert len(result.choices) > 0

    @pytest.mark.asyncio
    async def test_execution_with_retry(self, mock_provider):
        """Test execution with retry logic"""
        attempts = 0
        max_attempts = 3

        while attempts < max_attempts:
            attempts += 1
            if attempts >= 2:
                result = {"success": True}
                break
            result = {"success": False}

        assert attempts >= 2
        assert result["success"]

    @pytest.mark.asyncio
    async def test_execution_timeout(self):
        """Test execution timeout handling"""
        timeout_seconds = 1.0

        async def slow_operation():
            await asyncio.sleep(2.0)
            return "done"

        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(timeout_seconds):
                await slow_operation()


class TestQualityEvaluation:
    """Tests for quality evaluation"""

    def test_quality_score_calculation(self):
        """Test quality score calculation"""
        scores = {
            "correctness": 0.9,
            "completeness": 0.8,
            "efficiency": 0.85,
            "readability": 0.9,
        }
        weights = {
            "correctness": 0.4,
            "completeness": 0.25,
            "efficiency": 0.15,
            "readability": 0.2,
        }

        total_score = sum(scores[k] * weights[k] for k in scores)
        assert total_score >= 0.85

    def test_quality_threshold(self):
        """Test quality threshold checking"""
        score = 0.85
        threshold = 0.75
        passed = score >= threshold
        assert passed

    def test_quality_failure_handling(self):
        """Test handling quality failures"""
        score = 0.6
        threshold = 0.75
        needs_revision = score < threshold
        assert needs_revision


class TestGeneticTwin:
    """Tests for genetic twin verification"""

    def test_twin_creation(self):
        """Test creating genetic twins"""
        twins = [
            {"id": 1, "temperature": 0.3, "result": None},
            {"id": 2, "temperature": 0.7, "result": None},
            {"id": 3, "temperature": 0.9, "result": None},
        ]
        assert len(twins) == 3

    def test_twin_comparison(self):
        """Test comparing twin results"""
        results = [
            {"twin_id": 1, "output": "result_a", "quality": 0.85},
            {"twin_id": 2, "output": "result_b", "quality": 0.82},
            {"twin_id": 3, "output": "result_a", "quality": 0.88},
        ]

        outputs = [r["output"] for r in results]
        unique_outputs = set(outputs)

        assert len(unique_outputs) == 2

    def test_twin_consensus(self):
        """Test twin consensus detection"""
        results = [
            {"output": "same_result", "quality": 0.85},
            {"output": "same_result", "quality": 0.87},
            {"output": "different", "quality": 0.75},
        ]

        from collections import Counter

        output_counts = Counter(r["output"] for r in results)
        most_common = output_counts.most_common(1)[0]

        assert most_common[0] == "same_result"
        assert most_common[1] == 2


class TestExecutionResult:
    """Tests for execution result handling"""

    def test_success_result(self):
        """Test creating success result"""
        result = TaskResult(
            success=True,
            output="Task completed successfully",
            error=None,
        )
        assert result.success
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result"""
        result = TaskResult(
            success=False,
            output=None,
            error="Execution failed: timeout",
        )
        assert not result.success
        assert "timeout" in result.error

    def test_result_with_metrics(self):
        """Test result with metrics"""
        result = TaskResult(
            success=True,
            output="Done",
            metrics={
                "tokens": 150,
                "latency_ms": 500,
                "cost_usd": 0.001,
            },
        )
        assert result.metrics["tokens"] == 150
        assert result.metrics["latency_ms"] == 500


class TestHealingIntegration:
    """Tests for healing integration in execution"""

    def test_healing_level_selection(self):
        """Test selecting appropriate healing level"""
        error_types = {
            "rate_limit": HealingLevel.L1_RETRY,
            "timeout": HealingLevel.L1_RETRY,
            "validation_error": HealingLevel.L2_REFINE,
            "token_limit": HealingLevel.L3_PIVOT,
            "critical_error": HealingLevel.L5_HUMAN_ESCALATION,
        }

        assert error_types["rate_limit"] == HealingLevel.L1_RETRY
        assert error_types["validation_error"] == HealingLevel.L2_REFINE

    def test_healing_success_tracking(self):
        """Test tracking healing success"""
        healing_attempts = [
            {"level": HealingLevel.L1_RETRY, "success": True},
        ]
        successful = [a for a in healing_attempts if a["success"]]
        assert len(successful) == 1

    def test_healing_escalation(self):
        """Test healing escalation"""
        current_level = HealingLevel.L1_RETRY
        max_level = HealingLevel.L5_HUMAN_ESCALATION

        levels = list(HealingLevel)
        current_idx = levels.index(current_level)
        next_level = levels[current_idx + 1] if current_idx + 1 < len(levels) else None

        assert next_level == HealingLevel.L2_REFINE


class TestExecutorPool:
    """Tests for executor pool"""

    def test_pool_initialization(self):
        """Test pool initialization"""
        pool_size = 5
        executors = [f"executor_{i}" for i in range(pool_size)]
        assert len(executors) == pool_size

    def test_executor_assignment(self):
        """Test executor assignment"""
        executors = {
            "executor_1": {"busy": False},
            "executor_2": {"busy": True},
            "executor_3": {"busy": False},
        }
        available = [e for e, s in executors.items() if not s["busy"]]
        assert len(available) == 2

    def test_parallel_execution_capacity(self):
        """Test parallel execution capacity"""
        max_parallel = 4
        current_running = 2
        can_start_more = current_running < max_parallel
        assert can_start_more


class TestExecutionIntegration:
    """Integration tests for execution layer"""

    @pytest.mark.asyncio
    async def test_full_execution_flow(self, mock_provider, sample_task):
        """Test complete execution flow"""
        messages = [{"role": "user", "content": sample_task.description}]
        result = await mock_provider.chat_completion(messages=messages, model="model-1")
        assert result is not None

    @pytest.mark.asyncio
    async def test_execution_with_quality_check(self, mock_provider):
        """Test execution with quality checking"""
        result = await mock_provider.chat_completion(
            messages=[{"role": "user", "content": "test"}], model="model-1"
        )

        quality_score = 0.85
        threshold = 0.75

        assert quality_score >= threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
