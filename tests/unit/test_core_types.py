"""
Unit tests for GAAP core components
"""

import pytest

from gaap.core.exceptions import (
    GAAPException,
    ProviderError,
    ProviderRateLimitError,
    TaskError,
    TaskTimeoutError,
)
from gaap.core.types import (
    HealingLevel,
    LayerType,
    Message,
    MessageRole,
    ModelTier,
    Task,
    TaskComplexity,
    TaskPriority,
    TaskResult,
    TaskType,
)


class TestTaskTypes:
    """Test core type classes"""

    def test_task_creation(self):
        """Test creating a task"""
        task = Task(
            id="test-1",
            type=TaskType.CODE_GENERATION,
            description="Write a function",
            priority=TaskPriority.HIGH,
        )
        assert task.id == "test-1"
        assert task.type == TaskType.CODE_GENERATION
        assert task.priority == TaskPriority.HIGH

    def test_task_result(self):
        """Test creating a task result"""
        result = TaskResult(
            success=True,
            output="def hello(): pass",
            error=None,
        )
        assert result.success
        assert "def hello" in result.output

    def test_message_creation(self):
        """Test creating messages"""
        msg = Message(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_message_to_dict(self):
        """Test message serialization"""
        msg = Message(role=MessageRole.ASSISTANT, content="Response", name="bot")
        d = msg.to_dict()
        assert d["role"] == "assistant"
        assert d["content"] == "Response"
        assert d["name"] == "bot"


class TestEnums:
    """Test enum values"""

    def test_task_priority(self):
        """Test TaskPriority enum"""
        priorities = list(TaskPriority)
        assert TaskPriority.CRITICAL in priorities
        assert TaskPriority.HIGH in priorities
        assert TaskPriority.NORMAL in priorities
        assert TaskPriority.LOW in priorities

    def test_layer_type(self):
        """Test LayerType enum"""
        assert LayerType.INTERFACE.value == 0
        assert LayerType.STRATEGIC.value == 1
        assert LayerType.TACTICAL.value == 2
        assert LayerType.EXECUTION.value == 3

    def test_model_tier(self):
        """Test ModelTier enum"""
        assert ModelTier.TIER_1_STRATEGIC.name == "TIER_1_STRATEGIC"
        assert ModelTier.TIER_4_PRIVATE.name == "TIER_4_PRIVATE"

    def test_healing_level(self):
        """Test HealingLevel enum"""
        levels = list(HealingLevel)
        assert len(levels) == 5
        assert HealingLevel.L1_RETRY in levels
        assert HealingLevel.L5_HUMAN_ESCALATION in levels


class TestExceptions:
    """Test custom exceptions"""

    def test_gaap_exception(self):
        """Test base GAAPException"""
        exc = GAAPException(
            message="Test error",
            details={"key": "value"},
            suggestions=["Try again"],
        )
        assert exc.message == "Test error"
        assert exc.details["key"] == "value"
        assert exc.recoverable

    def test_provider_error(self):
        """Test ProviderError"""
        exc = ProviderError(
            message="Provider failed",
            details={"provider": "test"},
        )
        assert exc.error_code == "GAAP_PRV_001"
        assert exc.error_category == "provider"

    def test_rate_limit_error(self):
        """Test ProviderRateLimitError"""
        exc = ProviderRateLimitError(
            provider_name="groq",
            retry_after=60,
        )
        assert exc.recoverable
        assert "groq" in exc.message
        assert exc.details.get("retry_after_seconds") == 60

    def test_task_timeout_error(self):
        """Test TaskTimeoutError"""
        exc = TaskTimeoutError(
            task_id="task-123",
            timeout_seconds=30.0,
        )
        assert exc.error_code == "GAAP_TSK_005"
        assert exc.recoverable

    def test_exception_to_dict(self):
        """Test exception serialization"""
        exc = TaskError(
            message="Task failed",
            details={"task_id": "t1"},
        )
        d = exc.to_dict()
        assert d["error_code"] == "GAAP_TSK_001"
        assert d["message"] == "Task failed"


class TestTaskComplexity:
    """Test TaskComplexity calculations"""

    def test_complexity_values(self):
        """Test complexity enum"""
        complexities = list(TaskComplexity)
        assert TaskComplexity.SIMPLE in complexities
        assert TaskComplexity.COMPLEX in complexities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
