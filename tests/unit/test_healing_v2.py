"""
Tests for GAAP Healing System v2 (Reflexion & Semantic Classification)

Implements tests for:
- docs/evolution_plan_2026/26_HEALING_AUDIT_SPEC.md
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any

import pytest

from gaap.healing import (
    ErrorCategory,
    ErrorClassifier,
    HealingConfig,
    PatternDetectionConfig,
    PostMortemConfig,
    ReflexionConfig,
    ReflexionEngine,
    Reflection,
    ReflectionDepth,
    SelfHealingSystem,
    SemanticClassifierConfig,
    SemanticErrorClassifier,
    create_healing_config,
)
from gaap.healing.healer import ErrorContext, RecoveryAction
from gaap.core.types import HealingLevel, Task, TaskType


def async_test(coro):
    """Decorator to run async tests without pytest-asyncio."""

    def wrapper(*args, **kwargs):
        return asyncio.run(coro(*args, **kwargs))

    return wrapper


class TestReflexionEngine:
    """Tests for ReflexionEngine."""

    def test_reflection_creation(self):
        reflection = Reflection(
            failure_analysis="Test failure",
            root_cause="Root cause",
            proposed_fix="Fix it",
            confidence=0.8,
        )

        assert reflection.failure_analysis == "Test failure"
        assert reflection.root_cause == "Root cause"
        assert reflection.proposed_fix == "Fix it"
        assert reflection.confidence == 0.8

    def test_reflection_to_dict(self):
        reflection = Reflection(
            failure_analysis="Test",
            proposed_fix="Fix",
            lessons_learned=["Lesson 1", "Lesson 2"],
        )

        data = reflection.to_dict()

        assert data["failure_analysis"] == "Test"
        assert data["proposed_fix"] == "Fix"
        assert len(data["lessons_learned"]) == 2

    def test_reflection_from_dict(self):
        data = {
            "failure_analysis": "Test",
            "root_cause": "Cause",
            "proposed_fix": "Fix",
            "confidence": 0.9,
            "alternative_approaches": ["Alt 1"],
            "lessons_learned": ["Lesson"],
            "depth": "MODERATE",
        }

        reflection = Reflection.from_dict(data)

        assert reflection.failure_analysis == "Test"
        assert reflection.root_cause == "Cause"
        assert reflection.confidence == 0.9

    def test_reflection_to_prompt_context(self):
        reflection = Reflection(
            failure_analysis="Syntax error in code",
            root_cause="Missing closing bracket",
            proposed_fix="Add closing bracket",
            alternative_approaches=["Use linter", "Check brackets"],
            lessons_learned=["Always validate syntax"],
        )

        context = reflection.to_prompt_context()

        assert "PREVIOUS ATTEMPT FAILED" in context
        assert "Syntax error in code" in context
        assert "Missing closing bracket" in context
        assert "Add closing bracket" in context
        assert "Use linter" in context

    def test_engine_fallback_reflection(self):
        engine = ReflexionEngine(llm_provider=None)

        error = SyntaxError("invalid syntax")
        reflection = engine._fallback_reflection(
            error=error,
            task_description="Write a function",
            previous_output="def foo(",
        )

        assert reflection.failure_analysis != ""
        assert reflection.proposed_fix is not None
        assert reflection.depth == ReflectionDepth.SURFACE

    def test_engine_fallback_timeout(self):
        engine = ReflexionEngine(llm_provider=None)

        error = TimeoutError("Operation timed out")
        reflection = engine._fallback_reflection(
            error=error,
            task_description="Long computation",
            previous_output="",
        )

        assert "timeout" in reflection.failure_analysis.lower()
        assert reflection.confidence >= 0.4

    def test_engine_fallback_network(self):
        engine = ReflexionEngine(llm_provider=None)

        error = ConnectionError("Connection refused")
        reflection = engine._fallback_reflection(
            error=error,
            task_description="API call",
            previous_output="",
        )

        assert (
            "network" in reflection.failure_analysis.lower()
            or "connection" in reflection.failure_analysis.lower()
        )

    def test_refine_prompt(self):
        engine = ReflexionEngine(llm_provider=None)

        reflection = Reflection(
            failure_analysis="Wrong output",
            proposed_fix="Use different algorithm",
        )

        refined = engine.refine_prompt("Original task", reflection)

        assert "Original task" in refined
        assert "PREVIOUS ATTEMPT FAILED" in refined
        assert "Wrong output" in refined


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = ""):
        self.response = response
        self.calls: list[dict] = []

    async def complete(self, messages: list[dict], **kwargs) -> str:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self.response


class TestReflexionEngineAsync:
    """Async tests for ReflexionEngine."""

    @async_test
    async def test_reflect_with_llm(self):
        llm_response = """
**Failure Analysis:** The code failed to parse

**Root Cause:** Missing import

**Proposed Fix:** Add import statement

**Alternative Approaches:**
1. Use try/except
2. Check imports

**Lessons Learned:**
- Always check imports

**Confidence:** 0.85
"""
        mock_provider = MockLLMProvider(response=llm_response)
        engine = ReflexionEngine(llm_provider=mock_provider)

        reflection = await engine.reflect(
            error=ImportError("No module named 'foo'"),
            task_description="Import and use foo",
            previous_output="import foo",
        )

        assert reflection.failure_analysis != ""
        assert reflection.root_cause is not None
        assert reflection.confidence > 0

    @async_test
    async def test_reflect_uses_cache(self):
        mock_provider = MockLLMProvider(response="**Failure Analysis:** Test\n**Confidence:** 0.5")
        engine = ReflexionEngine(llm_provider=mock_provider)

        error = ValueError("test")

        await engine.reflect(error, "task1", "")
        await engine.reflect(error, "task1", "")

        assert len(mock_provider.calls) == 1

    @async_test
    async def test_reflect_fallback_on_llm_error(self):
        class FailingProvider:
            async def complete(self, messages, **kwargs):
                raise RuntimeError("LLM failed")

        engine = ReflexionEngine(llm_provider=FailingProvider())

        reflection = await engine.reflect(
            error=ValueError("test"),
            task_description="test task",
        )

        assert reflection.failure_analysis != ""


class TestSemanticErrorClassifier:
    """Tests for SemanticErrorClassifier."""

    @async_test
    async def test_classify_with_regex_match(self):
        classifier = SemanticErrorClassifier(llm_provider=None)

        error = ConnectionError("Connection refused")
        result = await classifier.classify(error, "test task")

        assert result == ErrorCategory.TRANSIENT

    @async_test
    async def test_classify_unknown_without_llm(self):
        classifier = SemanticErrorClassifier(llm_provider=None)

        error = ValueError("Some unknown error pattern xyz123")
        result = await classifier.classify(error, "test task")

        assert result == ErrorCategory.UNKNOWN

    @async_test
    async def test_classify_with_llm(self):
        mock_provider = MockLLMProvider(response="LOGIC")
        classifier = SemanticErrorClassifier(llm_provider=mock_provider)

        error = ValueError("Output doesn't match expected format")
        result = await classifier.classify(error, "Generate JSON output")

        assert result == ErrorCategory.LOGIC


class TestSelfHealingSystemV2:
    """Tests for SelfHealingSystem with Reflexion integration."""

    def test_healer_accepts_reflexion_engine(self):
        mock_provider = MockLLMProvider(
            response="**Failure Analysis:** Test\n**Proposed Fix:** Fix it\n**Confidence:** 0.8"
        )
        reflexion = ReflexionEngine(llm_provider=mock_provider)
        healer = SelfHealingSystem(
            max_level=HealingLevel.L2_REFINE,
            reflexion_engine=reflexion,
        )
        assert healer._reflexion_engine is not None

    def test_healer_accepts_failure_store(self):
        healer = SelfHealingSystem(failure_store=None)
        assert healer._failure_store is None

    @async_test
    async def test_heal_with_reflexion_refinement(self):
        mock_provider = MockLLMProvider(
            response="**Failure Analysis:** Test\n**Proposed Fix:** Fix it\n**Confidence:** 0.8"
        )
        reflexion = ReflexionEngine(llm_provider=mock_provider)
        healer = SelfHealingSystem(
            max_level=HealingLevel.L2_REFINE,
            reflexion_engine=reflexion,
        )

        call_count = 0

        async def execute(task: Task):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt failed")
            return "success"

        task = Task(id="test-1", description="Test task")
        result = await healer.heal(
            error=ValueError("Test error"),
            task=task,
            execute_func=execute,
        )

        assert result.success
        assert result.action == RecoveryAction.REFINE_PROMPT

    @async_test
    async def test_heal_semantic_classification(self):
        mock_provider = MockLLMProvider(response="TRANSIENT")
        classifier = SemanticErrorClassifier(llm_provider=mock_provider)

        error = ValueError("Some weird pattern that regex won't catch")
        result = await classifier.classify(error, "test task")

        assert result == ErrorCategory.TRANSIENT

    @async_test
    async def test_reflection_history_recorded(self):
        mock_provider = MockLLMProvider(
            response="**Failure Analysis:** Test\n**Proposed Fix:** Fix it\n**Confidence:** 0.8"
        )
        reflexion = ReflexionEngine(llm_provider=mock_provider)
        healer = SelfHealingSystem(
            max_level=HealingLevel.L2_REFINE,
            reflexion_engine=reflexion,
        )

        async def execute(task: Task):
            raise ValueError("Always fails")

        task = Task(id="test-3", description="Test task")

        await healer.heal(
            error=ValueError("Test error"),
            task=task,
            execute_func=execute,
        )

        assert "test-3" in healer._reflection_history


class TestReflexionWithFailureStore:
    """Tests for Reflexion integration with FailureStore."""

    @async_test
    async def test_failure_recorded_on_escalation(self):
        from gaap.meta_learning.failure_store import FailureStore, FailedTrace

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FailureStore(storage_path=str(tmpdir))

            healer = SelfHealingSystem(
                max_level=HealingLevel.L2_REFINE,
                failure_store=store,
            )

            async def always_fail(task: Task):
                raise ValueError("Always fails")

            task = Task(id="test-fail-1", description="Test failure task")

            result = await healer.heal(
                error=ValueError("Test error"),
                task=task,
                execute_func=always_fail,
            )

            assert not result.success
            assert result.action == RecoveryAction.ESCALATE

            assert len(store._failures) > 0


class TestReflectionDepth:
    """Tests for reflection depth progression."""

    def test_surface_depth(self):
        reflection = Reflection(
            failure_analysis="Quick analysis",
            depth=ReflectionDepth.SURFACE,
        )
        assert reflection.depth == ReflectionDepth.SURFACE

    def test_deep_depth_for_repeated_failures(self):
        engine = ReflexionEngine(llm_provider=None, enable_deep_reflection=True)

        assert engine._enable_deep is True


class TestErrorContextIntegration:
    """Tests for ErrorContext with new features."""

    def test_error_context_creation(self):
        ctx = ErrorContext(
            error=ValueError("test"),
            category=ErrorCategory.LOGIC,
            message="test message",
            task_id="task-1",
        )

        assert ctx.category == ErrorCategory.LOGIC
        assert ctx.task_id == "task-1"

    def test_error_context_to_dict(self):
        ctx = ErrorContext(
            error=ValueError("test"),
            category=ErrorCategory.SYNTAX,
            message="syntax error",
            task_id="task-2",
            provider="test-provider",
            model="test-model",
        )

        data = ctx.to_dict()

        assert data["category"] == "SYNTAX"
        assert data["task_id"] == "task-2"
        assert data["provider"] == "test-provider"


class TestHealingConfig:
    """Tests for HealingConfig and presets."""

    def test_default_config(self):
        config = HealingConfig()

        assert config.max_healing_level == 4
        assert config.max_retries_per_level == 1
        assert config.exponential_backoff is True
        assert config.jitter is True
        assert config.enable_learning is True

    def test_conservative_preset(self):
        config = HealingConfig.conservative()

        assert config.max_healing_level == 3
        assert config.reflexion.enable_deep_reflexion is False
        assert config.pattern_detection.detection_threshold == 2

    def test_aggressive_preset(self):
        config = HealingConfig.aggressive()

        assert config.max_healing_level == 5
        assert config.max_retries_per_level == 2
        assert config.reflexion.enable_deep_reflexion is True
        assert config.enable_parallel_recovery is True

    def test_fast_preset(self):
        config = HealingConfig.fast()

        assert config.max_healing_level == 2
        assert config.reflexion.enabled is False
        assert config.pattern_detection.enabled is False
        assert config.enable_learning is False

    def test_balanced_preset(self):
        config = HealingConfig.balanced()

        assert config.max_healing_level == 4
        assert config.reflexion.enabled is True
        assert config.pattern_detection.enabled is True

    def test_development_preset(self):
        config = HealingConfig.development()

        assert config.max_healing_level == 5
        assert config.base_delay_seconds == 0.0
        assert config.enable_observability is False

    def test_config_to_dict(self):
        config = HealingConfig.conservative()
        data = config.to_dict()

        assert data["max_healing_level"] == 3
        assert "reflexion" in data
        assert "pattern_detection" in data

    def test_config_from_dict(self):
        data = {
            "max_healing_level": 2,
            "max_retries_per_level": 3,
            "reflexion": {"enabled": False},
            "pattern_detection": {"detection_threshold": 5},
        }

        config = HealingConfig.from_dict(data)

        assert config.max_healing_level == 2
        assert config.max_retries_per_level == 3
        assert config.reflexion.enabled is False
        assert config.pattern_detection.detection_threshold == 5

    def test_create_healing_config_factory(self):
        config = create_healing_config("aggressive")
        assert config.max_healing_level == 5

        config = create_healing_config("fast", max_healing_level=3)
        assert config.max_healing_level == 3

    def test_config_validation(self):
        with pytest.raises(ValueError):
            HealingConfig(max_retries_per_level=0)

        with pytest.raises(ValueError):
            HealingConfig(base_delay_seconds=-1)

        with pytest.raises(ValueError):
            HealingConfig(max_delay_seconds=0.5, base_delay_seconds=1.0)


class TestPatternDetection:
    """Tests for pattern detection in healing system."""

    def test_error_signature_computation(self):
        healer = SelfHealingSystem()

        ctx1 = ErrorContext(
            error=ValueError("Error 123"),
            category=ErrorCategory.LOGIC,
            message="Error 123",
            task_id="task-1",
        )
        ctx2 = ErrorContext(
            error=ValueError("Error 456"),
            category=ErrorCategory.LOGIC,
            message="Error 456",
            task_id="task-2",
        )

        sig1 = healer._compute_error_signature(ctx1)
        sig2 = healer._compute_error_signature(ctx2)

        assert sig1 == sig2

    def test_pattern_detection_disabled(self):
        config = HealingConfig(pattern_detection=PatternDetectionConfig(enabled=False))
        healer = SelfHealingSystem(config=config)

        ctx = ErrorContext(
            error=ValueError("Test"),
            category=ErrorCategory.LOGIC,
            message="Test",
            task_id="task-1",
        )

        pattern = healer._detect_failure_pattern(ctx)
        assert pattern is None

    def test_pattern_detection_threshold(self):
        config = HealingConfig(
            pattern_detection=PatternDetectionConfig(
                enabled=True,
                detection_threshold=2,
            )
        )
        healer = SelfHealingSystem(config=config)

        ctx = ErrorContext(
            error=ValueError("Test error"),
            category=ErrorCategory.LOGIC,
            message="Test error",
            task_id="task-1",
        )

        assert healer._detect_failure_pattern(ctx) is None
        assert healer._detect_failure_pattern(ctx) is not None

    def test_auto_escalate_disabled(self):
        config = HealingConfig(
            pattern_detection=PatternDetectionConfig(
                auto_escalate_patterns=False,
            )
        )
        healer = SelfHealingSystem(config=config)

        assert healer._should_auto_escalate("some-pattern") is False

    def test_pattern_cooldown(self):
        config = HealingConfig(
            pattern_detection=PatternDetectionConfig(
                auto_escalate_patterns=True,
                pattern_cooldown_minutes=30,
            )
        )
        healer = SelfHealingSystem(config=config)

        assert healer._should_auto_escalate("pattern-1") is True
        assert healer._should_auto_escalate("pattern-1") is False

    def test_escalation_level_by_occurrences(self):
        healer = SelfHealingSystem()

        healer._pattern_history["pattern-1"] = [{"timestamp": 1.0} for _ in range(3)]
        assert healer._get_pattern_escalation_level("pattern-1") == HealingLevel.L3_PIVOT

        healer._pattern_history["pattern-2"] = [{"timestamp": 1.0} for _ in range(5)]
        assert healer._get_pattern_escalation_level("pattern-2") == HealingLevel.L5_HUMAN_ESCALATION


class TestDelayCalculation:
    """Tests for delay calculation with backoff and jitter."""

    def test_no_delay_first_attempt(self):
        healer = SelfHealingSystem()

        delay = healer._calculate_delay(HealingLevel.L1_RETRY, 0)
        assert delay == 0.0

    def test_exponential_backoff(self):
        config = HealingConfig(
            base_delay_seconds=1.0,
            exponential_backoff=True,
            jitter=False,
        )
        healer = SelfHealingSystem(config=config)

        delay1 = healer._calculate_delay(HealingLevel.L1_RETRY, 1)
        delay2 = healer._calculate_delay(HealingLevel.L1_RETRY, 2)

        assert delay2 > delay1

    def test_max_delay_cap(self):
        config = HealingConfig(
            base_delay_seconds=1.0,
            max_delay_seconds=2.0,
            exponential_backoff=True,
            jitter=False,
        )
        healer = SelfHealingSystem(config=config)

        delay = healer._calculate_delay(HealingLevel.L4_STRATEGY_SHIFT, 5)
        assert delay <= 2.0

    def test_jitter_adds_randomness(self):
        config = HealingConfig(
            base_delay_seconds=1.0,
            exponential_backoff=False,
            jitter=True,
        )
        healer = SelfHealingSystem(config=config)

        delays = [healer._calculate_delay(HealingLevel.L1_RETRY, 1) for _ in range(10)]

        assert len(set(delays)) > 1


class TestReflexionConfig:
    """Tests for ReflexionConfig."""

    def test_default_config(self):
        config = ReflexionConfig()

        assert config.enabled is True
        assert config.model == "gpt-4o-mini"
        assert config.enable_deep_reflexion is True
        assert config.cache_reflections is True

    def test_config_to_dict_and_from_dict(self):
        config = ReflexionConfig(
            enabled=False,
            model="gpt-4",
            deep_reflexion_threshold=5,
        )

        data = config.to_dict()
        restored = ReflexionConfig.from_dict(data)

        assert restored.enabled is False
        assert restored.model == "gpt-4"
        assert restored.deep_reflexion_threshold == 5


class TestHealerWithConfig:
    """Tests for SelfHealingSystem with config."""

    def test_healer_uses_config_max_level(self):
        config = HealingConfig(max_healing_level=2)
        healer = SelfHealingSystem(config=config)

        assert healer.max_level == HealingLevel.L2_REFINE

    def test_healer_config_property(self):
        config = HealingConfig.conservative()
        healer = SelfHealingSystem(config=config)

        assert healer.config.max_healing_level == 3

    @async_test
    async def test_heal_respects_config_retries(self):
        config = HealingConfig(
            max_healing_level=1,
            max_retries_per_level=2,
            base_delay_seconds=0.0,
        )
        healer = SelfHealingSystem(config=config)

        call_count = 0

        async def execute(task: Task):
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        task = Task(id="test-retries", description="Test")
        result = await healer.heal(
            error=ValueError("Test"),
            task=task,
            execute_func=execute,
        )

        assert not result.success
        assert call_count == 2
