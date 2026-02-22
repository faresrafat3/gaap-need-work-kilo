"""
The Gauntlet: E2E Scenario Runner

End-to-end tests that verify the full GAAP system works correctly.
These tests simulate real-world scenarios and check outputs.

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md
"""

import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaap.core.types import TaskPriority, TaskType
from gaap.layers import Layer1Strategic
from gaap.layers.layer0_interface import IntentType, StructuredIntent


@pytest.fixture
def mock_provider_for_gauntlet():
    """Provider mock with realistic responses for gauntlet tests."""
    provider = MagicMock()
    provider.name = "gauntlet-mock"
    provider.default_model = "test-model"

    async def mock_chat(messages, model=None, **kwargs):
        from gaap.core.types import (
            ChatCompletionChoice,
            ChatCompletionResponse,
            Message,
            MessageRole,
            Usage,
        )

        return ChatCompletionResponse(
            id="gauntlet-response",
            model=model or "test-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content='{"paradigm": "modular_monolith", "data_strategy": "single_database", "communication": "rest", "components": [{"name": "App", "responsibility": "Main application", "type": "module"}], "decisions": [{"aspect": "architecture", "choice": "modular_monolith", "reasoning": "Simple for MVP"}], "risks": [], "phases": [{"name": "Setup", "tasks": ["Create project"], "duration": "1 day"}], "complexity_score": 0.5, "estimated_time": "1 week"}',
                    ),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
        )

    provider.chat_completion = AsyncMock(side_effect=mock_chat)
    return provider


def create_intent(
    request_id: str,
    intent_type: IntentType = IntentType.CODE_GENERATION,
    goals: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
) -> StructuredIntent:
    """Helper to create StructuredIntent with defaults."""
    return StructuredIntent(
        request_id=request_id,
        timestamp=datetime.now(),
        intent_type=intent_type,
        explicit_goals=goals or [],
        constraints=constraints or {},
    )


class TestE2ECodeGeneration:
    """E2E tests for code generation scenarios."""

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_simple_function_generation(self, mock_provider_for_gauntlet) -> None:
        """Test generation of a simple function."""
        layer1 = Layer1Strategic(provider=mock_provider_for_gauntlet)

        intent = create_intent(
            request_id="gauntlet-001",
            intent_type=IntentType.CODE_GENERATION,
            goals=["Create a factorial function"],
            constraints={"language": "Python", "style": "Include docstring"},
        )

        spec = await layer1.process(intent)

        assert spec is not None
        assert spec.paradigm is not None
        assert spec.metadata.get("strategy_source") in ("llm", "fallback", "mcts")

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_api_design_scenario(self, mock_provider_for_gauntlet) -> None:
        """Test API design scenario."""
        layer1 = Layer1Strategic(provider=mock_provider_for_gauntlet)

        intent = create_intent(
            request_id="gauntlet-002",
            intent_type=IntentType.PLANNING,
            goals=["Design a REST API for user management"],
            constraints={"framework": "FastAPI", "features": "authentication"},
        )
        intent.metadata = {"complexity": None, "priority": TaskPriority.HIGH}

        spec = await layer1.process(intent)

        assert spec is not None
        assert spec.communication is not None

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_mcts_for_critical_task(self, mock_provider_for_gauntlet) -> None:
        """Test MCTS is used for critical tasks."""
        layer1 = Layer1Strategic(provider=mock_provider_for_gauntlet, enable_mcts=True)

        intent = create_intent(
            request_id="gauntlet-003",
            intent_type=IntentType.PLANNING,
            goals=["Design scalable microservices system"],
            constraints={},
        )
        intent.metadata = {
            "complexity": None,
            "priority": TaskPriority.CRITICAL,
        }

        spec = await layer1.process(intent)

        assert spec is not None
        stats = layer1.get_stats()
        assert stats["mcts_enabled"] is True


class TestE2EGauntletRunner:
    """Tests using the GauntletRunner for file-based assertions."""

    @pytest.mark.gauntlet
    def test_gauntlet_runner_file_checks(self, gauntlet_runner, tmp_path) -> None:
        """Test gauntlet runner can check file existence."""
        test_file = tmp_path / "app.py"
        test_file.write_text("print('hello')")

        assert gauntlet_runner.check_file_exists("app.py")
        assert not gauntlet_runner.check_file_exists("nonexistent.py")

    @pytest.mark.gauntlet
    def test_gauntlet_runner_content_checks(self, gauntlet_runner, tmp_path) -> None:
        """Test gauntlet runner can check file content."""
        test_file = tmp_path / "code.py"
        test_file.write_text("def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)")

        assert gauntlet_runner.check_file_contains("code.py", "factorial")
        assert not gauntlet_runner.check_file_contains("code.py", "fibonacci")

    @pytest.mark.gauntlet
    def test_gauntlet_runner_multiple_files(self, gauntlet_runner, tmp_path) -> None:
        """Test checking multiple files at once."""
        (tmp_path / "app.py").write_text("# app")
        (tmp_path / "requirements.txt").write_text("pytest")
        (tmp_path / "README.md").write_text("# Project")

        results = gauntlet_runner.check_files_exist(
            [
                "app.py",
                "requirements.txt",
                "README.md",
                "missing.py",
            ]
        )

        assert results["app.py"] is True
        assert results["requirements.txt"] is True
        assert results["README.md"] is True
        assert results["missing.py"] is False

    @pytest.mark.gauntlet
    def test_gauntlet_summary(self, gauntlet_runner, tmp_path) -> None:
        """Test gauntlet summary generation."""
        (tmp_path / "exists.py").write_text("pass")

        gauntlet_runner.check_file_exists("exists.py")
        gauntlet_runner.check_file_exists("missing.py")
        gauntlet_runner.check_file_contains("exists.py", "pass")

        summary = gauntlet_runner.get_summary()

        assert summary["total_checks"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1


class TestSemanticAssertions:
    """Tests for LLM-as-a-Judge semantic assertions."""

    @pytest.mark.gauntlet
    def test_semantic_similarity_pass(self, semantic_judge) -> None:
        """Test semantic similarity assertion passes."""
        semantic_judge.assert_similar(
            "The function calculates factorial",
            "The factorial calculation function",
            threshold=0.3,
        )

    @pytest.mark.gauntlet
    def test_semantic_similarity_fail(self, semantic_judge) -> None:
        """Test semantic similarity assertion fails appropriately."""
        with pytest.raises(AssertionError, match="Semantic similarity"):
            semantic_judge.assert_similar(
                "Python is a programming language",
                "The weather is sunny today",
                threshold=0.5,
            )

    @pytest.mark.gauntlet
    def test_concept_presence(self, semantic_judge) -> None:
        """Test concept presence assertion."""
        semantic_judge.assert_contains_concepts(
            "This function implements a REST API with authentication",
            ["REST", "API", "authentication"],
        )

    @pytest.mark.gauntlet
    def test_concept_missing(self, semantic_judge) -> None:
        """Test concept presence assertion fails when missing."""
        with pytest.raises(AssertionError, match="Missing concepts"):
            semantic_judge.assert_contains_concepts(
                "This is a simple function",
                ["REST", "database"],
            )

    @pytest.mark.gauntlet
    def test_response_quality_evaluation(self, semantic_judge) -> None:
        """Test response quality evaluation."""
        score = semantic_judge.evaluate_response_quality(
            "This API uses REST architecture with JSON responses and proper error handling",
            ["REST", "JSON", "error handling"],
        )
        assert score == 1.0

        score = semantic_judge.evaluate_response_quality(
            "This is a simple solution",
            ["REST", "JSON", "error handling"],
        )
        assert score == 0.0


class TestChaosTesting:
    """Tests for chaos/fault injection."""

    @pytest.mark.chaos
    def test_chaos_monkey_network_failure(self, chaos_monkey) -> None:
        """Test chaos monkey can inject network failures."""
        chaos_monkey.failure_rate = 1.0

        assert chaos_monkey.inject_network_failure() is True

        stats = chaos_monkey.get_stats()
        assert stats["injected_failures"] >= 1

    @pytest.mark.chaos
    def test_chaos_monkey_timeout_injection(self, chaos_monkey) -> None:
        """Test chaos monkey can inject timeouts."""
        chaos_monkey.failure_rate = 1.0

        timeout = chaos_monkey.inject_timeout(5.0)
        assert timeout == 50.0

    @pytest.mark.chaos
    def test_chaos_monkey_data_corruption(self, chaos_monkey) -> None:
        """Test chaos monkey can corrupt data."""
        chaos_monkey.failure_rate = 1.0

        data = {"key1": "value1", "key2": "value2"}
        corrupted = chaos_monkey.corrupt_data(data)

        assert corrupted != data
        assert None in corrupted.values()

    @pytest.mark.chaos
    def test_chaos_monkey_no_failure(self, chaos_monkey) -> None:
        """Test chaos monkey respects failure rate."""
        chaos_monkey.failure_rate = 0.0

        assert chaos_monkey.inject_network_failure() is False
        assert chaos_monkey.corrupt_data({"a": 1}) == {"a": 1}

    @pytest.mark.chaos
    @pytest.mark.asyncio
    async def test_resilient_provider_with_chaos(self, resilient_provider) -> None:
        """Test resilient provider handles chaos."""
        from gaap.core.types import Message, MessageRole

        messages = [Message(role=MessageRole.USER, content="test")]

        for _ in range(10):
            try:
                await resilient_provider.chat_completion(messages)
            except ConnectionError:
                pass

        stats = resilient_provider.get_stats()
        assert "chaos_stats" in stats


class TestVCRIntegration:
    """Tests for VCR cassette integration."""

    @pytest.mark.vcr
    @pytest.mark.gauntlet
    def test_vcr_config_available(self, vcr_config) -> None:
        """Test VCR configuration is available."""
        assert vcr_config["record_mode"] == "once"
        assert "cassette_library_dir" in vcr_config
        assert "filter_headers" in vcr_config


class TestLayerIntegration:
    """Integration tests for layer interactions."""

    @pytest.mark.integration
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_layer1_to_spec_flow(self, mock_provider_for_gauntlet) -> None:
        """Test complete flow from intent to architecture spec."""
        layer1 = Layer1Strategic(
            provider=mock_provider_for_gauntlet,
            tot_depth=3,
            mad_rounds=2,
            enable_mcts=False,
        )

        intent = create_intent(
            request_id="integration-001",
            intent_type=IntentType.PLANNING,
            goals=["Build a web service"],
            constraints={"language": "Python"},
        )

        spec = await layer1.process(intent)

        assert spec.spec_id is not None
        assert spec.paradigm is not None
        assert spec.decisions is not None
        assert spec.timestamp is not None

        stats = layer1.get_stats()
        assert stats["specs_created"] == 1
