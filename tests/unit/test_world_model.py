"""Tests for World Model"""

import pytest

from gaap.core.world_model import Action, Prediction, RiskLevel, WorldModel


class MockProvider:
    """Mock provider for testing"""

    default_model = "test-model"

    async def chat_completion(self, messages, model=None, temperature=0.7, max_tokens=1000):
        """Mock chat completion"""
        from gaap.core.types import ChatCompletionResponse, Choice, Message, MessageRole

        return ChatCompletionResponse(
            id="test",
            choices=[
                Choice(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content='{"risk": 0.3, "issues": ["test issue"], "suggestions": ["test suggestion"], "confidence": 0.8, "reasoning": "test"}',
                    ),
                    finish_reason="stop",
                )
            ],
            usage={"total_tokens": 100},
        )


@pytest.fixture
def world_model():
    """Create a world model without LLM"""
    return WorldModel(provider=None, enable_llm_prediction=False)


@pytest.fixture
def world_model_with_llm():
    """Create a world model with mock LLM"""
    return WorldModel(provider=MockProvider(), enable_llm_prediction=True)


class TestAction:
    """Tests for Action dataclass"""

    def test_action_creation(self):
        """Test creating an action"""
        action = Action(
            name="delete_file",
            description="Delete a file",
            task_type="file_operation",
            is_destructive=True,
        )

        assert action.name == "delete_file"
        assert action.is_destructive is True
        assert action.requires_external_access is False

    def test_action_defaults(self):
        """Test action default values"""
        action = Action(name="test", description="Test action")

        assert action.task_type == "general"
        assert action.is_destructive is False
        assert action.parameters == {}


class TestPrediction:
    """Tests for Prediction dataclass"""

    def test_prediction_creation(self):
        """Test creating a prediction"""
        prediction = Prediction(
            success_probability=0.7,
            risk_level=0.3,
            potential_issues=["Issue 1"],
            suggestions=["Suggestion 1"],
            similar_past_outcomes=["Past 1"],
            confidence=0.8,
            reasoning="Test reasoning",
        )

        assert prediction.success_probability == 0.7
        assert prediction.risk_level == 0.3
        assert len(prediction.potential_issues) == 1


class TestWorldModel:
    """Tests for WorldModel"""

    @pytest.mark.asyncio
    async def test_predict_outcome_without_llm(self, world_model):
        """Test prediction without LLM"""
        action = Action(
            name="read_file",
            description="Read a file",
            task_type="file_operation",
        )

        prediction = await world_model.predict_outcome(action)

        assert prediction.success_probability >= 0
        assert prediction.risk_level >= 0
        assert prediction.confidence > 0

    @pytest.mark.asyncio
    async def test_predict_destructive_action(self, world_model):
        """Test prediction for destructive action"""
        action = Action(
            name="delete_file",
            description="Delete a file permanently",
            is_destructive=True,
        )

        prediction = await world_model.predict_outcome(action)

        assert prediction.risk_level > 0
        assert any("destructive" in i.lower() for i in prediction.potential_issues)

    @pytest.mark.asyncio
    async def test_predict_external_access_action(self, world_model):
        """Test prediction for external access action"""
        action = Action(
            name="fetch_url",
            description="Fetch data from URL",
            requires_external_access=True,
        )

        prediction = await world_model.predict_outcome(action)

        assert prediction.risk_level > 0

    @pytest.mark.asyncio
    async def test_should_proceed_safe(self, world_model):
        """Test should_proceed for safe action"""
        prediction = Prediction(
            success_probability=0.9,
            risk_level=0.1,
            potential_issues=[],
            suggestions=[],
            similar_past_outcomes=[],
            confidence=0.9,
            reasoning="Safe",
        )

        should, reason = await world_model.should_proceed(prediction)

        assert should is True
        assert "Safe" in reason or "proceed" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_proceed_critical_risk(self, world_model):
        """Test should_proceed for critical risk"""
        prediction = Prediction(
            success_probability=0.1,
            risk_level=0.95,
            potential_issues=["Critical issue"],
            suggestions=[],
            similar_past_outcomes=[],
            confidence=0.8,
            reasoning="Dangerous",
        )

        should, reason = await world_model.should_proceed(prediction)

        assert should is False
        assert "critical" in reason.lower() or "risk" in reason.lower()

    @pytest.mark.asyncio
    async def test_should_proceed_high_risk_low_confidence(self, world_model):
        """Test should_proceed for high risk with low confidence"""
        prediction = Prediction(
            success_probability=0.3,
            risk_level=0.8,
            potential_issues=["High risk issue"],
            suggestions=[],
            similar_past_outcomes=[],
            confidence=0.3,
            reasoning="Uncertain",
        )

        should, reason = await world_model.should_proceed(prediction)

        assert should is False
        assert "confidence" in reason.lower() or "information" in reason.lower()

    @pytest.mark.asyncio
    async def test_predict_with_context(self, world_model):
        """Test prediction with context"""
        action = Action(
            name="deploy",
            description="Deploy to production",
        )

        context = {
            "environment": "production",
            "has_backup": True,
        }

        prediction = await world_model.predict_outcome(action, context)

        assert prediction is not None

    def test_get_stats(self, world_model):
        """Test getting statistics"""
        stats = world_model.get_stats()

        assert "predictions_made" in stats
        assert "llm_enabled" in stats


class TestWorldModelWithLLM:
    """Tests for WorldModel with LLM"""

    @pytest.mark.asyncio
    async def test_predict_with_llm(self, world_model_with_llm):
        """Test prediction with LLM"""
        action = Action(
            name="test_action",
            description="Test action for LLM prediction",
        )

        prediction = await world_model_with_llm.predict_outcome(action)

        assert prediction is not None
        assert prediction.processing_time_ms >= 0


class TestRiskLevels:
    """Tests for risk level classification"""

    def test_risk_level_values(self):
        """Test risk level enum values"""
        assert RiskLevel.SAFE.value == 0.0
        assert RiskLevel.LOW.value == 0.3
        assert RiskLevel.MEDIUM.value == 0.5
        assert RiskLevel.HIGH.value == 0.7
        assert RiskLevel.CRITICAL.value == 0.9
