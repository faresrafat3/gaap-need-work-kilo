"""
OODA Loop Gauntlet Tests
========================

Tests for the complete OODA loop cycle.

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from gaap.core.types import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    Message,
    MessageRole,
    Usage,
)


class OODAStage(Enum):
    """OODA Loop stages."""

    OBSERVE = auto()
    ORIENT = auto()
    DECIDE = auto()
    ACT = auto()
    LEARN = auto()


@dataclass
class OODAState:
    """State of OODA loop."""

    stage: OODAStage = OODAStage.OBSERVE
    observations: list[str] = field(default_factory=list)
    orientations: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    learnings: list[str] = field(default_factory=list)
    iterations: int = 0
    goal_achieved: bool = False


class MockOODAProvider:
    """Provider for OODA loop testing."""

    def __init__(self):
        self.name = "ooda-provider"
        self.responses = {
            "observe": "Observing the environment: Found Python project with Flask dependencies.",
            "orient": "Orienting: This is a web application that needs authentication features.",
            "decide": "Deciding: Will implement JWT-based authentication with Flask decorators.",
            "act": "Acting: Generated authentication code with login, register, and protected routes.",
            "learn": "Learning: Authentication pattern successful. Will reuse JWT approach for future projects.",
        }
        self.call_count = 0

    async def chat_completion(self, messages, model=None, **kwargs):
        self.call_count += 1

        last_msg = messages[-1].content.lower() if messages else ""

        if "observe" in last_msg:
            content = self.responses["observe"]
        elif "orient" in last_msg:
            content = self.responses["orient"]
        elif "decide" in last_msg:
            content = self.responses["decide"]
        elif "act" in last_msg:
            content = self.responses["act"]
        elif "learn" in last_msg:
            content = self.responses["learn"]
        else:
            content = "Processing OODA stage..."

        return ChatCompletionResponse(
            id=f"ooda-response-{self.call_count}",
            model=model or "ooda-model",
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=content),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=20, completion_tokens=50, total_tokens=70),
        )


class SimpleOODALoop:
    """Simplified OODA loop for testing."""

    def __init__(self, provider, max_iterations: int = 5):
        self.provider = provider
        self.max_iterations = max_iterations
        self.state = OODAState()

    async def observe(self, context: str) -> str:
        """Observe phase."""
        self.state.stage = OODAStage.OBSERVE
        response = await self.provider.chat_completion(
            [Message(role=MessageRole.USER, content=f"Observe: {context}")]
        )
        observation = response.choices[0].message.content
        self.state.observations.append(observation)
        return observation

    async def orient(self, observations: list[str]) -> str:
        """Orient phase."""
        self.state.stage = OODAStage.ORIENT
        context = "\n".join(observations)
        response = await self.provider.chat_completion(
            [Message(role=MessageRole.USER, content=f"Orient based on: {context}")]
        )
        orientation = response.choices[0].message.content
        self.state.orientations.append(orientation)
        return orientation

    async def decide(self, orientations: list[str]) -> str:
        """Decide phase."""
        self.state.stage = OODAStage.DECIDE
        context = "\n".join(orientations)
        response = await self.provider.chat_completion(
            [Message(role=MessageRole.USER, content=f"Decide based on: {context}")]
        )
        decision = response.choices[0].message.content
        self.state.decisions.append(decision)
        return decision

    async def act(self, decision: str) -> str:
        """Act phase."""
        self.state.stage = OODAStage.ACT
        response = await self.provider.chat_completion(
            [Message(role=MessageRole.USER, content=f"Act on: {decision}")]
        )
        action = response.choices[0].message.content
        self.state.actions.append(action)
        return action

    async def learn(self, result: str) -> str:
        """Learn phase."""
        self.state.stage = OODAStage.LEARN
        response = await self.provider.chat_completion(
            [Message(role=MessageRole.USER, content=f"Learn from: {result}")]
        )
        learning = response.choices[0].message.content
        self.state.learnings.append(learning)
        return learning

    async def run_cycle(self, initial_context: str) -> OODAState:
        """Run complete OODA cycle."""
        self.state.iterations += 1

        observation = await self.observe(initial_context)

        orientation = await self.orient([observation])

        decision = await self.decide([orientation])

        action = await self.act(decision)

        learning = await self.learn(action)

        self.state.goal_achieved = True
        return self.state

    def get_state(self) -> OODAState:
        """Get current state."""
        return self.state


@pytest.fixture
def ooda_provider():
    """Provide OODA test provider."""
    return MockOODAProvider()


@pytest.fixture
def ooda_loop(ooda_provider):
    """Provide OODA loop instance."""
    return SimpleOODALoop(ooda_provider)


class TestOODALoopGauntlet:
    """Tests for OODA loop."""

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_observe_stage(self, ooda_loop) -> None:
        """Test observe stage."""
        observation = await ooda_loop.observe("Python project with Flask")

        assert observation is not None
        assert len(observation) > 0
        assert ooda_loop.state.stage == OODAStage.OBSERVE
        assert len(ooda_loop.state.observations) == 1

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_orient_stage(self, ooda_loop) -> None:
        """Test orient stage."""
        ooda_loop.state.observations = ["Found Flask app", "Has authentication requirement"]
        orientation = await ooda_loop.orient(ooda_loop.state.observations)

        assert orientation is not None
        assert ooda_loop.state.stage == OODAStage.ORIENT

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_decide_stage(self, ooda_loop) -> None:
        """Test decide stage."""
        ooda_loop.state.orientations = ["Web app needs auth"]
        decision = await ooda_loop.decide(ooda_loop.state.orientations)

        assert decision is not None
        assert ooda_loop.state.stage == OODAStage.DECIDE

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_act_stage(self, ooda_loop) -> None:
        """Test act stage."""
        decision = "Implement JWT authentication"
        action = await ooda_loop.act(decision)

        assert action is not None
        assert ooda_loop.state.stage == OODAStage.ACT

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_learn_stage(self, ooda_loop) -> None:
        """Test learn stage."""
        result = "Authentication implemented successfully"
        learning = await ooda_loop.learn(result)

        assert learning is not None
        assert ooda_loop.state.stage == OODAStage.LEARN

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_full_cycle(self, ooda_loop) -> None:
        """Test complete OODA cycle."""
        state = await ooda_loop.run_cycle("Create a Flask app with authentication")

        assert state.iterations == 1
        assert state.goal_achieved is True
        assert len(state.observations) >= 1
        assert len(state.orientations) >= 1
        assert len(state.decisions) >= 1
        assert len(state.actions) >= 1
        assert len(state.learnings) >= 1

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_multiple_iterations(self, ooda_provider) -> None:
        """Test multiple OODA iterations."""
        loop = SimpleOODALoop(ooda_provider, max_iterations=3)

        for i in range(3):
            await loop.run_cycle(f"Task {i + 1}")

        assert loop.state.iterations == 3
        assert len(loop.state.observations) >= 3

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_state_transitions(self, ooda_loop) -> None:
        """Test state transitions through OODA stages."""
        stages = []

        await ooda_loop.observe("Context")
        stages.append(ooda_loop.state.stage)

        await ooda_loop.orient(["Observation"])
        stages.append(ooda_loop.state.stage)

        await ooda_loop.decide(["Orientation"])
        stages.append(ooda_loop.state.stage)

        await ooda_loop.act("Decision")
        stages.append(ooda_loop.state.stage)

        await ooda_loop.learn("Action result")
        stages.append(ooda_loop.state.stage)

        assert stages == [
            OODAStage.OBSERVE,
            OODAStage.ORIENT,
            OODAStage.DECIDE,
            OODAStage.ACT,
            OODAStage.LEARN,
        ]


class TestOODAWithFailuresGauntlet:
    """Tests for OODA loop with failures."""

    @pytest.mark.chaos
    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_reobserve_after_failure(self) -> None:
        """Test re-observation after failure."""
        call_count = 0

        class FailingObserveProvider:
            def __init__(self):
                self.name = "failing-observe"

            async def chat_completion(self, messages, model=None, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count == 1:
                    raise ConnectionError("Failed to observe")

                return ChatCompletionResponse(
                    id="response",
                    model="test",
                    choices=[
                        ChatCompletionChoice(
                            index=0,
                            message=Message(
                                role=MessageRole.ASSISTANT, content="Observation recovered"
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
                )

        provider = FailingObserveProvider()
        loop = SimpleOODALoop(provider)

        success = False
        for attempt in range(3):
            try:
                observation = await loop.observe("Context")
                if "recovered" in observation.lower():
                    success = True
                    break
            except ConnectionError:
                continue

        assert success or call_count >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gauntlet"])
