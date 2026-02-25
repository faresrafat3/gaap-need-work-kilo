"""
Tests for Graph of Thoughts (GoT) Strategic Engine
===================================================

Tests:
- ThoughtNode creation and relationships
- GoTStrategic exploration
- Generate/Aggregate/Refine operations
- Evidence-based scoring
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from gaap.layers.got_strategic import (
    ThoughtNode,
    ThoughtStatus,
    GoTGraph,
    GoTStrategic,
    create_got_strategic,
)
from gaap.layers.layer1_strategic import (
    ArchitectureDecision,
    ArchitectureParadigm,
    ArchitectureSpec,
    DataStrategy,
    CommunicationPattern,
)
from gaap.layers.layer0_interface import StructuredIntent, IntentType, ImplicitRequirements
from datetime import datetime


class TestThoughtNode:
    """Tests for ThoughtNode class."""

    def test_node_creation(self):
        decision = ArchitectureDecision(
            aspect="paradigm",
            choice="microservices",
            reasoning="Scalability requirements",
            trade_offs=["Complexity"],
            confidence=0.8,
        )

        node = ThoughtNode(
            id="test-1",
            content=decision,
        )

        assert node.id == "test-1"
        assert node.score == 0.0
        assert node.valid is True
        assert node.status == ThoughtStatus.DRAFT
        assert len(node.parents) == 0
        assert len(node.children) == 0

    def test_node_with_parents(self):
        decision1 = ArchitectureDecision(
            aspect="paradigm",
            choice="microservices",
            reasoning="Scalability",
            trade_offs=[],
            confidence=0.8,
        )
        decision2 = ArchitectureDecision(
            aspect="aggregated",
            choice="hybrid",
            reasoning="Best of both",
            trade_offs=[],
            confidence=0.9,
        )

        parent = ThoughtNode(id="parent", content=decision1)
        child = ThoughtNode(id="child", content=decision2, parents=[parent])

        assert len(child.parents) == 1
        assert child.parents[0] == parent

    def test_add_child_relationship(self):
        parent = ThoughtNode(
            id="parent",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
            ),
        )
        child = ThoughtNode(
            id="child",
            content=ArchitectureDecision(
                aspect="test", choice="b", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        parent.add_child(child)

        assert child in parent.children
        assert parent in child.parents

    def test_get_ancestors(self):
        root = ThoughtNode(
            id="root",
            content=ArchitectureDecision(
                aspect="test", choice="root", reasoning="", trade_offs=[], confidence=0.5
            ),
        )
        middle = ThoughtNode(
            id="middle",
            content=ArchitectureDecision(
                aspect="test", choice="middle", reasoning="", trade_offs=[], confidence=0.5
            ),
        )
        leaf = ThoughtNode(
            id="leaf",
            content=ArchitectureDecision(
                aspect="test", choice="leaf", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        root.add_child(middle)
        middle.add_child(leaf)

        ancestors = leaf.get_ancestors()

        assert root in ancestors
        assert middle in ancestors

    def test_evidence_storage(self):
        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
            ),
            evidence=["Microservices scale better", "Industry benchmark shows 3x improvement"],
        )

        assert len(node.evidence) == 2
        assert "Microservices" in node.evidence[0]

    def test_status_transitions(self):
        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        assert node.status == ThoughtStatus.DRAFT

        node.status = ThoughtStatus.EVALUATED
        assert node.status == ThoughtStatus.EVALUATED

        node.status = ThoughtStatus.AGGREGATED
        assert node.status == ThoughtStatus.AGGREGATED

    def test_to_dict(self):
        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="paradigm",
                choice="microservices",
                reasoning="test",
                trade_offs=[],
                confidence=0.8,
            ),
            score=0.75,
            generation=2,
        )

        d = node.to_dict()

        assert d["id"] == "test"
        assert d["score"] == 0.75
        assert d["generation"] == 2


class TestGoTGraph:
    """Tests for GoTGraph class."""

    def test_graph_creation(self):
        graph = GoTGraph()

        assert len(graph.nodes) == 0
        assert graph.root is None

    def test_add_node(self):
        graph = GoTGraph()
        node = ThoughtNode(
            id="test-1",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        result = graph.add_node(node)

        assert result is True
        assert len(graph.nodes) == 1

    def test_max_nodes_limit(self):
        graph = GoTGraph(max_nodes=2)

        for i in range(3):
            node = ThoughtNode(
                id=f"node-{i}",
                content=ArchitectureDecision(
                    aspect="test", choice=str(i), reasoning="", trade_offs=[], confidence=0.5
                ),
            )
            graph.add_node(node)

        assert len(graph.nodes) == 2

    def test_get_best_node(self):
        graph = GoTGraph()

        for i in range(5):
            node = ThoughtNode(
                id=f"node-{i}",
                content=ArchitectureDecision(
                    aspect="test", choice=str(i), reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.2 * (i + 1),
            )
            graph.add_node(node)

        best = graph.get_best_node()

        assert best is not None
        assert best.score == 1.0

    def test_get_nodes_by_generation(self):
        graph = GoTGraph()

        for gen in range(3):
            for i in range(2):
                node = ThoughtNode(
                    id=f"node-{gen}-{i}",
                    content=ArchitectureDecision(
                        aspect="test",
                        choice=f"{gen}-{i}",
                        reasoning="",
                        trade_offs=[],
                        confidence=0.5,
                    ),
                    generation=gen,
                )
                graph.add_node(node)

        gen_1_nodes = graph.get_nodes_by_generation(1)

        assert len(gen_1_nodes) == 2
        assert all(n.generation == 1 for n in gen_1_nodes)

    def test_prune_low_score_nodes(self):
        graph = GoTGraph()

        for i in range(5):
            node = ThoughtNode(
                id=f"node-{i}",
                content=ArchitectureDecision(
                    aspect="test", choice=str(i), reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.1 * i,
            )
            graph.add_node(node)

        pruned = graph.prune_low_score_nodes(threshold=0.25)

        assert pruned == 3


class TestGoTStrategic:
    """Tests for GoTStrategic engine."""

    @pytest.fixture
    def intent(self):
        return StructuredIntent(
            request_id="test-req-1",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a scalable web application"],
            implicit_requirements=ImplicitRequirements(
                scalability="high",
                performance="real_time",
                budget="standard",
            ),
            metadata={"original_text": "Build a scalable web application"},
        )

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.chat_completion = AsyncMock()
        provider.default_model = "test-model"
        return provider

    def test_initialization(self):
        got = GoTStrategic()

        assert got.max_nodes == 50
        assert got.max_generations == 4

    def test_initialization_with_provider(self, mock_provider):
        got = GoTStrategic(provider=mock_provider)

        assert got.provider == mock_provider

    def test_fallback_generate_diverse(self, intent):
        got = GoTStrategic()

        nodes = got._generate_fallback(intent, 3)

        assert len(nodes) == 3

    def test_fallback_score_node(self, intent):
        got = GoTStrategic()

        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="test", choice="microservices", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        score = got._score_fallback(node, intent)

        assert 0.0 <= score <= 1.0

    def test_fallback_aggregate(self):
        got = GoTStrategic()

        nodes = [
            ThoughtNode(
                id="n1",
                content=ArchitectureDecision(
                    aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.8,
            ),
            ThoughtNode(
                id="n2",
                content=ArchitectureDecision(
                    aspect="test", choice="b", reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.6,
            ),
        ]

        aggregated = got._aggregate_best_parts(nodes)

        assert aggregated is not None
        assert aggregated.status == ThoughtStatus.AGGREGATED
        assert len(aggregated.parents) == 2

    def test_fallback_refine(self):
        got = GoTStrategic()

        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="Original", trade_offs=[], confidence=0.5
            ),
            generation=1,
        )

        refined = got._refine_fallback(node, "Test feedback")

        assert refined is not None
        assert refined.status == ThoughtStatus.REFINED
        assert refined.generation == 2
        assert node in refined.parents

    @pytest.mark.asyncio
    async def test_explore_without_provider(self, intent):
        got = GoTStrategic(max_nodes=20, max_generations=2)

        spec = await got.explore(intent, {})

        assert spec is not None
        assert isinstance(spec, ArchitectureSpec)

    @pytest.mark.asyncio
    async def test_explore_with_wisdom_and_pitfalls(self, intent):
        got = GoTStrategic(max_nodes=20, max_generations=2)

        mock_wisdom = MagicMock()
        mock_wisdom.principle = "Always validate input"

        mock_failure = MagicMock()
        mock_failure.error = "Missing validation caused crash"

        mock_correction = MagicMock()
        mock_correction.solution = "Add input validation"

        spec = await got.explore(
            intent,
            {},
            wisdom=[mock_wisdom],
            pitfalls=[(mock_failure, [mock_correction])],
        )

        assert spec is not None


class TestGoTStrategicWithProvider:
    """Tests for GoTStrategic with LLM provider."""

    @pytest.fixture
    def intent(self):
        return StructuredIntent(
            request_id="test-req-2",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a real-time chat application"],
            implicit_requirements=ImplicitRequirements(
                scalability="high",
                performance="real_time",
            ),
            metadata={"original_text": "Build a real-time chat application"},
        )

    @pytest.fixture
    def mock_provider(self):
        provider = MagicMock()
        provider.default_model = "test-model"
        return provider

    @pytest.mark.asyncio
    async def test_llm_generate(self, intent, mock_provider):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        [
            {
                "paradigm": "microservices",
                "data_strategy": "cqrs",
                "communication": "grpc",
                "reasoning": "Real-time requirements",
                "trade_offs": ["Complexity"]
            }
        ]
        """
        mock_provider.chat_completion = AsyncMock(return_value=mock_response)

        got = GoTStrategic(provider=mock_provider)
        nodes = await got.generate(intent, n=1)

        assert len(nodes) >= 1

    @pytest.mark.asyncio
    async def test_llm_aggregate(self, intent, mock_provider):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """
        {
            "paradigm": "event_driven",
            "data_strategy": "event_sourcing",
            "communication": "event_bus",
            "reasoning": "Combined best aspects",
            "trade_offs": ["Learning curve"],
            "combined_from": ["async from n1", "scale from n2"]
        }
        """
        mock_provider.chat_completion = AsyncMock(return_value=mock_response)

        nodes = [
            ThoughtNode(
                id="n1",
                content=ArchitectureDecision(
                    aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.8,
            ),
            ThoughtNode(
                id="n2",
                content=ArchitectureDecision(
                    aspect="test", choice="b", reasoning="", trade_offs=[], confidence=0.5
                ),
                score=0.7,
            ),
        ]

        got = GoTStrategic(provider=mock_provider)
        aggregated = await got.aggregate(nodes, intent)

        assert aggregated is not None
        assert aggregated.status == ThoughtStatus.AGGREGATED

    @pytest.mark.asyncio
    async def test_llm_score_node(self, intent, mock_provider):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"score": 85, "evidence": ["Strong alignment with requirements", "Industry proven pattern"], "reasoning": "Good fit"}'
        mock_provider.chat_completion = AsyncMock(return_value=mock_response)

        got = GoTStrategic(provider=mock_provider)

        node = ThoughtNode(
            id="test",
            content=ArchitectureDecision(
                aspect="test", choice="a", reasoning="", trade_offs=[], confidence=0.5
            ),
        )

        score = await got.score_node(node, intent)

        assert 0.0 <= score <= 1.0
        assert len(node.evidence) >= 1


class TestFactoryFunction:
    """Tests for factory function."""

    def test_create_got_strategic(self):
        got = create_got_strategic()

        assert isinstance(got, GoTStrategic)

    def test_create_got_strategic_with_provider(self):
        provider = MagicMock()
        got = create_got_strategic(provider=provider)

        assert got.provider == provider
