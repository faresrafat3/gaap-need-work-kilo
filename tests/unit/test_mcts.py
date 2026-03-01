"""
Tests for MCTS Strategic Planning Module.

Tests cover:
    - MCTSNode: Node operations, UCT calculation, tree traversal
    - MCTSConfig: Configuration for different priorities/complexities
    - ValueOracle: Heuristic and LLM-based evaluation
    - MCTSStrategic: Full MCTS search loop
"""

from unittest.mock import MagicMock

import pytest

from gaap.core.types import TaskComplexity, TaskPriority
from gaap.layers.strategic.mcts_engine import (
    MCTSConfig,
    MCTSNode,
    MCTSPhase,
    MCTSStrategic,
    NodeType,
    ValueOracle,
    create_mcts,
    create_mcts_for_priority,
)


class TestMCTSConfig:
    """Tests for MCTSConfig."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MCTSConfig()
        assert config.iterations == 50
        assert config.exploration_weight == 1.414
        assert config.expansion_factor == 3
        assert config.max_depth == 5
        assert config.rollout_depth == 5
        assert config.parallel_rollouts is True
        assert config.min_iterations == 10

    def test_for_priority_critical(self) -> None:
        """Test config for CRITICAL priority (50 iterations)."""
        config = MCTSConfig.for_priority(TaskPriority.CRITICAL)
        assert config.iterations == 50

    def test_for_priority_high(self) -> None:
        """Test config for HIGH priority."""
        config = MCTSConfig.for_priority(TaskPriority.HIGH)
        assert config.iterations == 30

    def test_for_priority_normal(self) -> None:
        """Test config for NORMAL priority."""
        config = MCTSConfig.for_priority(TaskPriority.NORMAL)
        assert config.iterations == 20

    def test_for_priority_low(self) -> None:
        """Test config for LOW priority."""
        config = MCTSConfig.for_priority(TaskPriority.LOW)
        assert config.iterations == 10

    def test_for_priority_background(self) -> None:
        """Test config for BACKGROUND priority."""
        config = MCTSConfig.for_priority(TaskPriority.BACKGROUND)
        assert config.iterations == 10

    def test_for_complexity_architectural(self) -> None:
        """Test config for ARCHITECTURAL complexity."""
        config = MCTSConfig.for_complexity(TaskComplexity.ARCHITECTURAL)
        assert config.iterations == 50
        assert config.max_depth == 6

    def test_for_complexity_complex(self) -> None:
        """Test config for COMPLEX complexity."""
        config = MCTSConfig.for_complexity(TaskComplexity.COMPLEX)
        assert config.iterations == 30
        assert config.max_depth == 5

    def test_for_complexity_moderate(self) -> None:
        """Test config for MODERATE complexity."""
        config = MCTSConfig.for_complexity(TaskComplexity.MODERATE)
        assert config.iterations == 20
        assert config.max_depth == 4

    def test_for_complexity_simple(self) -> None:
        """Test config for SIMPLE complexity."""
        config = MCTSConfig.for_complexity(TaskComplexity.SIMPLE)
        assert config.iterations == 10
        assert config.max_depth == 3

    def test_for_complexity_trivial(self) -> None:
        """Test config for TRIVIAL complexity."""
        config = MCTSConfig.for_complexity(TaskComplexity.TRIVIAL)
        assert config.iterations == 10
        assert config.max_depth == 3


class TestMCTSNode:
    """Tests for MCTSNode."""

    def test_node_creation(self) -> None:
        """Test basic node creation."""
        node = MCTSNode(content="Test Decision")
        assert node.content == "Test Decision"
        assert node.visits == 0
        assert node.value == 0.0
        assert node.depth == 0
        assert node.node_type == NodeType.DECISION
        assert not node.expanded
        assert not node.pruned

    def test_node_with_parent(self) -> None:
        """Test node with parent."""
        parent = MCTSNode(content="Parent")
        child = MCTSNode(content="Child", parent=parent, depth=1)
        assert child.parent == parent
        assert child.depth == 1
        assert parent.children == []

    def test_is_leaf(self) -> None:
        """Test is_leaf property."""
        node = MCTSNode()
        assert node.is_leaf
        node.children.append(MCTSNode())
        assert not node.is_leaf

    def test_is_terminal(self) -> None:
        """Test is_terminal property."""
        node = MCTSNode()
        assert not node.is_terminal
        node.node_type = NodeType.TERMINAL
        assert node.is_terminal

    def test_is_root(self) -> None:
        """Test is_root property."""
        node = MCTSNode()
        assert node.is_root
        child = MCTSNode(parent=node)
        assert not child.is_root

    def test_average_value_no_visits(self) -> None:
        """Test average_value with no visits."""
        node = MCTSNode()
        assert node.average_value == 0.0

    def test_average_value_with_visits(self) -> None:
        """Test average_value with visits."""
        node = MCTSNode(visits=10, value=7.5)
        assert node.average_value == 0.75

    def test_uct_score_no_visits(self) -> None:
        """Test UCT score with no visits (should be infinity)."""
        node = MCTSNode()
        assert node.uct_score() == float("inf")

    def test_uct_score_with_visits_no_parent(self) -> None:
        """Test UCT score with visits but no parent."""
        node = MCTSNode(visits=10, value=5.0)
        assert node.uct_score() == 0.5

    def test_uct_score_full_calculation(self) -> None:
        """Test full UCT score calculation."""
        parent = MCTSNode(visits=100)
        child = MCTSNode(parent=parent, visits=10, value=7.0)
        parent.children.append(child)

        import math

        expected = 0.7 + 1.414 * math.sqrt(math.log(100) / 10)
        assert abs(child.uct_score() - expected) < 0.001

    def test_add_child(self) -> None:
        """Test adding a child node."""
        parent = MCTSNode(content="Parent")
        child = parent.add_child("Child Decision")

        assert len(parent.children) == 1
        assert child.content == "Child Decision"
        assert child.parent == parent
        assert child.depth == 1

    def test_update(self) -> None:
        """Test updating node statistics."""
        node = MCTSNode()
        node.update(0.8)

        assert node.visits == 1
        assert node.value == 0.8

        node.update(0.6)
        assert node.visits == 2
        assert node.value == 1.4

    def test_get_path(self) -> None:
        """Test getting path from root."""
        root = MCTSNode(content="Root")
        child1 = root.add_child("Child1")
        child2 = child1.add_child("Child2")

        path = child2.get_path()
        assert len(path) == 3
        assert path[0].content == "Root"
        assert path[1].content == "Child1"
        assert path[2].content == "Child2"

    def test_get_best_child(self) -> None:
        """Test getting best child by UCT."""
        parent = MCTSNode(visits=100)
        child1 = parent.add_child("A")
        child2 = parent.add_child("B")

        child1.visits = 10
        child1.value = 8.0
        child2.visits = 10
        child2.value = 5.0

        best = parent.get_best_child()
        assert best == child1

    def test_get_best_child_no_children(self) -> None:
        """Test getting best child with no children."""
        node = MCTSNode()
        assert node.get_best_child() is None

    def test_get_most_visited_child(self) -> None:
        """Test getting most visited child."""
        parent = MCTSNode()
        child1 = parent.add_child("A")
        child2 = parent.add_child("B")

        child1.visits = 5
        child2.visits = 15

        most_visited = parent.get_most_visited_child()
        assert most_visited == child2

    def test_get_most_visited_child_no_children(self) -> None:
        """Test getting most visited child with no children."""
        node = MCTSNode()
        assert node.get_most_visited_child() is None

    def test_prune(self) -> None:
        """Test pruning a node and its descendants."""
        root = MCTSNode()
        child1 = root.add_child("A")
        grandchild = child1.add_child("AA")
        child2 = root.add_child("B")

        child1.prune()

        assert child1.pruned
        assert grandchild.pruned
        assert not child2.pruned
        assert not root.pruned

    def test_to_dict(self) -> None:
        """Test converting node to dictionary."""
        node = MCTSNode(content="Test", visits=5, value=3.5, depth=2)
        d = node.to_dict()

        assert d["content"] == "Test"
        assert d["visits"] == 5
        assert d["value"] == 3.5
        assert d["avg_value"] == 0.7
        assert d["depth"] == 2
        assert d["node_type"] == "decision"


class TestValueOracle:
    """Tests for ValueOracle."""

    def test_oracle_creation(self) -> None:
        """Test oracle creation."""
        oracle = ValueOracle()
        assert oracle.provider is None
        assert oracle.model == "llama-3.3-70b-versatile"
        assert oracle._cache == {}

    def test_oracle_with_provider(self) -> None:
        """Test oracle with provider."""
        mock_provider = MagicMock()
        oracle = ValueOracle(provider=mock_provider, model="custom-model")
        assert oracle.provider == mock_provider
        assert oracle.model == "custom-model"

    def test_heuristic_evaluate_microservices_scalability(self) -> None:
        """Test heuristic evaluation for microservices with scalability."""
        oracle = ValueOracle()
        node = MCTSNode(content="microservices architecture")
        context = {"scalability_required": True}

        value = oracle._heuristic_evaluate(node, context, None)
        assert value > 0.5

    def test_heuristic_evaluate_microservices_budget(self) -> None:
        """Test heuristic evaluation for microservices with budget constraints."""
        oracle = ValueOracle()
        node = MCTSNode(content="microservices architecture")
        context = {"budget_conscious": True}

        value = oracle._heuristic_evaluate(node, context, None)
        assert value < 0.5

    def test_heuristic_evaluate_monolith_budget(self) -> None:
        """Test heuristic evaluation for monolith with budget."""
        oracle = ValueOracle()
        node = MCTSNode(content="monolith architecture")
        context = {"budget_conscious": True}

        value = oracle._heuristic_evaluate(node, context, None)
        assert value > 0.5

    def test_heuristic_evaluate_kubernetes_expertise(self) -> None:
        """Test heuristic evaluation for Kubernetes with expertise."""
        oracle = ValueOracle()
        node = MCTSNode(content="kubernetes deployment")
        context = {"team_ops_expertise": True}

        value = oracle._heuristic_evaluate(node, context, None)
        assert value > 0.5

    def test_heuristic_evaluate_kubernetes_no_expertise(self) -> None:
        """Test heuristic evaluation for Kubernetes without expertise."""
        oracle = ValueOracle()
        node = MCTSNode(content="kubernetes deployment")
        context = {"team_ops_expertise": False}

        value = oracle._heuristic_evaluate(node, context, None)
        assert value < 0.5

    def test_heuristic_evaluate_depth_penalty(self) -> None:
        """Test heuristic evaluation with depth penalty."""
        oracle = ValueOracle()
        shallow_node = MCTSNode(content="test", depth=2)
        deep_node = MCTSNode(content="test", depth=5)

        shallow_value = oracle._heuristic_evaluate(shallow_node, {}, None)
        deep_value = oracle._heuristic_evaluate(deep_node, {}, None)

        assert shallow_value > deep_value

    def test_parse_value_valid_float(self) -> None:
        """Test parsing valid float value."""
        oracle = ValueOracle()
        assert oracle._parse_value("0.85") == 0.85
        assert oracle._parse_value("0.5") == 0.5

    def test_parse_value_with_text(self) -> None:
        """Test parsing value from text with number."""
        oracle = ValueOracle()
        assert oracle._parse_value("The probability is 0.75") == 0.75

    def test_parse_value_clamp_high(self) -> None:
        """Test parsing value that gets clamped (too high)."""
        oracle = ValueOracle()
        assert oracle._parse_value("1.5") == 1.0

    def test_parse_value_clamp_low(self) -> None:
        """Test parsing value that gets clamped (too low)."""
        oracle = ValueOracle()
        assert oracle._parse_value("-0.5") == 0.0

    def test_parse_value_no_number(self) -> None:
        """Test parsing with no number returns default."""
        oracle = ValueOracle()
        assert oracle._parse_value("no number here") == 0.5

    def test_clear_cache(self) -> None:
        """Test clearing the cache."""
        oracle = ValueOracle()
        oracle._cache["test"] = 0.5
        oracle.clear_cache()
        assert oracle._cache == {}

    @pytest.mark.asyncio
    async def test_evaluate_caches_result(self) -> None:
        """Test that evaluation result is cached."""
        oracle = ValueOracle()
        node = MCTSNode(id="test123", content="test")

        value1 = await oracle.evaluate(node, {})
        value2 = await oracle.evaluate(node, {})

        assert value1 == value2
        assert len(oracle._cache) == 1


class TestMCTSStrategic:
    """Tests for MCTSStrategic."""

    def test_mcts_creation(self) -> None:
        """Test MCTS creation."""
        mcts = MCTSStrategic()
        assert mcts.config is not None
        assert mcts.oracle is not None
        assert mcts._root is None

    def test_mcts_with_custom_config(self) -> None:
        """Test MCTS with custom config."""
        config = MCTSConfig(iterations=100, exploration_weight=2.0)
        mcts = MCTSStrategic(config=config)
        assert mcts.config.iterations == 100
        assert mcts.config.exploration_weight == 2.0

    @pytest.mark.asyncio
    async def test_search_creates_root(self) -> None:
        """Test that search creates a root node."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=5))
        intent = MagicMock()

        best, stats = await mcts.search(intent, root_content="Test Root")

        assert mcts._root is not None
        assert mcts._root.content == "Test Root"

    @pytest.mark.asyncio
    async def test_search_returns_stats(self) -> None:
        """Test that search returns statistics."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=5))
        intent = MagicMock()

        best, stats = await mcts.search(intent)

        assert "iterations" in stats
        assert "nodes_created" in stats
        assert "rollouts" in stats
        assert stats["iterations"] == 5

    @pytest.mark.asyncio
    async def test_search_expands_nodes(self) -> None:
        """Test that search expands nodes."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=10))
        intent = MagicMock()

        await mcts.search(intent)

        assert mcts._root is not None
        assert len(mcts._root.children) > 0 or mcts._root.visits > 0

    @pytest.mark.asyncio
    async def test_search_updates_visits(self) -> None:
        """Test that search updates visit counts."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=20))
        intent = MagicMock()

        await mcts.search(intent)

        assert mcts._root is not None
        assert mcts._root.visits == 20

    def test_select_returns_leaf(self) -> None:
        """Test that select returns a leaf node."""
        mcts = MCTSStrategic()

        root = MCTSNode(content="Root", visits=10)
        child = root.add_child("Child")
        child.visits = 5

        selected = mcts._select(root)
        assert selected.is_leaf or selected.is_terminal

    def test_backpropagate_updates_chain(self) -> None:
        """Test backpropagation updates all nodes in chain."""
        mcts = MCTSStrategic()

        root = MCTSNode(content="Root")
        child1 = root.add_child("C1")
        child2 = child1.add_child("C2")

        mcts._backpropagate(child2, 0.8)

        assert root.visits == 1
        assert child1.visits == 1
        assert child2.visits == 1
        assert root.value == 0.8
        assert child1.value == 0.8
        assert child2.value == 0.8

    def test_get_best_path_empty(self) -> None:
        """Test get_best_path with no tree."""
        mcts = MCTSStrategic()
        assert mcts.get_best_path() == []

    @pytest.mark.asyncio
    async def test_get_best_path_returns_path(self) -> None:
        """Test get_best_path returns a valid path."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=10))
        intent = MagicMock()

        await mcts.search(intent)
        path = mcts.get_best_path()

        assert len(path) >= 1
        assert path[0] == mcts._root

    def test_get_tree_stats_no_tree(self) -> None:
        """Test get_tree_stats with no tree."""
        mcts = MCTSStrategic()
        stats = mcts.get_tree_stats()
        assert "error" in stats

    @pytest.mark.asyncio
    async def test_get_tree_stats_with_tree(self) -> None:
        """Test get_tree_stats with tree."""
        mcts = MCTSStrategic(config=MCTSConfig(iterations=5))
        intent = MagicMock()

        await mcts.search(intent)
        stats = mcts.get_tree_stats()

        assert "root_visits" in stats
        assert "children_count" in stats
        assert "tree_depth" in stats

    def test_get_default_actions_level_0(self) -> None:
        """Test default actions for level 0 (architecture)."""
        mcts = MCTSStrategic()
        actions = mcts._get_default_actions(0)

        assert "microservices" in actions
        assert "monolith" in actions

    def test_get_default_actions_level_1(self) -> None:
        """Test default actions for level 1 (data strategy)."""
        mcts = MCTSStrategic()
        actions = mcts._get_default_actions(1)

        assert "single_database" in actions
        assert "cqrs" in actions

    def test_get_default_actions_level_2(self) -> None:
        """Test default actions for level 2 (communication)."""
        mcts = MCTSStrategic()
        actions = mcts._get_default_actions(2)

        assert "rest" in actions
        assert "grpc" in actions

    def test_get_default_actions_level_out_of_range(self) -> None:
        """Test default actions for out-of-range level."""
        mcts = MCTSStrategic()
        actions = mcts._get_default_actions(100)

        assert len(actions) == 1
        assert "explore_alternative" in actions


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mcts_default(self) -> None:
        """Test create_mcts with defaults."""
        mcts = create_mcts()
        assert mcts.config.iterations == 50
        assert mcts.config.exploration_weight == 1.414

    def test_create_mcts_custom(self) -> None:
        """Test create_mcts with custom parameters."""
        mcts = create_mcts(iterations=100, exploration_weight=2.0)
        assert mcts.config.iterations == 100
        assert mcts.config.exploration_weight == 2.0

    def test_create_mcts_for_priority_critical(self) -> None:
        """Test create_mcts_for_priority with CRITICAL."""
        mcts = create_mcts_for_priority(TaskPriority.CRITICAL)
        assert mcts.config.iterations == 50

    def test_create_mcts_for_priority_normal(self) -> None:
        """Test create_mcts_for_priority with NORMAL."""
        mcts = create_mcts_for_priority(TaskPriority.NORMAL)
        assert mcts.config.iterations == 20


class TestNodeTypeEnum:
    """Tests for NodeType enum."""

    def test_node_type_values(self) -> None:
        """Test NodeType enum values."""
        assert NodeType.DECISION.value == "decision"
        assert NodeType.CHANCE.value == "chance"
        assert NodeType.TERMINAL.value == "terminal"


class TestMCTSPhaseEnum:
    """Tests for MCTSPhase enum."""

    def test_mcts_phase_values(self) -> None:
        """Test MCTSPhase enum values."""
        assert MCTSPhase.SELECTION.value == "selection"
        assert MCTSPhase.EXPANSION.value == "expansion"
        assert MCTSPhase.SIMULATION.value == "simulation"
        assert MCTSPhase.BACKPROPAGATION.value == "backpropagation"


class TestIntegration:
    """Integration tests for MCTS."""

    @pytest.mark.asyncio
    async def test_full_mcts_loop(self) -> None:
        """Test full MCTS loop execution."""
        config = MCTSConfig(iterations=30, max_depth=4)
        mcts = MCTSStrategic(config=config)

        intent = MagicMock()
        context = {"scalability_required": True, "budget_conscious": False}

        best_node, stats = await mcts.search(intent, context, "System Design")

        assert stats["iterations"] == 30
        assert stats["rollouts"] == 30
        assert mcts._root.visits == 30

    @pytest.mark.asyncio
    async def test_mcts_with_different_iterations(self) -> None:
        """Test MCTS with different iteration counts."""
        for iterations in [5, 10, 20, 50]:
            mcts = MCTSStrategic(config=MCTSConfig(iterations=iterations))
            intent = MagicMock()

            _, stats = await mcts.search(intent)

            assert stats["iterations"] == iterations

    @pytest.mark.asyncio
    async def test_mcts_finds_better_paths(self) -> None:
        """Test that MCTS finds good paths through exploration."""
        config = MCTSConfig(iterations=100, exploration_weight=1.5)
        mcts = MCTSStrategic(config=config)

        context = {"scalability_required": True}
        intent = MagicMock()

        await mcts.search(intent, context, "Scalable System")

        path = mcts.get_best_path()
        assert len(path) >= 1

        for node in path:
            assert node.visits >= 0
