"""
Enhanced tests for Layer 1 - Strategic Layer
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gaap.layers.layer1_strategic import (
    ArchitectureParadigm,
    DataStrategy,
    CommunicationPattern,
    ArchitectureDecision,
    ArchitectureSpec,
    ToTNode,
    ToTStrategic,
    MADArchitecturePanel,
    StrategicPlanner,
    Layer1Strategic,
    EpistemicCheckResult,
)


class TestArchitectureEnums:
    """Tests for architecture enums"""

    def test_architecture_paradigm_values(self):
        """Test all architecture paradigm values"""
        assert ArchitectureParadigm.MONOLITH.value == "monolith"
        assert ArchitectureParadigm.MODULAR_MONOLITH.value == "modular_monolith"
        assert ArchitectureParadigm.MICROSERVICES.value == "microservices"
        assert ArchitectureParadigm.SERVERLESS.value == "serverless"
        assert ArchitectureParadigm.EVENT_DRIVEN.value == "event_driven"
        assert ArchitectureParadigm.LAYERED.value == "layered"
        assert ArchitectureParadigm.HEXAGONAL.value == "hexagonal"

    def test_data_strategy_values(self):
        """Test all data strategy values"""
        assert DataStrategy.SINGLE_DB.value == "single_database"
        assert DataStrategy.POLYGLOT.value == "polyglot"
        assert DataStrategy.CQRS.value == "cqrs"
        assert DataStrategy.EVENT_SOURCING.value == "event_sourcing"

    def test_communication_pattern_values(self):
        """Test all communication pattern values"""
        assert CommunicationPattern.REST.value == "rest"
        assert CommunicationPattern.GRAPHQL.value == "graphql"
        assert CommunicationPattern.GRPC.value == "grpc"
        assert CommunicationPattern.MESSAGE_QUEUE.value == "message_queue"
        assert CommunicationPattern.EVENT_BUS.value == "event_bus"


class TestArchitectureDecision:
    """Tests for ArchitectureDecision dataclass"""

    def test_create_decision(self):
        """Test creating an architecture decision"""
        decision = ArchitectureDecision(
            aspect="paradigm",
            choice="microservices",
            reasoning="Scalability requirements",
            trade_offs=["Complexity", "Cost"],
            confidence=0.85,
        )
        assert decision.aspect == "paradigm"
        assert decision.choice == "microservices"
        assert decision.confidence == 0.85

    def test_decision_confidence_bounds(self):
        """Test confidence bounds"""
        decision = ArchitectureDecision(
            aspect="test",
            choice="test",
            reasoning="test",
            trade_offs=[],
            confidence=0.5,
        )
        assert 0 <= decision.confidence <= 1.0


class TestArchitectureSpec:
    """Tests for ArchitectureSpec dataclass"""

    def test_create_spec(self):
        """Test creating an architecture spec"""
        spec = ArchitectureSpec(spec_id="spec-001")
        assert spec.spec_id == "spec-001"
        assert spec.paradigm == ArchitectureParadigm.MODULAR_MONOLITH
        assert spec.data_strategy == DataStrategy.SINGLE_DB
        assert spec.communication == CommunicationPattern.REST

    def test_spec_to_dict(self):
        """Test converting spec to dictionary"""
        spec = ArchitectureSpec(spec_id="spec-001")
        spec.paradigm = ArchitectureParadigm.MICROSERVICES
        spec.data_strategy = DataStrategy.CQRS
        spec.communication = CommunicationPattern.REST

        spec.decisions.append(
            ArchitectureDecision(
                aspect="paradigm",
                choice="microservices",
                reasoning="test",
                trade_offs=[],
                confidence=0.8,
            )
        )

        result = spec.to_dict()
        assert result["spec_id"] == "spec-001"
        assert result["paradigm"] == "microservices"
        assert result["data_strategy"] == "cqrs"
        assert len(result["decisions"]) == 1

    def test_spec_default_values(self):
        """Test spec default values"""
        spec = ArchitectureSpec(spec_id="test")
        assert spec.explored_paths == 0
        assert spec.selected_path_score == 0.0
        assert spec.debate_rounds == 0
        assert spec.consensus_reached is False
        assert isinstance(spec.components, list)
        assert isinstance(spec.tech_stack, dict)
        assert isinstance(spec.decisions, list)
        assert isinstance(spec.risks, list)


class TestToTNode:
    """Tests for ToTNode"""

    def test_create_node(self):
        """Test creating a ToT node"""
        node = ToTNode(id="node-1", level=0, content="Test content", score=0.8)
        assert node.id == "node-1"
        assert node.level == 0
        assert node.content == "Test content"
        assert node.score == 0.8
        assert node.explored is False
        assert node.pruned is False

    def test_node_hierarchy(self):
        """Test node parent-child relationships"""
        parent = ToTNode(id="parent", level=0, content="Parent")
        child = ToTNode(id="child", level=1, content="Child", parent=parent)
        parent.children.append(child)

        assert child.parent == parent
        assert len(parent.children) == 1


class TestToTStrategic:
    """Tests for Tree of Thoughts Strategic"""

    def test_tot_initialization(self):
        """Test ToT initialization"""
        tot = ToTStrategic(max_depth=3, branching_factor=2)
        assert tot.max_depth == 3
        assert tot.branching_factor == 2
        assert tot._explored_nodes == 0

    @pytest.mark.asyncio
    async def test_tot_explore_research_intent(self):
        """Test ToT exploration with research intent"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.RESEARCH
        intent.explicit_goals = ["Research AI trends"]
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.scalability = None
        intent.implicit_requirements.budget = None
        intent.implicit_requirements.performance = None
        intent.implicit_requirements.security = None
        intent.implicit_requirements.timeline = None

        tot = ToTStrategic(max_depth=2, branching_factor=2)
        spec, root = await tot.explore(intent)

        assert spec is not None
        assert root is not None
        assert "research_methodology" in spec.metadata

    @pytest.mark.asyncio
    async def test_tot_explore_debugging_intent(self):
        """Test ToT exploration with debugging intent"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.DEBUGGING
        intent.explicit_goals = ["Fix authentication bug"]
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.scalability = None
        intent.implicit_requirements.budget = None
        intent.implicit_requirements.performance = None
        intent.implicit_requirements.security = None
        intent.implicit_requirements.timeline = None

        tot = ToTStrategic(max_depth=2, branching_factor=2)
        spec, root = await tot.explore(intent)

        assert spec is not None
        assert "diagnostic_approach" in spec.metadata

    @pytest.mark.asyncio
    async def test_tot_explore_standard_intent(self):
        """Test ToT exploration with standard intent"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.CODE_GENERATION
        intent.explicit_goals = ["Build a web API"]
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.scalability = True
        intent.implicit_requirements.budget = "budget_conscious"
        intent.implicit_requirements.performance = None
        intent.implicit_requirements.security = None
        intent.implicit_requirements.timeline = None

        tot = ToTStrategic(max_depth=2, branching_factor=2)
        spec, root = await tot.explore(intent)

        assert spec is not None

    def test_tot_generate_options_research(self):
        """Test option generation for research intent"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.RESEARCH
        intent.explicit_goals = []
        intent.implicit_requirements = MagicMock()

        tot = ToTStrategic()
        options = tot._generate_options(0, intent)
        assert "systematic_review" in options

    def test_tot_generate_options_debugging(self):
        """Test option generation for debugging intent"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.DEBUGGING
        intent.explicit_goals = []
        intent.implicit_requirements = MagicMock()

        tot = ToTStrategic()
        options = tot._generate_options(0, intent)
        assert "reproduce_first" in options

    def test_tot_evaluate_option(self):
        """Test option evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.CODE_GENERATION
        intent.explicit_goals = ["Build API", "Build Cache", "Build Queue", "Build Worker"]
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.scalability = True
        intent.implicit_requirements.budget = "budget_conscious"
        intent.implicit_requirements.performance = None

        tot = ToTStrategic()
        score = tot._evaluate_option("microservices", 0, intent)
        assert score > 0.5

    def test_tot_select_best_path(self):
        """Test best path selection"""
        root = ToTNode(id="root", level=0, content="root", score=0.5)
        child1 = ToTNode(id="c1", level=1, content="child1", score=0.3)
        child2 = ToTNode(id="c2", level=1, content="child2", score=0.8)
        root.children = [child1, child2]

        tot = ToTStrategic()
        path = tot._select_best_path(root)

        assert len(path) == 2
        assert path[-1].score == 0.8


class TestMADArchitecturePanel:
    """Tests for MAD Architecture Panel"""

    def test_mad_initialization(self):
        """Test MAD panel initialization"""
        panel = MADArchitecturePanel(max_rounds=3, consensus_threshold=0.8)
        assert panel.max_rounds == 3
        assert panel.consensus_threshold == 0.8
        assert panel._llm_failures == 0

    @pytest.mark.asyncio
    async def test_mad_debate_fallback(self):
        """Test MAD debate with fallback (no LLM)"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.CODE_GENERATION
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.budget = None
        intent.implicit_requirements.timeline = None
        intent.implicit_requirements.security = None

        panel = MADArchitecturePanel(max_rounds=2)
        result_spec, consensus = await panel.debate(spec, intent)

        assert result_spec is not None
        assert isinstance(consensus, bool)

    @pytest.mark.asyncio
    async def test_mad_evaluate_all_fallback(self):
        """Test fallback evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.CODE_GENERATION
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.budget = None
        intent.implicit_requirements.timeline = None
        intent.implicit_requirements.security = None

        panel = MADArchitecturePanel()
        evaluations = await panel._evaluate_all(spec, intent, 0)

        assert len(evaluations) == 4
        assert all("score" in e for e in evaluations)

    def test_mad_scalability_eval(self):
        """Test scalability evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        spec.paradigm = ArchitectureParadigm.MICROSERVICES

        intent = MagicMock(spec=StructuredIntent)
        intent.implicit_requirements = MagicMock()

        panel = MADArchitecturePanel()
        result = panel._scalability_eval(spec, intent)

        assert "score" in result
        assert result["score"] > 0.5

    def test_mad_pragmatism_eval(self):
        """Test pragmatism evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        spec.paradigm = ArchitectureParadigm.MICROSERVICES

        intent = MagicMock(spec=StructuredIntent)
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.budget = "budget_conscious"
        intent.implicit_requirements.timeline = None

        panel = MADArchitecturePanel()
        result = panel._pragmatism_eval(spec, intent)

        assert "score" in result
        assert "issues" in result

    def test_mad_cost_eval(self):
        """Test cost evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        spec.paradigm = ArchitectureParadigm.MONOLITH

        intent = MagicMock(spec=StructuredIntent)
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.budget = None

        panel = MADArchitecturePanel()
        result = panel._cost_eval(spec, intent)

        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_mad_robustness_eval(self):
        """Test robustness evaluation"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        spec.communication = CommunicationPattern.MESSAGE_QUEUE

        intent = MagicMock(spec=StructuredIntent)
        intent.implicit_requirements = MagicMock()
        intent.implicit_requirements.security = True

        panel = MADArchitecturePanel()
        result = panel._robustness_eval(spec, intent)

        assert "score" in result

    def test_mad_apply_critiques(self):
        """Test applying critiques to spec"""
        evaluations = [
            {
                "critic": "scalability",
                "score": 0.6,
                "suggestions": ["Fix issue 1"],
                "issues": ["Issue 1"],
            },
            {"critic": "pragmatism", "score": 0.8, "suggestions": [], "issues": []},
        ]

        spec = ArchitectureSpec(spec_id="test")
        result = MADArchitecturePanel()._apply_critiques(spec, evaluations)

        assert len(result.risks) > 0


class TestStrategicPlanner:
    """Tests for Strategic Planner"""

    def test_planner_initialization(self):
        """Test planner initialization"""
        planner = StrategicPlanner()
        assert planner is not None

    @pytest.mark.asyncio
    async def test_create_plan(self):
        """Test creating a plan"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        intent = MagicMock(spec=StructuredIntent)
        intent.intent_type = IntentType.CODE_GENERATION

        planner = StrategicPlanner()
        plan = await planner.create_plan(spec, intent)

        assert "phases" in plan
        assert "milestones" in plan
        assert isinstance(plan["phases"], list)

    def test_determine_phases(self):
        """Test determining phases"""
        from gaap.layers.layer0_interface import StructuredIntent, IntentType

        spec = ArchitectureSpec(spec_id="test")
        intent = MagicMock(spec=StructuredIntent)

        planner = StrategicPlanner()
        phases = planner._determine_phases(spec, intent)

        assert len(phases) > 0
        assert all("name" in p for p in phases)

    def test_create_milestones(self):
        """Test creating milestones"""
        phases = [{"name": "Phase 1"}, {"name": "Phase 2"}]

        planner = StrategicPlanner()
        milestones = planner._create_milestones(phases)

        assert len(milestones) == 2


class TestLayer1Strategic:
    """Tests for Layer 1 Strategic"""

    def test_layer1_initialization(self):
        """Test Layer 1 initialization"""
        layer1 = Layer1Strategic(tot_depth=3, mad_rounds=2)
        assert layer1 is not None
        assert layer1.tot is not None
        assert layer1.mad_panel is not None
        assert layer1.planner is not None

    def test_layer1_initialization_no_mcts(self):
        """Test Layer 1 initialization without MCTS"""
        layer1 = Layer1Strategic(enable_mcts=False)
        assert layer1._enable_mcts is False

    def test_layer1_stats_initialization(self):
        """Test Layer 1 statistics initialization"""
        layer1 = Layer1Strategic()
        assert layer1._specs_created == 0
        assert layer1._llm_strategies == 0
        assert layer1._fallback_strategies == 0


class TestEpistemicCheckResult:
    """Tests for EpistemicCheckResult"""

    def test_create_epistemic_result(self):
        """Test creating epistemic check result"""
        result = EpistemicCheckResult(
            needs_research=True,
            unknown_terms=["kubernetes", "terraform"],
            critical_gaps=["security"],
            confidence=0.7,
        )
        assert result.needs_research is True
        assert len(result.unknown_terms) == 2
        assert result.confidence == 0.7

    def test_epistemic_default_values(self):
        """Test epistemic default values"""
        result = EpistemicCheckResult()
        assert result.needs_research is False
        assert result.confidence == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
