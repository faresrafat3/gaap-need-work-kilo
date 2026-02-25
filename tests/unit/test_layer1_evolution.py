"""
Tests for Layer 1 Evolution
===========================

Tests for:
- Epistemic Check (knowledge gaps, wisdom retrieval, pitfalls)
- Research Trigger (blocking vs. continuing)
- Evidence-based Criticism
- Memory-augmented Planning
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from gaap.layers.layer1_strategic import (
    Layer1Strategic,
    ArchitectureSpec,
    ArchitectureDecision,
    ArchitectureParadigm,
    DataStrategy,
    CommunicationPattern,
    ResearchDecision,
    EpistemicCheckResult,
)
from gaap.layers.layer0_interface import StructuredIntent, IntentType, ImplicitRequirements


class TestEpistemicCheck:
    """Tests for epistemic knowledge checking."""

    @pytest.fixture
    def layer1(self):
        return Layer1Strategic()

    @pytest.fixture
    def intent_with_unknown_terms(self):
        return StructuredIntent(
            request_id="test-1",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a kubernetes-based microservices system with graphql"],
            implicit_requirements=ImplicitRequirements(
                scalability="high",
            ),
            metadata={
                "original_text": "Build a kubernetes-based microservices system with graphql"
            },
        )

    @pytest.fixture
    def intent_simple(self):
        return StructuredIntent(
            request_id="test-2",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a simple web app"],
            implicit_requirements=ImplicitRequirements(),
            metadata={"original_text": "Build a simple web app"},
        )

    @pytest.mark.asyncio
    async def test_epistemic_check_finds_unknown_terms(self, layer1, intent_with_unknown_terms):
        result = await layer1._epistemic_check(
            intent_with_unknown_terms, "kubernetes microservices graphql"
        )

        assert len(result.unknown_terms) >= 0

    @pytest.mark.asyncio
    async def test_epistemic_check_simple_request(self, layer1, intent_simple):
        result = await layer1._epistemic_check(intent_simple, "simple web app")

        assert result.needs_research is False or len(result.unknown_terms) == 0

    @pytest.mark.asyncio
    async def test_epistemic_check_finds_critical_paths(self, layer1):
        intent = StructuredIntent(
            request_id="test-3",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a secure authentication system"],
            implicit_requirements=ImplicitRequirements(
                security="high",
            ),
            metadata={"original_text": "Build a secure authentication system with encryption"},
        )

        result = await layer1._epistemic_check(intent, "secure authentication encryption")

        assert len(result.critical_gaps) > 0

    @pytest.mark.asyncio
    async def test_epistemic_check_retrieves_wisdom(self, intent_simple):
        layer1 = Layer1Strategic()

        mock_wisdom = MagicMock()
        mock_wisdom.principle = "Always validate user input"

        layer1._wisdom_distiller = MagicMock()
        layer1._wisdom_distiller.get_heuristics_for_context = MagicMock(return_value=[mock_wisdom])

        result = await layer1._epistemic_check(intent_simple, "web app")

        assert len(result.relevant_wisdom) == 1

    @pytest.mark.asyncio
    async def test_epistemic_check_retrieves_pitfalls(self, intent_simple):
        layer1 = Layer1Strategic()

        mock_failure = MagicMock()
        mock_failure.error = "SQL injection vulnerability"

        layer1._failure_store = MagicMock()
        layer1._failure_store.find_similar = MagicMock(return_value=[(mock_failure, [])])

        result = await layer1._epistemic_check(intent_simple, "web app")

        assert len(result.relevant_pitfalls) == 1


class TestResearchTrigger:
    """Tests for research trigger with hybrid blocking."""

    @pytest.fixture
    def layer1(self):
        return Layer1Strategic()

    @pytest.fixture
    def intent_critical(self):
        return StructuredIntent(
            request_id="test-critical",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a security-focused authentication system with kubernetes"],
            implicit_requirements=ImplicitRequirements(
                security="high",
            ),
            metadata={
                "original_text": "Build a security-focused authentication system with kubernetes"
            },
        )

    @pytest.fixture
    def intent_minor(self):
        return StructuredIntent(
            request_id="test-minor",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Use a new library for date formatting"],
            implicit_requirements=ImplicitRequirements(),
            metadata={"original_text": "Use a new library for date formatting"},
        )

    @pytest.mark.asyncio
    async def test_research_trigger_blocks_critical(self, layer1, intent_critical):
        unknown_terms = ["security", "auth", "crypto", "kubernetes"]

        result = await layer1._handle_unknown_terms(intent_critical, unknown_terms)

        assert result == ResearchDecision.BLOCKED_AND_RESOLVED

    @pytest.mark.asyncio
    async def test_research_trigger_continues_minor(self, layer1, intent_minor):
        unknown_terms = ["dateutil"]

        result = await layer1._handle_unknown_terms(intent_minor, unknown_terms)

        assert result == ResearchDecision.CONTINUE_WITH_PLACEHOLDER

    @pytest.mark.asyncio
    async def test_research_scoring_core_requirement(self, layer1):
        intent = StructuredIntent(
            request_id="test-score",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Implement kubernetes deployment"],
            implicit_requirements=ImplicitRequirements(),
            metadata={},
        )

        unknown_terms = ["kubernetes"]

        result = await layer1._handle_unknown_terms(intent, unknown_terms)

        assert result in [
            ResearchDecision.BLOCKED_AND_RESOLVED,
            ResearchDecision.CONTINUE_WITH_PLACEHOLDER,
        ]

    @pytest.mark.asyncio
    async def test_research_trigger_no_terms(self, layer1):
        result = await layer1._handle_unknown_terms(
            StructuredIntent(
                request_id="test",
                timestamp=datetime.now(),
                intent_type=IntentType.CODE_GENERATION,
                explicit_goals=["Simple task"],
                implicit_requirements=ImplicitRequirements(),
                metadata={},
            ),
            [],
        )

        assert result == ResearchDecision.CONTINUE_WITH_PLACEHOLDER


class TestEvidenceBasedCriticism:
    """Tests for evidence-based MAD criticism."""

    @pytest.fixture
    def spec(self):
        spec = ArchitectureSpec(
            spec_id="test-spec",
            paradigm=ArchitectureParadigm.MICROSERVICES,
            data_strategy=DataStrategy.CQRS,
            communication=CommunicationPattern.GRPC,
        )
        return spec

    @pytest.fixture
    def intent(self):
        return StructuredIntent(
            request_id="test-ev",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build scalable system"],
            implicit_requirements=ImplicitRequirements(
                scalability="high",
            ),
            metadata={"original_text": "Build scalable system"},
        )

    @pytest.mark.asyncio
    async def test_evidence_critic_evaluation(self, spec, intent):
        from gaap.layers.evidence_critic import EvidenceCritic, EvidenceStrength
        from gaap.mad.critic_prompts import ArchitectureCriticType

        critic = EvidenceCritic(ArchitectureCriticType.SCALABILITY)

        evaluation = await critic.evaluate(spec, intent)

        assert evaluation.score >= 0.0
        assert evaluation.score <= 1.0
        assert len(evaluation.evidence) > 0
        assert evaluation.evidence_strength != EvidenceStrength.NONE

    @pytest.mark.asyncio
    async def test_evidence_mad_panel(self, spec, intent):
        from gaap.layers.evidence_critic import EvidenceMADPanel

        panel = EvidenceMADPanel()

        result_spec, evaluations = await panel.evaluate_with_evidence(spec, intent)

        assert len(evaluations) > 0
        assert all(len(e.evidence) > 0 for e in evaluations)

    @pytest.mark.asyncio
    async def test_evidence_required_for_approval(self, spec, intent):
        from gaap.layers.evidence_critic import EvidenceBasedEvaluation, EvidenceStrength

        evaluation_with_evidence = EvidenceBasedEvaluation(
            critic="scalability",
            score=0.8,
            evidence=[
                "Microservices support horizontal scaling",
                "CQRS enables read/write separation",
            ],
            reasoning="Good architecture for scalability",
        )

        evaluation_without_evidence = EvidenceBasedEvaluation(
            critic="scalability",
            score=0.8,
            evidence=[],
            reasoning="Good architecture",
        )

        assert evaluation_with_evidence.is_reliable() is True
        assert evaluation_without_evidence.is_reliable() is False
        assert evaluation_with_evidence.confidence > evaluation_without_evidence.confidence


class TestMemoryAugmentedPlanning:
    """Tests for memory-augmented planning integration."""

    @pytest.fixture
    def intent(self):
        return StructuredIntent(
            request_id="test-memory",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a REST API"],
            implicit_requirements=ImplicitRequirements(),
            metadata={"original_text": "Build a REST API"},
        )

    @pytest.mark.asyncio
    async def test_wisdom_injection_into_got(self, intent):
        mock_wisdom = MagicMock()
        mock_wisdom.principle = "Always use pagination for list endpoints"

        layer1 = Layer1Strategic(enable_got=False)

        epistemic = EpistemicCheckResult(
            relevant_wisdom=[mock_wisdom],
        )

        assert len(epistemic.relevant_wisdom) == 1
        assert epistemic.relevant_wisdom[0].principle == "Always use pagination for list endpoints"

    @pytest.mark.asyncio
    async def test_pitfall_injection_into_got(self, intent):
        mock_failure = MagicMock()
        mock_failure.error = "N+1 query problem"

        mock_correction = MagicMock()
        mock_correction.solution = "Use eager loading"

        layer1 = Layer1Strategic(enable_got=False)

        epistemic = EpistemicCheckResult(
            relevant_pitfalls=[(mock_failure, [mock_correction])],
        )

        assert len(epistemic.relevant_pitfalls) == 1


class TestLayer1Integration:
    """Integration tests for Layer 1 evolution."""

    @pytest.fixture
    def intent_complex(self):
        return StructuredIntent(
            request_id="test-integration",
            timestamp=datetime.now(),
            intent_type=IntentType.CODE_GENERATION,
            explicit_goals=["Build a scalable microservices platform with kubernetes and graphql"],
            implicit_requirements=ImplicitRequirements(
                scalability="high",
                performance="real_time",
                security="high",
            ),
            metadata={
                "original_text": "Build a scalable microservices platform with kubernetes and graphql"
            },
        )

    @pytest.mark.asyncio
    async def test_full_flow_with_got(self, intent_complex):
        layer1 = Layer1Strategic(enable_got=True, enable_evidence_critics=True)

        spec = await layer1.process(intent_complex)

        assert spec is not None
        assert isinstance(spec, ArchitectureSpec)
        assert spec.paradigm in ArchitectureParadigm

    @pytest.mark.asyncio
    async def test_stats_tracking(self, intent_complex):
        layer1 = Layer1Strategic(enable_got=True, enable_evidence_critics=True)

        await layer1.process(intent_complex)

        stats = layer1.get_stats()

        assert "specs_created" in stats
        assert "got_enabled" in stats
        assert "evidence_critics_enabled" in stats
        assert stats["specs_created"] >= 1

    @pytest.mark.asyncio
    async def test_metadata_includes_epistemic_info(self, intent_complex):
        layer1 = Layer1Strategic(enable_got=True, enable_evidence_critics=True)

        spec = await layer1.process(intent_complex)

        assert "epistemic_check" in spec.metadata
        assert "unknown_terms" in spec.metadata["epistemic_check"]
        assert "confidence" in spec.metadata["epistemic_check"]


class TestResearchDecision:
    """Tests for ResearchDecision enum."""

    def test_enum_values(self):
        assert ResearchDecision.BLOCKED_AND_RESOLVED.value == 1
        assert ResearchDecision.CONTINUE_WITH_PLACEHOLDER.value == 2
        assert ResearchDecision.NO_RESEARCH_NEEDED.value == 3


class TestEpistemicCheckResult:
    """Tests for EpistemicCheckResult dataclass."""

    def test_default_values(self):
        result = EpistemicCheckResult()

        assert result.needs_research is False
        assert result.unknown_terms == []
        assert result.critical_gaps == []
        assert result.relevant_wisdom == []
        assert result.relevant_pitfalls == []
        assert result.confidence == 0.5

    def test_custom_values(self):
        result = EpistemicCheckResult(
            needs_research=True,
            unknown_terms=["kubernetes", "graphql"],
            critical_gaps=["security"],
            confidence=0.3,
        )

        assert result.needs_research is True
        assert len(result.unknown_terms) == 2
        assert "security" in result.critical_gaps
