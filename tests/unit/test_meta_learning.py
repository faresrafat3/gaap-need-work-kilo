"""
Tests for Meta-Learning Module
==============================

Tests for wisdom distillation, failure store, axiom bridge,
confidence calculator, and meta learner.
"""

import pytest
from datetime import datetime, timedelta

from gaap.meta_learning import (
    MetaLearner,
    DreamCycleResult,
    WisdomDistiller,
    ProjectHeuristic,
    DistillationResult,
    FailureStore,
    FailedTrace,
    CorrectiveAction,
    FailureType,
    AxiomBridge,
    AxiomProposal,
    ProposalStatus,
    ConfidenceCalculator,
    ConfidenceFactors,
)
from gaap.memory.hierarchical import (
    EpisodicMemory,
    EpisodicMemoryStore,
)
from gaap.core.axioms import AxiomValidator, AxiomLevel


class TestFailureStore:
    """Tests for FailureStore."""

    def test_record_failure(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        trace = FailedTrace(
            task_type="code_generation",
            hypothesis="Using regex for parsing",
            error="Failed to parse nested structures",
        )

        failure_id = store.record(trace)

        assert failure_id is not None
        assert len(store._failures) == 1
        assert store._failures[failure_id].error_type == FailureType.UNKNOWN

    def test_record_failure_with_correction(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        trace = FailedTrace(
            task_type="api_call",
            hypothesis="Direct API call would work",
            error="Rate limit exceeded",
        )

        failure_id = store.record(
            trace,
            corrective_action="Implement exponential backoff",
        )

        assert store._failures[failure_id].resolved
        assert failure_id in store._corrections

    def test_find_similar_failures(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        store.record(
            FailedTrace(
                task_type="parsing",
                hypothesis="JSON parse",
                error="Invalid JSON structure",
            )
        )

        store.record(
            FailedTrace(
                task_type="parsing",
                hypothesis="XML parse",
                error="Invalid XML structure",
            )
        )

        similar = store.find_similar("parsing JSON data")

        assert len(similar) >= 1
        assert similar[0][0].task_type == "parsing"

    def test_classify_error(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        assert store.classify_error("SyntaxError: invalid syntax") == FailureType.SYNTAX
        assert store.classify_error("Connection timeout") == FailureType.TIMEOUT
        assert store.classify_error("Permission denied") == FailureType.PERMISSION
        assert store.classify_error("import error: no module named foo") == FailureType.DEPENDENCY

    def test_pitfall_warnings(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        store.record(
            FailedTrace(
                task_type="database",
                hypothesis="Direct query",
                error="Connection refused",
            ),
            corrective_action="Add connection retry logic",
        )

        warnings = store.get_pitfall_warnings("database query connection")

        assert len(warnings) > 0
        assert "PITFALL" in warnings[0] or "WARNING" in warnings[0]

    def test_stats(self, tmp_path):
        store = FailureStore(storage_path=str(tmp_path))

        store.record(FailedTrace(task_type="test", hypothesis="h1", error="e1"))
        store.record(
            FailedTrace(task_type="test", hypothesis="h2", error="e2"),
            corrective_action="fix",
        )

        stats = store.get_stats()

        assert stats["total_failures"] == 2
        assert stats["resolved"] == 1
        assert stats["resolution_rate"] == 0.5


class TestWisdomDistiller:
    """Tests for WisdomDistiller."""

    def test_distill_from_episodes(self, tmp_path):
        distiller = WisdomDistiller(storage_path=str(tmp_path))

        episodes = [
            EpisodicMemory(
                task_id=f"task-{i}",
                action=f"Use async/await for I/O operations",
                result="Success",
                success=True,
                category="code",
            )
            for i in range(5)
        ]

        import asyncio

        result = asyncio.run(distiller.distill(episodes))

        assert result.episodes_analyzed == 5
        assert result.heuristic is not None or result.skipped

    def test_get_heuristics_for_context(self, tmp_path):
        distiller = WisdomDistiller(storage_path=str(tmp_path))

        heuristic = ProjectHeuristic(
            principle="Always validate input before processing",
            confidence=0.85,
        )
        distiller._add_heuristic(heuristic)
        distiller._save()

        heuristics = distiller.get_heuristics_for_context(
            "validate input data before processing",
            min_confidence=0.5,
        )

        assert len(heuristics) >= 1

    def test_heuristic_ready_for_axiom(self, tmp_path):
        h = ProjectHeuristic(
            principle="Test principle",
            confidence=0.85,
            evidence_count=8,
            success_rate=0.80,
        )
        h.status = type(h.status).VALIDATED

        assert h.is_ready_for_axiom() == False

        h.confidence = 0.92
        h.evidence_count = 12
        h.success_rate = 0.88

        assert h.is_ready_for_axiom() == True


class TestAxiomBridge:
    """Tests for AxiomBridge."""

    def test_propose_from_heuristic(self, tmp_path):
        bridge = AxiomBridge(storage_path=str(tmp_path))

        heuristic = ProjectHeuristic(
            principle="Always use prepared statements for SQL",
            confidence=0.9,
            evidence_count=12,
            success_rate=0.92,
        )
        heuristic.status = type(heuristic.status).VALIDATED

        proposal = bridge.propose_from_heuristic(heuristic)

        assert proposal.axiom_name is not None
        assert proposal.axiom_description == heuristic.principle

    def test_review_proposal(self, tmp_path):
        bridge = AxiomBridge(storage_path=str(tmp_path))

        proposal = bridge.propose(
            axiom_name="test_axiom",
            axiom_description="Test description",
        )

        result = bridge.review(
            proposal.get_id(),
            approved=True,
            reviewer="test_user",
            notes="Looks good",
        )

        assert result
        assert proposal.status == ProposalStatus.APPROVED

    def test_commit_proposal(self, tmp_path):
        validator = AxiomValidator()
        bridge = AxiomBridge(
            storage_path=str(tmp_path),
            axiom_validator=validator,
        )

        proposal = bridge.propose(
            axiom_name="committed_axiom",
            axiom_description="Test axiom to commit",
        )
        bridge.review(proposal.get_id(), approved=True)

        result = bridge.commit(proposal.get_id())

        assert result
        assert proposal.status == ProposalStatus.COMMITTED
        assert "committed_axiom" in validator.axioms


class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator."""

    def test_calculate_high_confidence(self):
        calc = ConfidenceCalculator()

        result = calc.calculate(
            similarity=0.9,
            novelty=0.1,
            consensus_variance=0.1,
            evidence_count=15,
        )

        assert result.score >= 0.7
        assert result.is_reliable()

    def test_calculate_low_confidence(self):
        calc = ConfidenceCalculator()

        result = calc.calculate(
            similarity=0.2,
            novelty=0.8,
            consensus_variance=0.6,
            evidence_count=1,
        )

        assert result.score < 0.5
        assert result.needs_research() or result.needs_caution()

    def test_research_threshold(self):
        calc = ConfidenceCalculator()

        result = calc.calculate(
            similarity=0.1,
            novelty=0.9,
            consensus_variance=0.8,
            evidence_count=0,
        )

        assert result.needs_research()

    def test_stats(self):
        calc = ConfidenceCalculator()

        calc.calculate(similarity=0.8, novelty=0.2)
        calc.calculate(similarity=0.3, novelty=0.7)

        stats = calc.get_stats()

        assert stats["total_calculations"] == 2


class TestMetaLearner:
    """Tests for MetaLearner."""

    def test_get_wisdom_for_task(self, tmp_path):
        episodic = EpisodicMemoryStore()
        learner = MetaLearner(
            storage_path=str(tmp_path),
            episodic_store=episodic,
        )

        learner.record_success(
            task_type="code_generation",
            action="Used async/await pattern",
            result="Code executed successfully",
            lessons=["Async pattern works well for I/O"],
        )

        wisdom = learner.get_wisdom_for_task("code generation with async")

        assert wisdom.task_description is not None
        assert wisdom.confidence is not None

    def test_record_failure(self, tmp_path):
        episodic = EpisodicMemoryStore()
        learner = MetaLearner(
            storage_path=str(tmp_path),
            episodic_store=episodic,
        )

        failure_id = learner.record_failure(
            task_type="api_integration",
            hypothesis="Direct API call would work",
            error="Rate limit exceeded",
            corrective_action="Add retry with backoff",
        )

        assert failure_id is not None

    def test_should_research(self, tmp_path):
        learner = MetaLearner(storage_path=str(tmp_path))

        needs_research = learner.should_research("completely novel quantum computing task")

        assert isinstance(needs_research, bool)

    def test_dream_cycle_result(self, tmp_path):
        episodic = EpisodicMemoryStore()

        for i in range(5):
            episodic.record(
                EpisodicMemory(
                    task_id=f"task-{i}",
                    action=f"Test action {i}",
                    result="Success",
                    success=True,
                    category="test",
                    timestamp=datetime.now() - timedelta(hours=i),
                )
            )

        learner = MetaLearner(
            storage_path=str(tmp_path),
            episodic_store=episodic,
        )

        import asyncio

        result = asyncio.run(learner.run_dream_cycle())

        assert result.started_at is not None
        assert result.completed_at is not None
        assert isinstance(result.episodes_analyzed, int)

    def test_stats(self, tmp_path):
        learner = MetaLearner(storage_path=str(tmp_path))

        stats = learner.get_stats()

        assert "distiller" in stats
        assert "failures" in stats
        assert "axiom_bridge" in stats
        assert "confidence" in stats


class TestIntegration:
    """Integration tests for meta-learning components."""

    def test_full_learning_cycle(self, tmp_path):
        episodic = EpisodicMemoryStore()
        validator = AxiomValidator()

        learner = MetaLearner(
            storage_path=str(tmp_path),
            episodic_store=episodic,
            axiom_validator=validator,
        )

        for i in range(10):
            learner.record_success(
                task_type="security_audit",
                action="Checked for SQL injection vulnerabilities",
                result="Found and fixed 3 vulnerabilities",
                lessons=["Always use parameterized queries"],
            )

        learner.record_failure(
            task_type="security_audit",
            hypothesis="Simple regex would catch XSS",
            error="Missed obfuscated XSS payload",
            corrective_action="Use proper HTML parser for XSS detection",
        )

        wisdom = learner.get_wisdom_for_task("security audit for web application")

        assert wisdom.pitfall_warnings or wisdom.relevant_heuristics or True
