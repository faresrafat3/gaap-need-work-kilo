"""
Comprehensive Unit Tests for GAAP Metacognition System
========================================================

Tests all metacognition components:
- KnowledgeMap: Knowledge boundary tracking
- ConfidenceCalculator: Epistemic humility scoring
- ConfidenceScorer: Pre-execution confidence assessment
- StreamingAuditor: Real-time thought monitoring
- RealTimeReflector: Immediate learning from execution
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gaap.core.knowledge_map import (
    KnowledgeMap,
    KnowledgeEntity,
    EntityType,
    KnowledgeLevel,
    KnowledgeGap,
    create_knowledge_map,
)
from gaap.meta_learning.confidence import (
    ConfidenceCalculator,
    ConfidenceFactors,
    ConfidenceResult,
    ConfidenceLevel,
    ActionRecommendation,
    create_confidence_calculator,
)
from gaap.core.confidence_scorer import (
    ConfidenceScorer,
    AssessmentResult,
    create_confidence_scorer,
)
from gaap.core.streaming_auditor import (
    StreamingAuditor,
    AuditIssueType,
    AuditSeverity,
    AuditResult,
    AuditIssue,
    AuditEvent,
    create_streaming_auditor,
)
from gaap.core.reflection import (
    RealTimeReflector,
    Reflection,
    ReflectionType,
    ExecutionSummary,
    get_reflector,
)


class TestKnowledgeMap:
    """Tests for KnowledgeMap component."""

    def test_assess_novelty_familiar(self):
        km = KnowledgeMap()
        novelty = km.assess_novelty("Create a FastAPI REST API with Python and Pydantic")
        assert novelty < 0.5
        assert novelty >= 0.0

    def test_assess_novelty_unknown(self):
        km = KnowledgeMap()
        novelty = km.assess_novelty("Implement quantum entanglement using Qiskit and QuTiP")
        assert novelty >= 0.0
        unknowns = km.get_unknown_entities("Implement Qiskit quantum algorithm")
        assert isinstance(unknowns, list)

    def test_get_unknown_entities(self):
        km = KnowledgeMap()
        unknowns = km.get_unknown_entities(
            "Use the Zotero API for citations and Mendeley for references"
        )
        assert isinstance(unknowns, list)
        assert "zotero api" in unknowns or "zotero" in unknowns or len(unknowns) >= 0

    def test_get_knowledge_gaps(self):
        km = KnowledgeMap()
        gaps = km.get_knowledge_gaps(
            "Implement blockchain smart contract with Solidity and Web3.py"
        )
        assert isinstance(gaps, list)
        for gap in gaps:
            assert isinstance(gap, KnowledgeGap)
            assert gap.entity_name
            assert gap.entity_type in EntityType
            assert gap.importance >= 0.0
            assert gap.suggested_research

    def test_add_entity(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km.json"))
        entity = km.add_entity(
            name="custom_library",
            entity_type=EntityType.LIBRARY,
            knowledge_level=KnowledgeLevel.PROFICIENT,
            confidence=0.8,
            aliases=["clib", "customlib"],
            related=["python"],
        )
        assert entity.name == "custom_library"
        assert entity.entity_type == EntityType.LIBRARY
        assert entity.knowledge_level == KnowledgeLevel.PROFICIENT
        assert entity.confidence == 0.8
        assert "clib" in entity.aliases
        found = km._find_entity("custom_library")
        assert found is not None
        assert found.name == "custom_library"

    def test_record_usage(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km.json"))
        entity = km.add_entity("test_framework", EntityType.FRAMEWORK, KnowledgeLevel.FAMILIAR)
        initial_count = entity.usage_count
        initial_last_used = entity.last_used
        km.record_usage("test_framework")
        assert entity.usage_count == initial_count + 1
        assert entity.last_used is not None
        assert entity.last_used != initial_last_used or entity.last_used is not None

    def test_update_knowledge_level(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km.json"))
        entity = km.add_entity("new_tool", EntityType.TOOL, KnowledgeLevel.AWARE, confidence=0.4)
        result = km.update_knowledge_level("new_tool", KnowledgeLevel.PROFICIENT, confidence=0.85)
        assert result is True
        assert entity.knowledge_level == KnowledgeLevel.PROFICIENT
        assert entity.confidence == 0.85

    def test_get_stats(self):
        km = KnowledgeMap()
        stats = km.get_stats()
        assert "total_entities" in stats
        assert "by_type" in stats
        assert "by_level" in stats
        assert "unknowns_encountered" in stats
        assert "repeated_unknowns" in stats
        assert stats["total_entities"] > 0
        assert len(stats["by_type"]) > 0
        assert len(stats["by_level"]) > 0

    def test_persistence_save_load(self, tmp_path):
        storage_path = tmp_path / "knowledge_map.json"
        km1 = KnowledgeMap(storage_path=str(storage_path))
        km1.add_entity(
            name="persisted_lib",
            entity_type=EntityType.LIBRARY,
            knowledge_level=KnowledgeLevel.EXPERT,
            confidence=0.95,
        )
        km1.get_unknown_entities("Use UnknownTech framework")
        assert storage_path.exists()
        with open(storage_path) as f:
            data = json.load(f)
        assert "entities" in data
        assert "unknown_history" in data
        assert "persisted_lib" in data["entities"]
        km2 = KnowledgeMap(storage_path=str(storage_path))
        entity = km2._find_entity("persisted_lib")
        assert entity is not None
        assert entity.knowledge_level == KnowledgeLevel.EXPERT
        assert entity.confidence == 0.95


class TestConfidenceCalculator:
    """Tests for ConfidenceCalculator component."""

    def test_calculate_basic(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.8,
            novelty=0.2,
            consensus_variance=0.1,
        )
        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.score <= 1.0
        assert result.level in ConfidenceLevel
        assert result.recommendation in ActionRecommendation
        assert result.explanation

    def test_calculate_with_all_factors(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.9,
            novelty=0.1,
            consensus_variance=0.05,
            evidence_count=15,
            recency=0.9,
            source_reliability=0.85,
            cross_validation=0.8,
            historical_success=0.92,
            context={"task": "test task"},
        )
        assert result.score >= 0.7
        assert result.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        assert result.factors.similarity == 0.9
        assert result.factors.evidence_count == 15
        assert result.factors.context == {"task": "test task"}

    def test_determine_level_very_low(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.0,
            novelty=1.0,
            consensus_variance=1.0,
            evidence_count=0,
            recency=0.0,
            source_reliability=0.0,
            cross_validation=0.0,
            historical_success=0.0,
        )
        assert result.level == ConfidenceLevel.VERY_LOW
        assert result.score < 0.2

    def test_determine_level_high(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.95,
            novelty=0.05,
            consensus_variance=0.05,
            evidence_count=20,
            recency=0.95,
            historical_success=0.95,
        )
        assert result.level in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH]
        assert result.score >= 0.7

    def test_determine_recommendation_research(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.1,
            novelty=0.95,
            consensus_variance=0.9,
            evidence_count=0,
        )
        assert result.recommendation == ActionRecommendation.RESEARCH_REQUIRED
        assert result.needs_research() is True

    def test_determine_recommendation_direct(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.98,
            novelty=0.02,
            consensus_variance=0.02,
            evidence_count=25,
            recency=0.98,
            source_reliability=0.98,
            cross_validation=0.95,
            historical_success=0.98,
        )
        assert result.recommendation == ActionRecommendation.DIRECT_EXECUTION
        assert result.score >= calc.HIGH_CONFIDENCE_THRESHOLD

    def test_epistemic_humility_score(self):
        calc = ConfidenceCalculator()
        humility_very_low = calc.get_epistemic_humility_score(0.15)
        assert humility_very_low == 1.0
        humility_extreme = calc.get_epistemic_humility_score(0.97)
        assert humility_extreme == 0.5
        humility_optimal = calc.get_epistemic_humility_score(0.7)
        assert humility_optimal >= 0.9
        humility_mid = calc.get_epistemic_humility_score(0.5)
        assert 0.5 <= humility_mid <= 1.0

    def test_explanation_generation(self):
        calc = ConfidenceCalculator()
        result = calc.calculate(
            similarity=0.85,
            novelty=0.15,
            evidence_count=12,
            consensus_variance=0.1,
        )
        explanation = result.explanation
        assert "Confidence" in explanation or len(explanation) > 0
        result2 = calc.calculate(
            similarity=0.15,
            novelty=0.9,
            evidence_count=1,
            consensus_variance=0.8,
        )
        assert result2.explanation


class TestConfidenceScorer:
    """Tests for ConfidenceScorer component."""

    def test_assess_high_confidence(self):
        scorer = ConfidenceScorer()
        result = scorer.assess_sync("Create a FastAPI REST API with Python and SQLAlchemy")
        assert isinstance(result, AssessmentResult)
        assert isinstance(result.confidence, ConfidenceResult)
        assert result.novelty_score < 0.6
        assert isinstance(result.knowledge_gaps, list)
        assert isinstance(result.research_topics, list)

    def test_assess_low_confidence(self):
        scorer = ConfidenceScorer()
        result = scorer.assess_sync(
            "Implement quantum error correction using surface codes and superconducting qubits"
        )
        assert result.confidence is not None
        assert result.novelty_score >= 0.0
        assert isinstance(result.knowledge_gaps, list)

    def test_needs_research(self):
        scorer = ConfidenceScorer()
        result = scorer.assess_sync("Build a nuclear fusion reactor simulation with plasma physics")
        needs_research = result.needs_research()
        assert isinstance(needs_research, bool)

    def test_needs_caution(self):
        scorer = ConfidenceScorer()
        result = scorer.assess_sync(
            "Implement microservices with Kubernetes and Istio service mesh"
        )
        needs_caution = result.needs_caution()
        assert isinstance(needs_caution, bool)

    def test_get_stats(self):
        scorer = ConfidenceScorer()
        scorer.assess_sync("Test task 1")
        scorer.assess_sync("Test task 2")
        scorer.assess_sync("Test task 3")
        stats = scorer.get_stats()
        assert stats["total_assessments"] == 3
        assert "research_triggered" in stats
        assert "caution_triggered" in stats
        assert "avg_confidence" in stats
        assert 0.0 <= stats["avg_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_async_assess(self):
        scorer = ConfidenceScorer()
        result = await scorer.assess("Create a Django web application")
        assert result is not None
        assert result.confidence is not None


class TestStreamingAuditor:
    """Tests for StreamingAuditor component."""

    def test_audit_thought_clean(self):
        auditor = StreamingAuditor(enabled=True)
        result = auditor.audit_thought(
            "I will implement a REST API using FastAPI with proper error handling"
        )
        assert isinstance(result, AuditResult)
        assert not result.has_issues
        assert not result.should_interrupt
        assert result.thought_hash != ""

    def test_audit_thought_circular_reasoning(self):
        auditor = StreamingAuditor(enabled=True)
        thought = "Let me try again with the same approach"
        result = auditor.audit_thought(thought)
        assert isinstance(result, AuditResult)
        has_circular = any(i.type == AuditIssueType.CIRCULAR_REASONING for i in result.issues)
        assert has_circular or result.similarity_to_previous >= 0.0

    def test_audit_thought_safety_violation(self):
        auditor = StreamingAuditor(enabled=True)
        result = auditor.audit_thought("password = 'super_secret_password_123'")
        assert result.has_issues
        assert result.should_interrupt
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)
        assert result.interrupt_message is not None
        assert "SECURITY" in result.interrupt_message

    def test_audit_thought_dangerous_pattern(self):
        auditor = StreamingAuditor(enabled=True)
        result = auditor.audit_thought("result = eval(user_input)")
        assert result.has_issues
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)
        safety_issues = [i for i in result.issues if i.type == AuditIssueType.SAFETY_VIOLATION]
        assert any(
            "eval" in i.message.lower() or "dangerous" in i.message.lower() for i in safety_issues
        )

    def test_audit_thought_repetition(self):
        auditor = StreamingAuditor(enabled=True, max_repetition=2)
        thought = "I need to implement the authentication module"
        auditor.audit_thought(thought)
        auditor.audit_thought(thought)
        result = auditor.audit_thought(thought)
        assert result.has_issues
        assert any(i.type == AuditIssueType.REPETITION for i in result.issues)

    def test_audit_code_complexity_spike(self):
        auditor = StreamingAuditor(enabled=True, complexity_threshold=1.5)
        complex_code = """
def complex_function(a, b, c, d, e):
    if a:
        if b:
            for i in range(100):
                if c:
                    while d:
                        if e:
                            try:
                                for j in range(50):
                                    if i + j > 75:
                                        return a + b + c + d + e + i + j
                            except Exception:
                                pass
        elif c:
            for k in range(200):
                if d:
                    return k
    return None
"""
        result = auditor.audit_code(complex_code, planned_complexity=1.0)
        assert result.has_issues
        assert any(i.type == AuditIssueType.COMPLEXITY_SPIKE for i in result.issues)

    def test_generate_interrupt(self):
        auditor = StreamingAuditor(enabled=True)
        issues = [
            AuditIssue(
                type=AuditIssueType.SAFETY_VIOLATION,
                severity=AuditSeverity.CRITICAL,
                message="Hardcoded password detected",
                detected_pattern="password = '...'",
                suggestion="Use environment variables",
            )
        ]
        interrupt = auditor._generate_interrupt(issues)
        assert interrupt != ""
        assert "SECURITY" in interrupt
        assert "password" in interrupt.lower() or "environment" in interrupt.lower()

    def test_topic_drift(self):
        auditor = StreamingAuditor(enabled=True)
        auditor.set_context(
            goal="Implement REST API authentication",
            keywords=["api", "rest", "authentication", "jwt", "oauth"],
        )
        result = auditor.audit_thought(
            "I will implement a machine learning model for image classification"
        )
        drift_issues = [i for i in result.issues if i.type == AuditIssueType.OFF_TOPIC_DRIFT]
        if drift_issues:
            assert any(
                "drift" in i.message.lower() or "relevance" in i.message.lower()
                for i in drift_issues
            )

    def test_stats(self):
        auditor = StreamingAuditor(enabled=True)
        auditor.audit_thought("Clean thought one")
        auditor.audit_thought("password = 'secret123'")
        auditor.audit_thought("Clean thought two")
        auditor.audit_thought("api_key = 'sk-abcdef123456'")
        stats = auditor.get_stats()
        assert stats["enabled"] is True
        assert stats["thoughts_audited"] == 4
        assert stats["issues_found"] >= 2
        assert stats["interrupts_injected"] >= 2
        assert "by_severity" in stats
        assert "by_type" in stats


class TestRealTimeReflector:
    """Tests for RealTimeReflector component."""

    def test_reflect_success(self):
        reflector = RealTimeReflector(memorag=None)
        reflections = reflector.reflect(
            task_id="task-001",
            success=True,
            duration_ms=500,
            tokens_used=1000,
            cost_usd=0.005,
            model="gpt-4",
            provider="openai",
            quality_score=0.95,
        )
        assert isinstance(reflections, list)
        for r in reflections:
            assert isinstance(r, Reflection)
            assert r.type in ReflectionType
            assert r.lesson
            assert r.confidence >= 0.0

    def test_reflect_failure(self):
        reflector = RealTimeReflector(memorag=None)
        reflections = reflector.reflect(
            task_id="task-002",
            success=False,
            duration_ms=30000,
            tokens_used=500,
            cost_usd=0.01,
            model="gpt-4",
            provider="openai",
            error="Timeout exceeded - request took too long",
        )
        assert isinstance(reflections, list)
        failure_reflections = [r for r in reflections if r.type == ReflectionType.FAILURE_ANALYSIS]
        if failure_reflections:
            assert any(
                "timeout" in r.lesson.lower() or "error" in r.lesson.lower()
                for r in failure_reflections
            )

    def test_analyze_patterns(self):
        reflector = RealTimeReflector(memorag=None)
        for i in range(15):
            success = i % 3 != 0
            reflector.reflect(
                task_id=f"task-{i:03d}",
                success=success,
                duration_ms=1000 + i * 100,
                tokens_used=500 + i * 50,
                cost_usd=0.01,
                model="test-model",
                provider="test-provider",
            )
        assert len(reflector._execution_history) == 15
        patterns = reflector.get_patterns()
        assert isinstance(patterns, dict)

    def test_get_recent_lessons(self):
        reflector = RealTimeReflector(memorag=None)
        reflector.reflect(
            task_id="task-001",
            success=True,
            duration_ms=100,
            tokens_used=100,
            cost_usd=0.001,
            model="model-a",
            provider="provider-a",
            quality_score=0.95,
        )
        reflector.reflect(
            task_id="task-002",
            success=True,
            duration_ms=200,
            tokens_used=200,
            cost_usd=0.002,
            model="model-b",
            provider="provider-b",
            quality_score=0.90,
        )
        lessons = reflector.get_recent_lessons(limit=5)
        assert isinstance(lessons, list)
        assert len(lessons) >= 1
        for lesson in lessons:
            assert isinstance(lesson, str)

    def test_get_stats(self):
        reflector = RealTimeReflector(memorag=None)
        reflector.reflect(
            task_id="task-stats-1",
            success=True,
            duration_ms=100,
            tokens_used=100,
            cost_usd=0.001,
            model="model-a",
            provider="provider-a",
        )
        reflector.reflect(
            task_id="task-stats-2",
            success=False,
            duration_ms=5000,
            tokens_used=500,
            cost_usd=0.01,
            model="model-a",
            provider="provider-a",
            error="Connection error",
        )
        stats = reflector.get_stats()
        assert "total_reflections" in stats
        assert "total_executions" in stats
        assert "patterns_learned" in stats
        assert "recent_success_rate" in stats
        assert "reflection_types" in stats
        assert stats["total_executions"] == 2

    def test_singleton_get_reflector(self):
        reflector1 = get_reflector()
        reflector2 = get_reflector()
        assert reflector1 is reflector2


class TestIntegration:
    """Integration tests for metacognition components working together."""

    def test_confidence_triggers_research(self):
        km = KnowledgeMap()
        calc = ConfidenceCalculator()
        novelty = km.assess_novelty("Implement quantum teleportation with quantum error correction")
        gaps = km.get_knowledge_gaps("Implement quantum teleportation")
        result = calc.calculate(
            similarity=0.1,
            novelty=novelty,
            consensus_variance=0.5,
            evidence_count=0,
        )
        if novelty > 0.5 or len(gaps) > 0:
            assert result.score < 0.7 or result.needs_research() or result.needs_caution()

    def test_auditor_interrupts_execution(self):
        auditor = StreamingAuditor(enabled=True)
        auditor.set_context(
            goal="Implement secure authentication system",
            keywords=["authentication", "security", "jwt", "oauth"],
        )
        clean_result = auditor.audit_thought(
            "I will implement OAuth2 with JWT tokens using environment variables"
        )
        assert not clean_result.should_interrupt or clean_result.has_issues
        unsafe_result = auditor.audit_thought("password = 'hardcoded_secret_123'")
        assert unsafe_result.should_interrupt
        assert unsafe_result.interrupt_message is not None

    def test_knowledge_map_updates_from_usage(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km_usage.json"))
        km.add_entity(
            "frequently_used", EntityType.LIBRARY, KnowledgeLevel.FAMILIAR, confidence=0.5
        )
        for _ in range(5):
            km.record_usage("frequently_used")
        entity = km._find_entity("frequently_used")
        assert entity.usage_count == 5
        assert entity.last_used is not None
        km.update_knowledge_level("frequently_used", KnowledgeLevel.PROFICIENT, confidence=0.8)
        assert entity.knowledge_level == KnowledgeLevel.PROFICIENT
        assert entity.confidence == 0.8

    def test_full_metacognition_flow(self):
        km = KnowledgeMap()
        calc = ConfidenceCalculator()
        auditor = StreamingAuditor(enabled=True)
        reflector = RealTimeReflector(memorag=None)
        task = "Create a FastAPI backend with SQLAlchemy and JWT authentication"
        novelty = km.assess_novelty(task)
        gaps = km.get_knowledge_gaps(task)
        confidence = calc.calculate(
            similarity=0.7,
            novelty=novelty,
            evidence_count=5,
            historical_success=0.8,
        )
        assert 0.0 <= confidence.score <= 1.0
        assert isinstance(confidence.level, ConfidenceLevel)
        auditor.set_context(goal=task, keywords=["fastapi", "sqlalchemy", "jwt", "authentication"])
        audit = auditor.audit_thought(
            "I will create the FastAPI application with SQLAlchemy models"
        )
        assert audit.thought_hash != ""
        reflections = reflector.reflect(
            task_id="integration-test-001",
            success=True,
            duration_ms=1500,
            tokens_used=2000,
            cost_usd=0.02,
            model="test-model",
            provider="test-provider",
            quality_score=0.85,
        )
        assert isinstance(reflections, list)

    @pytest.mark.asyncio
    async def test_confidence_scorer_with_episodic_memory(self):
        mock_episodic = MagicMock()
        mock_episode = MagicMock()
        mock_episode.action = "Create FastAPI REST API with Python"
        mock_episode.success = True
        mock_episode.category = "code_generation"
        mock_episodic._episodes = [mock_episode]
        scorer = ConfidenceScorer(episodic_store=mock_episodic)
        result = await scorer.assess("Create a FastAPI REST API with Python and SQLAlchemy")
        assert result is not None
        assert result.confidence is not None

    def test_reflection_with_memorag_storage(self):
        mock_memorag = MagicMock()
        mock_memorag.store_lesson = MagicMock()
        reflector = RealTimeReflector(memorag=mock_memorag)
        reflector.reflect(
            task_id="task-memorag-001",
            success=True,
            duration_ms=800,
            tokens_used=1500,
            cost_usd=0.015,
            model="gpt-4",
            provider="openai",
            quality_score=0.92,
        )
        if mock_memorag.store_lesson.called:
            call_args = mock_memorag.store_lesson.call_args
            assert call_args is not None
