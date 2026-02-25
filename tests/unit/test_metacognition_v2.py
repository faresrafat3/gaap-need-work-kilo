"""
Unit Tests for GAAP Metacognition Module (v2)
Tests: StreamingAuditor, KnowledgeMap, ConfidenceScorer
"""

import pytest
from datetime import datetime, timedelta

from gaap.core.streaming_auditor import (
    StreamingAuditor,
    AuditIssueType,
    AuditSeverity,
    AuditResult,
    AuditIssue,
    create_streaming_auditor,
)
from gaap.core.knowledge_map import (
    KnowledgeMap,
    KnowledgeEntity,
    EntityType,
    KnowledgeLevel,
    KnowledgeGap,
    create_knowledge_map,
)
from gaap.core.confidence_scorer import (
    ConfidenceScorer,
    AssessmentResult,
    create_confidence_scorer,
)


class TestStreamingAuditor:
    """Tests for StreamingAuditor."""

    def test_create_auditor(self):
        auditor = create_streaming_auditor()
        assert auditor is not None
        assert auditor.enabled

    def test_audit_safe_thought(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("I will implement a REST API using FastAPI")

        assert not result.has_issues
        assert not result.should_interrupt
        assert result.thought_hash != ""

    def test_detect_circular_reasoning(self):
        auditor = StreamingAuditor(enabled=True)

        thought = "I need to implement the authentication module"
        auditor.audit_thought(thought)
        auditor.audit_thought(thought)
        result = auditor.audit_thought(thought)

        assert result.has_issues or result.similarity_to_previous > 0.5

    def test_detect_safety_violation(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("password = 'super_secret_123'")

        assert result.has_issues
        assert result.should_interrupt
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)
        assert result.interrupt_message is not None
        assert "SECURITY" in result.interrupt_message

    def test_detect_hardcoded_api_key(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("api_key = 'sk-1234567890abcdef'")

        assert result.has_issues
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)

    def test_detect_dangerous_eval(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("result = eval(user_input)")

        assert result.has_issues
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)

    def test_detect_repetition(self):
        auditor = StreamingAuditor(enabled=True, max_repetition=2)

        thought = "I need to implement the authentication module"

        auditor.audit_thought(thought)
        auditor.audit_thought(thought)
        result = auditor.audit_thought(thought)

        assert result.has_issues
        assert any(i.type == AuditIssueType.REPETITION for i in result.issues)

    def test_audit_code_complexity(self):
        auditor = StreamingAuditor(enabled=True)

        simple_code = "def add(a, b):\n    return a + b"
        result = auditor.audit_code(simple_code, planned_complexity=1.0)
        assert not result.has_issues

    def test_set_context_and_drift_detection(self):
        auditor = StreamingAuditor(enabled=True)

        auditor.set_context(
            goal="Implement REST API with authentication",
            keywords=["api", "rest", "authentication", "fastapi"],
        )

        result = auditor.audit_thought("I will implement the REST endpoints using FastAPI")
        assert result.thought_hash != ""

    def test_get_stats(self):
        auditor = StreamingAuditor(enabled=True)

        auditor.audit_thought("Safe thought 1")
        auditor.audit_thought("password = 'secret'")
        auditor.audit_thought("Safe thought 2")

        stats = auditor.get_stats()

        assert stats["thoughts_audited"] == 3
        assert stats["issues_found"] >= 1
        assert stats["interrupts_injected"] >= 1

    def test_reset(self):
        auditor = StreamingAuditor(enabled=True)

        auditor.audit_thought("Test thought")
        auditor.set_context("Test goal", ["test"])

        auditor.reset()

        assert len(auditor._thought_history) == 0
        assert len(auditor._audit_log) == 0


class TestKnowledgeMap:
    """Tests for KnowledgeMap."""

    def test_create_knowledge_map(self):
        km = create_knowledge_map()
        assert km is not None

    def test_builtin_knowledge_loaded(self):
        km = KnowledgeMap()

        entity = km._find_entity("python")
        assert entity is not None
        assert entity.knowledge_level == KnowledgeLevel.EXPERT

        entity = km._find_entity("fastapi")
        assert entity is not None
        assert entity.knowledge_level == KnowledgeLevel.EXPERT

    def test_assess_novelty_familiar(self):
        km = KnowledgeMap()

        novelty = km.assess_novelty("Create a FastAPI REST API with Python")

        assert novelty < 0.5

    def test_assess_novelty_unknown(self):
        km = KnowledgeMap()

        novelty = km.assess_novelty("Implement quantum entanglement simulation using Qiskit")

        assert novelty >= 0.0

    def test_get_unknown_entities(self):
        km = KnowledgeMap()

        unknowns = km.get_unknown_entities("Use the Qiskit library for quantum computing")

        assert isinstance(unknowns, list)

    def test_get_knowledge_gaps(self):
        km = KnowledgeMap()

        gaps = km.get_knowledge_gaps("Build a neural network with TensorFlow")

        assert isinstance(gaps, list)
        for gap in gaps:
            assert isinstance(gap, KnowledgeGap)
            assert gap.entity_name
            assert gap.suggested_research

    def test_add_entity(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km.json"))

        entity = km.add_entity(
            name="custom_lib",
            entity_type=EntityType.LIBRARY,
            knowledge_level=KnowledgeLevel.FAMILIAR,
            confidence=0.6,
            aliases=["clib"],
        )

        assert entity.name == "custom_lib"
        assert km._find_entity("custom_lib") is not None

    def test_record_usage(self, tmp_path):
        km = KnowledgeMap(storage_path=str(tmp_path / "km.json"))

        entity = km.add_entity("test_lib", EntityType.LIBRARY)
        initial_count = entity.usage_count

        km.record_usage("test_lib")

        assert entity.usage_count == initial_count + 1

    def test_get_stats(self):
        km = KnowledgeMap()

        stats = km.get_stats()

        assert "total_entities" in stats
        assert "by_type" in stats
        assert "by_level" in stats
        assert stats["total_entities"] > 0


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""

    def test_create_scorer(self):
        scorer = create_confidence_scorer()
        assert scorer is not None

    def test_assess_familiar_task(self):
        scorer = ConfidenceScorer()

        result = scorer.assess_sync("Create a REST API using FastAPI and Python")

        assert result.confidence is not None
        assert result.novelty_score < 0.6
        assert isinstance(result.knowledge_gaps, list)
        assert isinstance(result.research_topics, list)

    def test_assess_novel_task(self):
        scorer = ConfidenceScorer()

        result = scorer.assess_sync("Implement blockchain smart contract with Rust and Substrate")

        assert result.confidence is not None
        assert result.novelty_score >= 0.0

    def test_needs_research_method(self):
        scorer = ConfidenceScorer()

        result = scorer.assess_sync("Build quantum computer simulator")

        assert isinstance(result.needs_research(), bool)

    def test_get_epistemic_humility(self):
        scorer = ConfidenceScorer()

        humility_low = scorer.get_epistemic_humility(0.2)
        humility_mid = scorer.get_epistemic_humility(0.5)
        humility_high = scorer.get_epistemic_humility(0.95)

        assert humility_low >= humility_mid
        assert humility_mid >= humility_high

    def test_get_stats(self):
        scorer = ConfidenceScorer()

        scorer.assess_sync("Test task 1")
        scorer.assess_sync("Test task 2")

        stats = scorer.get_stats()

        assert stats["total_assessments"] == 2


class TestIntegration:
    """Integration tests for metacognition components."""

    def test_full_assessment_flow(self):
        km = KnowledgeMap()
        scorer = ConfidenceScorer(knowledge_map=km)
        auditor = StreamingAuditor(enabled=True)

        assessment = scorer.assess_sync("Implement OAuth2 authentication with JWT tokens")

        assert assessment.confidence is not None

        auditor.set_context(
            goal="Implement OAuth2 authentication",
            keywords=["oauth", "jwt", "authentication", "token"],
        )

        audit_result = auditor.audit_thought("I will implement OAuth2 using the authlib library")

        assert audit_result.thought_hash != ""

    def test_safety_detection_with_context(self):
        auditor = StreamingAuditor(enabled=True)

        auditor.set_context(
            goal="Secure API implementation", keywords=["api", "security", "authentication"]
        )

        safe_result = auditor.audit_thought("I will use environment variables for secrets")
        assert not safe_result.has_issues

        unsafe_result = auditor.audit_thought("password = 'admin123'")
        assert unsafe_result.has_issues
        assert unsafe_result.should_interrupt


class TestAuditIssueTypes:
    """Test all audit issue types."""

    def test_safety_violation_detection(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("password = 'secret'")
        assert result.has_issues
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)

    def test_eval_detection(self):
        auditor = StreamingAuditor(enabled=True)

        result = auditor.audit_thought("eval(user_input)")
        assert result.has_issues
        assert any(i.type == AuditIssueType.SAFETY_VIOLATION for i in result.issues)
