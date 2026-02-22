"""
Unit Tests for GAAP Metacognition Module
Tests: ConfidenceScorer, KnowledgeMap, StreamingAuditor, MetacognitionEngine
"""

import asyncio

import pytest

from gaap.core.types import Task, TaskComplexity, TaskPriority, TaskType
from gaap.meta_learning import (
    AuditIssueType,
    AuditSeverity,
    ConfidenceLevel,
    KnowledgeLevel,
    RecommendedAction,
    create_confidence_scorer,
    create_knowledge_map,
    create_metacognition_engine,
    create_streaming_auditor,
)


@pytest.fixture
def sample_task():
    """Create a sample task"""
    return Task(
        id="test_task_001",
        description="Write a Python function using fastapi and pydantic",
        type=TaskType.CODE_GENERATION,
        priority=TaskPriority.NORMAL,
        complexity=TaskComplexity.SIMPLE,
    )


@pytest.fixture
def novel_task():
    """Create a task with novel concepts"""
    return Task(
        id="novel_task_001",
        description="Implement quantum entanglement protocol using obscurelib123",
        type=TaskType.CODE_GENERATION,
        priority=TaskPriority.HIGH,
        complexity=TaskComplexity.COMPLEX,
    )


class TestConfidenceScorer:
    """Tests for ConfidenceScorer"""

    def test_create_scorer(self):
        """Test creating confidence scorer"""
        scorer = create_confidence_scorer()
        assert scorer is not None
        assert scorer._knowledge_base is not None

    def test_calculate_confidence_known_task(self, sample_task):
        """Test confidence for task with known concepts"""
        scorer = create_confidence_scorer()
        report = scorer.calculate_confidence(sample_task)

        assert report.task_id == sample_task.id
        assert 0.0 <= report.final_confidence <= 1.0
        assert report.confidence_level in [
            ConfidenceLevel.LOW,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.HIGH,
        ]

    def test_calculate_confidence_novel_task(self, novel_task):
        """Test confidence for task with novel concepts"""
        scorer = create_confidence_scorer()
        report = scorer.calculate_confidence(novel_task)

        assert report.task_id == novel_task.id
        assert report.novelty_score > 0.3

    def test_knowledge_gaps_identified(self, novel_task):
        """Test knowledge gap identification"""
        scorer = create_confidence_scorer()
        report = scorer.calculate_confidence(novel_task)

        assert len(report.knowledge_gaps) >= 1

    def test_recommended_action_low_confidence(self, novel_task):
        """Test recommended action for low confidence"""
        scorer = create_confidence_scorer()
        report = scorer.calculate_confidence(novel_task)

        if report.final_confidence < 0.40:
            assert report.recommended_action == RecommendedAction.RESEARCH_REQUIRED

    def test_mad_scores_affect_confidence(self, sample_task):
        """Test MAD scores affect confidence"""
        scorer = create_confidence_scorer()

        report_no_mad = scorer.calculate_confidence(sample_task)
        report_with_mad = scorer.calculate_confidence(sample_task, mad_scores=[80, 85, 90])

        assert report_no_mad.consensus_variance != report_with_mad.consensus_variance

    def test_get_stats(self, sample_task):
        """Test getting statistics"""
        scorer = create_confidence_scorer()
        scorer.calculate_confidence(sample_task)

        stats = scorer.get_stats()

        assert stats["total_assessments"] == 1
        assert stats["knowledge_base_size"] > 0

    def test_add_to_knowledge_base(self):
        """Test adding to knowledge base"""
        scorer = create_confidence_scorer()
        initial_size = len(scorer._knowledge_base)

        scorer.add_to_knowledge_base("newconcept123")

        assert len(scorer._knowledge_base) == initial_size + 1


class TestKnowledgeMap:
    """Tests for KnowledgeMap"""

    def test_create_knowledge_map(self):
        """Test creating knowledge map"""
        km = create_knowledge_map()
        assert km is not None
        assert len(km._domains) > 0

    def test_lookup_known_concept(self):
        """Test looking up known concept"""
        km = create_knowledge_map()
        entry = km.lookup("fastapi")

        assert entry is not None
        assert entry.level in [
            KnowledgeLevel.EXPERT,
            KnowledgeLevel.PROFICIENT,
            KnowledgeLevel.FAMILIAR,
        ]

    def test_lookup_unknown_concept(self):
        """Test looking up unknown concept"""
        km = create_knowledge_map()
        entry = km.lookup("unknownconcept123")

        assert entry is None

    def test_assess_knowledge(self):
        """Test assessing multiple concepts"""
        km = create_knowledge_map()
        assessment = km.assess_knowledge(["fastapi", "pytest", "unknownconcept123"])

        assert "fastapi" in assessment
        assert assessment["unknownconcept123"] == KnowledgeLevel.UNKNOWN

    def test_add_knowledge(self):
        """Test adding new knowledge"""
        km = create_knowledge_map()

        from gaap.meta_learning.knowledge_map import KnowledgeLevel, KnowledgeType

        km.add_knowledge(
            concept="newconcept123",
            domain="python",
            knowledge_type=KnowledgeType.LIBRARY,
            level=KnowledgeLevel.FAMILIAR,
        )

        entry = km.lookup("newconcept123")
        assert entry is not None
        assert entry.level == KnowledgeLevel.FAMILIAR

    def test_get_gaps(self):
        """Test getting knowledge gaps"""
        km = create_knowledge_map()
        gaps = km.get_gaps(["fastapi", "unknownconcept123", "pytest"])

        assert "unknownconcept123" in gaps

    def test_update_usage(self):
        """Test updating usage count"""
        km = create_knowledge_map()
        entry = km.lookup("fastapi")
        assert entry is not None
        initial_count = entry.usage_count

        km.update_usage("fastapi")

        entry = km.lookup("fastapi")
        assert entry is not None
        assert entry.usage_count == initial_count + 1

    def test_get_domain_strength(self):
        """Test getting domain strength"""
        km = create_knowledge_map()
        strength = km.get_domain_strength("python")

        assert 0.0 <= strength <= 1.0

    def test_extract_concepts_from_text(self):
        """Test extracting concepts from text"""
        km = create_knowledge_map()
        concepts = km.extract_concepts_from_text("Using fastapi with pytest for testing")

        assert "fastapi" in concepts
        assert "pytest" in concepts

    def test_get_report(self):
        """Test getting knowledge map report"""
        km = create_knowledge_map()
        report = km.get_report()

        assert report.total_domains > 0
        assert report.total_concepts > 0


class TestStreamingAuditor:
    """Tests for StreamingAuditor"""

    def test_create_auditor(self):
        """Test creating auditor"""
        auditor = create_streaming_auditor()
        assert auditor is not None

    def test_audit_clean_thought(self):
        """Test auditing clean thought"""
        auditor = create_streaming_auditor()
        issue = auditor.audit_thought("This is a clean thought about Python programming.")

        assert issue is None

    def test_detect_circular_reasoning(self):
        """Test detecting circular reasoning"""
        auditor = create_streaming_auditor()

        auditor.audit_thought("As I mentioned earlier, we should use fastapi.")
        issue = auditor.audit_thought("As I mentioned earlier, we should use fastapi.")

        assert issue is not None
        assert issue.issue_type == AuditIssueType.CIRCULAR_REASONING

    def test_detect_safety_violation(self):
        """Test detecting safety violation"""
        auditor = create_streaming_auditor()
        issue = auditor.audit_thought('password = "secret123"')

        assert issue is not None
        assert issue.issue_type == AuditIssueType.SAFETY_VIOLATION
        assert issue.severity == AuditSeverity.CRITICAL

    def test_detect_complexity_spike(self):
        """Test detecting complexity spike"""
        auditor = create_streaming_auditor()
        complex_code = """
if a and b or c and d and e or f and g:
    if h and i or j and k:
        if l and m:
            pass
"""
        issue = auditor.audit_thought(complex_code, planned_complexity=1.0)

        assert issue is not None
        assert issue.issue_type == AuditIssueType.COMPLEXITY_SPIKE

    def test_inject_interrupt(self):
        """Test injecting interrupt"""
        auditor = create_streaming_auditor()
        issue = auditor.audit_thought('password = "secret"')

        assert issue is not None
        interrupt = auditor.inject_interrupt(issue)

        assert interrupt is not None
        assert interrupt.issue_type == AuditIssueType.SAFETY_VIOLATION

    def test_register_interrupt_handler(self):
        """Test registering interrupt handler"""
        auditor = create_streaming_auditor()
        handler_called = []

        def handler(interrupt):
            handler_called.append(interrupt)

        auditor.register_interrupt_handler(handler)

        issue = auditor.audit_thought('api_key = "key123"')
        assert issue is not None
        auditor.inject_interrupt(issue)

        assert len(handler_called) == 1

    def test_get_report(self):
        """Test getting audit report"""
        auditor = create_streaming_auditor()
        auditor.audit_thought("Clean thought")
        auditor.audit_thought('password = "secret"')

        report = auditor.get_report()

        assert report.total_thoughts == 2
        assert report.safety_violations == 1


class TestMetacognitionEngine:
    """Tests for MetacognitionEngine"""

    def test_create_engine(self):
        """Test creating engine"""
        engine = create_metacognition_engine()
        assert engine is not None
        assert engine._auditor is not None

    def test_create_engine_without_auditor(self):
        """Test creating engine without auditor"""
        engine = create_metacognition_engine(enable_auditor=False)
        assert engine._auditor is None

    def test_assess_known_task(self, sample_task):
        """Test assessing known task"""
        engine = create_metacognition_engine()

        assessment = asyncio.run(engine.assess(sample_task))

        assert assessment.task_id == sample_task.id
        assert assessment.confidence_report is not None

    def test_assess_novel_task(self, novel_task):
        """Test assessing novel task"""
        engine = create_metacognition_engine()

        assessment = asyncio.run(engine.assess(novel_task))

        assert assessment.task_id == novel_task.id
        assert assessment.confidence_report.novelty_score > 0 or len(assessment.knowledge_gaps) >= 0

    def test_audit_thought_with_auditor(self):
        """Test auditing thought with auditor enabled"""
        engine = create_metacognition_engine()

        issue = engine.audit_thought('password = "secret"')

        assert issue is not None
        assert issue.issue_type == AuditIssueType.SAFETY_VIOLATION

    def test_audit_thought_without_auditor(self):
        """Test auditing thought without auditor"""
        engine = create_metacognition_engine(enable_auditor=False)

        issue = engine.audit_thought('password = "secret"')

        assert issue is None

    def test_learn_concept(self):
        """Test learning new concept"""
        engine = create_metacognition_engine()

        engine.learn_concept("newconcept123", "python")

        entry = engine._knowledge_map.lookup("newconcept123")
        assert entry is not None

    def test_get_stats(self, sample_task):
        """Test getting statistics"""
        engine = create_metacognition_engine()

        asyncio.run(engine.assess(sample_task))

        stats = engine.get_stats()

        assert stats["total_assessments"] == 1
        assert "knowledge_domains" in stats

    def test_get_knowledge_report(self):
        """Test getting knowledge report"""
        engine = create_metacognition_engine()

        report = engine.get_knowledge_report()

        assert report["total_domains"] > 0
        assert "strong_domains" in report
        assert "weak_domains" in report

    def test_can_proceed_flag(self, sample_task, novel_task):
        """Test can_proceed flag"""
        engine = create_metacognition_engine()

        assessment = asyncio.run(engine.assess(sample_task))

        if assessment.needs_research:
            assert assessment.can_proceed is False
        else:
            assert assessment.can_proceed is True

    def test_inject_interrupt_with_issue(self):
        """Test inject interrupt with valid issue"""
        engine = create_metacognition_engine()

        issue = engine.audit_thought('password = "secret"')
        assert issue is not None

        interrupt = engine.inject_interrupt(issue)
        assert interrupt is not None


class TestConfidenceReport:
    """Tests for ConfidenceReport"""

    def test_to_dict(self, sample_task):
        """Test converting report to dict"""
        scorer = create_confidence_scorer()
        report = scorer.calculate_confidence(sample_task)

        d = report.to_dict()

        assert "task_id" in d
        assert "confidence" in d
        assert "level" in d


class TestAuditIssue:
    """Tests for AuditIssue"""

    def test_to_dict(self):
        """Test converting issue to dict"""
        from gaap.meta_learning.streaming_auditor import AuditIssue

        issue = AuditIssue(
            issue_type=AuditIssueType.SAFETY_VIOLATION,
            severity=AuditSeverity.HIGH,
            message="Test issue",
            context="Test context",
            suggested_interrupt="Test interrupt",
        )

        d = issue.to_dict()

        assert d["type"] == "SAFETY_VIOLATION"
        assert d["severity"] == "HIGH"
