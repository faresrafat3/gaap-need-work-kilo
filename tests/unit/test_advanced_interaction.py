"""
Comprehensive Unit Tests for GAAP Advanced Interaction System
==============================================================

Tests all advanced interaction components:
- PersonaRegistry: Persona registration and retrieval
- PersonaSwitcher: Dynamic persona switching
- SemanticDistiller: Context compression and distillation
- ContrastiveReasoner: Contrastive reasoning paths
- SemanticConstraints: Linguistic constraint enforcement
"""

import pytest
from unittest.mock import MagicMock, patch

from gaap.core.persona import (
    Persona,
    PersonaTier,
    PersonaRegistry,
    PersonaSwitcher,
)
from gaap.core.semantic_distiller import (
    SemanticMatrix,
    SemanticDistiller,
    DistillationResult,
)
from gaap.core.contrastive import (
    ContrastivePath,
    ContrastiveResult,
    ContrastiveReasoner,
)
from gaap.core.semantic_pressure import (
    Constraint,
    ConstraintSeverity,
    ConstraintViolation,
    SemanticConstraints,
)
from gaap.core.types import Message


class TestPersonaRegistry:
    """Tests for PersonaRegistry component."""

    def test_get_persona(self):
        registry = PersonaRegistry()
        persona = registry.get_persona("DEBUGGING")
        assert persona is not None
        assert persona.name == "Forensic Pathologist"

    def test_get_persona_unknown_intent(self):
        registry = PersonaRegistry()
        persona = registry.get_persona("UNKNOWN_INTENT")
        assert persona is not None
        assert persona.name == "Strategic Architect"

    def test_get_persona_by_name(self):
        registry = PersonaRegistry()
        persona = registry.get_persona_by_name("forensic_pathologist")
        assert persona is not None
        assert persona.name == "Forensic Pathologist"
        assert persona.tier == PersonaTier.ADAPTIVE

    def test_get_persona_by_name_not_found(self):
        registry = PersonaRegistry()
        persona = registry.get_persona_by_name("nonexistent_persona")
        assert persona is None

    def test_core_persona(self):
        registry = PersonaRegistry()
        core_personas = registry.list_personas_by_tier(PersonaTier.CORE)
        assert len(core_personas) >= 3
        names = [p.name for p in core_personas]
        assert "Strategic Architect" in names
        assert "Code Practitioner" in names
        assert "Quality Guardian" in names

    def test_adaptive_persona(self):
        registry = PersonaRegistry()
        adaptive_personas = registry.list_personas_by_tier(PersonaTier.ADAPTIVE)
        assert len(adaptive_personas) >= 5
        names = [p.name for p in adaptive_personas]
        assert "Forensic Pathologist" in names
        assert "Civil Engineer" in names
        assert "The Thief" in names

    def test_register_custom_persona(self):
        registry = PersonaRegistry()
        custom = Persona(
            name="Custom Expert",
            description="A custom test persona",
            tier=PersonaTier.TASK,
            values=["Test value 1", "Test value 2"],
            expertise=["Custom expertise"],
            constraints=["Test constraint"],
        )
        registry.register_persona(custom)
        retrieved = registry.get_persona_by_name("custom_expert")
        assert retrieved is not None
        assert retrieved.name == "Custom Expert"

    def test_list_personas(self):
        registry = PersonaRegistry()
        all_personas = registry.list_personas()
        assert "strategic_architect" in all_personas
        assert "forensic_pathologist" in all_personas
        assert len(all_personas) >= 8

    def test_intent_persona_mapping(self):
        registry = PersonaRegistry()
        mapping = registry.INTENT_PERSONA_MAP
        assert mapping["DEBUGGING"] == "forensic_pathologist"
        assert mapping["CODE_REVIEW"] == "quality_guardian"
        assert mapping["REFACTORING"] == "civil_engineer"
        assert mapping["RESEARCH"] == "academic_peer_reviewer"
        assert mapping["CODE_GENERATION"] == "senior_developer"
        assert mapping["PLANNING"] == "strategic_architect"

    def test_persona_to_dict(self):
        registry = PersonaRegistry()
        persona = registry.get_persona_by_name("strategic_architect")
        assert persona is not None
        data = persona.to_dict()
        assert data["name"] == "Strategic Architect"
        assert data["tier"] == "CORE"
        assert "values" in data
        assert "expertise" in data
        assert "constraints" in data


class TestPersonaSwitcher:
    """Tests for PersonaSwitcher component."""

    def test_switch(self):
        registry = PersonaRegistry()
        switcher = PersonaSwitcher(registry)
        persona = switcher.switch("DEBUGGING")
        assert persona is not None
        assert persona.name == "Forensic Pathologist"

    def test_switch_architecture_intent(self):
        switcher = PersonaSwitcher()
        persona = switcher.switch("REFACTORING")
        assert persona.name == "Civil Engineer"
        assert persona.tier == PersonaTier.ADAPTIVE

    def test_switch_security_intent(self):
        switcher = PersonaSwitcher()
        persona = switcher.switch("SECURITY")
        persona_name = persona.name.lower().replace(" ", "_")
        assert persona_name in ["the_thief", "strategic_architect"]

    def test_get_current(self):
        switcher = PersonaSwitcher()
        current = switcher.get_current()
        assert current is not None
        assert current.name == "Strategic Architect"

    def test_get_current_after_switch(self):
        switcher = PersonaSwitcher()
        switcher.switch("DEBUGGING")
        current = switcher.get_current()
        assert current.name == "Forensic Pathologist"

    def test_system_prompt(self):
        switcher = PersonaSwitcher()
        persona = switcher.switch("DEBUGGING")
        prompt = switcher.get_system_prompt(persona)
        assert prompt != ""
        assert "Forensic Pathologist" in prompt
        assert "debugging" in prompt.lower() or "diagnostic" in prompt.lower()

    def test_system_prompt_default(self):
        switcher = PersonaSwitcher()
        prompt = switcher.get_system_prompt()
        assert prompt != ""
        assert "Strategic Architect" in prompt

    def test_get_history(self):
        switcher = PersonaSwitcher()
        switcher.switch("DEBUGGING")
        switcher.switch("REFACTORING")
        history = switcher.get_history()
        assert len(history) == 2
        assert history[0][0] == "DEBUGGING"
        assert history[1][0] == "REFACTORING"

    def test_reset(self):
        switcher = PersonaSwitcher()
        switcher.switch("DEBUGGING")
        switcher.switch("REFACTORING")
        switcher.reset()
        current = switcher.get_current()
        assert current.name == "Strategic Architect"
        history = switcher.get_history()
        assert len(history) == 0

    def test_system_prompt_includes_values(self):
        switcher = PersonaSwitcher()
        persona = switcher.switch("CODE_GENERATION")
        prompt = switcher.get_system_prompt(persona)
        assert "Production-ready code" in prompt or "values" in prompt.lower()


class TestSemanticDistiller:
    """Tests for SemanticDistiller component."""

    def _create_messages(self, contents: list[str]) -> list[Message]:
        messages = []
        for content in contents:
            msg = MagicMock(spec=Message)
            msg.content = content
            msg.role = "user"
            messages.append(msg)
        return messages

    def test_distill(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "We decided to use FastAPI for the backend.",
                "The database will be PostgreSQL.",
                "We should implement authentication with JWT tokens.",
            ]
        )
        matrix = distiller.distill(messages)
        assert isinstance(matrix, SemanticMatrix)
        assert len(matrix.facts) > 0 or len(matrix.decisions) > 0

    def test_distill_extracts_decisions(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "We decided to use Python for this project.",
                "I chose FastAPI as the web framework.",
            ]
        )
        matrix = distiller.distill(messages)
        assert len(matrix.decisions) >= 1

    def test_distill_extracts_facts(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "The API uses REST architecture.",
                "We are using Python and FastAPI.",
            ]
        )
        matrix = distiller.distill(messages)
        assert len(matrix.facts) >= 1

    def test_distill_extracts_risks(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "Risk: The database might become a bottleneck.",
                "Warning: This could lead to performance issues.",
            ]
        )
        matrix = distiller.distill(messages)
        assert len(matrix.pending_risks) >= 1

    def test_distill_extracts_actions(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "TODO: Add unit tests for the API.",
                "We need to implement error handling.",
            ]
        )
        matrix = distiller.distill(messages)
        assert len(matrix.action_items) >= 1

    def test_should_distill(self):
        distiller = SemanticDistiller(distill_interval=5)
        assert distiller.should_distill(5) is True
        assert distiller.should_distill(10) is True
        assert distiller.should_distill(3) is False
        assert distiller.should_distill(7) is False

    def test_should_distill_custom_interval(self):
        distiller = SemanticDistiller(distill_interval=3)
        assert distiller.should_distill(3) is True
        assert distiller.should_distill(6) is True
        assert distiller.should_distill(4) is False

    def test_archive(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "We decided on Python for the backend.",
                "The API will use JWT authentication.",
            ]
        )
        matrix = distiller.distill(messages)
        assert distiller.get_matrix() is not None
        archived = distiller.archive_to_episodic(messages)
        assert archived is not None
        assert distiller.get_matrix() is None

    def test_get_active_context(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "We decided to use FastAPI.",
                "The database is PostgreSQL.",
                "Risk: Scalability might be an issue.",
            ]
        )
        distiller.distill(messages)
        context = distiller.get_active_context()
        assert isinstance(context, list)
        assert len(context) > 0

    def test_get_statistics(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(
            [
                "We decided to use FastAPI.",
                "The database is PostgreSQL.",
            ]
        )
        distiller.distill(messages)
        stats = distiller.get_statistics()
        assert stats["distillation_count"] == 1
        assert stats["message_count"] == 2

    def test_reset(self):
        distiller = SemanticDistiller()
        messages = self._create_messages(["Test message"])
        distiller.distill(messages)
        distiller.reset()
        stats = distiller.get_statistics()
        assert stats["distillation_count"] == 0
        assert stats["message_count"] == 0

    def test_matrix_merge(self):
        matrix1 = SemanticMatrix(
            facts=["Fact 1", "Fact 2"],
            decisions=[{"what": "Decision 1"}],
            pending_risks=["Risk 1"],
        )
        matrix2 = SemanticMatrix(
            facts=["Fact 2", "Fact 3"],
            decisions=[{"what": "Decision 2"}],
            pending_risks=["Risk 2"],
        )
        merged = matrix1.merge(matrix2)
        assert "Fact 1" in merged.facts
        assert "Fact 2" in merged.facts
        assert "Fact 3" in merged.facts
        assert len(merged.decisions) == 2

    def test_matrix_to_dict(self):
        matrix = SemanticMatrix(
            facts=["Test fact"],
            decisions=[{"what": "Test decision"}],
            pending_risks=["Test risk"],
            summary="Test summary",
        )
        data = matrix.to_dict()
        assert data["facts"] == ["Test fact"]
        assert data["summary"] == "Test summary"

    def test_matrix_from_dict(self):
        data = {
            "facts": ["Fact from dict"],
            "decisions": [{"what": "Decision from dict"}],
            "pending_risks": ["Risk from dict"],
            "summary": "Summary from dict",
        }
        matrix = SemanticMatrix.from_dict(data)
        assert matrix.facts == ["Fact from dict"]
        assert matrix.summary == "Summary from dict"


class TestContrastiveReasoner:
    """Tests for ContrastiveReasoner component."""

    def test_generate_paths(self):
        reasoner = ContrastiveReasoner()
        path_a, path_b = reasoner.generate_paths("Should we use microservices or monolith?")
        assert path_a is not None
        assert path_b is not None
        assert path_a.name != path_b.name

    def test_generate_paths_architecture(self):
        reasoner = ContrastiveReasoner()
        path_a, path_b = reasoner.generate_paths(
            "What architecture should we choose for the system?"
        )
        assert path_a.name in [
            "Monolith",
            "Microservices",
            "Conservative Approach",
            "Aggressive Approach",
        ]
        assert path_b.name in [
            "Monolith",
            "Microservices",
            "Conservative Approach",
            "Aggressive Approach",
        ]

    def test_generate_paths_database(self):
        reasoner = ContrastiveReasoner()
        path_a, path_b = reasoner.generate_paths("Which database should we use for storage?")
        assert path_a.name in [
            "SQL Database",
            "NoSQL Database",
            "Conservative Approach",
            "Aggressive Approach",
        ]

    def test_generate_paths_generic(self):
        reasoner = ContrastiveReasoner()
        path_a, path_b = reasoner.generate_paths("Some random question without keywords")
        assert path_a.name == "Conservative Approach"
        assert path_b.name == "Aggressive Approach"

    def test_synthesize(self):
        reasoner = ContrastiveReasoner()
        path_a = ContrastivePath(
            name="Option A",
            reasoning="First option",
            pros=["Pro 1", "Pro 2"],
            cons=["Con 1"],
            risks=["Risk 1"],
        )
        path_b = ContrastivePath(
            name="Option B",
            reasoning="Second option",
            pros=["Pro 3"],
            cons=["Con 2", "Con 3"],
            risks=["Risk 2"],
        )
        synthesis = reasoner.synthesize(path_a, path_b)
        assert "Option A" in synthesis
        assert "Option B" in synthesis

    def test_path_score(self):
        path = ContrastivePath(
            name="Test Path",
            reasoning="Test reasoning",
            pros=["Pro 1", "Pro 2", "Pro 3"],
            cons=["Con 1"],
            risks=["Risk 1"],
            estimated_cost=0.3,
            confidence=0.8,
        )
        score = path.score()
        assert 0.0 <= score <= 1.0

    def test_path_score_high_pros(self):
        high_pro_path = ContrastivePath(
            name="High Pros",
            pros=["Pro 1", "Pro 2", "Pro 3", "Pro 4", "Pro 5"],
            cons=[],
            risks=[],
            estimated_cost=0.1,
            confidence=1.0,
        )
        low_pro_path = ContrastivePath(
            name="Low Pros",
            pros=["Pro 1"],
            cons=["Con 1", "Con 2"],
            risks=["Risk 1"],
            estimated_cost=0.9,
            confidence=0.5,
        )
        assert high_pro_path.score() > low_pro_path.score()

    def test_reason_about(self):
        reasoner = ContrastiveReasoner()
        result = reasoner.reason_about("Should we use microservices?")
        assert isinstance(result, ContrastiveResult)
        assert result.path_a is not None
        assert result.path_b is not None
        assert result.final_decision != ""

    def test_reason_about_with_context(self):
        reasoner = ContrastiveReasoner()
        context = {
            "budget": "budget_conscious",
            "timeline": "urgent",
        }
        result = reasoner.reason_about(
            "What architecture should we use?",
            context=context,
        )
        assert result is not None
        assert result.confidence >= 0.0

    def test_get_winning_path(self):
        path_a = ContrastivePath(
            name="Winner",
            pros=["Pro 1", "Pro 2", "Pro 3"],
            cons=[],
            risks=[],
            estimated_cost=0.1,
            confidence=1.0,
        )
        path_b = ContrastivePath(
            name="Loser",
            pros=[],
            cons=["Con 1", "Con 2"],
            risks=["Risk 1", "Risk 2"],
            estimated_cost=0.9,
            confidence=0.5,
        )
        result = ContrastiveResult(path_a=path_a, path_b=path_b)
        winner = result.get_winning_path()
        assert winner.name == "Winner"

    def test_result_to_dict(self):
        path_a = ContrastivePath(name="Path A", pros=["Pro 1"])
        path_b = ContrastivePath(name="Path B", pros=["Pro 2"])
        result = ContrastiveResult(
            path_a=path_a,
            path_b=path_b,
            synthesis="Test synthesis",
            final_decision="Test decision",
            confidence=0.8,
        )
        data = result.to_dict()
        assert data["synthesis"] == "Test synthesis"
        assert data["final_decision"] == "Test decision"
        assert data["confidence"] == 0.8

    def test_add_custom_paths(self):
        reasoner = ContrastiveReasoner()
        custom_path_a = ContrastivePath(
            name="Custom A",
            reasoning="Custom option A",
            pros=["Custom pro"],
        )
        custom_path_b = ContrastivePath(
            name="Custom B",
            reasoning="Custom option B",
            pros=["Custom pro"],
        )
        reasoner.add_custom_paths("custom_decision", custom_path_a, custom_path_b)
        assert "custom_decision" in reasoner.ARCHITECTURE_PATHS
        stored_a, stored_b = reasoner.ARCHITECTURE_PATHS["custom_decision"]
        assert stored_a.name == "Custom A"
        assert stored_b.name == "Custom B"

    def test_get_statistics(self):
        reasoner = ContrastiveReasoner()
        reasoner.reason_about("Decision 1")
        reasoner.reason_about("Decision 2")
        stats = reasoner.get_statistics()
        assert stats["reasoning_count"] == 2


class TestSemanticConstraints:
    """Tests for SemanticConstraints component."""

    def test_check_text_clean(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("The function returns a list of integers.")
        assert violations == []

    def test_check_text_vague_terms(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("We need to ensure robust and efficient performance.")
        assert len(violations) >= 3
        matched = [v.matched_text.lower() for v in violations]
        assert "ensure" in matched
        assert "robust" in matched
        assert "efficient" in matched

    def test_check_text_metric_terms(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("The algorithm is fast and scalable.")
        metric_violations = [v for v in violations if v.constraint.category == "metric"]
        assert len(metric_violations) >= 2

    def test_check_text_placeholder(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("TODO: Implement this function")
        placeholder_violations = [v for v in violations if v.constraint.category == "placeholder"]
        assert len(placeholder_violations) == 1
        assert placeholder_violations[0].constraint.severity == ConstraintSeverity.ERROR

    def test_check_text_etc(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("Supports Python, JavaScript, etc.")
        incomplete_violations = [v for v in violations if v.constraint.category == "incomplete"]
        assert len(incomplete_violations) >= 1

    def test_check_text_severity(self):
        constraints = SemanticConstraints()
        text = "TODO: Fix this. We need to ensure robust behavior."
        violations = constraints.check_text_severity(text, min_severity=ConstraintSeverity.ERROR)
        for v in violations:
            assert v.constraint.severity == ConstraintSeverity.ERROR

    def test_apply_pressure_prompt(self):
        constraints = SemanticConstraints()
        prompt = constraints.apply_pressure_prompt()
        assert "Specific" in prompt
        assert "Measurable" in prompt
        assert "Actionable" in prompt
        assert "ensure" in prompt.lower()

    def test_apply_pressure_prompt_short(self):
        constraints = SemanticConstraints()
        prompt = constraints.apply_pressure_prompt_short()
        assert "vague" in prompt.lower()
        assert "ensure" in prompt.lower()

    def test_fix_text(self):
        constraints = SemanticConstraints()
        original = "We need to ensure robust performance."
        fixed, remaining = constraints.fix_text(original)
        assert "ensure" not in fixed or "robust" not in fixed

    def test_violation_to_dict(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text("This is robust.")
        if violations:
            data = violations[0].to_dict()
            assert "pattern" in data
            assert "severity" in data
            assert "matched_text" in data

    def test_get_statistics(self):
        constraints = SemanticConstraints()
        constraints.check_text("This is robust and efficient.")
        constraints.check_text("Clean text without issues.")
        stats = constraints.get_statistics()
        assert stats["total_checks"] == 2
        assert stats["total_violations"] >= 2

    def test_add_custom_constraint(self):
        constraints = SemanticConstraints()
        constraints.add_custom_constraint(
            pattern=r"\bdeprecated\b",
            requirement="Avoid deprecated APIs",
            severity=ConstraintSeverity.WARNING,
            category="api",
        )
        violations = constraints.check_text("This is a deprecated method.")
        deprecated_violations = [v for v in violations if "deprecated" in v.matched_text.lower()]
        assert len(deprecated_violations) >= 1

    def test_get_constraints_by_category(self):
        constraints = SemanticConstraints()
        categories = constraints.get_constraints_by_category()
        assert "vague" in categories
        assert "metric" in categories
        assert "quantifier" in categories
        assert "placeholder" in categories

    def test_constraint_matches(self):
        constraint = Constraint(
            pattern=r"\btest\b",
            requirement="Test requirement",
            severity=ConstraintSeverity.INFO,
        )
        assert constraint.matches("This is a test sentence.") is True
        assert constraint.matches("No match here.") is False

    def test_constraint_find_all(self):
        constraint = Constraint(
            pattern=r"\b\w+ing\b",
            requirement="Avoid gerunds",
            severity=ConstraintSeverity.INFO,
        )
        matches = constraint.find_all("Running and jumping are fun.")
        assert len(matches) >= 2

    def test_quantifier_detection(self):
        constraints = SemanticConstraints()
        violations = constraints.check_text(
            "This shows significant improvement with substantial benefits."
        )
        quantifier_violations = [v for v in violations if v.constraint.category == "quantifier"]
        assert len(quantifier_violations) >= 2


class TestIntegration:
    """Integration tests for advanced interaction components working together."""

    def test_persona_with_constraints(self):
        registry = PersonaRegistry()
        constraints = SemanticConstraints()
        persona = registry.get_persona("DEBUGGING")
        prompt = persona.system_prompt_template
        violations = constraints.check_text(prompt)
        critical_violations = [
            v for v in violations if v.constraint.severity == ConstraintSeverity.ERROR
        ]
        assert len(critical_violations) == 0

    def test_distillation_preserves_decisions(self):
        distiller = SemanticDistiller()
        messages = [
            MagicMock(content="We decided to use Python 3.11 for this project.", role="user"),
            MagicMock(content="The architecture will be microservices.", role="user"),
            MagicMock(content="Risk: Database scaling could be challenging.", role="user"),
        ]
        matrix = distiller.distill(messages)
        context = distiller.get_active_context()
        context_text = " ".join(context).lower()
        has_decision_or_fact = (
            len(matrix.decisions) > 0
            or len(matrix.facts) > 0
            or "python" in context_text
            or "microservices" in context_text
        )
        assert has_decision_or_fact or len(matrix.pending_risks) > 0

    def test_contrastive_with_distillation(self):
        reasoner = ContrastiveReasoner()
        distiller = SemanticDistiller()
        result = reasoner.reason_about("Should we use microservices architecture?")
        decision_text = f"Decision: {result.final_decision}. Rationale: {result.decision_rationale}"
        messages = [MagicMock(content=decision_text, role="assistant")]
        matrix = distiller.distill(messages)
        assert matrix is not None

    def test_full_interaction_flow(self):
        registry = PersonaRegistry()
        switcher = PersonaSwitcher(registry)
        distiller = SemanticDistiller()
        reasoner = ContrastiveReasoner()
        constraints = SemanticConstraints()
        persona = switcher.switch("DEBUGGING")
        assert persona.name == "Forensic Pathologist"
        prompt = switcher.get_system_prompt(persona)
        assert prompt != ""
        result = reasoner.reason_about("Should we refactor this code?")
        assert result.final_decision != ""
        messages = [
            MagicMock(content=result.final_decision, role="assistant"),
            MagicMock(content=result.decision_rationale, role="assistant"),
        ]
        matrix = distiller.distill(messages)
        assert matrix is not None
        violations = constraints.check_text(result.final_decision)
        for v in violations:
            assert v.constraint.severity != ConstraintSeverity.ERROR

    def test_persona_history_tracking(self):
        switcher = PersonaSwitcher()
        switcher.switch("DEBUGGING")
        switcher.switch("REFACTORING")
        switcher.switch("CODE_GENERATION")
        history = switcher.get_history()
        assert len(history) == 3
        intents = [h[0] for h in history]
        assert intents == ["DEBUGGING", "REFACTORING", "CODE_GENERATION"]

    def test_constraints_pressure_in_prompt(self):
        constraints = SemanticConstraints()
        pressure = constraints.apply_pressure_prompt()
        registry = PersonaRegistry()
        persona = registry.get_persona_by_name("senior_developer")
        assert persona is not None
        combined = f"{persona.system_prompt_template}\n\n{pressure}"
        assert len(combined) > len(persona.system_prompt_template)


class TestEdgeCases:
    """Edge case tests for advanced interaction components."""

    def test_empty_distillation(self):
        distiller = SemanticDistiller()
        messages = [MagicMock(content="", role="user")]
        matrix = distiller.distill(messages)
        assert isinstance(matrix, SemanticMatrix)

    def test_very_long_text_distillation(self):
        distiller = SemanticDistiller(max_facts=5, max_decisions=3)
        long_text = "We decided to use Python. " * 100
        messages = [MagicMock(content=long_text, role="user")]
        matrix = distiller.distill(messages)
        assert len(matrix.facts) <= 5
        assert len(matrix.decisions) <= 3

    def test_special_characters_in_constraints(self):
        constraints = SemanticConstraints()
        text = "Price: $100.00 (50% off!). Email: test@example.com"
        violations = constraints.check_text(text)
        assert isinstance(violations, list)

    def test_unicode_in_distillation(self):
        distiller = SemanticDistiller()
        messages = [
            MagicMock(content="Привет мир! 你好世界! 🌍", role="user"),
            MagicMock(content="日本語テスト", role="user"),
        ]
        matrix = distiller.distill(messages)
        assert isinstance(matrix, SemanticMatrix)

    def test_concurrent_switches(self):
        switcher = PersonaSwitcher()
        for intent in ["DEBUGGING", "REFACTORING", "CODE_GENERATION", "TESTING"]:
            persona = switcher.switch(intent)
            assert persona is not None
        history = switcher.get_history()
        assert len(history) == 4

    def test_constraint_severity_ordering(self):
        constraints = SemanticConstraints()
        text = "TODO: Fix this. We need to ensure robust behavior. This is fast."
        all_violations = constraints.check_text(text)
        errors = [v for v in all_violations if v.constraint.severity == ConstraintSeverity.ERROR]
        warnings = [
            v for v in all_violations if v.constraint.severity == ConstraintSeverity.WARNING
        ]
        assert len(errors) >= 1
        assert len(warnings) >= 2
