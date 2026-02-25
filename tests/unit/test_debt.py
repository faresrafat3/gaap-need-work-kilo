"""
Tests for Technical Debt Agent
================================

Tests for debt scanning, interest calculation, and refinancing.

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from gaap.maintenance import (
    DebtConfig,
    DebtItem,
    DebtPriority,
    DebtScanner,
    DebtType,
    InterestCalculator,
    InterestFactors,
    ProposalStatus,
    RefinancingEngine,
    RefactoringProposal,
    ScanResult,
    create_debt_config,
)


class TestDebtConfig:
    """Tests for DebtConfig."""

    def test_default_config(self):
        config = DebtConfig()

        assert config.enabled is True
        assert config.complexity_warning == 10
        assert config.complexity_critical == 15
        assert "TODO" in config.markers
        assert "FIXME" in config.markers

    def test_conservative_preset(self):
        config = DebtConfig.conservative()

        assert config.complexity_warning == 15
        assert config.complexity_critical == 20
        assert len(config.markers) == 2

    def test_aggressive_preset(self):
        config = DebtConfig.aggressive()

        assert config.complexity_warning == 8
        assert config.complexity_critical == 12
        assert len(config.markers) >= 5

    def test_development_preset(self):
        config = DebtConfig.development()

        assert config.llm_enabled is True
        assert config.complexity_warning == 10

    def test_config_to_dict_and_from_dict(self):
        config = DebtConfig.aggressive()
        data = config.to_dict()

        restored = DebtConfig.from_dict(data)

        assert restored.complexity_warning == config.complexity_warning
        assert restored.markers == config.markers

    def test_create_debt_config_factory(self):
        config = create_debt_config("conservative")
        assert config.complexity_warning == 15

        config = create_debt_config("default", complexity_warning=5)
        assert config.complexity_warning == 5


class TestDebtScanner:
    """Tests for DebtScanner."""

    def test_scan_simple_file(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("""
# TODO: This needs to be implemented
# FIXME: This is broken
def hello():
    pass  # This function does nothing
""")

        scanner = DebtScanner()
        items = scanner.scan_file(test_file)

        todo_items = [i for i in items if i.type == DebtType.TODO]
        fixme_items = [i for i in items if i.type == DebtType.FIXME]

        assert len(todo_items) == 1
        assert len(fixme_items) == 1
        assert "implemented" in todo_items[0].message.lower()

    def test_scan_complexity(self, tmp_path):
        test_file = tmp_path / "complex.py"
        content = """
def complex_function(a, b, c, d, e, f, g):
    if a:
        if b:
            if c:
                return 1
            elif d:
                return 2
            else:
                return 3
        elif e:
            return 4
        else:
            return 5
    else:
        return 0
"""
        test_file.write_text(content)

        config = DebtConfig(complexity_warning=5)
        scanner = DebtScanner(config=config)
        items = scanner.scan_file(test_file)

        complexity_items = [i for i in items if i.type == DebtType.COMPLEXITY]

        assert len(complexity_items) >= 1

    def test_scan_long_function(self, tmp_path):
        test_file = tmp_path / "long.py"
        lines = ["    x = 1"] * 60
        test_file.write_text("\n".join(["def long_function():"] + lines))

        config = DebtConfig(long_function_lines=50)
        scanner = DebtScanner(config=config)
        items = scanner.scan_file(test_file)

        long_items = [i for i in items if i.type == DebtType.LONG_FUNCTION]

        assert len(long_items) >= 1

    def test_scan_directory(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "module.py").write_text("# TODO: Module todo\npass")
        (tmp_path / "src" / "utils.py").write_text("# FIXME: Utility fixme\npass")

        scanner = DebtScanner()
        result = scanner.scan_directory(tmp_path)

        assert result.scanned_files >= 2
        assert result.total_debt_items >= 2

    def test_scan_result_to_dict(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("# TODO: Test\npass")

        scanner = DebtScanner()
        result = scanner.scan_directory(tmp_path)

        data = result.to_dict()

        assert "scanned_files" in data
        assert "total_debt_items" in data
        assert "items" in data


class TestInterestCalculator:
    """Tests for InterestCalculator."""

    def test_calculate_interest(self):
        config = DebtConfig()
        calculator = InterestCalculator(config=config)

        debt = DebtItem(
            id="test-1",
            type=DebtType.FIXME,
            file_path="auth/security.py",
            line_number=10,
            end_line=10,
            message="Critical security issue",
            priority=DebtPriority.CRITICAL,
        )

        interest = calculator.calculate(debt)

        assert 0.0 <= interest <= 1.0
        assert interest > 0.5

    def test_prioritize_debts(self):
        calculator = InterestCalculator()

        debts = [
            DebtItem(
                id=f"debt-{i}",
                type=DebtType.TODO,
                file_path=f"file{i}.py",
                line_number=i,
                end_line=i,
                message=f"Item {i}",
                priority=DebtPriority.MEDIUM,
            )
            for i in range(10)
        ]

        prioritized = calculator.prioritize(debts)

        assert len(prioritized) == 10

    def test_interest_factors(self):
        factors = InterestFactors(
            file_criticality=0.8,
            code_age_days=100,
            reference_count=5,
            test_coverage=0.3,
        )

        data = factors.to_dict()

        assert data["file_criticality"] == 0.8
        assert data["code_age_days"] == 100

    def test_calculate_criticality(self):
        calculator = InterestCalculator()

        assert calculator._calculate_criticality("auth/login.py") >= 0.8
        assert calculator._calculate_criticality("tests/test_utils.py") <= 0.5
        assert calculator._calculate_criticality("utils/helpers/format.py") <= 0.5


class TestRefinancingEngine:
    """Tests for RefinancingEngine."""

    def test_create_proposal(self, tmp_path):
        config = DebtConfig(storage_path=str(tmp_path / "debt"))
        engine = RefinancingEngine(config=config, project_root=tmp_path)

        debt = DebtItem(
            id="test-debt-1",
            type=DebtType.TODO,
            file_path="test.py",
            line_number=10,
            end_line=10,
            message="Test todo",
            priority=DebtPriority.MEDIUM,
        )

        import asyncio

        proposal = asyncio.run(engine.propose(debt, use_llm=False))

        assert proposal.id is not None
        assert proposal.debt_item.id == debt.id
        assert proposal.status == ProposalStatus.PENDING

    def test_proposal_to_dict_and_from_dict(self, tmp_path):
        config = DebtConfig(storage_path=str(tmp_path / "debt"))
        engine = RefinancingEngine(config=config, project_root=tmp_path)

        debt = DebtItem(
            id="test-debt-2",
            type=DebtType.FIXME,
            file_path="fix.py",
            line_number=5,
            end_line=5,
            message="Fix this",
            priority=DebtPriority.HIGH,
        )

        import asyncio

        proposal = asyncio.run(engine.propose(debt, use_llm=False))

        data = proposal.to_dict()
        restored = RefactoringProposal.from_dict(data)

        assert restored.id == proposal.id
        assert restored.debt_item.id == debt.id

    def test_list_proposals(self, tmp_path):
        config = DebtConfig(storage_path=str(tmp_path / "debt"))
        engine = RefinancingEngine(config=config, project_root=tmp_path)

        debt1 = DebtItem(
            id="debt-1",
            type=DebtType.TODO,
            file_path="a.py",
            line_number=1,
            end_line=1,
            message="Todo 1",
            priority=DebtPriority.LOW,
        )
        debt2 = DebtItem(
            id="debt-2",
            type=DebtType.FIXME,
            file_path="b.py",
            line_number=1,
            end_line=1,
            message="Fixme 1",
            priority=DebtPriority.HIGH,
        )

        import asyncio

        asyncio.run(engine.propose(debt1, use_llm=False))
        asyncio.run(engine.propose(debt2, use_llm=False))

        proposals = engine.list_proposals()

        assert len(proposals) == 2

    def test_update_proposal_status(self, tmp_path):
        config = DebtConfig(storage_path=str(tmp_path / "debt"))
        engine = RefinancingEngine(config=config, project_root=tmp_path)

        debt = DebtItem(
            id="debt-status",
            type=DebtType.TODO,
            file_path="c.py",
            line_number=1,
            end_line=1,
            message="Status test",
            priority=DebtPriority.MEDIUM,
        )

        import asyncio

        proposal = asyncio.run(engine.propose(debt, use_llm=False))

        updated = engine.update_status(
            proposal.id,
            ProposalStatus.APPROVED,
            notes="Looks good",
        )

        assert updated is not None
        assert updated.status == ProposalStatus.APPROVED
        assert updated.review_notes == "Looks good"


class TestDebtItem:
    """Tests for DebtItem."""

    def test_debt_item_creation(self):
        item = DebtItem(
            id="test-item",
            type=DebtType.TODO,
            file_path="test.py",
            line_number=10,
            end_line=10,
            message="Test message",
            priority=DebtPriority.HIGH,
        )

        assert item.id == "test-item"
        assert item.type == DebtType.TODO
        assert item.priority == DebtPriority.HIGH

    def test_debt_item_hash_and_equality(self):
        item1 = DebtItem(
            id="same-id",
            type=DebtType.TODO,
            file_path="a.py",
            line_number=1,
            end_line=1,
            message="Test",
        )
        item2 = DebtItem(
            id="same-id",
            type=DebtType.TODO,
            file_path="a.py",
            line_number=1,
            end_line=1,
            message="Test",
        )

        assert hash(item1) == hash(item2)
        assert item1 == item2

    def test_debt_item_to_dict_and_from_dict(self):
        item = DebtItem(
            id="test-dict",
            type=DebtType.COMPLEXITY,
            file_path="complex.py",
            line_number=50,
            end_line=100,
            message="Complex function",
            priority=DebtPriority.CRITICAL,
            complexity=25,
            interest_score=0.8,
        )

        data = item.to_dict()
        restored = DebtItem.from_dict(data)

        assert restored.id == item.id
        assert restored.type == item.type
        assert restored.complexity == 25
        assert restored.interest_score == 0.8
