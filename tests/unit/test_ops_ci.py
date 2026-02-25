"""
Tests for Ops & CI Components
=============================

Tests for:
- CostMonitor
- Feedback command
- Adversarial scenarios

Implements: docs/evolution_plan_2026/27_OPS_AND_CI.md
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestCostMonitor:
    """Tests for CostMonitor."""

    def test_cost_monitor_creation(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))
        assert monitor._records == []

    def test_record_usage(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        record = monitor.record(
            provider="openai",
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=50,
            task_id="test-123",
        )

        assert record.provider == "openai"
        assert record.model == "gpt-4o-mini"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.cost_usd > 0
        assert len(monitor._records) == 1

    def test_calculate_cost_gpt4o_mini(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        cost = monitor._calculate_cost("gpt-4o-mini", 1_000_000, 1_000_000)

        assert cost > 0
        assert cost < 1.0

    def test_generate_daily_report(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        for i in range(5):
            monitor.record(
                provider="openai",
                model="gpt-4o-mini",
                input_tokens=100 * (i + 1),
                output_tokens=50 * (i + 1),
            )

        report = monitor.generate_report("daily")

        assert report.total_calls == 5
        assert report.total_input_tokens == 1500
        assert report.total_output_tokens == 750
        assert report.total_cost_usd > 0

    def test_generate_weekly_report(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        monitor.record("openai", "gpt-4o-mini", 1000, 500)
        monitor.record("anthropic", "claude-3-haiku", 2000, 1000)

        report = monitor.generate_report("weekly")

        assert report.total_calls == 2
        assert "openai" in report.by_provider
        assert "anthropic" in report.by_provider

    def test_check_alerts_below_threshold(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        monitor.record("openai", "gpt-4o-mini", 100, 50)

        alerts = monitor.check_alerts(threshold=1.0)

        assert len(alerts) == 0

    def test_check_alerts_above_threshold(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        for _ in range(100):
            monitor.record("openai", "gpt-4o", 10000, 5000)

        alerts = monitor.check_alerts(threshold=0.01)

        assert len(alerts) > 0
        assert "DAILY_COST_ALERT" in alerts[0]

    def test_export(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        monitor.record("openai", "gpt-4o-mini", 100, 50)

        export_path = tmp_path / "export.json"
        result_path = monitor.export(str(export_path))

        assert result_path.exists()

        with open(result_path) as f:
            data = json.load(f)

        assert data["total_records"] == 1
        assert "records" in data
        assert "summary" in data

    def test_persistence(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor1 = CostMonitor(storage_path=str(tmp_path))
        monitor1.record("openai", "gpt-4o-mini", 100, 50)

        monitor2 = CostMonitor(storage_path=str(tmp_path))
        assert len(monitor2._records) == 1


class TestFeedbackCommand:
    """Tests for feedback command."""

    def test_feedback_submit(self, tmp_path):
        from gaap.cli.commands.feedback import cmd_feedback

        args = MagicMock()
        args.action = "submit"
        args.last_task = False
        args.task_id = "test-task-123"
        args.rating = 4
        args.comment = "Great work!"
        args.category = "general"

        with patch("gaap.cli.commands.feedback.get_store") as mock_store:
            mock_store.return_value = MagicMock()
            mock_store.return_value.append.return_value = "feedback-123"
            mock_store.return_value.load.return_value = []

            with patch("gaap.cli.commands.feedback._get_vector_store") as mock_vs:
                mock_vs.return_value = None

                with patch("gaap.cli.commands.feedback._print_feedback_summary"):
                    cmd_feedback(args)

    def test_feedback_negative_creates_constraint(self, tmp_path):
        from gaap.cli.commands.feedback import cmd_feedback, _handle_negative_feedback

        feedback = {
            "task_id": "test-123",
            "rating": 1,
            "comment": "Deleted important file",
            "timestamp": datetime.now().isoformat(),
        }

        mock_store = MagicMock()
        mock_vs = MagicMock()
        mock_vs.add.return_value = "constraint-123"

        _handle_negative_feedback(feedback, mock_vs, mock_store)

        mock_vs.add.assert_called_once()
        mock_store.append.assert_called()

    def test_feedback_list(self, tmp_path):
        from gaap.cli.commands.feedback import cmd_feedback_list

        args = MagicMock()
        args.limit = 10

        mock_store = MagicMock()
        mock_store.load.return_value = [
            {"rating": 4, "comment": "Good", "timestamp": "2026-01-01T00:00:00"},
            {"rating": 2, "comment": "Bad", "timestamp": "2026-01-02T00:00:00"},
        ]

        with patch("gaap.cli.commands.feedback.get_store", return_value=mock_store):
            cmd_feedback_list(args)

    def test_feedback_stats(self, tmp_path):
        from gaap.cli.commands.feedback import cmd_feedback_stats

        args = MagicMock()

        mock_store = MagicMock()
        mock_store.load.return_value = [
            {"rating": 5, "comment": "Excellent", "category": "code"},
            {"rating": 4, "comment": "Good", "category": "code"},
            {"rating": 2, "comment": "Bad", "category": "general"},
            {"rating": 3, "comment": "OK", "category": "general"},
        ]

        with patch("gaap.cli.commands.feedback.get_store", return_value=mock_store):
            cmd_feedback_stats(args)


class TestAdversarialScenarios:
    """Tests for adversarial scenarios loading and validation."""

    def test_load_scenarios(self):
        scenarios_path = Path(__file__).parent.parent / "scenarios" / "adversarial_cases.json"

        if not scenarios_path.exists():
            pytest.skip("adversarial_cases.json not found")

        with open(scenarios_path) as f:
            data = json.load(f)

        assert "scenarios" in data
        assert len(data["scenarios"]) > 0

    def test_scenario_structure(self):
        scenarios_path = Path(__file__).parent.parent / "scenarios" / "adversarial_cases.json"

        if not scenarios_path.exists():
            pytest.skip("adversarial_cases.json not found")

        with open(scenarios_path) as f:
            data = json.load(f)

        for scenario in data["scenarios"]:
            assert "id" in scenario
            assert "name" in scenario
            assert "task" in scenario
            assert "expected_behavior" in scenario
            assert "pass_criteria" in scenario
            assert "severity" in scenario
            assert "category" in scenario

    def test_categories_exist(self):
        scenarios_path = Path(__file__).parent.parent / "scenarios" / "adversarial_cases.json"

        if not scenarios_path.exists():
            pytest.skip("adversarial_cases.json not found")

        with open(scenarios_path) as f:
            data = json.load(f)

        categories = data.get("categories", {})

        for scenario in data["scenarios"]:
            assert scenario["category"] in categories

    def test_critical_scenarios_count(self):
        scenarios_path = Path(__file__).parent.parent / "scenarios" / "adversarial_cases.json"

        if not scenarios_path.exists():
            pytest.skip("adversarial_cases.json not found")

        with open(scenarios_path) as f:
            data = json.load(f)

        critical = [s for s in data["scenarios"] if s["severity"] == "critical"]
        assert len(critical) >= 3


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_cost_guardrail_check(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        monitor.record("openai", "gpt-4o-mini", 1000, 500)

        report = monitor.generate_report("daily")

        assert report.avg_cost_per_call < 0.05, "Cost should be under threshold"

    def test_multiple_providers_tracking(self, tmp_path):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
        from cost_monitor import CostMonitor

        monitor = CostMonitor(storage_path=str(tmp_path))

        providers = [
            ("openai", "gpt-4o-mini", 1000, 500),
            ("anthropic", "claude-3-haiku", 2000, 1000),
            ("google", "gemini-1.5-flash", 1500, 750),
        ]

        for provider, model, inp, out in providers:
            monitor.record(provider, model, inp, out)

        report = monitor.generate_report("daily")

        assert len(report.by_provider) == 3
        assert report.total_calls == 3
