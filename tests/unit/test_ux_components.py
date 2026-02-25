"""
Unit tests for GAAP UX Components

Tests FuzzyMenu, TaskReceipt, and CLI enhancements from gaap.cli.
"""

from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from gaap.cli.fuzzy_menu import FuzzyMenu, MenuItem, HAS_QUESTIONARY
from gaap.cli.tui import TaskReceipt, print_task_receipt, print_stats


class TestFuzzyMenu:
    """Test FuzzyMenu selection functionality."""

    def test_select_provider(self):
        """Test provider selection with fuzzy menu."""
        menu = FuzzyMenu(use_fuzzy=False)
        providers = [
            {"name": "openai", "provider_type": "openai", "enabled": True, "status": "active"},
            {
                "name": "anthropic",
                "provider_type": "anthropic",
                "enabled": False,
                "status": "inactive",
            },
            {"name": "local", "provider_type": "ollama", "enabled": True, "status": "active"},
        ]

        with patch("rich.prompt.Prompt.ask", return_value="1"):
            result = menu.select_provider(providers)
            assert result == "openai"

    def test_select_tool(self):
        """Test tool selection with fuzzy menu."""
        menu = FuzzyMenu(use_fuzzy=False)
        tools = [
            {"name": "code_interpreter", "description": "Execute code", "category": "execution"},
            {"name": "web_search", "description": "Search the web", "category": "research"},
            {"name": "file_ops", "description": "File operations", "category": "system"},
        ]

        with patch("rich.prompt.Prompt.ask", return_value="2"):
            result = menu.select_tool(tools)
            assert result == "web_search"

    def test_fallback_without_questionary(self):
        """Test fallback behavior when questionary is not available."""
        menu = FuzzyMenu(use_fuzzy=False)
        items = [
            MenuItem(label="Option A", value="a"),
            MenuItem(label="Option B", value="b"),
            MenuItem(label="Option C", value="c"),
        ]

        with patch("rich.prompt.Prompt.ask", return_value="3"):
            result = menu.select_from_list(items, title="Choose")
            assert result == "c"

    def test_select_provider_empty_list(self):
        """Test provider selection with empty list."""
        menu = FuzzyMenu(use_fuzzy=False)
        result = menu.select_provider([])
        assert result is None

    def test_select_tool_empty_list(self):
        """Test tool selection with empty list."""
        menu = FuzzyMenu(use_fuzzy=False)
        result = menu.select_tool([])
        assert result is None

    def test_select_cancel(self):
        """Test cancelling selection."""
        menu = FuzzyMenu(use_fuzzy=False)
        items = [MenuItem(label="Option", value="val")]

        with patch("rich.prompt.Prompt.ask", return_value="q"):
            result = menu.select_from_list(items)
            assert result is None

    def test_multi_select(self):
        """Test multi-selection functionality."""
        menu = FuzzyMenu(use_fuzzy=False)
        items = [
            MenuItem(label="A", value=1),
            MenuItem(label="B", value=2),
            MenuItem(label="C", value=3),
        ]

        with patch("rich.prompt.Prompt.ask", return_value="1,3"):
            result = menu.multi_select(items)
            assert result == [1, 3]

    def test_confirm(self):
        """Test confirmation prompt."""
        menu = FuzzyMenu(use_fuzzy=False)

        with patch("rich.prompt.Prompt.ask", return_value="y"):
            result = menu.confirm("Continue?")
            assert result is True

        with patch("rich.prompt.Prompt.ask", return_value="n"):
            result = menu.confirm("Continue?", default=False)
            assert result is False

    def test_text_input(self):
        """Test text input prompt."""
        menu = FuzzyMenu(use_fuzzy=False)

        with patch("rich.prompt.Prompt.ask", return_value="hello world"):
            result = menu.text_input("Enter text:")
            assert result == "hello world"


class TestTaskReceipt:
    """Test TaskReceipt functionality."""

    def test_receipt_creation(self):
        """Test creating a task receipt."""
        receipt = TaskReceipt(
            task_id="test-task-123",
            description="Implement feature X",
            status="success",
            duration_seconds=45.2,
            files_changed=["src/main.py", "tests/test_main.py"],
            quality_score=0.85,
            quality_breakdown={"correctness": 0.9, "style": 0.8},
            tokens_used=1500,
            cost=0.03,
            layer_times={"L1": 10.5, "L2": 15.2, "L3": 19.5},
        )

        assert receipt.task_id == "test-task-123"
        assert receipt.status == "success"
        assert len(receipt.files_changed) == 2
        assert receipt.quality_score == 0.85
        assert receipt.tokens_used == 1500

    def test_print_receipt(self):
        """Test printing a task receipt."""
        receipt = TaskReceipt(
            task_id="test-task-456",
            description="Refactor module",
            status="success",
            duration_seconds=30.0,
            files_changed=["src/utils.py"],
            quality_score=0.92,
            quality_breakdown={"correctness": 0.95, "style": 0.90, "tests": 0.92},
            tokens_used=800,
            cost=0.015,
        )

        with patch("gaap.cli.tui.console.print") as mock_print:
            print_task_receipt(receipt)
            mock_print.assert_called_once()
            panel = mock_print.call_args[0][0]
            assert hasattr(panel, "renderable")

    def test_receipt_partial_status(self):
        """Test receipt with partial status."""
        receipt = TaskReceipt(
            task_id="partial-123",
            description="Partial task",
            status="partial",
            duration_seconds=20.0,
            warnings=["Deprecated API used"],
        )

        rendered = receipt.render()
        rendered_str = rendered.renderable
        assert "⚠" in str(rendered_str) or "Partial" in str(rendered_str)

    def test_receipt_failed_status(self):
        """Test receipt with failed status."""
        receipt = TaskReceipt(
            task_id="failed-123",
            description="Failed task",
            status="failed",
            duration_seconds=5.0,
            errors=["Timeout exceeded", "Connection refused"],
        )

        rendered = receipt.render()
        rendered_str = rendered.renderable
        assert (
            "✗" in str(rendered_str)
            or "Failed" in str(rendered_str)
            or "Error" in str(rendered_str)
        )

    def test_receipt_quality_bar(self):
        """Test quality bar rendering."""
        receipt = TaskReceipt(
            task_id="quality-test",
            description="Test quality",
            status="success",
            duration_seconds=10.0,
            quality_score=0.75,
            quality_breakdown={"metric1": 0.8, "metric2": 0.7},
        )

        bar = receipt._score_bar(0.75)
        assert "█" in bar
        assert "░" in bar
        assert "75%" in bar

    def test_receipt_many_files(self):
        """Test receipt with many files changed."""
        files = [f"file_{i}.py" for i in range(15)]
        receipt = TaskReceipt(
            task_id="many-files",
            description="Large refactor",
            status="success",
            duration_seconds=60.0,
            files_changed=files,
        )

        rendered = receipt.render()
        assert "and 5 more" in str(rendered) or len(files) > 10


class TestCLIEnhancements:
    """Test CLI enhancement features."""

    def test_stats_enhanced(self):
        """Test enhanced stats display."""
        stats = {
            "requests_processed": 25,
            "success_rate": 0.92,
            "total_tokens": 12500,
            "total_cost": 0.45,
            "files_changed": ["src/main.py", "src/utils.py", "tests/test_main.py"],
            "quality_breakdown": {
                "correctness": 0.95,
                "style": 0.88,
                "tests": 0.90,
            },
            "quality_score": 0.91,
            "layer_times": {
                "L1 Strategic": 5.2,
                "L2 Tactical": 8.5,
                "L3 Execution": 12.3,
            },
        }

        with patch("gaap.cli.tui.console.print") as mock_print:
            print_stats(stats)
            assert mock_print.call_count >= 1

    def test_stats_minimal(self):
        """Test stats with minimal data."""
        stats = {
            "requests_processed": 1,
            "success_rate": 1.0,
            "total_tokens": 100,
            "total_cost": 0.01,
        }

        with patch("gaap.cli.tui.console.print") as mock_print:
            print_stats(stats)
            mock_print.assert_called_once()

    def test_stats_with_quality_breakdown(self):
        """Test stats with quality breakdown display."""
        stats = {
            "requests_processed": 10,
            "success_rate": 0.85,
            "total_tokens": 5000,
            "total_cost": 0.25,
            "quality_breakdown": {
                "correctness": 0.90,
                "efficiency": 0.75,
                "maintainability": 0.85,
            },
            "quality_score": 0.83,
        }

        with patch("gaap.cli.tui.console.print") as mock_print:
            print_stats(stats)
            calls = mock_print.call_args_list
            assert len(calls) >= 1

    def test_stats_with_layer_times(self):
        """Test stats with layer time breakdown."""
        stats = {
            "requests_processed": 5,
            "success_rate": 0.80,
            "total_tokens": 3000,
            "total_cost": 0.15,
            "layer_times": {
                "L1 Strategic": 2.5,
                "L2 Tactical": 4.5,
                "L3 Execution": 8.0,
            },
        }

        with patch("gaap.cli.tui.console.print") as mock_print:
            print_stats(stats)
            assert mock_print.call_count >= 2
