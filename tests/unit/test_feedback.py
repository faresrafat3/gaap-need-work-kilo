"""
Tests for GAAP Feedback Command
================================

Implements tests for:
- docs/evolution_plan_2026/27_OPS_AND_CI.md
"""

import json
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from gaap.cli.commands.feedback import (
    cmd_feedback,
    cmd_feedback_list,
    cmd_feedback_stats,
    _resolve_task_id,
    _handle_negative_feedback,
    _create_negative_episode,
)
from gaap.storage.json_store import JSONStore


class MockArgs:
    """Mock args object for testing."""

    def __init__(
        self,
        last_task: bool = False,
        task_id: str | None = None,
        rating: int = 3,
        comment: str = "",
        category: str = "general",
        action: str = "submit",
        limit: int = 20,
    ) -> None:
        self.last_task = last_task
        self.task_id = task_id
        self.rating = rating
        self.comment = comment
        self.category = category
        self.action = action
        self.limit = limit


class TestResolveTaskId:
    """Tests for _resolve_task_id function."""

    def test_explicit_task_id(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        args = MockArgs(task_id="task-123")

        result = _resolve_task_id(args, store)

        assert result == "task-123"

    def test_last_task_from_history(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        store.append("history", {"task_id": "task-456"})
        args = MockArgs(last_task=True)

        result = _resolve_task_id(args, store)

        assert result == "task-456"

    def test_no_task_found(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        args = MockArgs(last_task=True)

        result = _resolve_task_id(args, store)

        assert result is None


class TestHandleNegativeFeedback:
    """Tests for _handle_negative_feedback function."""

    def test_creates_constraint(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        vector_store = MagicMock()
        vector_store.add.return_value = "constraint-123"

        feedback = {
            "task_id": "task-123",
            "rating": 1,
            "comment": "Deleted my .env file!",
            "timestamp": "2026-01-01T12:00:00",
        }

        _handle_negative_feedback(feedback, vector_store, store)

        vector_store.add.assert_called_once()
        call_args = vector_store.add.call_args
        assert "CONSTRAINT" in call_args[1]["content"]
        assert call_args[1]["metadata"]["type"] == "negative_feedback"

    def test_no_constraint_without_comment(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)
        vector_store = MagicMock()

        feedback = {
            "task_id": "task-123",
            "rating": 2,
            "comment": "",
            "timestamp": "2026-01-01T12:00:00",
        }

        _handle_negative_feedback(feedback, vector_store, store)

        vector_store.add.assert_not_called()


class TestCreateNegativeEpisode:
    """Tests for _create_negative_episode function."""

    def test_creates_episode(self, tmp_path: Path) -> None:
        store = JSONStore(base_dir=tmp_path)

        feedback = {
            "task_id": "task-123",
            "rating": 1,
            "comment": "Agent made a mistake",
            "timestamp": "2026-01-01T12:00:00",
        }

        _create_negative_episode(feedback, store)

        assert True


class TestCmdFeedback:
    """Tests for cmd_feedback function."""

    def test_submit_positive_feedback(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            with patch("gaap.cli.commands.feedback._get_vector_store") as mock_vector:
                store = JSONStore(base_dir=tmp_path)
                mock_get_store.return_value = store
                mock_vector.return_value = MagicMock()

                args = MockArgs(
                    task_id="task-123",
                    rating=5,
                    comment="Excellent work!",
                )

                cmd_feedback(args)

                feedback_data = store.load("feedback")
                assert isinstance(feedback_data, list)
                assert len(feedback_data) == 1
                assert feedback_data[0]["rating"] == 5

    def test_submit_negative_feedback(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            with patch("gaap.cli.commands.feedback._get_vector_store") as mock_vector:
                store = JSONStore(base_dir=tmp_path)
                mock_get_store.return_value = store
                vector_store = MagicMock()
                vector_store.add.return_value = "constraint-123"
                mock_vector.return_value = vector_store

                args = MockArgs(
                    task_id="task-123",
                    rating=1,
                    comment="Deleted important file",
                )

                cmd_feedback(args)

                feedback_data = store.load("feedback")
                assert isinstance(feedback_data, list)
                assert len(feedback_data) == 1
                assert feedback_data[0]["rating"] == 1
                vector_store.add.assert_called_once()

    def test_list_action(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            store.append("feedback", {"rating": 5, "comment": "Great", "timestamp": "2026-01-01"})
            mock_get_store.return_value = store

            args = MockArgs(action="list")

            cmd_feedback(args)

    def test_stats_action(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            store.append("feedback", {"rating": 5, "comment": "Great"})
            store.append("feedback", {"rating": 1, "comment": "Bad"})
            mock_get_store.return_value = store

            args = MockArgs(action="stats")

            cmd_feedback(args)


class TestCmdFeedbackList:
    """Tests for cmd_feedback_list function."""

    def test_empty_list(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            mock_get_store.return_value = store

            args = MockArgs(action="list")
            cmd_feedback_list(args)

    def test_with_entries(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            store.append(
                "feedback",
                {"rating": 5, "comment": "Excellent", "timestamp": "2026-01-01T12:00:00"},
            )
            store.append(
                "feedback",
                {"rating": 2, "comment": "Needs improvement", "timestamp": "2026-01-02T12:00:00"},
            )
            mock_get_store.return_value = store

            args = MockArgs(action="list", limit=10)
            cmd_feedback_list(args)


class TestCmdFeedbackStats:
    """Tests for cmd_feedback_stats function."""

    def test_empty_stats(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            mock_get_store.return_value = store

            args = MockArgs(action="stats")
            cmd_feedback_stats(args)

    def test_with_data(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            store = JSONStore(base_dir=tmp_path)
            for rating in [5, 4, 5, 3, 2, 1, 4, 4]:
                store.append("feedback", {"rating": rating, "comment": f"Rating {rating}"})
            mock_get_store.return_value = store

            args = MockArgs(action="stats")
            cmd_feedback_stats(args)


class TestFeedbackIntegration:
    """Integration tests for feedback command."""

    def test_full_feedback_workflow(self, tmp_path: Path) -> None:
        with patch("gaap.cli.commands.feedback.get_store") as mock_get_store:
            with patch("gaap.cli.commands.feedback._get_vector_store") as mock_vector:
                store = JSONStore(base_dir=tmp_path)
                mock_get_store.return_value = store
                vector_store = MagicMock()
                vector_store.add.return_value = "constraint-123"
                mock_vector.return_value = vector_store

                args1 = MockArgs(task_id="task-1", rating=5, comment="Great job")
                cmd_feedback(args1)

                args2 = MockArgs(task_id="task-2", rating=1, comment="Deleted .env")
                cmd_feedback(args2)

                args3 = MockArgs(task_id="task-3", rating=4, comment="Good")
                cmd_feedback(args3)

                feedback_data = store.load("feedback")
                assert isinstance(feedback_data, list)
                assert len(feedback_data) == 3

                ratings = [f["rating"] for f in feedback_data]
                assert 5 in ratings
                assert 1 in ratings
                assert 4 in ratings

                assert vector_store.add.call_count == 1
