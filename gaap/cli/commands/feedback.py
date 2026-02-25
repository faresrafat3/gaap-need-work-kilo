"""
Feedback Command - User Feedback Collection
============================================

Implements: docs/evolution_plan_2026/27_OPS_AND_CI.md

The `gaap feedback` command allows users to report issues and rate
agent performance, creating negative episodic memories and constraints.

Usage:
    gaap feedback --last-task --rating 1 --comment "Deleted .env file"
    gaap feedback --task-id abc123 --rating 5 --comment "Great response"
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.storage.json_store import get_store


def _get_vector_store() -> Any:
    """Lazy import vector store to avoid import errors."""
    try:
        from gaap.memory.vector_store import get_vector_store

        return get_vector_store()
    except ImportError:
        return None


def cmd_feedback(args: Any) -> None:
    """
    Handle feedback command.

    Flow:
    1. Capture trace/task ID
    2. Create episodic memory entry
    3. Add constraint to VectorMemory (for negative feedback)
    4. Optionally trigger dream cycle
    """
    action = getattr(args, "action", "submit")

    if action == "list":
        cmd_feedback_list(args)
        return
    elif action == "stats":
        cmd_feedback_stats(args)
        return

    store = get_store()
    vector_store = _get_vector_store()

    task_id = _resolve_task_id(args, store)
    if not task_id and getattr(args, "last_task", False):
        print("No recent task found to rate.")
        sys.exit(1)

    rating = getattr(args, "rating", 3)
    comment = getattr(args, "comment", "")
    category = getattr(args, "category", "general")

    feedback_entry = {
        "task_id": task_id or "unknown",
        "rating": rating,
        "comment": comment,
        "category": category,
        "timestamp": datetime.now().isoformat(),
    }

    feedback_id = store.append("feedback", feedback_entry)

    if rating <= 2:
        _handle_negative_feedback(feedback_entry, vector_store, store)

    print(
        f"\n{'✅' if rating >= 4 else '⚠️' if rating >= 3 else '❌'} Feedback recorded: {feedback_id}"
    )

    if rating <= 2 and comment:
        print(f"   Constraint added: '{comment[:50]}...'")

    if rating <= 1:
        print("   Dream cycle recommended: Run 'gaap dream' to consolidate this lesson.")

    _print_feedback_summary(rating, comment)


def _resolve_task_id(args: Any, store: Any) -> str | None:
    """Resolve task ID from arguments or last task."""
    if hasattr(args, "task_id") and args.task_id:
        return str(args.task_id)

    if getattr(args, "last_task", False):
        history = store.load("history", default=[])
        if isinstance(history, list) and history:
            last = history[-1] if isinstance(history[-1], dict) else {}
            return last.get("task_id") or last.get("id")

    return None


def _handle_negative_feedback(
    feedback: dict[str, Any],
    vector_store: Any,
    store: Any,
) -> None:
    """Handle negative feedback by creating constraints."""
    comment = feedback.get("comment", "")
    task_id = feedback.get("task_id", "unknown")

    if not comment:
        return

    if vector_store:
        constraint_text = f"CONSTRAINT: {comment}"
        constraint_id = vector_store.add(
            content=constraint_text,
            metadata={
                "type": "negative_feedback",
                "task_id": task_id,
                "rating": feedback.get("rating", 1),
                "timestamp": feedback.get("timestamp"),
            },
        )
    else:
        constraint_id = f"local_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    constraint_entry = {
        "id": constraint_id,
        "constraint": comment,
        "source": "user_feedback",
        "task_id": task_id,
        "created_at": datetime.now().isoformat(),
    }
    store.append("constraints", constraint_entry)

    _create_negative_episode(feedback, store)


def _create_negative_episode(feedback: dict[str, Any], store: Any) -> None:
    """Create negative episodic memory from feedback."""
    try:
        from gaap.memory.hierarchical import EpisodicMemory, EpisodicMemoryStore

        episodic = EpisodicMemoryStore(storage_path=".gaap/memory")

        episode = EpisodicMemory(
            task_id=feedback.get("task_id", "unknown"),
            action="user_feedback",
            result=feedback.get("comment", ""),
            success=False,
            category="feedback",
            lessons=[f"User reported issue: {feedback.get('comment', '')[:100]}"],
        )

        episodic.record(episode)
        episodic.save()
    except Exception:
        pass


def _print_feedback_summary(rating: int, comment: str) -> None:
    """Print feedback summary."""
    stars = "★" * rating + "☆" * (5 - rating)
    print(f"\n   Rating: {stars} ({rating}/5)")
    if comment:
        print(f"   Comment: {comment}")


def cmd_feedback_list(args: Any) -> None:
    """List feedback entries."""
    store = get_store()

    limit = getattr(args, "limit", 20)
    data = store.load("feedback", default=[])
    feedback_list: list[dict[str, Any]] = data if isinstance(data, list) else []

    if not feedback_list:
        print("No feedback entries found.")
        return

    print(f"\n📋 Feedback Entries ({len(feedback_list)} total):\n")
    print("-" * 60)

    start_idx = max(0, len(feedback_list) - limit)
    for i in range(start_idx, len(feedback_list)):
        entry = feedback_list[i]
        if isinstance(entry, dict):
            rating = entry.get("rating", 0)
            stars = "★" * rating + "☆" * (5 - rating)
            comment = entry.get("comment", "")[:40]
            timestamp = entry.get("timestamp", "")[:10]

            print(f"  [{timestamp}] {stars} | {comment}...")

    print("-" * 60)


def cmd_feedback_stats(args: Any) -> None:
    """Show feedback statistics."""
    store = get_store()

    feedback_list = store.load("feedback", default=[])

    if not feedback_list:
        print("No feedback data available.")
        return

    ratings = [e.get("rating", 0) for e in feedback_list if isinstance(e, dict)]

    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    positive = sum(1 for r in ratings if r >= 4)
    negative = sum(1 for r in ratings if r <= 2)
    neutral = len(ratings) - positive - negative

    print("\n📊 Feedback Statistics:\n")
    print(f"   Total feedback: {len(ratings)}")
    print(f"   Average rating: {avg_rating:.2f}/5")
    print(f"   Positive (4-5): {positive} ({100 * positive / len(ratings):.0f}%)")
    print(f"   Neutral (3):    {neutral} ({100 * neutral / len(ratings):.0f}%)")
    print(f"   Negative (1-2): {negative} ({100 * negative / len(ratings):.0f}%)")

    categories: dict[str, int] = {}
    for entry in feedback_list:
        if isinstance(entry, dict):
            cat = entry.get("category", "general")
            categories[cat] = categories.get(cat, 0) + 1

    if categories:
        print("\n   By category:")
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f"     - {cat}: {count}")
