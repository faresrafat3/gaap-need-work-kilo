"""
Dream Processor - Nightly Log Consolidation & Learning
======================================================

Analyzes daily execution logs and extracts lessons learned.
Runs during "sleep" cycles to improve future performance.

Usage:
    processor = DreamProcessor()
    lessons = await processor.process_daily_logs()
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gaap.memory import VECTOR_MEMORY_AVAILABLE, LessonStore

logger = logging.getLogger("gaap.dream")


@dataclass
class DreamResult:
    """Result of dreaming cycle"""

    lessons_learned: int
    patterns_found: int
    errors_analyzed: int
    processing_time_ms: float
    lessons: list[str] = field(default_factory=list)


class DreamProcessor:
    """
    Processes execution logs during "sleep" cycles.

    Features:
    - Analyzes successful and failed executions
    - Extracts patterns and lessons
    - Stores lessons in vector memory for semantic retrieval
    - Identifies recurring issues
    """

    def __init__(
        self,
        log_dir: str | None = None,
        lesson_store: LessonStore | None = None,
    ):
        self.log_dir = Path(log_dir or Path.home() / ".gaap" / "logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if not VECTOR_MEMORY_AVAILABLE:
            logger.warning(
                "Vector memory not available. DreamProcessor will use fallback mode. "
                "Install chromadb for full functionality: pip install chromadb"
            )
            self._fallback_mode = True
            self._fallback_lessons: list[dict[str, Any]] = []
        else:
            self._fallback_mode = False

        if self._fallback_mode:
            self._lesson_store = None
        else:
            self._lesson_store = lesson_store or LessonStore()
        self._logger = logger

    async def process_daily_logs(self, days: int = 1) -> DreamResult:
        """
        Process logs and extract lessons.

        Args:
            days: Number of days to look back

        Returns:
            DreamResult with lessons learned
        """
        start_time = time.time()
        lessons = []
        patterns_found = 0
        errors_analyzed = 0

        cutoff = datetime.now() - timedelta(days=days)
        log_files = self._get_log_files(since=cutoff)

        if not log_files:
            self._logger.info("No log files to process")
            return DreamResult(
                lessons_learned=0,
                patterns_found=0,
                errors_analyzed=0,
                processing_time_ms=0,
            )

        for log_file in log_files:
            try:
                entries = self._parse_log_file(log_file)
                for entry in entries:
                    if entry.get("level") == "ERROR":
                        errors_analyzed += 1
                        lesson = self._extract_error_lesson(entry)
                        if lesson:
                            lessons.append(lesson)

                    elif entry.get("success") is True:
                        pattern = self._extract_success_pattern(entry)
                        if pattern:
                            patterns_found += 1

            except Exception as e:
                self._logger.debug(f"Failed to process {log_file}: {e}")

        for lesson in lessons:
            if self._fallback_mode:
                self._fallback_lessons.append(
                    {
                        "lesson": lesson,
                        "context": "dream_processor",
                        "category": "learned",
                        "success": False,
                    }
                )
            else:
                self._lesson_store.store_lesson(
                    lesson=lesson,
                    context="dream_processor",
                    category="learned",
                    success=False,
                )

        processing_time = (time.time() - start_time) * 1000

        self._logger.info(
            f"Dream cycle complete: {len(lessons)} lessons, "
            f"{patterns_found} patterns, {errors_analyzed} errors in {processing_time:.0f}ms"
        )

        return DreamResult(
            lessons_learned=len(lessons),
            patterns_found=patterns_found,
            errors_analyzed=errors_analyzed,
            processing_time_ms=processing_time,
            lessons=lessons,
        )

    async def consolidate_memories(self) -> dict[str, Any]:
        """
        Consolidate short-term memories into long-term lessons.

        This is a higher-level synthesis that combines multiple
        related lessons into general principles.
        """
        if self._fallback_mode:
            all_lessons = self._fallback_lessons
            if len(all_lessons) < 5:
                return {"consolidated": 0, "reason": "Not enough lessons to consolidate"}

            categories: dict[str, list[str]] = {}
            for result in all_lessons:
                cat = result.get("category", "general")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result["lesson"])

            consolidated = 0
            for category, items in categories.items():
                if len(items) >= 3:
                    principle = self._synthesize_principle(items)
                    if principle:
                        self._fallback_lessons.append(
                            {
                                "lesson": f"PRINCIPLE: {principle}",
                                "context": "dream_consolidation",
                                "category": category,
                                "success": True,
                            }
                        )
                        consolidated += 1

            return {
                "consolidated": consolidated,
                "categories_processed": len(categories),
                "fallback_mode": True,
            }

        all_lessons = self._lesson_store.search("", n=100)

        if len(all_lessons) < 5:
            return {"consolidated": 0, "reason": "Not enough lessons to consolidate"}

        categories: dict[str, list[str]] = {}
        for result in all_lessons:
            cat = result.metadata.get("category", "general")
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(result.content)

        consolidated = 0
        for category, items in categories.items():
            if len(items) >= 3:
                principle = self._synthesize_principle(items)
                if principle:
                    self._lesson_store.store_lesson(
                        lesson=f"PRINCIPLE: {principle}",
                        context="dream_consolidation",
                        category=category,
                        success=True,
                    )
                    consolidated += 1

        return {
            "consolidated": consolidated,
            "categories_processed": len(categories),
        }

    def _get_log_files(self, since: datetime) -> list[Path]:
        """Get log files modified since date"""
        files = []
        for f in self.log_dir.glob("*.json"):
            if datetime.fromtimestamp(f.stat().st_mtime) >= since:
                files.append(f)
        return files

    def _parse_log_file(self, filepath: Path) -> list[dict[str, Any]]:
        """Parse JSON log file"""
        entries = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.debug(f"Dream processing error: {e}")
        return entries

    def _extract_error_lesson(self, entry: dict[str, Any]) -> str | None:
        """Extract lesson from error entry"""
        error_msg = entry.get("message", "") or entry.get("error", "")
        task_type = entry.get("task_type", "unknown")

        if not error_msg:
            return None

        if "timeout" in error_msg.lower():
            return f"For {task_type}: Increase timeout or optimize for faster execution"
        elif "rate limit" in error_msg.lower():
            return f"For {task_type}: Implement rate limiting with exponential backoff"
        elif "not found" in error_msg.lower():
            return f"For {task_type}: Validate existence of resources before access"
        elif "permission" in error_msg.lower() or "access denied" in error_msg.lower():
            return f"For {task_type}: Check permissions before attempting operation"
        elif "connection" in error_msg.lower():
            return f"For {task_type}: Implement connection retry with fallback endpoints"

        return None

    def _extract_success_pattern(self, entry: dict[str, Any]) -> str | None:
        """Extract pattern from successful execution"""
        return None

    def _synthesize_principle(self, items: list[str]) -> str | None:
        """Synthesize a general principle from multiple lessons"""
        if len(items) < 3:
            return None

        common_words: dict[str, int] = {}
        for item in items:
            words = set(item.lower().split())
            for word in words:
                if len(word) > 4:
                    common_words[word] = common_words.get(word, 0) + 1

        recurring = [w for w, c in common_words.items() if c >= len(items) // 2]

        if recurring:
            return f"Common themes: {', '.join(recurring[:3])}"

        return None


async def run_dream_cycle() -> DreamResult:
    """Run a complete dream cycle"""
    processor = DreamProcessor()
    return await processor.process_daily_logs()


def run_dream_cycle_sync() -> DreamResult:
    """Synchronous wrapper for dream cycle"""
    return asyncio.run(run_dream_cycle())
