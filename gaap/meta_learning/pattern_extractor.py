from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Pattern:
    """نمط مكتشف"""

    id: str
    pattern_type: str
    trigger: str
    frequency: int
    success_rate: float
    sample_tasks: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    discovered_at: datetime = field(default_factory=datetime.now)


class PatternExtractor:
    """مستخرج الأنماط من التاريخ"""

    def __init__(self, min_occurrences: int = 3):
        self.min_occurrences = min_occurrences
        self._patterns: dict[str, Pattern] = {}
        self._task_counter: Counter = Counter()
        self._success_by_category: defaultdict[str, list[bool]] = defaultdict(list)

    def analyze_tasks(self, task_history: list[dict[str, Any]]) -> list[Pattern]:
        """تحليل تاريخ المهام واستخراج الأنماط"""
        patterns_found: list[Pattern] = []

        for task in task_history:
            task_type = task.get("type", "unknown")
            success = task.get("success", False)
            model = task.get("model", "")
            provider = task.get("provider", "")
            prompt_type = task.get("prompt_type", "")

            self._task_counter[task_type] += 1
            self._success_by_category[task_type].append(success)

            if provider and model:
                provider_model_key = f"{provider}:{model}"
                self._success_by_category[provider_model_key].append(success)

            if prompt_type:
                self._success_by_category[f"prompt:{prompt_type}"].append(success)

        self._extract_type_patterns(patterns_found)
        self._extract_provider_patterns(patterns_found)
        self._extract_prompt_patterns(patterns_found)
        self._extract_error_patterns(task_history, patterns_found)

        return patterns_found

    def _extract_type_patterns(self, patterns: list[Pattern]) -> None:
        """استخراج أنماط أنواع المهام"""
        for task_type, successes in self._success_by_category.items():
            if ":" in task_type:
                continue

            count = len(successes)
            if count >= self.min_occurrences:
                success_rate = sum(successes) / count

                pattern = Pattern(
                    id=f"type_{task_type}",
                    pattern_type="task_type",
                    trigger=task_type,
                    frequency=count,
                    success_rate=success_rate,
                    metadata={"category": "task_type"},
                )
                patterns.append(pattern)
                self._patterns[pattern.id] = pattern

    def _extract_provider_patterns(self, patterns: list[Pattern]) -> None:
        """استخراج أنماط المزودين"""
        for key, successes in self._success_by_category.items():
            if not key.startswith("provider:") and ":" not in key:
                continue

            if key.startswith("provider:"):
                continue

            parts = key.split(":")
            if len(parts) >= 2:
                provider = parts[0]
                model = ":".join(parts[1:])

                count = len(successes)
                if count >= self.min_occurrences:
                    success_rate = sum(successes) / count

                    pattern = Pattern(
                        id=f"provider_{provider}_{model}",
                        pattern_type="provider_model",
                        trigger=f"{provider}/{model}",
                        frequency=count,
                        success_rate=success_rate,
                        metadata={"provider": provider, "model": model},
                    )
                    patterns.append(pattern)
                    self._patterns[pattern.id] = pattern

    def _extract_prompt_patterns(self, patterns: list[Pattern]) -> None:
        """استخراج أنماط الـ prompts"""
        for key, successes in self._success_by_category.items():
            if not key.startswith("prompt:"):
                continue

            prompt_type = key.split(":", 1)[1]
            count = len(successes)

            if count >= self.min_occurrences:
                success_rate = sum(successes) / count

                pattern = Pattern(
                    id=f"prompt_{prompt_type}",
                    pattern_type="prompt_style",
                    trigger=prompt_type,
                    frequency=count,
                    success_rate=success_rate,
                    metadata={"prompt_type": prompt_type},
                )
                patterns.append(pattern)
                self._patterns[pattern.id] = pattern

    def _extract_error_patterns(
        self, task_history: list[dict[str, Any]], patterns: list[Pattern]
    ) -> None:
        """استخراج أنماط الأخطاء"""
        error_counter: Counter = Counter()
        error_success: dict[str, list[bool]] = defaultdict(list)

        for task in task_history:
            error = task.get("error", "")
            if error:
                error_type = self._classify_error(error)
                error_counter[error_type] += 1
                error_success[error_type].append(task.get("success", False))

        for error_type, count in error_counter.items():
            if count >= self.min_occurrences:
                successes = error_success[error_type]
                success_rate = 1.0 - (sum(successes) / len(successes))

                pattern = Pattern(
                    id=f"error_{error_type}",
                    pattern_type="error_recovery",
                    trigger=error_type,
                    frequency=count,
                    success_rate=success_rate,
                    metadata={"error_type": error_type, "failure_rate": success_rate},
                )
                patterns.append(pattern)
                self._patterns[pattern.id] = pattern

    def _classify_error(self, error: str) -> str:
        """تصنيف الخطأ"""
        error_lower = error.lower()

        if "rate limit" in error_lower or "429" in error_lower:
            return "rate_limit"
        elif "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "auth" in error_lower or "401" in error_lower or "403" in error_lower:
            return "authentication"
        elif "not found" in error_lower or "404" in error_lower:
            return "not_found"
        elif "invalid" in error_lower or "400" in error_lower:
            return "invalid_request"
        elif "500" in error_lower or "502" in error_lower or "503" in error_lower:
            return "server_error"
        else:
            return "unknown"

    def get_best_provider_for_task(self, task_type: str) -> tuple[str, float] | None:
        """الحصول على أفضل مزود لنوع مهمة"""
        best = None
        best_rate = 0.0

        for pattern in self._patterns.values():
            if pattern.pattern_type == "provider_model" and task_type in pattern.metadata:
                if pattern.success_rate > best_rate:
                    best_rate = pattern.success_rate
                    best = pattern.trigger

        return (best, best_rate) if best else None

    def get_patterns_by_type(self, pattern_type: str) -> list[Pattern]:
        """الحصول على أنماط من نوع معين"""
        return [p for p in self._patterns.values() if p.pattern_type == pattern_type]

    def get_high_success_patterns(self, min_rate: float = 0.8) -> list[Pattern]:
        """الحصول على الأنماط ذات نسبة النجاح العالية"""
        return [p for p in self._patterns.values() if p.success_rate >= min_rate]

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "total_patterns": len(self._patterns),
            "by_type": Counter(p.pattern_type for p in self._patterns.values()),
            "avg_success_rate": sum(p.success_rate for p in self._patterns.values())
            / max(len(self._patterns), 1),
            "high_success_patterns": len(self.get_high_success_patterns()),
        }
