from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class OutcomeAnalysis:
    """تحليل نتيجة مهمة"""

    task_id: str
    success: bool
    score: float
    latency_ms: float
    cost_usd: float
    error: str | None
    lessons: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FailureAnalysis:
    """تحليل الفشل"""

    task_id: str
    failure_reason: str
    severity: str
    root_cause: str
    suggested_fix: str
    similar_failures: int = 0


@dataclass
class SuccessAnalysis:
    """تحليل النجاح"""

    task_id: str
    success_factors: list[str]
    quality_score: float
    optimization_possible: bool = False


class ExperienceAnalyzer:
    """محلل الخبرات - يحلل النتائج ويستخرج الدروس"""

    def __init__(self, success_threshold: float = 70.0):
        self.success_threshold = success_threshold
        self._outcomes: list[OutcomeAnalysis] = []
        self._failure_patterns: dict[str, int] = defaultdict(int)

    def analyze_outcomes(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        """تحليل جميع النتائج"""
        analyses = {
            "successes": [],
            "failures": [],
            "statistics": {},
        }

        for task in history:
            outcome = self._analyze_single_outcome(task)
            self._outcomes.append(outcome)

            if outcome.success:
                analyses["successes"].append(self._analyze_success(outcome))
            else:
                analyses["failures"].append(self._analyze_failure(outcome))

        analyses["statistics"] = self._calculate_statistics()
        analyses["lessons"] = self._extract_lessons()

        return analyses

    def _analyze_single_outcome(self, task: dict[str, Any]) -> OutcomeAnalysis:
        """تحليل نتيجة واحدة"""
        success = task.get("success", False)
        score = task.get("quality_score", 0.0)

        if not success and "error" in task:
            error = task.get("error", "")
            lessons = self._extract_error_lessons(error)
        elif success and score < self.success_threshold:
            lessons = ["Quality below threshold - consider refining prompts"]
        elif success:
            lessons = ["Task completed successfully"]
        else:
            lessons = ["Task failed - check error details"]

        return OutcomeAnalysis(
            task_id=task.get("id", ""),
            success=success,
            score=score,
            latency_ms=task.get("latency_ms", 0.0),
            cost_usd=task.get("cost_usd", 0.0),
            error=task.get("error"),
            lessons=lessons,
            timestamp=task.get("timestamp", datetime.now()),
        )

    def _analyze_success(self, outcome: OutcomeAnalysis) -> SuccessAnalysis:
        """تحليل النجاح"""
        factors = []

        if outcome.score >= 90:
            factors.append("Excellent quality score")
        elif outcome.score >= 80:
            factors.append("Good quality score")

        if outcome.latency_ms < 2000:
            factors.append("Fast response time")
        elif outcome.latency_ms > 10000:
            factors.append("Slow response - may need optimization")

        if outcome.cost_usd < 0.01:
            factors.append("Cost-effective")
        elif outcome.cost_usd > 1.0:
            factors.append("High cost - consider cheaper model")

        return SuccessAnalysis(
            task_id=outcome.task_id,
            success_factors=factors,
            quality_score=outcome.score,
            optimization_possible=outcome.latency_ms > 5000 or outcome.cost_usd > 0.5,
        )

    def _analyze_failure(self, outcome: OutcomeAnalysis) -> FailureAnalysis:
        """تحليل الفشل"""
        if not outcome.error:
            return FailureAnalysis(
                task_id=outcome.task_id,
                failure_reason="Unknown",
                severity="unknown",
                root_cause="No error message provided",
                suggested_fix="Add error logging",
            )

        error = outcome.error.lower()
        reason = "Unknown error"
        severity = "medium"
        root_cause = "Unknown"
        fix = "Review error details"

        if "rate limit" in error or "429" in error:
            reason = "Rate limit exceeded"
            severity = "low"
            root_cause = "Too many requests to provider"
            fix = "Implement exponential backoff, use multiple providers"
            self._failure_patterns["rate_limit"] += 1

        elif "timeout" in error:
            reason = "Request timeout"
            severity = "medium"
            root_cause = "Provider taking too long to respond"
            fix = "Increase timeout or switch to faster provider"
            self._failure_patterns["timeout"] += 1

        elif "auth" in error or "401" in error or "403" in error:
            reason = "Authentication failure"
            severity = "high"
            root_cause = "Invalid or expired API key"
            fix = "Update API credentials"
            self._failure_patterns["auth"] += 1

        elif "not found" in error or "404" in error:
            reason = "Resource not found"
            severity = "low"
            root_cause = "Invalid model or endpoint"
            fix = "Verify model name and endpoint"
            self._failure_patterns["not_found"] += 1

        elif "invalid" in error or "400" in error:
            reason = "Invalid request"
            severity = "medium"
            root_cause = "Malformed request or invalid parameters"
            fix = "Review request format and parameters"
            self._failure_patterns["invalid_request"] += 1

        elif "500" in error or "502" in error or "503" in error:
            reason = "Provider server error"
            severity = "medium"
            root_cause = "Provider internal error"
            fix = "Retry with exponential backoff"
            self._failure_patterns["server_error"] += 1

        elif "model" in error and "not found" in error:
            reason = "Model not available"
            severity = "medium"
            root_cause = "Model not offered by provider"
            fix = "Use alternative model or provider"
            self._failure_patterns["model_not_found"] += 1

        else:
            reason = "Generic error"
            self._failure_patterns["unknown"] += 1

        return FailureAnalysis(
            task_id=outcome.task_id,
            failure_reason=reason,
            severity=severity,
            root_cause=root_cause,
            suggested_fix=fix,
        )

    def _extract_error_lessons(self, error: str) -> list[str]:
        """استخراج دروس من الأخطاء"""
        lessons = []
        error_lower = error.lower()

        if "rate limit" in error_lower:
            lessons.append("Rate limiting detected - implement better backoff strategy")
            lessons.append("Consider adding more providers for load distribution")

        if "timeout" in error_lower:
            lessons.append("Timeout occurred - check provider latency")
            lessons.append("Consider increasing timeout or switching providers")

        if "invalid" in error_lower:
            lessons.append("Invalid request - review prompt format")
            lessons.append("Check for special characters or formatting issues")

        if "context" in error_lower and "length" in error_lower:
            lessons.append("Context length exceeded - simplify prompt or use truncation")
            lessons.append("Consider breaking task into smaller parts")

        return lessons

    def _extract_lessons(self) -> list[str]:
        """استخراج دروس عامة من جميع النتائج"""
        lessons = []

        if not self._outcomes:
            return lessons

        success_count = sum(1 for o in self._outcomes if o.success)
        total = len(self._outcomes)
        success_rate = success_count / total if total > 0 else 0

        if success_rate < 0.5:
            lessons.append("Low success rate - review error patterns and provider configuration")

        if success_rate > 0.9:
            lessons.append("High success rate - current configuration is working well")

        if self._failure_patterns:
            top_failure = max(self._failure_patterns.items(), key=lambda x: x[1])
            lessons.append(f"Most common issue: {top_failure[0]} ({top_failure[1]} occurrences)")

        avg_cost = sum(o.cost_usd for o in self._outcomes) / total if total > 0 else 0
        if avg_cost > 0.5:
            lessons.append("High average cost - consider using cheaper models for simple tasks")

        return lessons

    def _calculate_statistics(self) -> dict[str, Any]:
        """حساب الإحصائيات"""
        if not self._outcomes:
            return {}

        success_count = sum(1 for o in self._outcomes if o.success)
        total = len(self._outcomes)

        return {
            "total_tasks": total,
            "successes": success_count,
            "failures": total - success_count,
            "success_rate": success_count / total if total > 0 else 0,
            "avg_latency_ms": sum(o.latency_ms for o in self._outcomes) / total,
            "avg_cost_usd": sum(o.cost_usd for o in self._outcomes) / total,
            "avg_quality_score": sum(o.score for o in self._outcomes) / total,
            "failure_patterns": dict(self._failure_patterns),
        }

    def get_recommendations(self) -> list[str]:
        """الحصول على توصيات للتحسين"""
        recommendations = []
        stats = self._calculate_statistics()

        if stats.get("success_rate", 0) < 0.7:
            recommendations.append("Consider implementing more robust error handling")

        if stats.get("avg_cost_usd", 0) > 0.3:
            recommendations.append("Use cheaper models for simple tasks to reduce costs")

        if stats.get("avg_latency_ms", 0) > 8000:
            recommendations.append("Consider faster providers or caching for common queries")

        for failure, count in self._failure_patterns.items():
            if count >= 3:
                recommendations.append(f"Address recurring {failure} issues ({count} times)")

        return recommendations
