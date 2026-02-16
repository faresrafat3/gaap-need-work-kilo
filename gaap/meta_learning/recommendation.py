from dataclasses import dataclass, field
from typing import Any


@dataclass
class Recommendation:
    """توصية"""

    id: str
    category: str
    title: str
    description: str
    priority: str
    confidence: float
    action: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RecommendationEngine:
    """محرك التوصيات - يولد توصيات بناءً على التحليل"""

    def __init__(self) -> None:
        self._recommendations: list[Recommendation] = []

    def generate_recommendations(
        self,
        patterns: list[Any],
        analyses: dict[str, Any],
        procedures: list[Any],
    ) -> list[Recommendation]:
        """توليد توصيات من التحليلات"""
        recommendations = []

        recommendations.extend(self._provider_recommendations(patterns, analyses))
        recommendations.extend(self._prompt_recommendations(patterns, analyses))
        recommendations.extend(self._optimization_recommendations(analyses))
        recommendations.extend(self._procedure_recommendations(procedures))
        recommendations.extend(self._error_recommendations(analyses))

        self._recommendations = recommendations
        return recommendations

    def _provider_recommendations(
        self, patterns: list[Any], analyses: dict[str, Any]
    ) -> list[Recommendation]:
        """توصيات المزودين"""
        recs: list[Recommendation] = []

        if not hasattr(patterns, "__iter__"):
            return recs

        provider_patterns = [
            p for p in patterns if hasattr(p, "pattern_type") and p.pattern_type == "provider_model"
        ]

        stats = analyses.get("statistics", {})
        success_rate = stats.get("success_rate", 0)

        if success_rate < 0.7:
            recs.append(
                Recommendation(
                    id="rec_provider_1",
                    category="provider",
                    title="Low Success Rate Detected",
                    description=f"Current success rate is {success_rate:.1%}, consider switching providers",
                    priority="high",
                    confidence=0.8,
                    action="Switch to backup provider or adjust prompts",
                    metadata={"current_rate": success_rate},
                )
            )

        best_pattern = None
        best_rate = 0.0
        for p in provider_patterns:
            if hasattr(p, "success_rate") and p.success_rate > best_rate:
                best_rate = p.success_rate
                best_pattern = p

        if best_pattern and best_rate > 0.9:
            recs.append(
                Recommendation(
                    id="rec_provider_2",
                    category="provider",
                    title="High-Performing Provider Found",
                    description=f"Provider {best_pattern.trigger} has {best_rate:.1%} success rate",
                    priority="medium",
                    confidence=best_rate,
                    action=f"Prefer {best_pattern.trigger} for similar tasks",
                    metadata={"provider": best_pattern.trigger, "rate": best_rate},
                )
            )

        return recs

    def _prompt_recommendations(
        self, patterns: list[Any], analyses: dict[str, Any]
    ) -> list[Recommendation]:
        """توصيات الـ prompts"""
        recs = []

        prompt_patterns = [
            p for p in patterns if hasattr(p, "pattern_type") and "prompt" in p.pattern_type
        ]

        for p in prompt_patterns:
            if hasattr(p, "success_rate") and p.success_rate > 0.85:
                recs.append(
                    Recommendation(
                        id=f"rec_prompt_{p.id}",
                        category="prompt",
                        title="Effective Prompt Style Found",
                        description=f"Prompt style '{p.trigger}' has {p.success_rate:.1%} success rate",
                        priority="medium",
                        confidence=p.success_rate,
                        action=f"Use {p.trigger} prompt style for similar tasks",
                        metadata={"prompt_type": p.trigger, "rate": p.success_rate},
                    )
                )

        return recs

    def _optimization_recommendations(self, analyses: dict[str, Any]) -> list[Recommendation]:
        """توصيات التحسين"""
        recs = []
        stats = analyses.get("statistics", {})

        avg_cost = stats.get("avg_cost_usd", 0)
        if avg_cost > 0.5:
            recs.append(
                Recommendation(
                    id="rec_opt_cost",
                    category="optimization",
                    title="High Average Cost",
                    description=f"Average cost is ${avg_cost:.3f}, consider using cheaper models",
                    priority="high",
                    confidence=0.7,
                    action="Use llama-3.1-8b or gemini-flash for simple tasks",
                    metadata={"avg_cost": avg_cost},
                )
            )

        avg_latency = stats.get("avg_latency_ms", 0)
        if avg_latency > 8000:
            recs.append(
                Recommendation(
                    id="rec_opt_latency",
                    category="optimization",
                    title="High Average Latency",
                    description=f"Average latency is {avg_latency:.0f}ms, consider faster providers",
                    priority="medium",
                    confidence=0.7,
                    action="Use Groq or Cerebras for time-sensitive tasks",
                    metadata={"avg_latency": avg_latency},
                )
            )

        avg_quality = stats.get("avg_quality_score", 0)
        if avg_quality < 60:
            recs.append(
                Recommendation(
                    id="rec_opt_quality",
                    category="optimization",
                    title="Low Quality Score",
                    description=f"Average quality is {avg_quality:.1f}, prompts may need refinement",
                    priority="high",
                    confidence=0.8,
                    action="Review and refine prompts for better results",
                    metadata={"avg_quality": avg_quality},
                )
            )

        return recs

    def _procedure_recommendations(self, procedures: list[Any]) -> list[Recommendation]:
        """توصيات الإجراءات"""
        recs = []

        if len(procedures) > 5:
            recs.append(
                Recommendation(
                    id="rec_proc_1",
                    category="procedure",
                    title="Multiple Procedures Available",
                    description=f"{len(procedures)} learned procedures available",
                    priority="low",
                    confidence=0.9,
                    action="Use learned procedures to speed up similar tasks",
                    metadata={"procedure_count": len(procedures)},
                )
            )

        return recs

    def _error_recommendations(self, analyses: dict[str, Any]) -> list[Recommendation]:
        """توصيات معالجة الأخطاء"""
        recs: list[Recommendation] = []
        stats = analyses.get("statistics", {})
        failure_patterns = stats.get("failure_patterns", {})

        if not failure_patterns:
            return recs

        for error_type, count in failure_patterns.items():
            if count >= 3:
                recs.append(
                    Recommendation(
                        id=f"rec_error_{error_type}",
                        category="error_handling",
                        title=f"Recurring {error_type} Errors",
                        description=f"Error '{error_type}' occurred {count} times",
                        priority="high" if count > 5 else "medium",
                        confidence=0.8,
                        action=self._get_error_action(error_type),
                        metadata={"error_type": error_type, "count": count},
                    )
                )

        return recs

    def _get_error_action(self, error_type: str) -> str:
        """الحصول على إجراء للخطأ"""
        actions = {
            "rate_limit": "Implement exponential backoff and add fallback providers",
            "timeout": "Increase timeout or switch to faster provider",
            "auth": "Update API credentials and implement key rotation",
            "model_not_found": "Verify model names and use provider-specific model lists",
            "server_error": "Implement retry with exponential backoff",
        }
        return actions.get(error_type, "Review error and implement appropriate handling")

    def get_recommendations_by_priority(self, priority: str) -> list[Recommendation]:
        """الحصول على توصيات حسب الأولوية"""
        return [r for r in self._recommendations if r.priority == priority]

    def get_recommendations_by_category(self, category: str) -> list[Recommendation]:
        """الحصول على توصيات حسب الفئة"""
        return [r for r in self._recommendations if r.category == category]

    def dismiss_recommendation(self, recommendation_id: str) -> bool:
        """رفض توصية"""
        for i, rec in enumerate(self._recommendations):
            if rec.id == recommendation_id:
                self._recommendations.pop(i)
                return True
        return False

    def get_all_recommendations(self) -> list[Recommendation]:
        """الحصول على جميع التوصيات"""
        return self._recommendations

    def clear(self) -> None:
        """مسح التوصيات"""
        self._recommendations.clear()
