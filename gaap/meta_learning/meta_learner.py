import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from gaap.meta_learning.experience_analyzer import ExperienceAnalyzer
from gaap.meta_learning.pattern_extractor import PatternExtractor
from gaap.meta_learning.procedure_learner import ProcedureLearner
from gaap.meta_learning.recommendation import Recommendation, RecommendationEngine


@dataclass
class LearningResult:
    """نتيجة التعلم"""

    patterns_extracted: int
    procedures_learned: int
    recommendations_generated: int
    analysis_summary: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class MetaLearner:
    """
    المعلم الفوقي - يحلل الخبرات ويستخرج الأنماط

    الوظائف:
    - استخراج الأنماط من تاريخ المهام
    - تحليل النتائج (نجاح/فشل)
    - تعلم إجراءات جديدة
    - توليد توصيات
    """

    def __init__(
        self,
        min_pattern_occurrences: int = 3,
        success_threshold: float = 70.0,
        auto_learn: bool = True,
    ):
        self.min_pattern_occurrences = min_pattern_occurrences
        self.success_threshold = success_threshold
        self.auto_learn = auto_learn

        self.pattern_extractor = PatternExtractor(min_occurrences=min_pattern_occurrences)
        self.experience_analyzer = ExperienceAnalyzer(success_threshold=success_threshold)
        self.procedure_learner = ProcedureLearner(min_success_rate=0.8, min_samples=3)
        self.recommendation_engine = RecommendationEngine()

        self._logger = logging.getLogger("gaap.meta_learning")

        self._last_learning: LearningResult | None = None

    async def learn_from_history(self, task_history: list[dict[str, Any]]) -> LearningResult:
        """التعلم من تاريخ المهام"""
        self._logger.info(f"Learning from {len(task_history)} tasks")

        patterns = self.pattern_extractor.analyze_tasks(task_history)
        self._logger.info(f"Extracted {len(patterns)} patterns")

        analyses = self.experience_analyzer.analyze_outcomes(task_history)
        self._logger.info(
            f"Analyzed {len(analyses.get('successes', []))} successes, {len(analyses.get('failures', []))} failures"
        )

        procedures = self._learn_procedures(task_history)
        self._logger.info(f"Learned {len(procedures)} procedures")

        recommendations = self.recommendation_engine.generate_recommendations(
            patterns=patterns,
            analyses=analyses,
            procedures=procedures,
        )
        self._logger.info(f"Generated {len(recommendations)} recommendations")

        self._last_learning = LearningResult(
            patterns_extracted=len(patterns),
            procedures_learned=len(procedures),
            recommendations_generated=len(recommendations),
            analysis_summary=analyses.get("statistics", {}),
        )

        return self._last_learning

    def _learn_procedures(self, task_history: list[dict[str, Any]]) -> list[Any]:
        """التعلم من المهام الناجحة"""
        procedures = []

        task_prompts: dict[str, list[dict[str, Any]]] = {}

        for task in task_history:
            task_type = task.get("type", "unknown")
            prompt = task.get("prompt", "")
            success = task.get("success", False)
            quality = task.get("quality_score", 0.0)

            if not prompt:
                continue

            if task_type not in task_prompts:
                task_prompts[task_type] = []

            task_prompts[task_type].append(
                {
                    "prompt": prompt,
                    "success": success,
                    "quality": quality,
                }
            )

        for task_type, prompts in task_prompts.items():
            for task_data in prompts:
                proc = self.procedure_learner.learn_from_task(
                    task_type=task_type,
                    prompt=task_data["prompt"],
                    success=task_data["success"],
                    quality_score=task_data["quality"],
                )
                if proc:
                    procedures.append(proc)

        return procedures

    def get_recommendations(self, priority: str | None = None) -> list[Recommendation]:
        """الحصول على التوصيات"""
        if priority:
            return self.recommendation_engine.get_recommendations_by_priority(priority)
        return self.recommendation_engine.get_all_recommendations()

    def get_best_procedure(self, task_type: str) -> Any | None:
        """الحصول على أفضل إجراء للمهمة"""
        return self.procedure_learner.get_procedure(task_type)

    def get_patterns(self, pattern_type: str | None = None) -> list[Any]:
        """الحصول على الأنماط"""
        if pattern_type:
            return self.pattern_extractor.get_patterns_by_type(pattern_type)
        return list(self.pattern_extractor._patterns.values())

    def get_high_success_patterns(self, min_rate: float = 0.8) -> list[Any]:
        """الحصول على الأنماط ذات النجاح العالي"""
        return self.pattern_extractor.get_high_success_patterns(min_rate)

    def get_analysis_summary(self) -> dict[str, Any]:
        """ملخص التحليل"""
        return {
            "patterns": self.pattern_extractor.get_stats(),
            "procedures": self.procedure_learner.get_stats(),
            "last_learning": {
                "timestamp": self._last_learning.timestamp if self._last_learning else None,
                "patterns_extracted": (
                    self._last_learning.patterns_extracted if self._last_learning else 0
                ),
                "procedures_learned": (
                    self._last_learning.procedures_learned if self._last_learning else 0
                ),
                "recommendations_generated": (
                    self._last_learning.recommendations_generated if self._last_learning else 0
                ),
            },
        }

    def apply_recommendation(self, recommendation: Recommendation) -> dict[str, Any]:
        """تطبيق توصية"""
        self._logger.info(f"Applying recommendation: {recommendation.title}")

        if recommendation.category == "provider":
            return {
                "action": "switch_provider",
                "target": recommendation.metadata.get("provider"),
                "confidence": recommendation.confidence,
            }
        elif recommendation.category == "prompt":
            return {
                "action": "update_prompt_style",
                "style": recommendation.metadata.get("prompt_type"),
                "confidence": recommendation.confidence,
            }
        elif recommendation.category == "optimization":
            return {
                "action": recommendation.action,
                "target": recommendation.metadata,
                "confidence": recommendation.confidence,
            }
        elif recommendation.category == "error_handling":
            return {
                "action": "implement_error_handling",
                "error_type": recommendation.metadata.get("error_type"),
                "confidence": recommendation.confidence,
            }

        return {"action": "unknown", "confidence": 0.0}

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "pattern_extractor": self.pattern_extractor.get_stats(),
            "procedure_learner": self.procedure_learner.get_stats(),
            "recommendations": {
                "total": len(self.recommendation_engine.get_all_recommendations()),
                "high_priority": len(
                    self.recommendation_engine.get_recommendations_by_priority("high")
                ),
                "medium_priority": len(
                    self.recommendation_engine.get_recommendations_by_priority("medium")
                ),
            },
        }

    def save(self, storage_path: str) -> bool:
        """حفظ الـ meta-learning data للقرص"""
        import json

        try:
            Path(storage_path).mkdir(parents=True, exist_ok=True)

            patterns_data = {
                pid: {
                    "id": p.id,
                    "pattern_type": p.pattern_type,
                    "trigger": p.trigger,
                    "frequency": p.frequency,
                    "success_rate": p.success_rate,
                    "sample_tasks": p.sample_tasks,
                    "metadata": p.metadata,
                    "discovered_at": p.discovered_at.isoformat() if p.discovered_at else None,
                }
                for pid, p in self.pattern_extractor._patterns.items()
            }

            recommendations_data = [
                {
                    "id": r.id,
                    "category": r.category,
                    "title": r.title,
                    "description": r.description,
                    "priority": r.priority,
                    "confidence": r.confidence,
                    "action": r.action,
                    "metadata": r.metadata,
                }
                for r in self.recommendation_engine.get_all_recommendations()
            ]

            data = {
                "patterns": patterns_data,
                "procedures": self.procedure_learner.get_stats(),
                "recommendations": recommendations_data,
                "saved_at": datetime.now().isoformat(),
            }

            filepath = Path(storage_path) / "meta_learning.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)

            self._logger.info(f"Saved meta-learning data to {filepath}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to save meta-learning data: {e}")
            return False

    def load(self, storage_path: str) -> bool:
        """تحميل الـ meta-learning data من القرص"""
        import json

        try:
            filepath = Path(storage_path) / "meta_learning.json"
            if not filepath.exists():
                self._logger.info(f"No existing meta-learning file at {filepath}")
                return False

            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            from gaap.meta_learning.pattern_extractor import Pattern

            for pid, pdata in data.get("patterns", {}).items():
                discovered_at = pdata.get("discovered_at")
                if discovered_at:
                    try:
                        discovered_at = datetime.fromisoformat(discovered_at)
                    except Exception:
                        discovered_at = datetime.now()

                pattern = Pattern(
                    id=pdata.get("id", pid),
                    pattern_type=pdata.get("pattern_type", ""),
                    trigger=pdata.get("trigger", ""),
                    frequency=pdata.get("frequency", 0),
                    success_rate=pdata.get("success_rate", 0.0),
                    sample_tasks=pdata.get("sample_tasks", []),
                    metadata=pdata.get("metadata", {}),
                    discovered_at=discovered_at,
                )
                self.pattern_extractor._patterns[pid] = pattern

            self._logger.info(
                f"Loaded {len(self.pattern_extractor._patterns)} patterns from {filepath}"
            )
            return True

        except Exception as e:
            self._logger.error(f"Failed to load meta-learning data: {e}")
            return False
