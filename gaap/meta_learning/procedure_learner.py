from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LearnedProcedure:
    """إجراء مكتسب"""

    id: str
    name: str
    task_type: str
    prompt_template: str
    success_rate: float
    sample_count: int
    variables: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    use_count: int = 0


class ProcedureLearner:
    """معلم الإجراءات - يتعلم procedures جديدة من المهام الناجحة"""

    def __init__(self, min_success_rate: float = 0.8, min_samples: int = 3):
        self.min_success_rate = min_success_rate
        self.min_samples = min_samples
        self._procedures: dict[str, LearnedProcedure] = {}
        self._task_templates: dict[str, list[dict[str, Any]]] = {}

    def learn_from_task(
        self,
        task_type: str,
        prompt: str,
        success: bool,
        quality_score: float,
    ) -> LearnedProcedure | None:
        """التعلم من مهمة واحدة"""
        if task_type not in self._task_templates:
            self._task_templates[task_type] = []

        self._task_templates[task_type].append(
            {
                "prompt": prompt,
                "success": success,
                "quality": quality_score,
                "timestamp": datetime.now(),
            }
        )

        if len(self._task_templates[task_type]) >= self.min_samples:
            return self._extract_procedure(task_type)

        return None

    def _extract_procedure(self, task_type: str) -> LearnedProcedure | None:
        """استخراج إجراء من المهام المتشابهة"""
        templates = self._task_templates.get(task_type, [])
        if len(templates) < self.min_samples:
            return None

        successful = [t for t in templates if t["success"] and t["quality"] >= 70]
        if not successful:
            return None

        success_rate = len(successful) / len(templates)
        if success_rate < self.min_success_rate:
            return None

        template = self._extract_common_template([t["prompt"] for t in successful])
        variables = self._extract_variables(template)

        procedure = LearnedProcedure(
            id=f"proc_{task_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=f"{task_type}_procedure",
            task_type=task_type,
            prompt_template=template,
            success_rate=success_rate,
            sample_count=len(successful),
            variables=variables,
        )

        self._procedures[procedure.id] = procedure
        return procedure

    def _extract_common_template(self, prompts: list[str]) -> str:
        """استخراج قالب مشترك من الـ prompts"""
        if not prompts:
            return ""

        if len(prompts) == 1:
            return prompts[0]

        common = prompts[0]
        for prompt in prompts[1:]:
            common = self._find_common(common, prompt)

        return self._generalize(common)

    def _find_common(self, s1: str, s2: str) -> str:
        """إيجاد المشترك بين نصين"""
        common = []
        s1_parts = s1.split()
        s2_parts = s2.split()

        for p1, p2 in zip(s1_parts, s2_parts):
            if p1 == p2:
                common.append(p1)
            else:
                break

        return " ".join(common) if common else s1[:50]

    def _generalize(self, template: str) -> str:
        """تعميم القالب"""
        import re

        template = re.sub(r"\d+", "{number}", template)
        template = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "{email}", template)
        template = re.sub(r"http[s]?://[^\s]+", "{url}", template)

        return template

    def _extract_variables(self, template: str) -> list[str]:
        """استخراج المتغيرات من القالب"""
        import re

        variables = re.findall(r"\{(\w+)\}", template)
        return list(set(variables))

    def get_procedure(self, task_type: str) -> LearnedProcedure | None:
        """الحصول على إجراء مناسب للمهمة"""
        best = None
        best_rate = 0.0

        for proc in self._procedures.values():
            if proc.task_type == task_type and proc.success_rate > best_rate:
                best_rate = proc.success_rate
                best = proc

        if best:
            best.use_count += 1
            best.last_used = datetime.now()

        return best

    def get_all_procedures(self) -> list[LearnedProcedure]:
        """الحصول على جميع الإجراءات"""
        return list(self._procedures.values())

    def get_procedures_by_task_type(self, task_type: str) -> list[LearnedProcedure]:
        """الحصول على إجراءات نوع معين"""
        return [p for p in self._procedures.values() if p.task_type == task_type]

    def delete_procedure(self, procedure_id: str) -> bool:
        """حذف إجراء"""
        if procedure_id in self._procedures:
            del self._procedures[procedure_id]
            return True
        return False

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "total_procedures": len(self._procedures),
            "by_task_type": self._get_task_type_counts(),
            "avg_success_rate": self._get_avg_success_rate(),
            "total_uses": sum(p.use_count for p in self._procedures.values()),
        }

    def _get_task_type_counts(self) -> dict[str, int]:
        """عدد الإجراءات لكل نوع مهمة"""
        counts: dict[str, int] = {}
        for proc in self._procedures.values():
            counts[proc.task_type] = counts.get(proc.task_type, 0) + 1
        return counts

    def _get_avg_success_rate(self) -> float:
        """متوسط نسبة النجاح"""
        if not self._procedures:
            return 0.0
        return sum(p.success_rate for p in self._procedures.values()) / len(self._procedures)
