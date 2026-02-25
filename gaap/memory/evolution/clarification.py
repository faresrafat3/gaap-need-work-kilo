"""
Clarification System Module
===========================

Smart clarification system that asks + suggests.

When a query is ambiguous, this system:
1. Generates a clarifying question
2. Provides likely suggestions
3. Learns from user responses

Reference: docs/evolution_plan_2026/25_MEMORY_AUDIT_SPEC.md
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.evolution.clarification")


class ClarificationType(Enum):
    """نوع التوضيح"""

    DOMAIN = auto()  # توضيح التخصص
    INTENT = auto()  # توضيح النية
    SCOPE = auto()  # توضيح النطاق
    DETAIL = auto()  # توضيح التفاصيل
    CONTEXT = auto()  # توضيح السياق


@dataclass
class ClarificationRequest:
    """
    طلب توضيح

    Attributes:
        query: الاستعلام الأصلي
        context: السياق المتاح
        ambiguity_score: درجة الغموض
        detected_domain: التخصص المكتشف
        history: تاريخ الاستعلامات
    """

    query: str
    context: dict[str, Any] = field(default_factory=dict)
    ambiguity_score: float = 0.5
    detected_domain: str | None = None
    history: list[str] = field(default_factory=list)


@dataclass
class Suggestion:
    """
    اقتراح للمستخدم

    Attributes:
        text: نص الاقتراح
        domain: التخصص
        confidence: الثقة
        reasoning: السبب
        action: الإجراء المقترح
    """

    text: str
    domain: str = "general"
    confidence: float = 0.5
    reasoning: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "domain": self.domain,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "action": self.action,
        }


@dataclass
class ClarificationResponse:
    """
    رد التوضيح

    Attributes:
        question: سؤال التوضيح
        suggestions: قائمة الاقتراحات
        clarification_type: نوع التوضيح
        confidence: الثقة في الاقتراحات
        allow_custom: السماح بإدخال مخصص
    """

    question: str
    suggestions: list[Suggestion]
    clarification_type: ClarificationType
    confidence: float
    allow_custom: bool = True
    follow_up_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "type": self.clarification_type.name,
            "confidence": self.confidence,
            "allow_custom": self.allow_custom,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class ClarificationSystem:
    """
    نظام التوضيح الذكي

    Features:
    - يولد أسئلة توضيحية ذكية
    - يقترح خيارات محتملة
    - يتعلم من إجابات المستخدم
    - يدعم سياق المحادثة
    """

    CLARIFICATION_PROMPT = """Generate a clarification for an ambiguous query.

Query: "{query}"
Context: {context}
Recent History: {history}
Detected Domain: {detected_domain}

The query is ambiguous. Create:
1. A friendly clarification question in Arabic
2. 3-4 specific suggestions based on context and history

Respond in JSON:
{{
  "question": "مش فاهم بالظبط تقصد إيه...",
  "suggestions": [
    {{
      "text": "إعداد قاعدة البيانات",
      "domain": "database",
      "confidence": 0.8,
      "reasoning": "Based on recent database queries",
      "action": "show_database_setup"
    }}
  ],
  "clarification_type": "DOMAIN|INTENT|SCOPE|DETAIL|CONTEXT"
}}"""

    DOMAIN_SUGGESTIONS: dict[str, list[Suggestion]] = {
        "python": [
            Suggestion("تثبيت مكتبة Python", "python", 0.7, "عملية تثبيت شائعة"),
            Suggestion("تصحيح خطأ في الكود", "python", 0.7, "مشكلة شائعة"),
            Suggestion("تحسين أداء الكود", "python", 0.6, "تحسين الأداء"),
        ],
        "database": [
            Suggestion("إعداد قاعدة بيانات جديدة", "database", 0.8, "إعداد قاعدة"),
            Suggestion("عمل migration", "database", 0.8, "نقل البيانات"),
            Suggestion("تحسين استعلامات SQL", "database", 0.7, "تحسين الأداء"),
            Suggestion("تصحيح مشكلة اتصال", "database", 0.7, "مشاكل الاتصال"),
        ],
        "devops": [
            Suggestion("إعداد Docker container", "devops", 0.8, "Docker setup"),
            Suggestion("تصحيح Kubernetes", "devops", 0.7, "K8s debugging"),
            Suggestion("إعداد CI/CD pipeline", "devops", 0.7, "CI/CD"),
        ],
        "api": [
            Suggestion("إنشاء endpoint جديد", "api", 0.8, "API development"),
            Suggestion("تصحيح طلبات API", "api", 0.7, "API debugging"),
            Suggestion("إضافة authentication", "api", 0.7, "Security"),
        ],
        "frontend": [
            Suggestion("إنشاء component جديد", "frontend", 0.8, "Component creation"),
            Suggestion("تحسين styling", "frontend", 0.6, "Styling"),
            Suggestion("إدارة state", "frontend", 0.7, "State management"),
        ],
        "security": [
            Suggestion("تأمين API", "security", 0.8, "API security"),
            Suggestion("إضافة authentication", "security", 0.8, "Auth"),
            Suggestion("فحص ثغرات", "security", 0.7, "Security audit"),
        ],
    }

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        learning_enabled: bool = True,
    ):
        self._llm_provider = llm_provider
        self._model = model
        self._learning_enabled = learning_enabled

        self._user_preferences: dict[str, int] = {}
        self._query_patterns: dict[str, list[str]] = {}
        self._successful_clarifications: list[dict[str, Any]] = []

        self._logger = logger

    async def clarify(
        self,
        request: ClarificationRequest,
    ) -> ClarificationResponse:
        """
        توليد توضيح لاستعلام غامض

        Args:
            request: طلب التوضيح

        Returns:
            رد التوضيح مع السؤال والاقتراحات
        """
        suggestions = await self._generate_suggestions(request)

        question = await self._generate_question(request, suggestions)

        clarification_type = self._determine_type(request)

        confidence = self._calculate_confidence(suggestions)

        response = ClarificationResponse(
            question=question,
            suggestions=suggestions[:4],
            clarification_type=clarification_type,
            confidence=confidence,
        )

        self._logger.info(
            f"Generated clarification: {clarification_type.name}, "
            f"{len(suggestions)} suggestions, confidence={confidence:.2f}"
        )

        return response

    async def _generate_suggestions(
        self,
        request: ClarificationRequest,
    ) -> list[Suggestion]:
        """توليد الاقتراحات"""

        suggestions = []

        if request.detected_domain:
            domain_suggestions = self.DOMAIN_SUGGESTIONS.get(request.detected_domain, [])
            suggestions.extend(domain_suggestions)

        history_suggestions = self._get_history_suggestions(request.history)
        suggestions.extend(history_suggestions)

        context_suggestions = self._get_context_suggestions(request.context)
        suggestions.extend(context_suggestions)

        preference_suggestions = self._get_preference_suggestions()
        suggestions.extend(preference_suggestions)

        if self._llm_provider and len(suggestions) < 3:
            llm_suggestions = await self._llm_suggestions(request)
            suggestions.extend(llm_suggestions)

        return self._rank_suggestions(suggestions)

    def _get_history_suggestions(
        self,
        history: list[str],
    ) -> list[Suggestion]:
        """اقتراحات من التاريخ"""

        suggestions = []

        for query in reversed(history[-5:]):
            if len(query) > 10:
                domain = self._infer_domain(query)

                suggestions.append(
                    Suggestion(
                        text=f"متابعة: {query[:50]}...",
                        domain=domain,
                        confidence=0.6,
                        reasoning="من تاريخ استعلاماتك",
                    )
                )

        return suggestions[:2]

    def _get_context_suggestions(
        self,
        context: dict[str, Any],
    ) -> list[Suggestion]:
        """اقتراحات من السياق"""

        suggestions = []

        if task := context.get("task_type"):
            task_suggestions = {
                "code_generation": Suggestion("كتابة كود جديد", "python", 0.7),
                "debugging": Suggestion("تصحيح خطأ", "python", 0.8),
                "database": Suggestion("عملية قاعدة بيانات", "database", 0.8),
                "api": Suggestion("تصميم API", "api", 0.8),
            }

            if suggestion := task_suggestions.get(str(task).lower()):
                suggestions.append(suggestion)

        if recent_topic := context.get("recent_topic"):
            suggestions.append(
                Suggestion(
                    text=f"توضيح بخصوص {recent_topic}",
                    domain=context.get("domain", "general"),
                    confidence=0.6,
                    reasoning="من السياق الحالي",
                )
            )

        return suggestions

    def _get_preference_suggestions(self) -> list[Suggestion]:
        """اقتراحات من تفضيلات المستخدم"""

        suggestions = []

        sorted_prefs = sorted(
            self._user_preferences.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:2]

        for domain, count in sorted_prefs:
            if count >= 2:
                domain_suggestions = self.DOMAIN_SUGGESTIONS.get(domain, [])
                if domain_suggestions:
                    suggestions.append(domain_suggestions[0])

        return suggestions

    async def _llm_suggestions(
        self,
        request: ClarificationRequest,
    ) -> list[Suggestion]:
        """اقتراحات باستخدام LLM"""

        if not self._llm_provider:
            return []

        try:
            context_str = str(request.context)[:200]
            history_str = ", ".join(request.history[-3:])

            prompt = self.CLARIFICATION_PROMPT.format(
                query=request.query,
                context=context_str,
                history=history_str,
                detected_domain=request.detected_domain or "unknown",
            )

            response = await self._llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=0.3,
                max_tokens=300,
            )

            return self._parse_llm_suggestions(response)

        except Exception as e:
            self._logger.debug(f"LLM suggestions failed: {e}")
            return []

    def _parse_llm_suggestions(
        self,
        response: str,
    ) -> list[Suggestion]:
        """تحليل اقتراحات LLM"""

        import json

        suggestions = []

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                for item in data.get("suggestions", []):
                    suggestions.append(
                        Suggestion(
                            text=item.get("text", ""),
                            domain=item.get("domain", "general"),
                            confidence=float(item.get("confidence", 0.5)),
                            reasoning=item.get("reasoning", ""),
                            action=item.get("action", ""),
                        )
                    )

        except Exception as e:
            self._logger.debug(f"Failed to parse LLM suggestions: {e}")

        return suggestions

    def _rank_suggestions(
        self,
        suggestions: list[Suggestion],
    ) -> list[Suggestion]:
        """ترتيب الاقتراحات"""

        seen = set()
        unique = []

        for s in suggestions:
            key = s.text.lower()[:30]
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return sorted(unique, key=lambda x: x.confidence, reverse=True)

    async def _generate_question(
        self,
        request: ClarificationRequest,
        suggestions: list[Suggestion],
    ) -> str:
        """توليد سؤال التوضيح"""

        if self._llm_provider:
            try:
                context_str = str(request.context)[:200]
                history_str = ", ".join(request.history[-3:])

                prompt = self.CLARIFICATION_PROMPT.format(
                    query=request.query,
                    context=context_str,
                    history=history_str,
                    detected_domain=request.detected_domain or "unknown",
                )

                response = await self._llm_provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._model,
                    temperature=0.3,
                    max_tokens=100,
                )

                return self._parse_question(response)

            except Exception as e:
                self._logger.debug(f"LLM question generation failed: {e}")

        if suggestions:
            return "مش فاهم بالظبط تقصد إيه. هل تقصد واحد من دول؟"

        return "ممكن توضح أكتر قصدك إيه؟"

    def _parse_question(self, response: str) -> str:
        """تحليل السؤال من الرد"""

        import json

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])
                question: str = data.get("question", "مش فاهم بالظبط تقصد إيه؟")
                return question

        except Exception:
            pass

        return "مش فاهم بالظبط تقصد إيه؟"

    def _determine_type(
        self,
        request: ClarificationRequest,
    ) -> ClarificationType:
        """تحديد نوع التوضيح"""

        if not request.detected_domain:
            return ClarificationType.DOMAIN

        if request.ambiguity_score > 0.7:
            return ClarificationType.INTENT

        if len(request.query.split()) < 4:
            return ClarificationType.DETAIL

        return ClarificationType.CONTEXT

    def _calculate_confidence(
        self,
        suggestions: list[Suggestion],
    ) -> float:
        """حساب الثقة في الاقتراحات"""

        if not suggestions:
            return 0.0

        top_confidence = suggestions[0].confidence if suggestions else 0.0

        count_factor = min(len(suggestions) / 4.0, 1.0)

        return top_confidence * 0.7 + count_factor * 0.3

    def _infer_domain(self, text: str) -> str:
        """استنتاج التخصص من النص"""

        text_lower = text.lower()

        domain_keywords = {
            "python": ["python", "django", "flask", "pip"],
            "database": ["sql", "postgres", "mysql", "mongo", "database"],
            "devops": ["docker", "kubernetes", "deploy", "ci"],
            "api": ["api", "endpoint", "rest", "graphql"],
            "frontend": ["react", "vue", "angular", "css", "html"],
        }

        for domain, keywords in domain_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return domain

        return "general"

    def learn_from_response(
        self,
        query: str,
        selected_suggestion: str,
        domain: str,
    ) -> None:
        """
        التعلم من اختيار المستخدم

        Args:
            query: الاستعلام الأصلي
            selected_suggestion: الاقتراح المختار
            domain: التخصص
        """
        if not self._learning_enabled:
            return

        self._user_preferences[domain] = self._user_preferences.get(domain, 0) + 1

        query_key = query[:50].lower()
        if query_key not in self._query_patterns:
            self._query_patterns[query_key] = []
        self._query_patterns[query_key].append(selected_suggestion)

        self._successful_clarifications.append(
            {
                "query": query,
                "suggestion": selected_suggestion,
                "domain": domain,
                "timestamp": datetime.now().isoformat(),
            }
        )

        self._logger.debug(f"Learned from clarification: {domain}")

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات النظام"""

        return {
            "total_clarifications": len(self._successful_clarifications),
            "user_preferences": self._user_preferences,
            "patterns_learned": len(self._query_patterns),
            "learning_enabled": self._learning_enabled,
        }
