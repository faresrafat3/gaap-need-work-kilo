"""
Specialist Agent Module
=======================

Intelligent domain specialization agent that:
- Detects the domain/specialization from queries
- Learns from usage patterns
- Adapts retrieval scope (narrow vs broad)
- Provides domain-specific context
"""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.agents.specialist")


class ScopeType(Enum):
    """نوع نطاق البحث"""

    NARROW = auto()  # ضيق: domain محدد جداً
    MODERATE = auto()  # متوسط: domain مع بعض المتعلقات
    BROAD = auto()  # واسع: كل المجالات المتعلقة
    GENERAL = auto()  # عام: كل الذاكرة


class Domain(Enum):
    """التخصصات المتاحة"""

    PYTHON = "python"
    DATABASE = "database"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    API = "api"
    SECURITY = "security"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    GENERAL = "general"


@dataclass
class DomainDecision:
    """
    قرار التخصص

    يحتوي على:
    - التخصص المحدد
    - مستوى الثقة
    - هل يحتاج تأكيد؟
    - اقتراحات بديلة
    """

    domain: str
    confidence: float
    scope: ScopeType = ScopeType.MODERATE
    needs_confirmation: bool = False
    alternative_domains: list[str] = field(default_factory=list)
    reasoning: str = ""
    context_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "confidence": self.confidence,
            "scope": self.scope.name,
            "needs_confirmation": self.needs_confirmation,
            "alternatives": self.alternative_domains,
            "reasoning": self.reasoning,
        }


@dataclass
class DomainProfile:
    """
    ملف تعريف التخصص

    يحتوي على معلومات عن:
    - الكلمات المفتاحية
    - الأنماط الشائعة
    - العلاقات مع تخصصات أخرى
    """

    name: str
    keywords: list[str]
    patterns: list[str]
    related_domains: list[str]
    typical_queries: list[str] = field(default_factory=list)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


class SpecialistAgent:
    """
    وكيل التخصص الذكي

    القدرات:
    - كشف التخصص من الاستعلام
    - التكيف مع السياق
    - التعلم من أنماط الاستخدام
    - تحديد نطاق البحث
    - اقتراح تخصصات بديلة
    """

    DOMAIN_PROFILES: dict[str, DomainProfile] = {
        "python": DomainProfile(
            name="python",
            keywords=[
                "python",
                "django",
                "flask",
                "fastapi",
                "pip",
                "pytest",
                "asyncio",
                "pandas",
                "numpy",
            ],
            patterns=["import ", "from ", "def ", "class ", "async def"],
            related_domains=["database", "api", "testing"],
            typical_queries=["كيف أستخدم", "خطأ في Python", "تثبيت مكتبة"],
        ),
        "database": DomainProfile(
            name="database",
            keywords=[
                "sql",
                "postgres",
                "mysql",
                "mongodb",
                "migration",
                "query",
                "table",
                "index",
                "database",
            ],
            patterns=["SELECT ", "INSERT ", "CREATE TABLE", "migration"],
            related_domains=["python", "api", "devops"],
            typical_queries=["إعداد قاعدة بيانات", "migration", "تحسين استعلام"],
        ),
        "devops": DomainProfile(
            name="devops",
            keywords=[
                "docker",
                "kubernetes",
                "ci",
                "cd",
                "deploy",
                "container",
                "aws",
                "gcp",
                "azure",
            ],
            patterns=["docker ", "kubectl ", "docker-compose"],
            related_domains=["database", "security", "architecture"],
            typical_queries=["إعداد Docker", "deployment", "CI/CD"],
        ),
        "frontend": DomainProfile(
            name="frontend",
            keywords=[
                "react",
                "vue",
                "angular",
                "css",
                "html",
                "javascript",
                "typescript",
                "dom",
                "component",
            ],
            patterns=["useState", "useEffect", "className", "v-if", "ngIf"],
            related_domains=["api", "testing"],
            typical_queries=["إنشاء component", "styling", "state management"],
        ),
        "api": DomainProfile(
            name="api",
            keywords=[
                "api",
                "rest",
                "graphql",
                "endpoint",
                "request",
                "response",
                "json",
                "http",
                "authentication",
            ],
            patterns=["fetch(", "axios.", "GET /", "POST /", "mutation {"],
            related_domains=["python", "database", "security"],
            typical_queries=["إنشاء endpoint", "تصحيح طلب", "authentication"],
        ),
        "security": DomainProfile(
            name="security",
            keywords=[
                "security",
                "auth",
                "jwt",
                "token",
                "encryption",
                "ssl",
                "https",
                "cors",
                "xss",
                "injection",
            ],
            patterns=["bcrypt", "jwt.sign", "hash(", "encrypt"],
            related_domains=["api", "devops"],
            typical_queries=["تأمين API", "authentication", "encryption"],
        ),
        "architecture": DomainProfile(
            name="architecture",
            keywords=[
                "architecture",
                "design",
                "pattern",
                "microservice",
                "monolith",
                "scalability",
                "clean architecture",
            ],
            patterns=["layer", "service", "repository", "factory pattern"],
            related_domains=["python", "api", "database"],
            typical_queries=["تصميم النظام", "microservices", "best practices"],
        ),
        "testing": DomainProfile(
            name="testing",
            keywords=[
                "test",
                "pytest",
                "unittest",
                "jest",
                "coverage",
                "mock",
                "fixture",
                "integration test",
            ],
            patterns=["def test_", "it(", "describe(", "assert "],
            related_domains=["python", "frontend", "api"],
            typical_queries=["كتابة اختبار", "mock", "coverage"],
        ),
        "general": DomainProfile(
            name="general",
            keywords=[],
            patterns=[],
            related_domains=[],
            typical_queries=[],
        ),
    }

    DOMAIN_DETECTION_PROMPT = """Analyze the query and determine the most relevant domain.

Query: "{query}"
Available domains: {domains}

Respond in JSON:
{{
  "primary_domain": "domain_name",
  "confidence": 0.0-1.0,
  "scope": "NARROW|MODERATE|BROAD|GENERAL",
  "alternatives": ["domain1", "domain2"],
  "reasoning": "brief explanation"
}}"""

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
        learning_rate: float = 0.1,
    ):
        self._llm_provider = llm_provider
        self._model = model
        self._learning_rate = learning_rate

        self._domain_history: list[str] = []
        self._domain_transitions: dict[str, dict[str, int]] = {}
        self._query_domain_map: dict[str, str] = {}
        self._domain_confidence: dict[str, float] = {}

        self._logger = logger

    async def determine_domain(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> DomainDecision:
        """
        تحديد التخصص المناسب

        Process:
        1. فحص السياق المباشر
        2. فحص التاريخ
        3. تحليل الكلمات المفتاحية
        4. (اختياري) LLM للتحليل العميق
        5. دمج النتائج
        """
        context = context or {}

        keyword_decision = self._analyze_keywords(query)

        history_decision = self._analyze_history()

        context_decision = self._analyze_context(context)

        final_decision = self._merge_decisions(
            query=query,
            keyword_decision=keyword_decision,
            history_decision=history_decision,
            context_decision=context_decision,
        )

        if self._llm_provider and final_decision.confidence < 0.6:
            llm_decision = await self._llm_domain_detection(query)
            if llm_decision.confidence > final_decision.confidence:
                final_decision = llm_decision

        self._domain_history.append(final_decision.domain)
        self._update_transitions(final_decision.domain)

        self._logger.info(
            f"Domain decision: {final_decision.domain} "
            f"(confidence={final_decision.confidence:.2f}, "
            f"scope={final_decision.scope.name})"
        )

        return final_decision

    def _analyze_keywords(self, query: str) -> DomainDecision:
        """تحليل الكلمات المفتاحية"""

        query_lower = query.lower()
        query_words = set(query_lower.split())

        scores: dict[str, float] = {}
        evidence: dict[str, list[str]] = {}

        for domain_name, profile in self.DOMAIN_PROFILES.items():
            if domain_name == "general":
                continue

            keyword_matches = [kw for kw in profile.keywords if kw in query_lower]

            pattern_matches = [p for p in profile.patterns if p.lower() in query_lower]

            score = 0.0
            if keyword_matches:
                score += len(keyword_matches) * 0.3
            if pattern_matches:
                score += len(pattern_matches) * 0.4

            total_keywords = len(profile.keywords)
            if total_keywords > 0:
                coverage = len(keyword_matches) / min(total_keywords, 10)
                score += coverage * 0.3

            if score > 0:
                scores[domain_name] = min(1.0, score)
                evidence[domain_name] = keyword_matches + pattern_matches

        if not scores:
            return DomainDecision(
                domain="general",
                confidence=0.3,
                scope=ScopeType.GENERAL,
                reasoning="No domain-specific keywords found",
            )

        best_domain = max(scores, key=lambda d: scores[d])
        best_score = scores[best_domain]

        alternatives = sorted(
            [d for d in scores if d != best_domain],
            key=lambda d: scores[d],
            reverse=True,
        )[:2]

        scope = self._determine_scope(best_score)

        return DomainDecision(
            domain=best_domain,
            confidence=best_score,
            scope=scope,
            alternative_domains=alternatives,
            reasoning=f"Keywords matched: {', '.join(evidence.get(best_domain, []))}",
            context_evidence=evidence.get(best_domain, []),
        )

    def _analyze_history(self) -> DomainDecision | None:
        """تحليل التاريخ للتخصص"""

        if not self._domain_history:
            return None

        recent = self._domain_history[-5:]

        domain_counts: dict[str, int] = {}
        for d in recent:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        most_common = max(domain_counts, key=lambda d: domain_counts[d])
        count = domain_counts[most_common]

        confidence = min(1.0, count / len(recent) * 0.8)

        return DomainDecision(
            domain=most_common,
            confidence=confidence * 0.6,
            scope=ScopeType.MODERATE,
            reasoning=f"Based on recent history ({count}/{len(recent)} queries)",
        )

    def _analyze_context(self, context: dict[str, Any]) -> DomainDecision | None:
        """تحليل السياق المباشر"""

        explicit_domain = context.get("domain") or context.get("active_domain")

        if explicit_domain:
            return DomainDecision(
                domain=explicit_domain,
                confidence=0.9,
                scope=ScopeType.NARROW,
                reasoning="Explicitly specified in context",
            )

        task_type = context.get("task_type")
        if task_type:
            task_domain_map = {
                "code_generation": "python",
                "debugging": "python",
                "database": "database",
                "api_design": "api",
                "deployment": "devops",
                "testing": "testing",
            }

            mapped_domain = task_domain_map.get(str(task_type).lower())
            if mapped_domain:
                return DomainDecision(
                    domain=mapped_domain,
                    confidence=0.7,
                    scope=ScopeType.MODERATE,
                    reasoning=f"Inferred from task type: {task_type}",
                )

        return None

    def _merge_decisions(
        self,
        query: str,
        keyword_decision: DomainDecision,
        history_decision: DomainDecision | None,
        context_decision: DomainDecision | None,
    ) -> DomainDecision:
        """دمج القرارات"""

        if context_decision and context_decision.confidence > 0.8:
            return context_decision

        candidates = [keyword_decision]
        if history_decision:
            candidates.append(history_decision)
        if context_decision:
            candidates.append(context_decision)

        domain_votes: dict[str, float] = {}
        domain_reasons: dict[str, list[str]] = {}

        for decision in candidates:
            domain = decision.domain
            weight = decision.confidence

            domain_votes[domain] = domain_votes.get(domain, 0) + weight

            if domain not in domain_reasons:
                domain_reasons[domain] = []
            domain_reasons[domain].append(decision.reasoning)

        if keyword_decision.alternative_domains:
            for alt in keyword_decision.alternative_domains:
                profile = self.DOMAIN_PROFILES.get(keyword_decision.domain)
                if profile and alt in profile.related_domains:
                    domain_votes[alt] = domain_votes.get(alt, 0) + 0.2

        best_domain = max(domain_votes, key=lambda d: domain_votes[d])
        total_votes = sum(domain_votes.values())
        confidence = domain_votes[best_domain] / total_votes if total_votes > 0 else 0.5

        alternatives = [
            d
            for d in sorted(domain_votes, key=lambda d: domain_votes[d], reverse=True)
            if d != best_domain
        ][:2]

        scope = self._determine_scope(confidence)

        needs_confirmation = 0.4 < confidence < 0.6 and len(alternatives) > 0

        return DomainDecision(
            domain=best_domain,
            confidence=confidence,
            scope=scope,
            needs_confirmation=needs_confirmation,
            alternative_domains=alternatives,
            reasoning="; ".join(domain_reasons.get(best_domain, ["Merged decision"])),
        )

    def _determine_scope(self, confidence: float) -> ScopeType:
        """تحديد نطاق البحث"""

        if confidence >= 0.8:
            return ScopeType.NARROW
        elif confidence >= 0.6:
            return ScopeType.MODERATE
        elif confidence >= 0.4:
            return ScopeType.BROAD
        else:
            return ScopeType.GENERAL

    async def _llm_domain_detection(self, query: str) -> DomainDecision:
        """استخدام LLM لكشف التخصص"""

        if not self._llm_provider:
            return DomainDecision(domain="general", confidence=0.3, scope=ScopeType.GENERAL)

        try:
            domains = list(self.DOMAIN_PROFILES.keys())
            prompt = self.DOMAIN_DETECTION_PROMPT.format(
                query=query,
                domains=domains,
            )

            response = await self._llm_provider.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                temperature=0.1,
                max_tokens=200,
            )

            return self._parse_llm_response(response)

        except Exception as e:
            self._logger.warning(f"LLM domain detection failed: {e}")
            return DomainDecision(domain="general", confidence=0.3, scope=ScopeType.GENERAL)

    def _parse_llm_response(self, response: str) -> DomainDecision:
        """تحليل استجابة LLM"""

        import json

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                domain = data.get("primary_domain", "general")
                if domain not in self.DOMAIN_PROFILES:
                    domain = "general"

                confidence = float(data.get("confidence", 0.5))
                scope_str = data.get("scope", "MODERATE")

                try:
                    scope = ScopeType[scope_str]
                except KeyError:
                    scope = ScopeType.MODERATE

                return DomainDecision(
                    domain=domain,
                    confidence=confidence,
                    scope=scope,
                    alternative_domains=data.get("alternatives", []),
                    reasoning=data.get("reasoning", "LLM analysis"),
                )
        except Exception as e:
            self._logger.warning(f"Failed to parse LLM response: {e}")

        return DomainDecision(domain="general", confidence=0.3, scope=ScopeType.GENERAL)

    def _update_transitions(self, domain: str) -> None:
        """تحديث مصفوفة الانتقالات"""

        if self._domain_history:
            prev_domain = self._domain_history[-2] if len(self._domain_history) >= 2 else None

            if prev_domain:
                if prev_domain not in self._domain_transitions:
                    self._domain_transitions[prev_domain] = {}

                self._domain_transitions[prev_domain][domain] = (
                    self._domain_transitions[prev_domain].get(domain, 0) + 1
                )

    def get_related_domains(self, domain: str) -> list[str]:
        """الحصول على التخصصات المرتبطة"""

        profile = self.DOMAIN_PROFILES.get(domain)
        if profile:
            return profile.related_domains
        return []

    def get_domain_profile(self, domain: str) -> DomainProfile | None:
        """الحصول على ملف تعريف التخصص"""

        return self.DOMAIN_PROFILES.get(domain)

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الوكيل"""

        domain_counts: dict[str, int] = {}
        for d in self._domain_history:
            domain_counts[d] = domain_counts.get(d, 0) + 1

        return {
            "total_queries": len(self._domain_history),
            "domain_distribution": domain_counts,
            "recent_domains": self._domain_history[-5:],
            "has_llm": self._llm_provider is not None,
        }
