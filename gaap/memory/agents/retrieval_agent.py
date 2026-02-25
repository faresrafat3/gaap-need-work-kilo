"""
Retrieval Agent Module
======================

Intelligent retrieval agent that:
- Analyzes query complexity
- Handles ambiguous queries with clarification + suggestions
- Multi-stage retrieval with reranking
- Integrates with specialist agent for domain-specific retrieval
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.agents.retrieval")


class QueryType(Enum):
    """نوع الاستعلام"""

    SPECIFIC = auto()  # محدد: "كيف أصلح خطأ ImportError؟"
    BROAD = auto()  # واسع: "أخبرني عن قواعد البيانات"
    AMBIGUOUS = auto()  # غامض: "ازاي أظبط ده؟"
    COMPARISON = auto()  # مقارنة: "الفرق بين PostgreSQL و MySQL"
    HOW_TO = auto()  # كيفية: "كيف أعمل migration؟"
    DEBUGGING = auto()  # تصحيح: "ليه الكود مش شغال؟"


@dataclass
class RetrievalContext:
    """
    سياق عملية الاسترجاع

    يحتوي على كل المعلومات اللي ممكن تساعد في الاسترجاع الذكي
    """

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    previous_queries: list[str] = field(default_factory=list)
    active_domain: str | None = None
    user_preferences: dict[str, Any] = field(default_factory=dict)
    session_context: dict[str, Any] = field(default_factory=dict)
    task_type: str | None = None
    max_results: int = 5
    min_confidence: float = 0.3
    include_related: bool = True


@dataclass
class ClarificationResponse:
    """
    رد التوضيح - يسأل ويقترح في نفس الوقت

    مثال:
    "مش فاهم بالظبط تقصد إيه. هل تقصد:
     1. إعداد قاعدة البيانات؟
     2. تكوين الـ API؟
     أو وضح أكتر..."
    """

    question: str
    suggestions: list[dict[str, Any]]
    original_query: str
    confidence: float
    needs_user_input: bool = True
    detected_domain: str | None = None
    context_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "suggestions": self.suggestions,
            "original_query": self.original_query,
            "confidence": self.confidence,
            "needs_user_input": self.needs_user_input,
            "detected_domain": self.detected_domain,
        }


@dataclass
class RetrievalResult:
    """
    نتيجة الاسترجاع
    """

    items: list[dict[str, Any]]
    query_type: QueryType
    domain: str
    confidence: float
    clarification: ClarificationResponse | None = None
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0
    rerank_time_ms: float = 0.0
    sources: list[str] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "query_type": self.query_type.name,
            "domain": self.domain,
            "confidence": self.confidence,
            "needs_clarification": self.needs_clarification,
            "clarification": self.clarification.to_dict() if self.clarification else None,
            "total_candidates": self.total_candidates,
            "retrieval_time_ms": self.retrieval_time_ms,
            "sources": self.sources,
        }


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM provider."""

    async def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for Vector Store."""

    def search(self, query: str, n_results: int, filter_meta: dict | None) -> list[Any]: ...


@runtime_checkable
class Reranker(Protocol):
    """Protocol for Reranker."""

    async def rerank(self, request: Any) -> list[Any]: ...


@runtime_checkable
class KnowledgeGraph(Protocol):
    """Protocol for Knowledge Graph."""

    def get_neighbors(self, node_id: str, depth: int) -> set[str]: ...


@runtime_checkable
class SpecialistAgentProtocol(Protocol):
    """Protocol for Specialist Agent."""

    async def determine_domain(self, query: str, context: dict) -> Any: ...


class RetrievalAgent:
    """
    وكيل الاسترجاع الذكي

    القدرات:
    - تحليل نوع الاستعلام
    - كشف الغموض والسؤال + الاقتراح
    - استرجاع متعدد المراحل
    - التكامل مع التخصص
    - Re-ranking ذكي
    """

    CLARIFICATION_PROMPT = """You are an intelligent assistant helping clarify a user's question.

## User's Question
"{query}"

## Context from Conversation
{context}

## User's History (Recent Topics)
{history}

## Task
The question is ambiguous or unclear. Based on the context and history:
1. Generate a short, friendly clarification question in Arabic
2. Suggest 3-4 likely interpretations as numbered options

Respond in JSON format:
{{
  "clarification_question": "مش فاهم بالظبط تقصد إيه...",
  "suggestions": [
    {{"text": "إعداد قاعدة البيانات", "domain": "database", "confidence": 0.8}},
    {{"text": "تكوين الـ API", "domain": "api", "confidence": 0.6}},
    ...
  ]
}}

Keep the question conversational and helpful."""

    QUERY_ANALYSIS_PROMPT = """Analyze this query and classify it.

Query: "{query}"

Respond in JSON:
{{
  "type": "SPECIFIC|BROAD|AMBIGUOUS|COMPARISON|HOW_TO|DEBUGGING",
  "confidence": 0.0-1.0,
  "keywords": ["keyword1", "keyword2"],
  "implied_domain": "domain or null",
  "is_ambiguous": true/false,
  "needs_context": true/false
}}"""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        reranker: Reranker | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        specialist_agent: SpecialistAgentProtocol | None = None,
        llm_provider: LLMProvider | None = None,
        model: str = "gpt-4o-mini",
    ):
        self._vector_store = vector_store
        self._reranker = reranker
        self._knowledge_graph = knowledge_graph
        self._specialist_agent = specialist_agent
        self._llm_provider = llm_provider
        self._model = model

        self._query_history: list[str] = []
        self._domain_history: list[str] = []
        self._logger = logger

    async def retrieve(
        self,
        query: str,
        context: RetrievalContext | None = None,
    ) -> RetrievalResult:
        """
        الاسترجاع الذكي

        Process:
        1. تحليل الاستعلام
        2. إذا غامض → توضيح + اقتراحات
        3. تحديد التخصص
        4. استرجاع متعدد المراحل
        5. Re-ranking
        """
        start_time = time.time()
        context = context or RetrievalContext()

        self._query_history.append(query)

        analysis = await self._analyze_query(query, context)

        if analysis.get("is_ambiguous", False) and analysis.get("confidence", 0) < 0.6:
            clarification = await self._generate_clarification(query, context)

            return RetrievalResult(
                items=[],
                query_type=QueryType.AMBIGUOUS,
                domain=clarification.detected_domain or "general",
                confidence=analysis.get("confidence", 0.3),
                clarification=clarification,
                retrieval_time_ms=(time.time() - start_time) * 1000,
            )

        domain = await self._determine_domain(query, context, analysis)
        self._domain_history.append(domain)

        candidates = await self._multi_stage_retrieve(query, domain, context)

        ranked_items = await self._rerank_results(query, candidates, context)

        result = RetrievalResult(
            items=ranked_items[: context.max_results],
            query_type=QueryType[analysis.get("type", "SPECIFIC")],
            domain=domain,
            confidence=analysis.get("confidence", 0.5),
            total_candidates=len(candidates),
            retrieval_time_ms=(time.time() - start_time) * 1000,
            sources=self._get_sources(candidates),
        )

        self._logger.info(
            f"Retrieval completed: {result.query_type.name}, "
            f"domain={domain}, confidence={result.confidence:.2f}, "
            f"candidates={result.total_candidates}"
        )

        return result

    async def _analyze_query(
        self,
        query: str,
        context: RetrievalContext,
    ) -> dict[str, Any]:
        """تحليل الاستعلام لتحديد نوعه وخصائصه"""

        if self._llm_provider:
            try:
                prompt = self.QUERY_ANALYSIS_PROMPT.format(query=query)
                response = await self._llm_provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._model,
                    temperature=0.1,
                    max_tokens=200,
                )

                return self._parse_analysis(response)
            except Exception as e:
                self._logger.warning(f"LLM query analysis failed: {e}")

        return self._heuristic_analysis(query, context)

    def _heuristic_analysis(
        self,
        query: str,
        context: RetrievalContext,
    ) -> dict[str, Any]:
        """تحليل استدلالي بدون LLM"""

        query_lower = query.lower()

        ambiguous_indicators = ["ده", "دي", "كده", "هنا", "المشكلة", "الخطأ", "الأمر"]
        is_ambiguous = (
            any(ind in query_lower for ind in ambiguous_indicators) and len(query.split()) < 5
        )

        how_to_indicators = ["كيف", "ازاي", "كيفية", "طريقة", "خطوات"]
        is_how_to = any(ind in query_lower for ind in how_to_indicators)

        debug_indicators = ["ليه", "مش شغال", "خطأ", "غلط", "مش بيشتغل"]
        is_debugging = any(ind in query_lower for ind in debug_indicators)

        comparison_indicators = ["فرق", "مقارنة", "أحسن", "أفضل", "vs", "أو"]
        is_comparison = any(ind in query_lower for ind in comparison_indicators)

        broad_indicators = ["عن", "معلومات", "شرح", "كل", "جميع"]
        is_broad = any(ind in query_lower for ind in broad_indicators)

        if is_ambiguous:
            query_type = "AMBIGUOUS"
            confidence = 0.4
        elif is_how_to:
            query_type = "HOW_TO"
            confidence = 0.7
        elif is_debugging:
            query_type = "DEBUGGING"
            confidence = 0.7
        elif is_comparison:
            query_type = "COMPARISON"
            confidence = 0.6
        elif is_broad:
            query_type = "BROAD"
            confidence = 0.5
        else:
            query_type = "SPECIFIC"
            confidence = 0.8

        keywords = [w for w in query_lower.split() if len(w) > 2][:5]

        implied_domain = self._detect_domain_from_keywords(keywords)

        return {
            "type": query_type,
            "confidence": confidence,
            "keywords": keywords,
            "implied_domain": implied_domain,
            "is_ambiguous": is_ambiguous,
            "needs_context": is_ambiguous,
        }

    def _detect_domain_from_keywords(self, keywords: list[str]) -> str | None:
        """كشف التخصص من الكلمات المفتاحية"""

        domain_keywords = {
            "python": ["python", "django", "flask", "fastapi", "pip"],
            "database": ["database", "sql", "postgres", "mysql", "mongodb", "migration"],
            "devops": ["docker", "kubernetes", "ci", "cd", "deploy", "container"],
            "frontend": ["react", "vue", "angular", "css", "html", "javascript"],
            "api": ["api", "rest", "graphql", "endpoint", "request"],
            "security": ["security", "auth", "encryption", "ssl", "token"],
        }

        for domain, dkeywords in domain_keywords.items():
            if any(kw in dk for kw in keywords for dk in dkeywords):
                return domain

        return None

    async def _generate_clarification(
        self,
        query: str,
        context: RetrievalContext,
    ) -> ClarificationResponse:
        """توليد سؤال توضيحي مع اقتراحات"""

        suggestions = await self._get_suggestions(query, context)

        if self._llm_provider:
            try:
                context_str = self._format_conversation_context(context)
                history_str = self._format_query_history()

                prompt = self.CLARIFICATION_PROMPT.format(
                    query=query,
                    context=context_str,
                    history=history_str,
                )

                response = await self._llm_provider.complete(
                    messages=[{"role": "user", "content": prompt}],
                    model=self._model,
                    temperature=0.3,
                    max_tokens=300,
                )

                parsed = self._parse_clarification(response)

                return ClarificationResponse(
                    question=parsed.get("clarification_question", "مش فاهم بالظبط تقصد إيه؟"),
                    suggestions=parsed.get("suggestions", suggestions),
                    original_query=query,
                    confidence=0.5,
                    detected_domain=parsed.get("suggestions", [{}])[0].get("domain")
                    if parsed.get("suggestions")
                    else None,
                    context_used=context_str,
                )
            except Exception as e:
                self._logger.warning(f"LLM clarification failed: {e}")

        return ClarificationResponse(
            question="مش فاهم بالظبط تقصد إيه. ممكن توضح أكتر؟",
            suggestions=suggestions,
            original_query=query,
            confidence=0.5,
            detected_domain=context.active_domain,
        )

    async def _get_suggestions(
        self,
        query: str,
        context: RetrievalContext,
    ) -> list[dict[str, Any]]:
        """الحصول على اقتراحات من التاريخ والسياق"""

        suggestions = []

        if context.active_domain:
            domain_suggestions = self._get_domain_suggestions(context.active_domain)
            suggestions.extend(domain_suggestions)

        history_suggestions = self._get_history_suggestions()
        suggestions.extend(history_suggestions)

        if context.conversation_history:
            context_suggestions = self._analyze_conversation_context(context.conversation_history)
            suggestions.extend(context_suggestions)

        seen = set()
        unique = []
        for s in suggestions:
            key = s.get("text", "")
            if key not in seen:
                seen.add(key)
                unique.append(s)

        return sorted(unique, key=lambda x: x.get("confidence", 0), reverse=True)[:4]

    def _get_domain_suggestions(self, domain: str) -> list[dict[str, Any]]:
        """اقتراحات بناءً على التخصص"""

        domain_suggestions_map = {
            "python": [
                {"text": "إعداد بيئة Python", "domain": "python", "confidence": 0.7},
                {"text": "تثبيت مكتبة", "domain": "python", "confidence": 0.6},
            ],
            "database": [
                {"text": "إعداد قاعدة البيانات", "domain": "database", "confidence": 0.7},
                {"text": "عمل migration", "domain": "database", "confidence": 0.7},
                {"text": "تحسين استعلامات", "domain": "database", "confidence": 0.6},
            ],
            "devops": [
                {"text": "إعداد Docker", "domain": "devops", "confidence": 0.7},
                {"text": "تصحيح Kubernetes", "domain": "devops", "confidence": 0.6},
            ],
            "api": [
                {"text": "إنشاء endpoint جديد", "domain": "api", "confidence": 0.7},
                {"text": "تصحيح طلبات API", "domain": "api", "confidence": 0.6},
            ],
        }

        return domain_suggestions_map.get(domain, [])

    def _get_history_suggestions(self) -> list[dict[str, Any]]:
        """اقتراحات من تاريخ الاستعلامات"""

        suggestions = []

        for query in reversed(self._query_history[-5:]):
            if len(query) > 10:
                domain = self._detect_domain_from_keywords(query.lower().split())
                suggestions.append(
                    {
                        "text": f"متابعة: {query[:50]}...",
                        "domain": domain or "general",
                        "confidence": 0.5,
                    }
                )

        return suggestions[:2]

    def _analyze_conversation_context(
        self,
        history: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """تحليل سياق المحادثة للاقتراحات"""

        suggestions = []

        recent_messages = history[-3:] if history else []

        for msg in recent_messages:
            content = msg.get("content", "").lower()
            domain = self._detect_domain_from_keywords(content.split())

            if domain:
                suggestions.append(
                    {
                        "text": f"توضيح بخصوص {domain}",
                        "domain": domain,
                        "confidence": 0.6,
                    }
                )

        return suggestions[:2]

    async def _determine_domain(
        self,
        query: str,
        context: RetrievalContext,
        analysis: dict[str, Any],
    ) -> str:
        """تحديد التخصص"""

        if self._specialist_agent:
            try:
                decision = await self._specialist_agent.determine_domain(
                    query,
                    {"analysis": analysis, "context": context.session_context},
                )
                domain: str = decision.domain
                return domain
            except Exception as e:
                self._logger.warning(f"Specialist agent failed: {e}")

        if context.active_domain:
            return context.active_domain

        if analysis.get("implied_domain"):
            implied_domain = analysis.get("implied_domain")
            if implied_domain:
                return str(implied_domain)

        if self._domain_history:
            return self._domain_history[-1]

        return "general"

    async def _multi_stage_retrieve(
        self,
        query: str,
        domain: str,
        context: RetrievalContext,
    ) -> list[dict[str, Any]]:
        """استرجاع متعدد المراحل"""

        candidates = []

        if self._vector_store:
            filter_meta = {"domain": domain} if domain != "general" else None

            try:
                results = self._vector_store.search(
                    query=query,
                    n_results=context.max_results * 3,
                    filter_meta=filter_meta,
                )

                for r in results:
                    candidates.append(
                        {
                            "content": getattr(r, "content", str(r)),
                            "metadata": getattr(r, "metadata", {}),
                            "source": "vector_store",
                            "original_score": 1.0,
                        }
                    )
            except Exception as e:
                self._logger.warning(f"Vector store search failed: {e}")

        if context.include_related and self._knowledge_graph and candidates:
            try:
                related = self._get_related_from_graph(candidates[:3])
                candidates.extend(related)
            except Exception as e:
                self._logger.debug(f"Knowledge graph lookup failed: {e}")

        return candidates

    def _get_related_from_graph(
        self,
        top_candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """الحصول على المحتوى المرتبط من الرسم المعرفي"""

        related = []

        for candidate in top_candidates:
            node_id = candidate.get("metadata", {}).get("node_id")

            if node_id and self._knowledge_graph:
                neighbors = self._knowledge_graph.get_neighbors(node_id, depth=1)

                for neighbor_id in neighbors:
                    related.append(
                        {
                            "content": f"Related: {neighbor_id}",
                            "metadata": {"node_id": neighbor_id},
                            "source": "knowledge_graph",
                            "original_score": 0.5,
                        }
                    )

        return related[:5]

    async def _rerank_results(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        context: RetrievalContext,
    ) -> list[dict[str, Any]]:
        """إعادة ترتيب النتائج"""

        if not candidates:
            return []

        if not self._reranker:
            return sorted(candidates, key=lambda x: x.get("original_score", 0), reverse=True)

        try:
            from gaap.memory.rerankers import RerankRequest

            request = RerankRequest(
                query=query,
                candidates=[c["content"] for c in candidates],
                top_k=context.max_results,
                context={"domain": context.active_domain},
                metadata_list=[c.get("metadata", {}) for c in candidates],
            )

            reranked = await self._reranker.rerank(request)

            results = []
            for r in reranked:
                idx = request.candidates.index(r.content) if r.content in request.candidates else -1
                original = candidates[idx] if idx >= 0 else {}

                results.append(
                    {
                        "content": r.content,
                        "score": r.score,
                        "original_score": r.original_score,
                        "rank": r.rank,
                        "source": r.source,
                        "reasoning": r.reasoning,
                        "metadata": original.get("metadata", {}),
                    }
                )

            return results

        except Exception as e:
            self._logger.warning(f"Reranking failed: {e}")
            return candidates

    def _parse_analysis(self, response: str) -> dict[str, Any]:
        """تحليل استجابة LLM للتحليل"""

        import json

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                result: dict[str, Any] = json.loads(response[json_start:json_end])
                return result
        except Exception:
            pass

        return {"type": "SPECIFIC", "confidence": 0.5, "is_ambiguous": False}

    def _parse_clarification(self, response: str) -> dict[str, Any]:
        """تحليل استجابة LLM للتوضيح"""

        import json

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                result: dict[str, Any] = json.loads(response[json_start:json_end])
                return result
        except Exception:
            pass

        return {
            "clarification_question": "مش فاهم بالظبط تقصد إيه؟",
            "suggestions": [],
        }

    def _format_conversation_context(self, context: RetrievalContext) -> str:
        """تنسيق سياق المحادثة"""

        if not context.conversation_history:
            return "No recent conversation."

        parts = []
        for msg in context.conversation_history[-3:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")[:100]
            parts.append(f"{role}: {content}")

        return "\n".join(parts)

    def _format_query_history(self) -> str:
        """تنسيق تاريخ الاستعلامات"""

        if not self._query_history:
            return "No previous queries."

        return "\n".join(f"- {q}" for q in self._query_history[-5:])

    def _get_sources(self, candidates: list[dict[str, Any]]) -> list[str]:
        """الحصول على مصادر النتائج"""

        sources = set()
        for c in candidates:
            source = c.get("source", "unknown")
            sources.add(source)
        return list(sources)

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الوكيل"""

        return {
            "total_queries": len(self._query_history),
            "recent_domains": self._domain_history[-5:],
            "has_vector_store": self._vector_store is not None,
            "has_reranker": self._reranker is not None,
            "has_knowledge_graph": self._knowledge_graph is not None,
            "has_specialist_agent": self._specialist_agent is not None,
        }
