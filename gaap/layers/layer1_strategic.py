# Layer 1: Strategic Layer
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from gaap.core.base import BaseLayer
from gaap.core.types import LayerType, Message, MessageRole
from gaap.layers.layer0_interface import StructuredIntent
from gaap.mad.critic_prompts import (
    ARCH_SYSTEM_PROMPTS,
    ArchitectureCriticType,
    build_architecture_prompt,
)
from gaap.mad.response_parser import (
    CriticParseError,
    fallback_architecture_evaluation,
    parse_architecture_critic_response,
)

# =============================================================================
# Logger Setup
# =============================================================================


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# =============================================================================
# Enums
# =============================================================================


class ArchitectureParadigm(Enum):
    """أنماط البنية المعمارية"""

    MONOLITH = "monolith"
    MODULAR_MONOLITH = "modular_monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"


class DataStrategy(Enum):
    """استراتيجيات البيانات"""

    SINGLE_DB = "single_database"
    POLYGLOT = "polyglot"
    CQRS = "cqrs"
    EVENT_SOURCING = "event_sourcing"


class CommunicationPattern(Enum):
    """أنماط الاتصال"""

    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    EVENT_BUS = "event_bus"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ArchitectureDecision:
    """قرار معماري"""

    aspect: str  # الجانب (مثل: paradigm, database, etc.)
    choice: str  # الاختيار
    reasoning: str  # السبب
    trade_offs: list[str]  # التنازلات
    confidence: float  # الثقة


@dataclass
class ArchitectureSpec:
    """مواصفات معمارية"""

    spec_id: str
    timestamp: datetime

    # القرارات الأساسية
    paradigm: ArchitectureParadigm = ArchitectureParadigm.MODULAR_MONOLITH
    data_strategy: DataStrategy = DataStrategy.SINGLE_DB
    communication: CommunicationPattern = CommunicationPattern.REST

    # المكونات
    components: list[dict[str, Any]] = field(default_factory=list)

    # التقنيات
    tech_stack: dict[str, str] = field(default_factory=dict)

    # القرارات
    decisions: list[ArchitectureDecision] = field(default_factory=list)

    # المخاطر
    risks: list[dict[str, Any]] = field(default_factory=list)

    # الموارد
    estimated_resources: dict[str, Any] = field(default_factory=dict)

    # الـ ToT
    explored_paths: int = 0
    selected_path_score: float = 0.0

    # الـ MAD
    debate_rounds: int = 0
    consensus_reached: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "paradigm": self.paradigm.value,
            "data_strategy": self.data_strategy.value,
            "communication": self.communication.value,
            "components": self.components,
            "tech_stack": self.tech_stack,
            "decisions": [
                {"aspect": d.aspect, "choice": d.choice, "reasoning": d.reasoning}
                for d in self.decisions
            ],
            "risks": self.risks,
            "consensus": self.consensus_reached,
        }


@dataclass
class ToTNode:
    """عقدة في شجرة الأفكار"""

    id: str
    level: int
    content: str
    score: float = 0.0
    children: list["ToTNode"] = field(default_factory=list)
    parent: Optional["ToTNode"] = None
    explored: bool = False
    pruned: bool = False


# =============================================================================
# Tree of Thoughts Strategic
# =============================================================================


class ToTStrategic:
    """
    شجرة الأفكار الاستراتيجية

    تستكشف فضاء الحلول على 5 مستويات:
    - L0: النمط المعماري
    - L1: استراتيجية البيانات
    - L2: نمط الاتصال
    - L3: البنية التحتية
    - L4: المراقبة والأمان
    """

    def __init__(self, max_depth: int = 5, branching_factor: int = 4):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self._logger = get_logger("gaap.layer1.tot")
        self._explored_nodes = 0

    async def explore(
        self, intent: StructuredIntent, context: dict[str, Any] | None = None
    ) -> tuple[ArchitectureSpec, ToTNode]:
        """استكشاف فضاء الحلول"""

        # إنشاء الجذر
        root = ToTNode(
            id="root",
            level=0,
            content=intent.explicit_goals[0] if intent.explicit_goals else "Design system",
        )

        # بناء الشجرة
        await self._build_tree(root, intent, context)

        # تقييم واختيار أفضل مسار
        best_path = self._select_best_path(root)

        # تحويل المواصفات
        spec = self._path_to_spec(best_path, intent)
        spec.explored_paths = self._explored_nodes

        return spec, root

    async def _build_tree(
        self, node: ToTNode, intent: StructuredIntent, context: dict[str, Any] | None
    ) -> None:
        """بناء الشجرة"""
        if node.level >= self.max_depth:
            return

        # توليد الخيارات
        options = self._generate_options(node.level, intent)

        # تقييم وتقليم
        scored_options = []
        for opt in options[: self.branching_factor]:
            score = self._evaluate_option(opt, node.level, intent)
            if score > 0.3:  # تقليم الدرجات المنخفضة
                child = ToTNode(
                    id=f"{node.id}_{len(node.children)}",
                    level=node.level + 1,
                    content=opt,
                    score=score,
                    parent=node,
                )
                node.children.append(child)
                scored_options.append((child, score))
                self._explored_nodes += 1

        # ترتيب واستكشاف أفضل الخيارات
        scored_options.sort(key=lambda x: x[1], reverse=True)

        for child, _ in scored_options[:2]:  # استكشاف أفضل 2
            child.explored = True
            await self._build_tree(child, intent, context)

    def _generate_options(self, level: int, intent: StructuredIntent) -> list[str]:
        """توليد الخيارات حسب المستوى"""
        options_map: dict[int, list[Any]] = {
            0: list(ArchitectureParadigm),
            1: list(DataStrategy),
            2: list(CommunicationPattern),
            3: ["kubernetes", "docker", "serverless", "vm"],
            4: ["prometheus", "datadog", "cloudwatch", "custom"],
        }

        options = options_map.get(level, [])
        return [opt.value if hasattr(opt, "value") else str(opt) for opt in options]

    def _evaluate_option(self, option: str, level: int, intent: StructuredIntent) -> float:
        """تقييم خيار"""
        score = 0.5  # درجة أساسية

        implicit = intent.implicit_requirements

        # تعديلات بناءً على المتطلبات
        if level == 0:  # النمط المعماري
            if implicit.scalability and option == "microservices":
                score += 0.3
            if implicit.budget == "budget_conscious" and option in ("monolith", "modular_monolith"):
                score += 0.2
            if len(intent.explicit_goals) > 3 and option == "microservices":
                score += 0.1

        elif level == 1:  # استراتيجية البيانات
            if implicit.scalability and option == "cqrs":
                score += 0.2
            if implicit.performance == "real_time" and option == "event_sourcing":
                score += 0.2

        elif level == 2:  # نمط الاتصال
            if implicit.performance == "high_throughput" and option == "grpc":
                score += 0.3
            if option == "rest":
                score += 0.1  # REST هو الافتراضي الآمن

        return min(score, 1.0)

    def _select_best_path(self, root: ToTNode) -> list[ToTNode]:
        """اختيار أفضل مسار"""
        path = [root]
        current = root

        while current.children:
            # اختيار أفضل طفل
            best_child = max(current.children, key=lambda x: x.score)
            path.append(best_child)
            current = best_child

        return path

    def _path_to_spec(self, path: list[ToTNode], intent: StructuredIntent) -> ArchitectureSpec:
        """تحويل المسار لمواصفات"""
        spec = ArchitectureSpec(spec_id=f"spec_{int(time.time() * 1000)}", timestamp=datetime.now())

        # استخراج القرارات من المسار
        for node in path[1:]:  # تخطي الجذر
            level = node.level
            content = node.content

            if level == 1:
                with contextlib.suppress(BaseException):
                    spec.paradigm = ArchitectureParadigm(content)
            elif level == 2:
                with contextlib.suppress(BaseException):
                    spec.data_strategy = DataStrategy(content)
            elif level == 3:
                with contextlib.suppress(BaseException):
                    spec.communication = CommunicationPattern(content)

        # إضافة قرار
        spec.decisions.append(
            ArchitectureDecision(
                aspect="architecture_paradigm",
                choice=spec.paradigm.value,
                reasoning=f"Selected based on requirements: {intent.implicit_requirements.scalability or 'balanced'}",
                trade_offs=["Complexity trade-off", "Team expertise required"],
                confidence=0.85,
            )
        )

        spec.selected_path_score = sum(n.score for n in path) / len(path) if path else 0

        return spec


# =============================================================================
# MAD Architecture Panel
# =============================================================================


class MADArchitecturePanel:
    """
    لجنة الجدل المعماري (Multi-Agent Debate)

    تتكون من 4+ نقاد بمنظورات مختلفة:
    - Scalability Critic: التوسع
    - Pragmatism Critic: الواقعية
    - Cost Critic: التكلفة
    - Robustness Critic: المتانة
    - Maintainability Critic: الصيانة
    - Security Architecture Critic: الأمان
    """

    ARCH_CRITICS = [
        ArchitectureCriticType.SCALABILITY,
        ArchitectureCriticType.PRAGMATISM,
        ArchitectureCriticType.COST,
        ArchitectureCriticType.ROBUSTNESS,
    ]

    def __init__(
        self,
        max_rounds: int = 3,
        consensus_threshold: float = 0.85,
        provider: Any = None,
        critic_model: str | None = None,
    ) -> None:
        self.max_rounds = max_rounds
        self.consensus_threshold = consensus_threshold
        self.provider = provider
        self.critic_model = critic_model or "llama-3.3-70b-versatile"
        self._logger = get_logger("gaap.layer1.mad")
        self._llm_failures = 0

    async def debate(
        self, spec: ArchitectureSpec, intent: StructuredIntent
    ) -> tuple[ArchitectureSpec, bool]:
        """إجراء الجدل"""

        for round_num in range(self.max_rounds):
            # تقييم كل ناقد
            evaluations = await self._evaluate_all(spec, intent, round_num)

            # حساب الإجماع
            avg_score = sum(e["score"] for e in evaluations) / len(evaluations)

            # هل وصلنا لإجماع؟
            if avg_score >= self.consensus_threshold:
                spec.consensus_reached = True
                spec.debate_rounds = round_num + 1

                self._logger.info(
                    f"MAD consensus reached at round {round_num + 1}: score={avg_score:.2f}"
                )

                return spec, True

            # تحسين بناءً على النقد
            spec = self._apply_critiques(spec, evaluations)

        spec.debate_rounds = self.max_rounds
        spec.consensus_reached = False

        return spec, False

    async def _evaluate_all(
        self, spec: ArchitectureSpec, intent: StructuredIntent, round_num: int
    ) -> list[dict[str, Any]]:
        """تقييم جميع النقاد - يستخدم LLM إذا توفر"""
        if self.provider is None:
            return self._evaluate_all_fallback(spec, intent)

        evaluations = []
        for critic_type in self.ARCH_CRITICS:
            result = await self._evaluate_with_llm(spec, intent, critic_type)
            evaluations.append(result)

        return evaluations

    async def _evaluate_with_llm(
        self, spec: ArchitectureSpec, intent: StructuredIntent, critic_type: ArchitectureCriticType
    ) -> dict[str, Any]:
        """تقييم بناقد محدد باستخدام LLM"""
        try:
            system_prompt = ARCH_SYSTEM_PROMPTS.get(
                critic_type, ARCH_SYSTEM_PROMPTS[ArchitectureCriticType.SCALABILITY]
            )
            user_prompt = build_architecture_prompt(spec, intent)

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt),
                Message(role=MessageRole.USER, content=user_prompt),
            ]

            response = await self.provider.chat_completion(
                messages=messages,
                model=self.critic_model,
                temperature=0.3,
                max_tokens=2048,
            )

            if not response.choices or not response.choices[0].message.content:
                self._logger.warning(f"LLM call failed for {critic_type.name}, using fallback")
                self._llm_failures += 1
                return self._get_fallback_eval(spec, intent, critic_type)

            parsed = parse_architecture_critic_response(
                response.choices[0].message.content, critic_type
            )

            return {
                "critic": critic_type.name.lower(),
                "score": parsed["score"] / 100.0,
                "issues": parsed["issues"],
                "suggestions": parsed["suggestions"],
                "reasoning": parsed["reasoning"],
            }

        except CriticParseError as e:
            self._logger.warning(f"Parse error for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._get_fallback_eval(spec, intent, critic_type)
        except Exception as e:
            self._logger.warning(f"LLM evaluation failed for {critic_type.name}: {e}")
            self._llm_failures += 1
            return self._get_fallback_eval(spec, intent, critic_type)

    def _get_fallback_eval(
        self, spec: ArchitectureSpec, intent: StructuredIntent, critic_type: ArchitectureCriticType
    ) -> dict[str, Any]:
        """تقييم احتياطي"""
        result = fallback_architecture_evaluation(critic_type, spec, intent)
        return {
            "critic": critic_type.name.lower(),
            "score": result["score"] / 100.0,
            "issues": result["issues"],
            "suggestions": result["suggestions"],
            "reasoning": result["reasoning"],
        }

    def _evaluate_all_fallback(
        self, spec: ArchitectureSpec, intent: StructuredIntent
    ) -> list[dict[str, Any]]:
        """تقييم جميع النقاد بالـ fallback (بدون LLM)"""
        evaluations = []
        evaluations.append(self._scalability_eval(spec, intent))
        evaluations.append(self._pragmatism_eval(spec, intent))
        evaluations.append(self._cost_eval(spec, intent))
        evaluations.append(self._robustness_eval(spec, intent))
        return evaluations

    def _scalability_eval(self, spec: ArchitectureSpec, intent: StructuredIntent) -> dict[str, Any]:
        """تقييم التوسع"""
        score = 0.5
        issues = []

        if spec.paradigm == ArchitectureParadigm.MICROSERVICES:
            score += 0.3
        elif spec.paradigm == ArchitectureParadigm.MONOLITH:
            score -= 0.2
            issues.append("Monolith may limit horizontal scaling")

        if spec.data_strategy == DataStrategy.CQRS:
            score += 0.1

        return {
            "critic": "scalability",
            "score": min(max(score, 0), 1),
            "issues": issues,
            "suggestions": ["Consider horizontal scaling patterns"] if score < 0.7 else [],
        }

    def _pragmatism_eval(self, spec: ArchitectureSpec, intent: StructuredIntent) -> dict[str, Any]:
        """تقييم الواقعية"""
        score = 0.7  # افتراضي جيد
        issues = []

        # التحقق من التعقيد الزائد
        if spec.paradigm == ArchitectureParadigm.MICROSERVICES:
            if intent.implicit_requirements.budget == "budget_conscious":
                score -= 0.3
                issues.append("Microservices may be over-engineering for budget constraints")

        # التحقق من الجدول الزمني
        if intent.implicit_requirements.timeline == "urgent":
            if spec.paradigm != ArchitectureParadigm.MODULAR_MONOLITH:
                score -= 0.1
                issues.append("Modular monolith may be faster to implement")

        return {
            "critic": "pragmatism",
            "score": min(max(score, 0), 1),
            "issues": issues,
            "suggestions": ["Start simple, evolve later"] if score < 0.7 else [],
        }

    def _cost_eval(self, spec: ArchitectureSpec, intent: StructuredIntent) -> dict[str, Any]:
        """تقييم التكلفة"""
        score = 0.6
        issues = []

        # تقدير التكلفة النسبية
        cost_factors = {
            ArchitectureParadigm.SERVERLESS: 0.7,
            ArchitectureParadigm.MONOLITH: 0.8,
            ArchitectureParadigm.MODULAR_MONOLITH: 0.75,
            ArchitectureParadigm.MICROSERVICES: 0.4,
        }

        score = cost_factors.get(spec.paradigm, 0.5)

        if intent.implicit_requirements.budget == "budget_conscious":
            if spec.paradigm == ArchitectureParadigm.MICROSERVICES:
                issues.append("High operational costs expected")

        return {
            "critic": "cost",
            "score": score,
            "issues": issues,
            "suggestions": ["Consider managed services to reduce ops cost"] if score < 0.6 else [],
        }

    def _robustness_eval(self, spec: ArchitectureSpec, intent: StructuredIntent) -> dict[str, Any]:
        """تقييم المتانة"""
        score = 0.6
        issues: list[str] = []

        if intent.implicit_requirements and intent.implicit_requirements.security:
            score += 0.2

        if spec.communication in (
            CommunicationPattern.MESSAGE_QUEUE,
            CommunicationPattern.EVENT_BUS,
        ):
            score += 0.1  # أفضل للتحمل

        return {
            "critic": "robustness",
            "score": min(max(score, 0), 1),
            "issues": issues,
            "suggestions": ["Add circuit breakers", "Implement retry logic"] if score < 0.7 else [],
        }

    def _apply_critiques(
        self, spec: ArchitectureSpec, evaluations: list[dict[str, Any]]
    ) -> ArchitectureSpec:
        """تطبيق النقادات"""

        # جمع الاقتراحات
        all_suggestions = []
        for eval_result in evaluations:
            if eval_result["score"] < 0.7:
                all_suggestions.extend(eval_result.get("suggestions", []))

        # إضافة المخاطر
        for eval_result in evaluations:
            for issue in eval_result.get("issues", []):
                spec.risks.append(
                    {
                        "source": eval_result["critic"],
                        "issue": issue,
                        "severity": "medium" if eval_result["score"] > 0.5 else "high",
                    }
                )

        return spec


# =============================================================================
# Strategic Planner
# =============================================================================


class StrategicPlanner:
    """المخطط الاستراتيجي"""

    def __init__(self) -> None:
        self._logger = get_logger("gaap.layer1.planner")

    async def create_plan(self, spec: ArchitectureSpec, intent: StructuredIntent) -> dict[str, Any]:
        """إنشاء خطة"""
        plan: dict[str, Any] = {
            "phases": [],
            "milestones": [],
            "resource_allocation": {},
        }

        # المراحل
        phases = self._determine_phases(spec, intent)
        plan["phases"] = phases

        # المعالم
        milestones = self._create_milestones(phases)
        plan["milestones"] = milestones

        return plan

    def _determine_phases(
        self, spec: ArchitectureSpec, intent: StructuredIntent
    ) -> list[dict[str, Any]]:
        """تحديد المراحل"""
        phases = [
            {
                "name": "Foundation",
                "tasks": ["Setup project structure", "Configure build system"],
                "duration": "1-2 days",
            },
            {
                "name": "Core Development",
                "tasks": ["Implement core components", "Database setup"],
                "duration": "1-2 weeks",
            },
            {
                "name": "Integration",
                "tasks": ["API integration", "Authentication"],
                "duration": "3-5 days",
            },
            {
                "name": "Testing & QA",
                "tasks": ["Unit tests", "Integration tests"],
                "duration": "3-5 days",
            },
        ]

        return phases

    def _create_milestones(self, phases: list[dict]) -> list[dict]:
        """إنشاء معالم"""
        return [{"name": f"{p['name']} Complete", "phase": p["name"]} for p in phases]


# =============================================================================
# Layer 1 Strategic
# =============================================================================


class Layer1Strategic(BaseLayer):
    """
    طبقة التخطيط الاستراتيجي (LLM-Powered)

    المسؤوليات:
    - تحويل الطلب المبهم إلى خطة معمارية مخصصة
    - استخدام الـ LLM لتحليل الطلب واقتراح المعمارية الأنسب
    - Fallback ذكي عبر ToT/MAD إذا فشل الـ LLM
    - توليد مواصفات معمارية مرتبطة بالطلب الفعلي
    """

    STRATEGY_PROMPT = """You are an expert software architect. Analyze the user's request and produce an architecture specification.

## User Request
{original_text}

## Intent Analysis
- Intent Type: {intent_type}
- Goals: {goals}
- Constraints: {constraints}

## Instructions
Analyze the request and determine the best architecture approach. Consider:
1. The scale and scope of what's being asked
2. Whether this is a simple utility, a library, a service, or a full system
3. What technologies and patterns are most appropriate
4. What risks exist

For SIMPLE requests (single function, script, algorithm), use minimal architecture.
For COMPLEX requests (full systems, services), use more sophisticated architecture.

## Output Format
Return ONLY a valid JSON object with these fields:
```json
{{
  "paradigm": "one of: monolith|modular_monolith|microservices|serverless|event_driven|layered|hexagonal",
  "data_strategy": "one of: single_database|polyglot|cqrs|event_sourcing",
  "communication": "one of: rest|graphql|grpc|message_queue|event_bus",
  "tech_stack": {{
    "language": "primary language",
    "framework": "framework if applicable",
    "tools": ["relevant tools"]
  }},
  "components": [
    {{
      "name": "Component name",
      "responsibility": "What it does",
      "type": "module|service|library|util"
    }}
  ],
  "decisions": [
    {{
      "aspect": "What was decided",
      "choice": "The decision",
      "reasoning": "Why this choice"
    }}
  ],
  "risks": [
    {{
      "issue": "Potential risk",
      "severity": "low|medium|high",
      "mitigation": "How to mitigate"
    }}
  ],
  "phases": [
    {{
      "name": "Phase name",
      "tasks": ["Specific tasks for THIS request"],
      "duration": "Estimated duration"
    }}
  ],
  "complexity_score": 0.5,
  "estimated_time": "Realistic time estimate"
}}
```

CRITICAL: Your response must be SPECIFIC to \"{original_text}\". Do NOT generate generic architecture. A request for a simple function should get a simple spec with 1-2 components. A request for a full system should get a comprehensive spec.

Return ONLY the JSON object, no markdown fences, no explanation."""

    PARADIGM_MAP = {
        "monolith": ArchitectureParadigm.MONOLITH,
        "modular_monolith": ArchitectureParadigm.MODULAR_MONOLITH,
        "microservices": ArchitectureParadigm.MICROSERVICES,
        "serverless": ArchitectureParadigm.SERVERLESS,
        "event_driven": ArchitectureParadigm.EVENT_DRIVEN,
        "layered": ArchitectureParadigm.LAYERED,
        "hexagonal": ArchitectureParadigm.HEXAGONAL,
    }

    DATA_STRATEGY_MAP = {
        "single_database": DataStrategy.SINGLE_DB,
        "polyglot": DataStrategy.POLYGLOT,
        "cqrs": DataStrategy.CQRS,
        "event_sourcing": DataStrategy.EVENT_SOURCING,
    }

    COMM_MAP = {
        "rest": CommunicationPattern.REST,
        "graphql": CommunicationPattern.GRAPHQL,
        "grpc": CommunicationPattern.GRPC,
        "message_queue": CommunicationPattern.MESSAGE_QUEUE,
        "event_bus": CommunicationPattern.EVENT_BUS,
    }

    def __init__(
        self,
        tot_depth: int = 5,
        tot_branching: int = 4,
        mad_rounds: int = 3,
        consensus_threshold: float = 0.85,
        provider: Any = None,
    ) -> None:
        super().__init__(LayerType.STRATEGIC)

        self._provider = provider

        # Fallback components (used when LLM is unavailable)
        self.tot = ToTStrategic(max_depth=tot_depth, branching_factor=tot_branching)
        self.mad_panel = MADArchitecturePanel(
            max_rounds=mad_rounds, consensus_threshold=consensus_threshold
        )
        self.planner = StrategicPlanner()

        self._logger = get_logger("gaap.layer1")

        # الإحصائيات
        self._specs_created = 0
        self._llm_strategies = 0
        self._fallback_strategies = 0

    async def process(self, input_data: Any) -> ArchitectureSpec:
        """معالجة المدخل"""
        start_time = time.time()

        # استخراج الـ Intent
        if isinstance(input_data, StructuredIntent):
            intent = input_data
        else:
            raise ValueError("Expected StructuredIntent from Layer 0")

        self._logger.info(f"Strategic planning for request {intent.request_id}")

        original_text = intent.metadata.get("original_text", "")

        # محاولة التخطيط بالـ LLM أولاً
        spec = None
        source = "fallback"

        if self._provider and original_text:
            try:
                spec = await self._llm_strategize(intent, original_text)
                if spec:
                    source = "llm"
                    self._llm_strategies += 1
                    self._logger.info("LLM strategy generation successful")
            except Exception as e:
                self._logger.warning(f"LLM strategy failed, using fallback: {e}")

        # Fallback: ToT + MAD + Planner
        if spec is None:
            spec, tree = await self.tot.explore(intent)
            spec, consensus = await self.mad_panel.debate(spec, intent)
            plan = await self.planner.create_plan(spec, intent)
            spec.metadata["plan"] = plan
            self._fallback_strategies += 1
            source = "fallback"

        # حفظ بيانات النية الأصلية لاستخدامها في L2
        spec.metadata["original_intent"] = {
            "request_id": intent.request_id,
            "intent_type": intent.intent_type.name,
            "explicit_goals": intent.explicit_goals,
            "constraints": intent.constraints,
            "context_snapshot": intent.context_snapshot,
            "recommended_tools": intent.recommended_tools,
            "original_text": original_text,
        }
        spec.metadata["strategy_source"] = source

        # تقدير الموارد (إذا لم يحددها الـ LLM)
        if not spec.estimated_resources:
            spec.estimated_resources = self._estimate_resources(spec, intent)

        self._specs_created += 1

        elapsed = (time.time() - start_time) * 1000
        self._logger.info(
            f"Architecture spec created: paradigm={spec.paradigm.value}, "
            f"source={source}, time={elapsed:.0f}ms"
        )

        return spec

    async def _llm_strategize(
        self, intent: StructuredIntent, original_text: str
    ) -> ArchitectureSpec | None:
        """تخطيط استراتيجي باستخدام الـ LLM"""
        prompt = self.STRATEGY_PROMPT.format(
            original_text=original_text,
            intent_type=intent.intent_type.name,
            goals=", ".join(intent.explicit_goals) if intent.explicit_goals else "Not specified",
            constraints=", ".join(intent.constraints) if intent.constraints else "None",
        )

        messages = [
            Message(
                role=MessageRole.SYSTEM,
                content="You are a precise software architecture engine. Output only valid JSON objects.",
            ),
            Message(role=MessageRole.USER, content=prompt),
        ]

        response = await self._provider.chat_completion(
            messages=messages,
            model=self._provider.default_model,
        )

        raw = response.choices[0].message.content
        data = self._parse_llm_response(raw)

        if not data:
            self._logger.warning("Failed to parse LLM strategy response")
            return None

        return self._build_spec_from_llm(data, intent)

    def _parse_llm_response(self, raw: str) -> dict[str, Any] | None:
        """تنظيف واستخراج JSON من رد الـ LLM"""
        if not raw:
            return None

        import re as _re

        cleaned = raw.strip()

        # إزالة thinking tags
        cleaned = _re.sub(r"<think>.*?</think>", "", cleaned, flags=_re.DOTALL).strip()

        # إزالة markdown code fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            start_idx = 1
            end_idx = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end_idx = i
                    break
            cleaned = "\n".join(lines[start_idx:end_idx])

        # محاولة parse مباشرة
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        # استخراج JSON object من النص
        import re as _re2

        brace_match = _re2.search(r"\{[\s\S]*\}", cleaned)
        if brace_match:
            try:
                data = json.loads(brace_match.group())
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass

        self._logger.error(f"Could not parse LLM strategy response: {cleaned[:200]}...")
        return None

    def _build_spec_from_llm(
        self, data: dict[str, Any], intent: StructuredIntent
    ) -> ArchitectureSpec:
        """بناء ArchitectureSpec من بيانات الـ LLM"""
        spec = ArchitectureSpec(spec_id=f"spec_{int(time.time() * 1000)}", timestamp=datetime.now())

        # النمط المعماري
        paradigm_str = str(data.get("paradigm", "modular_monolith")).lower().strip()
        spec.paradigm = self.PARADIGM_MAP.get(paradigm_str, ArchitectureParadigm.MODULAR_MONOLITH)

        # استراتيجية البيانات
        ds_str = str(data.get("data_strategy", "single_database")).lower().strip()
        spec.data_strategy = self.DATA_STRATEGY_MAP.get(ds_str, DataStrategy.SINGLE_DB)

        # نمط الاتصال
        comm_str = str(data.get("communication", "rest")).lower().strip()
        spec.communication = self.COMM_MAP.get(comm_str, CommunicationPattern.REST)

        # Tech stack
        tech = data.get("tech_stack", {})
        if isinstance(tech, dict):
            spec.tech_stack = tech

        # Components
        components = data.get("components", [])
        if isinstance(components, list):
            spec.components = components

        # Decisions
        decisions = data.get("decisions", [])
        if isinstance(decisions, list):
            for d in decisions:
                if isinstance(d, dict):
                    spec.decisions.append(
                        ArchitectureDecision(
                            aspect=str(d.get("aspect", "general")),
                            choice=str(d.get("choice", "")),
                            reasoning=str(d.get("reasoning", "")),
                            trade_offs=d.get("trade_offs", []),
                            confidence=float(d.get("confidence", 0.8)),
                        )
                    )

        # Risks
        risks = data.get("risks", [])
        if isinstance(risks, list):
            spec.risks = risks

        # Phases → plan
        phases = data.get("phases", [])
        if isinstance(phases, list) and phases:
            spec.metadata["plan"] = {
                "phases": phases,
                "milestones": [
                    {"name": f"{p.get('name', 'Phase')} Complete", "phase": p.get("name", "")}
                    for p in phases
                    if isinstance(p, dict)
                ],
            }

        # Resource estimates from LLM
        complexity = data.get("complexity_score", 0.5)
        est_time = data.get("estimated_time", "1-2 weeks")
        spec.estimated_resources = {
            "complexity_score": float(complexity) if isinstance(complexity, (int, float)) else 0.5,
            "estimated_time": str(est_time),
        }

        # Mark as LLM-generated
        spec.explored_paths = 1
        spec.selected_path_score = 0.9
        spec.consensus_reached = True
        spec.debate_rounds = 1

        return spec

    def _estimate_resources(
        self, spec: ArchitectureSpec, intent: StructuredIntent
    ) -> dict[str, Any]:
        """تقدير الموارد (fallback)"""
        estimates = {
            "estimated_time": "2-4 weeks",
            "complexity_score": 0.7,
            "team_size_recommendation": "2-3 developers",
        }

        if spec.paradigm == ArchitectureParadigm.MICROSERVICES:
            estimates["estimated_time"] = "4-8 weeks"
            estimates["team_size_recommendation"] = "4-6 developers"
            estimates["complexity_score"] = 0.9

        return estimates

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "layer": "L1_Strategic",
            "specs_created": self._specs_created,
            "llm_strategies": self._llm_strategies,
            "fallback_strategies": self._fallback_strategies,
            "tot_nodes_explored": self.tot._explored_nodes,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_strategic_layer(
    tot_depth: int = 5, mad_rounds: int = 3, provider: Any = None
) -> Layer1Strategic:
    """إنشاء طبقة استراتيجية"""
    return Layer1Strategic(tot_depth=tot_depth, mad_rounds=mad_rounds, provider=provider)
