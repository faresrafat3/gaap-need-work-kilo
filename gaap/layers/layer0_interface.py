# Layer 0: Interface Layer
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from gaap.core.base import BaseLayer
from gaap.core.types import LayerType, TaskComplexity
from gaap.security.firewall import PromptFirewall

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


class IntentType(Enum):
    """أنواع النوايا"""

    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    DEBUGGING = auto()
    REFACTORING = auto()
    DOCUMENTATION = auto()
    TESTING = auto()
    RESEARCH = auto()
    ANALYSIS = auto()
    PLANNING = auto()
    QUESTION = auto()
    CONVERSATION = auto()
    UNKNOWN = auto()


class RoutingTarget(Enum):
    """أهداف التوجيه"""

    STRATEGIC = "layer1_strategic"  # مهمة معقدة تحتاج تخطيط
    TACTICAL = "layer2_tactical"  # مهمة واضحة تحتاج تفصيل
    DIRECT = "layer3_execution"  # مهمة بسيطة可以直接 تنفيذ


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ImplicitRequirements:
    """المتطلبات الضمنية المستخرجة"""

    performance: str | None = None
    security: str | None = None
    scalability: str | None = None
    compliance: list[str] = field(default_factory=list)
    budget: str | None = None
    timeline: str | None = None


@dataclass
class StructuredIntent:
    """النية المهيكلة"""

    request_id: str
    timestamp: datetime

    # الأمان
    security_scan: dict[str, Any] = field(default_factory=dict)

    # التصنيف
    intent_type: IntentType = IntentType.UNKNOWN
    confidence: float = 0.0

    # التوجيه
    routing_target: RoutingTarget = RoutingTarget.STRATEGIC
    routing_reason: str = ""

    # المتطلبات
    explicit_goals: list[str] = field(default_factory=list)
    implicit_requirements: ImplicitRequirements = field(default_factory=ImplicitRequirements)

    # القيود
    constraints: dict[str, Any] = field(default_factory=dict)

    # السياق
    context_snapshot: dict[str, Any] = field(default_factory=dict)

    # التوصيات
    recommended_critics: list[str] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)

    # المetadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "intent_type": self.intent_type.name,
            "confidence": self.confidence,
            "routing_target": self.routing_target.value,
            "explicit_goals": self.explicit_goals,
            "constraints": self.constraints,
            "recommended_critics": self.recommended_critics,
        }


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier:
    """مصنف النوايا"""

    # أنماط التصنيف
    INTENT_PATTERNS = {
        IntentType.CODE_GENERATION: [
            r"write\s+.{0,40}(function|class|module|script|code|program|algorithm|implementation)",
            r"create\s+.{0,40}(function|class|module|script|code|program|algorithm|implementation|server|app|api)",
            r"implement\s+.{0,40}(function|class|module|algorithm|search|sort|pattern|handler|endpoint)",
            r"implement\s+\w+",
            r"build\s+(a|an|the)\s+\w+",
            r"develop\s+\w+",
            r"(write|code|make)\s+(me\s+)?(a|an)\s+\w+",
            r"(binary|linear|merge|quick|bubble)\s*search|sort",
            r"اكتب\s+(كود|دالة|برنامج)",
            r"أنشئ\s+\w+",
        ],
        IntentType.CODE_REVIEW: [
            r"review\s+(this|the)\s+(code|implementation)",
            r"check\s+(this|the)\s+code",
            r"analyze\s+(this|the)\s+code",
            r"راجع\s+(الكود|البرنامج)",
        ],
        IntentType.DEBUGGING: [
            r"debug\s+(this|the)\s+\w+",
            r"fix\s+(this|the|a)\s+(error|bug|issue)",
            r"why\s+(is|does|doesn\'t)\s+\w+",
            r"what\'?s?\s+wrong\s+with",
            r"صلح\s+(الخطأ|المشكلة)",
            r"حل\s+(الخطأ|المشكلة)",
        ],
        IntentType.REFACTORING: [
            r"refactor\s+(this|the)\s+\w+",
            r"improve\s+(this|the)\s+(code|performance)",
            r"optimize\s+\w+",
            r"restructure\s+\w+",
            r"أعد\s+هيكلة",
            r"حسّن\s+(الكود|الأداء)",
        ],
        IntentType.DOCUMENTATION: [
            r"write\s+(documentation|docs|comments)",
            r"document\s+(this|the)\s+\w+",
            r"add\s+comments\s+to",
            r"generate\s+(docs|documentation)",
            r"اكتب\s+(توثيق|وثائق)",
        ],
        IntentType.TESTING: [
            r"write\s+(tests?|test\s+cases?)",
            r"create\s+(tests?|test\s+suite)",
            r"generate\s+tests?",
            r"اكتب\s+(اختبارات|اختبار)",
        ],
        IntentType.RESEARCH: [
            r"research\s+\w+",
            r"find\s+(information|out)\s+about",
            r"what\s+is\s+\w+",
            r"how\s+does\s+\w+\s+work",
            r"ابحث\s+عن",
            r"ما\s+هو",
        ],
        IntentType.PLANNING: [
            r"plan\s+(a|an|the)\s+\w+",
            r"design\s+(a|an|the)\s+(architecture|system)",
            r"create\s+(a|an)?\s*(architecture|design|plan)",
            r"خطط\s+لـ",
            r"صمم\s+(نظام|بنية)",
        ],
        IntentType.QUESTION: [
            r"^(what|how|why|when|where|who|which)",
            r"^(هل|كيف|لماذا|متى|أين|من)",
            r"\?$",
        ],
        IntentType.ANALYSIS: [
            r"analyze\s+\w+",
            r"examine\s+\w+",
            r"evaluate\s+\w+",
            r"assess\s+\w+",
            r"حلل\s+\w+",
        ],
    }

    # مؤشرات التعقيد
    COMPLEXITY_INDICATORS = {
        "high": [
            "architecture",
            "system",
            "microservices",
            "distributed",
            "scale",
            "بنية",
            "نظام موزع",
        ],
        "medium": ["module", "component", "service", "api", "وحدة", "مكون"],
        "low": ["function", "method", "variable", "helper", "دالة", "متغير"],
    }

    def __init__(self):
        self._logger = get_logger("gaap.layer0.classifier")

    def classify(self, text: str) -> tuple[IntentType, float]:
        """تصنيف النص"""
        text_lower = text.lower()
        scores: dict[IntentType, float] = {}

        for intent_type, patterns in self.INTENT_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    score += len(matches) * 1.0

            if score > 0:
                scores[intent_type] = score

        if not scores:
            return IntentType.UNKNOWN, 0.0

        # أفضل تطابق
        best_intent = max(scores.items(), key=lambda x: x[1])

        # حساب الثقة
        total_score = sum(scores.values())
        confidence = best_intent[1] / total_score if total_score > 0 else 0.0

        return best_intent[0], min(confidence, 1.0)

    def estimate_complexity(self, text: str) -> TaskComplexity:
        """تقدير التعقيد"""
        text_lower = text.lower()

        high_count = sum(1 for kw in self.COMPLEXITY_INDICATORS["high"] if kw in text_lower)
        medium_count = sum(1 for kw in self.COMPLEXITY_INDICATORS["medium"] if kw in text_lower)
        low_count = sum(1 for kw in self.COMPLEXITY_INDICATORS["low"] if kw in text_lower)

        # طول النص كعامل إضافي
        length_factor = len(text.split()) / 100

        if high_count > 0 or length_factor > 2:
            return TaskComplexity.ARCHITECTURAL
        elif medium_count > 1 or length_factor > 1:
            return TaskComplexity.COMPLEX
        elif medium_count > 0 or length_factor > 0.5:
            return TaskComplexity.MODERATE
        elif low_count > 0:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.TRIVIAL


# =============================================================================
# Request Parser
# =============================================================================


class RequestParser:
    """محلل الطلبات"""

    # أنماط استخراج المتطلبات الضمنية
    REQUIREMENT_PATTERNS = {
        "performance": [
            (r"(fast|quick|speedy|high\s+performance)", "high_throughput"),
            (r"(real-?time|instant|immediate)", "real_time"),
            (r"(slow|latency|response\s+time)", "latency_optimization"),
        ],
        "security": [
            (r"(secure|security|encrypted|auth)", "security_required"),
            (r"(gdpr|compliance|privacy)", "compliance_required"),
            (r"(pci|hipaa|sox)", "regulatory_compliance"),
        ],
        "scalability": [
            (r"(scalable|scale|distributed)", "horizontal_scaling"),
            (r"(million|billion|large\s+scale)", "high_scale"),
        ],
        "budget": [
            (r"(budget|cheap|cost-?effective|affordable)", "budget_conscious"),
            (r"(enterprise|production)", "production_grade"),
        ],
        "timeline": [
            (r"(\d+)\s*(days?|weeks?|months?)", "timeline_constraint"),
            (r"(asap|urgent|quickly|soon)", "urgent"),
        ],
    }

    def __init__(self):
        self._logger = get_logger("gaap.layer0.parser")

    def parse(self, text: str) -> tuple[list[str], ImplicitRequirements, dict[str, Any]]:
        """تحليل الطلب"""
        # استخراج الأهداف الصريحة
        goals = self._extract_goals(text)

        # استخراج المتطلبات الضمنية
        implicit = self._extract_implicit_requirements(text)

        # استخراج القيود
        constraints = self._extract_constraints(text)

        return goals, implicit, constraints

    def _extract_goals(self, text: str) -> list[str]:
        """استخراج الأهداف"""
        goals = []

        # أنماط الأهداف
        goal_patterns = [
            r"(?:build|create|develop|implement|write)\s+(?:a|an|the)?\s*(.+?)(?:\s+that|\s+with|\s+for|\s*$)",
            r"(?:need|want|require)\s+(?:to\s+)?(.+?)(?:\s+that|\s+with|\s+for|\s*$)",
        ]

        for pattern in goal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            goals.extend(matches)

        # تنظيف
        goals = [g.strip() for g in goals if len(g.strip()) > 5]

        return goals[:5]  # حد أقصى 5 أهداف

    def _extract_implicit_requirements(self, text: str) -> ImplicitRequirements:
        """استخراج المتطلبات الضمنية"""
        text_lower = text.lower()

        implicit = ImplicitRequirements()

        # الأداء
        for pattern, value in self.REQUIREMENT_PATTERNS["performance"]:
            if re.search(pattern, text_lower):
                implicit.performance = value
                break

        # الأمان
        for pattern, value in self.REQUIREMENT_PATTERNS["security"]:
            if re.search(pattern, text_lower):
                implicit.security = value
                break

        # القابلية للتوسع
        for pattern, value in self.REQUIREMENT_PATTERNS["scalability"]:
            if re.search(pattern, text_lower):
                implicit.scalability = value
                break

        # الامتثال
        compliance_keywords = ["gdpr", "hipaa", "pci", "sox", "iso"]
        for kw in compliance_keywords:
            if kw in text_lower:
                implicit.compliance.append(kw.upper())

        # الميزانية
        for pattern, value in self.REQUIREMENT_PATTERNS["budget"]:
            if re.search(pattern, text_lower):
                implicit.budget = value
                break

        # الجدول الزمني
        for pattern, value in self.REQUIREMENT_PATTERNS["timeline"]:
            if re.search(pattern, text_lower):
                implicit.timeline = value
                break

        return implicit

    def _extract_constraints(self, text: str) -> dict[str, Any]:
        """استخراج القيود"""
        constraints = {}

        # قيود اللغة
        lang_pattern = r"(?:using|in|with)\s+(python|javascript|typescript|java|go|rust)"
        lang_match = re.search(lang_pattern, text, re.IGNORECASE)
        if lang_match:
            constraints["language"] = lang_match.group(1).lower()

        # قيود الإطار
        framework_pattern = r"(?:using|with)\s+(react|vue|angular|django|flask|fastapi|express)"
        framework_match = re.search(framework_pattern, text, re.IGNORECASE)
        if framework_match:
            constraints["framework"] = framework_match.group(1).lower()

        # قيود النظام الأساسي
        platform_pattern = r"(?:on|for)\s+(aws|azure|gcp|docker|kubernetes)"
        platform_match = re.search(platform_pattern, text, re.IGNORECASE)
        if platform_match:
            constraints["platform"] = platform_match.group(1).lower()

        return constraints


# =============================================================================
# Layer 0 Interface
# =============================================================================


class Layer0Interface(BaseLayer):
    """
    طبقة الواجهة - المدخل الرئيسي لنظام GAAP

    المسؤوليات:
    - فحص أمني للمدخلات (Prompt Firewall)
    - تصنيف نية المستخدم
    - استخراج المتطلبات الضمنية
    - توجيه الطلب للطبقة المناسبة
    - تهيئة السياق الأولي
    """

    def __init__(self, firewall_strictness: str = "high", enable_behavioral_analysis: bool = True):
        super().__init__(LayerType.INTERFACE)

        # المكونات
        self.firewall = PromptFirewall(strictness=firewall_strictness)
        self.classifier = IntentClassifier()
        self.parser = RequestParser()

        self._enable_behavioral = enable_behavioral_analysis
        self._logger = get_logger("gaap.layer0")

        # الإحصائيات
        self._requests_processed = 0
        self._requests_blocked = 0
        self._intent_distribution: dict[str, int] = {}

    async def process(self, input_data: Any) -> StructuredIntent:
        """معالجة المدخل"""
        start_time = time.time()

        # استخراج النص
        if isinstance(input_data, str):
            text = input_data
            context = {}
        elif isinstance(input_data, dict):
            text = input_data.get("text", "")
            context = input_data.get("context", {})
        else:
            raise ValueError("Invalid input format")

        # إنشاء معرف الطلب
        request_id = self._generate_request_id()

        self._logger.info(f"Processing request {request_id}")

        # 1. الفحص الأمني
        security_result = self.firewall.scan(text, context)

        structured = StructuredIntent(
            request_id=request_id,
            timestamp=datetime.now(),
            security_scan={
                "is_safe": security_result.is_safe,
                "risk_level": security_result.risk_level.name,
                "detected_patterns": security_result.detected_patterns[:5],
                "scan_time_ms": security_result.scan_time_ms,
            },
        )

        # إذا لم يكن آمناً
        if not security_result.is_safe:
            self._requests_blocked += 1
            structured.routing_target = RoutingTarget.DIRECT
            structured.routing_reason = f"Security risk: {security_result.risk_level.name}"
            structured.metadata["blocked"] = True
            return structured

        # 1.5. حفظ النص الأصلي للاستخدام في الطبقات اللاحقة
        structured.metadata["original_text"] = text

        # 2. تصنيف النية
        intent_type, confidence = self.classifier.classify(text)
        structured.intent_type = intent_type
        structured.confidence = confidence

        # تحديث توزيع النوايا
        intent_name = intent_type.name
        self._intent_distribution[intent_name] = self._intent_distribution.get(intent_name, 0) + 1

        # 3. تحليل الطلب
        goals, implicit, constraints = self.parser.parse(text)
        structured.explicit_goals = goals
        structured.implicit_requirements = implicit
        structured.constraints = constraints

        # 4. تقدير التعقيد
        complexity = self.classifier.estimate_complexity(text)
        structured.metadata["complexity"] = complexity.name

        # 5. تحديد التوجيه
        routing_target, routing_reason = self._determine_routing(
            intent_type, complexity, confidence, text
        )
        structured.routing_target = routing_target
        structured.routing_reason = routing_reason

        # 6. تحديد النقاد الموصى بهم
        structured.recommended_critics = self._recommend_critics(intent_type, implicit, constraints)

        # 7. تحديد الأدوات الموصى بها
        structured.recommended_tools = self._recommend_tools(intent_type, complexity)

        # 8. حفظ السياق
        structured.context_snapshot = {
            "text_length": len(text),
            "word_count": len(text.split()),
            "has_code_blocks": "```" in text,
            "has_questions": "?" in text,
        }

        self._requests_processed += 1

        elapsed = (time.time() - start_time) * 1000
        self._logger.info(
            f"Request {request_id} processed in {elapsed:.1f}ms: "
            f"intent={intent_type.name}, routing={routing_target.value}"
        )

        return structured

    def _determine_routing(
        self, intent_type: IntentType, complexity: TaskComplexity, confidence: float, text: str
    ) -> tuple[RoutingTarget, str]:
        """تحديد التوجيه"""

        # الأنواع التي تحتاج تخطيط استراتيجي
        strategic_intents = {
            IntentType.PLANNING,
            IntentType.ANALYSIS,
        }

        # الأنواع التي تحتاج تفصيل تكتيكي
        tactical_intents = {
            IntentType.CODE_GENERATION,
            IntentType.REFACTORING,
            IntentType.TESTING,
        }

        # الأنواع التي يمكن تنفيذها مباشرة
        direct_intents = {
            IntentType.QUESTION,
            IntentType.CONVERSATION,
            IntentType.DOCUMENTATION,
        }

        # قرار التوجيه
        if intent_type in strategic_intents or complexity == TaskComplexity.ARCHITECTURAL:
            return RoutingTarget.STRATEGIC, "Complex task requiring strategic planning"

        if intent_type in tactical_intents or complexity in (
            TaskComplexity.COMPLEX,
            TaskComplexity.MODERATE,
        ):
            return RoutingTarget.TACTICAL, "Task requiring tactical decomposition"

        if intent_type in direct_intents or complexity in (
            TaskComplexity.SIMPLE,
            TaskComplexity.TRIVIAL,
        ):
            return RoutingTarget.DIRECT, "Simple task for direct execution"

        # إذا كانت الثقة منخفضة، نذهب للاستراتيجي
        if confidence < 0.5:
            return RoutingTarget.STRATEGIC, "Low confidence, requires analysis"

        # الافتراضي
        return RoutingTarget.TACTICAL, "Default tactical routing"

    def _recommend_critics(
        self, intent_type: IntentType, implicit: ImplicitRequirements, constraints: dict[str, Any]
    ) -> list[str]:
        """توصية بالنقاد"""
        critics = ["logic"]  # دائماً ناقد المنطق

        # بناءً على النية
        intent_critics = {
            IntentType.CODE_GENERATION: ["performance", "style"],
            IntentType.CODE_REVIEW: ["security", "performance"],
            IntentType.DEBUGGING: ["logic", "security"],
            IntentType.REFACTORING: ["performance", "style"],
            IntentType.TESTING: ["logic"],
            IntentType.PLANNING: ["scalability"],
        }

        critics.extend(intent_critics.get(intent_type, []))

        # بناءً على المتطلبات الضمنية
        if implicit.security:
            critics.append("security")
        if implicit.performance:
            critics.append("performance")
        if implicit.compliance:
            critics.append("compliance")

        return list(set(critics))

    def _recommend_tools(self, intent_type: IntentType, complexity: TaskComplexity) -> list[str]:
        """توصية بالأدوات"""
        tools = []

        if intent_type == IntentType.RESEARCH:
            tools.extend(["perplexity", "web_search"])

        if complexity in (TaskComplexity.COMPLEX, TaskComplexity.ARCHITECTURAL):
            tools.append("tot_strategic")

        if intent_type == IntentType.DEBUGGING:
            tools.append("self_healing")

        return tools

    def _generate_request_id(self) -> str:
        """توليد معرف طلب"""
        import uuid

        return f"req_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الطبقة"""
        return {
            "layer": "L0_Interface",
            "requests_processed": self._requests_processed,
            "requests_blocked": self._requests_blocked,
            "block_rate": self._requests_blocked / max(self._requests_processed, 1),
            "intent_distribution": self._intent_distribution,
            "firewall_stats": self.firewall.get_stats(),
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_interface(
    firewall_strictness: str = "high", enable_behavioral: bool = True
) -> Layer0Interface:
    """إنشاء طبقة الواجهة"""
    return Layer0Interface(
        firewall_strictness=firewall_strictness, enable_behavioral_analysis=enable_behavioral
    )
