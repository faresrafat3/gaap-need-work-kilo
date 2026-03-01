"""
Layer 0: Interface Layer

The entry point for all GAAP requests. Provides:
- Security scanning (7-layer firewall)
- Intent classification (11 types)
- Complexity estimation
- Smart routing decisions

Classes:
    - IntentType: Intent classification types
    - RoutingTarget: Routing decision targets
    - ImplicitRequirements: Extracted implicit requirements
    - StructuredIntent: Fully classified intent
    - IntentClassifier: Classifies user intents
    - RequestParser: Parses and validates requests
    - Layer0Interface: Main Layer 0 interface

Usage:
    from gaap.layers import Layer0Interface

    layer0 = Layer0Interface()
    intent = await layer0.process("Write a binary search function")
    print(f"Intent: {intent.intent_type}")
    print(f"Route to: {intent.routing_target}")
"""

import json

# Layer 0: Interface Layer
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from gaap.core.base import BaseLayer, BaseProvider
from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import LayerType, TaskComplexity
from gaap.security.firewall import PromptFirewall

# =============================================================================
# Logger Setup
# =============================================================================



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

    # NEW: Metacognition fields
    confidence_assessment: dict[str, Any] | None = None
    knowledge_gaps: list[str] = field(default_factory=list)
    research_required: bool = False
    research_topics: list[str] = field(default_factory=list)
    epistemic_humility_score: float = 0.0
    novelty_score: float = 0.0
    caution_mode: bool = False

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
            "research_required": self.research_required,
            "knowledge_gaps": self.knowledge_gaps,
            "caution_mode": self.caution_mode,
        }


# =============================================================================
# Layer 0 Interface
# =============================================================================


class Layer0Interface(BaseLayer):
    """
    Layer 0: Interface Layer - Main entry point for GAAP system.

    Responsibilities:
    - Security scanning (7-layer firewall)
    - Intent classification (11 types)
    - Implicit requirement extraction
    - Routing decisions (Strategic/Tactical/Direct)
    - Context initialization

    Attributes:
        firewall: Prompt firewall for security scanning
        classifier: Intent classifier
        parser: Request parser
        _enable_behavioral: Enable behavioral analysis
        _logger: Logger instance
        _requests_processed: Request counter
        _requests_blocked: Blocked request counter
        _intent_distribution: Intent type distribution

    Usage:
        >>> layer0 = Layer0Interface(firewall_strictness="high")
        >>> intent = await layer0.process("Write a function")
        >>> print(f"Intent: {intent.intent_type.name}")
        >>> print(f"Route: {intent.routing_target.value}")
    """

    def __init__(
        self,
        firewall_strictness: str = "high",
        enable_behavioral_analysis: bool = True,
        provider: BaseProvider | None = None,
        episodic_store: Any | None = None,
        enable_metacognition: bool = True,
    ) -> None:
        """
        Initialize Layer 0 interface.

        Args:
            firewall_strictness: Security strictness level (low/medium/high)
            enable_behavioral_analysis: Enable behavioral analysis flag
            provider: LLM Provider for intent analysis
            episodic_store: Episodic memory store for confidence assessment
            enable_metacognition: Enable metacognition and confidence assessment
        """
        super().__init__(LayerType.INTERFACE)

        # Components
        self.firewall = PromptFirewall(strictness=firewall_strictness)
        self.provider = provider

        self._enable_behavioral = enable_behavioral_analysis
        self._enable_metacognition = enable_metacognition
        self._logger = get_logger("gaap.layer0")

        # Statistics
        self._requests_processed = 0
        self._requests_blocked = 0
        self._intent_distribution: dict[str, int] = {}

        # NEW: Metacognition components
        self._knowledge_map: "KnowledgeMap | None" = None
        self._confidence_scorer: "ConfidenceScorer | None" = None
        if enable_metacognition:
            from gaap.core.confidence_scorer import ConfidenceScorer
            from gaap.core.knowledge_map import KnowledgeMap

            self._knowledge_map = KnowledgeMap()
            self._confidence_scorer = ConfidenceScorer(
                episodic_store=episodic_store,
                knowledge_map=self._knowledge_map,
            )

    async def _analyze_intent_llm(self, text: str) -> dict[str, Any]:
        """Analyze intent using LLM provider."""
        default_res = {
            "intent_type": "UNKNOWN",
            "confidence": 0.5,
            "complexity": "MODERATE",
            "explicit_goals": [],
            "implicit_requirements": {},
            "constraints": {},
        }
        if not self.provider:
            self._logger.warning("No LLM provider tied to Layer0. Falling back to default intent.")
            return default_res

        system_prompt = """
You are the Layer 0 Intent Analyzer for the GAAP General-purpose AI Architecture Platform.
Analyze the user's input and return a pure JSON object mapping strictly to this schema:
{
  "intent_type": "CODE_GENERATION" | "CODE_REVIEW" | "DEBUGGING" | "REFACTORING" | "DOCUMENTATION" | "TESTING" | "RESEARCH" | "ANALYSIS" | "PLANNING" | "QUESTION" | "CONVERSATION" | "UNKNOWN",
  "confidence": float (0.0 to 1.0),
  "complexity": "TRIVIAL" | "SIMPLE" | "MODERATE" | "COMPLEX" | "ARCHITECTURAL",
  "explicit_goals": ["goal 1", "goal 2"],
  "implicit_requirements": {
    "performance": "real_time" | "high_throughput" | null,
    "security": "security_required" | "compliance_required" | null,
    "scalability": "horizontal_scaling" | "high_scale" | null,
    "compliance": ["GDPR", "HIPAA", ...],
    "budget": "budget_conscious" | "production_grade" | null,
    "timeline": "urgent" | "timeline_constraint" | null
  },
  "constraints": {
    "language": "python" | "javascript" | ...,
    "framework": "react" | "django" | ...,
    "platform": "aws" | "docker" | ...
  }
}
Return ONLY valid JSON.
"""
        try:
            from gaap.core.types import Message, MessageRole

            messages = [
                Message(role=MessageRole.SYSTEM, content=system_prompt.strip()),
                Message(role=MessageRole.USER, content=text),
            ]
            model = getattr(self.provider, "default_model", None) or "llama-3.3-70b-versatile"
            response = await self.provider.chat_completion(messages=messages, model=model)
            raw = response.choices[0].message.content if response.choices else ""
            # extract json block
            matches = re.findall(r"```(?:json)?\s*({.*})\s*```", raw, re.DOTALL)
            if matches:
                return json.loads(matches[0])  # type: ignore[no-any-return]
            else:
                return json.loads(raw)  # type: ignore[no-any-return]
        except Exception as e:
            self._logger.error(f"Failed to analyze intent via LLM: {e}")
            return default_res

    async def process(self, input_data: Any) -> StructuredIntent:
        """
        Process user input and return structured intent.

        Args:
            input_data: User input (string or dict with text/context)

        Returns:
            StructuredIntent with classification and routing

        Raises:
            ValueError: If input format is invalid

        Process Flow:
            1. Extract text from input
            2. Generate request ID
            3. Security scan (firewall)
            4. Intent classification
            5. Parse requirements
            6. Estimate complexity
            7. Determine routing

        Example:
            >>> layer0 = Layer0Interface()
            >>> intent = await layer0.process("Write a binary search")
            >>> print(f"Intent: {intent.intent_type.name}")
            'CODE_GENERATION'
        """
        start_time = time.time()

        # Extract text
        if isinstance(input_data, str):
            text = input_data
            context = {}
        elif isinstance(input_data, dict):
            text = input_data.get("text", "")
            context = input_data.get("context", {})
        else:
            raise ValueError("Invalid input format")

        # Generate request ID
        request_id = self._generate_request_id()

        self._logger.info(f"Processing request {request_id}")

        # 1. Security scan
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

        # If not safe
        if not security_result.is_safe:
            self._requests_blocked += 1
            structured.routing_target = RoutingTarget.DIRECT
            structured.routing_reason = f"Security risk: {security_result.risk_level.name}"
            structured.metadata["blocked"] = True
            return structured

        # Save original text
        structured.metadata["original_text"] = text

        # 2-4. Intent classification, parsing, and complexity estimation via LLM
        analysis = await self._analyze_intent_llm(text)

        try:
            structured.intent_type = IntentType[analysis.get("intent_type", "UNKNOWN")]
        except KeyError:
            structured.intent_type = IntentType.UNKNOWN

        structured.confidence = float(analysis.get("confidence", 0.5))

        # Update distribution
        intent_name = structured.intent_type.name
        self._intent_distribution[intent_name] = self._intent_distribution.get(intent_name, 0) + 1

        structured.explicit_goals = analysis.get("explicit_goals", [])

        imp_reqs = analysis.get("implicit_requirements", {})
        structured.implicit_requirements = ImplicitRequirements(
            performance=imp_reqs.get("performance"),
            security=imp_reqs.get("security"),
            scalability=imp_reqs.get("scalability"),
            compliance=imp_reqs.get("compliance", []),
            budget=imp_reqs.get("budget"),
            timeline=imp_reqs.get("timeline"),
        )
        structured.constraints = analysis.get("constraints", {})

        try:
            complexity = TaskComplexity[analysis.get("complexity", "MODERATE")]
        except KeyError:
            complexity = TaskComplexity.MODERATE

        structured.metadata["complexity"] = complexity.name

        # 5. NEW: Metacognition - Confidence Assessment
        if self._enable_metacognition and self._confidence_scorer:
            assessment = await self._confidence_scorer.assess(text, structured.intent_type.name)

            structured.confidence_assessment = assessment.to_dict()
            structured.knowledge_gaps = assessment.knowledge_gaps
            structured.novelty_score = assessment.novelty_score
            structured.epistemic_humility_score = assessment.epistemic_humility

            if assessment.needs_research():
                structured.research_required = True
                structured.research_topics = assessment.research_topics
                self._logger.info(
                    f"Research required for task: {assessment.confidence.score:.0%} confidence, "
                    f"gaps: {assessment.knowledge_gaps[:3]}"
                )
            elif assessment.needs_caution():
                structured.caution_mode = True
                structured.metadata["twin_verification"] = "increased"
                self._logger.info(
                    f"Caution mode enabled: {assessment.confidence.score:.0%} confidence"
                )

        # 6. Determine routing (enhanced with confidence)
        if structured.research_required:
            routing_target = RoutingTarget.TACTICAL
            routing_reason = (
                f"Low confidence ({structured.confidence_assessment['confidence']['score']:.0%}) - research required"
                if structured.confidence_assessment
                else "Research required"
            )
        else:
            routing_target, routing_reason = self._determine_routing(
                structured.intent_type, complexity, structured.confidence, text
            )

        if structured.caution_mode:
            routing_reason += " (caution mode)"

        structured.routing_target = routing_target
        structured.routing_reason = routing_reason

        # 7. Recommend critics
        structured.recommended_critics = self._recommend_critics(
            structured.intent_type, structured.implicit_requirements, structured.constraints
        )

        # 7. Recommend tools
        structured.recommended_tools = self._recommend_tools(structured.intent_type, complexity)

        # 8. Save context snapshot
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
            f"intent={structured.intent_type.name}, routing={routing_target.value}"
        )

        return structured

    def _determine_routing(
        self,
        intent_type: IntentType,
        complexity: TaskComplexity,
        confidence: float,
        text: str,
    ) -> tuple[RoutingTarget, str]:
        """
        Determine routing target based on intent and complexity.

        Args:
            intent_type: Classified intent type
            complexity: Estimated task complexity
            confidence: Classification confidence (0.0-1.0)
            text: Original input text

        Returns:
            Tuple of (routing_target, routing_reason)

        Routing Logic:
            - STRATEGIC: Planning, Analysis, or ARCHITECTURAL complexity
            - TACTICAL: Code generation, Refactoring, Testing, or COMPLEX
            - DIRECT: Questions, Conversation, Documentation, or SIMPLE

        Example:
            >>> layer0 = Layer0Interface()
            >>> target, reason = layer0._determine_routing(
            ...     IntentType.CODE_GENERATION, TaskComplexity.MODERATE, 0.8, "text"
            ... )
            >>> print(target.value)
            'layer2_tactical'
        """
        # Types requiring strategic planning
        strategic_intents = {
            IntentType.PLANNING,
            IntentType.ANALYSIS,
        }

        # Types requiring tactical decomposition
        tactical_intents = {
            IntentType.CODE_GENERATION,
            IntentType.REFACTORING,
            IntentType.TESTING,
        }

        # Types for direct execution
        direct_intents = {
            IntentType.QUESTION,
            IntentType.CONVERSATION,
            IntentType.DOCUMENTATION,
        }

        # Routing decision
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

        # Low confidence → strategic
        if confidence < 0.5:
            return RoutingTarget.STRATEGIC, "Low confidence, requires analysis"

        # Default
        return RoutingTarget.TACTICAL, "Default tactical routing"

    def _recommend_critics(
        self,
        intent_type: IntentType,
        implicit: ImplicitRequirements,
        constraints: dict[str, Any],
    ) -> list[str]:
        """
        Recommend MAD critics based on intent and requirements.

        Args:
            intent_type: Classified intent type
            implicit: Implicit requirements
            constraints: Task constraints

        Returns:
            List of critic names (logic, security, performance, etc.)

        Critic Selection Logic:
            - Always include 'logic' critic
            - Add intent-specific critics (performance, style, etc.)
            - Add requirement-based critics (security, compliance)

        Example:
            >>> layer0 = Layer0Interface()
            >>> critics = layer0._recommend_critics(IntentType.CODE_GENERATION, ...)
            >>> print(critics)
            ['logic', 'performance', 'style']
        """
        critics = ["logic"]  # Always include logic critic

        # Intent-based critics
        intent_critics = {
            IntentType.CODE_GENERATION: ["performance", "style"],
            IntentType.CODE_REVIEW: ["security", "performance"],
            IntentType.DEBUGGING: ["logic", "security"],
            IntentType.REFACTORING: ["performance", "style"],
            IntentType.TESTING: ["logic"],
            IntentType.PLANNING: ["scalability"],
        }

        critics.extend(intent_critics.get(intent_type, []))

        # Requirement-based critics
        if implicit.security:
            critics.append("security")
        if implicit.performance:
            critics.append("performance")
        if implicit.compliance:
            critics.append("compliance")

        return list(set(critics))

    def _recommend_tools(self, intent_type: IntentType, complexity: TaskComplexity) -> list[str]:
        """
        Recommend tools based on intent and complexity.

        Args:
            intent_type: Classified intent type
            complexity: Estimated task complexity

        Returns:
            List of recommended tool names

        Tool Selection Logic:
            - Research → perplexity, web_search
            - Complex/Architectural → tot_strategic
            - Debugging → self_healing

        Example:
            >>> layer0 = Layer0Interface()
            >>> tools = layer0._recommend_tools(IntentType.RESEARCH, TaskComplexity.MODERATE)
            >>> print(tools)
            ['perplexity', 'web_search']
        """
        tools = []

        if intent_type == IntentType.RESEARCH:
            tools.extend(["perplexity", "web_search"])

        if complexity in (TaskComplexity.COMPLEX, TaskComplexity.ARCHITECTURAL):
            tools.append("tot_strategic")

        if intent_type == IntentType.DEBUGGING:
            tools.append("self_healing")

        return tools

    def _generate_request_id(self) -> str:
        """
        Generate unique request ID.

        Returns:
            Unique request ID string (format: req_{timestamp}_{uuid})

        Example:
            >>> layer0._generate_request_id()
            'req_1708234567890_a1b2c3d4'
        """
        import uuid

        return f"req_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"

    def get_stats(self) -> dict[str, Any]:
        """
        Get layer statistics.

        Returns:
            Dictionary with layer statistics including:
            - layer: Layer name
            - requests_processed: Total requests processed
            - requests_blocked: Requests blocked by firewall
            - block_rate: Ratio of blocked requests
            - intent_distribution: Distribution of intent types
            - firewall_stats: Firewall statistics

        Example:
            >>> layer0 = Layer0Interface()
            >>> stats = layer0.get_stats()
            >>> print(f"Processed: {stats['requests_processed']}")
        """
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
    firewall_strictness: str = "high",
    enable_behavioral: bool = True,
) -> Layer0Interface:
    """
    Create Layer0Interface instance.

    Factory function for creating Layer 0 interface.

    Args:
        firewall_strictness: Security strictness (low/medium/high)
        enable_behavioral: Enable behavioral analysis

    Returns:
        Configured Layer0Interface instance

    Example:
        >>> layer0 = create_interface(firewall_strictness="high")
        >>> intent = await layer0.process("Write a function")
    """
    return Layer0Interface(
        firewall_strictness=firewall_strictness,
        enable_behavioral_analysis=enable_behavioral,
    )
