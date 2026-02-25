"""
Semantic Distiller for GAAP System

Provides context compression and conversation state management.

Classes:
    - SemanticMatrix: Compressed conversation state
    - SemanticDistiller: Context compression engine

Usage:
    from gaap.core.semantic_distiller import SemanticDistiller, SemanticMatrix

    distiller = SemanticDistiller()
    if distiller.should_distill(message_count):
        matrix = distiller.distill(messages)
        context = distiller.get_active_context()
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from gaap.core.logging import get_standard_logger as get_logger
from gaap.core.types import Message


@dataclass
class SemanticMatrix:
    """
    Compressed conversation state for efficient context management.

    Attributes:
        facts: Extracted factual statements
        decisions: Key decisions made during conversation
        pending_risks: Identified risks requiring attention
        action_items: Pending action items
        entities: Named entities mentioned
        key_terms: Important technical terms
        summary: Brief conversation summary
        token_count: Approximate token count of compressed state
        created_at: Timestamp of creation
    """

    facts: list[str] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    pending_risks: list[str] = field(default_factory=list)
    action_items: list[str] = field(default_factory=list)
    entities: dict[str, str] = field(default_factory=dict)
    key_terms: list[str] = field(default_factory=list)
    summary: str = ""
    token_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "facts": self.facts,
            "decisions": self.decisions,
            "pending_risks": self.pending_risks,
            "action_items": self.action_items,
            "entities": self.entities,
            "key_terms": self.key_terms,
            "summary": self.summary,
            "token_count": self.token_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticMatrix":
        return cls(
            facts=data.get("facts", []),
            decisions=data.get("decisions", []),
            pending_risks=data.get("pending_risks", []),
            action_items=data.get("action_items", []),
            entities=data.get("entities", {}),
            key_terms=data.get("key_terms", []),
            summary=data.get("summary", ""),
            token_count=data.get("token_count", 0),
            created_at=data.get("created_at", time.time()),
        )

    def merge(self, other: "SemanticMatrix") -> "SemanticMatrix":
        """Merge another matrix into this one, deduplicating entries."""
        merged = SemanticMatrix(
            facts=list(set(self.facts + other.facts)),
            decisions=self.decisions + other.decisions,
            pending_risks=list(set(self.pending_risks + other.pending_risks)),
            action_items=list(set(self.action_items + other.action_items)),
            entities={**self.entities, **other.entities},
            key_terms=list(set(self.key_terms + other.key_terms)),
            summary=other.summary if other.summary else self.summary,
            token_count=self.token_count + other.token_count,
        )
        return merged


@dataclass
class DistillationResult:
    """Result of a distillation operation."""

    matrix: SemanticMatrix
    messages_processed: int
    compression_ratio: float
    quality_score: float


class SemanticDistiller:
    """
    Context compression engine for managing conversation state.

    Compresses conversation history into semantic matrices,
    enabling efficient context management for long conversations.

    Attributes:
        distill_interval: Number of messages between distillations
        max_facts: Maximum facts to retain
        max_decisions: Maximum decisions to retain
        max_risks: Maximum risks to retain
    """

    DECISION_PATTERNS = [
        r"(?:decided|chose|selected|picked|went with)\s+(.+?)(?:\.|$)",
        r"(?:we will|I'll|let's)\s+(?:use|implement|go with)\s+(.+?)(?:\.|$)",
        r"(?:the (?:best|right|correct) (?:approach|solution|way) is)\s+(.+?)(?:\.|$)",
    ]

    FACT_PATTERNS = [
        r"(?:the|a)\s+(\w+)\s+(?:is|are|has|have|contains?|provides?)\s+(.+?)(?:\.|$)",
        r"(?:note that|important:|fyi:?)\s*(.+?)(?:\.|$)",
        r"(?:confirmed|verified|checked)\s*(.+?)(?:\.|$)",
    ]

    RISK_PATTERNS = [
        r"(?:risk|warning|caution|danger|be careful)\s*:\s*(.+?)(?:\.|$)",
        r"(?:might|could|may)\s+(?:cause|lead to|result in)\s+(.+?)(?:\.|$)",
        r"(?:potential (?:issue|problem|concern))\s*:\s*(.+?)(?:\.|$)",
    ]

    ACTION_PATTERNS = [
        r"(?:todo|task|action)\s*:\s*(.+?)(?:\.|$)",
        r"(?:need to|should|must|have to)\s+(.+?)(?:\.|$)",
        r"(?:next step|follow.?up)\s*:\s*(.+?)(?:\.|$)",
    ]

    def __init__(
        self,
        distill_interval: int = 5,
        max_facts: int = 20,
        max_decisions: int = 10,
        max_risks: int = 10,
        max_action_items: int = 15,
        provider: Any = None,
    ) -> None:
        self.distill_interval = distill_interval
        self.max_facts = max_facts
        self.max_decisions = max_decisions
        self.max_risks = max_risks
        self.max_action_items = max_action_items
        self._provider = provider
        self._logger = get_logger("gaap.core.semantic_distiller")
        self._current_matrix: Optional[SemanticMatrix] = None
        self._archived_matrices: list[SemanticMatrix] = []
        self._message_count = 0
        self._distillation_count = 0

    def should_distill(self, message_count: int) -> bool:
        """
        Check if distillation should occur based on message count.

        Args:
            message_count: Current message count

        Returns:
            True if distillation should occur
        """
        return message_count > 0 and message_count % self.distill_interval == 0

    def distill(
        self,
        messages: list[Message],
        use_llm: bool = False,
    ) -> SemanticMatrix:
        """
        Distill messages into a semantic matrix.

        Args:
            messages: List of messages to distill
            use_llm: Whether to use LLM for enhanced extraction

        Returns:
            SemanticMatrix containing compressed state
        """
        self._logger.debug(f"Distilling {len(messages)} messages")

        matrix = SemanticMatrix()
        text_content = self._extract_text(messages)

        matrix.facts = self._extract_facts(text_content)[: self.max_facts]
        matrix.decisions = self._extract_decisions(text_content)[: self.max_decisions]
        matrix.pending_risks = self._extract_risks(text_content)[: self.max_risks]
        matrix.action_items = self._extract_actions(text_content)[: self.max_action_items]
        matrix.entities = self._extract_entities(text_content)
        matrix.key_terms = self._extract_key_terms(text_content)
        matrix.summary = self._generate_summary(text_content, matrix)
        matrix.token_count = self._estimate_tokens(matrix)

        if use_llm and self._provider:
            matrix = self._enhance_with_llm(matrix, text_content)

        self._current_matrix = matrix
        self._message_count += len(messages)
        self._distillation_count += 1

        self._logger.info(
            f"Distilled to {matrix.token_count} tokens "
            f"(compression: {self._calculate_compression(messages, matrix):.1%})"
        )

        return matrix

    def get_active_context(self) -> list[str]:
        """
        Get active context as list of strings.

        Returns:
            List of context strings for LLM consumption
        """
        if self._current_matrix is None:
            return []

        context = []
        matrix = self._current_matrix

        if matrix.summary:
            context.append(f"Summary: {matrix.summary}")

        if matrix.facts:
            context.append("Key Facts:")
            context.extend(f"  - {fact}" for fact in matrix.facts[:5])

        if matrix.decisions:
            context.append("Decisions Made:")
            for decision in matrix.decisions[:5]:
                context.append(f"  - {decision.get('what', 'Unknown')}: {decision.get('why', '')}")

        if matrix.pending_risks:
            context.append("Pending Risks:")
            context.extend(f"  - {risk}" for risk in matrix.pending_risks[:3])

        if matrix.action_items:
            context.append("Action Items:")
            context.extend(f"  - {item}" for item in matrix.action_items[:5])

        return context

    def archive_to_episodic(self, messages: list[Message]) -> Optional[SemanticMatrix]:
        """
        Archive current matrix to episodic memory.

        Args:
            messages: Messages to include in archival

        Returns:
            Archived matrix if successful
        """
        if self._current_matrix is None:
            return None

        archive = SemanticMatrix(
            facts=self._current_matrix.facts.copy(),
            decisions=self._current_matrix.decisions.copy(),
            pending_risks=self._current_matrix.pending_risks.copy(),
            action_items=self._current_matrix.action_items.copy(),
            entities=self._current_matrix.entities.copy(),
            key_terms=self._current_matrix.key_terms.copy(),
            summary=self._current_matrix.summary,
            token_count=self._current_matrix.token_count,
        )

        self._archived_matrices.append(archive)
        self._current_matrix = None

        self._logger.info(f"Archived matrix (total archives: {len(self._archived_matrices)})")

        return archive

    def get_matrix(self) -> Optional[SemanticMatrix]:
        """Get current semantic matrix."""
        return self._current_matrix

    def set_matrix(self, matrix: SemanticMatrix) -> None:
        """Set the current semantic matrix."""
        self._current_matrix = matrix

    def _extract_text(self, messages: list[Message]) -> str:
        """Extract all text content from messages."""
        parts = []
        for msg in messages:
            if msg.content:
                parts.append(msg.content)
        return "\n".join(parts)

    def _extract_facts(self, text: str) -> list[str]:
        """Extract factual statements from text."""
        facts = []
        text_lower = text.lower()

        for pattern in self.FACT_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    fact = " ".join(match)
                else:
                    fact = match
                if fact and len(fact) > 10:
                    facts.append(fact.strip())

        code_facts = self._extract_code_context(text)
        facts.extend(code_facts)

        return list(dict.fromkeys(facts))

    def _extract_code_context(self, text: str) -> list[str]:
        """Extract context about code from text."""
        facts = []

        lang_pattern = r"(?:using|in|with)\s+(python|javascript|typescript|rust|go|java|c\+\+)"
        for match in re.findall(lang_pattern, text, re.IGNORECASE):
            facts.append(f"Language: {match}")

        framework_pattern = r"(?:using|with)\s+(react|vue|django|flask|fastapi|express|next\.?js)"
        for match in re.findall(framework_pattern, text, re.IGNORECASE):
            facts.append(f"Framework: {match}")

        return facts

    def _extract_decisions(self, text: str) -> list[dict[str, Any]]:
        """Extract decisions from text."""
        decisions = []

        for pattern in self.DECISION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match and len(match) > 5:
                    decisions.append(
                        {
                            "what": match.strip(),
                            "why": "Extracted from conversation",
                            "when": time.time(),
                        }
                    )

        return decisions

    def _extract_risks(self, text: str) -> list[str]:
        """Extract risks from text."""
        risks = []

        for pattern in self.RISK_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match and len(match) > 5:
                    risks.append(match.strip())

        return list(dict.fromkeys(risks))

    def _extract_actions(self, text: str) -> list[str]:
        """Extract action items from text."""
        actions = []

        for pattern in self.ACTION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if match and len(match) > 5:
                    actions.append(match.strip())

        return list(dict.fromkeys(actions))

    def _extract_entities(self, text: str) -> dict[str, str]:
        """Extract named entities from text."""
        entities = {}

        file_pattern = r"[\w\-]+\.(py|js|ts|go|rs|java|cpp|c|h|json|yaml|yml|md)"
        for match in re.findall(file_pattern, text):
            entities[match] = "file"

        class_pattern = r"\b([A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*)\b"
        for match in re.findall(class_pattern, text):
            if len(match) > 3 and match not in entities:
                entities[match] = "class_or_type"

        return entities

    def _extract_key_terms(self, text: str) -> list[str]:
        """Extract key technical terms from text."""
        terms = []

        tech_terms = [
            "api",
            "rest",
            "graphql",
            "grpc",
            "websocket",
            "database",
            "cache",
            "queue",
            "microservice",
            "monolith",
            "authentication",
            "authorization",
            "encryption",
            "token",
            "test",
            "unit",
            "integration",
            "mock",
            "stub",
            "docker",
            "kubernetes",
            "container",
            "deployment",
        ]

        text_lower = text.lower()
        for term in tech_terms:
            if term in text_lower:
                terms.append(term)

        return list(dict.fromkeys(terms))

    def _generate_summary(self, text: str, matrix: SemanticMatrix) -> str:
        """Generate a brief summary of the conversation."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

        if not sentences:
            return ""

        first_sentence = sentences[0][:100] if sentences else ""

        key_points = []
        if matrix.facts:
            key_points.append(f"{len(matrix.facts)} facts")
        if matrix.decisions:
            key_points.append(f"{len(matrix.decisions)} decisions")
        if matrix.pending_risks:
            key_points.append(f"{len(matrix.pending_risks)} risks")

        summary = first_sentence
        if key_points:
            summary += f" [{', '.join(key_points)}]"

        return summary[:200]

    def _estimate_tokens(self, matrix: SemanticMatrix) -> int:
        """Estimate token count for a matrix."""
        total = 0

        total += len(matrix.facts) * 15
        total += len(matrix.decisions) * 30
        total += len(matrix.pending_risks) * 15
        total += len(matrix.action_items) * 15
        total += len(matrix.summary.split()) * 2
        total += len(matrix.key_terms) * 5

        return total

    def _calculate_compression(
        self,
        messages: list[Message],
        matrix: SemanticMatrix,
    ) -> float:
        """Calculate compression ratio."""
        original_tokens = sum(
            len(msg.content.split()) * 1.3 if msg.content else 0 for msg in messages
        )
        if original_tokens == 0:
            return 1.0

        return matrix.token_count / original_tokens

    def _enhance_with_llm(
        self,
        matrix: SemanticMatrix,
        text: str,
    ) -> SemanticMatrix:
        """Enhance distillation with LLM analysis."""
        return matrix

    def get_statistics(self) -> dict[str, Any]:
        """Get distillation statistics."""
        return {
            "message_count": self._message_count,
            "distillation_count": self._distillation_count,
            "archived_matrices": len(self._archived_matrices),
            "current_matrix_exists": self._current_matrix is not None,
            "distill_interval": self.distill_interval,
        }

    def reset(self) -> None:
        """Reset the distiller state."""
        self._current_matrix = None
        self._archived_matrices.clear()
        self._message_count = 0
        self._distillation_count = 0
        self._logger.info("Semantic distiller reset")
