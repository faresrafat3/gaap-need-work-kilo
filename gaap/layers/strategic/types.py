"""Strategic layer shared types.

This module contains type definitions shared between strategic layer components
to avoid circular import issues.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ArchitectureParadigm(Enum):
    """Architecture paradigm patterns."""

    MONOLITH = "monolith"
    MODULAR_MONOLITH = "modular_monolith"
    MICROSERVICES = "microservices"
    SERVERLESS = "serverless"
    EVENT_DRIVEN = "event_driven"
    LAYERED = "layered"
    HEXAGONAL = "hexagonal"


class DataStrategy(Enum):
    """Data management strategies."""

    SINGLE_DB = "single_database"
    POLYGLOT = "polyglot"
    CQRS = "cqrs"
    EVENT_SOURCING = "event_sourcing"


class CommunicationPattern(Enum):
    """Inter-service communication patterns."""

    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    EVENT_BUS = "event_bus"


@dataclass
class ArchitectureDecision:
    """Single architecture decision record."""

    aspect: str
    choice: str
    reasoning: str
    trade_offs: list[str]
    confidence: float


@dataclass
class ArchitectureSpec:
    """Complete architecture specification."""

    spec_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    paradigm: ArchitectureParadigm = field(
        default_factory=lambda: ArchitectureParadigm.MODULAR_MONOLITH
    )
    data_strategy: DataStrategy = field(default_factory=lambda: DataStrategy.SINGLE_DB)
    communication: CommunicationPattern = field(default_factory=lambda: CommunicationPattern.REST)
    components: list[dict[str, Any]] = field(default_factory=list)
    tech_stack: dict[str, str] = field(default_factory=dict)
    decisions: list[ArchitectureDecision] = field(default_factory=list)
    risks: list[dict[str, Any]] = field(default_factory=list)
    estimated_resources: dict[str, Any] = field(default_factory=dict)
    explored_paths: int = 0
    selected_path_score: float = 0.0
    debate_rounds: int = 0
    consensus_reached: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to dictionary."""
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
