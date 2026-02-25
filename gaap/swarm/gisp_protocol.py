"""
GISP Protocol v2.0 - GAAP Internal Swarm Protocol

Defines message types for decentralized agent communication.

Key Message Types:
- TASK_AUCTION: Orchestrator broadcasts task for bidding
- TASK_BID: Fractal submits a bid with utility score
- TASK_AWARD: Orchestrator awards task to winning bidder
- CONSENSUS_VOTE: Fractals vote on SOPs
- MEMORY_SHARE: Guild members share memories
- GUILD_FORM: Request to form a guild

Architecture:
    Orchestrator                  Fractals
         │                           │
         │──── TASK_AUCTION ────────>│
         │                           │
         │<──── TASK_BID ────────────│
         │<──── TASK_BID ────────────│
         │<──── TASK_BID ────────────│
         │                           │
         │──── TASK_AWARD ──────────>│
         │                           │
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Literal
import uuid


class MessageType(Enum):
    """أنواع رسائل GISP"""

    # Auction messages
    TASK_AUCTION = auto()
    TASK_BID = auto()
    TASK_AWARD = auto()
    TASK_RESULT = auto()

    # Consensus messages
    CONSENSUS_PROPOSAL = auto()
    CONSENSUS_VOTE = auto()
    CONSENSUS_RESULT = auto()

    # Guild messages
    GUILD_FORM = auto()
    GUILD_JOIN = auto()
    GUILD_LEAVE = auto()
    GUILD_DISSOLVE = auto()

    # Memory messages
    MEMORY_SHARE = auto()
    MEMORY_REQUEST = auto()

    # Status messages
    HEARTBEAT = auto()
    CAPABILITY_ANNOUNCE = auto()
    ERROR = auto()


class TaskPriority(Enum):
    """أولوية المهمة"""

    CRITICAL = auto()  # Must complete immediately
    HIGH = auto()  # Important but not urgent
    NORMAL = auto()  # Standard priority
    LOW = auto()  # Can wait
    BACKGROUND = auto()  # Run when idle


class TaskDomain(Enum):
    """مجالات المهام"""

    PYTHON = "python"
    SQL = "sql"
    SECURITY = "security"
    FRONTEND = "frontend"
    RESEARCH = "research"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"


@dataclass
class GISPMessage:
    """
    Base class for all GISP messages.

    All messages include:
    - Unique message ID
    - Timestamp
    - Sender/receiver info
    - Correlation ID for request-response patterns
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    sender_id: str = ""
    receiver_id: str = ""  # Empty = broadcast
    correlation_id: str = ""  # For request-response matching
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "correlation_id": self.correlation_id,
            "metadata": self.metadata,
        }


@dataclass
class TaskAuction(GISPMessage):
    """
    رسالة مزاد المهمة.

    يرسلها Orchestrator لطلب عروض من Fractals.

    Attributes:
        task_id: معرف المهمة
        task_description: وصف المهمة
        domain: مجال المهمة
        complexity: مستوى التعقيد (1-10)
        priority: أولوية المهمة
        requirements: متطلبات إضافية
        constraints: قيود (موارد، وقت، etc.)
        min_reputation: الحد الأدنى للسمعة المطلوبة
        timeout_seconds: مهلة المزاد
        reward_tokens: مكافأة ناجحة (tokens)
        penalty_tokens: عقوبة الفشل (tokens)

    Example:
        auction = TaskAuction(
            task_id="task_123",
            task_description="Implement user authentication with OAuth2",
            domain=TaskDomain.SECURITY,
            complexity=7,
            priority=TaskPriority.HIGH,
            min_reputation=0.6,
            timeout_seconds=30,
        )
    """

    message_type: Literal[MessageType.TASK_AUCTION] = MessageType.TASK_AUCTION
    task_id: str = ""
    task_description: str = ""
    domain: TaskDomain = TaskDomain.GENERAL
    complexity: int = 5  # 1-10
    priority: TaskPriority = TaskPriority.NORMAL
    requirements: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    min_reputation: float = 0.3
    timeout_seconds: int = 30
    reward_tokens: float = 100.0
    penalty_tokens: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "TASK_AUCTION",
            "task_id": self.task_id,
            "task_description": self.task_description,
            "domain": self.domain.value,
            "complexity": self.complexity,
            "priority": self.priority.name,
            "requirements": self.requirements,
            "constraints": self.constraints,
            "min_reputation": self.min_reputation,
            "timeout_seconds": self.timeout_seconds,
            "reward_tokens": self.reward_tokens,
            "penalty_tokens": self.penalty_tokens,
        }


@dataclass
class TaskBid(GISPMessage):
    """
    عرض من Fractal للمهمة.

    يحتوي على:
    - تقدير معدل النجاح
    - تقدير التكلفة والوقت
    - درجة المنفعة المحسوبة
    - المبرر

    Utility Score Formula:
    U = w1 * success_rate + w2 * reputation + w3 * (1/cost) + w4 * (1/time)

    Default weights:
    - w1 = 0.35 (success rate)
    - w2 = 0.25 (reputation)
    - w3 = 0.20 (cost efficiency)
    - w4 = 0.20 (time efficiency)
    """

    message_type: Literal[MessageType.TASK_BID] = MessageType.TASK_BID
    task_id: str = ""
    bidder_id: str = ""

    # Estimates
    estimated_success_rate: float = 0.5  # 0.0 to 1.0
    estimated_cost_tokens: float = 100.0
    estimated_time_seconds: float = 60.0

    # Computed
    utility_score: float = 0.0

    # Supporting info
    rationale: str = ""
    similar_tasks_completed: int = 0
    current_load: float = 0.0  # 0.0 to 1.0

    # Confidence
    confidence_in_estimate: float = 0.5  # How confident in the bid

    # Weights for utility calculation
    WEIGHT_SUCCESS: float = 0.35
    WEIGHT_REPUTATION: float = 0.25
    WEIGHT_COST: float = 0.20
    WEIGHT_TIME: float = 0.20

    def compute_utility_score(
        self,
        reputation: float,
        max_cost: float = 1000.0,
        max_time: float = 300.0,
    ) -> float:
        """
        حساب درجة المنفعة.

        Args:
            reputation: سمعة Fractal في المجال
            max_cost: أقصى تكلفة متوقعة (للتطبيع)
            max_time: أقصى وقت متوقع (للتطبيع)

        Returns:
            Utility score (0.0 to 1.0)
        """
        # Normalize cost and time (inverse - lower is better)
        cost_score = 1.0 - min(self.estimated_cost_tokens / max_cost, 1.0)
        time_score = 1.0 - min(self.estimated_time_seconds / max_time, 1.0)

        # Adjust for current load
        load_penalty = self.current_load * 0.3

        # Compute weighted score
        raw_score = (
            self.WEIGHT_SUCCESS * self.estimated_success_rate
            + self.WEIGHT_REPUTATION * reputation
            + self.WEIGHT_COST * cost_score
            + self.WEIGHT_TIME * time_score
        )

        # Apply load penalty and confidence weighting
        adjusted_score = raw_score * (1 - load_penalty) * self.confidence_in_estimate

        self.utility_score = max(0.0, min(1.0, adjusted_score))
        return self.utility_score

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "TASK_BID",
            "task_id": self.task_id,
            "bidder_id": self.bidder_id,
            "estimated_success_rate": self.estimated_success_rate,
            "estimated_cost_tokens": self.estimated_cost_tokens,
            "estimated_time_seconds": self.estimated_time_seconds,
            "utility_score": round(self.utility_score, 4),
            "rationale": self.rationale,
            "similar_tasks_completed": self.similar_tasks_completed,
            "current_load": self.current_load,
            "confidence_in_estimate": self.confidence_in_estimate,
        }


@dataclass
class TaskAward(GISPMessage):
    """
    إسناد المهمة لـ Fractal فائز.

    يرسله Orchestrator بعد انتهاء المزاد.
    """

    message_type: Literal[MessageType.TASK_AWARD] = MessageType.TASK_AWARD
    task_id: str = ""
    winner_id: str = ""
    utility_score: float = 0.0

    # Winning bid details
    winning_bid: TaskBid | None = None

    # Runner-ups (for backup assignment)
    runner_ups: list[str] = field(default_factory=list)

    # Task details
    deadline: datetime | None = None
    checkpoint_interval: int = 60  # seconds between checkpoints

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "TASK_AWARD",
            "task_id": self.task_id,
            "winner_id": self.winner_id,
            "utility_score": self.utility_score,
            "winning_bid": self.winning_bid.to_dict() if self.winning_bid else None,
            "runner_ups": self.runner_ups,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "checkpoint_interval": self.checkpoint_interval,
        }


@dataclass
class TaskResult(GISPMessage):
    """
    نتيجة تنفيذ المهمة.

    يرسله Fractal بعد الانتهاء.
    """

    message_type: Literal[MessageType.TASK_RESULT] = MessageType.TASK_RESULT
    task_id: str = ""
    fractal_id: str = ""

    success: bool = False
    output: Any = None
    error: str | None = None

    # Metrics
    actual_cost_tokens: float = 0.0
    actual_time_seconds: float = 0.0

    # Quality assessment
    quality_score: float = 0.0  # 0.0 to 1.0
    self_assessment: str = ""  # Fractal's own assessment

    # Epistemic humility
    predicted_success: bool = True  # Did fractal predict this outcome?
    confidence_before: float = 0.5
    confidence_after: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "TASK_RESULT",
            "task_id": self.task_id,
            "fractal_id": self.fractal_id,
            "success": self.success,
            "error": self.error,
            "actual_cost_tokens": self.actual_cost_tokens,
            "actual_time_seconds": self.actual_time_seconds,
            "quality_score": self.quality_score,
            "self_assessment": self.self_assessment,
            "predicted_success": self.predicted_success,
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
        }


@dataclass
class ConsensusVote(GISPMessage):
    """
    تصويت على SOP أو قرار.

    يستخدم في Guild لاتخاذ قرارات جماعية.

    قواعد الإجماع:
    - إذا وافق 3+ Fractals على SOP، يُضاف للذاكرة
    - إذا رفض 3+ Fractals، يُرفض
    - إذا كان التصويت متقارب، يُحفظ للمراجعة
    """

    message_type: Literal[MessageType.CONSENSUS_VOTE] = MessageType.CONSENSUS_VOTE
    proposal_id: str = ""
    proposal_type: str = ""  # "SOP", "GUILD_FORM", "POLICY"
    voter_id: str = ""

    vote: Literal["APPROVE", "REJECT", "ABSTAIN"] = "ABSTAIN"
    confidence: float = 0.5
    reasoning: str = ""

    # Weight of this vote (based on voter's reputation)
    vote_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "CONSENSUS_VOTE",
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "voter_id": self.voter_id,
            "vote": self.vote,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "vote_weight": self.vote_weight,
        }


@dataclass
class GuildForm(GISPMessage):
    """
    طلب تكوين Guild جديد.

    Guild = تجمع Fractals متخصصة في مجال معين.
    """

    message_type: Literal[MessageType.GUILD_FORM] = MessageType.GUILD_FORM
    guild_id: str = ""
    guild_name: str = ""
    domain: str = ""
    founder_id: str = ""

    # Requirements for joining
    min_reputation: float = 0.7
    min_tasks_completed: int = 10

    # Initial members
    invited_members: list[str] = field(default_factory=list)

    # Benefits
    shared_memory: bool = True
    priority_auctions: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "GUILD_FORM",
            "guild_id": self.guild_id,
            "guild_name": self.guild_name,
            "domain": self.domain,
            "founder_id": self.founder_id,
            "min_reputation": self.min_reputation,
            "min_tasks_completed": self.min_tasks_completed,
            "invited_members": self.invited_members,
            "shared_memory": self.shared_memory,
            "priority_auctions": self.priority_auctions,
        }


@dataclass
class MemoryShare(GISPMessage):
    """
    مشاركة ذكريات بين Fractals (داخل Guild).

    يسمح للـ Fractals بتبادل الخبرات دون المرور بـ Orchestrator.
    """

    message_type: Literal[MessageType.MEMORY_SHARE] = MessageType.MEMORY_SHARE
    source_fractal: str = ""
    target_fractal: str = ""  # Empty = broadcast to guild

    # Memory content
    memory_type: str = ""  # "episodic", "semantic", "procedural"
    domain: str = ""
    content: dict[str, Any] = field(default_factory=dict)

    # Metadata
    relevance_score: float = 0.0
    timestamp_original: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "MEMORY_SHARE",
            "source_fractal": self.source_fractal,
            "target_fractal": self.target_fractal,
            "memory_type": self.memory_type,
            "domain": self.domain,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "timestamp_original": self.timestamp_original.isoformat(),
        }


@dataclass
class CapabilityAnnounce(GISPMessage):
    """
    إعلان عن قدرات Fractal.

    يرسله Fractal عند التسجيل أو تحديث قدراته.
    """

    message_type: Literal[MessageType.CAPABILITY_ANNOUNCE] = MessageType.CAPABILITY_ANNOUNCE
    fractal_id: str = ""

    # Capabilities
    domains: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)

    # Resources
    max_concurrent_tasks: int = 1
    max_memory_mb: int = 1000
    gpu_available: bool = False

    # Preferences
    preferred_task_types: list[str] = field(default_factory=list)
    avoided_task_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "message_type": "CAPABILITY_ANNOUNCE",
            "fractal_id": self.fractal_id,
            "domains": self.domains,
            "skills": self.skills,
            "tools": self.tools,
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "max_memory_mb": self.max_memory_mb,
            "gpu_available": self.gpu_available,
            "preferred_task_types": self.preferred_task_types,
            "avoided_task_types": self.avoided_task_types,
        }


def parse_message(data: dict[str, Any]) -> GISPMessage:
    """
    Parse a message from dict.

    Factory function to create the correct message type.
    """
    msg_type = data.get("message_type", "")

    message_classes: dict[str, type[GISPMessage]] = {
        "TASK_AUCTION": TaskAuction,
        "TASK_BID": TaskBid,
        "TASK_AWARD": TaskAward,
        "TASK_RESULT": TaskResult,
        "CONSENSUS_VOTE": ConsensusVote,
        "GUILD_FORM": GuildForm,
        "MEMORY_SHARE": MemoryShare,
        "CAPABILITY_ANNOUNCE": CapabilityAnnounce,
    }

    msg_class = message_classes.get(msg_type)
    if not msg_class:
        raise ValueError(f"Unknown message type: {msg_type}")

    return msg_class(**data)
