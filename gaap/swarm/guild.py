"""
Guild System - Emergent Fractal Collectives

Guilds are self-organizing groups of Fractals that:
- Share a common domain expertise
- Pool their memories for faster learning
- Vote on SOPs (Standard Operating Procedures)
- Get priority in auctions for their domain

Guild Formation Rules:
1. Minimum 3 Fractals with reputation > 0.7 in domain
2. At least one founder with reputation > 0.8
3. Approval from existing guild members (if any)

Guild Benefits:
- Shared memory pool
- Priority auction access
- Collective voting power
- Knowledge consolidation
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from gaap.swarm.gisp_protocol import (
    ConsensusVote,
)
from gaap.swarm.reputation import ReputationStore


class GuildState(Enum):
    """حالة Guild"""

    FORMING = auto()  # قيد التشكيل
    ACTIVE = auto()  # نشطة
    DORMANT = auto()  # خاملة (أقل من 3 أعضاء)
    DISSOLVED = auto()  # منحلة


@dataclass
class GuildMembership:
    """عضوية في Guild"""

    fractal_id: str
    guild_id: str
    joined_at: datetime = field(default_factory=datetime.now)
    role: str = "member"  # "founder", "member", "senior"
    contributions: int = 0
    reputation_at_join: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractal_id": self.fractal_id,
            "guild_id": self.guild_id,
            "joined_at": self.joined_at.isoformat(),
            "role": self.role,
            "contributions": self.contributions,
            "reputation_at_join": self.reputation_at_join,
        }


@dataclass
class GuildProposal:
    """اقتراح للتصويت"""

    proposal_id: str
    proposal_type: str  # "SOP", "MEMBER_JOIN", "POLICY"
    content: str
    proposer_id: str
    created_at: datetime = field(default_factory=datetime.now)
    votes: dict[str, ConsensusVote] = field(default_factory=dict)
    status: str = "pending"  # "pending", "approved", "rejected"
    required_votes: int = 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "proposal_type": self.proposal_type,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "proposer_id": self.proposer_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "votes": {k: v.to_dict() for k, v in self.votes.items()},
        }


@dataclass
class GuildMemory:
    """ذاكرة مشتركة للـ Guild"""

    entries: list[dict[str, Any]] = field(default_factory=list)
    consolidated_sops: list[str] = field(default_factory=list)
    last_consolidation: datetime = field(default_factory=datetime.now)

    def add_entry(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        # Keep last 1000 entries
        if len(self.entries) > 1000:
            self.entries = self.entries[-1000:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries_count": len(self.entries),
            "consolidated_sops": self.consolidated_sops,
            "last_consolidation": self.last_consolidation.isoformat(),
        }


class Guild:
    """
    تجمع Fractals الذكي.

    Features:
    - Emergent formation based on reputation
    - Shared memory pool
    - Consensus-based decision making
    - SOP creation and storage

    Usage:
        guild = Guild(
            guild_id="python_guild",
            domain="python",
            reputation_store=reputation_store,
        )

        # Add member
        guild.add_member("coder_01", role="founder")

        # Propose SOP
        proposal = guild.create_proposal(
            proposal_type="SOP",
            content="Always use type hints in function definitions",
            proposer_id="coder_01",
        )

        # Vote
        guild.vote(proposal.proposal_id, vote)

        # Share memory
        guild.share_memory(entry, source_fractal="coder_01")
    """

    MIN_MEMBERS = 3
    MIN_FOUNDER_REPUTATION = 0.8
    MIN_MEMBER_REPUTATION = 0.7

    def __init__(
        self,
        guild_id: str,
        domain: str,
        reputation_store: ReputationStore,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.guild_id = guild_id
        self.domain = domain
        self._reputation = reputation_store
        self._config = config or {}
        self._logger = logging.getLogger(f"gaap.swarm.guild.{guild_id}")

        # State
        self._state = GuildState.FORMING
        self._name = f"{domain.title()} Guild"

        # Members
        self._members: dict[str, GuildMembership] = {}
        self._pending_invites: set[str] = set()

        # Memory
        self._memory = GuildMemory()

        # Proposals
        self._proposals: dict[str, GuildProposal] = {}
        self._approved_sops: list[str] = []

        # Stats
        self._created_at = datetime.now()
        self._tasks_completed = 0
        self._collective_reputation = 0.0

    @property
    def state(self) -> GuildState:
        """حالة Guild"""
        if len(self._members) >= self.MIN_MEMBERS:
            if self._state == GuildState.FORMING:
                self._state = GuildState.ACTIVE
                self._logger.info(
                    f"Guild {self.guild_id} is now ACTIVE with {len(self._members)} members"
                )
        elif self._state == GuildState.ACTIVE:
            self._state = GuildState.DORMANT

        return self._state

    @property
    def member_count(self) -> int:
        return len(self._members)

    @property
    def members(self) -> list[str]:
        return list(self._members.keys())

    def can_join(self, fractal_id: str) -> tuple[bool, str]:
        """
        التحقق من إمكانية الانضمام.

        Returns:
            (can_join, reason)
        """
        if fractal_id in self._members:
            return False, "Already a member"

        reputation = self._reputation.get_domain_reputation(fractal_id, self.domain)

        if reputation < self.MIN_MEMBER_REPUTATION:
            return (
                False,
                f"Reputation {reputation:.2f} below threshold {self.MIN_MEMBER_REPUTATION}",
            )

        return True, "Eligible"

    def add_member(
        self,
        fractal_id: str,
        role: str = "member",
    ) -> GuildMembership | None:
        """
        إضافة عضو جديد.

        For non-founders, requires existing member approval.
        """
        can_join, reason = self.can_join(fractal_id)

        if not can_join:
            self._logger.warning(f"Cannot add {fractal_id}: {reason}")
            return None

        reputation = self._reputation.get_domain_reputation(fractal_id, self.domain)

        membership = GuildMembership(
            fractal_id=fractal_id,
            guild_id=self.guild_id,
            role=role,
            reputation_at_join=reputation,
        )

        self._members[fractal_id] = membership
        self._update_collective_reputation()

        self._logger.info(
            f"Added {fractal_id} as {role} to {self.guild_id} (reputation={reputation:.2f})"
        )

        return membership

    def remove_member(self, fractal_id: str) -> bool:
        """إزالة عضو"""
        if fractal_id not in self._members:
            return False

        del self._members[fractal_id]
        self._update_collective_reputation()

        self._logger.info(f"Removed {fractal_id} from {self.guild_id}")
        return True

    def _update_collective_reputation(self) -> None:
        """تحديث السمعة الجماعية"""
        if not self._members:
            self._collective_reputation = 0.0
            return

        total = sum(
            self._reputation.get_domain_reputation(m.fractal_id, self.domain)
            for m in self._members.values()
        )
        self._collective_reputation = total / len(self._members)

    def create_proposal(
        self,
        proposal_type: str,
        content: str,
        proposer_id: str,
    ) -> GuildProposal | None:
        """
        إنشاء اقتراح للتصويت.
        """
        if proposer_id not in self._members:
            self._logger.warning(f"Non-member {proposer_id} cannot propose")
            return None

        proposal = GuildProposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            proposal_type=proposal_type,
            content=content,
            proposer_id=proposer_id,
            required_votes=max(3, len(self._members) // 2),
        )

        self._proposals[proposal.proposal_id] = proposal

        self._logger.info(
            f"Created {proposal_type} proposal {proposal.proposal_id} by {proposer_id}"
        )

        return proposal

    def vote(self, proposal_id: str, vote: ConsensusVote) -> bool:
        """
        تسجيل صوت.
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return False

        if vote.voter_id not in self._members:
            return False

        # Weight vote by reputation
        vote.vote_weight = self._reputation.get_domain_reputation(vote.voter_id, self.domain)

        proposal.votes[vote.voter_id] = vote

        # Check if we have enough votes
        self._evaluate_proposal(proposal)

        return True

    def _evaluate_proposal(self, proposal: GuildProposal) -> None:
        """تقييم الاقتراح"""
        if proposal.status != "pending":
            return

        # Count weighted votes
        approve_weight = 0.0
        reject_weight = 0.0

        for vote in proposal.votes.values():
            if vote.vote == "APPROVE":
                approve_weight += vote.vote_weight
            elif vote.vote == "REJECT":
                reject_weight += vote.vote_weight

        # Check thresholds
        if approve_weight >= proposal.required_votes:
            proposal.status = "approved"
            self._on_proposal_approved(proposal)
        elif reject_weight >= proposal.required_votes:
            proposal.status = "rejected"

    def _on_proposal_approved(self, proposal: GuildProposal) -> None:
        """عند الموافقة على اقتراح"""
        if proposal.proposal_type == "SOP":
            self._approved_sops.append(proposal.content)
            self._memory.consolidated_sops.append(proposal.proposal_id)
            self._logger.info(f"New SOP approved: {proposal.proposal_id}")

    def share_memory(
        self,
        entry: dict[str, Any],
        source_fractal: str,
    ) -> bool:
        """
        مشاركة ذاكرة بين الأعضاء.
        """
        if source_fractal not in self._members:
            return False

        entry["source"] = source_fractal
        entry["shared_at"] = datetime.now().isoformat()

        self._memory.add_entry(entry)

        # Update contribution count
        self._members[source_fractal].contributions += 1

        return True

    def get_shared_memory(
        self,
        query: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """
        الحصول على الذكريات المشتركة.
        """
        entries = self._memory.entries

        if query:
            # Simple text matching
            entries = [e for e in entries if query.lower() in str(e).lower()]

        return entries[-limit:]

    def get_best_member_for_task(
        self,
        task_complexity: int = 5,
    ) -> str | None:
        """
        الحصول على أفضل عضو للمهمة.

        Factors:
        - Reputation
        - Current load (if known)
        - Recent performance
        """
        if not self._members:
            return None

        best_member = None
        best_score = 0.0

        for fractal_id in self._members:
            reputation = self._reputation.get_domain_reputation(fractal_id, self.domain)

            # Higher complexity = need higher reputation
            required_rep = 0.5 + (task_complexity / 20)

            if reputation >= required_rep and reputation > best_score:
                best_score = reputation
                best_member = fractal_id

        return best_member

    def get_sops(self) -> list[str]:
        """الحصول على SOPs المعتمدة"""
        return self._approved_sops.copy()

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات Guild"""
        return {
            "guild_id": self.guild_id,
            "name": self._name,
            "domain": self.domain,
            "state": self.state.name,
            "member_count": self.member_count,
            "members": self.members,
            "collective_reputation": round(self._collective_reputation, 4),
            "approved_sops": len(self._approved_sops),
            "memory_entries": len(self._memory.entries),
            "tasks_completed": self._tasks_completed,
            "created_at": self._created_at.isoformat(),
        }

    def dissolve(self) -> None:
        """حل Guild"""
        self._state = GuildState.DISSOLVED
        self._members.clear()
        self._logger.info(f"Guild {self.guild_id} dissolved")

    @classmethod
    def can_form_guild(
        cls,
        domain: str,
        fractals: list[str],
        reputation_store: ReputationStore,
    ) -> tuple[bool, str]:
        """
        التحقق من إمكانية تكوين Guild.

        Rules:
        - At least MIN_MEMBERS fractals
        - At least one founder with reputation >= MIN_FOUNDER_REPUTATION
        - All members must have reputation >= MIN_MEMBER_REPUTATION
        """
        if len(fractals) < cls.MIN_MEMBERS:
            return False, f"Need at least {cls.MIN_MEMBERS} members"

        has_founder = False

        for fractal_id in fractals:
            reputation = reputation_store.get_domain_reputation(fractal_id, domain)

            if reputation >= cls.MIN_FOUNDER_REPUTATION:
                has_founder = True
            elif reputation < cls.MIN_MEMBER_REPUTATION:
                return False, f"{fractal_id} reputation {reputation:.2f} too low"

        if not has_founder:
            return (
                False,
                f"Need at least one founder with reputation >= {cls.MIN_FOUNDER_REPUTATION}",
            )

        return True, "Can form guild"
