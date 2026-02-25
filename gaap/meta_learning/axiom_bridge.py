"""
Axiom Bridge - Automated Axiom Proposal
======================================

Bridges learned heuristics to constitutional axioms.
When a heuristic reaches high confidence and success rate,
it can be proposed as a formal axiom.

Usage:
    bridge = AxiomBridge(axiom_validator)

    # Propose from heuristic
    proposal = bridge.propose(heuristic)
    if proposal.status == ProposalStatus.APPROVED:
        bridge.commit(proposal)
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

from gaap.storage.atomic import atomic_write

from gaap.core.axioms import Axiom, AxiomLevel, AxiomValidator

logger = logging.getLogger("gaap.meta_learning.axiom_bridge")


class ProposalStatus(Enum):
    """حالة اقتراح البديهية"""

    PENDING = auto()
    UNDER_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    COMMITTED = auto()
    DEPRECATED = auto()


class ProposalType(Enum):
    """نوع الاقتراح"""

    NEW_AXIOM = auto()
    MODIFY_AXIOM = auto()
    DEPRECATE_AXIOM = auto()
    STRENGTHEN_AXIOM = auto()


@dataclass
class AxiomProposal:
    """
    Proposal for a new or modified axiom.

    Tracks the lifecycle from heuristic to constitutional rule.
    """

    proposal_type: ProposalType
    axiom_name: str
    axiom_description: str
    axiom_level: AxiomLevel = AxiomLevel.GUIDELINE
    status: ProposalStatus = ProposalStatus.PENDING

    evidence: list[str] = field(default_factory=list)
    success_rate: float = 0.0
    sample_size: int = 0
    source_heuristic_id: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    reviewed_at: datetime | None = None
    reviewed_by: str | None = None
    committed_at: datetime | None = None

    rejection_reason: str | None = None
    review_notes: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def get_id(self) -> str:
        content = f"{self.axiom_name}:{self.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.get_id(),
            "proposal_type": self.proposal_type.name,
            "axiom_name": self.axiom_name,
            "axiom_description": self.axiom_description,
            "axiom_level": self.axiom_level.name,
            "status": self.status.name,
            "evidence": self.evidence,
            "success_rate": self.success_rate,
            "sample_size": self.sample_size,
            "source_heuristic_id": self.source_heuristic_id,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewed_by": self.reviewed_by,
            "committed_at": self.committed_at.isoformat() if self.committed_at else None,
            "rejection_reason": self.rejection_reason,
            "review_notes": self.review_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AxiomProposal":
        return cls(
            proposal_type=ProposalType[data.get("proposal_type", "NEW_AXIOM")],
            axiom_name=data.get("axiom_name", ""),
            axiom_description=data.get("axiom_description", ""),
            axiom_level=AxiomLevel[data.get("axiom_level", "GUIDELINE")],
            status=ProposalStatus[data.get("status", "PENDING")],
            evidence=data.get("evidence", []),
            success_rate=data.get("success_rate", 0.0),
            sample_size=data.get("sample_size", 0),
            source_heuristic_id=data.get("source_heuristic_id"),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            reviewed_at=datetime.fromisoformat(data["reviewed_at"])
            if data.get("reviewed_at")
            else None,
            reviewed_by=data.get("reviewed_by"),
            committed_at=datetime.fromisoformat(data["committed_at"])
            if data.get("committed_at")
            else None,
            rejection_reason=data.get("rejection_reason"),
            review_notes=data.get("review_notes"),
            metadata=data.get("metadata", {}),
        )


class AxiomBridge:
    """
    Manages the lifecycle of axiom proposals.

    Features:
    - Convert high-confidence heuristics to axiom proposals
    - Track proposal review status
    - Auto-approve based on criteria
    - Integrate with AxiomValidator
    - Maintain constitution.yaml
    """

    DEFAULT_STORAGE_PATH = ".gaap/memory/proposals"
    CONSTITUTION_PATH = ".gaap/constitution.yaml"

    AUTO_APPROVE_THRESHOLD = 0.9
    AUTO_APPROVE_MIN_SAMPLES = 10

    def __init__(
        self,
        storage_path: str | None = None,
        axiom_validator: AxiomValidator | None = None,
    ) -> None:
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self._validator = axiom_validator

        self._proposals: dict[str, AxiomProposal] = {}
        self._pending_queue: list[str] = []
        self._committed_axioms: dict[str, str] = {}

        self._logger = logger

        self._load()

    def propose_from_heuristic(
        self,
        heuristic: Any,
        force_level: AxiomLevel | None = None,
    ) -> AxiomProposal:
        """
        Create a proposal from a learned heuristic.

        Args:
            heuristic: ProjectHeuristic to convert
            force_level: Optional specific axiom level

        Returns:
            Created AxiomProposal
        """
        axiom_name = self._generate_axiom_name(heuristic.principle)

        level = force_level or self._determine_level(heuristic)

        proposal = AxiomProposal(
            proposal_type=ProposalType.NEW_AXIOM,
            axiom_name=axiom_name,
            axiom_description=heuristic.principle,
            axiom_level=level,
            status=ProposalStatus.PENDING,
            evidence=heuristic.examples[:5],
            success_rate=heuristic.success_rate,
            sample_size=heuristic.evidence_count,
            source_heuristic_id=heuristic.get_id(),
        )

        if self._should_auto_approve(heuristic):
            proposal.status = ProposalStatus.APPROVED
            proposal.reviewed_at = datetime.now()
            proposal.reviewed_by = "auto_approve"
            self._logger.info(f"Auto-approved axiom proposal: {axiom_name}")
        else:
            self._pending_queue.append(proposal.get_id())

        self._proposals[proposal.get_id()] = proposal
        self._save()

        return proposal

    def propose(
        self,
        axiom_name: str,
        axiom_description: str,
        axiom_level: AxiomLevel = AxiomLevel.GUIDELINE,
        evidence: list[str] | None = None,
        success_rate: float = 0.0,
        sample_size: int = 0,
    ) -> AxiomProposal:
        """
        Create a new axiom proposal manually.

        Args:
            axiom_name: Name for the axiom
            axiom_description: Description of the axiom
            axiom_level: Level of the axiom
            evidence: Supporting evidence
            success_rate: Success rate of this pattern
            sample_size: Number of supporting cases

        Returns:
            Created AxiomProposal
        """
        proposal = AxiomProposal(
            proposal_type=ProposalType.NEW_AXIOM,
            axiom_name=axiom_name,
            axiom_description=axiom_description,
            axiom_level=axiom_level,
            status=ProposalStatus.PENDING,
            evidence=evidence or [],
            success_rate=success_rate,
            sample_size=sample_size,
        )

        self._proposals[proposal.get_id()] = proposal
        self._pending_queue.append(proposal.get_id())
        self._save()

        return proposal

    def review(
        self,
        proposal_id: str,
        approved: bool,
        reviewer: str = "manual",
        notes: str | None = None,
    ) -> bool:
        """
        Review a pending proposal.

        Args:
            proposal_id: ID of proposal to review
            approved: Whether to approve
            reviewer: Reviewer identifier
            notes: Optional review notes

        Returns:
            True if review was successful
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal:
            return False

        proposal.reviewed_at = datetime.now()
        proposal.reviewed_by = reviewer
        proposal.review_notes = notes

        if approved:
            proposal.status = ProposalStatus.APPROVED
            self._logger.info(f"Proposal approved: {proposal.axiom_name} by {reviewer}")
        else:
            proposal.status = ProposalStatus.REJECTED
            proposal.rejection_reason = notes or "Rejected by reviewer"
            self._logger.info(f"Proposal rejected: {proposal.axiom_name} by {reviewer}")

        if proposal_id in self._pending_queue:
            self._pending_queue.remove(proposal_id)

        self._save()
        return True

    def commit(self, proposal_id: str) -> bool:
        """
        Commit an approved proposal to the axiom system.

        Args:
            proposal_id: ID of approved proposal

        Returns:
            True if commit was successful
        """
        proposal = self._proposals.get(proposal_id)
        if not proposal or proposal.status != ProposalStatus.APPROVED:
            return False

        if self._validator:
            axiom = Axiom(
                name=proposal.axiom_name,
                description=proposal.axiom_description,
                level=proposal.axiom_level,
                enabled=True,
                check_func="check_custom",
            )

            self._validator.axioms[proposal.axiom_name] = axiom

        proposal.status = ProposalStatus.COMMITTED
        proposal.committed_at = datetime.now()

        self._committed_axioms[proposal.axiom_name] = proposal_id

        self._save()
        self._update_constitution(proposal)

        self._logger.info(f"Committed axiom: {proposal.axiom_name} ({proposal.axiom_level.name})")

        return True

    def deprecate_axiom(
        self,
        axiom_name: str,
        reason: str,
    ) -> AxiomProposal | None:
        """
        Propose deprecation of an existing axiom.

        Args:
            axiom_name: Name of axiom to deprecate
            reason: Reason for deprecation

        Returns:
            Created deprecation proposal
        """
        proposal = AxiomProposal(
            proposal_type=ProposalType.DEPRECATE_AXIOM,
            axiom_name=axiom_name,
            axiom_description=f"DEPRECATE: {reason}",
            axiom_level=AxiomLevel.PREFERENCE,
            status=ProposalStatus.PENDING,
            rejection_reason=reason,
        )

        self._proposals[proposal.get_id()] = proposal
        self._pending_queue.append(proposal.get_id())
        self._save()

        return proposal

    def get_pending_proposals(self) -> list[AxiomProposal]:
        """Get all pending proposals."""
        return [self._proposals[pid] for pid in self._pending_queue if pid in self._proposals]

    def get_committed_axioms(self) -> list[AxiomProposal]:
        """Get all committed axiom proposals."""
        return [p for p in self._proposals.values() if p.status == ProposalStatus.COMMITTED]

    def get_stats(self) -> dict[str, Any]:
        """Get bridge statistics."""
        by_status: dict[str, int] = {}
        for p in self._proposals.values():
            by_status[p.status.name] = by_status.get(p.status.name, 0) + 1

        return {
            "total_proposals": len(self._proposals),
            "pending": len(self._pending_queue),
            "committed": len(self._committed_axioms),
            "by_status": by_status,
            "auto_approve_threshold": self.AUTO_APPROVE_THRESHOLD,
        }

    def _should_auto_approve(self, heuristic: Any) -> bool:
        """Check if heuristic meets auto-approval criteria."""
        return (
            heuristic.confidence >= self.AUTO_APPROVE_THRESHOLD
            and heuristic.evidence_count >= self.AUTO_APPROVE_MIN_SAMPLES
            and heuristic.success_rate >= 0.85
            and len(heuristic.counter_examples) == 0
            and heuristic.status.name == "VALIDATED"
        )

    def _determine_level(self, heuristic: Any) -> AxiomLevel:
        """Determine appropriate axiom level from heuristic."""
        if heuristic.confidence >= 0.95 and heuristic.evidence_count >= 20:
            return AxiomLevel.INVARIANT
        elif heuristic.confidence >= 0.85:
            return AxiomLevel.GUIDELINE
        else:
            return AxiomLevel.PREFERENCE

    def _generate_axiom_name(self, principle: str) -> str:
        """Generate a valid axiom name from principle."""
        words = principle.lower().split()[:5]
        name = "_".join(w for w in words if w.isalnum())
        name = "".join(c for c in name if c.isalnum() or c == "_")

        if len(name) < 3:
            name = f"axiom_{hashlib.md5(principle.encode()).hexdigest()[:6]}"

        return name

    def _update_constitution(self, proposal: AxiomProposal) -> bool:
        """Update constitution.yaml with new axiom."""
        import yaml

        constitution_path = Path(self.CONSTITUTION_PATH)

        try:
            constitution_path.parent.mkdir(parents=True, exist_ok=True)

            data: dict[str, Any] = {}
            if constitution_path.exists():
                with open(constitution_path) as f:
                    data = yaml.safe_load(f) or {}

            if "axioms" not in data:
                data["axioms"] = []

            axiom_entry = {
                "name": proposal.axiom_name,
                "description": proposal.axiom_description,
                "level": proposal.axiom_level.name,
                "added_at": proposal.committed_at.isoformat()
                if proposal.committed_at
                else datetime.now().isoformat(),
                "source": "meta_learning",
                "proposal_id": proposal.get_id(),
            }

            data["axioms"].append(axiom_entry)

            atomic_write(
                constitution_path,
                yaml.dump(data, default_flow_style=False, sort_keys=False),
            )

            self._logger.info(f"Updated constitution.yaml with {proposal.axiom_name}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to update constitution: {e}")
            return False

    def _save(self) -> bool:
        """Save proposals to disk."""
        try:
            filepath = self.storage_path / "proposals.json"

            atomic_write(
                filepath,
                json.dumps(
                    {pid: p.to_dict() for pid, p in self._proposals.items()},
                    indent=2,
                ),
            )

            return True

        except Exception as e:
            self._logger.error(f"Failed to save proposals: {e}")
            return False

    def _load(self) -> bool:
        """Load proposals from disk."""
        try:
            filepath = self.storage_path / "proposals.json"

            if not filepath.exists():
                return True

            with open(filepath) as f:
                data = json.load(f)

            self._proposals = {pid: AxiomProposal.from_dict(p) for pid, p in data.items()}

            self._pending_queue = [
                pid for pid, p in self._proposals.items() if p.status == ProposalStatus.PENDING
            ]

            self._committed_axioms = {
                p.axiom_name: pid
                for pid, p in self._proposals.items()
                if p.status == ProposalStatus.COMMITTED
            }

            self._logger.info(f"Loaded {len(self._proposals)} proposals from {self.storage_path}")
            return True

        except Exception as e:
            self._logger.error(f"Failed to load proposals: {e}")
            return False


def create_axiom_bridge(
    storage_path: str | None = None,
    axiom_validator: AxiomValidator | None = None,
) -> AxiomBridge:
    """Create an AxiomBridge instance."""
    return AxiomBridge(
        storage_path=storage_path,
        axiom_validator=axiom_validator,
    )
