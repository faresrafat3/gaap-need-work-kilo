"""
Reputation System for Swarm Intelligence

Implements a dynamic, domain-aware reputation tracking system that:
- Tracks success/failure rates per domain (not just overall)
- Applies time-decay to prevent stale reputations
- Rewards epistemic humility (predicting own failures)
- Penalizes silent failures more than predicted ones

The reputation score uses a Bayesian-inspired formula that:
- Starts with a prior (baseline reputation)
- Updates with evidence (successes/failures)
- Weighs recent performance more heavily
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any


class ReputationEvent(Enum):
    """أنواع أحداث السمعة"""

    SUCCESS = auto()
    FAILURE = auto()
    PREDICTED_FAILURE = auto()  # Fractal correctly predicted it would fail
    TIMEOUT = auto()
    QUALITY_ISSUE = auto()


@dataclass
class DomainExpertise:
    """
    خبرة Fractal في مجال معين.

    Uses exponential moving average for score calculation:
    score = α * new_evidence + (1-α) * old_score

    Where α is the learning rate (0.1 by default).
    """

    domain: str
    successes: int = 0
    failures: int = 0
    predicted_failures: int = 0  # Correctly predicted failures
    total_tasks: int = 0
    score: float = 0.5  # Start neutral
    confidence: float = 0.0  # How confident are we in this score?
    last_updated: datetime = field(default_factory=datetime.now)

    # Learning parameters
    LEARNING_RATE: float = 0.15
    CONFIDENCE_INCREMENT: float = 0.05
    MAX_CONFIDENCE: float = 1.0

    def record_success(self) -> None:
        """تسجيل نجاح"""
        self.successes += 1
        self.total_tasks += 1

        # Bayesian update: increase score
        old_score = self.score
        evidence = 1.0  # Success = 1.0

        self.score = self.LEARNING_RATE * evidence + (1 - self.LEARNING_RATE) * old_score

        # Increase confidence
        self.confidence = min(self.MAX_CONFIDENCE, self.confidence + self.CONFIDENCE_INCREMENT)
        self.last_updated = datetime.now()

    def record_failure(self, predicted: bool = False) -> None:
        """
        تسجيل فشل.

        Args:
            predicted: If True, the fractal correctly predicted this failure
                      (epistemic humility - reduced penalty)
        """
        if predicted:
            self.predicted_failures += 1
            # Reduced penalty for honest prediction
            penalty = 0.05
        else:
            self.failures += 1
            # Full penalty for unexpected failure
            penalty = 0.15

        self.total_tasks += 1

        # Bayesian update: decrease score
        old_score = self.score
        evidence = 0.0  # Failure = 0.0

        self.score = (1 - penalty) * old_score + penalty * evidence

        # Still increase confidence (we learned something)
        self.confidence = min(
            self.MAX_CONFIDENCE, self.confidence + self.CONFIDENCE_INCREMENT * 0.5
        )
        self.last_updated = datetime.now()

    def get_adjusted_score(self) -> float:
        """
        الحصول على درجة معدلة حسب الثقة.

        Low confidence = score pulled toward 0.5 (neutral)
        High confidence = score as-is
        """
        # Pull toward neutral based on lack of confidence
        neutrality_pull = 1 - self.confidence
        return self.score * (1 - neutrality_pull) + 0.5 * neutrality_pull

    def decay(self, decay_factor: float = 0.95) -> None:
        """
        Apply time decay to reputation.

        Args:
            decay_factor: Multiplier for scores (0.95 = 5% decay)
        """
        days_since_update = (datetime.now() - self.last_updated).days

        if days_since_update > 0:
            # Exponential decay based on days
            decay_amount = decay_factor**days_since_update
            self.score = 0.5 + (self.score - 0.5) * decay_amount

            # Also decay confidence
            self.confidence *= decay_amount

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "successes": self.successes,
            "failures": self.failures,
            "predicted_failures": self.predicted_failures,
            "total_tasks": self.total_tasks,
            "score": round(self.score, 4),
            "adjusted_score": round(self.get_adjusted_score(), 4),
            "confidence": round(self.confidence, 4),
            "last_updated": self.last_updated.isoformat(),
        }


@dataclass
class ReputationScore:
    """
    ملخص سمعة Fractal.
    """

    fractal_id: str
    overall_score: float
    domain_scores: dict[str, float]
    total_tasks: int
    reliability: float  # How consistent is the fractal?
    availability: float  # What fraction of auctions did they bid on?

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractal_id": self.fractal_id,
            "overall_score": round(self.overall_score, 4),
            "domain_scores": {k: round(v, 4) for k, v in self.domain_scores.items()},
            "total_tasks": self.total_tasks,
            "reliability": round(self.reliability, 4),
            "availability": round(self.availability, 4),
        }


@dataclass
class ReputationEntry:
    """
    سجل سمعة Fractal كامل.
    """

    fractal_id: str
    domains: dict[str, DomainExpertise] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)

    # Metadata
    specializations: list[str] = field(default_factory=list)
    preferred_domains: list[str] = field(default_factory=list)

    def get_or_create_domain(self, domain: str) -> DomainExpertise:
        """الحصول على أو إنشاء سجل مجال"""
        if domain not in self.domains:
            self.domains[domain] = DomainExpertise(domain=domain)
        return self.domains[domain]

    def get_overall_score(self) -> float:
        """
        Calculate overall score weighted by domain usage.

        More used domains have higher weight.
        """
        if not self.domains:
            return 0.5

        total_weight = 0.0
        weighted_sum = 0.0

        for domain_exp in self.domains.values():
            weight = domain_exp.total_tasks + 1  # +1 to avoid zero
            weighted_sum += domain_exp.get_adjusted_score() * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def get_reliability(self) -> float:
        """
        Calculate reliability (consistency).

        High reliability = consistent performance
        Low reliability = volatile performance
        """
        if not self.domains:
            return 0.5

        # Calculate variance in scores
        scores = [d.get_adjusted_score() for d in self.domains.values() if d.total_tasks > 0]
        if not scores:
            return 0.5

        mean = sum(scores) / len(scores)
        if len(scores) < 2:
            return 0.5

        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)

        # Convert to reliability (low std_dev = high reliability)
        # std_dev of 0 = reliability 1.0
        # std_dev of 0.5 = reliability 0.0
        reliability = max(0.0, 1.0 - std_dev * 2)
        return reliability

    def to_dict(self) -> dict[str, Any]:
        return {
            "fractal_id": self.fractal_id,
            "domains": {k: v.to_dict() for k, v in self.domains.items()},
            "overall_score": round(self.get_overall_score(), 4),
            "reliability": round(self.get_reliability(), 4),
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "specializations": self.specializations,
            "preferred_domains": self.preferred_domains,
        }


class ReputationStore:
    """
    مخزن السمعة المركزي للـ Swarm.

    Features:
    - Domain-specific reputation tracking
    - Time-based decay
    - Persistence to disk
    - Epistemic humility rewards

    Usage:
        store = ReputationStore()

        # Record events
        store.record_success("coder_01", "python")
        store.record_failure("coder_01", "sql", predicted=False)

        # Query reputation
        score = store.get_domain_reputation("coder_01", "python")  # 0.94
        top_coders = store.get_top_fractals("python", limit=3)
    """

    DEFAULT_STORAGE_PATH = ".gaap/reputation/reputation.json"
    DECAY_INTERVAL_DAYS = 7
    DECAY_FACTOR = 0.95

    def __init__(self, storage_path: str | None = None) -> None:
        self._storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self._entries: dict[str, ReputationEntry] = {}
        self._logger = logging.getLogger("gaap.swarm.reputation")

        # Auction participation tracking (for availability metric)
        self._auctions_participated: dict[str, int] = {}
        self._auctions_total: dict[str, int] = {}

        self._load()

    def _load(self) -> None:
        """تحميل السمعة من القرص"""
        if not self._storage_path.exists():
            self._logger.info("No existing reputation store found, starting fresh")
            return

        try:
            with open(self._storage_path) as f:
                data = json.load(f)

            for fractal_id, entry_data in data.get("entries", {}).items():
                entry = ReputationEntry(
                    fractal_id=fractal_id,
                    specializations=entry_data.get("specializations", []),
                    preferred_domains=entry_data.get("preferred_domains", []),
                )

                for domain, domain_data in entry_data.get("domains", {}).items():
                    expertise = DomainExpertise(
                        domain=domain,
                        successes=domain_data.get("successes", 0),
                        failures=domain_data.get("failures", 0),
                        predicted_failures=domain_data.get("predicted_failures", 0),
                        total_tasks=domain_data.get("total_tasks", 0),
                        score=domain_data.get("score", 0.5),
                        confidence=domain_data.get("confidence", 0.0),
                    )
                    entry.domains[domain] = expertise

                self._entries[fractal_id] = entry

            self._logger.info(f"Loaded reputation for {len(self._entries)} fractals")

        except Exception as e:
            self._logger.error(f"Failed to load reputation store: {e}")

    def save(self) -> None:
        """حفظ السمعة للقرص"""
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "2.0",
            "updated_at": datetime.now().isoformat(),
            "entries": {fid: entry.to_dict() for fid, entry in self._entries.items()},
            "auction_stats": {
                "participated": self._auctions_participated,
                "total": self._auctions_total,
            },
        }

        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self._logger.debug(f"Saved reputation for {len(self._entries)} fractals")

    def get_or_create_entry(self, fractal_id: str) -> ReputationEntry:
        """الحصول على أو إنشاء سجل Fractal"""
        if fractal_id not in self._entries:
            self._entries[fractal_id] = ReputationEntry(fractal_id=fractal_id)
        return self._entries[fractal_id]

    def record_success(self, fractal_id: str, domain: str) -> None:
        """
        تسجيل نجاح.

        Args:
            fractal_id: معرف Fractal
            domain: مجال المهمة (python, sql, security, etc.)
        """
        entry = self.get_or_create_entry(fractal_id)
        domain_exp = entry.get_or_create_domain(domain)
        domain_exp.record_success()
        entry.last_active = datetime.now()

        self._logger.info(f"Recorded success for {fractal_id} in {domain}")
        self.save()

    def record_failure(
        self,
        fractal_id: str,
        domain: str,
        predicted: bool = False,
    ) -> None:
        """
        تسجيل فشل.

        Args:
            fractal_id: معرف Fractal
            domain: مجال المهمة
            predicted: هل تنبأ Fractal بالفشل مسبقاً؟ (يقلل العقوبة)
        """
        entry = self.get_or_create_entry(fractal_id)
        domain_exp = entry.get_or_create_domain(domain)
        domain_exp.record_failure(predicted=predicted)
        entry.last_active = datetime.now()

        if predicted:
            self._logger.info(
                f"Recorded predicted failure for {fractal_id} in {domain} (epistemic humility)"
            )
        else:
            self._logger.warning(f"Recorded unexpected failure for {fractal_id} in {domain}")

        self.save()

    def record_auction_participation(
        self,
        fractal_id: str,
        participated: bool,
    ) -> None:
        """
        تسجيل مشاركة في المزاد (لحساب التوفر).
        """
        if fractal_id not in self._auctions_total:
            self._auctions_total[fractal_id] = 0
            self._auctions_participated[fractal_id] = 0

        self._auctions_total[fractal_id] += 1
        if participated:
            self._auctions_participated[fractal_id] += 1

    def get_domain_reputation(self, fractal_id: str, domain: str) -> float:
        """
        الحصول على سمعة Fractal في مجال معين.

        Returns:
            float: 0.0 to 1.0 (0.5 = neutral)
        """
        entry = self._entries.get(fractal_id)
        if not entry:
            return 0.5  # Unknown = neutral

        domain_exp = entry.domains.get(domain)
        if not domain_exp:
            return 0.5  # No experience = neutral

        return domain_exp.get_adjusted_score()

    def get_overall_reputation(self, fractal_id: str) -> float:
        """
        الحصول على السمعة العامة لـ Fractal.
        """
        entry = self._entries.get(fractal_id)
        if not entry:
            return 0.5

        return entry.get_overall_score()

    def get_reputation_score(self, fractal_id: str) -> ReputationScore:
        """
        الحصول على ملخص سمعة كامل.
        """
        entry = self._entries.get(fractal_id)

        if not entry:
            return ReputationScore(
                fractal_id=fractal_id,
                overall_score=0.5,
                domain_scores={},
                total_tasks=0,
                reliability=0.5,
                availability=0.5,
            )

        domain_scores = {domain: exp.get_adjusted_score() for domain, exp in entry.domains.items()}

        total_tasks = sum(d.total_tasks for d in entry.domains.values())

        # Calculate availability
        total_auctions = self._auctions_total.get(fractal_id, 0)
        participated = self._auctions_participated.get(fractal_id, 0)
        availability = participated / total_auctions if total_auctions > 0 else 0.5

        return ReputationScore(
            fractal_id=fractal_id,
            overall_score=entry.get_overall_score(),
            domain_scores=domain_scores,
            total_tasks=total_tasks,
            reliability=entry.get_reliability(),
            availability=availability,
        )

    def get_top_fractals(
        self,
        domain: str,
        limit: int = 5,
        min_confidence: float = 0.3,
    ) -> list[tuple[str, float]]:
        """
        الحصول على أفضل Fractals في مجال معين.

        Args:
            domain: المجال المطلوب
            limit: الحد الأقصى للنتائج
            min_confidence: الحد الأدنى للثقة

        Returns:
            List of (fractal_id, score) tuples
        """
        candidates = []

        for fractal_id, entry in self._entries.items():
            domain_exp = entry.domains.get(domain)
            if domain_exp and domain_exp.confidence >= min_confidence:
                score = domain_exp.get_adjusted_score()
                candidates.append((fractal_id, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        return candidates[:limit]

    def apply_decay(self, days: int = 7) -> int:
        """
        تطبيق انخفاض السمعة مع الوقت.

        Args:
            days: تطبيق الانخفاض للإدخالات غير المحدثة منذ هذا العدد من الأيام

        Returns:
            Number of entries decayed
        """
        decayed = 0
        threshold = datetime.now() - timedelta(days=days)

        for entry in self._entries.values():
            if entry.last_active < threshold:
                for domain_exp in entry.domains.values():
                    domain_exp.decay(self.DECAY_FACTOR)
                decayed += 1

        if decayed > 0:
            self._logger.info(f"Applied time decay to {decayed} fractal reputations")
            self.save()

        return decayed

    def get_fractal_specializations(self, fractal_id: str) -> list[str]:
        """
        الحصول على تخصصات Fractal بناءً على السمعة.

        Returns domains where score > 0.7
        """
        entry = self._entries.get(fractal_id)
        if not entry:
            return []

        specializations = []
        for domain, exp in entry.domains.items():
            if exp.get_adjusted_score() > 0.7 and exp.confidence > 0.5:
                specializations.append(domain)

        return specializations

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات المخزن"""
        total_fractals = len(self._entries)
        total_domains = set()

        for entry in self._entries.values():
            total_domains.update(entry.domains.keys())

        return {
            "total_fractals": total_fractals,
            "total_domains": len(total_domains),
            "domains": list(total_domains),
            "storage_path": str(self._storage_path),
        }
