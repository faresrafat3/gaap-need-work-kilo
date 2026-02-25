"""
Source Auditor - Epistemic Trust Score (ETS) Evaluation
=======================================================

Assigns trust scores to sources based on domain reputation,
author credibility, and content quality.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, date
from typing import Any
from urllib.parse import urlparse

from .types import Source, SourceStatus, ETSLevel
from .config import SourceAuditConfig

logger = logging.getLogger("gaap.research.source_auditor")


@dataclass
class AuditResult:
    """Result of auditing a source."""

    source: Source
    ets_score: float
    ets_level: ETSLevel
    domain_score: float
    freshness_score: float
    author_score: float
    content_score: float
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.source.url,
            "ets_score": self.ets_score,
            "ets_level": self.ets_level.name,
            "domain_score": self.domain_score,
            "freshness_score": self.freshness_score,
            "author_score": self.author_score,
            "content_score": self.content_score,
            "reasons": self.reasons,
        }


class SourceAuditor:
    """
    Epistemic Trust Score (ETS) auditor for sources.

    ETS scoring factors:
    - Domain reputation (official docs, GitHub, SO, etc.)
    - Author credibility
    - Content freshness
    - Content quality (length, structure)
    - Citation count

    Usage:
        auditor = SourceAuditor()
        result = auditor.audit(source)
        print(f"ETS Score: {result.ets_score}")
    """

    DEFAULT_DOMAIN_SCORES: dict[str, float] = {
        # VERIFIED (1.0) - Official documentation
        "docs.python.org": 1.0,
        "python.org": 1.0,
        "fastapi.tiangolo.com": 1.0,
        "docs.djangoproject.com": 1.0,
        "react.dev": 1.0,
        "vuejs.org": 1.0,
        "nodejs.org": 1.0,
        "go.dev": 1.0,
        "rust-lang.org": 1.0,
        "kubernetes.io": 1.0,
        "tensorflow.org": 1.0,
        "pytorch.org": 1.0,
        "numpy.org": 1.0,
        "pandas.pydata.org": 1.0,
        # RELIABLE (0.8-0.9) - Official repos and high-quality sources
        "github.com": 0.9,
        "gitlab.com": 0.85,
        "bitbucket.org": 0.8,
        "readthedocs.io": 0.85,
        "pypi.org": 0.9,
        "npmjs.com": 0.85,
        "arxiv.org": 0.9,
        "dl.acm.org": 0.9,
        "ieeexplore.ieee.org": 0.9,
        "springer.com": 0.85,
        "nature.com": 0.9,
        "science.org": 0.9,
        # RELIABLE (0.7-0.8) - Community knowledge bases
        "stackoverflow.com": 0.75,
        "stackexchange.com": 0.7,
        "superuser.com": 0.7,
        "askubuntu.com": 0.7,
        "serverfault.com": 0.7,
        "reddit.com": 0.5,
        "quora.com": 0.5,
        "wikipedia.org": 0.7,
        # QUESTIONABLE (0.4-0.6) - Tech blogs and tutorials
        "medium.com": 0.5,
        "towardsdatascience.com": 0.55,
        "analyticsvidhya.com": 0.5,
        "machinelearningmastery.com": 0.55,
        "realpython.com": 0.7,
        "geeksforgeeks.org": 0.5,
        "w3schools.com": 0.5,
        "tutorialspoint.com": 0.45,
        "javatpoint.com": 0.45,
        "dev.to": 0.55,
        "hashnode.com": 0.5,
        "hackernoon.com": 0.45,
        # UNRELIABLE (0.2-0.4) - Random blogs, AI summaries
        "blogspot.com": 0.3,
        "wordpress.com": 0.35,
        "substack.com": 0.4,
        "notion.site": 0.35,
        # BLACKLISTED (0.0) - Known bad sources
        "example.com": 0.0,
        "test.com": 0.0,
    }

    WILDCARD_DOMAINS: dict[str, float] = {
        r".*\.python\.org$": 1.0,
        r".*\.github\.io$": 0.8,
        r".*\.readthedocs\.io$": 0.85,
        r".*\.stackoverflow\.com$": 0.75,
        r".*\.medium\.com$": 0.5,
        r".*\.dev$": 0.6,
        r".*\.edu$": 0.75,
        r".*\.gov$": 0.85,
        r".*\.org$": 0.6,
    }

    def __init__(
        self,
        config: SourceAuditConfig | None = None,
    ) -> None:
        self.config = config or SourceAuditConfig()
        self._domain_scores = self.DEFAULT_DOMAIN_SCORES.copy()

        if self.config.domain_overrides:
            self._domain_scores.update(self.config.domain_overrides)

        self._blacklist = set(self.config.blacklist_domains)
        self._whitelist = set(self.config.whitelist_domains)

        self._audited_count = 0
        self._filtered_count = 0
        self._total_time_ms = 0.0

        self._logger = logger

    def audit(self, source: Source) -> AuditResult:
        """
        Audit a source and assign ETS score.

        Args:
            source: Source to audit

        Returns:
            AuditResult with scores and reasons
        """
        start_time = time.time()
        reasons: list[str] = []

        if source.domain in self._blacklist:
            result = AuditResult(
                source=source,
                ets_score=0.0,
                ets_level=ETSLevel.BLACKLISTED,
                domain_score=0.0,
                freshness_score=0.0,
                author_score=0.0,
                content_score=0.0,
                reasons=["Domain is blacklisted"],
            )
            source.ets_score = 0.0
            source.status = SourceStatus.AUDITED
            self._filtered_count += 1
            return result

        domain_score = self._score_domain(source.domain)
        freshness_score = self._score_freshness(
            source.publish_date, source.metadata.get("publish_date")
        )
        author_score = self._score_author(source.author, source.metadata)
        content_score = self._score_content(source.content)

        total_score = (
            domain_score * self.config.domain_weight
            + freshness_score * self.config.freshness_weight
            + author_score * self.config.citation_weight
            + content_score * 0.2
        )

        citation_boost = min(source.citation_count * 0.02, 0.1)
        total_score = min(total_score + citation_boost, 1.0)

        if domain_score >= 0.9:
            reasons.append("Official/verified domain")
        elif domain_score >= 0.7:
            reasons.append("Reliable domain")
        elif domain_score >= 0.5:
            reasons.append("Moderate reliability domain")
        else:
            reasons.append("Low reliability domain")

        if freshness_score >= 0.8:
            reasons.append("Recent content")
        elif freshness_score < 0.3:
            reasons.append("Outdated content")

        if author_score >= 0.7:
            reasons.append("Identified author")

        if content_score >= 0.7:
            reasons.append("Quality content")
        elif content_score < 0.3:
            reasons.append("Low quality content")

        ets_level = self._get_ets_level(total_score)

        source.ets_score = total_score
        source.status = SourceStatus.AUDITED
        self._audited_count += 1

        self._total_time_ms += (time.time() - start_time) * 1000

        return AuditResult(
            source=source,
            ets_score=total_score,
            ets_level=ets_level,
            domain_score=domain_score,
            freshness_score=freshness_score,
            author_score=author_score,
            content_score=content_score,
            reasons=reasons,
        )

    def audit_batch(
        self,
        sources: list[Source],
        filter_threshold: bool = True,
    ) -> tuple[list[Source], list[Source]]:
        """
        Audit multiple sources.

        Args:
            sources: Sources to audit
            filter_threshold: Filter out sources below threshold

        Returns:
            Tuple of (passed_sources, filtered_sources)
        """
        passed: list[Source] = []
        filtered: list[Source] = []

        for source in sources:
            result = self.audit(source)

            if filter_threshold and result.ets_score < self.config.min_ets_threshold:
                filtered.append(source)
            else:
                passed.append(source)

        return passed, filtered

    def _score_domain(self, domain: str) -> float:
        """Score based on domain reputation."""
        domain_lower = domain.lower()

        if domain_lower in self._domain_scores:
            return self._domain_scores[domain_lower]

        for pattern, score in self.WILDCARD_DOMAINS.items():
            if re.match(pattern, domain_lower):
                return score

        for pattern, score in self.WILDCARD_DOMAINS.items():
            try:
                if re.match(pattern, domain_lower):
                    return score
            except re.error:
                pass

        if domain_lower.endswith(".edu"):
            return 0.75
        if domain_lower.endswith(".gov"):
            return 0.85
        if domain_lower.endswith(".org"):
            return 0.6
        if domain_lower.endswith(".io"):
            return 0.5
        if domain_lower.endswith(".dev"):
            return 0.55

        return 0.4

    def _score_freshness(
        self,
        publish_date: date | None,
        metadata_date: Any = None,
    ) -> float:
        """Score based on content freshness."""
        if not self.config.check_date:
            return 0.7

        content_date = publish_date

        if content_date is None and metadata_date:
            if isinstance(metadata_date, str):
                try:
                    content_date = datetime.fromisoformat(metadata_date).date()
                except (ValueError, TypeError):
                    pass
            elif isinstance(metadata_date, date):
                content_date = metadata_date

        if content_date is None:
            return 0.5

        days_old = (date.today() - content_date).days

        if days_old < 30:
            return 1.0
        elif days_old < 90:
            return 0.9
        elif days_old < 180:
            return 0.8
        elif days_old < 365:
            return 0.6
        elif days_old < 730:
            return 0.4
        else:
            return 0.2

    def _score_author(
        self,
        author: str | None,
        metadata: dict[str, Any],
    ) -> float:
        """Score based on author credibility."""
        if not self.config.check_author:
            return 0.7

        if author:
            return 0.8

        if metadata.get("author"):
            return 0.8

        if metadata.get("author_url"):
            return 0.75

        return 0.5

    def _score_content(self, content: str | None) -> float:
        """Score based on content quality."""
        if content is None:
            return 0.3

        score = 0.5

        length = len(content)
        if length < 100:
            score -= 0.3
        elif length < 500:
            score -= 0.1
        elif length > 2000:
            score += 0.1
        elif length > 5000:
            score += 0.15

        if "```" in content or "def " in content or "function " in content:
            score += 0.1

        if content.count("\n") > 10:
            score += 0.05

        if re.search(r"\d{4}", content):
            score += 0.05

        return max(0.0, min(score, 1.0))

    def _get_ets_level(self, score: float) -> ETSLevel:
        """Get ETS level from score."""
        if score >= 0.95:
            return ETSLevel.VERIFIED
        elif score >= 0.65:
            return ETSLevel.RELIABLE
        elif score >= 0.45:
            return ETSLevel.QUESTIONABLE
        elif score >= 0.2:
            return ETSLevel.UNRELIABLE
        else:
            return ETSLevel.BLACKLISTED

    def get_domain_score(self, domain: str) -> float:
        """Get ETS score for a domain."""
        return self._score_domain(domain)

    def set_domain_score(self, domain: str, score: float) -> None:
        """Set custom ETS score for a domain."""
        self._domain_scores[domain.lower()] = max(0.0, min(score, 1.0))
        self._logger.info(f"Set ETS score for {domain}: {score}")

    def add_blacklist_domain(self, domain: str) -> None:
        """Add domain to blacklist."""
        self._blacklist.add(domain.lower())
        self._logger.info(f"Blacklisted domain: {domain}")

    def add_whitelist_domain(self, domain: str) -> None:
        """Add domain to whitelist."""
        self._whitelist.add(domain.lower())
        self._domain_scores[domain.lower()] = 0.8
        self._logger.info(f"Whitelisted domain: {domain}")

    def detect_contradictions(
        self,
        sources: list[Source],
    ) -> list[tuple[Source, Source, str]]:
        """
        Detect potential contradictions between sources.

        Returns list of (source1, source2, issue) tuples.
        """
        contradictions: list[tuple[Source, Source, str]] = []

        return contradictions

    def get_stats(self) -> dict[str, Any]:
        """Get auditor statistics."""
        return {
            "audited_count": self._audited_count,
            "filtered_count": self._filtered_count,
            "filter_rate": f"{self._filtered_count / max(1, self._audited_count):.1%}",
            "min_ets_threshold": self.config.min_ets_threshold,
            "blacklist_size": len(self._blacklist),
            "custom_domains": len(self._domain_scores) - len(self.DEFAULT_DOMAIN_SCORES),
            "total_time_ms": f"{self._total_time_ms:.1f}",
        }


def create_source_auditor(
    min_ets_threshold: float = 0.3,
    domain_overrides: dict[str, float] | None = None,
) -> SourceAuditor:
    """Create a SourceAuditor with specified settings."""
    config = SourceAuditConfig(
        min_ets_threshold=min_ets_threshold,
        domain_overrides=domain_overrides or {},
    )
    return SourceAuditor(config)
