"""
Synthesizer - Hypothesis Building and LLM Verification
======================================================

Builds formal hypotheses from claims and verifies them
using LLM cross-validation.

Implements: docs/evolution_plan_2026/17_DEEP_RESEARCH_AGENT_SPEC.md
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .types import (
    Source,
    Claim,
    Hypothesis,
    HypothesisStatus,
    Contradiction,
    AssociativeTriple,
)
from .config import SynthesizerConfig

if TYPE_CHECKING:
    from gaap.core.base import BaseProvider

logger = logging.getLogger("gaap.research.synthesizer")


EXTRACTION_PROMPT = """Extract factual claims from the following content.

Content:
{content}

Extract up to {max_claims} factual claims. Each claim should be a single, verifiable statement.

Return a JSON array of objects with "claim" and "confidence" (0.0-1.0) fields:
[{{"claim": "...", "confidence": 0.8}}, ...]

Return ONLY the JSON array, nothing else."""


HYPOTHESIS_PROMPT = """Build a formal hypothesis from the following claim and supporting sources.

Claim: {claim}

Supporting Sources:
{sources}

Create a formal hypothesis statement that:
1. Is specific and testable
2. Captures the essence of the claim
3. Can be verified or falsified with evidence

Return JSON:
{{
  "hypothesis": "The formal hypothesis statement",
  "confidence": 0.75,
  "reasoning": "Brief explanation of why this hypothesis makes sense"
}}

Return ONLY the JSON object, nothing else."""


VERIFICATION_PROMPT = """Verify the following hypothesis against multiple sources.

Hypothesis: {hypothesis}

Sources to verify against:
{sources}

For each source, determine if it:
- SUPPORTS the hypothesis
- CONTRADICTS the hypothesis  
- is NEUTRAL (neither supports nor contradicts)

Return JSON:
{{
  "verdict": "VERIFIED" or "FALSIFIED" or "CONFLICTED" or "UNVERIFIED",
  "supporting_sources": [0, 2, ...],
  "contradicting_sources": [1, ...],
  "confidence": 0.85,
  "reasoning": "Explanation of the verdict"
}}

Source indices correspond to the order in the sources list above.
Return ONLY the JSON object, nothing else."""


CONTRADICTION_PROMPT = """Analyze these hypotheses for contradictions.

Hypotheses:
{hypotheses}

Find any pairs of hypotheses that contradict each other.
A contradiction exists when two hypotheses make mutually exclusive claims that cannot both be true.

Return JSON:
{{
  "contradictions": [
    {{
      "hypothesis1_index": 0,
      "hypothesis2_index": 2,
      "severity": "high" or "medium" or "low",
      "explanation": "Why these contradict"
    }}
  ]
}}

Return ONLY the JSON object, nothing else."""


TRIPLE_PROMPT = """Extract knowledge triples (subject-predicate-object) from the following content.

Content:
{content}

Extract facts as triples where:
- Subject: an entity (e.g., "FastAPI", "Python", "asyncio")
- Predicate: a relationship (e.g., "supports", "requires", "implements")
- Object: another entity or value

Return JSON array:
[{{"subject": "...", "predicate": "...", "object": "..."}}, ...]

Return up to {max_triples} triples.
Return ONLY the JSON array, nothing else."""


class Synthesizer:
    """
    LLM-powered hypothesis synthesis and verification.

    Features:
    - Claim extraction from sources
    - Hypothesis building
    - Cross-validation between sources
    - Contradiction detection
    - Knowledge triple extraction

    Usage:
        synthesizer = Synthesizer(llm_provider=provider)

        claims = await synthesizer.extract_claims(content, source)
        hypothesis = await synthesizer.build_hypothesis(claims[0], sources)
        verified = await synthesizer.verify_hypothesis(hypothesis, all_sources)
    """

    def __init__(
        self,
        llm_provider: BaseProvider | None = None,
        config: SynthesizerConfig | None = None,
    ) -> None:
        self._provider = llm_provider
        self.config = config or SynthesizerConfig()

        self._claims_extracted = 0
        self._hypotheses_built = 0
        self._hypotheses_verified = 0
        self._contradictions_found = 0
        self._triples_extracted = 0
        self._llm_calls = 0
        self._total_time_ms = 0.0

        self._logger = logger

    def set_provider(self, provider: BaseProvider) -> None:
        """Set LLM provider."""
        self._provider = provider

    async def extract_claims(
        self,
        content: str,
        source: Source,
        max_claims: int | None = None,
    ) -> list[Claim]:
        """
        Extract factual claims from content.

        Args:
            content: Text content
            source: Source of the content
            max_claims: Maximum claims to extract

        Returns:
            List of Claim objects
        """
        start_time = time.time()
        max_claims = max_claims or self.config.extract_claims_per_source

        if not self._provider:
            return self._extract_claims_rule_based(content, source)

        try:
            prompt = EXTRACTION_PROMPT.format(
                content=content[:3000],
                max_claims=max_claims,
            )

            response = await self._call_llm(prompt)
            self._llm_calls += 1

            claims_data = self._parse_json_response(response)

            if not isinstance(claims_data, list):
                return []

            claims: list[Claim] = []
            for item in claims_data[:max_claims]:
                if isinstance(item, dict) and "claim" in item:
                    claims.append(
                        Claim(
                            text=item["claim"],
                            source=source,
                            confidence=item.get("confidence", 0.7),
                            topic=source.metadata.get("topic", ""),
                        )
                    )

            self._claims_extracted += len(claims)
            return claims

        except Exception as e:
            self._logger.warning(f"Claim extraction failed: {e}")
            return self._extract_claims_rule_based(content, source)
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    def _extract_claims_rule_based(
        self,
        content: str,
        source: Source,
    ) -> list[Claim]:
        """Rule-based claim extraction fallback."""
        claims: list[Claim] = []

        sentences = re.split(r"[.!?]+", content)

        claim_indicators = [
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "can",
            "cannot",
            "will",
            "will not",
            "should",
            "must",
            "supports",
            "requires",
            "implements",
            "provides",
        ]

        for sentence in sentences[: self.config.extract_claims_per_source]:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 300:
                continue

            if any(f" {ind} " in sentence.lower() for ind in claim_indicators):
                claims.append(
                    Claim(
                        text=sentence,
                        source=source,
                        confidence=0.6,
                    )
                )

        self._claims_extracted += len(claims)
        return claims

    async def build_hypothesis(
        self,
        claim: Claim,
        sources: list[Source],
    ) -> Hypothesis:
        """
        Build a formal hypothesis from a claim.

        Args:
            claim: The claim to build hypothesis from
            sources: Additional supporting sources

        Returns:
            Hypothesis object
        """
        start_time = time.time()

        hypothesis_id = hashlib.md5(claim.text.encode()).hexdigest()[:12]

        if not self._provider:
            return Hypothesis(
                id=hypothesis_id,
                statement=claim.text,
                status=HypothesisStatus.UNVERIFIED,
                supporting_sources=[claim.source],
                confidence=claim.confidence,
            )

        try:
            sources_text = "\n".join(
                [f"- {s.title}: {s.content[:200] if s.content else ''}..." for s in sources[:5]]
            )

            prompt = HYPOTHESIS_PROMPT.format(
                claim=claim.text,
                sources=sources_text,
            )

            response = await self._call_llm(prompt)
            self._llm_calls += 1

            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                result = {}

            hypothesis = Hypothesis(
                id=hypothesis_id,
                statement=result.get("hypothesis", claim.text),
                status=HypothesisStatus.UNVERIFIED,
                supporting_sources=[claim.source],
                confidence=result.get("confidence", claim.confidence),
                reasoning=result.get("reasoning", ""),
            )

            self._hypotheses_built += 1
            return hypothesis

        except Exception as e:
            self._logger.warning(f"Hypothesis building failed: {e}")
            return Hypothesis(
                id=hypothesis_id,
                statement=claim.text,
                status=HypothesisStatus.UNVERIFIED,
                supporting_sources=[claim.source],
                confidence=claim.confidence,
            )
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    async def verify_hypothesis(
        self,
        hypothesis: Hypothesis,
        sources: list[Source],
        cross_validate: bool | None = None,
    ) -> Hypothesis:
        """
        Verify a hypothesis against multiple sources.

        Args:
            hypothesis: Hypothesis to verify
            sources: Sources to verify against
            cross_validate: Whether to cross-validate

        Returns:
            Updated Hypothesis with verification status
        """
        start_time = time.time()
        cross_validate = (
            cross_validate if cross_validate is not None else self.config.cross_validate_enabled
        )

        if not self._provider or not cross_validate:
            return hypothesis

        if len(sources) < 2:
            hypothesis.status = HypothesisStatus.UNVERIFIED
            return hypothesis

        try:
            sources_text = "\n".join(
                [
                    f"[{i}] {s.title}: {s.content[:300] if s.content else s.metadata.get('snippet', '')}..."
                    for i, s in enumerate(sources[:10])
                ]
            )

            prompt = VERIFICATION_PROMPT.format(
                hypothesis=hypothesis.statement,
                sources=sources_text,
            )

            response = await self._call_llm(prompt)
            self._llm_calls += 1

            result = self._parse_json_response(response)

            if not isinstance(result, dict):
                return hypothesis

            verdict = result.get("verdict", "UNVERIFIED")

            if verdict == "VERIFIED":
                hypothesis.status = HypothesisStatus.VERIFIED
            elif verdict == "FALSIFIED":
                hypothesis.status = HypothesisStatus.FALSIFIED
            elif verdict == "CONFLICTED":
                hypothesis.status = HypothesisStatus.CONFLICTED
            else:
                hypothesis.status = HypothesisStatus.UNVERIFIED

            supporting_indices = result.get("supporting_sources", [])
            contradicting_indices = result.get("contradicting_sources", [])

            hypothesis.supporting_sources = [
                sources[i] for i in supporting_indices if i < len(sources)
            ]
            hypothesis.contradicting_sources = [
                sources[i] for i in contradicting_indices if i < len(sources)
            ]
            hypothesis.confidence = result.get("confidence", hypothesis.confidence)
            hypothesis.reasoning = result.get("reasoning", "")
            hypothesis.verification_timestamp = datetime.now()

            self._hypotheses_verified += 1
            return hypothesis

        except Exception as e:
            self._logger.warning(f"Hypothesis verification failed: {e}")
            return hypothesis
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    async def find_contradictions(
        self,
        hypotheses: list[Hypothesis],
    ) -> list[Contradiction]:
        """
        Find contradictions between hypotheses.

        Args:
            hypotheses: List of hypotheses to analyze

        Returns:
            List of Contradiction objects
        """
        start_time = time.time()

        if not self._provider or not self.config.detect_contradictions:
            return []

        if len(hypotheses) < 2:
            return []

        try:
            hypotheses_text = "\n".join([f"[{i}] {h.statement}" for i, h in enumerate(hypotheses)])

            prompt = CONTRADICTION_PROMPT.format(hypotheses=hypotheses_text)

            response = await self._call_llm(prompt)
            self._llm_calls += 1

            result = self._parse_json_response(response)

            if not isinstance(result, dict) or "contradictions" not in result:
                return []

            contradictions: list[Contradiction] = []

            for item in result.get("contradictions", []):
                i1 = item.get("hypothesis1_index", -1)
                i2 = item.get("hypothesis2_index", -1)

                if i1 < 0 or i2 < 0 or i1 >= len(hypotheses) or i2 >= len(hypotheses):
                    continue

                h1 = hypotheses[i1]
                h2 = hypotheses[i2]

                if h1.supporting_sources and h2.supporting_sources:
                    contradictions.append(
                        Contradiction(
                            claim1=h1.statement,
                            claim2=h2.statement,
                            source1=h1.supporting_sources[0],
                            source2=h2.supporting_sources[0],
                            severity=item.get("severity", "medium"),
                            resolution=item.get("explanation", ""),
                        )
                    )

            self._contradictions_found += len(contradictions)
            return contradictions

        except Exception as e:
            self._logger.warning(f"Contradiction detection failed: {e}")
            return []
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    async def extract_triples(
        self,
        content: str,
        source: Source,
        max_triples: int | None = None,
    ) -> list[AssociativeTriple]:
        """
        Extract knowledge triples from content.

        Args:
            content: Text content
            source: Source of content
            max_triples: Maximum triples to extract

        Returns:
            List of AssociativeTriple objects
        """
        start_time = time.time()
        max_triples = max_triples or self.config.max_triples_per_hypothesis

        if not self._provider:
            return self._extract_triples_rule_based(content, source, max_triples)

        try:
            prompt = TRIPLE_PROMPT.format(
                content=content[:2000],
                max_triples=max_triples,
            )

            response = await self._call_llm(prompt)
            self._llm_calls += 1

            triples_data = self._parse_json_response(response)

            if not isinstance(triples_data, list):
                return []

            triples: list[AssociativeTriple] = []
            for item in triples_data[:max_triples]:
                if isinstance(item, dict) and all(
                    k in item for k in ["subject", "predicate", "object"]
                ):
                    triples.append(
                        AssociativeTriple(
                            subject=item["subject"],
                            predicate=item["predicate"],
                            object=item["object"],
                            source=source,
                            confidence=0.8,
                        )
                    )

            self._triples_extracted += len(triples)
            return triples

        except Exception as e:
            self._logger.warning(f"Triple extraction failed: {e}")
            return self._extract_triples_rule_based(content, source, max_triples)
        finally:
            self._total_time_ms += (time.time() - start_time) * 1000

    def _extract_triples_rule_based(
        self,
        content: str,
        source: Source,
        max_triples: int,
    ) -> list[AssociativeTriple]:
        """Rule-based triple extraction fallback."""
        triples: list[AssociativeTriple] = []

        patterns = [
            (r"(\w+)\s+supports\s+(\w+)", "supports"),
            (r"(\w+)\s+requires\s+(\w+)", "requires"),
            (r"(\w+)\s+implements\s+(\w+)", "implements"),
            (r"(\w+)\s+is\s+(?:a\s+)?(\w+)", "is_a"),
            (r"(\w+)\s+uses\s+(\w+)", "uses"),
            (r"(\w+)\s+provides\s+(\w+)", "provides"),
        ]

        for pattern, predicate in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                if len(triples) >= max_triples:
                    break

                subject = match.group(1).strip()
                obj = match.group(2).strip()

                if len(subject) > 2 and len(obj) > 2:
                    triples.append(
                        AssociativeTriple(
                            subject=subject,
                            predicate=predicate,
                            object=obj,
                            source=source,
                            confidence=0.6,
                        )
                    )

        self._triples_extracted += len(triples)
        return triples

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM provider."""
        if not self._provider:
            raise ValueError("No LLM provider configured")

        try:
            from gaap.core.types import Message, MessageRole

            messages = [Message(role=MessageRole.USER, content=prompt)]
            model = getattr(self._provider, "default_model", "gpt-4o-mini")

            if hasattr(self._provider, "chat_completion"):
                response = await self._provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=self.config.llm_temperature,
                )
                return response.choices[0].message.content
            else:
                raise ValueError("Provider has no compatible completion method")
        except Exception as e:
            self._logger.error(f"LLM call failed: {e}")
            raise

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response."""
        response = response.strip()

        if response.startswith("```"):
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r"[\[{].*[\]}]", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
        return {}

    def get_stats(self) -> dict[str, Any]:
        """Get synthesizer statistics."""
        return {
            "claims_extracted": self._claims_extracted,
            "hypotheses_built": self._hypotheses_built,
            "hypotheses_verified": self._hypotheses_verified,
            "contradictions_found": self._contradictions_found,
            "triples_extracted": self._triples_extracted,
            "llm_calls": self._llm_calls,
            "total_time_ms": f"{self._total_time_ms:.1f}",
            "avg_time_per_call_ms": f"{self._total_time_ms / max(1, self._llm_calls):.1f}",
        }


def create_synthesizer(
    llm_provider: BaseProvider | None = None,
    max_hypotheses: int = 10,
    cross_validate: bool = True,
) -> Synthesizer:
    """Create a Synthesizer with specified settings."""
    config = SynthesizerConfig(
        max_hypotheses=max_hypotheses,
        cross_validate_enabled=cross_validate,
    )
    return Synthesizer(llm_provider=llm_provider, config=config)
