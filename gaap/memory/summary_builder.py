"""
Summary Builder for RAPTOR

Builds summaries using LLM for hierarchical document organization.
Provides text summarization, key concept extraction, and importance estimation.

Usage:
    from gaap.memory.summary_builder import SummaryBuilder

    builder = SummaryBuilder(llm_provider=my_llm)
    summary = await builder.summarize_texts(["doc1", "doc2"])
    concepts = builder.extract_key_concepts(text)
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("gaap.memory.summary_builder")


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers."""

    async def generate(self, prompt: str, **kwargs: Any) -> str: ...


class SimpleLLMProvider:
    """Simple LLM provider for testing without external dependencies."""

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        lines = prompt.split("\n")
        meaningful_lines = [l for l in lines if len(l.strip()) > 20]
        if meaningful_lines:
            combined = " ".join(meaningful_lines[:3])
            if len(combined) > 200:
                return combined[:200] + "..."
            return combined
        return "Summary not available."


@dataclass
class KeyConcept:
    """Extracted key concept."""

    name: str
    description: str
    relevance: float = 0.5
    category: str = "general"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "relevance": self.relevance,
            "category": self.category,
        }


@dataclass
class SummaryResult:
    """Result from summarization."""

    summary: str
    key_concepts: list[KeyConcept] = field(default_factory=list)
    importance_score: float = 0.5
    word_count: int = 0
    source_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "key_concepts": [c.to_dict() for c in self.key_concepts],
            "importance_score": self.importance_score,
            "word_count": self.word_count,
            "source_count": self.source_count,
        }


class SummaryBuilder:
    """
    Build summaries using LLM for RAPTOR tree.

    Provides:
    - Text summarization with configurable compression
    - Key concept extraction
    - Importance estimation
    - Project-level summary generation

    Attributes:
        llm: LLM provider for generation
        max_summary_length: Maximum summary length in characters
        compression_ratio: Target compression ratio

    Usage:
        builder = SummaryBuilder(llm_provider=my_llm)
        result = await builder.summarize_texts(["text1", "text2"])
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        max_summary_length: int = 500,
        compression_ratio: float = 0.3,
    ) -> None:
        self.llm = llm or SimpleLLMProvider()
        self.max_summary_length = max_summary_length
        self.compression_ratio = compression_ratio
        self._logger = logger

    async def summarize_texts(
        self,
        texts: list[str],
        style: str = "concise",
    ) -> SummaryResult:
        """
        Summarize multiple texts into one summary.

        Args:
            texts: List of text strings to summarize
            style: Summary style ("concise", "detailed", "bullet")

        Returns:
            SummaryResult with summary and metadata

        Example:
            result = await builder.summarize_texts(
                ["First document...", "Second document..."],
                style="concise"
            )
        """
        if not texts:
            return SummaryResult(summary="", source_count=0)

        combined = "\n\n---\n\n".join(texts)

        if len(combined) <= self.max_summary_length:
            return SummaryResult(
                summary=combined,
                word_count=len(combined.split()),
                source_count=len(texts),
            )

        prompt = self._build_summary_prompt(combined, style)

        try:
            summary = await self.llm.generate(prompt, max_tokens=self.max_summary_length)
        except Exception as e:
            self._logger.warning(f"LLM summarization failed: {e}, using fallback")
            summary = self._fallback_summary(combined)

        summary = summary[: self.max_summary_length]

        concepts = self.extract_key_concepts(summary)

        importance = self.estimate_importance_from_text(summary)

        return SummaryResult(
            summary=summary,
            key_concepts=concepts,
            importance_score=importance,
            word_count=len(summary.split()),
            source_count=len(texts),
        )

    async def create_project_summary(
        self,
        tree: Any,
        include_children: bool = True,
    ) -> str:
        """
        Create root-level summary for entire tree.

        Args:
            tree: SummaryTree instance
            include_children: Whether to include child summaries

        Returns:
            Project-level summary string

        Example:
            summary = await builder.create_project_summary(tree)
        """
        root = tree.get_root()
        if not root:
            return ""

        if root.summary:
            return root.summary

        summaries = []

        if include_children:
            for level in range(tree.get_max_level(), -1, -1):
                nodes = tree.get_nodes_at_level(level)
                for node in nodes[:10]:
                    if node.summary:
                        summaries.append(node.summary)

        if not summaries:
            summaries = [root.text] if root.text else []

        result = await self.summarize_texts(summaries, style="detailed")
        return result.summary

    def extract_key_concepts(
        self,
        text: str,
        max_concepts: int = 5,
    ) -> list[KeyConcept]:
        """
        Extract key concepts from text.

        Uses heuristic extraction based on:
        - Capitalized phrases
        - Repeated terms
        - Domain-specific patterns

        Args:
            text: Text to analyze
            max_concepts: Maximum concepts to extract

        Returns:
            List of KeyConcept objects

        Example:
            concepts = builder.extract_key_concepts(text, max_concepts=10)
        """
        concepts = []

        patterns = [
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            r"\b([A-Z]{2,})\b",
            r"\b(\w+(?:-\w+)+)\b",
        ]

        concept_counts: dict[str, int] = {}

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if len(match) > 2 and match.lower() not in STOPWORDS:
                    concept_counts[match.lower()] = concept_counts.get(match.lower(), 0) + 1

        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)

        for concept_name, count in sorted_concepts[:max_concepts]:
            relevance = min(count / 5, 1.0)

            context = self._extract_context(text, concept_name)

            concepts.append(
                KeyConcept(
                    name=concept_name,
                    description=context,
                    relevance=relevance,
                    category=self._categorize_concept(concept_name),
                )
            )

        return concepts

    def estimate_importance(
        self,
        node: Any,
    ) -> float:
        """
        Estimate importance of a tree node.

        Considers:
        - Node level (higher levels more important)
        - Text length
        - Keyword presence
        - Child count

        Args:
            node: SummaryTreeNode to evaluate

        Returns:
            Importance score (0.0 to 1.0)

        Example:
            score = builder.estimate_importance(node)
        """
        score = 0.5

        level_bonus = node.level * 0.05
        score += min(level_bonus, 0.2)

        text_length = len(node.text)
        if text_length > 1000:
            score += 0.1
        elif text_length > 500:
            score += 0.05

        importance_keywords = [
            "important",
            "critical",
            "key",
            "essential",
            "main",
            "primary",
            "significant",
            "core",
            "fundamental",
        ]
        text_lower = node.text.lower() + node.summary.lower()
        keyword_count = sum(1 for kw in importance_keywords if kw in text_lower)
        score += min(keyword_count * 0.03, 0.15)

        child_count = len(node.children)
        if child_count > 3:
            score += 0.1
        elif child_count > 0:
            score += 0.05

        return min(score, 1.0)

    def estimate_importance_from_text(
        self,
        text: str,
    ) -> float:
        """
        Estimate importance from text alone.

        Args:
            text: Text to analyze

        Returns:
            Importance score (0.0 to 1.0)
        """
        score = 0.5

        word_count = len(text.split())
        if word_count > 200:
            score += 0.1
        elif word_count > 100:
            score += 0.05

        importance_keywords = [
            "important",
            "critical",
            "key",
            "essential",
            "main",
            "primary",
        ]
        text_lower = text.lower()
        keyword_count = sum(1 for kw in importance_keywords if kw in text_lower)
        score += min(keyword_count * 0.05, 0.2)

        structure_indicators = ["##", "###", "- ", "* ", "1.", "2."]
        structure_count = sum(1 for ind in structure_indicators if ind in text)
        score += min(structure_count * 0.02, 0.1)

        return min(score, 1.0)

    async def summarize_node(
        self,
        node: Any,
        include_children: bool = True,
    ) -> str:
        """
        Summarize a single node, optionally including children.

        Args:
            node: SummaryTreeNode to summarize
            include_children: Whether to include child summaries

        Returns:
            Generated summary

        Example:
            summary = await builder.summarize_node(node, include_children=True)
        """
        texts = [node.text] if node.text else []

        if include_children and node.children:
            for child_id in node.children:
                child = getattr(node, "_tree", None)
                if child:
                    child_node = child.get_node(child_id)
                    if child_node and child_node.summary:
                        texts.append(child_node.summary)

        if not texts:
            return node.summary or ""

        result = await self.summarize_texts(texts)
        return result.summary

    def _build_summary_prompt(
        self,
        text: str,
        style: str,
    ) -> str:
        """Build prompt for summarization."""
        style_instructions = {
            "concise": "Provide a brief, concise summary capturing only the main points.",
            "detailed": "Provide a comprehensive summary covering all important details.",
            "bullet": "Summarize using bullet points for easy scanning.",
        }

        instruction = style_instructions.get(style, style_instructions["concise"])

        truncated_text = text[:2000] if len(text) > 2000 else text

        return f"""Summarize the following text.

{instruction}

Text:
{truncated_text}

Summary:"""

    def _fallback_summary(
        self,
        text: str,
    ) -> str:
        """Generate fallback summary when LLM fails."""
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            return text[: self.max_summary_length]

        target_length = int(len(text) * self.compression_ratio)
        summary_sentences = []
        current_length = 0

        for sentence in sentences[:10]:
            if current_length + len(sentence) > target_length:
                break
            summary_sentences.append(sentence)
            current_length += len(sentence)

        return ". ".join(summary_sentences) + "."

    def _extract_context(
        self,
        text: str,
        concept: str,
        context_words: int = 10,
    ) -> str:
        """Extract context around concept mention."""
        words = text.split()
        concept_lower = concept.lower()

        for i, word in enumerate(words):
            if word.lower() == concept_lower:
                start = max(0, i - context_words)
                end = min(len(words), i + context_words + 1)
                context = " ".join(words[start:end])
                return f"...{context}..."

        return f"Concept: {concept}"

    def _categorize_concept(
        self,
        concept: str,
    ) -> str:
        """Categorize a concept based on patterns."""
        tech_indicators = ["api", "code", "function", "class", "method", "data", "system"]
        business_indicators = ["business", "market", "customer", "product", "service"]
        process_indicators = ["process", "workflow", "step", "procedure", "method"]

        concept_lower = concept.lower()

        if any(ind in concept_lower for ind in tech_indicators):
            return "technical"
        if any(ind in concept_lower for ind in business_indicators):
            return "business"
        if any(ind in concept_lower for ind in process_indicators):
            return "process"

        return "general"


STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "ought",
    "used",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
}


class HierarchicalSummarizer:
    """
    Summarizer that maintains hierarchical context.

    Provides context-aware summarization for building
    RAPTOR trees with coherent summaries at each level.

    Attributes:
        base_summarizer: Base SummaryBuilder instance
        context_window: Number of sibling summaries to include

    Usage:
        summarizer = HierarchicalSummarizer(llm_provider=my_llm)
        summary = await summarizer.summarize_with_context(node, siblings)
    """

    def __init__(
        self,
        llm: LLMProvider | None = None,
        context_window: int = 3,
    ) -> None:
        self.base_summarizer = SummaryBuilder(llm=llm)
        self.context_window = context_window
        self._logger = logger

    async def summarize_with_context(
        self,
        node: Any,
        siblings: list[Any],
        parent_summary: str | None = None,
    ) -> str:
        """
        Summarize a node with sibling context.

        Args:
            node: Node to summarize
            siblings: Sibling nodes for context
            parent_summary: Optional parent summary for context

        Returns:
            Context-aware summary

        Example:
            summary = await summarizer.summarize_with_context(
                node, sibling_nodes, parent_summary
            )
        """
        texts = [node.text] if node.text else []

        for sibling in siblings[: self.context_window]:
            if sibling.id != node.id and sibling.summary:
                texts.append(f"[Context: {sibling.summary[:100]}]")

        if parent_summary:
            texts.insert(0, f"[Parent context: {parent_summary[:100]}]")

        result = await self.base_summarizer.summarize_texts(texts, style="concise")
        return result.summary

    async def build_level_summaries(
        self,
        nodes: list[Any],
        batch_size: int = 5,
    ) -> dict[str, str]:
        """
        Build summaries for all nodes at a level.

        Processes nodes in batches for efficiency.

        Args:
            nodes: List of nodes at same level
            batch_size: Number of nodes to process in parallel

        Returns:
            Dictionary of node ID to summary

        Example:
            summaries = await summarizer.build_level_summaries(nodes)
        """
        summaries = {}

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i : i + batch_size]

            tasks = []
            for node in batch:
                if not node.summary:
                    tasks.append(self.base_summarizer.summarize_texts([node.text]))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                result_idx = 0
                for node in batch:
                    if not node.summary:
                        result = results[result_idx]
                        result_idx += 1
                        if isinstance(result, Exception):
                            self._logger.warning(f"Summarization failed for {node.id}: {result}")
                            summaries[node.id] = node.text[:200]
                        else:
                            summaries[node.id] = result.summary

        return summaries


def create_summary_builder(
    llm_provider: LLMProvider | None = None,
    max_summary_length: int = 500,
) -> SummaryBuilder:
    """
    Create a SummaryBuilder with given configuration.

    Args:
        llm_provider: Optional LLM provider
        max_summary_length: Maximum summary length

    Returns:
        Configured SummaryBuilder instance
    """
    return SummaryBuilder(
        llm=llm_provider,
        max_summary_length=max_summary_length,
    )
