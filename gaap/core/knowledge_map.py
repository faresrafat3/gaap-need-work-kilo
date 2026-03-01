"""
KnowledgeMap - What the Agent Knows
===================================

Tracks the agent's knowledge boundaries to enable
"knowing what it doesn't know".

Maps:
- Known libraries and frameworks
- Known APIs and services
- Known code patterns
- Domain knowledge

Usage:
    km = KnowledgeMap()

    # Assess novelty
    novelty = km.assess_novelty("Create a GraphQL API using Strawberry")
    # Returns: 0.3 (somewhat familiar)

    # Get unknown entities
    unknowns = km.get_unknown_entities("Use the Zotero API for citations")
    # Returns: ["zotero api"]

    # Check knowledge gaps
    gaps = km.get_knowledge_gaps("Implement OAuth2 with PKCE")
    # Returns: ["PKCE extension details"]
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.core.knowledge_map")


class EntityType(Enum):
    """أنواع الكيانات المعرفية"""

    LIBRARY = auto()
    FRAMEWORK = auto()
    API = auto()
    SERVICE = auto()
    PATTERN = auto()
    CONCEPT = auto()
    DOMAIN = auto()
    LANGUAGE = auto()
    TOOL = auto()
    PROTOCOL = auto()


class KnowledgeLevel(Enum):
    """مستوى المعرفة"""

    EXPERT = auto()
    PROFICIENT = auto()
    FAMILIAR = auto()
    AWARE = auto()
    UNKNOWN = auto()


@dataclass
class KnowledgeEntity:
    """كيان معرفي"""

    name: str
    entity_type: EntityType
    knowledge_level: KnowledgeLevel = KnowledgeLevel.FAMILIAR
    confidence: float = 0.5
    last_used: datetime | None = None
    usage_count: int = 0
    source: str = "builtin"
    aliases: list[str] = field(default_factory=list)
    related: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity_type": self.entity_type.name,
            "knowledge_level": self.knowledge_level.name,
            "confidence": self.confidence,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
            "source": self.source,
            "aliases": self.aliases,
            "related": self.related,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEntity":
        return cls(
            name=data.get("name", ""),
            entity_type=EntityType[data.get("entity_type", "CONCEPT")],
            knowledge_level=KnowledgeLevel[data.get("knowledge_level", "FAMILIAR")],
            confidence=data.get("confidence", 0.5),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
            usage_count=data.get("usage_count", 0),
            source=data.get("source", "manual"),
            aliases=data.get("aliases", []),
            related=data.get("related", []),
            metadata=data.get("metadata", {}),
        )

    def record_usage(self) -> None:
        """Record that this entity was used."""
        self.usage_count += 1
        self.last_used = datetime.now()


@dataclass
class KnowledgeGap:
    """فجوة معرفية"""

    entity_name: str
    entity_type: EntityType
    context: str
    importance: float
    suggested_research: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_name": self.entity_name,
            "entity_type": self.entity_type.name,
            "context": self.context,
            "importance": self.importance,
            "suggested_research": self.suggested_research,
        }


class KnowledgeMap:
    """
    Maps the agent's knowledge boundaries.

    Features:
    - Track known entities (libraries, APIs, patterns)
    - Detect novel concepts in tasks
    - Identify knowledge gaps
    - Assess novelty scores
    - Persist knowledge state

    Usage:
        km = KnowledgeMap()

        novelty = km.assess_novelty("Use Pydantic for validation")
        # Returns 0.2 (Pydantic is well known)

        gaps = km.get_knowledge_gaps("Implement gRPC streaming")
        # Returns gaps if gRPC or streaming not well known
    """

    DEFAULT_STORAGE_PATH = ".gaap/knowledge_map.json"

    BUILTIN_KNOWLEDGE = {
        "python": (EntityType.LANGUAGE, KnowledgeLevel.EXPERT),
        "javascript": (EntityType.LANGUAGE, KnowledgeLevel.PROFICIENT),
        "typescript": (EntityType.LANGUAGE, KnowledgeLevel.PROFICIENT),
        "rust": (EntityType.LANGUAGE, KnowledgeLevel.FAMILIAR),
        "go": (EntityType.LANGUAGE, KnowledgeLevel.FAMILIAR),
        "java": (EntityType.LANGUAGE, KnowledgeLevel.PROFICIENT),
        "fastapi": (EntityType.FRAMEWORK, KnowledgeLevel.EXPERT),
        "django": (EntityType.FRAMEWORK, KnowledgeLevel.EXPERT),
        "flask": (EntityType.FRAMEWORK, KnowledgeLevel.EXPERT),
        "react": (EntityType.FRAMEWORK, KnowledgeLevel.PROFICIENT),
        "vue": (EntityType.FRAMEWORK, KnowledgeLevel.FAMILIAR),
        "express": (EntityType.FRAMEWORK, KnowledgeLevel.PROFICIENT),
        "sqlalchemy": (EntityType.LIBRARY, KnowledgeLevel.EXPERT),
        "pydantic": (EntityType.LIBRARY, KnowledgeLevel.EXPERT),
        "pytest": (EntityType.LIBRARY, KnowledgeLevel.EXPERT),
        "numpy": (EntityType.LIBRARY, KnowledgeLevel.PROFICIENT),
        "pandas": (EntityType.LIBRARY, KnowledgeLevel.PROFICIENT),
        "requests": (EntityType.LIBRARY, KnowledgeLevel.EXPERT),
        "aiohttp": (EntityType.LIBRARY, KnowledgeLevel.PROFICIENT),
        "httpx": (EntityType.LIBRARY, KnowledgeLevel.PROFICIENT),
        "redis": (EntityType.SERVICE, KnowledgeLevel.PROFICIENT),
        "postgresql": (EntityType.SERVICE, KnowledgeLevel.PROFICIENT),
        "mongodb": (EntityType.SERVICE, KnowledgeLevel.FAMILIAR),
        "docker": (EntityType.TOOL, KnowledgeLevel.PROFICIENT),
        "kubernetes": (EntityType.TOOL, KnowledgeLevel.FAMILIAR),
        "git": (EntityType.TOOL, KnowledgeLevel.EXPERT),
        "rest": (EntityType.PROTOCOL, KnowledgeLevel.EXPERT),
        "graphql": (EntityType.PROTOCOL, KnowledgeLevel.PROFICIENT),
        "grpc": (EntityType.PROTOCOL, KnowledgeLevel.FAMILIAR),
        "websocket": (EntityType.PROTOCOL, KnowledgeLevel.PROFICIENT),
        "oauth2": (EntityType.PROTOCOL, KnowledgeLevel.PROFICIENT),
        "jwt": (EntityType.PROTOCOL, KnowledgeLevel.EXPERT),
        "async": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "mvc": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "microservices": (EntityType.PATTERN, KnowledgeLevel.PROFICIENT),
        "clean architecture": (EntityType.PATTERN, KnowledgeLevel.PROFICIENT),
        "dependency injection": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "factory pattern": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "singleton": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "observer": (EntityType.PATTERN, KnowledgeLevel.EXPERT),
        "security": (EntityType.DOMAIN, KnowledgeLevel.PROFICIENT),
        "authentication": (EntityType.DOMAIN, KnowledgeLevel.PROFICIENT),
        "testing": (EntityType.DOMAIN, KnowledgeLevel.EXPERT),
        "database": (EntityType.DOMAIN, KnowledgeLevel.PROFICIENT),
        "api design": (EntityType.DOMAIN, KnowledgeLevel.EXPERT),
    }

    def __init__(
        self,
        storage_path: str | None = None,
        procedural_memory: Any = None,
    ) -> None:
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self._procedural = procedural_memory

        self._entities: dict[str, KnowledgeEntity] = {}
        self._unknown_history: dict[str, int] = {}
        self._keyword_index: dict[str, set[str]] = {}

        self._logger = logger

        self._load_builtin_knowledge()
        self._load()

    def assess_novelty(self, task_description: str) -> float:
        """
        Assess how novel a task is based on knowledge map.

        Args:
            task_description: Description of the task

        Returns:
            Novelty score (0.0 = familiar, 1.0 = completely novel)
        """
        entities = self._extract_entities(task_description)

        if not entities:
            return 0.0

        unknown_count = 0
        total_confidence = 0.0

        for entity_name, entity_type in entities:
            entity = self._find_entity(entity_name)

            if entity:
                if entity.knowledge_level == KnowledgeLevel.UNKNOWN:
                    unknown_count += 1
                else:
                    total_confidence += entity.confidence
            else:
                unknown_count += 1

        total_entities = len(entities)
        if total_entities == 0:
            return 0.0

        unknown_ratio = unknown_count / total_entities
        avg_confidence = total_confidence / max(total_entities - unknown_count, 1)

        novelty = unknown_ratio * 0.7 + (1 - avg_confidence) * 0.3

        return min(novelty, 1.0)

    def get_unknown_entities(self, text: str) -> list[str]:
        """
        Get list of unknown entities in text.

        Args:
            text: Text to analyze

        Returns:
            List of unknown entity names
        """
        entities = self._extract_entities(text)
        unknown = []

        for entity_name, _ in entities:
            entity = self._find_entity(entity_name)
            if not entity or entity.knowledge_level == KnowledgeLevel.UNKNOWN:
                unknown.append(entity_name)
                self._unknown_history[entity_name] = self._unknown_history.get(entity_name, 0) + 1

        return unknown

    def get_knowledge_gaps(self, task_description: str) -> list[KnowledgeGap]:
        """
        Identify knowledge gaps for a task.

        Args:
            task_description: Task description

        Returns:
            List of KnowledgeGap objects
        """
        entities = self._extract_entities(task_description)
        gaps = []

        for entity_name, entity_type in entities:
            entity = self._find_entity(entity_name)

            if not entity:
                gaps.append(
                    KnowledgeGap(
                        entity_name=entity_name,
                        entity_type=entity_type,
                        context=task_description[:200],
                        importance=self._calculate_importance(entity_name),
                        suggested_research=f"Research {entity_name} documentation and best practices",
                    )
                )
            elif entity.knowledge_level == KnowledgeLevel.UNKNOWN:
                gaps.append(
                    KnowledgeGap(
                        entity_name=entity_name,
                        entity_type=entity.entity_type,
                        context=task_description[:200],
                        importance=self._calculate_importance(entity_name) * 0.8,
                        suggested_research=f"Deepen understanding of {entity_name}",
                    )
                )
            elif entity.confidence < 0.5:
                gaps.append(
                    KnowledgeGap(
                        entity_name=entity_name,
                        entity_type=entity.entity_type,
                        context=task_description[:200],
                        importance=self._calculate_importance(entity_name) * 0.5,
                        suggested_research=f"Review {entity_name} concepts",
                    )
                )

        return sorted(gaps, key=lambda g: g.importance, reverse=True)

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        knowledge_level: KnowledgeLevel = KnowledgeLevel.FAMILIAR,
        confidence: float = 0.5,
        aliases: list[str] | None = None,
        related: list[str] | None = None,
    ) -> KnowledgeEntity:
        """
        Add a new entity to the knowledge map.

        Args:
            name: Entity name
            entity_type: Type of entity
            knowledge_level: Knowledge level
            confidence: Confidence level
            aliases: Alternative names
            related: Related entities

        Returns:
            Created KnowledgeEntity
        """
        entity = KnowledgeEntity(
            name=name.lower(),
            entity_type=entity_type,
            knowledge_level=knowledge_level,
            confidence=confidence,
            aliases=[a.lower() for a in (aliases or [])],
            related=[r.lower() for r in (related or [])],
            source="learned",
        )

        self._entities[entity.name] = entity

        for alias in entity.aliases:
            self._keyword_index[alias] = {entity.name}

        self._save()
        return entity

    def record_usage(self, entity_name: str) -> None:
        """Record usage of an entity."""
        entity = self._find_entity(entity_name)
        if entity:
            entity.record_usage()
            self._save()

    def update_knowledge_level(
        self,
        entity_name: str,
        level: KnowledgeLevel,
        confidence: float | None = None,
    ) -> bool:
        """Update knowledge level of an entity."""
        entity = self._find_entity(entity_name)
        if entity:
            entity.knowledge_level = level
            if confidence is not None:
                entity.confidence = confidence
            self._save()
            return True
        return False

    def get_repeated_unknowns(self, min_count: int = 3) -> list[tuple[str, int]]:
        """Get entities that have been unknown multiple times."""
        return [
            (name, count) for name, count in self._unknown_history.items() if count >= min_count
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge map statistics."""
        by_type: dict[str, int] = {}
        by_level: dict[str, int] = {}

        for entity in self._entities.values():
            by_type[entity.entity_type.name] = by_type.get(entity.entity_type.name, 0) + 1
            by_level[entity.knowledge_level.name] = by_level.get(entity.knowledge_level.name, 0) + 1

        return {
            "total_entities": len(self._entities),
            "by_type": by_type,
            "by_level": by_level,
            "unknowns_encountered": len(self._unknown_history),
            "repeated_unknowns": len(self.get_repeated_unknowns()),
        }

    def _find_entity(self, name: str) -> KnowledgeEntity | None:
        """Find entity by name or alias."""
        name_lower = name.lower()

        if name_lower in self._entities:
            return self._entities[name_lower]

        for entity in self._entities.values():
            if name_lower in entity.aliases:
                return entity

        for entity_name in self._keyword_index.get(name_lower, set()):
            if entity_name in self._entities:
                return self._entities[entity_name]

        return None

    def _extract_entities(self, text: str) -> list[tuple[str, EntityType]]:
        """Extract potential entities from text."""
        entities = []

        tech_patterns = [
            (r"\b([A-Z][a-z]+(?:API|SDK|CLI|UI|ML|AI|DB|ORM|HTTP|REST|SDK))\b", EntityType.API),
            (r"\b([A-Z][a-z]+(?:Framework|Library|Engine|Server|Client))\b", EntityType.LIBRARY),
            (
                r"\b(python|javascript|typescript|rust|golang|java|ruby|php|go|c\+\+|c#)\b",
                EntityType.LANGUAGE,
            ),
            (r"\b([a-z_]+(?:js|py|rs|go|ts))\b", EntityType.LIBRARY),
            (r"\b([A-Z][a-z]+(?:DB|Database|SQL|NoSQL))\b", EntityType.SERVICE),
            (r"\b(docker|kubernetes|k8s|terraform|ansible|jenkins)\b", EntityType.TOOL),
            (r"\b(oauth|oauth2|jwt|saml|ldap|grpc|graphql|rest|websocket)\b", EntityType.PROTOCOL),
            (r"\b(microservice|monolith|serverless|event[- ]driven)\b", EntityType.PATTERN),
        ]

        text_lower = text.lower()
        for pattern, etype in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.lower() if isinstance(match, str) else match[0].lower()
                if len(entity_name) > 2:
                    entities.append((entity_name, etype))

        for name in self._entities:
            if name in text_lower:
                entities.append((name, self._entities[name].entity_type))

        seen = set()
        unique_entities = []
        for entity in entities:
            if entity[0] not in seen:
                seen.add(entity[0])
                unique_entities.append(entity)

        return unique_entities

    def _calculate_importance(self, entity_name: str) -> float:
        """Calculate importance of an unknown entity."""
        count = self._unknown_history.get(entity_name, 0)
        return min(0.5 + count * 0.1, 1.0)

    def _load_builtin_knowledge(self) -> None:
        """Load built-in knowledge entities."""
        for name, (etype, level) in self.BUILTIN_KNOWLEDGE.items():
            confidence = {
                KnowledgeLevel.EXPERT: 0.95,
                KnowledgeLevel.PROFICIENT: 0.8,
                KnowledgeLevel.FAMILIAR: 0.6,
                KnowledgeLevel.AWARE: 0.4,
                KnowledgeLevel.UNKNOWN: 0.1,
            }.get(level, 0.5)

            self._entities[name.lower()] = KnowledgeEntity(
                name=name.lower(),
                entity_type=etype,
                knowledge_level=level,
                confidence=confidence,
                source="builtin",
            )

    def _save(self) -> bool:
        """Save knowledge map to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "entities": {
                    name: e.to_dict() for name, e in self._entities.items() if e.source != "builtin"
                },
                "unknown_history": self._unknown_history,
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            self._logger.error(f"Failed to save knowledge map: {e}")
            return False

    def _load(self) -> bool:
        """Load knowledge map from disk."""
        try:
            if not self.storage_path.exists():
                return True

            with open(self.storage_path) as f:
                data = json.load(f)

            for name, entity_data in data.get("entities", {}).items():
                self._entities[name] = KnowledgeEntity.from_dict(entity_data)

            self._unknown_history = data.get("unknown_history", {})

            return True

        except Exception as e:
            self._logger.error(f"Failed to load knowledge map: {e}")
            return False


def create_knowledge_map(
    storage_path: str | None = None,
) -> KnowledgeMap:
    """Create a KnowledgeMap instance."""
    return KnowledgeMap(storage_path=storage_path)
