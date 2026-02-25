"""
Artifact System - MetaGPT-Inspired Artifact-Centric Communication

Implements structured outputs from agents with:
- Type-safe artifact definitions
- Central registry for tracking
- Artifact linking and provenance
- Validation and quality gates

Inspired by MetaGPT: https://github.com/geekan/MetaGPT
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("gaap.core.artifacts")


class ArtifactType(Enum):
    """
    Types of artifacts in the system.

    Each type represents a different kind of
    structured output from agents.
    """

    PR = auto()
    SPEC = auto()
    TEST_RESULT = auto()
    CODE = auto()
    DOCUMENT = auto()
    DIAGRAM = auto()
    REVIEW = auto()
    PLAN = auto()
    REPORT = auto()
    CONFIGURATION = auto()
    DATA = auto()
    MODEL = auto()
    UNKNOWN = auto()


class ArtifactStatus(Enum):
    """Status of an artifact in its lifecycle"""

    DRAFT = auto()
    PENDING_REVIEW = auto()
    APPROVED = auto()
    REJECTED = auto()
    DEPRECATED = auto()
    ARCHIVED = auto()


@dataclass
class ArtifactMetadata:
    """
    Metadata for an artifact.

    Tracks additional information about the artifact
    such as version, tags, and custom attributes.
    """

    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)
    language: str | None = None
    framework: str | None = None
    dependencies: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tags": self.tags,
            "language": self.language,
            "framework": self.framework,
            "dependencies": self.dependencies,
            "custom": self.custom,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactMetadata:
        return cls(
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
            language=data.get("language"),
            framework=data.get("framework"),
            dependencies=data.get("dependencies", []),
            custom=data.get("custom", {}),
        )


@dataclass
class ArtifactLink:
    """
    Link between two artifacts.

    Represents a relationship such as:
    - PR links to SPEC
    - CODE links to TEST_RESULT
    - DOCUMENT links to CODE
    """

    source_id: str
    target_id: str
    relationship: str
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Artifact:
    """
    Structured output from agents.

    Artifacts are the primary communication mechanism
    between agents in the MetaGPT-inspired architecture.

    Attributes:
        id: Unique identifier
        type: Type of artifact
        name: Human-readable name
        content: The actual content (can be any type)
        metadata: Additional metadata
        created_by: ID of the agent that created this
        created_at: Creation timestamp
        status: Current status
        parent_id: ID of parent artifact (if derived)
        validation_errors: List of validation errors
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ArtifactType = ArtifactType.UNKNOWN
    name: str = ""
    content: Any = None
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    status: ArtifactStatus = ArtifactStatus.DRAFT
    parent_id: str | None = None
    validation_errors: list[str] = field(default_factory=list)

    def compute_hash(self) -> str:
        """Compute a hash of the artifact content"""
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the artifact.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        if not self.name:
            errors.append("Artifact name is required")

        if self.content is None:
            errors.append("Artifact content is required")

        type_validators = {
            ArtifactType.CODE: self._validate_code,
            ArtifactType.SPEC: self._validate_spec,
            ArtifactType.TEST_RESULT: self._validate_test_result,
            ArtifactType.DOCUMENT: self._validate_document,
            ArtifactType.PR: self._validate_pr,
        }

        validator = type_validators.get(self.type)
        if validator:
            type_errors = validator()
            errors.extend(type_errors)

        self.validation_errors = errors
        return len(errors) == 0, errors

    def _validate_code(self) -> list[str]:
        """Validate code artifact"""
        errors = []
        if not isinstance(self.content, str):
            errors.append("Code content must be a string")
        elif len(self.content) < 10:
            errors.append("Code content is too short")
        return errors

    def _validate_spec(self) -> list[str]:
        """Validate specification artifact"""
        errors = []
        if not isinstance(self.content, dict):
            errors.append("Spec content must be a dictionary")
        else:
            required_keys = ["description", "requirements"]
            for key in required_keys:
                if key not in self.content:
                    errors.append(f"Spec missing required key: {key}")
        return errors

    def _validate_test_result(self) -> list[str]:
        """Validate test result artifact"""
        errors = []
        if not isinstance(self.content, dict):
            errors.append("Test result content must be a dictionary")
        else:
            if "passed" not in self.content:
                errors.append("Test result missing 'passed' field")
            if "total" not in self.content:
                errors.append("Test result missing 'total' field")
        return errors

    def _validate_document(self) -> list[str]:
        """Validate document artifact"""
        errors = []
        if not isinstance(self.content, str):
            errors.append("Document content must be a string")
        elif len(self.content) < 50:
            errors.append("Document content is too short")
        return errors

    def _validate_pr(self) -> list[str]:
        """Validate PR artifact"""
        errors = []
        if not isinstance(self.content, dict):
            errors.append("PR content must be a dictionary")
        else:
            required_keys = ["title", "description", "changes"]
            for key in required_keys:
                if key not in self.content:
                    errors.append(f"PR missing required key: {key}")
        return errors

    def to_dict(self) -> dict[str, Any]:
        """Convert artifact to dictionary"""
        return {
            "id": self.id,
            "type": self.type.name,
            "name": self.name,
            "content": self.content,
            "metadata": self.metadata.to_dict(),
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "status": self.status.name,
            "parent_id": self.parent_id,
            "hash": self.compute_hash(),
            "validation_errors": self.validation_errors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Artifact:
        """Create artifact from dictionary"""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=ArtifactType[data.get("type", "UNKNOWN")],
            name=data.get("name", ""),
            content=data.get("content"),
            metadata=ArtifactMetadata.from_dict(data.get("metadata", {})),
            created_by=data.get("created_by", ""),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
            status=ArtifactStatus[data.get("status", "DRAFT")],
            parent_id=data.get("parent_id"),
            validation_errors=data.get("validation_errors", []),
        )


class ArtifactBuilder:
    """
    Builder for creating artifacts fluently.

    Usage:
        artifact = (
            ArtifactBuilder()
            .type(ArtifactType.CODE)
            .name("main.py")
            .content("def hello(): pass")
            .created_by("agent_01")
            .tag("python")
            .tag("core")
            .build()
        )
    """

    def __init__(self) -> None:
        self._type = ArtifactType.UNKNOWN
        self._name = ""
        self._content: Any = None
        self._metadata = ArtifactMetadata()
        self._created_by = ""
        self._status = ArtifactStatus.DRAFT
        self._parent_id: str | None = None

    def type(self, artifact_type: ArtifactType) -> ArtifactBuilder:
        """Set the artifact type"""
        self._type = artifact_type
        return self

    def name(self, name: str) -> ArtifactBuilder:
        """Set the artifact name"""
        self._name = name
        return self

    def content(self, content: Any) -> ArtifactBuilder:
        """Set the artifact content"""
        self._content = content
        return self

    def created_by(self, agent_id: str) -> ArtifactBuilder:
        """Set the creator agent ID"""
        self._created_by = agent_id
        return self

    def status(self, status: ArtifactStatus) -> ArtifactBuilder:
        """Set the artifact status"""
        self._status = status
        return self

    def parent(self, parent_id: str) -> ArtifactBuilder:
        """Set the parent artifact ID"""
        self._parent_id = parent_id
        return self

    def tag(self, tag: str) -> ArtifactBuilder:
        """Add a tag to metadata"""
        if tag not in self._metadata.tags:
            self._metadata.tags.append(tag)
        return self

    def version(self, version: str) -> ArtifactBuilder:
        """Set the version in metadata"""
        self._metadata.version = version
        return self

    def language(self, language: str) -> ArtifactBuilder:
        """Set the language in metadata"""
        self._metadata.language = language
        return self

    def framework(self, framework: str) -> ArtifactBuilder:
        """Set the framework in metadata"""
        self._metadata.framework = framework
        return self

    def dependency(self, dep: str) -> ArtifactBuilder:
        """Add a dependency to metadata"""
        if dep not in self._metadata.dependencies:
            self._metadata.dependencies.append(dep)
        return self

    def custom_metadata(self, key: str, value: Any) -> ArtifactBuilder:
        """Add custom metadata"""
        self._metadata.custom[key] = value
        return self

    def build(self) -> Artifact:
        """Build the artifact"""
        return Artifact(
            type=self._type,
            name=self._name,
            content=self._content,
            metadata=self._metadata,
            created_by=self._created_by,
            status=self._status,
            parent_id=self._parent_id,
        )


class ArtifactRegistry:
    """
    Central registry for all artifacts.

    Provides:
    - Registration and lookup
    - Type-based filtering
    - Artifact linking
    - Persistence support
    - Query capabilities

    Usage:
        registry = ArtifactRegistry()

        # Register
        artifact_id = registry.register(artifact)

        # Get by ID
        artifact = registry.get(artifact_id)

        # Get by type
        code_artifacts = registry.get_by_type(ArtifactType.CODE)

        # Link artifacts
        registry.link(pr_id, spec_id, "implements")
    """

    def __init__(self, storage_path: str | None = None) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._links: list[ArtifactLink] = []
        self._type_index: dict[ArtifactType, list[str]] = {t: [] for t in ArtifactType}
        self._creator_index: dict[str, list[str]] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        self._logger = logging.getLogger("gaap.core.artifacts.registry")

        if self._storage_path:
            self._load_from_storage()

    def register(self, artifact: Artifact) -> str:
        """
        Register an artifact.

        Returns:
            The artifact ID
        """
        if not artifact.id:
            artifact.id = str(uuid.uuid4())

        self._artifacts[artifact.id] = artifact

        if artifact.id not in self._type_index[artifact.type]:
            self._type_index[artifact.type].append(artifact.id)

        if artifact.created_by:
            if artifact.created_by not in self._creator_index:
                self._creator_index[artifact.created_by] = []
            if artifact.id not in self._creator_index[artifact.created_by]:
                self._creator_index[artifact.created_by].append(artifact.id)

        self._logger.debug(f"Registered artifact: {artifact.id} ({artifact.type.name})")

        if self._storage_path:
            self._save_to_storage()

        return artifact.id

    def get(self, artifact_id: str) -> Artifact | None:
        """Get an artifact by ID"""
        return self._artifacts.get(artifact_id)

    def get_by_type(self, artifact_type: ArtifactType) -> list[Artifact]:
        """Get all artifacts of a specific type"""
        ids = self._type_index.get(artifact_type, [])
        return [self._artifacts[id_] for id_ in ids if id_ in self._artifacts]

    def get_by_creator(self, creator_id: str) -> list[Artifact]:
        """Get all artifacts created by a specific agent"""
        ids = self._creator_index.get(creator_id, [])
        return [self._artifacts[id_] for id_ in ids if id_ in self._artifacts]

    def get_by_status(self, status: ArtifactStatus) -> list[Artifact]:
        """Get all artifacts with a specific status"""
        return [a for a in self._artifacts.values() if a.status == status]

    def get_by_tag(self, tag: str) -> list[Artifact]:
        """Get all artifacts with a specific tag"""
        return [a for a in self._artifacts.values() if tag in a.metadata.tags]

    def link(
        self,
        source_id: str,
        target_id: str,
        relationship: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactLink | None:
        """
        Create a link between two artifacts.

        Args:
            source_id: Source artifact ID
            target_id: Target artifact ID
            relationship: Type of relationship (e.g., "implements", "tests", "reviews")
            metadata: Optional metadata for the link

        Returns:
            The created link, or None if artifacts don't exist
        """
        if source_id not in self._artifacts or target_id not in self._artifacts:
            self._logger.warning(f"Cannot link: artifact not found ({source_id} -> {target_id})")
            return None

        link = ArtifactLink(
            source_id=source_id,
            target_id=target_id,
            relationship=relationship,
            metadata=metadata or {},
        )

        self._links.append(link)
        self._logger.debug(f"Created link: {source_id} -[{relationship}]-> {target_id}")

        if self._storage_path:
            self._save_to_storage()

        return link

    def get_links(self, artifact_id: str) -> list[ArtifactLink]:
        """Get all links for an artifact (both incoming and outgoing)"""
        return [
            link
            for link in self._links
            if link.source_id == artifact_id or link.target_id == artifact_id
        ]

    def get_linked_artifacts(
        self, artifact_id: str, relationship: str | None = None
    ) -> list[Artifact]:
        """Get artifacts linked to/from this artifact"""
        linked_ids = set()

        for link in self._links:
            if relationship and link.relationship != relationship:
                continue

            if link.source_id == artifact_id:
                linked_ids.add(link.target_id)
            elif link.target_id == artifact_id:
                linked_ids.add(link.source_id)

        return [self._artifacts[id_] for id_ in linked_ids if id_ in self._artifacts]

    def update_status(self, artifact_id: str, status: ArtifactStatus) -> bool:
        """Update the status of an artifact"""
        artifact = self._artifacts.get(artifact_id)
        if artifact:
            artifact.status = status
            self._logger.debug(f"Updated artifact {artifact_id} status to {status.name}")
            return True
        return False

    def delete(self, artifact_id: str) -> bool:
        """Delete an artifact"""
        if artifact_id not in self._artifacts:
            return False

        artifact = self._artifacts[artifact_id]

        self._type_index[artifact.type].remove(artifact_id)

        if artifact.created_by in self._creator_index:
            self._creator_index[artifact.created_by].remove(artifact_id)

        self._links = [
            link
            for link in self._links
            if link.source_id != artifact_id and link.target_id != artifact_id
        ]

        del self._artifacts[artifact_id]

        if self._storage_path:
            self._save_to_storage()

        return True

    def query(
        self,
        type: ArtifactType | None = None,
        status: ArtifactStatus | None = None,
        created_by: str | None = None,
        tag: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[Artifact]:
        """
        Query artifacts with multiple filters.

        All filters are optional and combined with AND logic.
        """
        results = list(self._artifacts.values())

        if type is not None:
            results = [a for a in results if a.type == type]

        if status is not None:
            results = [a for a in results if a.status == status]

        if created_by is not None:
            results = [a for a in results if a.created_by == created_by]

        if tag is not None:
            results = [a for a in results if tag in a.metadata.tags]

        if created_after is not None:
            results = [a for a in results if a.created_at >= created_after]

        if created_before is not None:
            results = [a for a in results if a.created_at <= created_before]

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics"""
        type_counts = {t.name: len(ids) for t, ids in self._type_index.items() if ids}

        status_counts = {}
        for status in ArtifactStatus:
            count = sum(1 for a in self._artifacts.values() if a.status == status)
            if count > 0:
                status_counts[status.name] = count

        return {
            "total_artifacts": len(self._artifacts),
            "total_links": len(self._links),
            "by_type": type_counts,
            "by_status": status_counts,
            "creators": len(self._creator_index),
        }

    def _save_to_storage(self) -> None:
        """Save registry to storage"""
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)

        artifacts_file = self._storage_path / "artifacts.json"
        links_file = self._storage_path / "links.json"

        with open(artifacts_file, "w") as f:
            json.dump(
                [a.to_dict() for a in self._artifacts.values()],
                f,
                indent=2,
                default=str,
            )

        with open(links_file, "w") as f:
            json.dump(
                [l.to_dict() for l in self._links],
                f,
                indent=2,
                default=str,
            )

    def _load_from_storage(self) -> None:
        """Load registry from storage"""
        if not self._storage_path:
            return

        artifacts_file = self._storage_path / "artifacts.json"
        links_file = self._storage_path / "links.json"

        if artifacts_file.exists():
            try:
                with open(artifacts_file) as f:
                    data = json.load(f)
                for item in data:
                    artifact = Artifact.from_dict(item)
                    self._artifacts[artifact.id] = artifact
                    self._type_index[artifact.type].append(artifact.id)
                    if artifact.created_by:
                        if artifact.created_by not in self._creator_index:
                            self._creator_index[artifact.created_by] = []
                        self._creator_index[artifact.created_by].append(artifact.id)
            except Exception as e:
                self._logger.error(f"Failed to load artifacts: {e}")

        if links_file.exists():
            try:
                with open(links_file) as f:
                    data = json.load(f)
                for item in data:
                    link = ArtifactLink(
                        source_id=item["source_id"],
                        target_id=item["target_id"],
                        relationship=item["relationship"],
                        created_at=datetime.fromisoformat(item["created_at"]),
                        metadata=item.get("metadata", {}),
                    )
                    self._links.append(link)
            except Exception as e:
                self._logger.error(f"Failed to load links: {e}")


def create_artifact_registry(storage_path: str | None = None) -> ArtifactRegistry:
    """Factory function to create an ArtifactRegistry"""
    return ArtifactRegistry(storage_path=storage_path)
