"""
Skill Cache - Persistent Storage for Synthesized Tools
======================================================

Manages persistent storage of synthesized tools with:
- JSON metadata alongside each Python skill
- Automatic checksum verification
- Category organization (coding, research, analysis, etc.)
- Import validation
- Thread-safe operations

File Structure:
    .gaap/skills/
    ├── metadata.json         # Index of all skills
    ├── coding/
    │   ├── tool_abc123.py
    │   └── tool_abc123.json
    ├── research/
    │   ├── tool_def456.py
    │   └── tool_def456.json
    └── analysis/
        └── ...
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from gaap.tools.synthesizer import SynthesizedTool

logger = logging.getLogger("gaap.tools.skill_cache")

CATEGORIES = ["coding", "research", "analysis", "automation", "utility", "other"]

SCHEMA_VERSION = 1


@dataclass
class SkillMetadata:
    id: str
    name: str
    description: str
    created_at: str
    last_used: str
    use_count: int
    dependencies: list[str]
    tags: list[str]
    file_path: str
    checksum: str
    category: str = "other"
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "file_path": self.file_path,
            "checksum": self.checksum,
            "category": self.category,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillMetadata:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            last_used=data.get("last_used", ""),
            use_count=data.get("use_count", 0),
            dependencies=data.get("dependencies", []),
            tags=data.get("tags", []),
            file_path=data.get("file_path", ""),
            checksum=data.get("checksum", ""),
            category=data.get("category", "other"),
            schema_version=data.get("schema_version", SCHEMA_VERSION),
        )


@dataclass
class SkillCacheStats:
    total_skills: int
    total_size_bytes: int
    categories: dict[str, int]
    most_used: list[SkillMetadata]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_skills": self.total_skills,
            "total_size_bytes": self.total_size_bytes,
            "categories": self.categories,
            "most_used": [s.to_dict() for s in self.most_used],
        }


class SkillCache:
    """
    Thread-safe persistent storage for synthesized tools.

    Features:
    - JSON metadata alongside each Python skill
    - Automatic checksum verification
    - Category organization
    - Import validation
    - Migration support for format changes

    Example:
        >>> cache = SkillCache()
        >>> skill_id = cache.store(synthesized_tool, {"tags": ["research"]})
        >>> tool = cache.retrieve(skill_id)
    """

    def __init__(self, cache_path: Path | str = ".gaap/skills"):
        self.cache_path = Path(cache_path)
        self._lock = threading.Lock()
        self._metadata_cache: dict[str, SkillMetadata] = {}
        self._initialized = False

        self._init_cache()

    def _init_cache(self) -> None:
        """Initialize cache directory structure and load metadata index."""
        with self._lock:
            if self._initialized:
                return

            self.cache_path.mkdir(parents=True, exist_ok=True)

            for category in CATEGORIES:
                (self.cache_path / category).mkdir(exist_ok=True)

            if str(self.cache_path.absolute()) not in sys.path:
                sys.path.insert(0, str(self.cache_path.absolute()))

            self._load_metadata_index()
            self._initialized = True

    def _get_metadata_index_path(self) -> Path:
        return self.cache_path / "metadata.json"

    def _load_metadata_index(self) -> None:
        """Load the main metadata index into memory."""
        index_path = self._get_metadata_index_path()
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                    for skill_data in data.get("skills", []):
                        metadata = SkillMetadata.from_dict(skill_data)
                        self._metadata_cache[metadata.id] = metadata
            except Exception as e:
                logger.warning(f"Failed to load metadata index: {e}")

    def _save_metadata_index(self) -> None:
        """Save the metadata index to disk."""
        index_path = self._get_metadata_index_path()
        data = {
            "version": SCHEMA_VERSION,
            "updated_at": datetime.now().isoformat(),
            "skills": [m.to_dict() for m in self._metadata_cache.values()],
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _calculate_checksum(self, code: str) -> str:
        """Calculate SHA256 checksum of code."""
        return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]

    def _determine_category(self, tags: list[str], description: str) -> str:
        """Determine skill category from tags and description."""
        desc_lower = description.lower()
        tags_lower = [t.lower() for t in tags]

        for category in CATEGORIES[:-1]:
            if category in tags_lower or category in desc_lower:
                return category

        if any(t in desc_lower for t in ["code", "function", "class", "module"]):
            return "coding"
        if any(t in desc_lower for t in ["search", "find", "lookup", "research"]):
            return "research"
        if any(t in desc_lower for t in ["analyze", "parse", "extract", "process"]):
            return "analysis"
        if any(t in desc_lower for t in ["automate", "schedule", "batch"]):
            return "automation"

        return "other"

    def _get_skill_path(self, skill_id: str, category: str) -> Path:
        """Get the file path for a skill."""
        return self.cache_path / category / f"tool_{skill_id}.py"

    def _get_metadata_path(self, skill_id: str, category: str) -> Path:
        """Get the metadata JSON path for a skill."""
        return self.cache_path / category / f"tool_{skill_id}.json"

    def store(self, skill: SynthesizedTool, metadata: dict[str, Any]) -> str:
        """
        Store a synthesized tool with metadata.

        Args:
            skill: The synthesized tool to store
            metadata: Additional metadata (tags, dependencies, etc.)

        Returns:
            skill_id: The ID of the stored skill
        """
        with self._lock:
            tags = metadata.get("tags", [])
            description = metadata.get("description", skill.description)
            category = self._determine_category(tags, description)

            skill_id = skill.id
            now = datetime.now().isoformat()

            file_path = self._get_skill_path(skill_id, category)
            meta_path = self._get_metadata_path(skill_id, category)

            checksum = self._calculate_checksum(skill.code)

            skill_meta = SkillMetadata(
                id=skill_id,
                name=skill.name,
                description=description,
                created_at=now,
                last_used=now,
                use_count=1,
                dependencies=metadata.get("dependencies", []),
                tags=tags,
                file_path=str(file_path),
                checksum=checksum,
                category=category,
            )

            file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(file_path, "w") as f:
                f.write(skill.code)

            with open(meta_path, "w") as f:
                json.dump(skill_meta.to_dict(), f, indent=2, ensure_ascii=False)

            self._metadata_cache[skill_id] = skill_meta
            self._save_metadata_index()

            logger.info(f"Stored skill {skill_id} in category '{category}'")
            return skill_id

    def retrieve(self, skill_id: str) -> SynthesizedTool | None:
        """
        Retrieve a synthesized tool by ID.

        Args:
            skill_id: The skill ID to retrieve

        Returns:
            SynthesizedTool or None if not found
        """
        with self._lock:
            metadata = self._metadata_cache.get(skill_id)
            if not metadata:
                return None

            file_path = Path(metadata.file_path)
            if not file_path.exists():
                logger.warning(f"Skill file not found: {file_path}")
                return None

            try:
                with open(file_path, "r") as f:
                    code = f.read()

                current_checksum = self._calculate_checksum(code)
                if current_checksum != metadata.checksum:
                    logger.warning(
                        f"Checksum mismatch for skill {skill_id}: "
                        f"expected {metadata.checksum}, got {current_checksum}"
                    )
                    return None

                module = self._load_module(skill_id, file_path)
                if module is None:
                    return None

                return SynthesizedTool(
                    id=skill_id,
                    name=metadata.name,
                    code=code,
                    description=metadata.description,
                    file_path=file_path,
                    is_safe=True,
                    module=module,
                )
            except Exception as e:
                logger.error(f"Failed to retrieve skill {skill_id}: {e}")
                return None

    def _load_module(self, skill_id: str, file_path: Path) -> Any:
        """Load a Python module from file."""
        try:
            spec = importlib.util.spec_from_file_location(f"tool_{skill_id}", file_path)
            if not spec or not spec.loader:
                raise ImportError(f"Failed to create module spec for {skill_id}")

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logger.error(f"Failed to load module for skill {skill_id}: {e}")
            return None

    def find_by_name(self, name: str) -> SynthesizedTool | None:
        """
        Find a skill by name.

        Args:
            name: The skill name to search for

        Returns:
            SynthesizedTool or None if not found
        """
        with self._lock:
            for skill_id, metadata in self._metadata_cache.items():
                if metadata.name == name:
                    return self.retrieve(skill_id)
        return None

    def find_by_tags(self, tags: list[str]) -> list[SynthesizedTool]:
        """
        Find skills matching any of the given tags.

        Args:
            tags: Tags to search for

        Returns:
            List of matching SynthesizedTool objects
        """
        results = []
        tags_lower = [t.lower() for t in tags]

        with self._lock:
            for skill_id, metadata in self._metadata_cache.items():
                skill_tags_lower = [t.lower() for t in metadata.tags]
                if any(t in skill_tags_lower for t in tags_lower):
                    tool = self.retrieve(skill_id)
                    if tool:
                        results.append(tool)

        return results

    def list_all(self) -> list[SkillMetadata]:
        """
        List all stored skills.

        Returns:
            List of SkillMetadata for all skills
        """
        with self._lock:
            return list(self._metadata_cache.values())

    def update_usage(self, skill_id: str) -> None:
        """
        Update usage statistics for a skill.

        Args:
            skill_id: The skill ID to update
        """
        with self._lock:
            metadata = self._metadata_cache.get(skill_id)
            if not metadata:
                return

            metadata.last_used = datetime.now().isoformat()
            metadata.use_count += 1

            meta_path = self._get_metadata_path(skill_id, metadata.category)
            if meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

            self._save_metadata_index()

    def delete(self, skill_id: str) -> bool:
        """
        Delete a skill from the cache.

        Args:
            skill_id: The skill ID to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            metadata = self._metadata_cache.get(skill_id)
            if not metadata:
                return False

            file_path = Path(metadata.file_path)
            meta_path = self._get_metadata_path(skill_id, metadata.category)

            try:
                if file_path.exists():
                    file_path.unlink()
                if meta_path.exists():
                    meta_path.unlink()

                del self._metadata_cache[skill_id]
                self._save_metadata_index()

                logger.info(f"Deleted skill {skill_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete skill {skill_id}: {e}")
                return False

    def cleanup_unused(self, days: int = 30) -> int:
        """
        Remove skills not used in the specified number of days.

        Args:
            days: Number of days of inactivity before cleanup

        Returns:
            Count of deleted skills
        """
        deleted_count = 0
        cutoff = datetime.now() - timedelta(days=days)

        with self._lock:
            to_delete = []

            for skill_id, metadata in self._metadata_cache.items():
                try:
                    last_used = datetime.fromisoformat(metadata.last_used)
                    if last_used < cutoff:
                        to_delete.append(skill_id)
                except Exception:
                    to_delete.append(skill_id)

        for skill_id in to_delete:
            if self.delete(skill_id):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} unused skills")

        return deleted_count

    def get_stats(self) -> SkillCacheStats:
        """
        Get statistics about the skill cache.

        Returns:
            SkillCacheStats with usage and storage information
        """
        with self._lock:
            total_size = 0
            categories: dict[str, int] = {cat: 0 for cat in CATEGORIES}

            for metadata in self._metadata_cache.values():
                categories[metadata.category] = categories.get(metadata.category, 0) + 1

                try:
                    file_path = Path(metadata.file_path)
                    if file_path.exists():
                        total_size += file_path.stat().st_size
                except Exception:
                    pass

            sorted_skills = sorted(
                self._metadata_cache.values(),
                key=lambda x: x.use_count,
                reverse=True,
            )
            most_used = sorted_skills[:5]

            return SkillCacheStats(
                total_skills=len(self._metadata_cache),
                total_size_bytes=total_size,
                categories=categories,
                most_used=most_used,
            )

    def validate_skill(self, skill_id: str) -> bool:
        """
        Validate that a skill can still be imported and executed.

        Args:
            skill_id: The skill ID to validate

        Returns:
            True if valid, False otherwise
        """
        tool = self.retrieve(skill_id)
        if tool is None:
            return False

        if tool.module is None:
            return False

        has_entry = hasattr(tool.module, "run") or hasattr(tool.module, "execute")
        return has_entry

    def migrate_skill(self, skill_id: str) -> bool:
        """
        Migrate a skill to the current schema version.

        Args:
            skill_id: The skill ID to migrate

        Returns:
            True if migration successful, False otherwise
        """
        with self._lock:
            metadata = self._metadata_cache.get(skill_id)
            if not metadata:
                return False

            if metadata.schema_version >= SCHEMA_VERSION:
                return True

            try:
                metadata.schema_version = SCHEMA_VERSION
                metadata.category = metadata.category or "other"

                meta_path = self._get_metadata_path(skill_id, metadata.category)
                with open(meta_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)

                self._save_metadata_index()
                logger.info(f"Migrated skill {skill_id} to version {SCHEMA_VERSION}")
                return True
            except Exception as e:
                logger.error(f"Failed to migrate skill {skill_id}: {e}")
                return False

    def repair_index(self) -> int:
        """
        Repair the metadata index by scanning actual files.

        Returns:
            Number of skills restored to index
        """
        restored = 0

        with self._lock:
            for category in CATEGORIES:
                category_path = self.cache_path / category
                if not category_path.exists():
                    continue

                for py_file in category_path.glob("tool_*.py"):
                    skill_id = py_file.stem.replace("tool_", "")
                    if skill_id in self._metadata_cache:
                        continue

                    meta_file = py_file.with_suffix(".json")
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r") as f:
                                data = json.load(f)
                                metadata = SkillMetadata.from_dict(data)
                                self._metadata_cache[skill_id] = metadata
                                restored += 1
                        except Exception as e:
                            logger.warning(f"Failed to restore skill {skill_id}: {e}")

            self._save_metadata_index()

        if restored > 0:
            logger.info(f"Restored {restored} skills to index")

        return restored


_cache_instance: SkillCache | None = None
_cache_lock = threading.Lock()


def get_skill_cache(cache_path: str = ".gaap/skills") -> SkillCache:
    """Get singleton SkillCache instance."""
    global _cache_instance

    with _cache_lock:
        if _cache_instance is None:
            _cache_instance = SkillCache(cache_path=cache_path)
        return _cache_instance
