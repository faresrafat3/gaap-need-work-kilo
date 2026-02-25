"""
Knowledge Configuration - Settings for Repository Ingestion
============================================================

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md

Configuration for the Library Eater - repository ingestion engine
that learns new libraries by reading their source code.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class Language(Enum):
    """Supported programming languages"""

    PYTHON = auto()
    JAVASCRIPT = auto()
    TYPESCRIPT = auto()
    UNKNOWN = auto()


@dataclass
class KnowledgeConfig:
    """
    Configuration for knowledge ingestion.

    Controls repository cloning, parsing, and storage behavior.
    """

    enabled: bool = True

    storage_path: str = ".gaap/knowledge"
    temp_clone_path: str = ".gaap/temp/repos"

    max_file_size_mb: int = 10
    max_files_per_repo: int = 1000
    max_total_size_mb: int = 100

    supported_languages: list[str] = field(
        default_factory=lambda: ["python", "javascript", "typescript"]
    )

    include_patterns: list[str] = field(
        default_factory=lambda: [
            "**/*.py",
            "**/*.js",
            "**/*.ts",
            "**/*.tsx",
        ]
    )

    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "**/test_*.py",
            "**/*_test.py",
            "**/__pycache__/**",
            "**/node_modules/**",
            "**/.venv/**",
            "**/venv/**",
            "**/dist/**",
            "**/build/**",
            "**/.git/**",
            "**/migrations/**",
        ]
    )

    test_directories: list[str] = field(
        default_factory=lambda: ["tests", "test", "tests_", "__tests__"]
    )

    example_directories: list[str] = field(
        default_factory=lambda: ["examples", "example", "docs/examples", "demos"]
    )

    top_functions_count: int = 10
    common_patterns_count: int = 5

    use_tree_sitter: bool = True
    fallback_to_ast: bool = True

    store_in_vector_memory: bool = True
    vector_namespace: str = "learned_libraries"

    clone_depth: int = 1
    clone_timeout_seconds: int = 120

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "storage_path": self.storage_path,
            "temp_clone_path": self.temp_clone_path,
            "max_file_size_mb": self.max_file_size_mb,
            "max_files_per_repo": self.max_files_per_repo,
            "max_total_size_mb": self.max_total_size_mb,
            "supported_languages": self.supported_languages,
            "include_patterns": self.include_patterns,
            "exclude_patterns": self.exclude_patterns,
            "test_directories": self.test_directories,
            "example_directories": self.example_directories,
            "top_functions_count": self.top_functions_count,
            "common_patterns_count": self.common_patterns_count,
            "use_tree_sitter": self.use_tree_sitter,
            "fallback_to_ast": self.fallback_to_ast,
            "store_in_vector_memory": self.store_in_vector_memory,
            "vector_namespace": self.vector_namespace,
            "clone_depth": self.clone_depth,
            "clone_timeout_seconds": self.clone_timeout_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeConfig":
        return cls(
            enabled=data.get("enabled", True),
            storage_path=data.get("storage_path", ".gaap/knowledge"),
            temp_clone_path=data.get("temp_clone_path", ".gaap/temp/repos"),
            max_file_size_mb=data.get("max_file_size_mb", 10),
            max_files_per_repo=data.get("max_files_per_repo", 1000),
            max_total_size_mb=data.get("max_total_size_mb", 100),
            supported_languages=data.get(
                "supported_languages", ["python", "javascript", "typescript"]
            ),
            include_patterns=data.get("include_patterns", []),
            exclude_patterns=data.get("exclude_patterns", []),
            test_directories=data.get("test_directories", ["tests", "test"]),
            example_directories=data.get("example_directories", ["examples", "example"]),
            top_functions_count=data.get("top_functions_count", 10),
            common_patterns_count=data.get("common_patterns_count", 5),
            use_tree_sitter=data.get("use_tree_sitter", True),
            fallback_to_ast=data.get("fallback_to_ast", True),
            store_in_vector_memory=data.get("store_in_vector_memory", True),
            vector_namespace=data.get("vector_namespace", "learned_libraries"),
            clone_depth=data.get("clone_depth", 1),
            clone_timeout_seconds=data.get("clone_timeout_seconds", 120),
        )

    @classmethod
    def default(cls) -> "KnowledgeConfig":
        return cls()

    @classmethod
    def fast(cls) -> "KnowledgeConfig":
        """Fast config - minimal parsing"""
        return cls(
            max_file_size_mb=5,
            max_files_per_repo=100,
            store_in_vector_memory=False,
            clone_depth=1,
        )

    @classmethod
    def deep(cls) -> "KnowledgeConfig":
        """Deep config - comprehensive analysis"""
        return cls(
            max_files_per_repo=5000,
            max_total_size_mb=500,
            top_functions_count=20,
            common_patterns_count=10,
            clone_depth=0,
        )

    def get_language_from_extension(self, ext: str) -> Language:
        """Get language from file extension"""
        ext = ext.lower().lstrip(".")
        mapping = {
            "py": Language.PYTHON,
            "js": Language.JAVASCRIPT,
            "ts": Language.TYPESCRIPT,
            "tsx": Language.TYPESCRIPT,
        }
        return mapping.get(ext, Language.UNKNOWN)


def create_knowledge_config(preset: str = "default", **kwargs: Any) -> KnowledgeConfig:
    """Create a KnowledgeConfig with optional preset."""
    presets = {
        "default": KnowledgeConfig.default,
        "fast": KnowledgeConfig.fast,
        "deep": KnowledgeConfig.deep,
    }

    config = presets.get(preset, KnowledgeConfig.default)()

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
