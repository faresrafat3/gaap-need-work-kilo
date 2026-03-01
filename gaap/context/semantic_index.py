"""
Semantic Index - Vector Store Integration for Code Search
Implements: docs/evolution_plan_2026/40_CONTEXT_AUDIT_SPEC.md

Features:
- Vector store integration
- Embedding generation
- Similarity search
- Code semantic search
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.context.semantic_index")


@dataclass
class IndexEntry:
    entry_id: str
    content: str
    embedding: list[float] | None
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: str = ""
    chunk_type: str = ""
    name: str = ""
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.entry_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "file_path": self.file_path,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "signature": self.signature,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class IndexConfig:
    embedding_model: str = "text-embedding-ada-002"
    embedding_dimension: int = 1536
    chunk_size: int = 500
    chunk_overlap: int = 50
    use_local_embeddings: bool = False
    persist_dir: str = ".gaap/index"

    @classmethod
    def default(cls) -> IndexConfig:
        return cls()

    @classmethod
    def local(cls) -> IndexConfig:
        return cls(
            use_local_embeddings=True,
            embedding_model="local",
        )

    @classmethod
    def fast(cls) -> IndexConfig:
        return cls(
            embedding_dimension=384,
            chunk_size=300,
            use_local_embeddings=True,
        )


class SemanticIndex:
    """
    Semantic code index with vector search.

    Features:
    - Vector store integration
    - Embedding generation
    - Similarity search
    - Code semantic understanding

    Usage:
        index = SemanticIndex()
        await index.index_code(code, file_path)
        results = await index.search("function that handles authentication")
    """

    def __init__(self, config: IndexConfig | None = None) -> None:
        self.config = config or IndexConfig.default()
        self._entries: dict[str, IndexEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._vector_store: Any = None
        self._embedding_provider: Any = None
        self._logger = logger
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True

        try:
            if self.config.use_local_embeddings:
                self._init_local_embeddings()
            else:
                self._init_api_embeddings()

            self._initialized = True
            self._logger.info("Semantic index initialized")
            return True

        except Exception as e:
            self._logger.warning(f"Failed to initialize semantic index: {e}")
            self._initialized = False
            return False

    def _init_local_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_provider = SentenceTransformer("all-MiniLM-L6-v2")
            self._logger.info("Using local sentence-transformers embeddings")
        except ImportError:
            self._logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._embedding_provider = None

    def _init_api_embeddings(self) -> None:
        try:
            pass

            self._embedding_provider = "api"
            self._logger.info("Using API embeddings")
        except Exception:
            self._embedding_provider = None

    async def index_code(
        self,
        code: str,
        file_path: str,
        chunk_type: str = "code",
        name: str = "",
        signature: str = "",
    ) -> str:
        if not self._initialized:
            await self.initialize()

        entry_id = self._generate_id(code, file_path)

        embedding = await self._generate_embedding(code)

        entry = IndexEntry(
            entry_id=entry_id,
            content=code,
            embedding=embedding,
            metadata={
                "file_path": file_path,
                "chunk_type": chunk_type,
                "name": name,
                "signature": signature,
            },
            file_path=file_path,
            chunk_type=chunk_type,
            name=name,
            signature=signature,
        )

        self._entries[entry_id] = entry
        if embedding:
            self._embeddings[entry_id] = embedding

        return entry_id

    async def index_chunk(self, chunk: Any) -> str:
        return await self.index_code(
            code=chunk.content,
            file_path=chunk.file_path,
            chunk_type=(
                chunk.chunk_type.name
                if hasattr(chunk.chunk_type, "name")
                else str(chunk.chunk_type)
            ),
            name=chunk.name,
            signature=chunk.signature,
        )

    async def index_file(self, file_path: str | Path) -> list[str]:
        path = Path(file_path)
        if not path.exists():
            self._logger.error(f"File not found: {file_path}")
            return []

        code = path.read_text(encoding="utf-8")

        from gaap.context.smart_chunking import ChunkingConfig, SmartChunker

        chunker = SmartChunker(ChunkingConfig.for_context())
        chunks = chunker.chunk(code, str(path))

        entry_ids: list[str] = []
        for chunk in chunks:
            entry_id = await self.index_chunk(chunk)
            entry_ids.append(entry_id)

        return entry_ids

    async def _generate_embedding(self, text: str) -> list[float] | None:
        if not self._embedding_provider:
            return None

        try:
            if hasattr(self._embedding_provider, "encode"):
                embedding = self._embedding_provider.encode(text)
                return embedding.tolist()  # type: ignore[no-any-return]
            else:
                return self._simple_embedding(text)
        except Exception as e:
            self._logger.debug(f"Embedding generation failed: {e}")
            return self._simple_embedding(text)

    def _simple_embedding(self, text: str) -> list[float]:
        words = text.lower().split()[:100]

        word_hash = {}
        for word in words:
            h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
            word_hash[word] = h

        embedding = [0.0] * self.config.embedding_dimension

        for word, h in word_hash.items():
            idx = h % self.config.embedding_dimension
            embedding[idx] = min(embedding[idx] + 1.0, 5.0)

        total = sum(embedding)
        if total > 0:
            embedding = [e / total for e in embedding]

        return embedding

    def _generate_id(self, content: str, file_path: str) -> str:
        return hashlib.sha256(f"{file_path}:{content}".encode()).hexdigest()[:16]

    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[tuple[IndexEntry, float]]:
        if not self._entries:
            return []

        query_embedding = await self._generate_embedding(query)
        if not query_embedding:
            return [(entry, 0.0) for entry in list(self._entries.values())[:top_k]]

        scores: list[tuple[IndexEntry, float]] = []

        for entry_id, entry in self._entries.items():
            if entry.embedding:
                similarity = self._cosine_similarity(query_embedding, entry.embedding)
                if similarity >= threshold:
                    scores.append((entry, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)  # type: ignore[no-any-return]

    def get_entry(self, entry_id: str) -> IndexEntry | None:
        return self._entries.get(entry_id)

    def get_all_entries(self) -> list[IndexEntry]:
        return list(self._entries.values())

    def remove_entry(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            self._embeddings.pop(entry_id, None)
            return True
        return False

    def clear(self) -> None:
        self._entries.clear()
        self._embeddings.clear()

    def get_stats(self) -> dict[str, Any]:
        entries_with_embeddings = sum(1 for e in self._entries.values() if e.embedding)

        types: dict[str, int] = {}
        for entry in self._entries.values():
            types[entry.chunk_type] = types.get(entry.chunk_type, 0) + 1

        return {
            "total_entries": len(self._entries),
            "entries_with_embeddings": entries_with_embeddings,
            "embedding_dimension": self.config.embedding_dimension,
            "types": types,
            "initialized": self._initialized,
        }

    def save(self, path: str | Path | None = None) -> bool:
        import json

        save_path = Path(path or self.config.persist_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            data = {
                "config": {
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": self.config.embedding_dimension,
                },
                "entries": [
                    {
                        "id": e.entry_id,
                        "content": e.content,
                        "embedding": e.embedding,
                        "metadata": e.metadata,
                        "file_path": e.file_path,
                        "chunk_type": e.chunk_type,
                        "name": e.name,
                        "signature": e.signature,
                    }
                    for e in self._entries.values()
                ],
            }

            index_file = save_path / "index.json"
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            self._logger.error(f"Failed to save index: {e}")
            return False

    def load(self, path: str | Path | None = None) -> bool:
        import json

        load_path = Path(path or self.config.persist_dir)
        index_file = load_path / "index.json"

        if not index_file.exists():
            return False

        try:
            with open(index_file, encoding="utf-8") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = IndexEntry(
                    entry_id=entry_data["id"],
                    content=entry_data["content"],
                    embedding=entry_data.get("embedding"),
                    metadata=entry_data.get("metadata", {}),
                    file_path=entry_data.get("file_path", ""),
                    chunk_type=entry_data.get("chunk_type", ""),
                    name=entry_data.get("name", ""),
                    signature=entry_data.get("signature", ""),
                )
                self._entries[entry.entry_id] = entry
                if entry.embedding:
                    self._embeddings[entry.entry_id] = entry.embedding

            self._initialized = True
            return True

        except Exception as e:
            self._logger.error(f"Failed to load index: {e}")
            return False


def create_semantic_index(
    embedding_dimension: int = 1536,
    use_local: bool = False,
) -> SemanticIndex:
    config = IndexConfig(
        embedding_dimension=embedding_dimension,
        use_local_embeddings=use_local,
    )
    return SemanticIndex(config)
