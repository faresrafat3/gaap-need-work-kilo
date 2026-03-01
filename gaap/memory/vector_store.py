"""
Vector Store Implementation (Evolution 2026)
Implements: docs/evolution_plan_2026/25_MEMORY_AUDIT_SPEC.md

Focus: Semantic retrieval using ChromaDB with fallback embedding.
"""

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("gaap.memory.vector")

CHROMADB_AVAILABLE = False
_chromadb = None
_Settings = None
try:
    import chromadb as _chromadb_module
    from chromadb.config import Settings as _Settings_class

    _chromadb = _chromadb_module
    _Settings = _Settings_class
    CHROMADB_AVAILABLE = True
except ImportError:
    pass


@dataclass
class VectorEntry:
    """A single unit of semantic memory."""

    id: str
    content: str
    metadata: dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class SimpleEmbeddingFunction:
    """
    Fallback embedding function using hash-based pseudo-embeddings.
    Used when sentence_transformers is not available.
    """

    name = "simple_hash_embedding"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        """Generate pseudo-embeddings from text hashes."""
        embeddings = []
        for text in texts:
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            embedding = [int(text_hash[i : i + 8], 16) / 0xFFFFFFFF for i in range(0, 64, 8)]
            embedding = [e * 2 - 1 for e in embedding]
            while len(embedding) < 384:
                embedding.extend(embedding[: min(384 - len(embedding), len(embedding))])
            embeddings.append(embedding[:384])
        return embeddings


class VectorStore:
    """
    Persistent Vector Database wrapper around ChromaDB.
    Handles embedding generation and semantic search with fallback.
    """

    def __init__(
        self,
        collection_name: str = "gaap_memory",
        persist_dir: str = ".gaap/memory/vector_db",
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._available = False
        self._fallback_store: dict[str, tuple[str, dict[str, Any]]] = {}

        if not CHROMADB_AVAILABLE:
            logger.warning(
                "ChromaDB not installed. Using disk-based fallback. "
                "Install chromadb for better performance: pip install chromadb"
            )
            self._fallback_store_path = Path(".gaap/vector_fallback.json")
            self._fallback_store_path.parent.mkdir(parents=True, exist_ok=True)
            self._fallback_store = self._load_fallback_store()
            return

        os.makedirs(persist_dir, exist_ok=True)

        try:
            self.client = _chromadb.PersistentClient(  # type: ignore[union-attr]
                path=persist_dir,
                settings=_Settings(allow_reset=True),  # type: ignore[misc]
            )

            self.embedding_fn = self._get_embedding_function()

            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
            logger.debug(f"VectorStore initialized at {persist_dir}")

        except Exception as e:
            logger.warning(f"VectorStore unavailable: {e}. Using in-memory fallback.")
            self._available = False

    def _get_embedding_function(self) -> Any:
        """Get embedding function with fallback."""
        try:
            from chromadb.utils import embedding_functions

            ef = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            logger.debug("Using SentenceTransformer embeddings")
            return ef
        except Exception as e:
            logger.warning(f"SentenceTransformer unavailable: {e}. Using hash-based fallback.")
            return SimpleEmbeddingFunction()

    @property
    def available(self) -> bool:
        return self._available

    def _load_fallback_store(self) -> dict:
        if self._fallback_store_path.exists():
            with open(self._fallback_store_path) as f:
                return json.load(f)
        return {}

    def _save_fallback_store(self) -> None:
        with open(self._fallback_store_path, "w") as f:
            json.dump(self._fallback_store, f)

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> str:
        """Add a text document to the vector store."""
        if not content.strip():
            return ""

        if not entry_id:
            entry_id = str(uuid.uuid4())

        meta = metadata or {}
        meta["timestamp"] = datetime.now().isoformat()

        if self._available:
            try:
                self.collection.add(documents=[content], metadatas=[meta], ids=[entry_id])
                return entry_id
            except Exception as e:
                logger.error(f"Error adding to VectorStore: {e}")
                return ""
        else:
            self._fallback_store[entry_id] = (content, meta)
            self._save_fallback_store()
            return entry_id

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_meta: dict[str, Any] | None = None,
    ) -> list[VectorEntry]:
        """Semantic search for relevant documents."""
        if self._available:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_meta,
                )

                entries = []
                if results["ids"] and results["ids"][0]:
                    ids = results["ids"][0]
                    documents = results["documents"][0] if results["documents"] else []
                    metadatas = results["metadatas"][0] if results["metadatas"] else []

                    for i, doc_id in enumerate(ids):
                        meta_raw = metadatas[i] if i < len(metadatas) else {}
                        meta: dict[str, Any] = dict(meta_raw) if meta_raw else {}
                        entries.append(
                            VectorEntry(
                                id=doc_id,
                                content=documents[i] if i < len(documents) else "",
                                metadata=meta,
                            )
                        )
                return entries

            except Exception as e:
                logger.error(f"Error searching VectorStore: {e}")
                return []
        else:
            results = []
            query_lower = query.lower()
            for entry_id, (content, meta) in self._fallback_store.items():
                if query_lower in content.lower():
                    if filter_meta:
                        if not all(meta.get(k) == v for k, v in filter_meta.items()):
                            continue
                    results.append(VectorEntry(id=entry_id, content=content, metadata=meta))
                    if len(results) >= n_results:
                        break
            return results

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID."""
        if self._available:
            try:
                self.collection.delete(ids=[entry_id])
                return True
            except Exception as e:
                logger.error(f"Error deleting from VectorStore: {e}")
                return False
        else:
            if entry_id in self._fallback_store:
                del self._fallback_store[entry_id]
                self._save_fallback_store()
                return True
            return False

    def count(self) -> int:
        """Return total count of embeddings."""
        if self._available:
            count: int = self.collection.count()
            return count
        return len(self._fallback_store)

    def reset(self) -> None:
        """Wipe the entire database."""
        if self._available:
            try:
                self.client.reset()
                logger.warning("VectorStore has been reset.")
            except Exception as e:
                logger.error(f"Error resetting VectorStore: {e}")
        else:
            self._fallback_store.clear()
            self._save_fallback_store()


_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Get singleton VectorStore instance."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance
