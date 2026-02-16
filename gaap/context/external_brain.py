# External Brain
import hashlib
import json
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BrainIndex:
    """فهرس في الدماغ الخارجي"""

    id: str
    content: str
    source: str
    content_type: str  # file, function, class, doc, pattern
    token_count: int
    embedding: list[float] | None = None  # للتضمين الدلالي
    keywords: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "content_type": self.content_type,
            "token_count": self.token_count,
            "keywords": self.keywords,
            "access_count": self.access_count,
        }


@dataclass
class SearchResult:
    """نتيجة بحث"""

    id: str
    content: str
    source: str
    relevance_score: float
    token_count: int
    content_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Text Embedding (Simple TF-IDF)
# =============================================================================


class SimpleEmbedding:
    """
    تضمين نصي مبسط (TF-IDF-like)
    للتشابه الدلالي دون نماذج خارجية
    """

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.doc_count = 0

    def fit(self, documents: list[str]) -> None:
        """بناء المفردات"""
        # بناء المفردات
        word_doc_count: dict[str, int] = defaultdict(int)

        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] += 1

        self.doc_count = len(documents)

        # حساب IDF
        for word, count in word_doc_count.items():
            self.idf[word] = math.log(self.doc_count / (1 + count))
            if word not in self.vocabulary:
                self.vocabulary[word] = len(self.vocabulary)

    def encode(self, text: str) -> list[float]:
        """تشفير النص"""
        words = self._tokenize(text)
        word_count: dict[str, int] = defaultdict(int)

        for word in words:
            word_count[word] += 1

        total_words = len(words) if words else 1

        # متجه TF-IDF
        vector = [0.0] * len(self.vocabulary)

        for word, count in word_count.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf.get(word, 1.0)
                vector[self.vocabulary[word]] = tf * idf

        return vector

    def _tokenize(self, text: str) -> list[str]:
        """تحويل النص لكلمات"""
        # تحويل للصيغة الصغيرة
        text = text.lower()
        # استخراج الكلمات
        words = re.findall(r"\b[a-z_][a-z0-9_]*\b", text)
        # إزالة الكلمات الشائعة
        stopwords = {
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
            "this",
            "that",
            "these",
            "those",
            "and",
            "but",
            "or",
            "if",
            "else",
            "then",
            "so",
            "than",
            "too",
            "very",
            "just",
            "now",
        }

        return [w for w in words if w not in stopwords and len(w) > 2]

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """حساب التشابه"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


# =============================================================================
# External Brain
# =============================================================================


class ExternalBrain:
    """
    الدماغ الخارجي - ذاكرة منظمة للوكلاء

    الميزات:
    - فهرسة الملفات والأكواد
    - بحث دلالي سريع
    - تخزين الأنماط والدروس
    - استرجاع ذكي للسياق
    """

    def __init__(
        self, project_path: str, storage_path: str | None = None, max_index_size: int = 100000
    ):
        self.project_path = project_path
        self.storage_path = storage_path or os.path.join(project_path, ".gaap", "brain")
        self.max_index_size = max_index_size

        self._logger = logging.getLogger("gaap.context.brain")

        # الفهارس
        self._indices: dict[str, BrainIndex] = {}
        self._keyword_index: dict[str, list[str]] = defaultdict(list)
        self._source_index: dict[str, list[str]] = defaultdict(list)

        # التضمين
        self._embedding = SimpleEmbedding()
        self._is_fitted = False

        # التخزين
        os.makedirs(self.storage_path, exist_ok=True)

    # =========================================================================
    # Indexing Methods
    # =========================================================================

    async def index_project(self, force: bool = False) -> int:
        """فهرسة المشروع"""
        # محاولة تحميل فهرس محفوظ
        if not force and await self._load_index():
            self._logger.info(f"Loaded existing index: {len(self._indices)} items")
            return len(self._indices)

        self._logger.info(f"Indexing project: {self.project_path}")

        count = 0
        documents = []

        for root, dirs, files in os.walk(self.project_path):
            # تجاهل المجلدات المخفية
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "__pycache__", "venv", ".git", ".gaap"]
            ]

            for file in files:
                if file.startswith("."):
                    continue

                file_path = os.path.join(root, file)

                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()

                    # فهرسة الملف
                    index = self._create_file_index(file_path, content)
                    self._indices[index.id] = index
                    documents.append(content)

                    # فهرسة المكونات الفرعية
                    sub_indices = self._index_file_components(file_path, content)
                    for sub_index in sub_indices:
                        self._indices[sub_index.id] = sub_index
                        documents.append(sub_index.content)

                    count += 1 + len(sub_indices)

                except Exception as e:
                    self._logger.debug(f"Could not index {file_path}: {e}")

        # بناء التضمين
        self._embedding.fit(documents)
        self._is_fitted = True

        # بناء فهرس الكلمات المفتاحية
        self._build_keyword_index()

        # حفظ الفهرس
        await self._save_index()

        self._logger.info(f"Indexed {count} items from {len(self._indices)} entries")
        return count

    def _create_file_index(self, file_path: str, content: str) -> BrainIndex:
        """إنشاء فهرس ملف"""
        index_id = self._generate_id(file_path)
        keywords = self._extract_keywords(content)

        return BrainIndex(
            id=index_id,
            content=content[:50000],  # حد الحجم
            source=file_path,
            content_type="file",
            token_count=len(content.split()) * 1.5,
            keywords=keywords,
            metadata={
                "file_name": os.path.basename(file_path),
                "ext": os.path.splitext(file_path)[1],
            },
        )

    def _index_file_components(self, file_path: str, content: str) -> list[BrainIndex]:
        """فهرسة مكونات الملف"""
        indices: list[BrainIndex] = []
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".py":
            indices = self._index_python_components(file_path, content)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            indices = self._index_js_components(file_path, content)

        return indices

    def _index_python_components(self, file_path: str, content: str) -> list[BrainIndex]:
        """فهرسة مكونات Python"""
        import ast

        indices: list[BrainIndex] = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    indices.append(
                        self._create_component_index(file_path, content, node, "class", node.name)
                    )
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    indices.append(
                        self._create_component_index(
                            file_path, content, node, "function", node.name
                        )
                    )
        except SyntaxError:
            pass

        return indices

    def _index_js_components(self, file_path: str, content: str) -> list[BrainIndex]:
        """فهرسة مكونات JavaScript"""
        indices: list[BrainIndex] = []

        # استخراج الفئات
        class_pattern = re.compile(r"class\s+(\w+)", re.MULTILINE)
        for match in class_pattern.finditer(content):
            indices.append(
                BrainIndex(
                    id=self._generate_id(f"{file_path}:class:{match.group(1)}"),
                    content=content[max(0, match.start() - 100) : match.start() + 1000],
                    source=file_path,
                    content_type="class",
                    token_count=500,
                    keywords=[match.group(1).lower()],
                )
            )

        # استخراج الدوال
        func_pattern = re.compile(r"function\s+(\w+)", re.MULTILINE)
        for match in func_pattern.finditer(content):
            indices.append(
                BrainIndex(
                    id=self._generate_id(f"{file_path}:func:{match.group(1)}"),
                    content=content[max(0, match.start() - 50) : match.start() + 500],
                    source=file_path,
                    content_type="function",
                    token_count=200,
                    keywords=[match.group(1).lower()],
                )
            )

        return indices

    def _create_component_index(
        self, file_path: str, content: str, node, content_type: str, name: str
    ) -> BrainIndex:
        """إنشاء فهرس مكون"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        lines = content.split("\n")
        component_content = "\n".join(lines[start_line - 1 : end_line])

        import ast

        docstring = ast.get_docstring(node) if hasattr(node, "body") else ""

        return BrainIndex(
            id=self._generate_id(f"{file_path}:{content_type}:{name}"),
            content=component_content[:10000],
            source=file_path,
            content_type=content_type,
            token_count=len(component_content.split()) * 1.5,
            keywords=self._extract_keywords(component_content),
            metadata={
                "name": name,
                "docstring": docstring[:500] if docstring else "",
                "start_line": start_line,
                "end_line": end_line,
            },
        )

    # =========================================================================
    # Search Methods
    # =========================================================================

    async def search(
        self, query: str, limit: int = 10, content_type: str | None = None, min_score: float = 0.1
    ) -> list[SearchResult]:
        """
        البحث في الدماغ الخارجي

        Args:
            query: نص البحث
            limit: الحد الأقصى للنتائج
            content_type: نوع المحتوى (file, function, class)
            min_score: الحد الأدنى للصلة

        Returns:
            قائمة النتائج
        """
        if not self._indices:
            await self.index_project()

        results: list[SearchResult] = []

        # البحث بالكلمات المفتاحية
        query_keywords = self._embedding._tokenize(query)
        keyword_scores: dict[str, float] = defaultdict(float)

        for keyword in query_keywords:
            if keyword in self._keyword_index:
                for index_id in self._keyword_index[keyword]:
                    keyword_scores[index_id] += 1.0

        # البحث الدلالي
        query_embedding = self._embedding.encode(query) if self._is_fitted else []

        # حساب الدرجات
        for index_id, index in self._indices.items():
            # فلترة بالنوع
            if content_type and index.content_type != content_type:
                continue

            score = 0.0

            # درجة الكلمات المفتاحية
            keyword_score = keyword_scores.get(index_id, 0) / max(len(query_keywords), 1)

            # درجة التشابه الدلالي
            semantic_score = 0.0
            if self._is_fitted and index.embedding:
                index_embedding = self._embedding.encode(index.content)
                semantic_score = SimpleEmbedding.cosine_similarity(query_embedding, index_embedding)

            # درجة تطابق النص
            text_score = 0.0
            query_lower = query.lower()
            if query_lower in index.content.lower():
                text_score = 0.5
            if query_lower in index.source.lower():
                text_score += 0.3

            # الدرجة المجمعة
            score = keyword_score * 0.4 + semantic_score * 0.3 + text_score * 0.3

            # تعديل بنوع المحتوى
            if index.content_type == "function":
                score *= 1.1
            elif index.content_type == "class":
                score *= 1.0
            elif index.content_type == "file":
                score *= 0.8

            if score >= min_score:
                results.append(
                    SearchResult(
                        id=index_id,
                        content=index.content,
                        source=index.source,
                        relevance_score=score,
                        token_count=index.token_count,
                        content_type=index.content_type,
                        metadata=index.metadata,
                    )
                )

        # ترتيب النتائج
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        # تحديث عداد الوصول
        for result in results[:limit]:
            if result.id in self._indices:
                self._indices[result.id].access_count += 1

        return results[:limit]

    async def get_by_id(self, index_id: str) -> BrainIndex | None:
        """الحصول على فهرس بالمعرف"""
        return self._indices.get(index_id)

    async def get_by_source(self, source: str) -> list[BrainIndex]:
        """الحصول على فهارس بمصدر"""
        return [
            self._indices[iid] for iid in self._source_index.get(source, []) if iid in self._indices
        ]

    async def get_recent(self, limit: int = 20) -> list[BrainIndex]:
        """الحصول على أحدث الفهارس"""
        sorted_indices = sorted(self._indices.values(), key=lambda x: x.created_at, reverse=True)
        return sorted_indices[:limit]

    async def get_most_accessed(self, limit: int = 20) -> list[BrainIndex]:
        """الحصول على الأكثر وصولاً"""
        sorted_indices = sorted(self._indices.values(), key=lambda x: x.access_count, reverse=True)
        return sorted_indices[:limit]

    # =========================================================================
    # Knowledge Storage
    # =========================================================================

    async def store_knowledge(
        self, key: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """تخزين معرفة مكتسبة"""
        index = BrainIndex(
            id=self._generate_id(f"knowledge:{key}"),
            content=content,
            source="knowledge_store",
            content_type="pattern",
            token_count=len(content.split()) * 1.5,
            keywords=self._extract_keywords(content),
            metadata=metadata or {},
        )

        self._indices[index.id] = index
        self._update_keyword_index(index)

        # حفظ
        await self._save_index()

    async def get_knowledge(self, key: str) -> BrainIndex | None:
        """الحصول على معرفة مخزنة"""
        index_id = self._generate_id(f"knowledge:{key}")
        return self._indices.get(index_id)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _generate_id(self, text: str) -> str:
        """توليد معرف"""
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _extract_keywords(self, content: str) -> list[str]:
        """استخراج الكلمات المفتاحية"""
        words = self._embedding._tokenize(content)

        # حساب التكرار
        word_count: dict[str, int] = defaultdict(int)
        for word in words:
            word_count[word] += 1

        # اختيار الأكثر تكراراً
        sorted_words = sorted(word_count.items(), key=lambda x: -x[1])
        return [w for w, _ in sorted_words[:20]]

    def _build_keyword_index(self) -> None:
        """بناء فهرس الكلمات المفتاحية"""
        self._keyword_index.clear()
        self._source_index.clear()

        for index_id, index in self._indices.items():
            self._update_keyword_index(index)

            # فهرس المصدر
            self._source_index[index.source].append(index_id)

    def _update_keyword_index(self, index: BrainIndex) -> None:
        """تحديث فهرس الكلمات المفتاحية"""
        for keyword in index.keywords:
            if index.id not in self._keyword_index[keyword]:
                self._keyword_index[keyword].append(index.id)

    async def _save_index(self) -> None:
        """حفظ الفهرس"""
        try:
            data = {
                "indices": {k: v.to_dict() for k, v in self._indices.items()},
                "keyword_index": dict(self._keyword_index),
            }

            index_path = os.path.join(self.storage_path, "brain_index.json")
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self._logger.warning(f"Could not save index: {e}")

    async def _load_index(self) -> bool:
        """تحميل الفهرس"""
        try:
            index_path = os.path.join(self.storage_path, "brain_index.json")

            if not os.path.exists(index_path):
                return False

            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)

            self._indices = {
                k: BrainIndex(
                    id=k,
                    content="",  # لا نحمل المحتوى الكامل
                    source=v.get("source", ""),
                    content_type=v.get("content_type", ""),
                    token_count=v.get("token_count", 0),
                    keywords=v.get("keywords", []),
                    metadata=v.get("metadata", {}),
                    access_count=v.get("access_count", 0),
                )
                for k, v in data.get("indices", {}).items()
            }

            self._keyword_index = defaultdict(list, data.get("keyword_index", {}))

            return True

        except Exception as e:
            self._logger.warning(f"Could not load index: {e}")
            return False

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات الدماغ"""
        content_types: dict[str, int] = defaultdict(int)
        for index in self._indices.values():
            content_types[index.content_type] += 1

        return {
            "total_indices": len(self._indices),
            "content_types": dict(content_types),
            "keyword_count": len(self._keyword_index),
            "is_fitted": self._is_fitted,
        }

    def clear(self) -> None:
        """مسح الدماغ"""
        self._indices.clear()
        self._keyword_index.clear()
        self._source_index.clear()
        self._is_fitted = False
