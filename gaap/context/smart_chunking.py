# Smart Chunking
import ast
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class ChunkType(Enum):
    """أنواع القطع"""

    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    MODULE_DOC = "module_doc"
    IMPORTS = "imports"
    LOGICAL_BLOCK = "logical_block"
    INTERFACE = "interface"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CodeChunk:
    """قطعة كود"""

    id: str
    chunk_type: ChunkType
    name: str
    content: str
    token_count: int
    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    dependencies: list[str] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    signature: str = ""
    docstring: str = ""
    importance: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.chunk_type.value,
            "name": self.name,
            "token_count": self.token_count,
            "file_path": self.file_path,
            "lines": f"{self.start_line}-{self.end_line}",
            "dependencies": len(self.dependencies),
        }


# =============================================================================
# Smart Chunker
# =============================================================================


class SmartChunker:
    """
    المقسم الذكي للكود

    الميزات:
    - تقسيم يحترم البنية البرمجية
    - استخراج الواجهات
    - تتبع التبعيات
    - تقليل السياق المطلوب
    """

    # الحدود
    MAX_CHUNK_TOKENS = 2000
    MIN_CHUNK_TOKENS = 50

    def __init__(self, project_path: str) -> None:
        self.project_path = project_path
        self._logger = logging.getLogger("gaap.context.chunker")
        self._chunk_cache: dict[str, list[CodeChunk]] = {}

    def _estimate_tokens(self, text: str) -> int:
        """تقدير عدد الرموز من النص"""
        return int(len(text.split()) * 1.5)

    # =========================================================================
    # Chunking Methods
    # =========================================================================

    async def chunk_file(
        self, file_path: str, max_chunk_size: int | None = None
    ) -> list[CodeChunk]:
        """تقسيم ملف إلى قطع"""
        max_chunk_size = max_chunk_size or self.MAX_CHUNK_TOKENS

        # التحقق من التخزين المؤقت
        cache_key = f"{file_path}:{max_chunk_size}"
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]

        # قراءة الملف
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self._logger.warning(f"Could not read {file_path}: {e}")
            return []

        ext = os.path.splitext(file_path)[1].lower()

        # تقسيم حسب نوع الملف
        if ext == ".py":
            chunks = self._chunk_python(file_path, content, max_chunk_size)
        elif ext in (".js", ".ts", ".jsx", ".tsx"):
            chunks = self._chunk_javascript(file_path, content, max_chunk_size)
        else:
            chunks = self._chunk_generic(file_path, content, max_chunk_size)

        # تخزين مؤقت
        self._chunk_cache[cache_key] = chunks

        return chunks

    def _chunk_python(self, file_path: str, content: str, max_chunk_size: int) -> list[CodeChunk]:
        """تقسيم ملف Python"""
        chunks: list[CodeChunk] = []
        lines = content.split("\n")

        # تجربة تحليل AST
        try:
            tree = ast.parse(content)

            # استخراج التوثيق
            module_doc = ast.get_docstring(tree)
            if module_doc:
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, "module_doc"),
                        chunk_type=ChunkType.MODULE_DOC,
                        name="module_docstring",
                        content=module_doc,
                        token_count=self._estimate_tokens(module_doc),
                        file_path=file_path,
                        start_line=1,
                        end_line=module_doc.count("\n") + 1,
                    )
                )

            # استخراج imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))

            if imports:
                import_content = "\n".join(imports)
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, "imports"),
                        chunk_type=ChunkType.IMPORTS,
                        name="imports",
                        content=import_content,
                        token_count=self._estimate_tokens(import_content),
                        file_path=file_path,
                    )
                )

            # استخراج الفئات والدوال
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_chunks = self._extract_class_chunks(
                        node, file_path, lines, max_chunk_size
                    )
                    chunks.extend(class_chunks)

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_chunk = self._extract_function_chunk(node, file_path, lines)
                    if func_chunk:
                        chunks.append(func_chunk)

        except SyntaxError:
            # الرجوع للتقسيم العام
            return self._chunk_generic(file_path, content, max_chunk_size)

        return chunks

    def _extract_class_chunks(
        self, node: ast.ClassDef, file_path: str, lines: list[str], max_chunk_size: int
    ) -> list[CodeChunk]:
        """استخراج قطع الفئة"""
        chunks: list[CodeChunk] = []

        # توثيق الفئة
        docstring = ast.get_docstring(node) or ""

        # توقيع الفئة
        methods = [
            n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        signature = f"class {node.name}"
        if node.bases:
            bases = [ast.unparse(b) for b in node.bases]
            signature += f"({', '.join(bases)})"

        # استخراج المحتوى
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = "\n".join(lines[start_line - 1 : end_line])

        # إذا كانت الفئة كبيرة، تقسيمها
        token_count = self._estimate_tokens(content)

        if token_count <= max_chunk_size:
            # فئة كاملة
            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, f"class_{node.name}"),
                    chunk_type=ChunkType.CLASS,
                    name=node.name,
                    content=content,
                    token_count=token_count,
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature,
                    docstring=docstring,
                    interfaces=methods,
                    metadata={"methods": methods},
                )
            )
        else:
            # تقسيم الفئة
            # أولاً: واجهة الفئة
            interface_content = f"{signature}:\n"
            interface_content += f'    """{docstring}"""\n'
            interface_content += f"    # Methods: {', '.join(methods)}\n"

            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, f"class_{node.name}_interface"),
                    chunk_type=ChunkType.INTERFACE,
                    name=f"{node.name}_interface",
                    content=interface_content,
                    token_count=self._estimate_tokens(interface_content),
                    file_path=file_path,
                    start_line=start_line,
                    signature=signature,
                    docstring=docstring,
                    interfaces=methods,
                )
            )

            # ثم: كل طريقة كقطعة منفصلة
            for method_node in node.body:
                if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_chunk = self._extract_function_chunk(
                        method_node, file_path, lines, class_name=node.name
                    )
                    if method_chunk:
                        chunks.append(method_chunk)

        return chunks

    def _extract_function_chunk(
        self, node: Any, file_path: str, lines: list[str], class_name: str | None = None
    ) -> CodeChunk | None:
        """استخراج قطعة دالة"""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        content = "\n".join(lines[start_line - 1 : end_line])

        # التوقيع
        args = [a.arg for a in node.args.args]
        if class_name and args and args[0] == "self":
            args = args[1:]  # إزالة self من العرض

        is_async = isinstance(node, ast.AsyncFunctionDef)
        signature = f"{'async ' if is_async else ''}def {node.name}({', '.join(args)})"

        # التوثيق
        docstring = ast.get_docstring(node) or ""

        # التبعيات
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                dependencies.append(child.id)
            elif isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                dependencies.append(f"{child.value.id}.{child.attr}")

        chunk_type = ChunkType.METHOD if class_name else ChunkType.FUNCTION
        name = f"{class_name}.{node.name}" if class_name else node.name

        return CodeChunk(
            id=self._generate_id(file_path, f"func_{name}"),
            chunk_type=chunk_type,
            name=name,
            content=content,
            token_count=self._estimate_tokens(content),
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            signature=signature,
            docstring=docstring,
            dependencies=list(set(dependencies))[:10],
            importance=self._calculate_importance(node, content),
        )

    def _calculate_importance(self, node: Any, content: str) -> float:
        """حساب أهمية الدالة"""
        importance = 0.5

        # دوال أكثر أهمية
        important_names = ["main", "run", "execute", "process", "handle", "init"]
        if node.name.lower() in important_names or any(
            n in node.name.lower() for n in important_names
        ):
            importance += 0.3

        # دوال موثقة أكثر أهمية
        if ast.get_docstring(node):
            importance += 0.1

        # دوال أطول قد تكون أكثر أهمية
        lines = content.count("\n")
        if lines > 50:
            importance += 0.1

        return min(importance, 1.0)

    def _chunk_javascript(
        self, file_path: str, content: str, max_chunk_size: int
    ) -> list[CodeChunk]:
        """تقسيم ملف JavaScript/TypeScript"""
        chunks: list[CodeChunk] = []

        # استخراج imports
        import_pattern = re.compile(r"^import\s+.+?;?\s*$", re.MULTILINE)
        imports = import_pattern.findall(content)

        if imports:
            import_content = "\n".join(imports)
            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, "imports"),
                    chunk_type=ChunkType.IMPORTS,
                    name="imports",
                    content=import_content,
                    token_count=self._estimate_tokens(import_content),
                    file_path=file_path,
                )
            )

        # استخراج الفئات
        class_pattern = re.compile(
            r"(export\s+)?(default\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?\s*\{", re.MULTILINE
        )

        for match in class_pattern.finditer(content):
            class_name = match.group(3)
            # البحث عن نهاية الفئة (مبسط)
            start = match.start()
            brace_count = 0
            end = start

            for i, char in enumerate(content[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            class_content = content[start:end]

            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, f"class_{class_name}"),
                    chunk_type=ChunkType.CLASS,
                    name=class_name,
                    content=class_content,
                    token_count=self._estimate_tokens(class_content),
                    file_path=file_path,
                    start_line=content[:start].count("\n") + 1,
                    end_line=content[:end].count("\n") + 1,
                )
            )

        # استخراج الدوال
        func_pattern = re.compile(
            r"(export\s+)?(async\s+)?function\s+(\w+)\s*\([^)]*\)", re.MULTILINE
        )

        for match in func_pattern.finditer(content):
            func_name = match.group(3)
            start = match.start()

            # البحث عن نهاية الدالة (مبسط)
            brace_count = 0
            end = start
            in_func = False

            for i, char in enumerate(content[start:], start):
                if char == "{":
                    brace_count += 1
                    in_func = True
                elif char == "}":
                    brace_count -= 1
                    if in_func and brace_count == 0:
                        end = i + 1
                        break

            func_content = content[start:end]

            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, f"func_{func_name}"),
                    chunk_type=ChunkType.FUNCTION,
                    name=func_name,
                    content=func_content,
                    token_count=self._estimate_tokens(func_content),
                    file_path=file_path,
                    start_line=content[:start].count("\n") + 1,
                    end_line=content[:end].count("\n") + 1,
                    signature=match.group(0),
                )
            )

        return chunks

    def _chunk_generic(self, file_path: str, content: str, max_chunk_size: int) -> list[CodeChunk]:
        """تقسيم عام للملفات"""
        chunks: list[CodeChunk] = []
        lines = content.split("\n")
        total_tokens = self._estimate_tokens(content)

        if total_tokens <= max_chunk_size:
            # ملف كامل
            chunks.append(
                CodeChunk(
                    id=self._generate_id(file_path, "full"),
                    chunk_type=ChunkType.FILE,
                    name=os.path.basename(file_path),
                    content=content,
                    token_count=total_tokens,
                    file_path=file_path,
                    start_line=1,
                    end_line=len(lines),
                )
            )
        else:
            # تقسيم بالأسطر
            chunk_lines: list[str] = []
            current_tokens: int = 0
            start_line = 1

            for i, line in enumerate(lines, 1):
                line_tokens = self._estimate_tokens(line)

                if current_tokens + line_tokens > max_chunk_size and chunk_lines:
                    # حفظ القطعة الحالية
                    chunk_content = "\n".join(chunk_lines)
                    chunks.append(
                        CodeChunk(
                            id=self._generate_id(file_path, f"chunk_{start_line}_{i - 1}"),
                            chunk_type=ChunkType.LOGICAL_BLOCK,
                            name=f"lines_{start_line}_{i - 1}",
                            content=chunk_content,
                            token_count=current_tokens,
                            file_path=file_path,
                            start_line=start_line,
                            end_line=i - 1,
                        )
                    )

                    chunk_lines = []
                    current_tokens = 0
                    start_line = i

                chunk_lines.append(line)
                current_tokens += line_tokens

            # القطعة الأخيرة
            if chunk_lines:
                chunk_content = "\n".join(chunk_lines)
                chunks.append(
                    CodeChunk(
                        id=self._generate_id(file_path, f"chunk_{start_line}_{len(lines)}"),
                        chunk_type=ChunkType.LOGICAL_BLOCK,
                        name=f"lines_{start_line}_{len(lines)}",
                        content=chunk_content,
                        token_count=current_tokens,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=len(lines),
                    )
                )

        return chunks

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _generate_id(self, file_path: str, suffix: str) -> str:
        """توليد معرف فريد"""
        base = f"{file_path}:{suffix}"
        return hashlib.md5(base.encode()).hexdigest()[:12]

    def get_relevant_chunks(
        self, chunks: list[CodeChunk], query: str, limit: int = 10
    ) -> list[CodeChunk]:
        """الحصول على القطع ذات الصلة"""
        query_lower = query.lower()
        scored: list[tuple] = []

        for chunk in chunks:
            score: float = 0

            # تطابق الاسم
            if query_lower in chunk.name.lower():
                score += 10

            # تطابق التوقيع
            if query_lower in chunk.signature.lower():
                score += 5

            # تطابق المحتوى
            if query_lower in chunk.content.lower():
                score += 3

            # تطابق التوثيق
            if query_lower in chunk.docstring.lower():
                score += 2

            # تعديل بالأهمية
            score = score * chunk.importance

            if score > 0:
                scored.append((chunk, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in scored[:limit]]

    def get_chunk_interfaces(self, chunks: list[CodeChunk]) -> list[CodeChunk]:
        """الحصول على واجهات فقط"""
        return [
            CodeChunk(
                id=f"{c.id}_interface",
                chunk_type=ChunkType.INTERFACE,
                name=c.name,
                content=c.signature
                + ('\n    """' + c.docstring[:200] + '"""' if c.docstring else ""),
                token_count=self._estimate_tokens(
                    c.signature + (c.docstring[:200] if c.docstring else "")
                ),
                file_path=c.file_path,
                signature=c.signature,
                interfaces=c.interfaces,
            )
            for c in chunks
            if c.signature
        ]

    def clear_cache(self) -> None:
        """مسح التخزين المؤقت"""
        self._chunk_cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """إحصائيات"""
        return {
            "cached_files": len(self._chunk_cache),
            "total_chunks": sum(len(chunks) for chunks in self._chunk_cache.values()),
        }
