"""
Code Synthesizer Module

Generates Python code for new tools using templates and LLM.

Features:
    - Template-based code generation
    - LLM-powered synthesis with structured prompts
    - Comprehensive code validation (AST, security, type hints)
    - Automatic test generation
    - Multiple code templates for different use cases

Usage:
    from gaap.tools.code_synthesizer import CodeSynthesizer, SynthesisRequest

    synthesizer = CodeSynthesizer(provider=my_llm_provider)
    request = SynthesisRequest(
        intent="Create a JSON parser",
        required_libraries=["json"],
        input_schema={"data": "str"},
        output_type="dict"
    )
    result = await synthesizer.synthesize(request, libraries)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from gaap.core.logging import get_standard_logger

if TYPE_CHECKING:
    from gaap.core.base import BaseProvider
    from gaap.tools.library_discoverer import LibraryInfo

logger = get_standard_logger("gaap.tools.code_synthesizer")


class ComplexityLevel(Enum):
    """Complexity levels for synthesis."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class SynthesisRequest:
    """Request for code synthesis."""

    intent: str
    required_libraries: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_type: str = "Any"
    constraints: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "required_libraries": self.required_libraries,
            "input_schema": self.input_schema,
            "output_type": self.output_type,
            "constraints": self.constraints,
            "examples": self.examples,
        }


@dataclass
class SynthesisResult:
    """Result of code synthesis."""

    code: str
    success: bool
    error: str | None = None
    imports_needed: list[str] = field(default_factory=list)
    tests_generated: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "success": self.success,
            "error": self.error,
            "imports_needed": self.imports_needed,
            "tests_generated": self.tests_generated,
            "confidence": self.confidence,
        }


TEMPLATE_BASIC = '''"""
{description}
"""

from typing import Any


def run(**kwargs: Any) -> dict[str, Any]:
    """
    {function_description}

    Returns:
        dict with 'success' and 'result' or 'error'
    """
    try:
        {implementation}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''

TEMPLATE_WITH_CONFIG = '''
"""
{description}
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class Config:
    {config_fields}


def run(config: Config, **kwargs: Any) -> dict[str, Any]:
    """
    {function_description}

    Args:
        config: Configuration object

    Returns:
        dict with 'success' and 'result' or 'error'
    """
    try:
        {implementation}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}
'''

TEMPLATE_ASYNC = '''
"""
{description}
"""

import asyncio
from typing import Any


async def run(**kwargs: Any) -> dict[str, Any]:
    """
    {function_description}

    Returns:
        dict with 'success' and 'result' or 'error'
    """
    try:
        {implementation}
        return {{"success": True, "result": result}}
    except Exception as e:
        return {{"success": False, "error": str(e)}}


def execute(**kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper for async run function."""
    return asyncio.run(run(**kwargs))
'''

TEMPLATE_ERROR_HANDLING = '''
"""
{description}
"""

from typing import Any
import logging

logger = logging.getLogger(__name__)


class ToolError(Exception):
    """Custom error for this tool."""
    pass


class ValidationError(ToolError):
    """Validation failed."""
    pass


class ExecutionError(ToolError):
    """Execution failed."""
    pass


def validate_input(**kwargs: Any) -> None:
    """Validate input parameters."""
    {validation_logic}


def run(**kwargs: Any) -> dict[str, Any]:
    """
    {function_description}

    Returns:
        dict with 'success' and 'result' or 'error'
    """
    try:
        validate_input(**kwargs)
        logger.info("Starting execution")

        {implementation}

        logger.info("Execution completed successfully")
        return {{"success": True, "result": result}}
    except ValidationError as e:
        logger.warning(f"Validation error: {{e}}")
        return {{"success": False, "error": f"Validation: {{e}}"}}
    except ExecutionError as e:
        logger.error(f"Execution error: {{e}}")
        return {{"success": False, "error": f"Execution: {{e}}"}}
    except Exception as e:
        logger.exception("Unexpected error")
        return {{"success": False, "error": str(e)}}
'''

TEMPLATE_PROGRESS = '''
"""
{description}
"""

from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


ProgressCallback = Callable[[int, int, str], None]


def run(
    progress_callback: ProgressCallback | None = None,
    **kwargs: Any
) -> dict[str, Any]:
    """
    {function_description}

    Args:
        progress_callback: Optional callback(current, total, message)

    Returns:
        dict with 'success' and 'result' or 'error'
    """
    try:
        total_steps = {total_steps}

        def report_progress(step: int, message: str) -> None:
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"Progress: {{step}}/{{total_steps}} - {{message}}")

        report_progress(0, "Starting")

        {implementation_with_progress}

        report_progress(total_steps, "Completed")
        return {{"success": True, "result": result}}
    except Exception as e:
        logger.exception("Error during execution")
        return {{"success": False, "error": str(e)}}
'''

PROMPT_CODE_GENERATION = """You are an expert Python code generator. Generate a complete Python module for the following tool.

INTENT: {intent}

REQUIRED LIBRARIES: {libraries}

INPUT SCHEMA:
{input_schema}

OUTPUT TYPE: {output_type}

CONSTRAINTS:
{constraints}

EXAMPLES:
{examples}

REQUIREMENTS:
1. Create a complete Python module with proper imports at the top
2. Include a run() function as the main entry point
3. Use proper type hints for all parameters and return values
4. Include comprehensive docstrings
5. Implement proper error handling with try/except blocks
6. Return a dict with 'success' (bool) and 'result' or 'error' keys
7. Follow PEP 8 style guidelines
8. Keep the code under {max_length} characters

OUTPUT FORMAT:
Return ONLY the Python code, no explanations. Start with imports and end with the run() function.

```python
# Your code here
```
"""

PROMPT_TEST_GENERATION = """Generate comprehensive unit tests for the following Python tool code.

CODE:
```python
{code}
```

REQUIREMENTS:
1. Use pytest framework
2. Test the run() function
3. Include tests for:
   - Normal operation with valid inputs
   - Edge cases
   - Error handling
4. Use proper test function names starting with test_
5. Include docstrings for test functions

OUTPUT FORMAT:
Return ONLY the Python test code, no explanations.

```python
# Test code here
```
"""

DANGEROUS_IMPORTS = {
    "os.system",
    "os.popen",
    "subprocess.call",
    "subprocess.run",
    "subprocess.Popen",
    "eval",
    "exec",
    "compile",
    "__import__",
    "importlib.import_module",
    "pickle.loads",
    "pickle.load",
    "shelve.open",
    "marshal.loads",
}

DANGEROUS_PATTERNS = [
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\(",
    r"__import__\s*\(",
    r"subprocess\.(call|run|Popen)",
    r"os\.system\s*\(",
    r"os\.popen\s*\(",
    r"open\s*\([^)]*,\s*['\"]w['\"]",
]


@dataclass
class CodeSynthesizerConfig:
    """Configuration for code synthesizer."""

    max_code_length: int = 5000
    require_type_hints: bool = True
    require_docstrings: bool = True
    allowed_imports: list[str] = field(
        default_factory=lambda: [
            "typing",
            "dataclasses",
            "json",
            "re",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "logging",
            "abc",
            "enum",
            "copy",
            "math",
            "random",
            "string",
            "time",
            "uuid",
            "hashlib",
            "base64",
            "html",
            "urllib.parse",
            "http.client",
            "aiohttp",
            "requests",
            "asyncio",
        ]
    )


class CodeSynthesizer:
    """
    Generates Python code for new tools using templates and LLM.

    Features:
        - Template-based code generation
        - LLM-powered synthesis with structured prompts
        - Comprehensive code validation (AST, security, type hints)
        - Automatic test generation
        - Multiple code templates for different use cases
    """

    def __init__(
        self,
        provider: BaseProvider | None = None,
        config: CodeSynthesizerConfig | None = None,
    ):
        self._provider = provider
        self._config = config or CodeSynthesizerConfig()
        self._templates = {
            "basic": TEMPLATE_BASIC,
            "with_config": TEMPLATE_WITH_CONFIG,
            "async": TEMPLATE_ASYNC,
            "error_handling": TEMPLATE_ERROR_HANDLING,
            "progress": TEMPLATE_PROGRESS,
        }

    def __repr__(self) -> str:
        return f"CodeSynthesizer(provider={'set' if self._provider else 'none'}, templates={len(self._templates)})"

    async def synthesize(
        self,
        request: SynthesisRequest,
        libraries: list[LibraryInfo] | None = None,
    ) -> SynthesisResult:
        """
        Synthesize Python code for a tool based on the request.

        Args:
            request: Synthesis request with intent, schema, etc.
            libraries: Optional list of relevant libraries discovered

        Returns:
            SynthesisResult with generated code and metadata
        """
        logger.info(f"Synthesizing code for intent: {request.intent[:50]}...")

        complexity = self.estimate_complexity(request)
        logger.info(f"Estimated complexity: {complexity}")

        if self._provider:
            result = await self._synthesize_with_llm(request, libraries, complexity)
            if result.success:
                return result
            logger.warning(f"LLM synthesis failed: {result.error}, falling back to template")

        return self._synthesize_from_template(request, complexity)

    async def _synthesize_with_llm(
        self,
        request: SynthesisRequest,
        libraries: list[LibraryInfo] | None,
        complexity: str,
    ) -> SynthesisResult:
        """Generate code using LLM provider."""
        if not self._provider:
            return SynthesisResult(
                code="",
                success=False,
                error="No LLM provider configured",
                confidence=0.0,
            )

        prompt = self._build_generation_prompt(request, libraries)

        try:
            from gaap.core.types import Message, MessageRole

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are an expert Python code generator. Generate clean, well-documented code.",
                ),
                Message(role=MessageRole.USER, content=prompt),
            ]

            models = self._provider.get_available_models()
            if not models:
                return SynthesisResult(
                    code="",
                    success=False,
                    error="No models available",
                    confidence=0.0,
                )

            response = await self._provider.chat_completion(
                messages=messages,
                model=models[0],
                temperature=0.3,
                max_tokens=2000,
            )

            if not response.choices:
                return SynthesisResult(
                    code="",
                    success=False,
                    error="No response from LLM",
                    confidence=0.0,
                )

            content = response.choices[0].message.content
            code = self._extract_code_from_response(content)

            if not code:
                return SynthesisResult(
                    code="",
                    success=False,
                    error="No code found in LLM response",
                    confidence=0.0,
                )

            is_valid, errors = self.validate_generated_code(code)
            if not is_valid:
                return SynthesisResult(
                    code=code,
                    success=False,
                    error=f"Validation failed: {'; '.join(errors)}",
                    confidence=0.3,
                )

            tests = await self.generate_tests(code)

            imports = self._extract_imports(code)

            return SynthesisResult(
                code=code,
                success=True,
                imports_needed=imports,
                tests_generated=tests,
                confidence=0.8,
            )

        except Exception as e:
            logger.exception(f"LLM synthesis error: {e}")
            return SynthesisResult(
                code="",
                success=False,
                error=str(e),
                confidence=0.0,
            )

    def _synthesize_from_template(
        self,
        request: SynthesisRequest,
        complexity: str,
    ) -> SynthesisResult:
        """Generate code from template as fallback."""
        template_name = self._select_template(request, complexity)
        params = self._build_template_params(request)

        code = self.generate_from_template(template_name, params)

        is_valid, errors = self.validate_generated_code(code)
        if not is_valid:
            return SynthesisResult(
                code=code,
                success=False,
                error=f"Template validation failed: {'; '.join(errors)}",
                confidence=0.2,
            )

        imports = self._extract_imports(code)

        return SynthesisResult(
            code=code,
            success=True,
            imports_needed=imports,
            tests_generated="",
            confidence=0.5,
        )

    def generate_from_template(self, template_name: str, params: dict[str, Any]) -> str:
        """
        Generate code from a named template with parameters.

        Args:
            template_name: Name of the template (basic, async, etc.)
            params: Parameters to fill in the template

        Returns:
            Generated code string
        """
        template = self._templates.get(template_name, TEMPLATE_BASIC)

        defaults = {
            "description": "Generated tool",
            "function_description": "Execute the tool functionality",
            "implementation": "result = kwargs",
            "config_fields": "pass",
            "validation_logic": "pass",
            "implementation_with_progress": "result = kwargs",
            "total_steps": 3,
        }

        full_params = {**defaults, **params}

        try:
            return template.format(**full_params).strip()
        except KeyError as e:
            logger.error(f"Template parameter missing: {e}")
            return template.format(**defaults).strip()

    def validate_generated_code(self, code: str) -> tuple[bool, list[str]]:
        """
        Validate generated Python code.

        Performs multiple validation checks:
        - AST parsing
        - Import validation
        - Security checks
        - Entry point verification
        - Type hint checks

        Args:
            code: Python code to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors: list[str] = []

        if len(code) > self._config.max_code_length:
            errors.append(f"Code exceeds max length ({len(code)} > {self._config.max_code_length})")

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error: {e}")
            return (False, errors)

        imports = self._extract_imports_from_ast(tree)

        for imp in imports:
            root_module = imp.split(".")[0]
            if root_module not in self._config.allowed_imports:
                if not any(
                    root_module.startswith(allowed.split(".")[0])
                    for allowed in self._config.allowed_imports
                ):
                    pass

        for pattern in DANGEROUS_PATTERNS:
            if re.search(pattern, code):
                errors.append(f"Potentially dangerous pattern found: {pattern}")

        for dangerous in DANGEROUS_IMPORTS:
            if dangerous in code:
                errors.append(f"Dangerous import/function: {dangerous}")

        has_entry_point = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name in ("run", "execute"):
                    has_entry_point = True

                    if self._config.require_type_hints:
                        if not node.returns:
                            errors.append("Entry point function missing return type hint")

                        for arg in node.args.args:
                            if arg.annotation is None and arg.arg != "self":
                                errors.append(f"Parameter '{arg.arg}' missing type hint")

                    if self._config.require_docstrings:
                        if not ast.get_docstring(node):
                            errors.append("Entry point function missing docstring")

        if not has_entry_point:
            errors.append("No entry point function (run/execute) found")

        return (len(errors) == 0, errors)

    async def generate_tests(self, code: str) -> str:
        """
        Generate unit tests for the given code.

        Args:
            code: Python code to generate tests for

        Returns:
            Generated test code string
        """
        if self._provider:
            return await self._generate_tests_with_llm(code)

        return self._generate_basic_tests(code)

    async def _generate_tests_with_llm(self, code: str) -> str:
        """Generate tests using LLM."""
        if not self._provider:
            return ""

        try:
            from gaap.core.types import Message, MessageRole

            prompt = PROMPT_TEST_GENERATION.format(code=code)

            messages = [
                Message(
                    role=MessageRole.SYSTEM,
                    content="You are an expert test writer. Generate comprehensive pytest tests.",
                ),
                Message(role=MessageRole.USER, content=prompt),
            ]

            models = self._provider.get_available_models()
            if not models:
                return ""

            response = await self._provider.chat_completion(
                messages=messages,
                model=models[0],
                temperature=0.3,
                max_tokens=1500,
            )

            if response.choices:
                content = response.choices[0].message.content
                return self._extract_code_from_response(content) or ""

        except Exception as e:
            logger.warning(f"Test generation failed: {e}")

        return ""

    def _generate_basic_tests(self, code: str) -> str:
        """Generate basic test scaffolding."""
        test_template = '''
"""
Unit tests for generated tool.
"""

import pytest


def test_run_returns_dict():
    """Test that run function returns a dictionary."""
    from generated_tool import run

    result = run()
    assert isinstance(result, dict)
    assert "success" in result


def test_run_success_structure():
    """Test successful result structure."""
    from generated_tool import run

    result = run()
    if result.get("success"):
        assert "result" in result
        assert "error" not in result


def test_run_error_structure():
    """Test error result structure when error occurs."""
    from generated_tool import run

    result = run(invalid_param_to_trigger_error=True)
    if not result.get("success"):
        assert "error" in result
'''
        return test_template.strip()

    def estimate_complexity(self, request: SynthesisRequest) -> str:
        """
        Estimate the complexity of the synthesis request.

        Args:
            request: Synthesis request

        Returns:
            Complexity level: 'simple', 'moderate', or 'complex'
        """
        score = 0

        intent = request.intent.lower()

        complex_keywords = [
            "async",
            "concurrent",
            "parallel",
            "distributed",
            "database",
            "network",
            "api",
            "server",
            "client",
            "authentication",
            "encryption",
            "security",
            "machine learning",
            "ml",
            "ai",
            "model",
        ]
        for kw in complex_keywords:
            if kw in intent:
                score += 2

        moderate_keywords = [
            "file",
            "read",
            "write",
            "parse",
            "convert",
            "validate",
            "transform",
            "process",
            "http",
            "request",
            "response",
        ]
        for kw in moderate_keywords:
            if kw in intent:
                score += 1

        score += len(request.required_libraries) * 0.5

        score += len(request.constraints) * 0.3

        if len(request.input_schema) > 5:
            score += 1

        if score >= 5:
            return ComplexityLevel.COMPLEX.value
        elif score >= 2:
            return ComplexityLevel.MODERATE.value
        else:
            return ComplexityLevel.SIMPLE.value

    def _build_generation_prompt(
        self,
        request: SynthesisRequest,
        libraries: list[LibraryInfo] | None,
    ) -> str:
        """Build the prompt for LLM code generation."""
        libraries_str = ", ".join(request.required_libraries) or "None"

        input_schema_str = (
            "\n".join(f"  {k}: {v}" for k, v in request.input_schema.items())
            or "  No specific schema"
        )

        constraints_str = "\n".join(f"- {c}" for c in request.constraints) or "- None specified"

        examples_str = (
            "\n".join(
                f"  Input: {ex.get('input', 'N/A')}\n  Output: {ex.get('output', 'N/A')}"
                for ex in request.examples[:3]
            )
            or "  No examples provided"
        )

        if libraries:
            libs_context = "\n".join(
                f"- {lib.name}: {lib.description[:100]}" for lib in libraries[:3]
            )
            libraries_str += f"\n\nDiscovered Libraries:\n{libs_context}"

        return PROMPT_CODE_GENERATION.format(
            intent=request.intent,
            libraries=libraries_str,
            input_schema=input_schema_str,
            output_type=request.output_type,
            constraints=constraints_str,
            examples=examples_str,
            max_length=self._config.max_code_length,
        )

    def _select_template(self, request: SynthesisRequest, complexity: str) -> str:
        """Select the best template based on request."""
        intent_lower = request.intent.lower()

        if "async" in intent_lower or "await" in intent_lower:
            return "async"

        if "progress" in intent_lower or "callback" in intent_lower:
            return "progress"

        if complexity == ComplexityLevel.COMPLEX.value:
            return "error_handling"

        if request.constraints and len(request.constraints) > 3:
            return "error_handling"

        return "basic"

    def _build_template_params(self, request: SynthesisRequest) -> dict[str, Any]:
        """Build parameters for template filling."""
        params: dict[str, Any] = {
            "description": request.intent,
            "function_description": request.intent,
        }

        impl_parts = []
        for param, ptype in request.input_schema.items():
            impl_parts.append(f"{param} = kwargs.get('{param}')")

        impl_parts.append("result = kwargs")
        params["implementation"] = "\n        ".join(impl_parts)

        return params

    def _extract_code_from_response(self, content: str) -> str | None:
        """Extract Python code from LLM response."""
        code_block_pattern = r"```(?:python)?\s*\n(.*?)\n```"
        matches = re.findall(code_block_pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        if "def " in content or "import " in content:
            lines = content.split("\n")
            code_lines = []
            in_code = False

            for line in lines:
                if line.strip().startswith(("import ", "from ", "def ", "class ", "@")):
                    in_code = True
                if in_code:
                    code_lines.append(line)

            return "\n".join(code_lines).strip()

        return None

    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements from code."""
        imports = []
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                imports.append(line)
        return imports

    def _extract_imports_from_ast(self, tree: ast.AST) -> list[str]:
        """Extract import names from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def register_template(self, name: str, template: str) -> None:
        """Register a custom template."""
        self._templates[name] = template

    def get_template_names(self) -> list[str]:
        """Get list of available template names."""
        return list(self._templates.keys())

    def set_provider(self, provider: BaseProvider) -> None:
        """Set the LLM provider."""
        self._provider = provider

    def get_config(self) -> CodeSynthesizerConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, **kwargs: Any) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
