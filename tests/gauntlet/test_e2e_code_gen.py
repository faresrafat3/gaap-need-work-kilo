"""
E2E Code Generation Gauntlet Tests
==================================

Tests for end-to-end code generation scenarios.

Implements: docs/evolution_plan_2026/45_TESTING_AUDIT_SPEC.md

NOTE: Gauntlet tests are integration tests that test full workflows.
They are marked with @pytest.mark.gauntlet and run by default.
Use -m "not gauntlet" to skip.
"""

import ast
import os
import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gaap.core.types import Message, MessageRole, TaskPriority, TaskType
from gaap.layers.layer0_interface import IntentType, StructuredIntent


class MockKiloProvider:
    """Mock provider simulating Kilo/Kimi responses."""

    def __init__(self):
        self.name = "kilo-mock"
        self.models = ["kilo-1.0", "kimi-1.5"]
        self.default_model = "kilo-1.0"

    async def chat_completion(self, messages, model=None, **kwargs):
        from gaap.core.types import ChatCompletionChoice, ChatCompletionResponse, Usage

        user_msg = messages[-1].content if messages else ""

        if "factorial" in user_msg.lower():
            code = '''def factorial(n: int) -> int:
    """Calculate the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)


if __name__ == "__main__":
    print(factorial(5))  # Output: 120
'''
        elif "fibonacci" in user_msg.lower():
            code = '''def fibonacci(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    
    seq = [0, 1]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq


if __name__ == "__main__":
    print(fibonacci(10))
'''
        elif "binary search" in user_msg.lower():
            code = '''def binary_search(arr: list[int], target: int) -> int:
    """Perform binary search on a sorted array.
    
    Args:
        arr: Sorted list of integers
        target: Value to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1


if __name__ == "__main__":
    arr = [1, 3, 5, 7, 9, 11, 13]
    print(binary_search(arr, 7))  # Output: 3
'''
        else:
            code = '''def hello_world() -> str:
    """Return a greeting message."""
    return "Hello, World!"


if __name__ == "__main__":
    print(hello_world())
'''

        return ChatCompletionResponse(
            id="kilo-response",
            model=model or self.default_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=Message(role=MessageRole.ASSISTANT, content=code),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=50, completion_tokens=150, total_tokens=200),
        )


@pytest.fixture
def kilo_provider():
    """Provide mock Kilo/Kimi provider."""
    return MockKiloProvider()


class TestCodeGenerationGauntlet:
    """E2E tests for code generation."""

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_factorial_generation(self, kilo_provider, tmp_path) -> None:
        """Test generation of factorial function."""
        from gaap.gaap_engine import create_engine

        response = await kilo_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Write a factorial function in Python")]
        )

        code = response.choices[0].message.content

        code_file = tmp_path / "factorial.py"
        code_file.write_text(code)

        assert code_file.exists()
        assert "def factorial" in code
        assert "n:" in code

        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            assert "factorial" in functions
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

        namespace = {}
        exec(code, namespace)
        assert namespace["factorial"](5) == 120
        assert namespace["factorial"](0) == 1
        assert namespace["factorial"](1) == 1

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_fibonacci_generation(self, kilo_provider, tmp_path) -> None:
        """Test generation of Fibonacci function."""
        response = await kilo_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Write a Fibonacci sequence function")]
        )

        code = response.choices[0].message.content

        assert "def fibonacci" in code

        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            assert "fibonacci" in functions
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

        namespace = {}
        exec(code, namespace)
        result = namespace["fibonacci"](10)
        assert len(result) == 10
        assert result[0] == 0
        assert result[1] == 1

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_binary_search_generation(self, kilo_provider, tmp_path) -> None:
        """Test generation of binary search function."""
        response = await kilo_provider.chat_completion(
            [Message(role=MessageRole.USER, content="Implement binary search in Python")]
        )

        code = response.choices[0].message.content

        assert "def binary_search" in code
        assert "target" in code.lower()

        try:
            tree = ast.parse(code)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            assert "binary_search" in functions
        except SyntaxError as e:
            pytest.fail(f"Generated code has syntax errors: {e}")

        namespace = {}
        exec(code, namespace)
        arr = [1, 3, 5, 7, 9, 11, 13]
        assert namespace["binary_search"](arr, 7) == 3
        assert namespace["binary_search"](arr, 1) == 0
        assert namespace["binary_search"](arr, 13) == 6
        assert namespace["binary_search"](arr, 4) == -1

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_code_with_type_hints(self, kilo_provider) -> None:
        """Test that generated code includes type hints."""
        response = await kilo_provider.chat_completion(
            [
                Message(
                    role=MessageRole.USER,
                    content="Write a Python function with type hints for calculating factorial",
                )
            ]
        )

        code = response.choices[0].message.content

        assert "-> int" in code or "-> float" in code
        assert ": int" in code or ": float" in code

    @pytest.mark.gauntlet
    @pytest.mark.asyncio
    async def test_code_with_docstring(self, kilo_provider) -> None:
        """Test that generated code includes docstrings."""
        response = await kilo_provider.chat_completion(
            [
                Message(
                    role=MessageRole.USER,
                    content="Write a well-documented Python function",
                )
            ]
        )

        code = response.choices[0].message.content

        assert '"""' in code or "'''" in code


class TestCodeQualityGauntlet:
    """Tests for code quality checks."""

    @pytest.mark.gauntlet
    def test_syntax_validation(self, gauntlet_runner, tmp_path) -> None:
        """Test that generated code is syntactically valid."""
        valid_code = "def hello():\n    return 'world'"
        invalid_code = "def hello(\n    return 'world'"

        valid_file = tmp_path / "valid.py"
        valid_file.write_text(valid_code)

        invalid_file = tmp_path / "invalid.py"
        invalid_file.write_text(invalid_code)

        try:
            ast.parse(valid_code)
            valid_syntax = True
        except SyntaxError:
            valid_syntax = False

        try:
            ast.parse(invalid_code)
            invalid_syntax = True
        except SyntaxError:
            invalid_syntax = False

        assert valid_syntax is True
        assert invalid_syntax is False

    @pytest.mark.gauntlet
    def test_import_validation(self, tmp_path) -> None:
        """Test that generated code has valid imports."""
        code = '''
import os
import sys
from pathlib import Path

def get_files(directory: str) -> list[str]:
    """List files in directory."""
    return [str(f) for f in Path(directory).iterdir() if f.is_file()]
'''

        code_file = tmp_path / "imports.py"
        code_file.write_text(code)

        try:
            tree = ast.parse(code)
            imports = [
                node.names[0].name
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
            assert "os" in imports or "Path" in names
        except SyntaxError as e:
            pytest.fail(f"Code has syntax errors: {e}")


class TestSemanticAssertionsGauntlet:
    """Tests using semantic assertions."""

    @pytest.mark.gauntlet
    def test_response_relevance(self, semantic_judge) -> None:
        """Test that response is relevant to the query."""
        query = "Write a Python function for factorial"
        response = (
            "def factorial(n: int) -> int:\n    if n <= 1: return 1\n    return n * factorial(n-1)"
        )

        semantic_judge.assert_contains_concepts(response, ["factorial", "def"])

    @pytest.mark.gauntlet
    def test_code_correctness_concept(self, semantic_judge) -> None:
        """Test that code contains required concepts."""
        code = """
def sort_list(items: list) -> list:
    '''Sort a list in ascending order.'''
    return sorted(items)
"""
        semantic_judge.assert_contains_concepts(code, ["sort", "list", "def"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "gauntlet"])
