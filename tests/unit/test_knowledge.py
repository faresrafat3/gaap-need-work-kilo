"""
Tests for Knowledge Ingestion Module
===================================

Tests for AST parsing, usage mining, cheat sheet generation,
and knowledge ingestion.

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaap.knowledge import (
    KnowledgeConfig,
    ASTParser,
    ParsedFile,
    ClassInfo,
    FunctionInfo,
    Parameter,
    UsageMiner,
    UsageExample,
    MiningResult,
    CheatSheetGenerator,
    ReferenceCard,
    FunctionSummary,
    KnowledgeIngestion,
    IngestionResult,
    LibraryKnowledge,
    create_knowledge_config,
    create_parser,
    create_usage_miner,
    create_cheat_sheet_generator,
    create_knowledge_ingestion,
)


class TestKnowledgeConfig:
    """Tests for KnowledgeConfig."""

    def test_default_config(self):
        config = KnowledgeConfig()

        assert config.enabled is True
        assert config.max_file_size_mb == 10
        assert config.max_files_per_repo == 1000
        assert "python" in config.supported_languages

    def test_fast_preset(self):
        config = KnowledgeConfig.fast()

        assert config.max_file_size_mb == 5
        assert config.max_files_per_repo == 100
        assert config.store_in_vector_memory is False

    def test_deep_preset(self):
        config = KnowledgeConfig.deep()

        assert config.max_files_per_repo == 5000
        assert config.top_functions_count == 20

    def test_config_to_dict_and_from_dict(self):
        config = KnowledgeConfig.deep()
        data = config.to_dict()

        restored = KnowledgeConfig.from_dict(data)

        assert restored.max_files_per_repo == config.max_files_per_repo
        assert restored.top_functions_count == config.top_functions_count

    def test_create_knowledge_config_factory(self):
        config = create_knowledge_config("fast")
        assert config.max_file_size_mb == 5

        config = create_knowledge_config("default", max_file_size_mb=20)
        assert config.max_file_size_mb == 20


class TestASTParser:
    """Tests for ASTParser."""

    def test_parse_simple_python_file(self, tmp_path):
        test_file = tmp_path / "simple.py"
        test_file.write_text(
            '"""Module docstring."""\n\n'
            "def hello(name: str) -> str:\n"
            '    """Say hello."""\n'
            '    return f"Hello, {name}!"\n\n'
            "class Greeter:\n"
            '    """A greeter class."""\n'
            "    def greet(self, name: str) -> str:\n"
            '        return f"Hi, {name}!"\n'
        )

        parser = ASTParser()
        result = parser.parse_file(test_file)

        assert result.language == "python"
        assert result.error is None
        assert result.module_docstring == "Module docstring."
        assert len(result.functions) == 1
        assert len(result.classes) == 1

    def test_parse_function_with_parameters(self, tmp_path):
        test_file = tmp_path / "params.py"
        test_file.write_text(
            "def complex_func(a: int, b: str = 'default') -> bool:\n"
            '    """A complex function."""\n'
            "    return True\n"
        )

        parser = ASTParser()
        result = parser.parse_file(test_file)

        assert len(result.functions) == 1
        func = result.functions[0]
        assert func.name == "complex_func"
        assert func.return_type == "bool"
        assert len(func.parameters) >= 2

    def test_parse_class_with_inheritance(self, tmp_path):
        test_file = tmp_path / "inherit.py"
        test_file.write_text(
            "class Animal:\n"
            '    """Base animal."""\n'
            "    pass\n\n"
            "class Dog(Animal):\n"
            '    """A dog."""\n'
            "    def bark(self) -> str:\n"
            '        return "Woof!"\n'
        )

        parser = ASTParser()
        result = parser.parse_file(test_file)

        assert len(result.classes) == 2

        dog_class = next(c for c in result.classes if c.name == "Dog")
        assert "Animal" in dog_class.bases
        assert len(dog_class.methods) == 1
        assert dog_class.methods[0].name == "bark"

    def test_parse_imports(self, tmp_path):
        test_file = tmp_path / "imports.py"
        test_file.write_text(
            "import os\nfrom typing import List, Dict\nfrom pathlib import Path as P\n"
        )

        parser = ASTParser()
        result = parser.parse_file(test_file)

        assert len(result.imports) == 3

    def test_parse_directory(self, tmp_path):
        (tmp_path / "module1.py").write_text("def func1(): pass")
        (tmp_path / "module2.py").write_text("def func2(): pass")
        (tmp_path / "subdir").mkdir()
        (tmp_path / "subdir" / "module3.py").write_text("def func3(): pass")

        parser = ASTParser()
        results = parser.parse_directory(tmp_path)

        assert len(results) == 3
        total_functions = sum(len(r.functions) for r in results)
        assert total_functions == 3

    def test_parse_unsupported_language(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello world")

        parser = ASTParser()
        result = parser.parse_file(test_file)

        assert result.language == "unknown"


class TestUsageMiner:
    """Tests for UsageMiner."""

    def test_mine_test_files(self, tmp_path):
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()

        test_code = (
            "def test_greeting():\n"
            '    """Test the greeting function."""\n'
            "    from mylib import greet\n"
            "    result = greet('World')\n"
            '    assert result == "Hello, World!"\n'
        )
        (tests_dir / "test_module.py").write_text(test_code)

        miner = UsageMiner()
        result = miner.mine_tests(tmp_path)

        assert result.total_files >= 1
        assert len(result.examples) >= 1

    def test_mine_examples(self, tmp_path):
        examples_dir = tmp_path / "examples"
        examples_dir.mkdir()

        example_code = (
            "def example_basic_usage():\n"
            '    """Example: Basic usage."""\n'
            "    from mylib import Client\n"
            "    client = Client()\n"
            "    client.connect()\n"
        )
        (examples_dir / "basic.py").write_text(example_code)

        miner = UsageMiner()
        result = miner.mine_examples(tmp_path)

        assert result.total_files >= 1
        assert len(result.examples) >= 1

    def test_infer_intent(self, tmp_path):
        miner = UsageMiner()

        assert "Testing" in miner._infer_intent("test_greeting")
        assert "Should" in miner._infer_intent("it_should_work")
        assert "Usage" in miner._infer_intent("basic_example")


class TestCheatSheetGenerator:
    """Tests for CheatSheetGenerator."""

    def test_generate_reference_card(self, tmp_path):
        test_file = tmp_path / "lib.py"
        test_file.write_text(
            "def important_function(x: int) -> int:\n"
            '    """An important function."""\n'
            "    return x * 2\n\n"
            "class ImportantClass:\n"
            '    """An important class."""\n'
            "    pass\n"
        )

        parser = ASTParser()
        parsed_files = parser.parse_directory(tmp_path)

        generator = CheatSheetGenerator()
        card = generator.generate(
            parsed_files=parsed_files,
            library_name="testlib",
            description="A test library",
        )

        assert card.library_name == "testlib"
        assert card.description == "A test library"
        assert card.total_files_analyzed >= 1
        assert len(card.top_functions) >= 1
        assert len(card.top_classes) >= 1

    def test_reference_card_to_markdown(self, tmp_path):
        parser = ASTParser()

        test_file = tmp_path / "lib.py"
        test_file.write_text("def foo(): pass")

        parsed = parser.parse_directory(tmp_path)

        generator = CheatSheetGenerator()
        card = generator.generate(parsed_files=parsed, library_name="test")

        markdown = card.to_markdown()

        assert "# test" in markdown
        assert "## Top Functions" in markdown

    def test_save_card(self, tmp_path):
        parser = ASTParser()

        test_file = tmp_path / "lib.py"
        test_file.write_text("def bar(): pass")

        parsed = parser.parse_directory(tmp_path)

        generator = CheatSheetGenerator()
        card = generator.generate(parsed_files=parsed, library_name="save_test")

        output_path = tmp_path / "output" / "card"
        saved_path = generator.save_card(card, output_path, format="json")

        assert saved_path.exists()
        assert saved_path.suffix == ".json"


class TestKnowledgeIngestion:
    """Tests for KnowledgeIngestion."""

    def test_ingest_local_directory(self, tmp_path):
        lib_dir = tmp_path / "mylib"
        lib_dir.mkdir()

        init_code = '"""My Library."""\n\ndef useful_function(x: int) -> int:\n    return x + 1\n'
        (lib_dir / "__init__.py").write_text(init_code)

        tests_dir = lib_dir / "tests"
        tests_dir.mkdir()
        test_code = "def test_func():\n    from mylib import useful_function\n    assert useful_function(1) == 2\n"
        (tests_dir / "test_lib.py").write_text(test_code)

        config = KnowledgeConfig(storage_path=str(tmp_path / "knowledge"))
        ingestion = KnowledgeIngestion(config=config)

        import asyncio

        result = asyncio.run(
            ingestion.ingest_repo(
                source=lib_dir,
                library_name="mylib",
                description="Test library",
            )
        )

        assert result.success is True
        assert result.library_name == "mylib"
        assert result.files_parsed >= 1
        assert result.output_path is not None

    def test_list_libraries(self, tmp_path):
        config = KnowledgeConfig(storage_path=str(tmp_path / "knowledge"))
        ingestion = KnowledgeIngestion(config=config)

        libraries = ingestion.list_libraries()

        assert isinstance(libraries, list)

    def test_load_library(self, tmp_path):
        storage_path = tmp_path / "knowledge"
        storage_path.mkdir(parents=True)

        lib_file = storage_path / "testlib.json"
        lib_file.write_text(
            json.dumps(
                {
                    "library_name": "testlib",
                    "version": "1.0.0",
                    "description": "Test library",
                    "reference_card": {
                        "library_name": "testlib",
                        "top_functions": [],
                        "top_classes": [],
                        "common_patterns": [],
                    },
                    "parsed_files": [],
                    "usage_examples": [],
                    "ingested_at": "2026-01-01T00:00:00",
                }
            )
        )

        config = KnowledgeConfig(storage_path=str(storage_path))
        ingestion = KnowledgeIngestion(config=config)

        knowledge = ingestion.load_library("testlib")

        assert knowledge is not None
        assert knowledge.library_name == "testlib"
        assert knowledge.version == "1.0.0"


class TestLibraryKnowledge:
    """Tests for LibraryKnowledge."""

    def test_get_context_for_prompt(self):
        from gaap.knowledge.cheat_sheet import ReferenceCard

        card = ReferenceCard(
            library_name="testlib",
            description="A test library",
        )

        knowledge = LibraryKnowledge(
            library_name="testlib",
            description="A test library",
            reference_card=card,
        )

        context = knowledge.get_context_for_prompt()

        assert "testlib" in context
        assert "A test library" in context

    def test_to_dict(self):
        knowledge = LibraryKnowledge(
            library_name="testlib",
            version="1.0.0",
        )

        data = knowledge.to_dict()

        assert data["library_name"] == "testlib"
        assert data["version"] == "1.0.0"
