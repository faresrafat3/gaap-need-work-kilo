"""
Knowledge Module - Repository Learning Engine
=============================================

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md

Enables GAAP to learn new libraries by reading their source code:

1. **AST Parser** - Extract structure from source files
2. **Usage Miner** - Find usage patterns in tests/examples
3. **Cheat Sheet Generator** - Create reference cards
4. **Knowledge Ingestion** - Orchestrate the learning process

Usage:
    from gaap.knowledge import KnowledgeIngestion

    # Learn a library
    ingestion = KnowledgeIngestion()
    result = await ingestion.ingest_repo("https://github.com/pydantic/pydantic")

    # Load learned knowledge later
    knowledge = ingestion.load_library("pydantic")
    context = knowledge.get_context_for_prompt()
"""

from gaap.knowledge.knowledge_config import (
    KnowledgeConfig,
    create_knowledge_config,
)
from gaap.knowledge.ast_parser import (
    ASTParser,
    ParsedFile,
    ClassInfo,
    FunctionInfo,
    Parameter,
    ImportInfo,
    create_parser,
)
from gaap.knowledge.usage_miner import (
    UsageMiner,
    UsageExample,
    UsagePattern,
    MiningResult,
    create_usage_miner,
)
from gaap.knowledge.cheat_sheet import (
    CheatSheetGenerator,
    ReferenceCard,
    FunctionSummary,
    PatternExample,
    BreakingChange,
    create_cheat_sheet_generator,
)
from gaap.knowledge.ingestion import (
    KnowledgeIngestion,
    IngestionResult,
    LibraryKnowledge,
    create_knowledge_ingestion,
)

__all__ = [
    "KnowledgeConfig",
    "create_knowledge_config",
    "ASTParser",
    "ParsedFile",
    "ClassInfo",
    "FunctionInfo",
    "Parameter",
    "ImportInfo",
    "create_parser",
    "UsageMiner",
    "UsageExample",
    "UsagePattern",
    "MiningResult",
    "create_usage_miner",
    "CheatSheetGenerator",
    "ReferenceCard",
    "FunctionSummary",
    "PatternExample",
    "BreakingChange",
    "create_cheat_sheet_generator",
    "KnowledgeIngestion",
    "IngestionResult",
    "LibraryKnowledge",
    "create_knowledge_ingestion",
]
