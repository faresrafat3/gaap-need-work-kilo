"""
Learn CLI Command - Repository Learning Interface
=================================================

CLI interface for the knowledge ingestion engine.

Usage:
    gaap learn <repo_url>          # Learn from GitHub repo
    gaap learn <local_path>        # Learn from local directory
    gaap knowledge list            # List learned libraries
    gaap knowledge show <lib>      # Show library details
    gaap knowledge delete <lib>    # Delete learned library

Implements: docs/evolution_plan_2026/28_KNOWLEDGE_INGESTION.md
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

from gaap.knowledge import (
    KnowledgeIngestion,
    KnowledgeConfig,
    create_knowledge_config,
)


def cmd_learn(args: Any) -> None:
    """Handle learn command - ingest a repository."""
    source = getattr(args, "source", None)

    if not source:
        print("Error: No source specified")
        print("Usage: gaap learn <repo_url_or_path>")
        sys.exit(1)

    config = _get_config(args)
    ingestion = KnowledgeIngestion(config=config)

    library_name = getattr(args, "name", None)
    description = getattr(args, "description", None)

    print(f"\n📚 Learning from: {source}")
    print("=" * 50)

    result = asyncio.run(
        ingestion.ingest_repo(
            source=source,
            library_name=library_name,
            description=description,
        )
    )

    if result.success:
        print(f"\n✅ Successfully learned: {result.library_name}")
        print(f"\n📊 Statistics:")
        print(f"   Files parsed:     {result.files_parsed}")
        print(f"   Functions found:  {result.functions_found}")
        print(f"   Classes found:    {result.classes_found}")
        print(f"   Examples mined:   {result.examples_mined}")
        print(f"   Patterns found:   {result.patterns_identified}")
        print(f"\n⏱️  Time:")
        print(f"   Parsing:    {result.parse_time_ms:.0f}ms")
        print(f"   Mining:     {result.mine_time_ms:.0f}ms")
        print(f"   Total:      {result.total_time_ms:.0f}ms")
        print(f"\n💾 Saved to: {result.output_path}")

        if result.reference_card:
            print(f"\n📋 Reference Card:")
            print(f"   Top functions: {len(result.reference_card.top_functions)}")
            print(f"   Top classes:   {len(result.reference_card.top_classes)}")
            print(f"   Patterns:      {len(result.reference_card.common_patterns)}")

            if result.reference_card.top_functions:
                print(f"\n   🔝 Top functions:")
                for func in result.reference_card.top_functions[:5]:
                    print(f"      - {func.signature}")
    else:
        print(f"\n❌ Failed to learn: {result.library_name}")
        for error in result.errors:
            print(f"   Error: {error}")
        sys.exit(1)


def cmd_knowledge(args: Any) -> None:
    """Handle knowledge command - manage learned libraries."""
    action = getattr(args, "action", "list")

    config = _get_config(args)
    ingestion = KnowledgeIngestion(config=config)

    if action == "list":
        cmd_knowledge_list(ingestion)
    elif action == "show":
        cmd_knowledge_show(ingestion, args)
    elif action == "delete":
        cmd_knowledge_delete(ingestion, args)
    elif action == "context":
        cmd_knowledge_context(ingestion, args)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


def cmd_knowledge_list(ingestion: KnowledgeIngestion) -> None:
    """List all learned libraries."""
    libraries = ingestion.list_libraries()

    if not libraries:
        print("\n📚 No libraries learned yet.")
        print("   Use 'gaap learn <repo>' to learn a library.")
        return

    print(f"\n📚 Learned Libraries ({len(libraries)}):\n")

    for lib_name in libraries:
        knowledge = ingestion.load_library(lib_name)
        if knowledge:
            desc = knowledge.description or "No description"
            desc_short = desc[:50] + "..." if len(desc) > 50 else desc
            print(f"  📦 {lib_name}")
            print(f"     {desc_short}")
            if knowledge.reference_card:
                print(
                    f"     Functions: {knowledge.reference_card.total_functions_found}, "
                    f"Classes: {knowledge.reference_card.total_classes_found}"
                )
            print()


def cmd_knowledge_show(ingestion: KnowledgeIngestion, args: Any) -> None:
    """Show details of a learned library."""
    library_name = getattr(args, "library", None)

    if not library_name:
        print("Error: No library specified")
        print("Usage: gaap knowledge show <library>")
        sys.exit(1)

    knowledge = ingestion.load_library(library_name)

    if not knowledge:
        print(f"Error: Library not found: {library_name}")
        sys.exit(1)

    print(f"\n📚 {knowledge.library_name}")
    print("=" * 50)

    if knowledge.description:
        print(f"\n{knowledge.description}")

    if knowledge.version:
        print(f"\nVersion: {knowledge.version}")

    if knowledge.reference_card:
        rc = knowledge.reference_card

        print(f"\n📊 Statistics:")
        print(f"   Files analyzed:    {rc.total_files_analyzed}")
        print(f"   Functions found:   {rc.total_functions_found}")
        print(f"   Classes found:     {rc.total_classes_found}")

        if rc.top_functions:
            print(f"\n🔝 Top Functions:")
            for func in rc.top_functions[:10]:
                print(f"   • {func.signature}")
                if func.docstring:
                    doc_short = func.docstring.split("\n")[0][:60]
                    print(f"     {doc_short}...")

        if rc.top_classes:
            print(f"\n🏗️  Key Classes:")
            for cls in rc.top_classes[:5]:
                bases = f"({', '.join(cls['bases'])})" if cls.get("bases") else ""
                print(f"   • {cls['name']}{bases}")
                print(f"     {cls['public_methods']} public methods")

        if rc.common_patterns:
            print(f"\n🔄 Common Patterns:")
            for pattern in rc.common_patterns[:3]:
                print(f"   • {pattern.description}")

        if rc.imports_used:
            print(f"\n📦 Common Imports:")
            print(f"   {', '.join(rc.imports_used[:10])}")


def cmd_knowledge_delete(ingestion: KnowledgeIngestion, args: Any) -> None:
    """Delete a learned library."""
    library_name = getattr(args, "library", None)

    if not library_name:
        print("Error: No library specified")
        print("Usage: gaap knowledge delete <library>")
        sys.exit(1)

    if ingestion.delete_library(library_name):
        print(f"✅ Deleted library: {library_name}")
    else:
        print(f"❌ Library not found: {library_name}")
        sys.exit(1)


def cmd_knowledge_context(ingestion: KnowledgeIngestion, args: Any) -> None:
    """Generate context for LLM prompts."""
    library_name = getattr(args, "library", None)
    max_tokens = getattr(args, "max_tokens", 4000)

    if not library_name:
        print("Error: No library specified")
        print("Usage: gaap knowledge context <library>")
        sys.exit(1)

    knowledge = ingestion.load_library(library_name)

    if not knowledge:
        print(f"Error: Library not found: {library_name}")
        sys.exit(1)

    context = knowledge.get_context_for_prompt(max_tokens=max_tokens)
    print(context)


def _get_config(args: Any) -> KnowledgeConfig:
    """Get knowledge configuration from args."""
    preset = getattr(args, "preset", "default")
    return create_knowledge_config(preset)
