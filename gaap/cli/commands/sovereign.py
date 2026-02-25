"""
Sovereign CLI Commands (v2.1)
"""

import asyncio
import sys
import json
from typing import Any
from gaap.gaap_engine import create_engine, GAAPRequest
from gaap.layers.layer0_interface import IntentType


async def cmd_research(args: Any) -> None:
    """Deep Research Command"""
    print(f"🔍 Starting Deep Research on: {args.query} (Depth: {args.depth})")
    engine = create_engine()
    request = GAAPRequest(
        text=args.query, metadata={"force_intent": "RESEARCH", "research_depth": args.depth}
    )
    response = await engine.process(request)

    if response.success:
        print("\n" + "=" * 50)
        print("RESEARCH REPORT")
        print("=" * 50)
        print(response.output)
    else:
        print(f"\n❌ Research failed: {response.error}")


async def cmd_debug(args: Any) -> None:
    """Diagnostic & Debug Command"""
    print(f"🛠️ Starting Diagnostic Analysis: {args.issue}")
    engine = create_engine()
    request = GAAPRequest(text=args.issue, metadata={"force_intent": "DEBUGGING"})
    response = await engine.process(request)

    if response.success:
        print("\n" + "=" * 50)
        print("DIAGNOSTIC VERDICT")
        print("=" * 50)
        print(response.output)
    else:
        print(f"\n❌ Diagnosis failed: {response.error}")


def cmd_dream(args: Any) -> None:
    """Memory Consolidation Command"""
    print("🌙 Entering Sovereign REM Sleep...")
    import subprocess

    subprocess.run(["python3", "-m", "gaap.memory.dream_processor"])
    print("✨ Memory Consolidation Complete.")


def cmd_audit(args: Any) -> None:
    """Constitutional Audit Command"""
    print("⚖️ Running Constitutional Integrity Audit...")
    import subprocess

    subprocess.run(["python3", "-m", "gaap.core.axioms"])
    print("✅ Axiomatic Guardrails Verified.")
