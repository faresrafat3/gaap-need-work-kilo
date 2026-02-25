"""
Debt CLI Command - Technical Debt Management
============================================

CLI interface for the technical debt agent.

Usage:
    gaap debt scan              # Scan for debt
    gaap debt report            # Generate report
    gaap debt list              # List all debt items
    gaap debt propose <id>      # Create refactoring proposal
    gaap debt stats             # Show statistics

Implements: docs/evolution_plan_2026/29_TECHNICAL_DEBT_AGENT.md
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from gaap.maintenance import (
    DebtConfig,
    DebtItem,
    DebtScanner,
    InterestCalculator,
    ProposalStatus,
    RefinancingEngine,
    create_debt_config,
)


def cmd_debt(args: Any) -> None:
    """Handle debt command."""
    action = getattr(args, "action", "scan")

    if action == "scan":
        cmd_debt_scan(args)
    elif action == "report":
        cmd_debt_report(args)
    elif action == "list":
        cmd_debt_list(args)
    elif action == "propose":
        asyncio.run(cmd_debt_propose(args))
    elif action == "stats":
        cmd_debt_stats(args)
    elif action == "proposals":
        cmd_debt_proposals(args)
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)


def cmd_debt_scan(args: Any) -> None:
    """Scan codebase for technical debt."""
    config = _get_config(args)
    scanner = DebtScanner(config=config)

    path = getattr(args, "path", ".") or "."
    root_path = Path(path)

    print(f"\n🔍 Scanning {root_path} for technical debt...\n")

    result = scanner.scan_directory(root_path)

    print(f"📁 Scanned {result.scanned_files} files")
    print(f"🐛 Found {result.total_debt_items} debt items")
    print(f"⏱️  Scan time: {result.scan_time_ms:.1f}ms")

    if result.errors:
        print(f"\n⚠️  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"   - {error}")

    if result.by_type:
        print("\n📊 By Type:")
        for dtype, count in sorted(result.by_type.items(), key=lambda x: -x[1]):
            print(f"   {dtype}: {count}")

    if result.by_priority:
        print("\n🚨 By Priority:")
        for priority, count in sorted(result.by_priority.items(), key=lambda x: -x[1]):
            print(f"   {priority}: {count}")

    if result.items:
        output_file = Path(config.storage_path) / "scan_results.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n💾 Results saved to {output_file}")


def cmd_debt_report(args: Any) -> None:
    """Generate detailed debt report with interest scores."""
    config = _get_config(args)

    results_file = Path(config.storage_path) / "scan_results.json"
    if not results_file.exists():
        print("No scan results found. Run 'gaap debt scan' first.")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    items = [DebtItem.from_dict(item) for item in data.get("items", [])]

    calculator = InterestCalculator(config=config)
    report = calculator.generate_report(items)

    print("\n" + "=" * 60)
    print("📊 TECHNICAL DEBT INTEREST REPORT")
    print("=" * 60)

    print(f"\n📈 Total Items: {report.total_items}")
    print(f"💰 Total Interest: {report.total_interest:.2f}")
    print(f"🔴 Critical Interest: {report.critical_interest_count}")
    print(f"🟠 High Interest: {report.high_interest_count}")

    if report.by_type:
        print("\n📊 Interest by Type:")
        for dtype, interest in sorted(report.by_type.items(), key=lambda x: -x[1]):
            print(f"   {dtype}: {interest:.2f}")

    if report.top_items:
        print("\n🔝 Top 10 Debt Items:")
        for i, (item, interest) in enumerate(report.top_items[:10], 1):
            print(f"\n   {i}. [{item.type.name}] {item.file_path}:{item.line_number}")
            print(f"      Interest: {interest:.2f} | Priority: {item.priority.name}")
            print(f"      {item.message[:60]}...")


def cmd_debt_list(args: Any) -> None:
    """List all debt items."""
    config = _get_config(args)

    results_file = Path(config.storage_path) / "scan_results.json"
    if not results_file.exists():
        print("No scan results found. Run 'gaap debt scan' first.")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    items = [DebtItem.from_dict(item) for item in data.get("items", [])]

    calculator = InterestCalculator(config=config)
    prioritized = calculator.prioritize(items)

    limit = getattr(args, "limit", 20)
    debt_type = getattr(args, "type", None)

    if debt_type:
        prioritized = [i for i in prioritized if i.type.name == debt_type.upper()]

    print(f"\n📋 Debt Items ({len(prioritized)} total):\n")

    for item in prioritized[:limit]:
        priority_emoji = {
            "CRITICAL": "🔴",
            "HIGH": "🟠",
            "MEDIUM": "🟡",
            "LOW": "🟢",
            "INFO": "ℹ️",
        }.get(item.priority.name, "⚪")

        print(f"{priority_emoji} [{item.type.name}] {item.file_path}:{item.line_number}")
        print(f"   {item.message[:60]}...")
        print(f"   Interest: {item.interest_score:.2f}")
        print()


def cmd_debt_stats(args: Any) -> None:
    """Show debt statistics."""
    config = _get_config(args)

    results_file = Path(config.storage_path) / "scan_results.json"
    if not results_file.exists():
        print("No scan results found. Run 'gaap debt scan' first.")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    items = [DebtItem.from_dict(item) for item in data.get("items", [])]

    total = len(items)
    by_type: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    total_interest = 0.0

    for item in items:
        by_type[item.type.name] = by_type.get(item.type.name, 0) + 1
        by_priority[item.priority.name] = by_priority.get(item.priority.name, 0) + 1
        total_interest += item.interest_score

    print("\n📊 Technical Debt Statistics\n")
    print(f"Total Debt Items: {total}")
    print(f"Total Interest Score: {total_interest:.2f}")
    print(f"Average Interest: {total_interest / max(total, 1):.2f}")

    print("\nBy Type:")
    for dtype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        bar = "█" * int(pct / 5)
        print(f"  {dtype:15} {count:4} {bar} {pct:.0f}%")

    print("\nBy Priority:")
    for priority, count in sorted(by_priority.items(), key=lambda x: -x[1]):
        pct = 100 * count / max(total, 1)
        bar = "█" * int(pct / 5)
        print(f"  {priority:15} {count:4} {bar} {pct:.0f}%")


async def cmd_debt_propose(args: Any) -> None:
    """Create a refactoring proposal."""
    config = _get_config(args)

    debt_id = getattr(args, "id", None)
    if not debt_id:
        print("Error: --id required for proposal")
        sys.exit(1)

    results_file = Path(config.storage_path) / "scan_results.json"
    if not results_file.exists():
        print("No scan results found. Run 'gaap debt scan' first.")
        sys.exit(1)

    with open(results_file) as f:
        data = json.load(f)

    items = [DebtItem.from_dict(item) for item in data.get("items", [])]

    target = None
    for item in items:
        if item.id == debt_id or item.id.startswith(debt_id):
            target = item
            break

    if not target:
        print(f"Debt item not found: {debt_id}")
        sys.exit(1)

    print(f"\n🔨 Creating proposal for: {target.id}")
    print(f"   Type: {target.type.name}")
    print(f"   File: {target.file_path}:{target.line_number}")
    print(f"   Message: {target.message[:50]}...")

    llm_provider = _get_llm_provider(args)

    engine = RefinancingEngine(
        config=config,
        llm_provider=llm_provider,
    )

    proposal = await engine.propose(target, use_llm=llm_provider is not None)

    print(f"\n✅ Proposal created: {proposal.id}")
    print(f"   Branch: {proposal.branch_name}")
    print(f"   LLM Generated: {proposal.llm_generated}")
    print(f"   Confidence: {proposal.confidence:.2f}")
    print(f"\n📝 Proposed Fix:\n")
    print(f"   {proposal.proposed_fix[:500]}...")


def cmd_debt_proposals(args: Any) -> None:
    """List existing proposals."""
    config = _get_config(args)

    engine = RefinancingEngine(config=config)

    status_filter = getattr(args, "status", None)
    status = None
    if status_filter:
        try:
            status = ProposalStatus[status_filter.upper()]
        except KeyError:
            pass

    proposals = engine.list_proposals(status=status)

    if not proposals:
        print("No proposals found.")
        return

    print(f"\n📋 Proposals ({len(proposals)} total):\n")

    for proposal in proposals:
        status_emoji = {
            "PENDING": "⏳",
            "IN_PROGRESS": "🔄",
            "READY_FOR_REVIEW": "👀",
            "APPROVED": "✅",
            "MERGED": "🔀",
            "REJECTED": "❌",
            "ABANDONED": "🗑️",
        }.get(proposal.status.name, "❓")

        print(f"{status_emoji} [{proposal.status.name}] {proposal.id}")
        print(f"   Debt: {proposal.debt_item.type.name} in {proposal.debt_item.file_path}")
        print(f"   Branch: {proposal.branch_name}")
        print(f"   Created: {proposal.created_at.strftime('%Y-%m-%d %H:%M')}")
        if proposal.test_results:
            passed = proposal.test_results.get("passed", False)
            print(f"   Tests: {'✅ Passed' if passed else '❌ Failed'}")
        print()


def _get_config(args: Any) -> DebtConfig:
    """Get debt configuration from args."""
    preset = getattr(args, "preset", "default")

    if preset == "conservative":
        return DebtConfig.conservative()
    elif preset == "aggressive":
        return DebtConfig.aggressive()
    elif preset == "development":
        return DebtConfig.development()
    else:
        return DebtConfig()


def _get_llm_provider(args: Any) -> Any:
    """Get LLM provider if available."""
    no_llm = getattr(args, "no_llm", False)
    if no_llm:
        return None

    try:
        from gaap.providers import get_provider

        return get_provider("groq")
    except Exception:
        return None
