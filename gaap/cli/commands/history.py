"""
History Commands
"""

import json
from datetime import datetime
from typing import Any


def cmd_history(args: Any) -> None:
    """History management commands"""
    action = args.action if hasattr(args, "action") else "list"

    if action == "list":
        _list_history(args.limit if hasattr(args, "limit") else 20)
    elif action == "show":
        _show_item(args.id if hasattr(args, "id") else None)
    elif action == "clear":
        _clear_history()
    elif action == "search":
        _search_history(args.query if hasattr(args, "query") else None)
    elif action == "export":
        _export_history(args.file if hasattr(args, "file") else None)
    else:
        _list_history()


def _list_history(limit: int = 20) -> None:
    """List conversation history"""
    from gaap.storage import load_history

    history = load_history(limit=limit)

    if not history:
        print("\n📜 No conversation history")
        return

    print(f"\n📜 Conversation History (last {len(history)} messages)")
    print("=" * 70)

    for i, item in enumerate(reversed(history[-limit:])):
        timestamp = item.get("timestamp", "Unknown")
        role = item.get("role", "unknown")
        content = item.get("content", "")

        role_icon = "👤" if role == "user" else "🤖" if role == "assistant" else "📝"
        content_preview = content[:50] + "..." if len(content) > 50 else content

        print(f"\n{i + 1}. [{timestamp[:16]}] {role_icon} {role}")
        print(f"   {content_preview}")

        if item.get("provider"):
            print(f"   Provider: {item['provider']}, Model: {item.get('model', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"Total messages: {len(history)}")


def _show_item(item_id: str | None = None) -> None:
    """Show a specific history item"""
    if not item_id:
        print("❌ Item ID required: gaap history show <id>")
        return

    from gaap.storage import get_store

    store = get_store()
    item = store.get_by_id("history", item_id)

    if not item:
        print(f"❌ Item not found: {item_id}")
        return

    print(f"\n📜 History Item: {item_id}")
    print("=" * 70)
    print(f"  Timestamp: {item.get('timestamp', 'Unknown')}")
    print(f"  Role: {item.get('role', 'unknown')}")
    print("  Content:")
    print("-" * 50)
    print(item.get("content", ""))
    print("-" * 50)

    if item.get("provider"):
        print(f"  Provider: {item['provider']}")
    if item.get("model"):
        print(f"  Model: {item['model']}")
    if item.get("tokens"):
        print(f"  Tokens: {item['tokens']}")
    if item.get("cost"):
        print(f"  Cost: ${item['cost']:.4f}")


def _clear_history() -> None:
    """Clear all history"""
    from gaap.storage import get_store

    print("\n⚠️  This will delete all conversation history.")
    confirm = input("Are you sure? (yes/no): ")

    if confirm.lower() == "yes":
        store = get_store()
        store.save("history", [])
        print("✅ History cleared")
    else:
        print("❌ Cancelled")


def _search_history(query: str | None = None) -> None:
    """Search history"""
    if not query:
        print("❌ Query required: gaap history search <query>")
        return

    from gaap.storage import load_history

    history = load_history(limit=1000)

    results = []
    for item in history:
        content = item.get("content", "")
        if query.lower() in content.lower():
            results.append(item)

    if not results:
        print(f"\n🔍 No results for: '{query}'")
        return

    print(f"\n🔍 Search Results: '{query}' ({len(results)} found)")
    print("=" * 70)

    for i, item in enumerate(results[:20]):
        content = item.get("content", "")
        role = item.get("role", "unknown")
        timestamp = item.get("timestamp", "")

        print(f"\n{i + 1}. [{timestamp[:16]}] {role}")
        print(f"   {content[:80]}...")


def _export_history(file_path: str | None = None) -> None:
    """Export history to file"""
    from pathlib import Path

    from gaap.storage import load_history

    history = load_history()

    if not file_path:
        file_path = f"gaap_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    path = Path(file_path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"✅ Exported {len(history)} messages to {file_path}")
