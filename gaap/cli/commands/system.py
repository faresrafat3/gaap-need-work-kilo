"""
System Commands
"""

import os
import platform
import sys


def cmd_version(args):
    """Show version information"""
    print("\n🤖 GAAP System")
    print("=" * 50)
    print("  Version: 1.0.0")
    print("  Author: GAAP Team")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.release()}")
    print("=" * 50)


def cmd_status(args):
    """Show system status"""
    print("\n📊 GAAP System Status")
    print("=" * 50)

    # Check environment
    env_vars = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "CEREBRAS_API_KEY": os.environ.get("CEREBRAS_API_KEY"),
        "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
    }

    print("\n🔌 API Keys:")
    for key, value in env_vars.items():
        status = "✅ Set" if value else "❌ Not set"
        print(f"  {key}: {status}")

    # Check storage
    from gaap.storage import load_history, load_stats

    stats = load_stats()
    history = load_history(limit=10000)

    print("\n📈 Statistics:")
    print(f"  Total Requests: {stats.get('total_requests', 0)}")
    print(f"  Total Tokens: {stats.get('total_tokens', 0)}")
    print(f"  Total Cost: ${stats.get('total_cost', 0):.4f}")
    print(f"  History Items: {len(history)}")

    # Check config
    from gaap.storage import load_config

    config = load_config()
    print("\n⚙️  Configuration:")
    print(f"  Default Provider: {config.get('default_provider', 'groq')}")
    print(f"  Default Model: {config.get('default_model', 'llama-3.3-70b')}")
    print(f"  Default Budget: ${config.get('default_budget', 10.0):.2f}")

    print("\n" + "=" * 50)


def cmd_doctor(args):
    """Run system diagnostics"""
    print("\n🔍 GAAP System Diagnostics")
    print("=" * 50)

    issues = []

    # Check Python version
    py_version = sys.version_info
    if py_version < (3, 10):
        issues.append("Python 3.10+ required")
    else:
        print("✅ Python version OK")

    # Check required packages
    required_packages = [
        "aiohttp",
        "httpx",
        "pyyaml",
        "structlog",
    ]

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} installed")
        except ImportError:
            issues.append(f"{pkg} not installed")
            print(f"❌ {pkg} not installed")

    # Check GAAP modules
    try:
        from gaap import GAAPEngine

        print("✅ GAAP core module OK")
    except ImportError as e:
        issues.append(f"GAAP import error: {e}")
        print(f"❌ GAAP import error: {e}")

    # Check storage
    try:
        from gaap.storage import get_store

        store = get_store()
        test_data = {"test": "data"}
        store.save("test", test_data)
        loaded = store.load("test")
        if loaded == test_data:
            print("✅ Storage working")
        else:
            issues.append("Storage read/write mismatch")
    except Exception as e:
        issues.append(f"Storage error: {e}")
        print(f"❌ Storage error: {e}")

    # Check API keys
    has_any_key = any(
        [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("CEREBRAS_API_KEY"),
            os.environ.get("MISTRAL_API_KEY"),
        ]
    )

    if has_any_key:
        print("✅ At least one API key configured")
    else:
        issues.append("No API keys configured")
        print("⚠️  No API keys configured")

    # Summary
    print("\n" + "=" * 50)
    if issues:
        print(f"\n⚠️  Found {len(issues)} issue(s):")
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n✅ All checks passed!")

    print("=" * 50)
