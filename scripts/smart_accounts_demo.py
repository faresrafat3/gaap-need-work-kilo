#!/usr/bin/env python3
"""
Smart Account Manager Demo

Shows how to use multiple AI providers with:
- Automatic load balancing
- Token health monitoring
- Multi-provider support (Kilo + Kimi)
- Parallel execution

Usage:
    # Set up accounts
    export KILO_TOKENS="token1,token2"
    export KIMI_TOKENS="token1,token2,token3"

    python scripts/smart_accounts_demo.py
"""

import asyncio
import os
import sys
import time
from pprint import pprint

DEFAULT_KILO_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbnYiOiJwcm9kdWN0aW9uIiwia2lsb1VzZXJJZCI6IjI2Y2Y1ZjVmLTg3ZmMtNDY0Yi05NWY3LTczOTVkY2Q1ZWM5MSIsImFwaVRva2VuUGVwcGVyIjpudWxsLCJ2ZXJzaW9uIjozLCJpYXQiOjE3NzE0OTk2NTksImV4cCI6MTkyOTE3OTY1OX0.Gu-hV1seeiacopMLJcQoafvYoqgkqe2XMIM7sPcA1ic"


async def demo_basic() -> None:
    print("=" * 70)
    print("║ SMART ACCOUNT MANAGER - Basic Usage")
    print("=" * 70)

    from gaap.providers.smart_accounts import SmartAccountManager, create_smart_manager

    manager = create_smart_manager()

    if not manager.total_accounts:
        print("\n⚠️ No accounts found in env, adding demo account...")
        manager.add_account(
            token=os.environ.get("KILO_TOKEN", DEFAULT_KILO_TOKEN),
            provider="kilo",
            name="demo_kilo",
        )

    print(f"\n📊 Accounts loaded: {manager.total_accounts}")
    for acc in manager._accounts:
        print(f"   - {acc.name} ({acc.provider})")
        if acc.days_until_expiry():
            print(f"     Expires in: {acc.days_until_expiry():.0f} days")

    print(f"\n{'─' * 70}")
    print("🚀 Executing single request...")
    print(f"{'─' * 70}")

    result = await manager.execute(
        prompt="What is 7+7? Just the number.", model="auto", task_type="fast"
    )

    print(f"\n  Account: {result.account}")
    print(f"  Provider: {result.provider}")
    print(f"  Model: {result.model}")
    print(f"  Response: {result.content[:100]}")
    print(f"  Tokens: {result.tokens}")
    print(f"  Time: {result.elapsed:.1f}s")
    print(f"  Status: {'✅' if result.success else '❌'}")

    await manager.close()


async def demo_parallel() -> None:
    print(f"\n{'=' * 70}")
    print("║ PARALLEL EXECUTION")
    print("=" * 70)

    from gaap.providers.smart_accounts import create_smart_manager

    manager = create_smart_manager()

    if not manager.total_accounts:
        manager.add_account(token=os.environ.get("KILO_TOKEN", DEFAULT_KILO_TOKEN), provider="kilo")

    prompts = [
        "What is 1+1? Just the number.",
        "What is 2+2? Just the number.",
        "What is 3+3? Just the number.",
    ]

    print(f"\n📊 Running {len(prompts)} prompts in parallel...")

    start = time.time()
    results = await manager.execute_parallel(prompts)
    elapsed = time.time() - start

    print(f"\n{'─' * 70}")
    for i, r in enumerate(results):
        print(f"[{i + 1}] {r.account}: {r.content[:50]}... ({r.tokens} tokens)")

    total_tokens = sum(r.tokens for r in results)
    print(f"\n{'─' * 70}")
    print(f"Total: {total_tokens} tokens in {elapsed:.1f}s")
    print(f"Throughput: {total_tokens / elapsed:.0f} tokens/sec")

    await manager.close()


async def demo_stats() -> None:
    print(f"\n{'=' * 70}")
    print("║ ACCOUNT STATISTICS")
    print("=" * 70)

    from gaap.providers.smart_accounts import create_smart_manager

    manager = create_smart_manager()

    if not manager.total_accounts:
        manager.add_account(token=os.environ.get("KILO_TOKEN", DEFAULT_KILO_TOKEN), provider="kilo")

    print("\n📊 Account Statistics:")
    stats = manager.get_stats()
    print(f"   Total accounts: {stats['total_accounts']}")
    print(f"   Active accounts: {stats['active_accounts']}")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total tokens: {stats['total_tokens']}")

    print("\n📋 Per-Account Details:")
    for acc in stats["accounts"]:
        print(f"   {acc['name']}:")
        print(f"      Provider: {acc['provider']}")
        print(f"      Status: {acc['status']}")
        print(f"      Requests: {acc['requests']}")
        print(f"      Tokens: {acc['tokens']}")
        if acc["days_until_expiry"]:
            print(f"      Expires in: {acc['days_until_expiry']:.0f} days")

    await manager.close()


async def demo_save_config() -> None:
    print(f"\n{'=' * 70}")
    print("║ CONFIG MANAGEMENT")
    print("=" * 70)

    from gaap.providers.smart_accounts import SmartAccountManager

    manager = SmartAccountManager()

    manager.add_account(token="demo_token_1", provider="kilo", name="kilo_main")

    manager.add_account(token="demo_token_2", provider="kimi", name="kimi_main")

    print(f"\n💾 Saving {manager.total_accounts} accounts to config...")

    saved = manager.save_to_config()
    if saved:
        print("   ✅ Config saved to ~/.gaap/accounts.json")
    else:
        print("   ❌ Failed to save config")

    print("\n📂 Config file contents:")
    import os

    config_path = os.path.expanduser("~/.gaap/accounts.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            print(f.read())


async def main() -> int:
    print("\n" + "=" * 70)
    print("║ SMART ACCOUNT MANAGER DEMO")
    print("=" * 70)

    await demo_basic()
    await demo_parallel()
    await demo_stats()
    await demo_save_config()

    print(f"\n{'=' * 70}")
    print("║ DONE!")
    print("=" * 70)

    print("""
💡 Key Points:

1. JWT tokens are valid for 5 YEARS - no refresh needed!
2. Add accounts via:
   - Environment: KILO_TOKENS="t1,t2" KIMI_TOKENS="t1,t2"
   - Config file: ~/.gaap/accounts.json
   - Code: manager.add_account(token, provider, name)

3. Automatic load balancing across accounts
4. Health monitoring for each account
5. Expiration tracking (auto-detected from JWT)

6. To add more Kilo accounts:
   - Open incognito browser
   - Go to kilo.ai
   - Sign up with different Google/GitHub
   - Get JWT token from cookies/CLI
""")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
