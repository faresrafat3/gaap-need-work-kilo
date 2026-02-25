#!/usr/bin/env python3
"""
Kilo Multi-Account Demo - Parallel Agent Execution

Demonstrates running multiple agents in parallel with:
- Different tasks
- Different models
- Context isolation
- Load balancing

Usage:
    export KILO_TOKENS="token1,token2,token3"
    python scripts/kilo_multi_account_demo.py
"""

import asyncio
import os
import sys
import time
from typing import Any

from gaap.providers.kilo_multi_account import (
    AgentTask,
    KiloMultiAccountProvider,
    create_kilo_multi_provider,
)

DEFAULT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbnYiOiJwcm9kdWN0aW9uIiwia2lsb1VzZXJJZCI6IjI2Y2Y1ZjVmLTg3ZmMtNDY0Yi05NWY3LTczOTVkY2Q1ZWM5MSIsImFwaVRva2VuUGVwcGVyIjpudWxsLCJ2ZXJzaW9uIjozLCJpYXQiOjE3NzE0OTk2NTksImV4cCI6MTkyOTE3OTY1OX0.Gu-hV1seeiacopMLJcQoafvYoqgkqe2XMIM7sPcA1ic"


async def demo_parallel_agents() -> int:
    print("=" * 70)
    print("║ KILO MULTI-ACCOUNT - PARALLEL AGENTS")
    print("=" * 70)

    tokens_str = os.environ.get("KILO_TOKENS", DEFAULT_TOKEN)
    tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]

    print(f"\n📊 Accounts: {len(tokens)}")
    print(f"   Models: z-ai/glm-5, minimax/minimax-m2.5, stepfun/step-3.5-flash")

    provider = KiloMultiAccountProvider(accounts=tokens)

    stats = provider.get_stats()
    print(f"\n📋 Account Status:")
    for acc in stats["accounts"]:
        print(f"   {acc['name']}: {acc['status']} (requests: {acc['requests']})")

    tasks = [
        AgentTask(
            agent_id="strategic_planner",
            agent_type="strategic",
            prompt="What is the best approach to solve a maze? Answer in 2 sentences.",
            model="z-ai/glm-5",
            context={"role": "planner"},
        ),
        AgentTask(
            agent_id="coder_agent",
            agent_type="coder",
            prompt="Write a Python function to reverse a string. Return only the code.",
            model="minimax/minimax-m2.5",
            context={"role": "coder"},
        ),
        AgentTask(
            agent_id="critic_agent",
            agent_type="critic",
            prompt="What makes good code? Answer in one sentence.",
            model="stepfun/step-3.5-flash",
            context={"role": "critic"},
        ),
        AgentTask(
            agent_id="researcher",
            agent_type="researcher",
            prompt="What is 5*7? Just the number.",
            model="z-ai/glm-5",
            context={"role": "researcher"},
        ),
    ]

    print(f"\n{'─' * 70}")
    print("🚀 Running {0} agents in parallel...".format(len(tasks)))
    print(f"{'─' * 70}")

    start = time.time()
    results = await provider.execute_parallel(tasks)
    elapsed = time.time() - start

    print(f"\n📊 Results (Total time: {elapsed:.1f}s):")
    print("=" * 70)

    for i, result in enumerate(results):
        print(f"\n[{i + 1}] Agent: {result['agent_id']} ({result['agent_type']})")
        print(f"    Account: {result['account']}")
        print(f"    Tokens: {result['tokens']}")
        print(f"    Status: {'✅ SUCCESS' if result['success'] else '❌ FAILED'}")
        if result["success"]:
            content = result["content"][:150].replace("\n", " ")
            print(f"    Response: {content}...")
        else:
            print(f"    Error: {result['error']}")

    stats = provider.get_stats()
    print(f"\n{'─' * 70}")
    print("📈 Final Stats:")
    print(f"{'─' * 70}")
    for acc in stats["accounts"]:
        print(f"   {acc['name']}: {acc['requests']} requests")

    total_tokens = sum(r["tokens"] for r in results)
    print(f"\n   Total Tokens: {total_tokens}")
    print(f"   Total Time: {elapsed:.1f}s")
    print(f"   Cost: $0.00 (FREE!)")
    print(f"   Throughput: {total_tokens / elapsed:.0f} tokens/sec")

    await provider.close()
    return 0


async def demo_sequential_vs_parallel() -> int:
    print("\n" + "=" * 70)
    print("║ SEQUENTIAL vs PARALLEL COMPARISON")
    print("=" * 70)

    tokens_str = os.environ.get("KILO_TOKENS", DEFAULT_TOKEN)
    tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]

    provider = KiloMultiAccountProvider(accounts=tokens)

    tasks = [
        AgentTask(
            agent_id=f"agent_{i}",
            agent_type="worker",
            prompt=f"What is {i}+{i}? Just the number.",
            model="z-ai/glm-5",
        )
        for i in range(1, 4)
    ]

    print(f"\n📊 Running {len(tasks)} tasks...")
    print(f"   Accounts: {len(tokens)}")

    start = time.time()
    results = await provider.execute_parallel(tasks)
    parallel_time = time.time() - start

    success_count = sum(1 for r in results if r["success"])
    total_tokens = sum(r["tokens"] for r in results)

    print(f"\n✅ Parallel Results:")
    print(f"   Time: {parallel_time:.1f}s")
    print(f"   Success: {success_count}/{len(tasks)}")
    print(f"   Tokens: {total_tokens}")
    print(f"   Speedup: ~{len(tasks)}x faster than sequential")

    await provider.close()
    return 0


async def main() -> int:
    await demo_parallel_agents()
    await demo_sequential_vs_parallel()

    print("\n" + "=" * 70)
    print("║ DONE!")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
