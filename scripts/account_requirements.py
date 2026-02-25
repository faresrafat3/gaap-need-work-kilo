#!/usr/bin/env python3
"""
Account Requirements Calculator for GAAP

Estimates how many accounts needed for:
- Context isolation per task
- Comfortable rate limits
- High quality results
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class PipelineStage:
    name: str
    llm_calls: int
    parallel: bool = False
    context_size: str = "medium"  # small, medium, large


@dataclass
class AccountCapacity:
    provider: str
    requests_per_minute: int
    requests_per_day: int
    tokens_per_minute: int
    tokens_per_day: int
    accounts_available: int


def calculate_requirements(
    pipeline_stages: list[PipelineStage],
    iterations_per_day: int = 100,
    quality_level: Literal["basic", "standard", "high"] = "high",
) -> dict:
    """Calculate account requirements."""

    # Provider capacities (estimated)
    kimi_capacity = AccountCapacity(
        provider="kimi",
        requests_per_minute=10,
        requests_per_day=500,
        tokens_per_minute=10000,
        tokens_per_day=100000,
        accounts_available=9,
    )

    kilo_capacity = AccountCapacity(
        provider="kilo",
        requests_per_minute=20,
        requests_per_day=1000,
        tokens_per_minute=20000,
        tokens_per_day=200000,
        accounts_available=1,  # Currently have
    )

    # Calculate total LLM calls
    total_sequential_calls = sum(s.llm_calls for s in pipeline_stages)
    max_parallel_calls = max((s.llm_calls for s in pipeline_stages if s.parallel), default=0)

    # Multipliers for quality
    quality_multiplier = {"basic": 1.0, "standard": 1.5, "high": 2.0}[quality_level]

    # Daily requirements
    daily_requests = total_sequential_calls * iterations_per_day * quality_multiplier
    daily_tokens = daily_requests * 1000  # Assume 1000 tokens avg per request

    # Peak requirements (parallel execution)
    peak_requests_per_minute = max_parallel_calls * 2 if max_parallel_calls else 5

    # Accounts needed
    kimi_accounts_for_rpm = peak_requests_per_minute / kimi_capacity.requests_per_minute
    kimi_accounts_for_rpd = daily_requests / kimi_capacity.requests_per_day
    kimi_needed = max(1, int(min(kimi_accounts_for_rpm, kimi_accounts_for_rpd)) + 1)

    kilo_accounts_for_rpm = peak_requests_per_minute / kilo_capacity.requests_per_minute
    kilo_accounts_for_rpd = daily_requests / kilo_capacity.requests_per_day
    kilo_needed = max(1, int(min(kilo_accounts_for_rpm, kilo_accounts_for_rpd)) + 1)

    # Context isolation
    # Each agent/task needs isolated context
    context_isolation_accounts = len([s for s in pipeline_stages if s.parallel])

    return {
        "pipeline": {
            "total_stages": len(pipeline_stages),
            "sequential_calls": total_sequential_calls,
            "max_parallel": max_parallel_calls,
        },
        "daily_usage": {
            "iterations": iterations_per_day,
            "total_requests": daily_requests,
            "total_tokens": daily_tokens,
            "quality_level": quality_level,
        },
        "kimi": {
            "current_accounts": kimi_capacity.accounts_available,
            "minimum_needed": kimi_needed,
            "recommended": kimi_needed * 2,
            "ideal": kimi_needed * 3,
        },
        "kilo": {
            "current_accounts": kilo_capacity.accounts_available,
            "minimum_needed": kilo_needed,
            "recommended": kilo_needed * 2,
            "ideal": kilo_needed * 3,
        },
        "total": {
            "current": kimi_capacity.accounts_available + kilo_capacity.accounts_available,
            "minimum": kimi_needed + kilo_needed,
            "recommended": (kimi_needed + kilo_needed) * 2,
            "ideal": (kimi_needed + kilo_needed) * 3 + context_isolation_accounts,
        },
        "context_isolation": {
            "parallel_stages": context_isolation_accounts,
            "accounts_for_isolation": context_isolation_accounts * 2,
        },
    }


def print_report(req: dict) -> None:
    """Print formatted report."""
    print("=" * 70)
    print("║ GAAP ACCOUNT REQUIREMENTS ANALYSIS")
    print("=" * 70)

    print("\n📊 PIPELINE ANALYSIS")
    print("-" * 40)
    print(f"  Total stages: {req['pipeline']['total_stages']}")
    print(f"  Sequential LLM calls: {req['pipeline']['sequential_calls']}")
    print(f"  Max parallel calls: {req['pipeline']['max_parallel']}")

    print("\n📈 DAILY USAGE ESTIMATE")
    print("-" * 40)
    print(f"  Iterations/day: {req['daily_usage']['iterations']}")
    print(f"  Total requests: {req['daily_usage']['total_requests']:.0f}")
    print(f"  Total tokens: {req['daily_usage']['total_tokens']:.0f}")
    print(f"  Quality level: {req['daily_usage']['quality_level']}")

    print("\n🤖 KIMI ACCOUNTS")
    print("-" * 40)
    print(f"  Current: {req['kimi']['current_accounts']}")
    print(f"  Minimum needed: {req['kimi']['minimum_needed']}")
    print(f"  Recommended: {req['kimi']['recommended']}")
    print(f"  Ideal: {req['kimi']['ideal']}")

    print("\n🔵 KILO ACCOUNTS")
    print("-" * 40)
    print(f"  Current: {req['kilo']['current_accounts']}")
    print(f"  Minimum needed: {req['kilo']['minimum_needed']}")
    print(f"  Recommended: {req['kilo']['recommended']}")
    print(f"  Ideal: {req['kilo']['ideal']}")

    print("\n📋 TOTAL ACCOUNTS")
    print("-" * 40)
    print(f"  Current: {req['total']['current']}")
    print(f"  Minimum: {req['total']['minimum']}")
    print(f"  Recommended: {req['total']['recommended']}")
    print(f"  Ideal (with context isolation): {req['total']['ideal']}")

    print("\n🔒 CONTEXT ISOLATION")
    print("-" * 40)
    print(f"  Parallel stages: {req['context_isolation']['parallel_stages']}")
    print(f"  Accounts for full isolation: {req['context_isolation']['accounts_for_isolation']}")

    print("\n" + "=" * 70)
    print("║ RECOMMENDATION")
    print("=" * 70)

    current = req["total"]["current"]
    ideal = req["total"]["ideal"]

    if current < req["total"]["minimum"]:
        print(f"\n⚠️ CRITICAL: Need at least {req['total']['minimum'] - current} more accounts!")
    elif current < req["total"]["recommended"]:
        print(f"\n💡 Good: Add {req['total']['recommended'] - current} more accounts for comfort")
    elif current < ideal:
        print(f"\n✅ Great: Add {ideal - current} more accounts for ideal setup")
    else:
        print("\n🎉 Perfect: You have enough accounts!")

    # How to get more accounts
    print("\n📝 HOW TO GET MORE ACCOUNTS:")
    print("-" * 40)
    print("""
  KIMI (9 accounts currently):
    - Kimi has built-in 9 accounts rotation
    - Already optimized for multi-account

  KILO (1 account currently):
    - Open incognito browser
    - Go to: https://kilo.ai
    - Sign up with different Google/GitHub account
    - Get JWT token from browser DevTools
    - Token valid for 5 YEARS!

  QUICK METHOD FOR KILO:
    1. Open https://kilo.ai in incognito
    2. Sign in with Google/GitHub
    3. Open DevTools (F12) → Application → Cookies
    4. Find 'kilo_token' or similar JWT
    5. Copy and add to KILO_TOKENS env var
""")


def main() -> None:
    # Define GAAP pipeline stages
    pipeline = [
        # Layer 0: Interface
        PipelineStage("firewall_scan", llm_calls=0),
        PipelineStage("intent_classification", llm_calls=1),
        # Layer 1: Strategic
        PipelineStage("strategic_planning", llm_calls=1, context_size="large"),
        PipelineStage("architecture_spec", llm_calls=1),
        # Layer 2: Tactical
        PipelineStage("task_decomposition", llm_calls=1),
        PipelineStage("dependency_analysis", llm_calls=1),
        # Layer 3: Execution (OODA Loop - can iterate multiple times)
        PipelineStage("ooda_observe", llm_calls=1),
        PipelineStage("ooda_decide", llm_calls=1),
        PipelineStage("ooda_act", llm_calls=1, parallel=True),  # Multiple agents
        PipelineStage("ooda_learn", llm_calls=1),
        # Self-Healing (when needed)
        PipelineStage("healing_retry", llm_calls=2, parallel=True),
        PipelineStage("healing_pivot", llm_calls=1),
        # Swarm Intelligence (when enabled)
        PipelineStage("swarm_coder", llm_calls=1, parallel=True),
        PipelineStage("swarm_critic", llm_calls=1, parallel=True),
        PipelineStage("swarm_researcher", llm_calls=1, parallel=True),
        PipelineStage("swarm_consensus", llm_calls=1),
    ]

    # Calculate for different scenarios
    print("\n" + "=" * 70)
    print("║ SCENARIO 1: Development/Testing (100 iterations/day)")
    print("=" * 70)
    req1 = calculate_requirements(pipeline, iterations_per_day=100, quality_level="standard")
    print_report(req1)

    print("\n\n" + "=" * 70)
    print("║ SCENARIO 2: Production (500 iterations/day)")
    print("=" * 70)
    req2 = calculate_requirements(pipeline, iterations_per_day=500, quality_level="high")
    print_report(req2)

    # Summary table
    print("\n\n" + "=" * 70)
    print("║ QUICK REFERENCE TABLE")
    print("=" * 70)
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│  USAGE LEVEL          │ KIMI  │ KILO  │ TOTAL │ STATUS            │
├─────────────────────────────────────────────────────────────────────┤
│  Light (hobby)        │   3   │   2   │   5   │ Basic             │
│  Medium (dev)         │   5   │   3   │   8   │ Good              │
│  High (production)    │   9   │   5   │  14   │ Recommended       │
│  Enterprise           │  15   │  10   │  25   │ Ideal             │
│  Heavy parallel       │  20   │  15   │  35   │ Maximum isolation │
└─────────────────────────────────────────────────────────────────────┘

  YOUR CURRENT STATUS: 9 Kimi + 1 Kilo = 10 accounts (Good!)

  NEXT STEP: Add 5-10 Kilo accounts for production readiness
""")


if __name__ == "__main__":
    main()
