"""
Provider Configuration Summary
==============================

Shows clean, filtered configuration for all providers
Only primary models kept for production use
"""

from typing import Any

from gaap.providers.multi_provider_config import (
    FAILED_PROVIDERS,
    WORKING_PROVIDERS,
)


def print_provider_summary() -> None:
    """Print clean provider summary"""

    print("=" * 90)
    print("🔧 CLEANED PROVIDER CONFIGURATION")
    print("=" * 90)
    print()

    # Working providers
    print("✅ WORKING PROVIDERS (4 providers)")
    print("-" * 90)

    for provider in WORKING_PROVIDERS:
        num_keys = len(provider.api_keys)
        num_models = len(provider.models)
        rpm_total = provider.limits.requests_per_minute * num_keys

        print(f"\n📌 {provider.name}")
        print(f"   Priority: {provider.priority}")
        print(f"   API Keys: {num_keys}")
        print(f"   Capacity: {rpm_total} RPM ({provider.limits.requests_per_minute} RPM per key)")
        print(f"   Models: {num_models}")

        for model in provider.models:
            print(f"      • {model}")

        print(f"   Base URL: {provider.base_url}")
        print(f"   Notes: {provider.notes}")

    print()
    print()

    # Failed providers
    print("❌ FAILED/UNAVAILABLE PROVIDERS (4 providers)")
    print("-" * 90)

    for provider in FAILED_PROVIDERS:
        num_keys = len(provider.api_keys)
        num_models = len(provider.models)

        print(f"\n⛔ {provider.name}")
        print(f"   API Keys: {num_keys}")
        print(f"   Models: {num_models}")

        for model in provider.models:
            print(f"      • {model}")

        print(f"   Notes: {provider.notes}")

    print()
    print()

    # Total capacity (working providers only)
    total_rpm = 0
    total_keys = 0
    provider_rows = []

    for provider in WORKING_PROVIDERS:
        num_keys = len(provider.api_keys)
        rpm_total = provider.limits.requests_per_minute * num_keys
        total_rpm += rpm_total
        total_keys += num_keys
        provider_rows.append(
            {
                "name": provider.name,
                "rpm": rpm_total,
                "keys": num_keys,
                "priority": provider.priority,
            }
        )

    print("📊 TOTAL CAPACITY (Working Providers Only)")
    print("-" * 90)
    print(f"Total RPM:        {total_rpm:,}")
    print(f"Total Keys:       {total_keys}")
    print(f"Active Providers: {len(WORKING_PROVIDERS)}")
    print()

    print("Breakdown by Provider:")
    print(f"{'Provider':<25} {'RPM':>10} {'Keys':>6} {'Models':>8} {'Priority':>10}")
    print("-" * 90)

    for p in provider_rows:
        print(f"{p['name']:<25} {p['rpm']:>10,} {p['keys']:>6} {1:>8} {p['priority']:>10}")

    print()
    print("=" * 90)
    print()

    # Recommendations
    print("💡 RECOMMENDATIONS:")
    print()
    print("1. Primary Workload:")
    print("   • Use Cerebras + Groq (420 RPM combined)")
    print("   • Both achieve 87% MMLU accuracy")
    print("   • Groq is fastest (227ms avg latency)")
    print()
    print("2. Backup/Overflow:")
    print("   • Use Mistral (60 RPM)")
    print("   • Use GitHub (15 RPM)")
    print()
    print("3. Models to Use:")
    print("   • Cerebras: llama3.3-70b")
    print("   • Groq: llama-3.3-70b-versatile")
    print("   • Mistral: mistral-large-latest")
    print("   • GitHub: gpt-4o-mini")
    print()
    print("4. Avoid:")
    print("   • OpenRouter (needs $10 credit)")
    print("   • Gemini (quota exhausted)")
    print("   • Cloudflare (needs account ID)")
    print()
    print("=" * 90)


def print_model_comparison() -> None:
    """Compare models across providers"""

    print()
    print("=" * 90)
    print("🔬 MODEL COMPARISON")
    print("=" * 90)
    print()

    models_data = [
        {
            "provider": "Groq",
            "model": "llama-3.3-70b-versatile",
            "latency": "227ms",
            "accuracy": "87.0%",
            "rpm": 210,
            "status": "✅ FASTEST",
        },
        {
            "provider": "Cerebras",
            "model": "llama3.3-70b",
            "latency": "511ms",
            "accuracy": "87.0%",
            "rpm": 210,
            "status": "✅ RELIABLE",
        },
        {
            "provider": "Mistral",
            "model": "mistral-large-latest",
            "latency": "603ms",
            "accuracy": "N/A",
            "rpm": 60,
            "status": "✅ VERIFIED",
        },
        {
            "provider": "GitHub",
            "model": "gpt-4o-mini",
            "latency": "1498ms",
            "accuracy": "N/A",
            "rpm": 15,
            "status": "✅ BACKUP",
        },
    ]

    print(f"{'Provider':<12} {'Model':<30} {'Latency':>10} {'MMLU':>8} {'RPM':>6} {'Status':<12}")
    print("-" * 90)

    for m in models_data:
        print(
            f"{m['provider']:<12} {m['model']:<30} {m['latency']:>10} {m['accuracy']:>8} {m['rpm']:>6} {m['status']:<12}"
        )

    print()
    print("=" * 90)


if __name__ == "__main__":
    print_provider_summary()
    print_model_comparison()
