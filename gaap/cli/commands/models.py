"""
Model Commands
"""

from typing import Any

MODELS_INFO = {
    "strategic": {
        "tier": "Tier 1 - Strategic",
        "models": [
            {"name": "gpt-4o", "context": 128000, "cost": "$2.50/1M tokens"},
            {"name": "claude-opus", "context": 200000, "cost": "$15/1M tokens"},
            {"name": "o1-preview", "context": 128000, "cost": "$15/1M tokens"},
        ],
    },
    "tactical": {
        "tier": "Tier 2 - Tactical",
        "models": [
            {"name": "llama-3.3-70b", "context": 128000, "cost": "Free (Groq)"},
            {"name": "gemini-1.5-pro", "context": 1000000, "cost": "Free tier"},
            {"name": "claude-sonnet", "context": 200000, "cost": "$3/1M tokens"},
        ],
    },
    "efficient": {
        "tier": "Tier 3 - Efficient",
        "models": [
            {"name": "llama-3.1-8b", "context": 128000, "cost": "Free (Groq)"},
            {"name": "gemini-1.5-flash", "context": 1000000, "cost": "Free tier"},
            {"name": "mistral-small", "context": 32000, "cost": "Free tier"},
        ],
    },
    "private": {
        "tier": "Tier 4 - Private/Local",
        "models": [
            {"name": "llama-3.2-local", "context": 8192, "cost": "Free (local)"},
            {"name": "mistral-local", "context": 32768, "cost": "Free (local)"},
        ],
    },
}


def cmd_models(args: Any) -> None:
    """Model management commands"""
    action = args.action if hasattr(args, "action") else "list"
    tier = args.tier if hasattr(args, "tier") else None

    if action == "list":
        _list_models(tier)
    elif action == "tiers":
        _show_tiers()
    elif action == "info":
        _show_model_info(args.model if hasattr(args, "model") else None)
    else:
        _list_models(tier)


def _list_models(tier: str | None = None) -> None:
    """List all models"""
    print("\n🤖 Available Models")
    print("=" * 70)

    for tier_key, tier_info in MODELS_INFO.items():
        if tier and tier.lower() not in tier_key.lower():
            continue

        print(f"\n📁 {tier_info['tier']}")
        print("-" * 50)

        for model in tier_info["models"]:
            context_str = _format_context(model["context"])
            print(f"  • {model['name']}")
            print(f"    Context: {context_str} | Cost: {model['cost']}")

    print("\n" + "=" * 70)


def _show_tiers() -> None:
    """Show model tier information"""
    print("\n📊 Model Tiers")
    print("=" * 70)

    tier_descriptions = {
        "strategic": "For complex planning, architecture decisions, and critical tasks",
        "tactical": "For code generation, analysis, and moderate complexity tasks",
        "efficient": "For simple tasks, quick responses, and high-volume operations",
        "private": "For sensitive data, offline usage, and privacy requirements",
    }

    for tier_key, tier_info in MODELS_INFO.items():
        print(f"\n{tier_info['tier']}")
        print(f"  {tier_descriptions.get(tier_key, '')}")
        print(f"  Models: {len(tier_info['models'])}")

    print("\n" + "=" * 70)


def _show_model_info(model_name: str | None = None) -> None:
    """Show detailed model info"""
    if not model_name:
        print("❌ Model name required: gaap models info <name>")
        return

    for _tier_key, tier_info in MODELS_INFO.items():
        for model in tier_info["models"]:
            if model["name"] == model_name:
                print(f"\n🤖 {model['name']}")
                print("=" * 50)
                print(f"  Tier: {tier_info['tier']}")
                print(f"  Context: {_format_context(model['context'])}")
                print(f"  Cost: {model['cost']}")
                print("=" * 50)
                return

    print(f"❌ Model not found: {model_name}")


def _format_context(tokens: int) -> str:
    """Format context size"""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M tokens"
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K tokens"
    return f"{tokens} tokens"
