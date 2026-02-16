"""
Provider Commands
"""

import asyncio
import os
from typing import Any

PROVIDERS_INFO: dict[str, dict[str, Any]] = {
    "groq": {
        "name": "Groq",
        "type": "Free Tier",
        "models": ["llama-3.3-70b", "llama-3.1-8b", "mixtral-8x7b"],
        "env_key": "GROQ_API_KEY",
        "status": "available",
    },
    "gemini": {
        "name": "Google Gemini",
        "type": "Free Tier",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "env_key": "GEMINI_API_KEY",
        "status": "available",
    },
    "cerebras": {
        "name": "Cerebras",
        "type": "Free Tier",
        "models": ["llama-3.3-70b"],
        "env_key": "CEREBRAS_API_KEY",
        "status": "available",
    },
    "mistral": {
        "name": "Mistral",
        "type": "Free Tier",
        "models": ["mistral-small", "mistral-medium"],
        "env_key": "MISTRAL_API_KEY",
        "status": "available",
    },
    "g4f": {
        "name": "G4F (Free)",
        "type": "Free",
        "models": ["auto", "gpt-4", "claude-3"],
        "env_key": None,
        "status": "available",
    },
}


def cmd_providers(args: Any) -> None:
    """Provider management commands"""
    action = args.action if hasattr(args, "action") else "list"

    if action == "list":
        _list_providers()
    elif action == "status":
        _show_status()
    elif action == "test":
        _test_provider(args.provider if hasattr(args, "provider") else None)
    elif action == "enable":
        _enable_provider(args.provider if hasattr(args, "provider") else None)
    elif action == "disable":
        _disable_provider(args.provider if hasattr(args, "provider") else None)
    else:
        _list_providers()


def _list_providers() -> None:
    """List all providers"""
    print("\n🔌 Available Providers")
    print("=" * 60)

    for key, info in PROVIDERS_INFO.items():
        has_key = "✅" if info["env_key"] is None or os.environ.get(info["env_key"]) else "❌"
        print(f"\n{info['name']} ({key})")
        print(f"  Type: {info['type']}")
        print(f"  API Key: {has_key}")
        print(f"  Models: {', '.join(info['models'])}")

    print("\n" + "=" * 60)


def _show_status() -> None:
    """Show detailed provider status"""
    print("\n📊 Provider Status")
    print("=" * 60)

    for _key, info in PROVIDERS_INFO.items():
        env_key = info["env_key"]
        has_key = os.environ.get(env_key) if env_key else True
        status = "🟢 Ready" if has_key else "🔴 No API Key"

        print(f"\n{info['name']}: {status}")
        if env_key:
            key_value = os.environ.get(env_key, "Not set")
            masked = key_value[:8] + "..." if len(key_value) > 8 else key_value
            print(f"  Key: {masked}")

    print("\n" + "=" * 60)


def _test_provider(provider_name: str | None = None) -> None:
    """Test a provider connection"""
    if not provider_name:
        print("❌ Provider name required: gaap providers test <name>")
        return

    if provider_name not in PROVIDERS_INFO:
        print(f"❌ Unknown provider: {provider_name}")
        return

    info = PROVIDERS_INFO[provider_name]
    env_key = info["env_key"]

    if env_key and not os.environ.get(env_key):
        print(f"❌ API key not set for {info['name']}")
        print(f"   Set {env_key} in .gaap_env or environment")
        return

    print(f"\n🔍 Testing {info['name']}...")

    async def test() -> None:
        try:
            prov: Any = None
            if provider_name == "groq":
                from gaap.providers.free_tier import GroqProvider

                prov = GroqProvider(api_key=os.environ.get("GROQ_API_KEY"))
            elif provider_name == "gemini":
                from gaap.providers.free_tier import GeminiProvider

                prov = GeminiProvider(api_key=os.environ.get("GEMINI_API_KEY"))
            else:
                print("⚠️  Test not implemented for this provider")
                return

            from gaap.core.types import Message, MessageRole

            messages = [Message(role=MessageRole.USER, content="Hi")]
            response = await prov.chat_completion(messages, model=info["models"][0])
            print("✅ Connection successful!")
            print(f"   Response: {response.choices[0].message.content[:50]}...")
        except Exception as e:
            print(f"❌ Connection failed: {e}")

    asyncio.run(test())


def _enable_provider(provider_name: str | None = None) -> None:
    """Enable a provider"""
    if not provider_name:
        print("❌ Provider name required: gaap providers enable <name>")
        return

    from gaap.storage import save_config

    save_config(f"provider_{provider_name}_enabled", True)
    print(f"✅ Provider '{provider_name}' enabled")


def _disable_provider(provider_name: str | None = None) -> None:
    """Disable a provider"""
    if not provider_name:
        print("❌ Provider name required: gaap providers disable <name>")
        return

    from gaap.storage import save_config

    save_config(f"provider_{provider_name}_enabled", False)
    print(f"✅ Provider '{provider_name}' disabled")
