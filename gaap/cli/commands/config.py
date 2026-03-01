"""
Config Commands
"""

import json
from typing import Any

DEFAULT_CONFIG = {
    "default_provider": "kimi",
    "default_model": "llama-3.3-70b",
    "default_budget": 10.0,
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 120,
    "enable_healing": True,
    "enable_memory": True,
    "enable_security": True,
    "log_level": "INFO",
}


def cmd_config(args: Any) -> None:
    """Configuration management commands"""
    action = args.action if hasattr(args, "action") else "show"

    if action == "show":
        _show_config()
    elif action == "get":
        _get_config(args.key if hasattr(args, "key") else None)
    elif action == "set":
        _set_config(
            args.key if hasattr(args, "key") else None,
            args.value if hasattr(args, "value") else None,
        )
    elif action == "reset":
        _reset_config()
    elif action == "export":
        _export_config()
    elif action == "import":
        _import_config(args.file if hasattr(args, "file") else None)
    else:
        _show_config()


def _show_config() -> None:
    """Show all configuration"""
    from gaap.storage import load_config

    config = load_config()

    if not config:
        config = DEFAULT_CONFIG.copy()

    print("\n⚙️  GAAP Configuration")
    print("=" * 50)

    for key, value in config.items():
        if isinstance(value, bool):
            value_str = "✅" if value else "❌"
        elif isinstance(value, str) and len(value) > 30:
            value_str = value[:27] + "..."
        else:
            value_str = str(value)

        print(f"  {key}: {value_str}")

    print("\n" + "=" * 50)
    print("Use 'gaap config get <key>' to get a specific value")
    print("Use 'gaap config set <key> <value>' to update")


def _get_config(key: str | None = None) -> None:
    """Get a specific config value"""
    if not key:
        print("❌ Key required: gaap config get <key>")
        return

    from gaap.storage import get_config, load_config

    if key == "all":
        config = load_config()
        print(json.dumps(config, indent=2))
        return

    value = get_config(key)
    if value is None:
        # Check default
        value = DEFAULT_CONFIG.get(key)

    if value is None:
        print(f"❌ Config key not found: {key}")
    else:
        print(f"{key}: {value}")


def _set_config(key: str | None = None, value: str | None = None) -> None:
    """Set a config value"""
    if not key or value is None:
        print("❌ Key and value required: gaap config set <key> <value>")
        return

    from gaap.storage import save_config

    # Parse value type
    parsed_value = _parse_value(value)

    save_config(key, parsed_value)
    print(f"✅ Set {key} = {parsed_value}")


def _reset_config() -> None:
    """Reset configuration to defaults"""
    from gaap.storage import get_store

    store = get_store()
    store.save("config", DEFAULT_CONFIG.copy())

    print("✅ Configuration reset to defaults")
    _show_config()


def _export_config() -> None:
    """Export configuration to JSON"""
    from gaap.storage import load_config

    config = load_config()
    print(json.dumps(config, indent=2))


def _import_config(file_path: str | None = None) -> None:
    """Import configuration from file"""
    if not file_path:
        print("❌ File path required: gaap config import <file>")
        return

    from pathlib import Path

    from gaap.storage import get_store

    try:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return

        with open(path) as f:
            config = json.load(f)

        store = get_store()
        store.save("config", config)
        print("✅ Configuration imported")
    except json.JSONDecodeError:
        print("❌ Invalid JSON file")
    except Exception as e:
        print(f"❌ Error: {e}")


def _parse_value(value: str) -> Any:
    """Parse string value to appropriate type"""
    # Boolean
    if value.lower() in ("true", "yes", "on", "1"):
        return True
    if value.lower() in ("false", "no", "off", "0"):
        return False

    # Number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        pass

    # JSON
    if value.startswith("{") or value.startswith("["):
        try:
            return json.loads(value)
        except Exception as e:
            print(f"Config error: {e}")

    # String
    return value


# Import Any at the top
from typing import Any
