#!/usr/bin/env python3
"""
Kilo Token - Simple Extractor

The JWT token is stored when you use:
1. Kilo Code CLI (kilo command)
2. Kilo Code VS Code Extension

Location of tokens:
- CLI: ~/.config/kilo/opencode.json
- VS Code: Extension storage

Usage:
    python scripts/get_kilo_token.py
"""

import json
import os
from pathlib import Path


def get_cli_token() -> str | None:
    """Get token from Kilo CLI config."""
    config_paths = [
        Path.home() / ".config" / "kilo" / "opencode.json",
        Path.home() / ".config" / "kilo" / "opencode.jsonc",
        Path.home() / ".kilo" / "config.json",
        Path.home() / ".opencode" / "config.json",
    ]

    for path in config_paths:
        if path.exists():
            try:
                with open(path) as f:
                    content = f.read()
                    # Try to parse JSON
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Try to remove comments
                        import re

                        content = re.sub(r"//.*", "", content)
                        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
                        data = json.loads(content)

                    # Look for token in various places
                    if "apiKey" in data:
                        return data["apiKey"]
                    if "token" in data:
                        return data["token"]
                    if "auth" in data and isinstance(data["auth"], dict):
                        if "token" in data["auth"]:
                            return data["auth"]["token"]
                        if "apiKey" in data["auth"]:
                            return data["auth"]["apiKey"]
                    if "providers" in data:
                        for provider in data.get("providers", []):
                            if provider.get("type") == "kilo" and "apiKey" in provider:
                                return provider["apiKey"]
            except Exception as e:
                print(f"Error reading {path}: {e}")

    return None


def get_vscode_token() -> str | None:
    """Get token from VS Code extension storage."""
    vscode_paths = [
        Path.home() / ".config" / "Code" / "User" / "globalStorage" / "kilo.kilo-code",
        Path.home() / ".vscode" / "extensions" / "kilo.kilo-code",
    ]

    for base_path in vscode_paths:
        if not base_path.exists():
            continue

        # Check various possible storage locations
        for json_file in base_path.rglob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        if "token" in data:
                            return data["token"]
                        if "apiKey" in data:
                            return data["apiKey"]
                        if "auth" in data:
                            return data["auth"].get("token") or data["auth"].get("apiKey")
            except Exception:
                pass

    return None


def check_kilo_cli_installed() -> bool:
    """Check if Kilo CLI is installed."""
    import shutil

    return shutil.which("kilo") is not None


def check_kilo_vscode_installed() -> bool:
    """Check if Kilo VS Code extension is installed."""
    vscode_ext_path = Path.home() / ".vscode" / "extensions"
    if vscode_ext_path.exists():
        for ext in vscode_ext_path.iterdir():
            if "kilo" in ext.name.lower():
                return True
    return False


def show_instructions() -> None:
    """Show instructions for getting token."""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║           HOW TO GET KILO JWT TOKEN                               ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  OPTION 1: Use Kilo CLI (RECOMMENDED)                             ║
║  ─────────────────────────────────────────                        ║
║  1. Install Kilo CLI:                                             ║
║     npm install -g @kilocode/cli                                  ║
║                                                                    ║
║  2. Run kilo and authenticate:                                    ║
║     kilo                                                          ║
║                                                                    ║
║  3. Use /connect command to sign in with Google/GitHub            ║
║                                                                    ║
║  4. Token will be saved to:                                       ║
║     ~/.config/kilo/opencode.json                                  ║
║                                                                    ║
║  OPTION 2: Use Kilo VS Code Extension                             ║
║  ─────────────────────────────────────────                        ║
║  1. Install from:                                                 ║
║     https://marketplace.visualstudio.com/items?itemName=kilo.kilo-code
║                                                                    ║
║  2. Open VS Code and click on Kilo icon                           ║
║                                                                    ║
║  3. Sign in with Google/GitHub                                    ║
║                                                                    ║
║  4. Token will be stored in VS Code's secret storage              ║
║                                                                    ║
║  OPTION 3: Use Kilo Web + Browser Console                         ║
║  ─────────────────────────────────────────                        ║
║  1. Go to: https://app.kilo.ai                                    ║
║                                                                    ║
║  2. Sign in with Google/GitHub                                    ║
║                                                                    ║
║  3. Open Browser Console (F12 → Console)                          ║
║                                                                    ║
║  4. Run this JavaScript:                                          ║
║     localStorage                                                  ║
║                                                                    ║
║  5. Look for any key with a value starting with "eyJ"             ║
║                                                                    ║
║  OPTION 4: From Network Tab (while using Kilo services)           ║
║  ─────────────────────────────────────────                        ║
║  1. Go to: https://app.kilo.ai                                    ║
║                                                                    ║
║  2. Open DevTools (F12)                                           ║
║                                                                    ║
║  3. Go to Network tab                                             ║
║                                                                    ║
║  4. Do any action that calls the API                              ║
║                                                                    ║
║  5. Look for requests to api.kilo.ai                              ║
║                                                                    ║
║  6. Check Authorization header                                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
""")


def main() -> int:
    print("=" * 70)
    print("║ KILO TOKEN FINDER")
    print("=" * 70)

    # Check what's installed
    cli_installed = check_kilo_cli_installed()
    vscode_installed = check_kilo_vscode_installed()

    print(f"\n📊 Status:")
    print(f"   Kilo CLI: {'✅ Installed' if cli_installed else '❌ Not installed'}")
    print(f"   Kilo VS Code: {'✅ Installed' if vscode_installed else '❌ Not installed'}")

    # Try to find token
    token = None
    source = None

    if cli_installed:
        token = get_cli_token()
        if token:
            source = "Kilo CLI"

    if not token and vscode_installed:
        token = get_vscode_token()
        if token:
            source = "VS Code Extension"

    if token:
        print(f"\n✅ Found token in {source}!")
        print(f"   Token: {token[:50]}...")

        # Validate
        if token.startswith("eyJ"):
            print("\n✅ Valid JWT format!")

            # Decode to show info
            try:
                import base64

                parts = token.split(".")
                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                data = json.loads(base64.urlsafe_b64decode(payload))
                print(f"   User ID: {data.get('kiloUserId', 'N/A')[:30]}...")
                if "exp" in data:
                    import time

                    exp_date = time.strftime("%Y-%m-%d", time.localtime(data["exp"]))
                    days = (data["exp"] - time.time()) / 86400
                    print(f"   Expires: {exp_date} ({days:.0f} days)")
            except Exception:
                pass

            # Save to GAAP config
            config_path = Path.home() / ".gaap" / "accounts.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)

            config = {"accounts": []}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            # Check duplicate
            exists = any(a["token"] == token for a in config["accounts"])
            if not exists:
                count = len([a for a in config["accounts"] if a["provider"] == "kilo"])
                config["accounts"].append(
                    {"token": token, "provider": "kilo", "name": f"kilo_{count + 1}"}
                )
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"\n💾 Saved to: {config_path}")

            return 0
    else:
        print("\n❌ Token not found in local storage")
        show_instructions()

        # Manual entry
        print("\n" + "─" * 70)
        print("Paste your token manually (or press Enter to skip):")
        try:
            manual_token = input("> ").strip()
            if manual_token and manual_token.startswith("eyJ"):
                config_path = Path.home() / ".gaap" / "accounts.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)

                config = {"accounts": []}
                if config_path.exists():
                    with open(config_path) as f:
                        config = json.load(f)

                count = len([a for a in config["accounts"] if a["provider"] == "kilo"])
                config["accounts"].append(
                    {"token": manual_token, "provider": "kilo", "name": f"kilo_{count + 1}"}
                )
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=2)
                print(f"\n✅ Token saved!")
        except EOFError:
            pass

    return 0


if __name__ == "__main__":
    main()
