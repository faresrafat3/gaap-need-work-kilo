#!/usr/bin/env python3
"""
Kilo Token Watcher

Watches ~/.local/share/kilo/auth.json for changes and automatically
saves new tokens to ~/.gaap/accounts.json

Run this in a separate terminal while you authenticate new accounts.

Usage:
    python scripts/watch_kilo_tokens.py
"""

import json
import time
from pathlib import Path


def decode_token(token: str) -> dict:
    import base64

    parts = token.split(".")
    payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def get_current_auth() -> dict | None:
    auth_path = Path.home() / ".local" / "share" / "kilo" / "auth.json"
    if auth_path.exists():
        with open(auth_path) as f:
            return json.load(f)
    return None


def get_saved_tokens() -> list[str]:
    config_path = Path.home() / ".gaap" / "accounts.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return [a["token"] for a in config.get("accounts", []) if a["provider"] == "kilo"]
    return []


def save_token(token: str) -> bool:
    config_path = Path.home() / ".gaap" / "accounts.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {"accounts": []}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Check duplicate
    if any(a["token"] == token for a in config["accounts"]):
        return False

    count = len([a for a in config["accounts"] if a["provider"] == "kilo"])
    config["accounts"].append({"token": token, "provider": "kilo", "name": f"kilo_{count + 1}"})

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return True


def main():
    print("=" * 60)
    print("║ KILO TOKEN WATCHER")
    print("=" * 60)
    print("\n👀 Watching for new Kilo tokens...")
    print("   Open a new terminal and run: kilo")
    print("   Then use: /connect to add a new account")
    print("   This script will auto-save new tokens")
    print("\n   Press Ctrl+C to stop\n")
    print("-" * 60)

    seen_tokens = set(get_saved_tokens())
    last_check = time.time()

    while True:
        try:
            auth = get_current_auth()
            if auth and "kilo" in auth:
                token = auth["kilo"].get("access") or auth["kilo"].get("key")
                if token and token not in seen_tokens:
                    # New token found!
                    data = decode_token(token)
                    user_id = data.get("kiloUserId", "unknown")[:8]

                    if save_token(token):
                        seen_tokens.add(token)
                        count = len(seen_tokens)
                        print(f"\n✅ [{time.strftime('%H:%M:%S')}] New token saved!")
                        print(f"   User: {user_id}...")
                        print(f"   Total Kilo accounts: {count}")
                        print("-" * 60)
                    else:
                        seen_tokens.add(token)

            time.sleep(1)

        except KeyboardInterrupt:
            print("\n\n👋 Stopped watching")
            print(f"   Total Kilo accounts: {len(seen_tokens)}")
            break
        except Exception as e:
            time.sleep(1)


if __name__ == "__main__":
    main()
