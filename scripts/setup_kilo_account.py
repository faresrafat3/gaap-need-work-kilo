#!/usr/bin/env python3
"""
Kilo Account Setup Assistant

Interactive guide to add new Kilo accounts:
1. Opens browser to Kilo login
2. Waits for you to authenticate
3. Helps extract JWT token
4. Saves to config automatically

Usage:
    python scripts/setup_kilo_account.py
"""

import json
import os
import sys
import time
import webbrowser
from pathlib import Path


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"║ {title}")
    print(f"{'=' * 70}")


def print_step(step: int, title: str) -> None:
    print(f"\n📌 STEP {step}: {title}")
    print("-" * 40)


def get_config_path() -> Path:
    return Path.home() / ".gaap" / "accounts.json"


def load_current_accounts() -> dict:
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {"accounts": []}


def save_account(token: str, name: str) -> bool:
    try:
        config_path = get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config = load_current_accounts()

        # Check if token already exists
        for acc in config["accounts"]:
            if acc["token"] == token:
                print(f"⚠️ Token already exists as '{acc['name']}'")
                return False

        config["accounts"].append({"token": token, "provider": "kilo", "name": name})

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return True
    except Exception as e:
        print(f"❌ Error saving: {e}")
        return False


def validate_jwt_token(token: str) -> dict | None:
    """Validate and decode JWT token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        import base64

        # Add padding if needed
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)

        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)

        # Check required fields
        if "kiloUserId" in data or "exp" in data:
            return data
        return None
    except Exception:
        return None


def extract_token_from_browser() -> None:
    """Guide user to extract token from browser."""
    print_step(1, "Open Kilo in Browser")

    print("""
The browser will open to: https://kilo.ai

If it doesn't open automatically, go there manually.
""")

    if input("Open browser now? (y/n): ").lower() == "y":
        webbrowser.open("https://kilo.ai")
        print("✅ Browser opened!")

    print_step(2, "Sign In")

    print("""
Sign in with one of these methods:
  - Google account
  - GitHub account
  - Email (if available)

Complete the sign-in process in the browser.
""")

    input("Press Enter when you're signed in...")

    print_step(3, "Open Developer Tools")

    print("""
In the browser:
1. Press F12 (or right-click → Inspect)
2. Go to the "Application" tab (or "Storage" in some browsers)
3. In the left sidebar, find "Local Storage" or "Cookies"
4. Click on "https://kilo.ai"
5. Look for a key named:
   - "kilo_token"
   - "token"
   - "auth_token"
   - Or any JWT-looking value (starts with "eyJ")

ALTERNATIVE METHOD:
1. Press F12 → "Network" tab
2. Refresh the page
3. Click on any request to "api.kilo.ai"
4. Look at "Headers" → "Authorization: Bearer eyJ..."
""")

    print_step(4, "Copy Token")

    print("""
The JWT token looks like:
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ...very long string...

Copy the ENTIRE token (including all three parts separated by dots)
""")


def manual_token_entry() -> str | None:
    """Allow manual token entry."""
    print_header("MANUAL TOKEN ENTRY")

    print("""
Paste your JWT token below.
It should start with "eyJ" and be quite long.
""")

    print("Paste token (or 'cancel' to exit):")
    token = input("> ").strip()

    if token.lower() == "cancel":
        return None

    # Clean up token (remove quotes, spaces, etc.)
    token = token.strip("\"'").strip()

    # Validate
    data = validate_jwt_token(token)
    if data:
        print("\n✅ Valid JWT token detected!")
        print(f"   User ID: {data.get('kiloUserId', 'N/A')[:20]}...")
        print(f"   Environment: {data.get('env', 'N/A')}")
        if "exp" in data:
            exp_date = time.strftime("%Y-%m-%d", time.localtime(data["exp"]))
            print(f"   Expires: {exp_date}")
        return token
    else:
        print("\n❌ Invalid token. Please check and try again.")
        return None


def interactive_setup() -> None:
    """Run interactive setup wizard."""
    print_header("KILO ACCOUNT SETUP ASSISTANT")

    current = load_current_accounts()
    kilo_count = len([a for a in current["accounts"] if a["provider"] == "kilo"])
    kimi_count = len([a for a in current["accounts"] if a["provider"] == "kimi"])

    print(f"""
Current Accounts:
  - Kilo: {kilo_count}
  - Kimi: {kimi_count}
  - Total: {kilo_count + kimi_count}

Recommended for production: 5-10 Kilo accounts
""")

    while True:
        print("\n" + "─" * 40)
        print("OPTIONS:")
        print("  1. Add new Kilo account (guided)")
        print("  2. Add new Kilo account (paste token)")
        print("  3. Add new Kimi account (paste token)")
        print("  4. View current accounts")
        print("  5. Export to environment variables")
        print("  6. Exit")
        print("─" * 40)

        choice = input("\nSelect option (1-6): ").strip()

        if choice == "1":
            extract_token_from_browser()
            token = manual_token_entry()
            if token:
                name = input(f"Enter account name (default: kilo_{kilo_count + 1}): ").strip()
                name = name or f"kilo_{kilo_count + 1}"
                if save_account(token, name):
                    print(f"\n✅ Account '{name}' saved successfully!")
                    kilo_count += 1

        elif choice == "2":
            token = manual_token_entry()
            if token:
                name = input(f"Enter account name (default: kilo_{kilo_count + 1}): ").strip()
                name = name or f"kilo_{kilo_count + 1}"
                if save_account(token, name):
                    print(f"\n✅ Account '{name}' saved successfully!")
                    kilo_count += 1

        elif choice == "3":
            print("\nKimi tokens can be obtained from:")
            print("  - Kimi web interface cookies")
            print("  - Each Kimi account has 9 sub-accounts built-in")
            token = input("\nPaste Kimi token (or 'cancel'): ").strip()
            if token.lower() != "cancel":
                name = input(f"Enter account name (default: kimi_{kimi_count + 1}): ").strip()
                name = name or f"kimi_{kimi_count + 1}"
                config = load_current_accounts()
                config["accounts"].append({"token": token, "provider": "kimi", "name": name})
                with open(get_config_path(), "w") as f:
                    json.dump(config, f, indent=2)
                print(f"\n✅ Account '{name}' saved!")
                kimi_count += 1

        elif choice == "4":
            print_header("CURRENT ACCOUNTS")
            config = load_current_accounts()
            for i, acc in enumerate(config["accounts"], 1):
                token_preview = acc["token"][:30] + "..."
                print(f"{i}. {acc['name']} ({acc['provider']})")
                print(f"   Token: {token_preview}")

        elif choice == "5":
            print_header("EXPORT TO ENVIRONMENT VARIABLES")
            config = load_current_accounts()

            kilo_tokens = [a["token"] for a in config["accounts"] if a["provider"] == "kilo"]
            kimi_tokens = [a["token"] for a in config["accounts"] if a["provider"] == "kimi"]

            print("\nAdd these to your .bashrc, .zshrc, or .env file:\n")
            print(f'export KILO_TOKENS="{",".join(kilo_tokens)}"')
            print(f'export KIMI_TOKENS="{",".join(kimi_tokens)}"')

            # Also save to .gaap_env
            env_path = Path.home() / ".gaap" / ".env"
            with open(env_path, "w") as f:
                f.write(f'KILO_TOKENS="{",".join(kilo_tokens)}"\n')
                f.write(f'KIMI_TOKENS="{",".join(kimi_tokens)}"\n')
            print(f"\n💾 Also saved to: {env_path}")

        elif choice == "6":
            print("\n👋 Goodbye!")
            print(
                f"\nFinal count: {kilo_count} Kilo + {kimi_count} Kimi = {kilo_count + kimi_count} accounts"
            )

            if kilo_count < 5:
                print(f"\n💡 Tip: Add {5 - kilo_count} more Kilo accounts for production use!")
            break

        else:
            print("Invalid option. Please try again.")


def quick_add_tokens() -> None:
    """Quick mode to add multiple tokens."""
    print_header("QUICK ADD MULTIPLE TOKENS")

    print("""
Paste your Kilo JWT tokens one by one.
Press Enter twice when done.

Each token should start with "eyJ" and be on a single line.
""")

    tokens = []
    while True:
        line = input(f"Token {len(tokens) + 1} (or empty to finish): ").strip()
        if not line:
            break

        data = validate_jwt_token(line)
        if data:
            tokens.append(line)
            print(f"  ✅ Valid token #{len(tokens)}")
        else:
            print("  ❌ Invalid token, skipping")

    if tokens:
        print(f"\n{len(tokens)} valid tokens found!")
        for i, token in enumerate(tokens):
            name = f"kilo_{i + 1}"
            save_account(token, name)
        print(f"\n✅ All {len(tokens)} tokens saved!")


def main() -> int:
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick":
            quick_add_tokens()
        elif sys.argv[1] == "--help":
            print(__doc__)
            print("""
Options:
    --quick    Quick add multiple tokens
    --help     Show this help
""")
        else:
            # Treat as token
            token = sys.argv[1]
            data = validate_jwt_token(token)
            if data:
                save_account(token, f"kilo_auto")
                print("✅ Token saved!")
            else:
                print("❌ Invalid token")
                return 1
    else:
        interactive_setup()

    return 0


if __name__ == "__main__":
    sys.exit(main())
