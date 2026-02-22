"""
WebChat Provider Registry & Utilities
=====================================

Provider factory, high-level webchat_call, and CLI.
"""

import logging
import time
from typing import Any

from .base import WebChatProvider, list_accounts
from .copilot import CopilotWebChat
from .deepseek import DeepSeekWebChat
from .glm import GLMWebChat
from .kimi import KimiWebChat

logger = logging.getLogger("gaap.providers.webchat")

_providers: dict[str, WebChatProvider] = {}


def get_provider(provider_name: str, account: str = "default") -> WebChatProvider:
    """Get or create a webchat provider instance."""
    key = f"{provider_name}:{account}"
    if key not in _providers:
        if provider_name == "glm":
            _providers[key] = GLMWebChat(account=account)
        elif provider_name == "kimi":
            _providers[key] = KimiWebChat(account=account)
        elif provider_name == "deepseek":
            _providers[key] = DeepSeekWebChat(account=account)
        elif provider_name == "copilot":
            _providers[key] = CopilotWebChat(account=account)
        else:
            raise ValueError(f"Unknown webchat provider: {provider_name}")
    return _providers[key]


def webchat_call(
    provider_name: str,
    messages: list[dict[str, str]],
    model: str = "",
    account: str = "default",
    timeout: int = 120,
) -> str:
    """
    High-level: call a webchat provider with smart account selection.

    If account_manager pools are configured, uses intelligent routing:
      - Auto-selects best account (least loaded, highest health)
      - Tracks rate limits per account
      - Auto-rotates on failure

    Falls back to simple auth check if no pools configured.
    """
    try:
        from ..account_manager import PoolManager

        mgr = PoolManager.instance()
        pool = mgr.pool(provider_name)

        if pool.accounts:
            should, reason, acct_label = pool.should_call(
                label=account if account != "default" else "",
                model=model,
            )
            if should and acct_label:
                provider = get_provider(provider_name, acct_label)
                if provider.is_authenticated:
                    start = time.time()
                    try:
                        result = provider.call(messages, model=model, timeout=timeout)
                        latency_ms = (time.time() - start) * 1000
                        pool.record_call(
                            acct_label,
                            success=True,
                            latency_ms=latency_ms,
                            tokens_used=len(result) // 4,
                        )
                        return result
                    except Exception as e:
                        latency_ms = (time.time() - start) * 1000
                        pool.record_call(
                            acct_label,
                            success=False,
                            latency_ms=latency_ms,
                            error_msg=str(e)[:100],
                        )

                        try:
                            from ..account_manager import detect_hard_cooldown

                            hc = detect_hard_cooldown(str(e))
                            if hc:
                                cd_sec, cd_reason = hc
                                acct_obj = pool.get_account(acct_label)
                                if acct_obj:
                                    acct_obj.rate_tracker.set_hard_cooldown(cd_sec, cd_reason)
                        except Exception as e:
                            logger.debug(f"Hard cooldown detection failed: {e}")
                            pass

                        fallback = pool.best_account(model)
                        if fallback and fallback.label != acct_label:
                            fb_provider = get_provider(provider_name, fallback.label)
                            if fb_provider.is_authenticated:
                                start2 = time.time()
                                try:
                                    result2 = fb_provider.call(
                                        messages, model=model, timeout=timeout
                                    )
                                    pool.record_call(
                                        fallback.label,
                                        success=True,
                                        latency_ms=(time.time() - start2) * 1000,
                                        tokens_used=len(result2) // 4,
                                    )
                                    return result2
                                except Exception as e:
                                    logger.debug(f"Fallback call failed: {e}")
                                    pass
                        raise
    except ImportError:
        pass

    provider = get_provider(provider_name, account)
    if provider.is_authenticated:
        return provider.call(messages, model=model, timeout=timeout)

    for acct in list_accounts(provider_name):
        if acct == account:
            continue
        provider = get_provider(provider_name, acct)
        if provider.is_authenticated:
            return provider.call(messages, model=model, timeout=timeout)

    raise RuntimeError(
        f"{provider_name}: No authenticated accounts. "
        f"Run: python -m gaap.providers.webchat_providers login {provider_name}"
    )


def check_all_webchat_auth() -> dict[str, list[dict[str, Any]]]:
    """Check auth status for all webchat providers & accounts."""
    result = {}
    for pname in ["glm", "kimi", "deepseek", "copilot"]:
        accounts = list_accounts(pname) or ["default"]
        statuses = []
        for acct in accounts:
            provider = get_provider(pname, acct)
            statuses.append(provider.check_auth())
        result[pname] = statuses
    return result


def _cli() -> None:
    import sys

    usage = """
Usage: python -m gaap.providers.webchat_providers <command> [args]

Commands:
  login <provider> [account]    Open browser for login (provider: glm | kimi | deepseek | copilot)
  status                        Show auth status for all providers
  reset <provider> [account]    Clear cached auth
  test <provider> [account]     Quick test call
  accounts <provider>           List cached accounts

Examples:
  python -m gaap.providers.webchat_providers login glm
  python -m gaap.providers.webchat_providers login kimi account2
  python -m gaap.providers.webchat_providers status
  python -m gaap.providers.webchat_providers test glm
"""

    args = sys.argv[1:]
    if not args:
        print(usage)
        return

    cmd = args[0]

    if cmd == "login":
        if len(args) < 2:
            print("Usage: login <provider> [account]")
            return
        pname = args[1]
        account = args[2] if len(args) > 2 else "default"
        provider = get_provider(pname, account)
        provider.warmup(force=True)

    elif cmd == "status":
        all_status = check_all_webchat_auth()
        print("=" * 60)
        print("WebChat Provider Auth Status")
        print("=" * 60)
        for pname, statuses in all_status.items():
            for s in statuses:
                icon = "✅" if s["valid"] else "❌"
                print(f"  {icon} {s['provider']}[{s['account']}]: {s['message']}")
        print("=" * 60)

    elif cmd == "reset":
        if len(args) < 2:
            print("Usage: reset <provider> [account]")
            return
        pname = args[1]
        account = args[2] if len(args) > 2 else "default"
        from .base import invalidate_auth

        if invalidate_auth(pname, account):
            print(f"  🗑️  Cleared {pname} [{account}] auth cache")
        else:
            print(f"  ℹ️  No cache to clear for {pname} [{account}]")

    elif cmd == "test":
        if len(args) < 2:
            print("Usage: test <provider> [account]")
            return
        pname = args[1]
        account = args[2] if len(args) > 2 else "default"
        provider = get_provider(pname, account)
        if not provider.is_authenticated:
            print(f"  ❌ {pname} [{account}]: Not authenticated")
            return
        print(f"  🧪 Testing {pname} [{account}]...")
        try:
            result = provider.call(
                [{"role": "user", "content": "Reply with only: OK"}],
                timeout=60,
            )
            print(f"  ✅ Response: '{result[:100]}'")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    elif cmd == "accounts":
        if len(args) < 2:
            print("Usage: accounts <provider>")
            return
        pname = args[1]
        accounts = list_accounts(pname)
        if accounts:
            print(f"  Cached accounts for {pname}: {', '.join(accounts)}")
        else:
            print(f"  No cached accounts for {pname}")

    else:
        print(f"Unknown command: {cmd}")
        print(usage)


if __name__ == "__main__":
    _cli()
