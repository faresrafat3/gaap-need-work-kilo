"""
WebChat Providers - Backward Compatibility Module
=================================================

This module re-exports everything from the new webchat/ subpackage
for backward compatibility.

The actual implementation is now in:
  gaap/providers/webchat/
    ├── __init__.py
    ├── base.py      (WebChatAuth, WebChatProvider ABC)
    ├── glm.py       (GLMWebChat)
    ├── kimi.py      (KimiWebChat)
    ├── deepseek.py  (DeepSeekWebChat)
    ├── copilot.py   (CopilotWebChat)
    └── registry.py  (get_provider, webchat_call, CLI)
"""

from gaap.providers.webchat import (
    WEBCHAT_CACHE_DIR,
    CopilotWebChat,
    DeepSeekWebChat,
    GLMWebChat,
    KimiWebChat,
    WebChatAuth,
    WebChatProvider,
    check_all_webchat_auth,
    get_provider,
    invalidate_auth,
    list_accounts,
    load_auth,
    save_auth,
    webchat_call,
)

__all__ = [
    "WEBCHAT_CACHE_DIR",
    "WebChatAuth",
    "WebChatProvider",
    "GLMWebChat",
    "KimiWebChat",
    "DeepSeekWebChat",
    "CopilotWebChat",
    "save_auth",
    "load_auth",
    "invalidate_auth",
    "list_accounts",
    "get_provider",
    "webchat_call",
    "check_all_webchat_auth",
]


def _cli() -> None:
    from gaap.providers.webchat.registry import _cli as _real_cli

    _real_cli()


if __name__ == "__main__":
    _cli()
