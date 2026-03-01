"""
WebChat Providers - Browser-Authenticated Web Chat Scraping
============================================================

Provides free access to GLM-5 (chat.z.ai), Kimi K2.5 (kimi.com),
DeepSeek V3.2 (chat.deepseek.com), and GitHub Copilot (GPT-5, Claude 4,
Gemini 3, etc.) via browser-based / OAuth authentication and direct API calls.

Pattern:
  1. Open browser → user logs in manually → capture auth token
     (or OAuth device flow for Copilot — no browser scraping needed)
  2. Cache token to disk (~/.config/gaap/webchat_auth/)
  3. Use curl_cffi with captured token for API calls (no browser needed)
  4. Token expires → re-open browser for fresh auth

Supports multiple accounts for parallelism.

Module Structure:
  - base.py: WebChatAuth dataclass, WebChatProvider ABC, shared utilities
  - glm.py: GLM-5 / GLM-4.7 provider (chat.z.ai)
  - kimi.py: Kimi K2.5 provider (kimi.com, Connect-RPC protocol)
  - deepseek.py: DeepSeek V3.2 provider (chat.deepseek.com, PoW-based)
  - copilot.py: GitHub Copilot provider (OAuth device flow)
  - registry.py: Provider factory, webchat_call(), CLI
"""

from .aistudio import AIStudioWebChat
from .base import (
    WEBCHAT_CACHE_DIR,
    WebChatAuth,
    WebChatProvider,
    invalidate_auth,
    list_accounts,
    load_auth,
    save_auth,
)
from .copilot import CopilotWebChat
from .deepseek import DeepSeekWebChat
from .glm import GLMWebChat
from .kimi import KimiWebChat
from .registry import (
    check_all_webchat_auth,
    get_provider,
    webchat_call,
)

__all__ = [
    "WEBCHAT_CACHE_DIR",
    "WebChatAuth",
    "WebChatProvider",
    "GLMWebChat",
    "KimiWebChat",
    "DeepSeekWebChat",
    "AIStudioWebChat",
    "CopilotWebChat",
    "save_auth",
    "load_auth",
    "invalidate_auth",
    "list_accounts",
    "get_provider",
    "webchat_call",
    "check_all_webchat_auth",
]
