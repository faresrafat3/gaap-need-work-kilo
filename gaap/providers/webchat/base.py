"""
WebChat Provider Base - Abstract base class for browser-authenticated providers.

Pattern:
  1. Open browser → user logs in manually → capture auth token
  2. Cache token to disk (~/.config/gaap/webchat_auth/)
  3. Use curl_cffi with captured token for API calls (no browser needed)
  4. Token expires → re-open browser for fresh auth
"""

import asyncio
import contextlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from .auth import (
    WEBCHAT_CACHE_DIR,
    WebChatAuth,
    invalidate_auth,
    list_accounts,
    load_auth,
    save_auth,
)

logger = logging.getLogger("gaap.providers.webchat")


class WebChatProvider(ABC):
    """Abstract base for browser-authenticated web chat providers."""

    PROVIDER_NAME: str = ""
    URL: str = ""
    DEFAULT_MODEL: str = ""
    MODELS: list[str] = []
    TOKEN_LIFETIME_SEC: int = 43200

    def __init__(self, account: str = "default"):
        self.account = account
        self._auth: WebChatAuth | None = None

    @property
    def auth(self) -> WebChatAuth | None:
        """Get current auth (from cache if needed)."""
        if self._auth is None or self._auth.is_expired:
            self._auth = load_auth(self.PROVIDER_NAME, self.account)
        return self._auth

    @property
    def is_authenticated(self) -> bool:
        return self.auth is not None and not self.auth.is_expired

    def check_auth(self) -> dict[str, Any]:
        """Check auth status. Returns status dict."""
        auth = self.auth
        if auth is None:
            return {
                "valid": False,
                "provider": self.PROVIDER_NAME,
                "account": self.account,
                "message": "No auth cached — browser login needed",
            }
        return {
            "valid": not auth.is_expired,
            "provider": self.PROVIDER_NAME,
            "account": self.account,
            "remaining_sec": auth.remaining_sec,
            "user_id": auth.user_id,
            "message": (
                f"Valid — {auth.remaining_sec // 60}m remaining"
                if not auth.is_expired
                else "Expired — needs re-login"
            ),
        }

    def invalidate(self) -> None:
        """Clear cached auth."""
        invalidate_auth(self.PROVIDER_NAME, self.account)
        self._auth = None

    async def browser_login(self, proxy: str | None = None, timeout: int = 180) -> WebChatAuth:
        """
        Open browser for user login and capture auth token.

        This opens a visible Chrome window. The user must:
        1. Log in to the service
        2. Wait for the provider to capture the token
        3. The browser closes automatically

        Returns the captured WebChatAuth.
        """
        try:
            from g4f.requests import get_nodriver
        except ImportError:
            raise RuntimeError(
                "zendriver required for browser auth. Install: pip install zendriver platformdirs"
            )

        try:
            import nodriver
        except ImportError:
            import zendriver as nodriver

        print(f"  Opening {self.URL} for login...")
        print(f"  Please log in. Timeout: {timeout}s")

        browser_profile = f"gaap_{self.PROVIDER_NAME}_{self.account}"
        browser, stop_browser = await get_nodriver(
            proxy=proxy or "", timeout=timeout, user_data_dir=browser_profile
        )

        try:
            page = await browser.get(self.URL)

            for _ in range(30):
                ready = await page.evaluate(
                    "!!document.querySelector('body')", return_by_value=True
                )
                if ready:
                    break
                await asyncio.sleep(1)

            user_agent: Any = await page.evaluate(
                "window.navigator.userAgent", return_by_value=True
            )
            if not isinstance(user_agent, str):
                user_agent = ""

            auth = await self._capture_auth(page, nodriver, timeout)
            auth.user_agent = user_agent
            auth.captured_at = time.time()
            if auth.expires_at <= 0:
                auth.expires_at = auth.captured_at + self.TOKEN_LIFETIME_SEC

            cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([self.URL])):
                cookies[c.name] = c.value
            auth.cookies = cookies

            save_auth(auth)
            self._auth = auth

            print(f"  Auth captured for {self.PROVIDER_NAME} [{self.account}]")
            return auth
        finally:
            await stop_browser()

    @abstractmethod
    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Provider-specific auth capture from browser page."""
        ...

    @abstractmethod
    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Send a chat completion request. Returns response text."""
        ...

    def warmup(
        self, proxy: str | None = None, force: bool = False, timeout: int = 300
    ) -> dict[str, Any]:
        """Synchronous warmup: check auth and login if needed."""
        status = self.check_auth()
        if status["valid"] and not force:
            print(f"  {self.PROVIDER_NAME} [{self.account}]: {status['message']}")
            return status

        print(f"  {self.PROVIDER_NAME} [{self.account}]: {status['message']}")

        loop = None
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_running_loop()

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(
                    lambda: asyncio.run(self.browser_login(proxy=proxy, timeout=timeout))
                ).result(timeout=timeout + 30)
        else:
            asyncio.run(self.browser_login(proxy=proxy, timeout=timeout))

        return self.check_auth()


__all__ = [
    "WEBCHAT_CACHE_DIR",
    "WebChatAuth",
    "WebChatProvider",
    "save_auth",
    "load_auth",
    "invalidate_auth",
    "list_accounts",
]
