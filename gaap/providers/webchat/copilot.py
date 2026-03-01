"""
GitHub Copilot WebChat Provider (api.githubcopilot.com)
=======================================================

GitHub Copilot via api.githubcopilot.com — OpenAI-compatible API.

Auth:   GitHub OAuth device flow (no browser scraping needed).
        One-time: visit github.com/login/device + enter code.
        Token persists indefinitely (GitHub OAuth tokens don't expire).
Token:  GitHub access_token → exchanged for short-lived Copilot API token
        (~30 min). Auto-refreshed transparently before each call.
API:    Standard OpenAI chat/completions with SSE streaming.
"""

import asyncio
import json
import logging
import time
from typing import Any

from .base import WebChatAuth, WebChatProvider, save_auth

logger = logging.getLogger("gaap.providers.webchat")


class CopilotWebChat(WebChatProvider):
    """
    GitHub Copilot via api.githubcopilot.com — OpenAI-compatible API.

    Auth:   GitHub OAuth device flow (no browser scraping needed).
            One-time: visit github.com/login/device + enter code.
            Token persists indefinitely (GitHub OAuth tokens don't expire).
    Token:  GitHub access_token → exchanged for short-lived Copilot API token
            (~30 min). Auto-refreshed transparently before each call.
    API:    Standard OpenAI chat/completions with SSE streaming.
    """

    PROVIDER_NAME = "copilot"
    URL = "https://github.com/copilot"
    DEFAULT_MODEL = "gpt-5"
    MODELS = [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "claude-opus-4.6",
        "claude-opus-4.6-fast",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "grok-code-fast-1",
    ]
    TOKEN_LIFETIME_SEC = 365 * 86400

    _CLIENT_ID = "Iv1.b507a08c87ecfe98"
    _COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
    _BASE_URL = "https://api.githubcopilot.com"
    _EDITOR_VERSION = "vscode/1.95.0"
    _EDITOR_PLUGIN_VERSION = "copilot/1.250.0"

    _MAX_RESPONSE_BYTES = 512 * 1024

    def __init__(self, account: str = "default"):
        super().__init__(account)
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0

    async def browser_login(self, proxy: str | None = None, timeout: int = 300) -> WebChatAuth:
        """
        GitHub OAuth device flow — no browser scraping needed.

        The user visits github.com/login/device and enters a short code.
        After authorization, the access_token is saved to disk and persists
        indefinitely (GitHub OAuth tokens don't expire).
        """
        from curl_cffi import requests as cf_requests

        r = cf_requests.post(
            "https://github.com/login/device/code",
            data=f"client_id={self._CLIENT_ID}&scope=read:user",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            impersonate="chrome",
            timeout=15,
        )
        if r.status_code != 200:
            raise RuntimeError(f"GitHub device auth failed: {r.status_code} — {r.text[:200]}")

        device_data = r.json()
        user_code = device_data["user_code"]
        device_code = device_data["device_code"]
        verification_uri = device_data.get("verification_uri", "https://github.com/login/device")
        interval = device_data.get("interval", 5)
        expires_in = device_data.get("expires_in", 900)

        print(f"\n{'=' * 60}")
        print("  GitHub Copilot OAuth Login")
        print(f"{'=' * 60}")
        print(f"\n  1. Visit:  {verification_uri}")
        print(f"  2. Enter:  {user_code}")
        print(f"\n{'=' * 60}")

        try:
            import webbrowser

            webbrowser.open(verification_uri)
            print("  (Browser opened automatically)")
        except Exception as e:
            logger.debug(f"Browser open failed: {e}")

        print("\n  Waiting for authorization", end="", flush=True)
        start = time.time()
        while time.time() - start < min(timeout, expires_in):
            await asyncio.sleep(interval)

            r = cf_requests.post(
                "https://github.com/login/oauth/access_token",
                data=(
                    f"grant_type=urn:ietf:params:oauth:grant-type:device_code"
                    f"&client_id={self._CLIENT_ID}"
                    f"&device_code={device_code}"
                ),
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                impersonate="chrome",
                timeout=15,
            )

            result = r.json()

            if "access_token" in result:
                print("\n  ✅ Authorization successful!")

                github_token = result["access_token"]
                try:
                    self._exchange_copilot_token(github_token)
                    print("  ✅ Copilot API access confirmed!")
                except Exception as e:
                    print(f"  ⚠️  Token obtained but Copilot exchange failed: {e}")
                    print("     (You may need Copilot Pro access)")

                auth = WebChatAuth(
                    provider=self.PROVIDER_NAME,
                    token=github_token,
                    account_label=self.account,
                    captured_at=time.time(),
                    expires_at=time.time() + self.TOKEN_LIFETIME_SEC,
                )

                save_auth(auth)
                self._auth = auth
                return auth

            error = result.get("error", "")
            if error == "authorization_pending":
                print(".", end="", flush=True)
                continue
            elif error == "slow_down":
                interval += 5
                continue
            elif error == "expired_token":
                raise RuntimeError("Device code expired. Try again.")
            elif error == "access_denied":
                raise RuntimeError("Authorization denied by user.")
            else:
                raise RuntimeError(f"OAuth error: {error} — {result.get('error_description', '')}")

        raise RuntimeError("OAuth timeout — no authorization received")

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Not used — Copilot uses OAuth device flow instead of browser scraping."""
        raise NotImplementedError(
            "CopilotWebChat uses OAuth device flow. Call browser_login() directly."
        )

    def _exchange_copilot_token(self, github_token: str) -> str:
        """Exchange GitHub OAuth token for a short-lived Copilot API token."""
        from curl_cffi import requests as cf_requests

        r = cf_requests.get(
            self._COPILOT_TOKEN_URL,
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/json",
                "User-Agent": "GithubCopilot/1.250.0",
                "Editor-Version": self._EDITOR_VERSION,
                "Editor-Plugin-Version": self._EDITOR_PLUGIN_VERSION,
                "Openai-Organization": "github-copilot",
                "X-GitHub-Api-Version": "2024-12-15",
            },
            impersonate="chrome",
            timeout=15,
        )

        if r.status_code == 401:
            raise PermissionError("GitHub token invalid/revoked. Re-run warmup(force=True).")
        if r.status_code != 200:
            raise RuntimeError(f"Copilot token exchange failed: {r.status_code} — {r.text[:200]}")

        data: dict[str, Any] = r.json()
        token: str = data.get("token", "") or ""
        if not token:
            raise RuntimeError(f"No token in Copilot exchange response: {r.text[:200]}")

        expires_at_str = data.get("expires_at", "")
        if expires_at_str:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                self._copilot_token_expires = dt.timestamp()
            except Exception as e:
                logger.debug(f"Copilot token expiry parsing failed: {e}")
                self._copilot_token_expires = time.time() + 1800
        else:
            self._copilot_token_expires = time.time() + 1800

        self._copilot_token = token
        return token

    def _get_copilot_token(self) -> str:
        """Get a valid Copilot API token, refreshing if needed."""
        if self._copilot_token and time.time() < self._copilot_token_expires - 60:
            return self._copilot_token

        auth = self.auth
        if not auth:
            raise RuntimeError(f"Copilot [{self.account}]: Not authenticated. Run warmup() first.")

        return self._exchange_copilot_token(auth.token)

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call OpenAI-compatible Copilot API with SSE streaming."""
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"Copilot [{self.account}]: Not authenticated. Run warmup() first.")

        model = model or self.DEFAULT_MODEL
        copilot_token = self._get_copilot_token()

        headers = {
            "Authorization": f"Bearer {copilot_token}",
            "Content-Type": "application/json",
            "Editor-Version": self._EDITOR_VERSION,
            "Editor-Plugin-Version": self._EDITOR_PLUGIN_VERSION,
            "Openai-Organization": "github-copilot",
            "Copilot-Integration-Id": "vscode-chat",
            "X-GitHub-Api-Version": "2024-12-15",
        }

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }

        r = cf_requests.post(
            f"{self._BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
            impersonate="chrome",
            timeout=timeout,
            stream=True,
        )

        if r.status_code in (401, 403):
            self._copilot_token = None
            copilot_token = self._get_copilot_token()
            headers["Authorization"] = f"Bearer {copilot_token}"

            r = cf_requests.post(
                f"{self._BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                impersonate="chrome",
                timeout=timeout,
                stream=True,
            )

        if r.status_code in (401, 403):
            raise PermissionError(
                f"Copilot auth rejected ({r.status_code}). "
                f"Run --webchat-login copilot to re-authenticate."
            )

        if r.status_code != 200:
            raise RuntimeError(f"Copilot API error {r.status_code}: {r.text[:200]}")

        return self._parse_openai_sse(r)

    def _parse_openai_sse(self, response: Any) -> str:
        """Parse standard OpenAI SSE streaming response."""
        result_parts = []
        total_len = 0

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace").strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                result_parts.append(content)
                                total_len += len(content)
                                if total_len > self._MAX_RESPONSE_BYTES:
                                    break
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

        result = "".join(result_parts).strip()
        del result_parts
        if not result:
            raise ValueError("Empty response from Copilot — no content in SSE stream")
        return result
