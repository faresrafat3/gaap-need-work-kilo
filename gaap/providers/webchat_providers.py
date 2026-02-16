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
"""

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import os
import struct
import subprocess
import time
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# =============================================================================
# Auth Cache
# =============================================================================

WEBCHAT_CACHE_DIR = Path.home() / ".config" / "gaap" / "webchat_auth"


@dataclass
class WebChatAuth:
    """Cached authentication state for a webchat provider."""

    provider: str
    token: str
    cookies: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    user_id: str = ""
    account_label: str = "default"
    captured_at: float = 0.0
    expires_at: float = 0.0  # 0 = unknown

    @property
    def is_expired(self) -> bool:
        if self.expires_at <= 0:
            # No known expiry → consider valid for 12 hours max
            return (time.time() - self.captured_at) > 43200
        return time.time() >= self.expires_at

    @property
    def remaining_sec(self) -> int:
        if self.expires_at <= 0:
            return max(0, int(43200 - (time.time() - self.captured_at)))
        return max(0, int(self.expires_at - time.time()))

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "token": self.token,
            "cookies": self.cookies,
            "headers": self.headers,
            "user_agent": self.user_agent,
            "user_id": self.user_id,
            "account_label": self.account_label,
            "captured_at": self.captured_at,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WebChatAuth":
        return cls(
            provider=d.get("provider", ""),
            token=d.get("token", ""),
            cookies=d.get("cookies", {}),
            headers=d.get("headers", {}),
            user_agent=d.get("user_agent", ""),
            user_id=d.get("user_id", ""),
            account_label=d.get("account_label", "default"),
            captured_at=d.get("captured_at", 0.0),
            expires_at=d.get("expires_at", 0.0),
        )


def _cache_path(provider: str, account: str = "default") -> Path:
    return WEBCHAT_CACHE_DIR / f"{provider}_{account}.json"


def save_auth(auth: WebChatAuth) -> Path:
    """Save auth to disk cache."""
    WEBCHAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(auth.provider, auth.account_label)
    path.write_text(json.dumps(auth.to_dict(), indent=2), encoding="utf-8")
    return path


def load_auth(provider: str, account: str = "default") -> WebChatAuth | None:
    """Load auth from disk cache. Returns None if missing or expired."""
    path = _cache_path(provider, account)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        auth = WebChatAuth.from_dict(data)
        if auth.is_expired:
            return None
        return auth
    except (OSError, json.JSONDecodeError):
        return None


def invalidate_auth(provider: str, account: str = "default") -> bool:
    """Delete cached auth file."""
    path = _cache_path(provider, account)
    if path.exists():
        path.unlink()
        return True
    return False


def list_accounts(provider: str) -> list[str]:
    """List all cached account labels for a provider."""
    if not WEBCHAT_CACHE_DIR.exists():
        return []
    prefix = f"{provider}_"
    accounts = []
    for f in WEBCHAT_CACHE_DIR.iterdir():
        if f.name.startswith(prefix) and f.name.endswith(".json"):
            label = f.name[len(prefix) : -5]
            accounts.append(label)
    return sorted(accounts)


# =============================================================================
# Base WebChat Provider
# =============================================================================


class WebChatProvider(ABC):
    """Abstract base for browser-authenticated web chat providers."""

    PROVIDER_NAME: str = ""
    URL: str = ""
    DEFAULT_MODEL: str = ""
    MODELS: list[str] = []
    TOKEN_LIFETIME_SEC: int = 43200  # 12h default

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
            import nodriver  # g4f aliases zendriver → nodriver
        except ImportError:
            import zendriver as nodriver

        print(f"  🌐 Opening {self.URL} for login...")
        print(f"  ⏳ Please log in. Timeout: {timeout}s")

        # Each account gets its own browser profile so sessions don't conflict
        browser_profile = f"gaap_{self.PROVIDER_NAME}_{self.account}"
        browser, stop_browser = await get_nodriver(
            proxy=proxy or "", timeout=timeout, user_data_dir=browser_profile
        )

        try:
            page = await browser.get(self.URL)

            # Wait for body to load
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

            # Capture cookies
            cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([self.URL])):
                cookies[c.name] = c.value
            auth.cookies = cookies

            # Save to cache
            save_auth(auth)
            self._auth = auth

            print(f"  ✅ Auth captured for {self.PROVIDER_NAME} [{self.account}]")
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
            print(f"  ✅ {self.PROVIDER_NAME} [{self.account}]: {status['message']}")
            return status

        print(f"  🔐 {self.PROVIDER_NAME} [{self.account}]: {status['message']}")

        # Run async browser login in sync context
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


# =============================================================================
# GLM-5 WebChat Provider (chat.z.ai)
# =============================================================================


class GLMWebChat(WebChatProvider):
    """
    GLM-5 / GLM-4.7 via chat.z.ai (Open WebUI based).

    Auth: Browser login → capture JWT from /api/v1/auths/ or intercepted headers.
    API:  POST /api/chat/completions with HMAC signature.
    SSE:  type=chat:completion with delta_content/edit_content.
    """

    PROVIDER_NAME = "glm"
    URL = "https://chat.z.ai"
    DEFAULT_MODEL = "GLM-5"
    MODELS = ["GLM-5", "GLM-4.7", "GLM-4.6", "GLM-4.5", "GLM-4.5V", "Z1-Rumination"]
    TOKEN_LIFETIME_SEC = 86400  # 24h

    # Model name → API ID mapping
    MODEL_ALIASES = {
        "GLM-5": "glm-5",
        "GLM-4.7": "glm-4.7",
        "GLM-4.6": "GLM-4-6-API-V1",
        "GLM-4.5": "GLM-4.5",
        "GLM-4.5V": "glm-4.5v",
        "Z1-Rumination": "Z1-Rumination",
    }

    @staticmethod
    def _is_real_user_token(token: str, cookies: dict[str, str] | None = None) -> dict[str, Any]:
        """
        Validate token against /api/v1/auths/ and check it's not a guest.
        Returns {'valid': bool, 'role': str, 'user_id': str, 'token': str}.
        """
        try:
            from curl_cffi import requests as cf_requests

            r = cf_requests.get(
                "https://chat.z.ai/api/v1/auths/",
                headers={"Authorization": f"Bearer {token}"},
                cookies=cookies or {},
                impersonate="chrome",
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                role = data.get("role", "guest")
                return {
                    "valid": role != "guest",
                    "role": role,
                    "user_id": data.get("id", ""),
                    "token": data.get("token", token),
                }
        except Exception:
            pass
        return {"valid": False, "role": "unknown", "user_id": "", "token": token}

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Capture auth token from chat.z.ai logged-in session."""
        captured: dict[str, Any] = {"token": None, "user_id": None}

        # Intercept API requests to grab Authorization header
        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            headers = event.request.headers if hasattr(event.request, "headers") else {}
            if "chat.z.ai/api" in url:
                auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and len(auth_header) > 20:
                    token = auth_header[7:]
                    # Only capture if it's a real user token (not guest)
                    info = self._is_real_user_token(token)
                    if info["valid"]:
                        captured["token"] = info["token"] or token
                        captured["user_id"] = info["user_id"]

        await page.send(nodriver_module.cdp.network.enable())
        page.add_handler(nodriver_module.cdp.network.RequestWillBeSent, on_request)

        print("  💡 سجل دخول في chat.z.ai في نافذة البراوزر")
        print("  💡 بعد التسجيل، ابعت أي رسالة في الشات عشان نمسك الـ token")
        print(f"  ⏳ مستنيك... (timeout: {timeout}s)")

        start = time.time()
        check_interval = 3
        last_status = ""
        while time.time() - start < timeout:
            if captured["token"]:
                break

            # Check localStorage token and validate it's a real user
            try:
                token = await page.evaluate("localStorage.getItem('token')", return_by_value=True)
                if token and len(token) > 20:
                    info = self._is_real_user_token(token)
                    if info["valid"]:
                        captured["token"] = info["token"] or token
                        captured["user_id"] = info["user_id"]
                        print(f"  🔑 Token captured! Role: {info['role']}")
                        break
                    elif info["role"] == "guest" and last_status != "guest":
                        last_status = "guest"
                        elapsed = int(time.time() - start)
                        print(f"  ⏳ [{elapsed}s] لسه guest... سجل دخول الأول")
            except Exception:
                pass

            # Print periodic waiting message
            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 15 == 0 and last_status != str(elapsed):
                last_status = str(elapsed)
                remaining = timeout - elapsed
                print(f"  ⏳ [{elapsed}s] مستنيك تسجل... (باقي {remaining}s)")

            await asyncio.sleep(check_interval)

        if not captured["token"]:
            raise RuntimeError(
                "مقدرتش أمسك token مسجل. تأكد إنك سجلت دخول في chat.z.ai وبعت رسالة في الشات."
            )

        # Get user ID if not captured from validation
        if not captured["user_id"] and captured["token"]:
            info = self._is_real_user_token(captured["token"])
            if info["user_id"]:
                captured["user_id"] = info["user_id"]

        return WebChatAuth(
            provider=self.PROVIDER_NAME,
            token=captured["token"],
            user_id=captured["user_id"] or "",
            account_label=self.account,
        )

    @staticmethod
    def _create_signature(sorted_payload: str, prompt: str, timestamp_ms: int) -> tuple[str, int]:
        """HMAC-SHA256 signature for chat.z.ai anti-bot.

        Algorithm (reverse-engineered from js-sha256 in prod-fe-1.0.240):
        1. base64 encode the prompt (UTF-8 bytes -> base64)
        2. data_string = sorted_payload|base64(prompt)|timestamp_ms
        3. time_window = timestamp_ms // (5 * 60 * 1000)
        4. base_key = HMAC-SHA256(SIGNING_KEY, str(time_window))
        5. signature = HMAC-SHA256(base_key, data_string)
        """
        SIGNING_KEY = "key-@@@@)))()((9))-xxxx&&&%%%%%"

        prompt_b64 = base64.b64encode(prompt.encode("utf-8")).decode("ascii")
        data_string = f"{sorted_payload}|{prompt_b64}|{timestamp_ms}"
        time_window = timestamp_ms // (5 * 60 * 1000)

        base_key = hmac.new(
            SIGNING_KEY.encode("utf-8"),
            str(time_window).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        sig = hmac.new(
            base_key.encode("utf-8"),
            data_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return sig, timestamp_ms

    def _build_signed_url(self, prompt: str) -> tuple[str, str]:
        """Build signed URL with all anti-bot params. Returns (url, signature)."""
        auth = self.auth
        if not auth:
            raise RuntimeError("Not authenticated")

        current_time_ms = int(time.time() * 1000)
        current_time = str(current_time_ms)
        request_id = str(uuid.uuid1())

        basic_params = {
            "timestamp": current_time,
            "requestId": request_id,
            "user_id": auth.user_id or "anonymous",
        }

        additional_params = {
            "version": "0.0.1",
            "platform": "web",
            "token": auth.token,
            "user_agent": auth.user_agent
            or "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36",
            "language": "en-US",
            "languages": "en-US,en",
            "timezone": "UTC",
            "cookie_enabled": "true",
            "screen_width": "1920",
            "screen_height": "1080",
            "screen_resolution": "1920x1080",
            "viewport_height": "900",
            "viewport_width": "1440",
            "viewport_size": "1440x900",
            "color_depth": "24",
            "pixel_ratio": "1",
            "current_url": f"{self.URL}/",
            "pathname": "/",
            "search": "",
            "hash": "",
            "host": "chat.z.ai",
            "hostname": "chat.z.ai",
            "protocol": "https:",
            "referrer": "",
            "title": "Z.ai - Free AI Chatbot & Agent powered by GLM-5 & GLM-4.7",
            "timezone_offset": "0",
            "local_time": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime()),
            "utc_time": time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime()),
            "is_mobile": "false",
            "is_touch": "false",
            "max_touch_points": "0",
            "browser_name": "Chrome",
            "os_name": "Linux",
        }

        all_params = {**basic_params, **additional_params}
        url_params = urllib.parse.urlencode(all_params)

        sorted_payload = ",".join(f"{k},{v}" for k, v in sorted(basic_params.items()))

        signature, timestamp = self._create_signature(
            sorted_payload, prompt.strip(), current_time_ms
        )

        full_url = (
            f"{self.URL}/api/v2/chat/completions?{url_params}&signature_timestamp={timestamp}"
        )
        return full_url, signature

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call GLM via chat.z.ai API with HMAC signature."""
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"GLM [{self.account}]: Not authenticated. Run warmup() first.")

        model = model or self.DEFAULT_MODEL
        model_id = self.MODEL_ALIASES.get(model, model)

        # Get last user message for signature
        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

        # Build signed URL
        url, signature = self._build_signed_url(user_prompt)

        str(uuid.uuid4())
        payload = {
            "stream": True,
            "model": model_id,
            "messages": messages,
            "signature_prompt": user_prompt,
            "params": {},
            "extra": {},
            "features": {
                "image_generation": False,
                "web_search": False,
                "auto_web_search": False,
                "preview_mode": True,
                "flags": [],
                "enable_thinking": True,
            },
            "variables": {},
        }

        headers = {
            "Authorization": f"Bearer {auth.token}",
            "Content-Type": "application/json",
            "X-FE-Version": "prod-fe-1.0.240",
            "X-Signature": signature,
            "Origin": self.URL,
            "Referer": f"{self.URL}/",
        }
        if auth.user_agent:
            headers["User-Agent"] = auth.user_agent

        r = cf_requests.post(
            url,
            json=payload,
            headers=headers,
            cookies=auth.cookies,
            impersonate="chrome",
            timeout=timeout,
            stream=True,
        )

        if r.status_code == 401 or r.status_code == 403:
            raise PermissionError(
                f"GLM auth rejected ({r.status_code}): {r.text[:150]}. "
                f"Run --webchat-login glm to re-login."
            )

        if r.status_code != 200:
            raise RuntimeError(f"GLM API error {r.status_code}: {r.text[:200]}")

        # Parse SSE streaming response
        return self._parse_sse_stream(r)

    # Maximum response text size (512KB) to prevent OOM
    _MAX_RESPONSE_BYTES = 512 * 1024

    def _parse_sse_stream(self, response: Any) -> str:
        """Parse SSE stream from chat.z.ai into plain text (streaming)."""
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
                        chunk_data = json.loads(data_str)
                        if chunk_data.get("type") == "chat:completion":
                            data = chunk_data.get("data", {})
                            phase = data.get("phase", "")
                            if phase == "thinking":
                                continue  # Skip thinking tokens
                            edit_content = data.get("edit_content", "")
                            if edit_content:
                                cleaned = edit_content.split("\n</details>\n")[-1]
                                if cleaned:
                                    result_parts.append(cleaned)
                                    total_len += len(cleaned)
                            else:
                                delta = data.get("delta_content", "")
                                if delta:
                                    result_parts.append(delta)
                                    total_len += len(delta)
                        # Safety: abort if response is too large
                        if total_len > self._MAX_RESPONSE_BYTES:
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

        result = "".join(result_parts).strip()
        del result_parts  # free list memory immediately
        if not result:
            raise ValueError("Empty response from GLM - no content in SSE stream")
        return result


# =============================================================================
# Kimi K2.5 WebChat Provider (kimi.com)
# =============================================================================


class KimiWebChat(WebChatProvider):
    """
    Kimi K2.5 Thinking via kimi.com web chat.

    Protocol: Connect-RPC over HTTP (not REST).
    Auth:     Browser login → capture JWT Bearer token.
    Service:  kimi.gateway.chat.v1.ChatService/Chat (server-streaming).
    Base URL: https://www.kimi.com/apiv2/
    Framing:  Connect streaming envelope: [flags(1B), length(4B BE), payload].

    Reverse-engineered from kimi.com JS bundle (common-CyC_syon.js).
    Protobuf schemas decoded from base64-encoded strings in the bundle.
    """

    PROVIDER_NAME = "kimi"
    URL = "https://www.kimi.com"
    DEFAULT_MODEL = "kimi"
    MODELS = ["kimi-k2.5-thinking", "kimi-k2", "kimi-k2-thinking", "kimi-research", "kimi"]
    DEFAULT_MODEL = "kimi-k2.5-thinking"
    TOKEN_LIFETIME_SEC = 2592000  # JWT valid ~30 days

    # Connect-RPC endpoint
    _CHAT_ENDPOINT = "/apiv2/kimi.gateway.chat.v1.ChatService/Chat"

    # Model → Scenario enum mapping (from Kimi web app JS bundle)
    MODEL_SCENARIOS = {
        "kimi-k2.5-thinking": "SCENARIO_K2D5",
        "kimi-k2": "SCENARIO_K2",
        "kimi-k2-thinking": "SCENARIO_K2_THINKING",
        "kimi-research": "SCENARIO_RESEARCH",
        "kimi": "SCENARIO_K2D5",  # default to K2.5
    }

    # Legacy alias
    _SCENARIO = "SCENARIO_K2D5"

    @staticmethod
    def _connect_envelope(flags: int, data: bytes) -> bytes:
        """Create a Connect streaming envelope: [flags(1B), length(4B BE), data]."""
        return struct.pack(">BI", flags, len(data)) + data

    @staticmethod
    def _parse_connect_stream(content: bytes) -> list:
        """Parse Connect streaming response into list of (flags, json_dict) tuples."""
        messages = []
        idx = 0
        while idx + 5 <= len(content):
            flags = content[idx]
            length = struct.unpack(">I", content[idx + 1 : idx + 5])[0]
            data = content[idx + 5 : idx + 5 + length]
            idx += 5 + length
            try:
                messages.append((flags, json.loads(data)))
            except (json.JSONDecodeError, UnicodeDecodeError):
                messages.append((flags, {"_raw": data}))
        return messages

    @staticmethod
    def _extract_user_id_from_jwt(token: str) -> str:
        """Extract 'sub' (user ID) from JWT payload without verification."""
        try:
            parts = token.split(".")
            if len(parts) < 2:
                return ""
            # Pad base64
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return str(payload.get("sub", ""))
        except Exception:
            return ""

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Capture auth from kimi.com by intercepting requests or localStorage."""
        captured: dict[str, Any] = {"token": None, "user_id": None}

        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            headers = event.request.headers if hasattr(event.request, "headers") else {}
            if "kimi.com/api" in url or "kimi.com/apiv2" in url:
                auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and len(auth_header) > 50:
                    token = auth_header[7:]
                    if token not in ("undefined", "null"):
                        captured["token"] = token
                        captured["user_id"] = self._extract_user_id_from_jwt(token)

        await page.send(nodriver_module.cdp.network.enable())
        page.add_handler(nodriver_module.cdp.network.RequestWillBeSent, on_request)

        print("  💡 Please log in to kimi.com in the browser window")
        print("  💡 After login, the token will be captured automatically")
        print(f"  ⏳ Waiting... (timeout: {timeout}s)")

        start = time.time()
        while time.time() - start < timeout:
            if captured["token"]:
                break

            # Try localStorage / cookie fallbacks
            for js_expr in [
                "localStorage.getItem('access_token')",
                "localStorage.getItem('token')",
                "document.cookie.split('; ')"
                ".find(c => c.startsWith('access_token='))"
                "?.split('=').slice(1).join('=')",
            ]:
                try:
                    val = await page.evaluate(js_expr, return_by_value=True)
                    if (
                        val
                        and isinstance(val, str)
                        and len(val) > 50
                        and val not in ("undefined", "null")
                    ):
                        captured["token"] = val
                        captured["user_id"] = self._extract_user_id_from_jwt(val)
                        break
                except Exception:
                    pass

            if captured["token"]:
                break

            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 15 == 0:
                print(f"  ⏳ [{elapsed}s] Waiting for login... ({timeout - elapsed}s left)")

            await asyncio.sleep(2)

        if not captured["token"]:
            raise RuntimeError(
                "Failed to capture auth token from kimi.com. "
                "Please ensure you logged in successfully."
            )

        # Validate token with a lightweight API call
        user_id = captured.get("user_id") or ""
        try:
            from curl_cffi import requests as cf_requests

            cookies = {}
            for c in await page.send(nodriver_module.cdp.network.get_cookies([self.URL])):
                cookies[c.name] = c.value

            r = cf_requests.get(
                f"{self.URL}/api/user",
                headers={
                    "Authorization": f"Bearer {captured['token']}",
                    "Referer": f"{self.URL}/",
                    "x-msh-platform": "web",
                },
                cookies=cookies,
                impersonate="chrome",
                timeout=10,
            )
            if r.status_code == 200:
                user_data = r.json()
                user_id = user_data.get("id", "") or user_id
                print(f"  🔑 Token validated! User: {user_id}")
        except Exception:
            pass

        # Parse JWT expiry
        expires_at = 0.0
        try:
            parts = captured["token"].split(".")
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp", 0)
            if exp:
                expires_at = float(exp)
        except Exception:
            pass

        return WebChatAuth(
            provider=self.PROVIDER_NAME,
            token=captured["token"],
            user_id=user_id,
            account_label=self.account,
            expires_at=expires_at,
        )

    def _resolve_scenario(self, model: str = "") -> str:
        """Resolve model name to Kimi scenario enum."""
        if not model:
            model = self.DEFAULT_MODEL
        return self.MODEL_SCENARIOS.get(model, self._SCENARIO)

    def _build_chat_request(self, user_content: str, model: str = "") -> dict:
        """Build a kimi.gateway.chat.v1.ChatRequest JSON payload.

        Uses useProtoFieldName=true convention (snake_case field names).
        Model param selects the scenario (K2.5, K2, thinking, research).
        """
        scenario = self._resolve_scenario(model)
        return {
            "chat_id": "",
            "kimiplus_id": "",
            "scenario": scenario,
            "message": {
                "id": "",
                "parent_id": "",
                "role": "user",
                "blocks": [
                    {
                        "id": "",
                        "text": {"content": user_content},
                    }
                ],
                "scenario": scenario,
                "labels": [],
            },
            "options": {},
        }

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call Kimi K2.5 via Connect-RPC (kimi.com/apiv2/).

        Takes standard OpenAI-style messages and sends the last user message
        as a new conversation via the Connect streaming protocol.
        """
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"Kimi [{self.account}]: Not authenticated. Run warmup() first.")

        # Extract user content from messages
        user_content = ""
        system_prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_content = content

        if not user_content:
            raise ValueError("No user message found in messages list")

        # Prepend system prompt if present
        if system_prompt:
            user_content = f"{system_prompt}\n\n{user_content}"

        # Build Connect-RPC request
        payload = self._build_chat_request(user_content, model=model)

        # Serialize to JSON and wrap in Connect streaming envelope
        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        body = self._connect_envelope(0, json_bytes)

        user_id = auth.user_id or self._extract_user_id_from_jwt(auth.token)

        headers = {
            "Authorization": f"Bearer {auth.token}",
            "Content-Type": "application/connect+json",
            "Connect-Protocol-Version": "1",
            "x-msh-platform": "web",
            "R-Timezone": "Africa/Cairo",
            "X-Traffic-Id": user_id,
            "Referer": f"{self.URL}/",
            "Origin": self.URL,
        }
        if auth.user_agent:
            headers["User-Agent"] = auth.user_agent

        r = cf_requests.post(
            f"{self.URL}{self._CHAT_ENDPOINT}",
            data=body,
            headers=headers,
            cookies=auth.cookies,
            impersonate="chrome",
            timeout=timeout,
        )

        if r.status_code in (401, 403):
            raise PermissionError(
                f"Kimi auth rejected ({r.status_code}). Run --webchat-login kimi to re-login."
            )

        if r.status_code != 200:
            raise RuntimeError(f"Kimi API error {r.status_code}: {r.content[:200]}")

        return self._parse_connect_response(r.content)

    # Maximum response text size (512KB) to prevent OOM
    _MAX_RESPONSE_BYTES = 512 * 1024

    def _parse_connect_response(self, content: bytes) -> str:
        """Parse Connect streaming envelopes and extract text content.

        Response format (from kimi.gateway.chat.v1.ChatResponse):
          - event.block.text.content  → the actual text
          - op: "set" | "append"      → how to apply the text
          - event.done               → signals completion
          - flags=0x02               → end-of-stream trailer
        """
        envelopes = self._parse_connect_stream(content)

        full_text = ""
        error_msg = ""

        for flags, msg in envelopes:
            if not isinstance(msg, dict):
                continue

            # Check for Connect-RPC error in end-stream trailer
            if flags == 0x02:
                err = msg.get("error", {})
                if err:
                    code = err.get("code", "unknown")
                    detail = err.get("message", "") or json.dumps(err)
                    error_msg = f"Kimi Connect-RPC error: {code} - {detail}"
                continue

            # Check for application-level error events
            if "error" in msg:
                err = msg["error"]
                error_msg = f"Kimi error: {err}" if isinstance(err, str) else json.dumps(err)
                continue

            op = msg.get("op", "")

            # Extract text from block events
            block = msg.get("block")
            if block and isinstance(block, dict):
                text_block = block.get("text")
                if text_block and isinstance(text_block, dict):
                    text_content = text_block.get("content", "")
                    if text_content:
                        if op == "append":
                            full_text += text_content
                        else:
                            # "set" replaces (usually the first/only chunk)
                            full_text = text_content

            # Safety: abort if response is too large
            if len(full_text) > self._MAX_RESPONSE_BYTES:
                break

        del envelopes  # free parsed envelopes immediately
        result = full_text.strip()
        del full_text
        if not result:
            if error_msg:
                raise RuntimeError(error_msg)
            raise ValueError("Empty response from Kimi - no text content in stream")

        return result


# =============================================================================
# DeepSeek WebChat Provider (chat.deepseek.com)
# =============================================================================


class DeepSeekWebChat(WebChatProvider):
    """
    DeepSeek V3.2 / R2 via chat.deepseek.com.

    Protocol: REST + custom SSE (text/event-stream).
    Auth:     Browser login (Google OAuth) → capture opaque Bearer token.
    PoW:      DeepSeekHashV1 — custom 23-round Keccak (NOT SHA3-256).
              Solved via Node.js subprocess (deepseek_pow_solver.js).

    Flow:
      1. POST /api/v0/chat_session/create → session_id
      2. POST /api/v0/chat/create_pow_challenge → PoW challenge
      3. Solve PoW via Node.js → answer
      4. POST /api/v0/chat/completion (+ x-ds-pow-response header) → SSE

    SSE format (custom, NOT OpenAI-compatible):
      - event: ready/update_session/title/close
      - data: {"v":{"response":{"fragments":[{"content":"..."}]}}}
      - data: {"p":"response/status","o":"SET","v":"FINISHED"}
    """

    PROVIDER_NAME = "deepseek"
    URL = "https://chat.deepseek.com"
    DEFAULT_MODEL = "deepseek"
    MODELS = ["deepseek", "deepseek-thinking"]
    TOKEN_LIFETIME_SEC = 2592000  # ~30 days (opaque token, long-lived)

    _BASE_API = "https://chat.deepseek.com/api/v0"
    _POW_SOLVER_PATH = Path(__file__).parent / "deepseek_pow_solver.js"

    # Standard headers for all DeepSeek API requests
    _DS_HEADERS = {
        "x-client-locale": "en_US",
        "x-app-version": "20241129.1",
        "x-client-version": "1.7.0",
        "x-client-platform": "web",
        "Origin": "https://chat.deepseek.com",
        "Referer": "https://chat.deepseek.com/",
    }

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Capture Bearer token from chat.deepseek.com requests."""
        captured: dict[str, Any] = {"token": None}

        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            headers = event.request.headers if hasattr(event.request, "headers") else {}
            if "chat.deepseek.com/api" in url:
                auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and len(auth_header) > 30:
                    token = auth_header[7:]
                    # Validate token with a quick API call
                    try:
                        from curl_cffi import requests as cf_requests

                        r = cf_requests.get(
                            f"{self._BASE_API}/users/current",
                            headers={
                                "Authorization": f"Bearer {token}",
                                **self._DS_HEADERS,
                            },
                            impersonate="chrome",
                            timeout=10,
                        )
                        if r.status_code == 200:
                            captured["token"] = token
                    except Exception:
                        # Accept anyway if we can't validate
                        captured["token"] = token

        await page.send(nodriver_module.cdp.network.enable())
        page.add_handler(nodriver_module.cdp.network.RequestWillBeSent, on_request)

        print("  💡 Please log in to chat.deepseek.com in the browser window")
        print("  💡 Use Google/email login, then send any message")
        print(f"  ⏳ Waiting... (timeout: {timeout}s)")

        start = time.time()
        while time.time() - start < timeout:
            if captured["token"]:
                print("  🔑 Bearer token captured!")
                break

            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 15 == 0:
                print(f"  ⏳ [{elapsed}s] Waiting for login... ({timeout - elapsed}s left)")

            await asyncio.sleep(2)

        if not captured["token"]:
            raise RuntimeError(
                "Failed to capture auth token from chat.deepseek.com. "
                "Ensure you logged in and sent a message."
            )

        return WebChatAuth(
            provider=self.PROVIDER_NAME,
            token=captured["token"],
            account_label=self.account,
        )

    def _api_headers(self) -> dict[str, str]:
        """Build common API headers with auth."""
        auth = self.auth
        if not auth:
            raise RuntimeError(f"DeepSeek [{self.account}]: Not authenticated.")
        headers = {
            "Authorization": f"Bearer {auth.token}",
            "Content-Type": "application/json",
            **self._DS_HEADERS,
        }
        if auth.user_agent:
            headers["User-Agent"] = auth.user_agent
        return headers

    def _create_session(self) -> str:
        """Create a new chat session. Returns session_id."""
        from curl_cffi import requests as cf_requests

        r = cf_requests.post(
            f"{self._BASE_API}/chat_session/create",
            json={"character_id": None},
            headers=self._api_headers(),
            cookies=self.auth.cookies if self.auth else {},
            impersonate="chrome",
            timeout=15,
        )

        if r.status_code in (401, 403):
            raise PermissionError(f"DeepSeek auth rejected ({r.status_code}). Re-login needed.")
        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek session create error {r.status_code}: {r.text[:200]}")

        data: dict[str, Any] = r.json()
        biz_data: dict[str, Any] = data.get("data", {}).get("biz_data", data.get("data", {}))
        session_id: str = biz_data.get("id", "") or ""
        if not session_id:
            raise RuntimeError(f"No session_id in response: {r.text[:300]}")
        return session_id

    def _get_pow_challenge(self) -> dict[str, Any]:
        """Get PoW challenge from server. Returns challenge dict."""
        from curl_cffi import requests as cf_requests

        r = cf_requests.post(
            f"{self._BASE_API}/chat/create_pow_challenge",
            json={"target_path": "/api/v0/chat/completion"},
            headers=self._api_headers(),
            cookies=self.auth.cookies if self.auth else {},
            impersonate="chrome",
            timeout=15,
        )

        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek PoW challenge error {r.status_code}: {r.text[:200]}")

        data = r.json()
        challenge: dict[str, Any] = data.get("data", {}).get("biz_data", {}).get("challenge", {})
        if not challenge or not challenge.get("challenge"):
            raise RuntimeError(f"No challenge in PoW response: {r.text[:300]}")
        return challenge

    def _solve_pow(self, challenge: dict[str, Any]) -> dict[str, Any]:
        """Solve PoW challenge via Node.js subprocess. Returns {answer, ...}."""
        solver_path = str(self._POW_SOLVER_PATH)
        if not os.path.exists(solver_path):
            raise RuntimeError(
                f"PoW solver not found at {solver_path}. "
                f"deepseek_pow_solver.js must be alongside this file."
            )

        result = subprocess.run(
            [
                "node",
                solver_path,
                challenge["algorithm"],
                challenge["challenge"],
                challenge["salt"],
                str(challenge["difficulty"]),
                str(challenge["expire_at"]),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(f"PoW solver failed: {result.stderr}")

        solution: dict[str, Any] = json.loads(result.stdout.strip())
        if "error" in solution:
            raise RuntimeError(f"PoW solver error: {solution['error']}")

        return solution

    def _build_pow_header(self, challenge: dict, answer: int) -> str:
        """Build the x-ds-pow-response header value (base64-encoded JSON)."""
        pow_response = {
            "algorithm": challenge["algorithm"],
            "challenge": challenge["challenge"],
            "salt": challenge["salt"],
            "answer": answer,
            "signature": challenge["signature"],
            "target_path": challenge["target_path"],
        }
        return base64.b64encode(json.dumps(pow_response, separators=(",", ":")).encode()).decode()

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call DeepSeek V3.2 via chat.deepseek.com.

        Flow: create_session → get_pow_challenge → solve_pow → completion (SSE).
        """
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"DeepSeek [{self.account}]: Not authenticated. Run warmup() first.")

        model = model or self.DEFAULT_MODEL
        thinking_enabled = "thinking" in model.lower()

        # Extract user content
        user_content = ""
        system_prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_content = content

        if not user_content:
            raise ValueError("No user message found in messages list")

        if system_prompt:
            user_content = f"{system_prompt}\n\n{user_content}"

        # Step 1: Create session
        session_id = self._create_session()

        # Step 2: Get PoW challenge
        challenge = self._get_pow_challenge()

        # Step 3: Solve PoW
        solution = self._solve_pow(challenge)
        pow_header = self._build_pow_header(challenge, solution["answer"])

        # Step 4: Completion with SSE
        payload = {
            "chat_session_id": session_id,
            "prompt": user_content,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": False,
        }
        # NOTE: parent_message_id must be OMITTED entirely (not 0, not "", not null)

        headers = self._api_headers()
        headers["x-ds-pow-response"] = pow_header

        r = cf_requests.post(
            f"{self._BASE_API}/chat/completion",
            json=payload,
            headers=headers,
            cookies=auth.cookies,
            impersonate="chrome",
            timeout=timeout,
            stream=True,
        )

        if r.status_code in (401, 403):
            raise PermissionError(
                f"DeepSeek auth rejected ({r.status_code}). "
                f"Run --webchat-login deepseek to re-login."
            )

        if r.status_code != 200:
            raise RuntimeError(f"DeepSeek API error {r.status_code}: {r.text[:200]}")

        return self._parse_sse_stream(r, thinking_enabled)

    # Maximum response text size (512KB) to prevent OOM
    _MAX_RESPONSE_BYTES = 512 * 1024

    def _parse_sse_stream(self, response: Any, include_thinking: bool = False) -> str:
        """Parse DeepSeek's custom SSE stream into plain text.

        SSE format uses a JSON patch protocol:
          1. Full fragment: {"v":{"response":{"fragments":[{"content":"...","type":"RESPONSE"}]}}}
          2. Patch append:  {"p":"response/fragments/-1/content","o":"APPEND","v":"text"}
          3. Short append:  {"v":"text"}  (string v = append to current fragment)
          4. Batch patch:   {"p":"response","o":"BATCH","v":[...]}
          5. Status:        {"p":"response/status","o":"SET","v":"FINISHED"}
          6. Events:        event: ready|update_session|title|close
        """
        content_parts: list = []  # accumulated text tokens
        total_len = 0

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                # Skip event type lines
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                    if event_type == "close":
                        break
                    continue

                if not line.startswith("data:"):
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                v = chunk.get("v")
                p = chunk.get("p", "")
                o = chunk.get("o", "")

                # Case 1: Short append — {"v": "text"} (no "p" or "o" keys)
                # This is a compact form meaning "append to current fragment"
                if isinstance(v, str) and not p and not o:
                    content_parts.append(v)
                    total_len += len(v)
                    if total_len > self._MAX_RESPONSE_BYTES:
                        break
                    continue

                # Case 2: Full response with fragments
                if isinstance(v, dict) and "response" in v:
                    response_obj = v["response"]
                    if isinstance(response_obj, dict):
                        for frag in response_obj.get("fragments", []):
                            frag_type = frag.get("type", "")
                            content = frag.get("content", "")
                            if frag_type == "RESPONSE" and content:
                                content_parts.append(content)
                                total_len += len(content)

                # Case 3: Patch append — {"p":"response/fragments/-1/content","o":"APPEND","v":"text"}
                if o == "APPEND" and "content" in p and isinstance(v, str):
                    content_parts.append(v)
                    total_len += len(v)

                # Safety: abort if response is too large
                if total_len > self._MAX_RESPONSE_BYTES:
                    break

        finally:
            response.close()

        result = "".join(content_parts).strip()
        del content_parts  # free list memory immediately
        if not result:
            raise ValueError("Empty response from DeepSeek - no content in SSE stream")

        return result


# =============================================================================
# GitHub Copilot WebChat Provider (api.githubcopilot.com)
# =============================================================================


class CopilotWebChat(WebChatProvider):
    """
    GitHub Copilot via api.githubcopilot.com — OpenAI-compatible API.

    Auth:   GitHub OAuth device flow (no browser scraping needed).
            One-time: visit github.com/login/device + enter code.
            Token persists indefinitely (GitHub OAuth tokens don't expire).
    Token:  GitHub access_token → exchanged for short-lived Copilot API token
            (~30 min). Auto-refreshed transparently before each call.
    API:    Standard OpenAI chat/completions with SSE streaming.

    Supports GPT-5 series, Claude 4 series, Gemini 3, Grok, and more.
    Requires GitHub account with Copilot access (free tier or Pro).

    Protocol (reverse-engineered from VS Code Copilot extension):
      1. Device flow: POST github.com/login/device/code → user_code
      2. User visits github.com/login/device → enters code → authorizes
      3. Poll: POST github.com/login/oauth/access_token → access_token
      4. Exchange: GET api.github.com/copilot_internal/v2/token → copilot_token
      5. API: POST api.githubcopilot.com/chat/completions (OpenAI format)
    """

    PROVIDER_NAME = "copilot"
    URL = "https://github.com/copilot"
    DEFAULT_MODEL = "gpt-5"
    MODELS = [
        # GPT-5 Series
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.1",
        "gpt-5.2",
        # GPT-5 Codex
        "gpt-5-codex",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        # Claude 4 Series
        "claude-opus-4.6",
        "claude-opus-4.6-fast",
        "claude-opus-4.5",
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-haiku-4.5",
        # Gemini Series
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        # GPT-4 Series
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        # Grok
        "grok-code-fast-1",
    ]
    TOKEN_LIFETIME_SEC = 365 * 86400  # GitHub OAuth tokens don't expire

    # GitHub OAuth — VS Code Copilot extension client ID
    _CLIENT_ID = "Iv1.b507a08c87ecfe98"
    _COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
    _BASE_URL = "https://api.githubcopilot.com"
    _EDITOR_VERSION = "vscode/1.95.0"
    _EDITOR_PLUGIN_VERSION = "copilot/1.250.0"

    def __init__(self, account: str = "default"):
        super().__init__(account)
        self._copilot_token: str | None = None
        self._copilot_token_expires: float = 0

    # ── Auth: OAuth Device Flow ──────────────────────────────────────────

    async def browser_login(self, proxy: str | None = None, timeout: int = 300) -> WebChatAuth:
        """
        GitHub OAuth device flow — no browser scraping needed.

        The user visits github.com/login/device and enters a short code.
        After authorization, the access_token is saved to disk and persists
        indefinitely (GitHub OAuth tokens don't expire).
        """
        from curl_cffi import requests as cf_requests

        # Step 1: Request device authorization
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

        # Try opening browser automatically
        try:
            import webbrowser

            webbrowser.open(verification_uri)
            print("  (Browser opened automatically)")
        except Exception:
            pass

        # Step 2: Poll for access token
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

                # Validate: can we actually get a Copilot token?
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

    # ── Token Exchange ───────────────────────────────────────────────────

    def _exchange_copilot_token(self, github_token: str) -> str:
        """Exchange GitHub OAuth token for a short-lived Copilot API token.

        Called internally; caches the result for ~30 minutes.
        """
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

        # Parse expiry
        expires_at_str = data.get("expires_at", "")
        if expires_at_str:
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(expires_at_str.replace("Z", "+00:00"))
                self._copilot_token_expires = dt.timestamp()
            except Exception:
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

    # ── API Call ─────────────────────────────────────────────────────────

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call OpenAI-compatible Copilot API with SSE streaming.

        Supports all models available in GitHub Copilot:
        GPT-5 series, Claude 4 series, Gemini 3, Grok, etc.
        """
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
            # Copilot token may have expired — refresh and retry once
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

    # Maximum response text size (512KB) to prevent OOM
    _MAX_RESPONSE_BYTES = 512 * 1024

    def _parse_openai_sse(self, response: Any) -> str:
        """Parse standard OpenAI SSE streaming response.

        Format:
          data: {"choices":[{"delta":{"content":"..."}}]}
          data: [DONE]
        """
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


# =============================================================================
# Provider Registry & Utilities
# =============================================================================

# Global provider instances (lazy-initialized)
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
    # --- Try smart routing via AccountPool ---
    try:
        from .account_manager import PoolManager

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

                        # Detect hard rate limits and set cooldown
                        try:
                            from .account_manager import detect_hard_cooldown

                            hc = detect_hard_cooldown(str(e))
                            if hc:
                                cd_sec, cd_reason = hc
                                acct_obj = pool.get_account(acct_label)
                                if acct_obj:
                                    acct_obj.rate_tracker.set_hard_cooldown(cd_sec, cd_reason)
                        except Exception:
                            pass

                        # Try fallback via pool
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
                                except Exception:
                                    pass
                        raise  # re-raise original error
    except ImportError:
        pass  # account_manager not available, use simple fallback

    # --- Simple fallback (no pool or pool empty) ---
    # Try the requested account first
    provider = get_provider(provider_name, account)
    if provider.is_authenticated:
        return provider.call(messages, model=model, timeout=timeout)

    # Try other cached accounts
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


# =============================================================================
# CLI
# =============================================================================


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
