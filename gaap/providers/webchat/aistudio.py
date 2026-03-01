"""
Google AI Studio WebChat Provider (aistudio.google.com)
====================================================

Gemini models via aistudio.google.com web chat.

Protocol: HTTP/REST with SAPISIDHASH auth
Auth:     Browser login → capture cookies (SAPISID, SID, etc.) + generate SAPISIDHASH
Base URL: https://alkalimakersuite-pa.clients6.google.com
Models:   gemini-3.1-pro-preview, gemini-3-pro-preview, gemini-2.5-pro, etc.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

from .base import WebChatAuth, WebChatProvider, save_auth

logger = logging.getLogger("gaap.providers.webchat")

API_HOST = "alkalimakersuite-pa.clients6.google.com"
API_BASE = (
    f"https://{API_HOST}/$rpc/google.internal.alkali.applications.makersuite.v1.MakerSuiteService"
)


class MessageFlowCapture:
    """Container for captured message flow data."""

    def __init__(self):
        self.auth: WebChatAuth | None = None
        self.requests: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.api_endpoint: str = ""
        self.api_headers: dict[str, str] = {}
        self.api_payload: dict[str, Any] = {}


class AIStudioWebChat(WebChatProvider):
    """
    Google AI Studio via aistudio.google.com web chat.

    Auth: Browser login → capture cookies (SAPISID, SID, etc.) + generate SAPISIDHASH
    Models: Gemini 3.1 Pro, Gemini 3 Pro, Gemini 2.5 Pro, etc.
    """

    PROVIDER_NAME = "aistudio"
    URL = "https://aistudio.google.com"
    DEFAULT_MODEL = "gemini-3.1-pro-preview"
    MODELS = [
        "gemini-3.1-pro-preview",
        "gemini-3.1-flash-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    TOKEN_LIFETIME_SEC = 3600

    _MAX_RESPONSE_BYTES = 512 * 1024

    @staticmethod
    def _generate_sapisidhash(
        sapisid_cookie: str, origin: str = "https://aistudio.google.com"
    ) -> str:
        """Generate SAPISIDHASH for Google API authentication."""
        timestamp = int(time.time() * 1000)
        data = f"{timestamp} {sapisid_cookie} {origin}"
        hash_value = hashlib.sha1(data.encode()).hexdigest()
        return f"{timestamp}_{hash_value}"

    @staticmethod
    def _generate_auth_header(
        cookies: dict[str, str], origin: str = "https://aistudio.google.com"
    ) -> str:
        """Generate the Authorization header with SAPISIDHASH."""
        sapisid = cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID")
        if not sapisid:
            raise ValueError("SAPISID cookie not found")

        sapisidhash = AIStudioWebChat._generate_sapisidhash(sapisid, origin)
        return f"SAPISIDHASH {sapisidhash}"

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Capture auth from aistudio.google.com - cookies for SAPISIDHASH auth."""
        captured: dict[str, Any] = {"cookies": {}, "user_id": None, "api_key": None}

        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            headers = event.request.headers if hasattr(event.request, "headers") else {}

            if API_HOST in url:
                api_key = headers.get("X-Goog-Api-Key", "") or headers.get("x-goog-api-key", "")
                if api_key and api_key != "undefined":
                    captured["api_key"] = api_key

        try:
            await page.send(nodriver_module.cdp.network.enable())
            page.add_handler(nodriver_module.cdp.network.RequestWillBeSent, on_request)
        except Exception as e:
            logger.debug(f"CDP network monitoring failed: {e}")

        print("  💡 Please log in to aistudio.google.com in the browser window")
        print("  💡 After login, the cookies will be captured automatically")
        print(f"  ⏳ Waiting... (timeout: {timeout}s)")

        start = time.time()
        cookies_captured = False

        while time.time() - start < timeout:
            try:
                cookies = {}
                for c in await page.send(nodriver_module.cdp.network.get_cookies([self.URL])):
                    cookies[c.name] = c.value

                if cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID"):
                    captured["cookies"] = cookies
                    cookies_captured = True
                    break
            except Exception as e:
                logger.debug(f"Cookie capture error: {e}")

            elapsed = int(time.time() - start)
            if elapsed > 0 and elapsed % 15 == 0:
                print(f"  ⏳ [{elapsed}s] Waiting for login... ({timeout - elapsed}s left)")

            await asyncio.sleep(2)

        if not cookies_captured:
            raise RuntimeError(
                "Failed to capture auth cookies from aistudio.google.com. "
                "Please ensure you logged in successfully."
            )

        if captured.get("api_key"):
            captured["headers"] = {"X-Goog-Api-Key": captured["api_key"]}

        try:
            user_id = captured["cookies"].get("SID", "") or captured["cookies"].get(
                "__Secure-1PSID", ""
            )
        except Exception:
            user_id = ""

        expires_at = time.time() + self.TOKEN_LIFETIME_SEC

        return WebChatAuth(
            provider=self.PROVIDER_NAME,
            token=captured["cookies"].get("SAPISID", ""),
            user_id=user_id,
            account_label=self.account,
            cookies=captured["cookies"],
            headers=captured.get("headers", {}),
            expires_at=expires_at,
        )

    def _build_chat_request(
        self, user_content: str, model: str = "", system_prompt: str = ""
    ) -> dict[str, Any]:
        """Build the chat request payload for MakerSuite API."""
        model = model or self.DEFAULT_MODEL

        return {
            "model": model,
            "messages": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": 0.9,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
            },
        }

    def call(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        timeout: int = 120,
        stream: bool = False,
    ) -> str:
        """Call Google AI Studio via MakerSuite API with SAPISIDHASH auth."""

        auth = self.auth
        if not auth:
            raise RuntimeError(
                f"AI Studio [{self.account}]: Not authenticated. Run warmup() first."
            )

        if not auth.cookies:
            raise RuntimeError(
                f"AI Studio [{self.account}]: No cookies available. Run warmup() to capture cookies."
            )

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

        origin = "https://aistudio.google.com"
        auth_header = self._generate_auth_header(auth.cookies, origin)

        captured_api_key = auth.headers.get("X-Goog-Api-Key", "")
        env_api_key = os.getenv("GEMINI_API_KEY", "")
        api_key = captured_api_key or env_api_key

        if not api_key:
            raise RuntimeError(
                "No API key available. Please set the GEMINI_API_KEY environment variable, "
                "or re-authenticate to capture an API key from the browser session."
            )

        payload = self._build_chat_request(user_content, model=model)

        headers_base = {
            "Authorization": auth_header,
            "Content-Type": "application/json+protobuf",
            "Referer": f"{origin}/",
            "Origin": origin,
        }

        if auth.user_agent:
            headers_base["User-Agent"] = auth.user_agent

        endpoint = f"{API_BASE}/StreamGenerateContent"

        attempts = []

        if captured_api_key:
            attempts.append(
                {
                    "name": "with_captured_api_key",
                    "url": f"{endpoint}?alt=json&key={captured_api_key}",
                    "api_key": captured_api_key,
                }
            )

        attempts.append(
            {
                "name": "without_api_key",
                "url": f"{endpoint}?alt=json",
                "api_key": None,
            }
        )

        if env_api_key:
            attempts.append(
                {
                    "name": "with_env_api_key",
                    "url": f"{endpoint}?alt=json&key={env_api_key}",
                    "api_key": env_api_key,
                }
            )

        last_error = None

        for attempt in attempts:
            try:
                headers = headers_base.copy()
                headers["X-Goog-Api-Key"] = attempt["api_key"] or ""

                url = attempt["url"]
                name = attempt["name"]

                logger.debug(f"Trying {name}: {url}")

                if stream:
                    result = self._stream_request(url, payload, headers, auth.cookies, timeout)
                else:
                    result = self._make_request(url, payload, headers, auth.cookies, timeout)

                if result and result.strip():
                    logger.debug(f"Success with {name}")
                    return result

            except Exception as e:
                logger.debug(f"Attempt {attempt['name']} failed: {e}")
                last_error = e
                continue

        try:
            return self._call_official_api(user_content, model, timeout, system_prompt)
        except Exception as e:
            raise RuntimeError(
                f"AI Studio API call failed after {len(attempts)} attempts. "
                f"Last error: {last_error}. "
                f"Official API fallback also failed: {e}. "
                "The session may have expired. Run warmup() to re-authenticate."
            )

    def _call_official_api(
        self, content: str, model: str, timeout: int, system_prompt: str = ""
    ) -> str:
        """Fallback to official Google Gemini API."""
        import requests

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Please set it to use the official Gemini API fallback."
            )

        model = model or "gemini-1.5-pro"

        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={api_key}"

        payload: dict[str, Any] = {
            "contents": [{"parts": [{"text": content}]}],
            "generationConfig": {
                "temperature": 0.9,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": 8192,
            },
        }

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code != 200:
            raise RuntimeError(f"Official API error: {response.status_code} - {response.text}")

        data = response.json()

        if "candidates" in data and len(data["candidates"]) > 0:
            candidate = data["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if len(parts) > 0 and "text" in parts[0]:
                    return parts[0]["text"]

        return str(data)

    def _make_request(
        self,
        url: str,
        payload: dict,
        headers: dict,
        cookies: dict,
        timeout: int,
    ) -> str:
        """Make a non-streaming request to the API."""
        from curl_cffi import requests as cf_requests

        response = cf_requests.post(
            url,
            json=payload,
            headers=headers,
            cookies=cookies,
            impersonate="chrome",
            timeout=timeout,
        )

        if response.status_code != 200:
            raise RuntimeError(f"API returned status {response.status_code}: {response.text[:500]}")

        return self._parse_response(response.text)

    def _stream_request(
        self,
        url: str,
        payload: dict,
        headers: dict,
        cookies: dict,
        timeout: int,
    ) -> str:
        """Make a streaming request to the API."""
        from curl_cffi import requests as cf_requests

        response = cf_requests.post(
            url,
            json=payload,
            headers=headers,
            cookies=cookies,
            impersonate="chrome",
            timeout=timeout,
            stream=True,
        )

        if response.status_code != 200:
            raise RuntimeError(f"API returned status {response.status_code}")

        full_response = []
        for line in response.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8")
                if decoded.startswith("data:"):
                    data_str = decoded[5:].strip()
                    parsed = json.loads(data_str)
                    text = self._extract_text_from_stream_response(parsed)
                    if text:
                        full_response.append(text)
            except Exception as e:
                logger.debug(f"Stream parse error: {e}")
                continue

        return "".join(full_response)

    def _extract_text_from_stream_response(self, data: Any) -> str:
        """Extract text from the streaming response format."""
        try:
            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        for part in item:
                            if isinstance(part, list):
                                for p in part:
                                    if isinstance(p, list) and len(p) >= 2 and p[0] is None:
                                        return p[1] if isinstance(p[1], str) else ""
            elif isinstance(data, dict):
                candidates = data.get("candidates", [])
                for candidate in candidates:
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        if "text" in part:
                            return part["text"]
        except Exception as e:
            logger.debug(f"Text extraction error: {e}")
        return ""

    def _parse_response(self, response_text: str) -> str:
        """Parse the API response and extract text."""
        try:
            data = json.loads(response_text)

            if isinstance(data, list) and len(data) > 0:
                for item in data:
                    if isinstance(item, list) and len(item) >= 2:
                        for part in item:
                            if isinstance(part, list):
                                for p in part:
                                    if isinstance(p, list) and len(p) >= 2 and p[0] is None:
                                        return p[1] if isinstance(p[1], str) else ""

            candidates = data.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                texts = []
                for part in parts:
                    if "text" in part:
                        texts.append(part["text"])
                if texts:
                    return "".join(texts)

        except json.JSONDecodeError as e:
            logger.debug(f"Response parse error: {e}")

        return response_text

    async def capture_message_flow(
        self, proxy: str | None = None, timeout: int = 300
    ) -> MessageFlowCapture:
        """
        Open browser, wait for login, keep open, then capture message flow.

        This method:
        1. Opens the browser
        2. Waits for user to login
        3. Keeps browser open after login
        4. Prompts user to send a message
        5. Monitors and captures the API requests when user sends message
        6. Extracts endpoint, headers, and payload from those requests
        7. Returns the captured auth AND API details

        Returns:
            MessageFlowCapture object containing auth and API details
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

        result = MessageFlowCapture()

        print(f"  Opening {self.URL} for login...")
        print(f"  Please log in. Timeout: {timeout}s")

        browser_profile = f"gaap_{self.PROVIDER_NAME}_{self.account}"
        browser, stop_browser = await get_nodriver(
            proxy=proxy or "", timeout=timeout, user_data_dir=browser_profile
        )

        captured_requests: list[dict[str, Any]] = []
        captured_responses: list[dict[str, Any]] = []

        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            if API_HOST in url and "/StreamGenerateContent" in url:
                try:
                    headers = (
                        dict(event.request.headers) if hasattr(event.request, "headers") else {}
                    )
                    post_data = None
                    if hasattr(event.request, "post_data") and event.request.post_data:
                        if isinstance(event.request.post_data, bytes):
                            post_data = event.request.post_data.decode("utf-8")
                        else:
                            post_data = str(event.request.post_data)

                    request_data = {
                        "url": url,
                        "method": "POST",
                        "headers": headers,
                        "post_data": post_data,
                    }
                    captured_requests.append(request_data)
                    logger.debug(f"Captured request: {url}")
                except Exception as e:
                    logger.debug(f"Request capture error: {e}")

        async def on_response(event: Any) -> None:
            url = event.response.url if hasattr(event.response, "url") else ""
            if API_HOST in url and "/StreamGenerateContent" in url:
                try:
                    headers = (
                        dict(event.response.headers) if hasattr(event.response, "headers") else {}
                    )
                    status = event.response.status if hasattr(event.response, "status") else 0

                    response_data = {
                        "url": url,
                        "status": status,
                        "headers": headers,
                    }
                    captured_responses.append(response_data)
                    logger.debug(f"Captured response: {url} - {status}")
                except Exception as e:
                    logger.debug(f"Response capture error: {e}")

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

            await page.send(nodriver.cdp.network.enable())
            page.add_handler(nodriver.cdp.network.RequestWillBeSent, on_request)
            page.add_handler(nodriver.cdp.network.ResponseReceived, on_response)

            print("  💡 Please log in to aistudio.google.com in the browser window")
            print("  💡 After login, the cookies will be captured automatically")
            print(f"  ⏳ Waiting for login... (timeout: {timeout}s)")

            start = time.time()
            cookies_captured = False

            while time.time() - start < timeout:
                try:
                    cookies = {}
                    for c in await page.send(nodriver.cdp.network.get_cookies([self.URL])):
                        cookies[c.name] = c.value

                    if cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID"):
                        cookies_captured = True
                        break
                except Exception as e:
                    logger.debug(f"Cookie capture error: {e}")

                elapsed = int(time.time() - start)
                if elapsed > 0 and elapsed % 15 == 0:
                    print(f"  ⏳ [{elapsed}s] Waiting for login... ({timeout - elapsed}s left)")

                await asyncio.sleep(2)

            if not cookies_captured:
                await stop_browser()
                raise RuntimeError(
                    "Failed to capture auth cookies from aistudio.google.com. "
                    "Please ensure you logged in successfully."
                )

            print("  ✅ Login detected! Cookies captured.")
            print("")
            print("  🔵 Browser will stay OPEN. Please send a message in the chat.")
            print("  ⏳ The request will be captured automatically...")
            print("  💡 When you're done, the browser will close automatically.")
            print("")

            message_sent = False
            request_captured = False

            while time.time() - start < timeout:
                if len(captured_requests) > 0:
                    request_captured = True
                    print(f"  ✅ API request captured! ({len(captured_requests)} request(s))")
                    break

                await asyncio.sleep(1)

            if not request_captured:
                print("  ⚠️ No API request detected yet. Continuing to wait...")

                wait_start = time.time()
                while time.time() - wait_start < 60:
                    if len(captured_requests) > 0:
                        request_captured = True
                        print(f"  ✅ API request captured! ({len(captured_requests)} request(s))")
                        break
                    await asyncio.sleep(1)

            cookies = {}
            for c in await page.send(nodriver.cdp.network.get_cookies([self.URL])):
                cookies[c.name] = c.value

            api_key = ""
            if captured_requests:
                req = captured_requests[0]
                api_key = req.get("headers", {}).get("X-Goog-Api-Key", "") or req.get(
                    "headers", {}
                ).get("x-goog-api-key", "")

            sapisid = cookies.get("SAPISID") or cookies.get("__Secure-1PAPISID", "")
            user_id = cookies.get("SID", "") or cookies.get("__Secure-1PSID", "")

            expires_at = time.time() + self.TOKEN_LIFETIME_SEC

            result.auth = WebChatAuth(
                provider=self.PROVIDER_NAME,
                token=sapisid,
                user_id=user_id,
                account_label=self.account,
                cookies=cookies,
                headers={"X-Goog-Api-Key": api_key} if api_key else {},
                expires_at=expires_at,
                user_agent=user_agent,
                captured_at=time.time(),
            )

            save_auth(result.auth)
            self._auth = result.auth

            result.requests = captured_requests
            result.responses = captured_responses

            if captured_requests:
                req = captured_requests[0]
                result.api_endpoint = req.get("url", "")
                result.api_headers = req.get("headers", {})

                post_data = req.get("post_data", "")
                if post_data:
                    try:
                        result.api_payload = json.loads(post_data)
                    except json.JSONDecodeError:
                        result.api_payload = {"raw": post_data}

            print("")
            print("  📡 Capture Summary:")
            print(f"     - Requests captured: {len(result.requests)}")
            print(f"     - API endpoint: {result.api_endpoint[:100]}...")
            if result.api_payload:
                print(f"     - Payload keys: {list(result.api_payload.keys())}")
            print(f"     - Auth saved: ✅")

            return result

        finally:
            print("  👋 Closing browser...")
            await stop_browser()

    def capture_message_flow_sync(
        self, proxy: str | None = None, timeout: int = 300
    ) -> MessageFlowCapture:
        """Synchronous wrapper for capture_message_flow()."""
        return asyncio.run(self.capture_message_flow(proxy=proxy, timeout=timeout))
