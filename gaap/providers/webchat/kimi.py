"""
Kimi K2.5 WebChat Provider (kimi.com)
=====================================

Kimi K2.5 Thinking via kimi.com web chat.

Protocol: Connect-RPC over HTTP (not REST).
Auth:     Browser login → capture JWT Bearer token.
Service:  kimi.gateway.chat.v1.ChatService/Chat (server-streaming).
Base URL: https://www.kimi.com/apiv2/
Framing:  Connect streaming envelope: [flags(1B), length(4B BE), payload].
"""

import asyncio
import base64
import json
import logging
import struct
import time
from typing import Any

from .base import WebChatAuth, WebChatProvider

logger = logging.getLogger("gaap.providers.webchat")


class KimiWebChat(WebChatProvider):
    """
    Kimi K2.5 Thinking via kimi.com web chat.

    Protocol: Connect-RPC over HTTP (not REST).
    Auth:     Browser login → capture JWT Bearer token.
    Service:  kimi.gateway.chat.v1.ChatService/Chat (server-streaming).
    Base URL: https://www.kimi.com/apiv2/
    Framing:  Connect streaming envelope: [flags(1B), length(4B BE), payload].
    """

    PROVIDER_NAME = "kimi"
    URL = "https://www.kimi.com"
    DEFAULT_MODEL = "kimi-k2.5-thinking"
    MODELS = ["kimi-k2.5-thinking", "kimi-k2", "kimi-k2-thinking", "kimi-research", "kimi"]
    TOKEN_LIFETIME_SEC = 2592000

    _CHAT_ENDPOINT = "/apiv2/kimi.gateway.chat.v1.ChatService/Chat"

    MODEL_SCENARIOS = {
        "kimi-k2.5-thinking": "SCENARIO_K2D5",
        "kimi-k2": "SCENARIO_K2",
        "kimi-k2-thinking": "SCENARIO_K2_THINKING",
        "kimi-research": "SCENARIO_RESEARCH",
        "kimi": "SCENARIO_K2D5",
    }

    _SCENARIO = "SCENARIO_K2D5"
    _MAX_RESPONSE_BYTES = 512 * 1024

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
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            return str(payload.get("sub", ""))
        except Exception as e:
            logger.debug(f"JWT parsing failed: {e}")
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
                except Exception as e:
                    logger.debug(f"localStorage read failed: {e}")

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
        except Exception as e:
            logger.debug(f"Token validation failed: {e}")

        expires_at = 0.0
        try:
            parts = captured["token"].split(".")
            payload_b64 = parts[1] + "=" * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64))
            exp = payload.get("exp", 0)
            if exp:
                expires_at = float(exp)
        except Exception as e:
            logger.debug(f"JWT expiry parsing failed: {e}")

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
        """Build a kimi.gateway.chat.v1.ChatRequest JSON payload."""
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
        """Call Kimi K2.5 via Connect-RPC (kimi.com/apiv2/)."""
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"Kimi [{self.account}]: Not authenticated. Run warmup() first.")

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

        payload = self._build_chat_request(user_content, model=model)

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

    def _parse_connect_response(self, content: bytes) -> str:
        """Parse Connect streaming envelopes and extract text content."""
        envelopes = self._parse_connect_stream(content)

        full_text = ""
        error_msg = ""

        for flags, msg in envelopes:
            if not isinstance(msg, dict):
                continue

            if flags == 0x02:
                err = msg.get("error", {})
                if err:
                    code = err.get("code", "unknown")
                    detail = err.get("message", "") or json.dumps(err)
                    error_msg = f"Kimi Connect-RPC error: {code} - {detail}"
                continue

            if "error" in msg:
                err = msg["error"]
                error_msg = f"Kimi error: {err}" if isinstance(err, str) else json.dumps(err)
                continue

            op = msg.get("op", "")

            block = msg.get("block")
            if block and isinstance(block, dict):
                text_block = block.get("text")
                if text_block and isinstance(text_block, dict):
                    text_content = text_block.get("content", "")
                    if text_content:
                        if op == "append":
                            full_text += text_content
                        else:
                            full_text = text_content

            if len(full_text) > self._MAX_RESPONSE_BYTES:
                break

        del envelopes
        result = full_text.strip()
        del full_text
        if not result:
            if error_msg:
                raise RuntimeError(error_msg)
            raise ValueError("Empty response from Kimi - no text content in stream")

        return result
