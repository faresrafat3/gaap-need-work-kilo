"""
DeepSeek WebChat Provider (chat.deepseek.com)
=============================================

DeepSeek V3.2 / R2 via chat.deepseek.com.

Protocol: REST + custom SSE (text/event-stream).
Auth:     Browser login (Google OAuth) → capture opaque Bearer token.
PoW:      DeepSeekHashV1 — custom 23-round Keccak (NOT SHA3-256).
          Solved via Node.js subprocess (deepseek_pow_solver.js).
"""

import asyncio
import base64
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import WebChatAuth, WebChatProvider, save_auth

logger = logging.getLogger("gaap.providers.webchat")


class DeepSeekWebChat(WebChatProvider):
    """
    DeepSeek V3.2 / R2 via chat.deepseek.com.

    Protocol: REST + custom SSE (text/event-stream).
    Auth:     Browser login (Google OAuth) → capture opaque Bearer token.
    PoW:      DeepSeekHashV1 — custom 23-round Keccak (NOT SHA3-256).
              Solved via Node.js subprocess (deepseek_pow_solver.js).
    """

    PROVIDER_NAME = "deepseek"
    URL = "https://chat.deepseek.com"
    DEFAULT_MODEL = "deepseek"
    MODELS = ["deepseek", "deepseek-thinking"]
    TOKEN_LIFETIME_SEC = 2592000

    _BASE_API = "https://chat.deepseek.com/api/v0"
    _POW_SOLVER_PATH = Path(__file__).parent.parent / "deepseek_pow_solver.js"

    _DS_HEADERS = {
        "x-client-locale": "en_US",
        "x-app-version": "20241129.1",
        "x-client-version": "1.7.0",
        "x-client-platform": "web",
        "Origin": "https://chat.deepseek.com",
        "Referer": "https://chat.deepseek.com/",
    }

    _MAX_RESPONSE_BYTES = 512 * 1024

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
                    except Exception as e:
                        logger.debug(f"Token validation failed: {e}")
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
        """Call DeepSeek V3.2 via chat.deepseek.com."""
        from curl_cffi import requests as cf_requests

        auth = self.auth
        if not auth:
            raise RuntimeError(f"DeepSeek [{self.account}]: Not authenticated. Run warmup() first.")

        model = model or self.DEFAULT_MODEL
        thinking_enabled = "thinking" in model.lower()

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

        session_id = self._create_session()

        challenge = self._get_pow_challenge()

        solution = self._solve_pow(challenge)
        pow_header = self._build_pow_header(challenge, solution["answer"])

        payload = {
            "chat_session_id": session_id,
            "prompt": user_content,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": False,
        }

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

    def _parse_sse_stream(self, response: Any, include_thinking: bool = False) -> str:
        """Parse DeepSeek's custom SSE stream into plain text."""
        content_parts: list = []
        total_len = 0

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

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

                if isinstance(v, str) and not p and not o:
                    content_parts.append(v)
                    total_len += len(v)
                    if total_len > self._MAX_RESPONSE_BYTES:
                        break
                    continue

                if isinstance(v, dict) and "response" in v:
                    response_obj = v["response"]
                    if isinstance(response_obj, dict):
                        for frag in response_obj.get("fragments", []):
                            frag_type = frag.get("type", "")
                            content = frag.get("content", "")
                            if frag_type == "RESPONSE" and content:
                                content_parts.append(content)
                                total_len += len(content)

                if o == "APPEND" and "content" in p and isinstance(v, str):
                    content_parts.append(v)
                    total_len += len(v)

                if total_len > self._MAX_RESPONSE_BYTES:
                    break

        finally:
            response.close()

        result = "".join(content_parts).strip()
        del content_parts
        if not result:
            raise ValueError("Empty response from DeepSeek - no content in SSE stream")

        return result
