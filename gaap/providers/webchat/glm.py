"""
GLM-5 WebChat Provider (chat.z.ai)
===================================

GLM-5 / GLM-4.7 via chat.z.ai (Open WebUI based).

Auth: Browser login → capture JWT from /api/v1/auths/ or intercepted headers.
API:  POST /api/chat/completions with HMAC signature.
SSE:  type=chat:completion with delta_content/edit_content.
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import threading
import time
import urllib.parse
import uuid
from typing import Any

from .base import WebChatAuth, WebChatProvider

logger = logging.getLogger("gaap.providers.webchat")


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
    TOKEN_LIFETIME_SEC = 86400

    MODEL_ALIASES = {
        "GLM-5": "glm-5",
        "GLM-4.7": "glm-4.7",
        "GLM-4.6": "GLM-4-6-API-V1",
        "GLM-4.5": "GLM-4.5",
        "GLM-4.5V": "glm-4.5v",
        "Z1-Rumination": "Z1-Rumination",
    }

    _MAX_RESPONSE_BYTES = 512 * 1024

    # Model detection cache settings
    _MODEL_CACHE_TTL = 300  # 5 minutes

    def __init__(self, account: str = "default") -> None:
        super().__init__(account)
        self._actual_model: str | None = None
        self._last_api_response: dict[str, Any] | None = None
        self._model_detected_at: float = 0
        self._model_lock = threading.Lock()

    def _is_model_cache_valid(self) -> bool:
        """Check if the cached model is still valid (within TTL)."""
        if not self._actual_model:
            return False
        return (time.time() - self._model_detected_at) < self._MODEL_CACHE_TTL

    def _extract_model_from_response(
        self, response: Any, response_data: dict[str, Any] | None = None
    ) -> str | None:
        """
        Extract the actual model from API response.
        Checks response headers, JSON body, and SSE data.
        """
        model: str | None = None

        # Try response headers first
        if hasattr(response, "headers"):
            headers = response.headers
            for header_name in ["X-Model-Used", "X-Model", "X-Glm-Model", "Model"]:
                if header_name in headers:
                    model = headers[header_name]
                    break

        # Try response body/model field
        if not model and response_data:
            if isinstance(response_data, dict):
                model = response_data.get("model") or response_data.get("model_id")

                # Try nested in choices
                if (
                    not model
                    and "choices" in response_data
                    and isinstance(response_data["choices"], list)
                ):
                    for choice in response_data["choices"]:
                        if isinstance(choice, dict) and "model" in choice:
                            model = choice["model"]
                            break

        # Normalize and validate detected model
        if model:
            model = self._normalize_model_name(model)

        return model

    def _normalize_model_name(self, model: str) -> str | None:
        """
        Normalize model name to internal naming convention.
        Returns None if model cannot be identified.
        """
        if not model:
            return None

        model_lower = model.lower().replace("-", "").replace("_", "").replace(" ", "")

        # Known model mappings
        model_mappings = {
            "glm5": "GLM-5",
            "glm4.7": "GLM-4.7",
            "glm47": "GLM-4.7",
            "glm4.6": "GLM-4.6",
            "glm46": "GLM-4.6",
            "glm4.5": "GLM-4.5",
            "glm45": "GLM-4.5",
            "glm4.5v": "GLM-4.5V",
            "glm45v": "GLM-4.5V",
            "glm4plus": "GLM-4-plus",
            "glm-4-plus": "GLM-4-plus",
            "z1rumination": "Z1-Rumination",
            "z1": "Z1-Rumination",
        }

        # Check direct mapping
        if model_lower in model_mappings:
            return model_mappings[model_lower]

        # Check if it contains known model strings
        for key, normalized in model_mappings.items():
            if key in model_lower:
                return normalized

        # Fallback: if it starts with glm-, return as-is with GLM- prefix
        if model_lower.startswith("glm"):
            return model.upper() if model.startswith("GLM") else f"GLM-{model[3:].upper()}"

        # Unknown model - return as generic GLM model
        logger.debug(f"Unknown model detected: {model}, using as-is")
        return model

    def _update_detected_model(self, model: str | None) -> None:
        """Update the detected model with thread-safe locking."""
        if not model:
            return

        with self._model_lock:
            self._actual_model = model
            self._model_detected_at = time.time()
            logger.debug(f"Detected model updated: {model}")

    def get_actual_model(self) -> str:
        """
        Get the actual model being used by the API.
        Returns cached model if valid, otherwise falls back to DEFAULT_MODEL.
        """
        with self._model_lock:
            if self._is_model_cache_valid():
                return self._actual_model or self.DEFAULT_MODEL
            return self.DEFAULT_MODEL

    def get_provider_info(self) -> dict[str, Any]:
        """Return full provider status including detected model information."""
        with self._model_lock:
            return {
                "provider": self.PROVIDER_NAME,
                "account": self.account,
                "default_model": self.DEFAULT_MODEL,
                "actual_model": self._actual_model,
                "model_cache_valid": self._is_model_cache_valid(),
                "model_detected_at": self._model_detected_at,
                "supported_models": self.MODELS,
                "is_authenticated": self.auth is not None,
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
        except Exception as e:
            logger.debug(f"Auth validation failed: {e}")
        return {"valid": False, "role": "unknown", "user_id": "", "token": token}

    async def _capture_auth(self, page: Any, nodriver_module: Any, timeout: int) -> WebChatAuth:
        """Capture auth token from chat.z.ai logged-in session."""
        captured: dict[str, Any] = {"token": None, "user_id": None}

        async def on_request(event: Any) -> None:
            url = event.request.url if hasattr(event.request, "url") else ""
            headers = event.request.headers if hasattr(event.request, "headers") else {}
            if "chat.z.ai/api" in url:
                auth_header = headers.get("Authorization", "") or headers.get("authorization", "")
                if auth_header.startswith("Bearer ") and len(auth_header) > 20:
                    token = auth_header[7:]
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
            except Exception as e:
                logger.debug(f"Auth capture iteration failed: {e}")

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
        """HMAC-SHA256 signature for chat.z.ai anti-bot."""
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

        user_prompt = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_prompt = msg.get("content", "")
                break

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

        # Store response reference for model detection
        self._last_api_response = {
            "headers": dict(r.headers) if hasattr(r, "headers") else {},
            "status_code": r.status_code,
        }

        # Try to extract model from headers immediately
        detected_model = self._extract_model_from_response(r)
        if detected_model:
            self._update_detected_model(detected_model)

        return self._parse_sse_stream(r)

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

                        # Try to extract model from SSE data
                        detected_model = self._extract_model_from_response(None, chunk_data)
                        if detected_model:
                            self._update_detected_model(detected_model)

                        if chunk_data.get("type") == "chat:completion":
                            data = chunk_data.get("data", {})
                            phase = data.get("phase", "")
                            if phase == "thinking":
                                continue
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
                        if total_len > self._MAX_RESPONSE_BYTES:
                            break
                    except json.JSONDecodeError:
                        continue
        finally:
            response.close()

        result = "".join(result_parts).strip()
        del result_parts
        if not result:
            raise ValueError("Empty response from GLM - no content in SSE stream")
        return result
