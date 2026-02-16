# mypy: ignore-errors
# Groq Provider
import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterator
from datetime import datetime
from typing import Any

import aiohttp

from gaap.core.exceptions import (
    ProviderAuthenticationError,
    ProviderNotAvailableError,
    ProviderRateLimitError,
    ProviderResponseError,
)
from gaap.core.types import (
    Message,
    MessageRole,
    ModelTier,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider, get_logger, register_provider

# =============================================================================
# Groq Configuration
# =============================================================================

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.3-70b-specdec",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama-3.2-1b-preview",
    "llama-3.2-3b-preview",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# تكاليف Groq (لكل 1M tokens)
GROQ_COSTS = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.3-70b-specdec": {"input": 0.59, "output": 0.99},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "llama-3.2-1b-preview": {"input": 0.04, "output": 0.04},
    "llama-3.2-3b-preview": {"input": 0.06, "output": 0.06},
    "llama-3.2-11b-vision-preview": {"input": 0.18, "output": 0.18},
    "llama-3.2-90b-vision-preview": {"input": 0.90, "output": 0.90},
    "mixtral-8x7b-32768": {"input": 0.27, "output": 0.27},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},
}

# حدود الطبقة المجانية
FREE_TIER_LIMITS = {
    "requests_per_minute": 30,
    "requests_per_day": 14400,
    "tokens_per_minute": 18000,
    "tokens_per_day": 500000,
}

# تعيين الأسماء المستعارة
MODEL_ALIASES = {
    "llama-70b": "llama-3.3-70b-versatile",
    "llama-8b": "llama-3.1-8b-instant",
    "mixtral": "mixtral-8x7b-32768",
    "gemma": "gemma2-9b-it",
}


# =============================================================================
# Groq Provider Implementation
# =============================================================================


@register_provider("groq")
class GroqProvider(BaseProvider):
    """
    مزود Groq - سرعة فائقة مع طبقة مجانية

    الميزات:
    - سرعة استجابة فائقة
    - طبقة مجانية سخية
    - متوافق مع OpenAI API
    - نماذج Llama و Mixtral
    """

    def __init__(
        self, api_key: str | None = None, default_model: str = "llama-3.1-8b-instant", **kwargs: Any
    ) -> None:
        # البحث عن API key من البيئة
        if api_key is None:
            import os

            api_key = os.environ.get("GROQ_API_KEY")

        super().__init__(
            name="groq",
            provider_type=ProviderType.FREE_TIER,
            models=GROQ_MODELS,
            api_key=api_key,
            base_url=GROQ_API_URL,
            rate_limit_rpm=FREE_TIER_LIMITS["requests_per_minute"],
            rate_limit_tpm=FREE_TIER_LIMITS["tokens_per_minute"],
            timeout=60.0,
            max_retries=3,
            default_model=default_model,
        )

        self._session: aiohttp.ClientSession | None = None
        self._logger = get_logger("gaap.provider.groq")

        # تتبع الاستخدام اليومي
        self._daily_requests = 0
        self._daily_tokens = 0
        self._last_reset: datetime | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """الحصول على جلسة HTTP"""
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    def _check_daily_limits(self) -> bool:
        """التحقق من الحدود اليومية"""
        now = datetime.now()

        # إعادة تعيين العداد كل يوم
        if self._last_reset is None or (now - self._last_reset).days >= 1:
            self._daily_requests = 0
            self._daily_tokens = 0
            self._last_reset = now

        return (
            self._daily_requests < FREE_TIER_LIMITS["requests_per_day"]
            and self._daily_tokens < FREE_TIER_LIMITS["tokens_per_day"]
        )

    async def _make_request(self, messages: list[Message], model: str, **kwargs) -> dict[str, Any]:
        """تنفيذ الطلب لـ Groq API"""

        # التحقق من الحدود اليومية
        if not self._check_daily_limits():
            raise ProviderRateLimitError(provider_name=self.name, retry_after=86400)  # غداً

        # حل الأسماء المستعارة
        actual_model = MODEL_ALIASES.get(model, model)

        # تحويل الرسائل
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        # بناء الطلب
        payload = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "top_p": kwargs.get("top_p", 1.0),
        }

        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]

        # تنفيذ الطلب
        session = await self._get_session()

        try:
            async with session.post(
                GROQ_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status == 401:
                    raise ProviderAuthenticationError(provider_name=self.name)

                if response.status == 429:
                    retry_after = int(response.headers.get("retry-after", 60))
                    raise ProviderRateLimitError(provider_name=self.name, retry_after=retry_after)

                if response.status != 200:
                    error_body = await response.text()
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                data = await response.json()

                # تحديث الاستخدام اليومي
                usage = data.get("usage", {})
                self._daily_requests += 1
                self._daily_tokens += usage.get("total_tokens", 0)

                return data

        except aiohttp.ClientError as e:
            raise ProviderNotAvailableError(provider_name=self.name, reason=str(e))

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs
    ) -> AsyncIterator[str]:
        """تدفق الاستجابة من Groq"""

        actual_model = MODEL_ALIASES.get(model, model)

        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        payload = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": True,
        }

        session = await self._get_session()

        try:
            async with session.post(
                GROQ_API_URL, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                if response.status != 200:
                    error_body = await response.text()
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                async for line in response.content:
                    line = line.decode("utf-8").strip()

                    if not line or line == "data: [DONE]":
                        continue

                    if line.startswith("data: "):
                        json_str = line[6:]
                        try:
                            chunk = json.loads(json_str)
                            content = (
                                chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            )
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except aiohttp.ClientError as e:
            raise ProviderNotAvailableError(provider_name=self.name, reason=str(e))

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """حساب التكلفة"""
        actual_model = MODEL_ALIASES.get(model, model)
        costs = GROQ_COSTS.get(actual_model, {"input": 0.0, "output": 0.0})

        return (input_tokens * costs["input"] / 1_000_000) + (
            output_tokens * costs["output"] / 1_000_000
        )

    def get_daily_usage(self) -> dict[str, Any]:
        """الحصول على الاستخدام اليومي"""
        return {
            "requests": self._daily_requests,
            "tokens": self._daily_tokens,
            "requests_limit": FREE_TIER_LIMITS["requests_per_day"],
            "tokens_limit": FREE_TIER_LIMITS["tokens_per_day"],
            "requests_remaining": FREE_TIER_LIMITS["requests_per_day"] - self._daily_requests,
            "tokens_remaining": FREE_TIER_LIMITS["tokens_per_day"] - self._daily_tokens,
        }

    def get_model_info(self, model: str) -> dict[str, Any]:
        """معلومات عن نموذج معين"""
        actual_model = MODEL_ALIASES.get(model, model)
        costs = GROQ_COSTS.get(actual_model, {})

        return {
            "name": actual_model,
            "provider": "groq",
            "cost_per_1m_input": costs.get("input", 0),
            "cost_per_1m_output": costs.get("output", 0),
            "available": actual_model in GROQ_MODELS,
        }

    async def close(self) -> None:
        """إغلاق الجلسة"""
        if self._session and not self._session.closed:
            await self._session.close()

    def shutdown(self) -> None:
        """إيقاف المزود"""
        asyncio.create_task(self.close())
        super().shutdown()

    def shutdown(self) -> None:
        """إيقاف المزود"""
        asyncio.create_task(self.close())
        super().shutdown()


# =============================================================================
# Convenience Functions
# =============================================================================


def create_groq_provider(
    api_key: str | None = None, default_model: str = "llama-3.1-8b-instant"
) -> GroqProvider:
    """إنشاء مزود Groq بسهولة"""
    return GroqProvider(api_key=api_key, default_model=default_model)


def list_groq_models() -> list[str]:
    """قائمة نماذج Groq"""
    return GROQ_MODELS.copy()


def get_groq_free_limits() -> dict[str, int]:
    """حدود الطبقة المجانية"""
    return FREE_TIER_LIMITS.copy()


# =============================================================================
# Gemini Provider (Free Tier)
# =============================================================================


@register_provider("gemini")
class GeminiProvider(BaseProvider):
    """
    مزود Google Gemini - طبقة مجانية

    الميزات:
    - نافذة سياق ضخمة (1M tokens)
    - طبقة مجانية سخية
    - نماذج متعددة (Flash, Pro)
    """

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    GEMINI_MODELS = [
        "gemini-2.5-flash",
    ]

    GEMINI_COSTS = {
        "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
        "gemini-2.5-pro": {"input": 0.0, "output": 0.0},
        "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-flash-8b": {"input": 0.0375, "output": 0.15},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    }

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        default_model: str = "gemini-2.5-flash",
        **kwargs,
    ):
        env_keys_raw = os.environ.get("GEMINI_API_KEYS", "")
        env_keys = [k.strip() for k in env_keys_raw.split(",") if k.strip()]

        keys: list[str] = []
        if api_keys:
            keys.extend([k.strip() for k in api_keys if k and k.strip()])

        if api_key:
            keys.insert(0, api_key.strip())
        else:
            single_env_key = os.environ.get("GEMINI_API_KEY")
            if single_env_key:
                keys.insert(0, single_env_key.strip())

        keys.extend(env_keys)

        # إزالة التكرار مع الحفاظ على الترتيب
        deduped_keys: list[str] = []
        seen = set()
        for key in keys:
            if key not in seen:
                deduped_keys.append(key)
                seen.add(key)

        self._api_keys: list[str] = deduped_keys
        self._api_key_index: int = 0

        if self._api_keys:
            api_key = self._api_keys[0]
        else:
            api_key = None

        super().__init__(
            name="gemini",
            provider_type=ProviderType.FREE_TIER,
            models=self.GEMINI_MODELS,
            api_key=api_key,
            rate_limit_rpm=15,
            rate_limit_tpm=1_000_000,  # 1M context!
            timeout=120.0,
            max_retries=3,
            default_model=default_model,
        )

        self._session: aiohttp.ClientSession | None = None
        self._logger = get_logger("gaap.provider.gemini")
        self._key_cooldown_until: dict[str, float] = {}

    def _current_api_key(self) -> str | None:
        """الحصول على المفتاح الحالي"""
        if not self._api_keys:
            return self.api_key
        return self._api_keys[self._api_key_index]

    def _is_key_available(self, key: str) -> bool:
        """هل المفتاح متاح (غير تحت التبريد)؟"""
        cooldown_until = self._key_cooldown_until.get(key, 0.0)
        return time.time() >= cooldown_until

    def _set_key_cooldown(self, key: str, seconds: float) -> None:
        """تفعيل تبريد مؤقت لمفتاح"""
        self._key_cooldown_until[key] = time.time() + max(seconds, 0.0)

    def _rotate_api_key(self) -> bool:
        """التبديل للمفتاح التالي إن وجد"""
        if len(self._api_keys) <= 1:
            return False

        total = len(self._api_keys)
        start = (self._api_key_index + 1) % total
        idx = start

        while True:
            candidate = self._api_keys[idx]
            if self._is_key_available(candidate):
                self._api_key_index = idx
                self.api_key = candidate
                self._logger.warning(
                    f"Gemini key rotated to index {self._api_key_index + 1}/{total}"
                )
                return True

            idx = (idx + 1) % total
            if idx == start:
                break

        return False

    async def _get_session(self) -> aiohttp.ClientSession:
        """الحصول على جلسة HTTP"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _make_request(self, messages: list[Message], model: str, **kwargs) -> dict[str, Any]:
        """تنفيذ الطلب لـ Gemini API"""

        session = await self._get_session()

        # تحويل الرسائل لتنسيق Gemini
        contents = []
        for msg in messages:
            role = "user" if msg.role == MessageRole.USER else "model"
            contents.append({"role": role, "parts": [{"text": msg.content}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 8192),
                "topP": kwargs.get("top_p", 1.0),
            },
        }

        if not (self._api_keys or self.api_key):
            raise ProviderAuthenticationError(provider_name=self.name)

        max_attempts = max(len(self._api_keys), 1)
        last_error: Exception | None = None

        for _ in range(max_attempts):
            current_key = self._current_api_key() or self.api_key
            url = f"{self.GEMINI_API_URL}/{model}:generateContent?key={current_key}"

            try:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._convert_response(data, model)

                    error_body = await response.text()
                    last_error = ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                    # تبريد المفتاح عند 429 أو 403/401
                    if response.status in (401, 403, 429):
                        retry_after = response.headers.get("retry-after")
                        try:
                            cooldown_seconds = float(retry_after) if retry_after else 30.0
                        except ValueError:
                            cooldown_seconds = 30.0
                        if current_key:
                            self._set_key_cooldown(current_key, cooldown_seconds)

                    if not self._rotate_api_key():
                        raise last_error

            except aiohttp.ClientError as e:
                last_error = ProviderNotAvailableError(provider_name=self.name, reason=str(e))
                if not self._rotate_api_key():
                    raise last_error

        if last_error:
            raise last_error

        raise ProviderResponseError(
            provider_name=self.name,
            status_code=500,
            response_body="Gemini request failed after key rotation attempts",
        )

    def _convert_response(self, gemini_response: dict[str, Any], model: str) -> dict[str, Any]:
        """تحويل استجابة Gemini لتنسيق OpenAI"""

        candidates = gemini_response.get("candidates", [])

        choices = []
        for i, candidate in enumerate(candidates):
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

            choices.append(
                {
                    "index": i,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": candidate.get("finishReason", "stop").lower(),
                }
            )

        usage = gemini_response.get("usageMetadata", {})

        return {
            "id": f"gemini-{int(time.time())}",
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
            },
        }

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs
    ) -> AsyncIterator[str]:
        """تدفق الاستجابة - NotImplemented للمختصر"""
        raise NotImplementedError("Gemini streaming not implemented in this version")

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """حساب التكلفة"""
        costs = self.GEMINI_COSTS.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * costs["input"] / 1_000_000) + (
            output_tokens * costs["output"] / 1_000_000
        )

    def get_model_tier(self, model: str) -> ModelTier:
        """تحديد مستوى نماذج Gemini بدقة أفضل للتوجيه"""
        model_lower = model.lower()
        if "gemini-2.5-flash" in model_lower:
            return ModelTier.TIER_2_TACTICAL
        if "pro" in model_lower:
            return ModelTier.TIER_1_STRATEGIC
        return super().get_model_tier(model)

    async def close(self) -> None:
        """إغلاق الجلسة"""
        if self._session and not self._session.closed:
            await self._session.close()


def create_gemini_provider(
    api_key: str | None = None, default_model: str = "gemini-2.5-flash"
) -> GeminiProvider:
    """إنشاء مزود Gemini بسهولة"""
    return GeminiProvider(api_key=api_key, default_model=default_model)
