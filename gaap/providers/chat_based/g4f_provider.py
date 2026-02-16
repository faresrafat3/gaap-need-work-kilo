# G4F Provider
import asyncio

# استيراد من المسار النسبي
import time
from collections.abc import AsyncGenerator
from typing import Any

from gaap.core.exceptions import (
    ProviderResponseError,
)
from gaap.core.types import (
    Message,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider, get_logger, register_provider

# =============================================================================
# G4F Models Mapping
# =============================================================================

G4F_MODELS = {
    # OpenAI Models
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    # Claude Models
    "claude-3-5-sonnet": "claude-3.5-sonnet",
    "claude-3-5-opus": "claude-3.5-opus",
    "claude-3-opus": "claude-3-opus",
    "claude-3-sonnet": "claude-3-sonnet",
    "claude-3-haiku": "claude-3-haiku",
    # Gemini Models
    "gemini-pro": "gemini-pro",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "gemini-1.5-flash": "gemini-1.5-flash",
    # Llama Models
    "llama-3-70b": "llama-3-70b",
    "llama-3-8b": "llama-3-8b",
    "llama-2-70b": "llama-2-70b",
    # Mistral Models
    "mixtral-8x7b": "mixtral-8x7b",
    "mistral-7b": "mistral-7b",
}

# تكاليف تقريبية (g4f مجاني لكن للتتبع)
G4F_COSTS = {
    "gpt-4o": {"input": 0.0, "output": 0.0},
    "gpt-4o-mini": {"input": 0.0, "output": 0.0},
    "claude-3-5-sonnet": {"input": 0.0, "output": 0.0},
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
    "llama-3-70b": {"input": 0.0, "output": 0.0},
}


# =============================================================================
# G4F Provider Implementation
# =============================================================================


@register_provider("g4f")
class G4FProvider(BaseProvider):
    """
    مزود G4F - وصول مجاني للنماذج

    الميزات:
    - مجاني بالكامل
    - دعم نماذج متعددة
    - تناوب تلقائي بين الموفرين
    - لا حاجة لمفتاح API
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        default_model: str = "gpt-4o-mini",
        provider: str | None = None,
        **kwargs: Any,
    ) -> None:
        # استخراج النماذج المدعومة
        models = list(G4F_MODELS.keys())

        super().__init__(
            name="g4f",
            provider_type=ProviderType.CHAT_BASED,
            models=models,
            api_key=api_key,
            base_url=base_url,
            rate_limit_rpm=30,  # حد أدنى للسلامة
            rate_limit_tpm=50000,
            timeout=60.0,
            max_retries=2,
            default_model=default_model,
        )

        self._preferred_provider = provider
        self._g4f_module = None
        self._logger = get_logger("gaap.provider.g4f")

    async def _ensure_g4f_loaded(self) -> None:
        """التأكد من تحميل مكتبة g4f"""
        if self._g4f_module is not None:
            return

        try:
            # محاولة استيراد g4f
            import g4f

            self._g4f_module = g4f
            self._logger.info("g4f library loaded successfully")
        except ImportError:
            self._logger.warning("g4f library not installed. Install with: pip install g4f")
            # سنستخدم وضع المحاكاة
            self._g4f_module = None

    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        """تنفيذ الطلب باستخدام g4f"""
        await self._ensure_g4f_loaded()

        # تحويل الرسائل
        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        # الحصول على اسم النموذج في g4f
        g4f_model = G4F_MODELS.get(model, model)

        if self._g4f_module is not None:
            # استخدام g4f الفعلي
            try:
                # تحديد المزود
                if self._preferred_provider:
                    provider = getattr(self._g4f_module.Provider, self._preferred_provider, None)
                else:
                    provider = None

                # استدعاء API
                response = await asyncio.to_thread(
                    self._g4f_module.ChatCompletion.create,
                    model=g4f_model,
                    messages=formatted_messages,
                    provider=provider,
                    **kwargs,
                )

                # تحويل الاستجابة
                if isinstance(response, str):
                    # استجابة نصية بسيطة
                    return {
                        "id": f"g4f-{int(time.time())}",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": response},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                else:
                    # استجابة منظمة
                    return response

            except Exception as e:
                self._logger.error(f"g4f request failed: {e}")
                raise ProviderResponseError(
                    provider_name=self.name, status_code=500, response_body=str(e)
                )
        else:
            # وضع المحاكاة للمطورين بدون g4f
            return await self._simulate_response(formatted_messages, g4f_model)

    async def _simulate_response(
        self, messages: list[dict[str, str]], model: str
    ) -> dict[str, Any]:
        """
        محاكاة استجابة للاختبار والتطوير
        عندما تكون مكتبة g4f غير متوفرة
        """
        self._logger.warning(
            f"Using simulation mode for model {model}. Install g4f for real responses."
        )

        # استخراج آخر رسالة مستخدم
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break

        # محاكاة استجابة
        simulated_response = (
            f"[SIMULATION MODE - g4f not installed]\n"
            f"Model: {model}\n"
            f"Your message: {user_message[:100]}...\n\n"
            f"To get real responses, install g4f:\n"
            f"pip install g4f"
        )

        # محاكاة تأخير الشبكة
        await asyncio.sleep(0.5)

        return {
            "id": f"sim-{int(time.time())}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": simulated_response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(simulated_response.split()),
                "total_tokens": len(user_message.split()) + len(simulated_response.split()),
            },
        }

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """تدفق الاستجابة"""
        await self._ensure_g4f_loaded()

        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        g4f_model = G4F_MODELS.get(model, model)

        if self._g4f_module is not None:
            try:
                # استخدام التدفق في g4f
                response = await asyncio.to_thread(
                    self._g4f_module.ChatCompletion.create,
                    model=g4f_model,
                    messages=formatted_messages,
                    stream=True,
                    **kwargs,
                )

                for chunk in response:
                    if isinstance(chunk, str):
                        yield chunk
                    elif isinstance(chunk, dict):
                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if content:
                            yield content

            except Exception as e:
                self._logger.error(f"g4f stream error: {e}")
                raise ProviderResponseError(
                    provider_name=self.name, status_code=500, response_body=str(e)
                )
        else:
            # محاكاة التدفق
            simulated = f"[Simulation] Model {model} response..."
            for word in simulated.split():
                yield word + " "
                await asyncio.sleep(0.05)

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """حساب التكلفة (مجاني)"""
        # g4f مجاني لكن نحتفظ بالتتبع
        costs = G4F_COSTS.get(model, {"input": 0.0, "output": 0.0})
        return (input_tokens * costs["input"] / 1000) + (output_tokens * costs["output"] / 1000)

    def get_available_providers(self) -> list[str]:
        """الحصول على قائمة المزودين المتاحين في g4f"""
        if self._g4f_module is None:
            return []

        try:
            providers = []
            for name in dir(self._g4f_module.Provider):
                if not name.startswith("_"):
                    providers.append(name)
            return providers
        except Exception:
            return []


# =============================================================================
# Convenience Functions
# =============================================================================


def create_g4f_provider(
    default_model: str = "gpt-4o-mini", provider: str | None = None
) -> G4FProvider:
    """إنشاء مزود g4f بسهولة"""
    return G4FProvider(default_model=default_model, provider=provider)


# =============================================================================
# Model Alias Support
# =============================================================================


def get_g4f_model_alias(model: str) -> str:
    """الحصول على اسم النموذج في g4f"""
    return G4F_MODELS.get(model, model)


def list_g4f_models() -> list[str]:
    """قائمة النماذج المدعومة"""
    return list(G4F_MODELS.keys())
