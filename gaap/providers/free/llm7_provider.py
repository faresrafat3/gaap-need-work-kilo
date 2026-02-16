from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class SimpleChatResponse:
    """Simple chat response"""

    id: str
    content: str
    model: str
    provider: str
    success: bool = True
    error: str | None = None
    tokens_used: int = 0


class LLM7Provider:
    """
    LLM7.io - Free OpenAI-compatible API

    No API key required!
    Models: gpt-3.5-turbo, gpt-4o, claude-3.5-sonnet, llama-3.1-405b, and more.
    """

    BASE_URL = "https://api.llm7.io/v1"

    MODELS = [
        "gpt-3.5-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "claude-3.5-sonnet",
        "claude-3-opus",
        "claude-3-haiku",
        "llama-3.1-405b-instruct",
        "llama-3.1-70b-instruct",
        "llama-3.1-8b-instruct",
        "llama-3-70b-instruct",
        "llama-3-8b-instruct",
        "mistral-large",
        "mistral-medium",
        "mistral-small",
        "mixtral-8x7b-instruct",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
    ]

    DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.name = "llm7"

    def get_available_models(self) -> list[str]:
        return self.MODELS

    def get_default_model(self) -> str:
        return self.DEFAULT_MODEL

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> SimpleChatResponse:
        """Execute chat completion via LLM7.io"""
        model = model or self.DEFAULT_MODEL

        if model not in self.MODELS:
            model = self.DEFAULT_MODEL

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
                if response.status != 200:
                    error_text = await response.text()
                    return SimpleChatResponse(
                        id="error",
                        content="",
                        model=model,
                        provider="llm7",
                        success=False,
                        error=f"HTTP {response.status}: {error_text}",
                    )

                data = await response.json()

                choice = data.get("choices", [{}])[0]
                message_data = choice.get("message", {})
                content = message_data.get("content", "")

                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)

                return SimpleChatResponse(
                    id=data.get("id", "llm7-" + str(hash(str(messages)))[:8]),
                    content=content,
                    model=model,
                    provider="llm7",
                    tokens_used=tokens_used,
                )

        except Exception as e:
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="llm7",
                success=False,
                error=str(e),
            )

    async def embeddings(
        self,
        text: str | list[str],
        model: str = "text-embedding-3-small",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Generate embeddings"""
        return {
            "success": False,
            "error": "Embeddings not supported on LLM7.io free tier",
        }
