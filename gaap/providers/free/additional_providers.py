from dataclasses import dataclass

import aiohttp


@dataclass
class SimpleChatResponse:
    id: str
    content: str
    model: str
    provider: str
    success: bool = True
    error: str | None = None
    tokens_used: int = 0


async def _make_request(
    url: str, payload: dict, headers: dict = None, timeout: int = 120
) -> tuple[int, dict | str]:
    """Helper for async HTTP requests"""
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.post(
                url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response,
        ):
            status = response.status
            if status != 200:
                text = await response.text()
                return status, text
            data = await response.json()
            return status, data
    except Exception as e:
        return 0, str(e)


class PuterProvider:
    """Puter.js - Free unlimited AI API (User-pays model)"""

    BASE_URL = "https://api.puter.com/v1"

    MODELS = [
        "gpt-5-nano",
        "gpt-5",
        "claude-sonnet-4-5",
        "claude-4-sonnet",
        "claude-4-haiku",
        "deepseek/deepseek-r1",
        "deepseek/deepseek-chat",
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "llama-3.1-70b",
        "llama-3.1-8b",
        "mistral-large",
        "mistral-small",
        "qwen/qwen-2.5-72b",
        "qwen/qwen-2.5-7b",
    ]

    DEFAULT_MODEL = "gpt-5-nano"

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        status, result = await _make_request(
            f"{self.BASE_URL}/ai/chat", payload, timeout=self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="puter",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("text", "") or result.get("message", {}).get("content", "")
        return SimpleChatResponse(
            id="puter-" + str(hash(str(messages)))[:8],
            content=content,
            model=model,
            provider="puter",
        )


class ScitelyProvider:
    """Scitely - Free Unlimited AI API (Nonprofit)"""

    BASE_URL = "https://api.scitely.com/v1"

    MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
        "qwen-plus",
        "qwen-turbo",
        "kimi-k2.5-thinking",
        "glm-4",
        "glm-4-flash",
        "llama-3.1-70b",
        "llama-3.1-8b",
        "mixtral-8x7b",
    ]

    DEFAULT_MODEL = "deepseek-chat"

    def __init__(self, api_key: str = "free", timeout: int = 120):
        self.api_key = api_key
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {"Authorization": f"Bearer {self.api_key}"}
        status, result = await _make_request(
            f"{self.BASE_URL}/chat/completions", payload, headers, self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="scitely",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return SimpleChatResponse(
            id=result.get("id", "scitely-" + str(hash(str(messages)))[:8]),
            content=content,
            model=model,
            provider="scitely",
        )


class OllamaProvider:
    """Ollama - Run LLMs locally"""

    BASE_URL = "http://localhost:11434"

    MODELS = [
        "llama3.2",
        "llama3.2:1b",
        "llama3.2:3b",
        "llama3.1",
        "llama3.1:70b",
        "mistral",
        "mixtral",
        "qwen2.5",
        "qwen2.5:7b",
        "qwen2.5:72b",
        "phi3",
        "phi3.5",
        "gemma2",
        "command-r",
    ]

    DEFAULT_MODEL = "llama3.2"

    def __init__(self, base_url: str | None = None, timeout: int = 120):
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens:
            payload["num_predict"] = max_tokens

        status, result = await _make_request(
            f"{self.base_url}/api/generate", payload, timeout=self.timeout
        )

        if status == 0 and "Connection" in str(result):
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="ollama",
                success=False,
                error="Ollama not running. Start: ollama serve",
            )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="ollama",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("response", "")
        return SimpleChatResponse(
            id="ollama-" + str(hash(str(messages)))[:8],
            content=content,
            model=model,
            provider="ollama",
        )


class MLvocaProvider:
    """MLvoca - Free Ollama-based API (no key required)"""

    BASE_URL = "https://mlvoca.com"

    MODELS = ["tinyllama", "deepseek-r1:1.5b"]

    DEFAULT_MODEL = "tinyllama"

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if temperature:
            payload["options"] = {"temperature": temperature}
        if max_tokens:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens

        status, result = await _make_request(
            f"{self.BASE_URL}/api/generate", payload, timeout=self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="mlvoca",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("response", "")
        return SimpleChatResponse(
            id="mlvoca-" + str(hash(str(messages)))[:8],
            content=content,
            model=model,
            provider="mlvoca",
        )


class OllamaFreeAPIProvider:
    """OllamaFreeAPI - Free distributed Ollama API"""

    BASE_URL = "https://api.ollama-free.com"

    MODELS = [
        "llama3.2",
        "llama3.2:1b",
        "llama3.2:3b",
        "llama3.1",
        "llama3.1:70b",
        "mistral",
        "mixtral",
        "qwen2.5",
        "qwen2.5:7b",
        "qwen2.5:72b",
        "phi3",
        "phi3.5",
        "gemma2",
        "deepseek-coder-v2",
        "llama3:8b",
    ]

    DEFAULT_MODEL = "llama3.2"

    def __init__(self, timeout: int = 120):
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if temperature:
            payload["options"] = {"temperature": temperature}
        if max_tokens:
            payload["options"] = payload.get("options", {})
            payload["options"]["num_predict"] = max_tokens

        status, result = await _make_request(
            f"{self.BASE_URL}/api/generate", payload, timeout=self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="ollamafree",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("response", "")
        return SimpleChatResponse(
            id="ollamafree-" + str(hash(str(messages)))[:8],
            content=content,
            model=model,
            provider="ollamafree",
        )


class OpenRouterProvider:
    """OpenRouter.ai - Aggregates 100+ models with free tier"""

    BASE_URL = "https://openrouter.ai/api/v1"

    MODELS = [
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
        "meta-llama/llama-3.1-8b-instruct",
        "meta-llama/llama-3.1-70b-instruct",
        "mistralai/mistral-7b-instruct",
        "mistralai/mixtral-8x7b-instruct",
        "anthropic/claude-3-haiku",
        "anthropic/claude-3-sonnet",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4o-mini",
    ]

    DEFAULT_MODEL = "google/gemma-2-9b-it"

    def __init__(self, api_key: str = "free", base_url: str | None = None, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        headers = {"Authorization": f"Bearer {self.api_key}"}
        status, result = await _make_request(
            f"{self.base_url}/chat/completions", payload, headers, self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="openrouter",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        return SimpleChatResponse(
            id=result.get("id", "or-" + str(hash(str(messages)))[:8]),
            content=content,
            model=model,
            provider="openrouter",
            tokens_used=usage.get("total_tokens", 0),
        )


class HuggingChatProvider:
    """HuggingChat - Free chat with Meta-Llama, Mistral, etc."""

    BASE_URL = "https://api.huggingface.co"

    MODELS = [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "microsoft/Phi-3-mini-128k-instruct",
        "google/gemma-2-9b-it",
    ]

    DEFAULT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    def __init__(self, timeout: int = 120):
        self.base_url = self.BASE_URL
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        status, result = await _make_request(
            f"{self.base_url}/chat/completions", payload, timeout=self.timeout
        )

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="huggingchat",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return SimpleChatResponse(
            id=result.get("id", "hc-" + str(hash(str(messages)))[:8]),
            content=content,
            model=model,
            provider="huggingchat",
        )


class PoeProvider:
    """Poe.com - Free tier with Claude, GPT-4"""

    BASE_URL = "https://poe.com/api/v1"

    MODELS = [
        "claude-3-5-sonnet",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "claude-3-opus",
        "claude-3-haiku",
    ]

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, p_b: str = "", p_lat: str = "", timeout: int = 120):
        self.p_b = p_b
        self.p_lat = p_lat
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL

        if not self.p_b or not self.p_lat:
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="poe",
                success=False,
                error="Poe requires p_b and p_lat cookies",
            )

        payload = {
            "query": messages[-1].get("content", ""),
            "variables": {"chatId": "", "modelName": model},
        }

        headers = {"Content-Type": "application/json", "Referer": "https://poe.com/"}
        cookies = {"p-b": self.p_b, "p-lat": self.p_lat}

        try:
            async with (
                aiohttp.ClientSession() as session,
                session.post(
                    f"{self.BASE_URL}/chat/ChatPoe",
                    json=payload,
                    headers=headers,
                    cookies=cookies,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response,
            ):
                if response.status != 200:
                    text = await response.text()
                    return SimpleChatResponse(
                        id="error",
                        content="",
                        model=model,
                        provider="poe",
                        success=False,
                        error=f"HTTP {response.status}: {text[:200]}",
                    )
                data = await response.json()
                content = data.get("data", {}).get("chatPoe", {}).get("text", "")
                return SimpleChatResponse(
                    id="poe-" + str(hash(str(messages)))[:8],
                    content=content,
                    model=model,
                    provider="poe",
                )
        except Exception as e:
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="poe",
                success=False,
                error=str(e),
            )


class YouChatProvider:
    """You.com - Free AI assistant"""

    BASE_URL = "https://youapi.ai/api/v1"

    MODELS = ["you", "gpt-3.5-turbo", "gpt-4"]

    DEFAULT_MODEL = "you"

    def __init__(self, timeout: int = 120):
        self.base_url = self.BASE_URL
        self.timeout = timeout

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> SimpleChatResponse:
        model = model or self.DEFAULT_MODEL

        prompt = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)
        payload = {"model": model, "prompt": prompt, "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens

        status, result = await _make_request(f"{self.base_url}/chat", payload, timeout=self.timeout)

        if status != 200:
            err = result[:200] if isinstance(result, str) else str(result)
            return SimpleChatResponse(
                id="error",
                content="",
                model=model,
                provider="youchat",
                success=False,
                error=f"HTTP {status}: {err}",
            )

        content = result.get("choices", [{}])[0].get("text", "")
        return SimpleChatResponse(
            id="yc-" + str(hash(str(messages)))[:8],
            content=content,
            model=model,
            provider="youchat",
        )
