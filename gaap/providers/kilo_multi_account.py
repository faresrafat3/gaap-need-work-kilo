"""
Kilo Multi-Account Provider

Run multiple Kilo accounts in parallel for:
- Higher throughput
- Context isolation per agent/task
- Automatic rotation and load balancing

Usage:
    from gaap.providers.kilo_multi_account import KiloMultiAccountProvider

    provider = KiloMultiAccountProvider(
        accounts=["token1", "token2", "token3"]
    )

    # Run parallel tasks
    results = await provider.execute_parallel([
        {"prompt": "Task 1", "model": "z-ai/glm-5"},
        {"prompt": "Task 2", "model": "minimax/minimax-m2.5"},
    ])
"""

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from gaap.core.exceptions import ProviderNotAvailableError, ProviderResponseError
from gaap.core.types import ChatCompletionResponse, Message, MessageRole, ProviderType
from gaap.core.logging import get_standard_logger as get_logger
from gaap.providers.base_provider import BaseProvider, register_provider

KILO_API_URL = "https://api.kilo.ai/api/openrouter/chat/completions"

FREE_MODELS = [
    "z-ai/glm-5",
    "minimax/minimax-m2.5",
    "stepfun/step-3.5-flash",
    "arcee-ai/trinity-large-preview",
    "corethink",
]


@dataclass
class KiloAccount:
    token: str
    name: str = ""
    status: str = "active"
    requests_count: int = 0
    last_used: float = 0.0
    cooldown_until: float = 0.0

    def is_available(self) -> bool:
        return self.status == "active" and time.time() >= self.cooldown_until


@dataclass
class AgentTask:
    agent_id: str
    agent_type: str
    prompt: str
    model: str = "z-ai/glm-5"
    context: dict[str, Any] = field(default_factory=dict)
    account_index: int | None = None


class KiloMultiAccountProvider(BaseProvider):
    """
    Multi-account Kilo provider for parallel execution.

    Features:
    - Multiple accounts for higher throughput
    - Context isolation per agent
    - Automatic load balancing
    - Free models by default

    Usage:
        provider = KiloMultiAccountProvider(
            accounts=["token1", "token2", "token3"]
        )

        # Parallel execution
        results = await provider.execute_parallel([
            AgentTask(agent_id="coder", agent_type="coder", prompt="Write a function"),
            AgentTask(agent_id="critic", agent_type="critic", prompt="Review code"),
        ])
    """

    def __init__(
        self,
        accounts: list[str] | None = None,
        default_model: str = "z-ai/glm-5",
        **kwargs: Any,
    ) -> None:
        accounts = accounts or []
        if not accounts:
            env_tokens = os.environ.get("KILO_TOKENS", "")
            if env_tokens:
                accounts = [t.strip() for t in env_tokens.split(",") if t.strip()]
            else:
                single_token = os.environ.get("KILO_API_KEY") or os.environ.get("KILO_TOKEN")
                if single_token:
                    accounts = [single_token]

        self._accounts: list[KiloAccount] = [
            KiloAccount(token=token, name=f"account_{i + 1}") for i, token in enumerate(accounts)
        ]
        self._account_index = 0
        self._sessions: dict[str, aiohttp.ClientSession] = {}

        super().__init__(
            name="kilo_multi",
            provider_type=kwargs.get("provider_type", ProviderType.FREE_TIER),
            models=FREE_MODELS,
            api_key=accounts[0] if accounts else None,
            rate_limit_rpm=100 * len(accounts) if accounts else 100,
            rate_limit_tpm=500000 * len(accounts) if accounts else 500000,
            timeout=120.0,
            max_retries=3,
            default_model=default_model,
        )

        self._logger = get_logger("gaap.provider.kilo_multi")

    @property
    def account_count(self) -> int:
        return len(self._accounts)

    @property
    def active_accounts(self) -> list[KiloAccount]:
        return [a for a in self._accounts if a.is_available()]

    def add_account(self, token: str, name: str = "") -> None:
        account = KiloAccount(token=token, name=name or f"account_{len(self._accounts) + 1}")
        self._accounts.append(account)
        self._logger.info(f"Added account: {account.name}")

    def get_next_account(self) -> KiloAccount | None:
        available = self.active_accounts
        if not available:
            return None

        for _ in range(len(self._accounts)):
            account = self._accounts[self._account_index % len(self._accounts)]
            self._account_index += 1
            if account.is_available():
                return account

        return available[0]

    async def _get_session(self, token: str) -> aiohttp.ClientSession:
        if token not in self._sessions or self._sessions[token].closed:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "User-Agent": "GAAP-MultiAccount",
                "X-KILOCODE-EDITORNAME": "GAAP",
            }
            self._sessions[token] = aiohttp.ClientSession(headers=headers)
        return self._sessions[token]

    def _get_free_model_name(self, model: str) -> str:
        base = model.replace(":free", "")
        if base in FREE_MODELS:
            return f"{base}:free"
        return model

    async def _make_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> dict[str, Any]:
        account = self.get_next_account()
        if not account:
            raise ProviderNotAvailableError(provider_name=self.name, reason="No available accounts")

        session = await self._get_session(account.token)
        actual_model = self._get_free_model_name(model)

        formatted_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        payload = {
            "model": actual_model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }

        try:
            async with session.post(
                KILO_API_URL,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                account.requests_count += 1
                account.last_used = time.time()

                if response.status == 401:
                    account.status = "invalid"
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=401,
                        response_body="Invalid token",
                    )

                if response.status == 429:
                    account.cooldown_until = time.time() + 60
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=429,
                        response_body="Rate limited",
                    )

                if response.status != 200:
                    error_body = await response.text()
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=response.status,
                        response_body=error_body,
                    )

                data: dict[str, Any] = await response.json()
                return dict(data)

        except aiohttp.ClientError as e:
            raise ProviderNotAvailableError(provider_name=self.name, reason=str(e))

    async def _stream_request(
        self, messages: list[Message], model: str, **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        yield ""  # pragma: no cover

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def execute_single(
        self,
        task: AgentTask,
    ) -> tuple[str, str, int]:
        account = (
            self._accounts[task.account_index]
            if task.account_index is not None
            else self.get_next_account()
        )
        if not account:
            raise ProviderNotAvailableError(provider_name=self.name, reason="No accounts")

        session = await self._get_session(account.token)
        model = self._get_free_model_name(task.model)

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": task.prompt}],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        start = time.time()
        async with session.post(
            KILO_API_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as response:
            account.requests_count += 1
            account.last_used = time.time()

            if response.status != 200:
                error = await response.text()
                raise ProviderResponseError(
                    provider_name=self.name,
                    status_code=response.status,
                    response_body=error,
                )

            data = await response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            tokens = data.get("usage", {}).get("total_tokens", 0)
            elapsed = time.time() - start

            return content, account.name, tokens

    async def execute_parallel(
        self,
        tasks: list[AgentTask],
        max_concurrent: int = 5,
    ) -> list[dict[str, Any]]:
        results = []
        semaphore = asyncio.Semaphore(min(max_concurrent, len(self._accounts)))

        async def run_task(task: AgentTask, idx: int) -> dict[str, Any]:
            async with semaphore:
                task.account_index = idx % len(self._accounts)
                try:
                    content, account_name, tokens = await self.execute_single(task)
                    return {
                        "agent_id": task.agent_id,
                        "agent_type": task.agent_type,
                        "content": content,
                        "account": account_name,
                        "tokens": tokens,
                        "success": True,
                        "error": None,
                    }
                except Exception as e:
                    return {
                        "agent_id": task.agent_id,
                        "agent_type": task.agent_type,
                        "content": "",
                        "account": "",
                        "tokens": 0,
                        "success": False,
                        "error": str(e),
                    }

        coroutines = [run_task(task, i) for i, task in enumerate(tasks)]
        results = await asyncio.gather(*coroutines)

        return results

    async def close(self) -> None:
        for session in self._sessions.values():
            if not session.closed:
                await session.close()
        self._sessions.clear()

    def shutdown(self) -> None:
        asyncio.create_task(self.close())
        super().shutdown()

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_accounts": len(self._accounts),
            "active_accounts": len(self.active_accounts),
            "accounts": [
                {
                    "name": a.name,
                    "status": a.status,
                    "requests": a.requests_count,
                    "available": a.is_available(),
                }
                for a in self._accounts
            ],
        }


def create_kilo_multi_provider(
    accounts: list[str] | None = None,
    default_model: str = "z-ai/glm-5",
) -> KiloMultiAccountProvider:
    return KiloMultiAccountProvider(accounts=accounts, default_model=default_model)
