# WebChat Bridge Provider
# يربط webchat_call() بواجهة BaseProvider للاستخدام في GAAPEngine
# Supports multi-account rotation via AccountPool
import asyncio
import gc
import time
from collections.abc import AsyncIterator
from typing import Any

from gaap.core.exceptions import ProviderResponseError
from gaap.core.memory_guard import get_guard
from gaap.core.types import (
    Message,
    ModelTier,
    ProviderType,
)
from gaap.providers.base_provider import BaseProvider, get_logger

# =============================================================================
# WebChat Models
# =============================================================================

WEBCHAT_MODELS = {
    "kimi": {
        "models": [
            "kimi-k2.5-thinking",
            "kimi-k2.5",
            "kimi-k2",
            "kimi-k1.5-long",
        ],
        "default": "kimi-k2.5-thinking",
        "tier": ModelTier.TIER_1_STRATEGIC,
    },
    "glm": {
        "models": ["glm-4-plus"],
        "default": "glm-4-plus",
        "tier": ModelTier.TIER_2_TACTICAL,
    },
    "deepseek": {
        "models": ["deepseek-v3"],
        "default": "deepseek-v3",
        "tier": ModelTier.TIER_2_TACTICAL,
    },
}


# =============================================================================
# WebChat Bridge Provider
# =============================================================================

class WebChatBridgeProvider(BaseProvider):
    """
    جسر بين webchat_call() وواجهة BaseProvider
    
    يسمح باستخدام مزودي WebChat (Kimi, GLM, DeepSeek)
    داخل GAAPEngine مع كل الطبقات (SmartRouter, FallbackManager, etc.)
    
    Multi-Account Support:
      - Auto-discovers authenticated accounts from disk
      - Uses AccountPool for smart rotation (best health score)
      - Detects hard rate limits (e.g., Kimi 3-hour cooldown)
      - Auto-switches to next available account on failure
    """

    def __init__(
        self,
        webchat_provider: str = "kimi",
        account: str = "default",
        timeout: int = 120,
        **kwargs
    ):
        self._webchat_provider = webchat_provider
        self._account = account  # preferred account (fallback to others)
        self._webchat_timeout = timeout
        self._cooldown_sec = 5  # ثواني راحة بين الطلبات لحماية الشبكة

        # Initialize AccountPool for multi-account support
        self._pool = None
        self._pool_initialized = False
        try:
            from .account_manager import PoolManager, auto_discover_accounts
            mgr = PoolManager.instance()
            auto_discover_accounts()
            self._pool = mgr.pool(webchat_provider)
            self._pool_initialized = True
        except Exception:
            pass

        # الحصول على تكوين المزود
        config = WEBCHAT_MODELS.get(webchat_provider, WEBCHAT_MODELS["kimi"])
        models = config["models"]
        default_model = config["default"]
        self._model_tier = config["tier"]

        super().__init__(
            name=f"webchat_{webchat_provider}",
            provider_type=ProviderType.CHAT_BASED,
            models=models,
            rate_limit_rpm=60,
            rate_limit_tpm=500000,
            timeout=float(timeout),
            max_retries=2,
            default_model=default_model,
        )

        self._logger = get_logger(f"gaap.provider.webchat_{webchat_provider}")

        # Log multi-account status
        if self._pool and self._pool.accounts:
            active = [a.label for a in self._pool.active_accounts]
            self._logger.info(
                f"Multi-account: {len(self._pool.accounts)} accounts "
                f"({len(active)} active: {', '.join(active)})"
            )
        else:
            self._logger.info(f"Single account: {account}")

    def _select_account(self, model: str = "") -> str:
        """Select the best available account using AccountPool.
        
        Returns the account label to use for the next call.
        Falls back to self._account if pool is not available.
        """
        if not self._pool or not self._pool.accounts:
            return self._account

        # Try smart selection
        should, reason, acct_label = self._pool.should_call(
            label="",  # don't lock to specific account
            model=model,
        )

        if should and acct_label:
            if acct_label != self._account:
                self._logger.info(f"Account rotation: using '{acct_label}' ({reason})")
            return acct_label

        # All accounts unavailable — log why
        self._logger.warning(f"All accounts unavailable: {reason}")

        # Return the one with shortest wait time
        min_wait = float("inf")
        best_label = self._account
        for acct in self._pool.accounts:
            wait = acct.rate_tracker.seconds_until_next_allowed
            if wait < min_wait:
                min_wait = wait
                best_label = acct.label

        return best_label

    def _record_call_result(
        self, account: str, success: bool,
        latency_ms: float = 0, tokens_used: int = 0,
        error_msg: str = ""
    ):
        """Record call result in AccountPool and detect hard cooldowns."""
        if not self._pool:
            return

        self._pool.record_call(
            account, success=success,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            error_msg=error_msg[:100] if error_msg else "",
        )

        # Detect hard rate limits (e.g., Kimi 3-hour ban)
        if not success and error_msg:
            try:
                from .account_manager import detect_hard_cooldown
                result = detect_hard_cooldown(error_msg)
                if result:
                    cooldown_sec, reason = result
                    acct = self._pool.get_account(account)
                    if acct:
                        acct.rate_tracker.set_hard_cooldown(cooldown_sec, reason)
                        import datetime
                        expires = datetime.datetime.now() + datetime.timedelta(seconds=cooldown_sec)
                        self._logger.warning(
                            f"🚫 Hard rate limit on '{account}': {reason} "
                            f"({cooldown_sec // 3600}h {(cooldown_sec % 3600) // 60}m cooldown, "
                            f"expires at {expires.strftime('%H:%M:%S')})"
                        )
            except Exception:
                pass

    def get_account_status(self) -> str:
        """Get a human-readable status of all accounts."""
        if not self._pool or not self._pool.accounts:
            return f"Single account: {self._account}"
        return self._pool.dashboard()

    def get_model_tier(self, model: str) -> ModelTier:
        """كل نماذج webchat تعتبر TIER_1 أو TIER_2"""
        return self._model_tier

    async def _make_request(
        self,
        messages: list[Message],
        model: str,
        **kwargs
    ) -> dict[str, Any]:
        """تنفيذ الطلب عبر webchat_call() مع إدارة ذاكرة + retry + multi-account"""
        # Memory check before making LLM call
        guard = get_guard()
        guard.check(context=f"webchat_bridge before {self._webchat_provider} call")

        # تحويل Message objects لقواميس
        formatted_messages = []
        for m in messages:
            formatted_messages.append({
                "role": m.role.value,
                "content": m.content,
            })

        # Select best account (via AccountPool or default)
        current_account = self._select_account(model)

        # Track which accounts we've tried to avoid infinite loops
        tried_accounts = set()

        # استدعاء webchat_call مع retry للأخطاء المؤقتة
        max_retries = 3
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                from .webchat_providers import webchat_call

                call_start = time.time()
                content = await asyncio.to_thread(
                    webchat_call,
                    self._webchat_provider,
                    formatted_messages,
                    model,
                    current_account,
                    self._webchat_timeout,
                )
                call_latency = (time.time() - call_start) * 1000

                if not content or not content.strip():
                    raise ProviderResponseError(
                        provider_name=self.name,
                        status_code=500,
                        response_body="Empty response from webchat provider"
                    )

                # Record success
                self._record_call_result(
                    current_account, success=True,
                    latency_ms=call_latency,
                    tokens_used=len(content.split()) * 2,
                )

                # تقدير عدد الرموز
                input_tokens = sum(len(m.content.split()) for m in messages) * 2
                output_tokens = len(content.split()) * 2

                result = {
                    "id": f"webchat-{self._webchat_provider}-{current_account}-{int(time.time())}",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "_account_used": current_account,
                }

                # تنظيف الذاكرة بعد كل طلب لتقليل الضغط
                del formatted_messages
                gc.collect()

                # Memory check after response (catch leaks early)
                guard.check(context=f"webchat_bridge after {self._webchat_provider} call")

                # cooldown بين الطلبات لتقليل ضغط النظام
                await asyncio.sleep(self._cooldown_sec)

                return result

            except ImportError as e:
                raise ProviderResponseError(
                    provider_name=self.name,
                    status_code=500,
                    response_body=f"webchat_providers not available: {e}"
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                full_err = str(e)

                # Record failure
                self._record_call_result(
                    current_account, success=False,
                    error_msg=full_err,
                )
                tried_accounts.add(current_account)

                # Check if this is a hard rate limit (Kimi 3-hour ban)
                is_hard_limit = False
                try:
                    from .account_manager import detect_hard_cooldown
                    if detect_hard_cooldown(full_err):
                        is_hard_limit = True
                except Exception:
                    pass

                # If hard limit, try switching to another account immediately
                if is_hard_limit and self._pool:
                    next_acct = self._pool.best_account(model)
                    if next_acct and next_acct.label not in tried_accounts:
                        self._logger.info(
                            f"Hard rate limit on '{current_account}', "
                            f"switching to '{next_acct.label}'"
                        )
                        current_account = next_acct.label
                        continue

                # Only retry transient connection errors, NOT timeouts.
                is_transient = any(k in err_str for k in [
                    "connect", "reset", "refused",
                    "temporarily", "429", "503", "network",
                    "exhausted", "concurrency",
                ]) and "timeout" not in err_str

                if is_transient and attempt < max_retries:
                    # Try switching account for concurrency/exhausted errors
                    if ("exhausted" in err_str or "concurrency" in err_str) and self._pool:
                        next_acct = self._pool.best_account(model)
                        if next_acct and next_acct.label not in tried_accounts:
                            self._logger.info(
                                f"Concurrency limit on '{current_account}', "
                                f"switching to '{next_acct.label}'"
                            )
                            current_account = next_acct.label
                            wait = 5  # shorter wait since we're switching accounts
                        else:
                            wait = 30 * attempt  # 30s, 60s
                    else:
                        wait = self._cooldown_sec * attempt * 2

                    self._logger.warning(
                        f"Transient error on '{current_account}' (attempt {attempt}/{max_retries}), "
                        f"retrying in {wait}s: {e}"
                    )
                    gc.collect()
                    await asyncio.sleep(wait)
                    continue

                self._logger.error(f"webchat_call failed on '{current_account}': {e}")
                gc.collect()
                raise ProviderResponseError(
                    provider_name=self.name,
                    status_code=500,
                    response_body=str(e)
                )

        # Should not reach here, but safety net
        gc.collect()
        raise ProviderResponseError(
            provider_name=self.name,
            status_code=500,
            response_body=str(last_error) if last_error else "Unknown error after retries"
        )

    async def _stream_request(
        self,
        messages: list[Message],
        model: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """WebChat لا يدعم التدفق — نستخدم الطلب العادي"""
        response = await self._make_request(messages, model, **kwargs)
        content = response["choices"][0]["message"]["content"]
        yield content

    def _calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """WebChat مجاني"""
        return 0.0


# =============================================================================
# Convenience Functions
# =============================================================================

def create_kimi_provider(
    model: str = "kimi-k2.5-thinking",
    account: str = "default",
    timeout: int = 120,
) -> WebChatBridgeProvider:
    """إنشاء مزود Kimi مع دعم الحسابات المتعددة.
    
    Auto-discovers all authenticated Kimi accounts and uses 
    AccountPool for smart rotation between them.
    """
    provider = WebChatBridgeProvider(
        webchat_provider="kimi",
        account=account,
        timeout=timeout,
    )
    provider.default_model = model
    return provider


def create_glm_provider(
    account: str = "default",
    timeout: int = 120,
) -> WebChatBridgeProvider:
    """إنشاء مزود GLM"""
    return WebChatBridgeProvider(
        webchat_provider="glm",
        account=account,
        timeout=timeout,
    )


def create_deepseek_provider(
    account: str = "default",
    timeout: int = 120,
) -> WebChatBridgeProvider:
    """إنشاء مزود DeepSeek"""
    return WebChatBridgeProvider(
        webchat_provider="deepseek",
        account=account,
        timeout=timeout,
    )
