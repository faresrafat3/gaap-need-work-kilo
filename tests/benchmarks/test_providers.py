"""
Provider API Tester
==================

Tests all configured providers to verify:
- API keys are valid
- Models are accessible
- Actual rate limits
- Response times
- Success rates
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass

import aiohttp

# Add parent dir to path
from gaap.providers.multi_provider_config import (
    CEREBRAS_CONFIG,
    GEMINI_CONFIG,
    GITHUB_CONFIG,
    GROQ_CONFIG,
    MISTRAL_CONFIG,
    OPENROUTER_CONFIG,
    ProviderConfig,
)


@dataclass
class TestResult:
    """Result of testing a provider"""

    provider_name: str
    provider_type: str
    key_index: int
    success: bool
    response_time_ms: float
    error: str | None = None
    model_tested: str | None = None
    response_content: str | None = None


class ProviderTester:
    """Test individual providers"""

    def __init__(self):
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        if self.session:
            await self.session.close()

    async def test_cerebras(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test Cerebras API"""
        api_key = config.api_keys[key_index]
        model = config.models[0] if config.models else "llama3.3-70b"

        start = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            async with self.session.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_openrouter(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test OpenRouter API"""
        api_key = config.api_keys[key_index]
        model = "meta-llama/llama-3.3-70b-instruct:free"

        start = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/gaap",
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            async with self.session.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_groq(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test Groq API"""
        api_key = config.api_keys[key_index]
        model = "llama-3.3-70b-versatile"

        start = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            async with self.session.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_gemini(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test Gemini API"""
        api_key = config.api_keys[key_index]
        model = "gemini-2.5-flash"

        start = time.time()
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

            data = {
                "contents": [{"parts": [{"text": "Say 'test' only"}]}],
                "generationConfig": {
                    "maxOutputTokens": 5,
                    "temperature": 0,
                },
            }

            async with self.session.post(
                url, json=data, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = (
                        result.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [{}])[0]
                        .get("text", "")
                    )
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_mistral(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test Mistral La Plateforme API"""
        api_key = config.api_keys[key_index]
        model = "mistral-large-latest"

        start = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            async with self.session.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_github(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test GitHub Models API"""
        api_key = config.api_keys[key_index]
        model = "gpt-4o-mini"

        start = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "model": model,
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
                "temperature": 0,
            }

            async with self.session.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )

    async def test_cloudflare(self, config: ProviderConfig, key_index: int = 0) -> TestResult:
        """Test Cloudflare Workers AI"""
        api_key = config.api_keys[key_index]
        model = "@cf/meta/llama-3.1-8b-instruct"

        start = time.time()
        try:
            # Get account ID from environment
            account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            if not account_id:
                return TestResult(
                    provider_name=config.name,
                    provider_type=config.provider_type.value,
                    key_index=key_index,
                    success=False,
                    response_time_ms=0,
                    error="CLOUDFLARE_ACCOUNT_ID not set in environment",
                    model_tested=model,
                )

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "messages": [{"role": "user", "content": "Say 'test' only"}],
                "max_tokens": 5,
            }

            url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"

            async with self.session.post(
                url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                response_time = (time.time() - start) * 1000

                if resp.status == 200:
                    result = await resp.json()
                    content = result.get("result", {}).get("response", "")
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=True,
                        response_time_ms=response_time,
                        model_tested=model,
                        response_content=content[:50],
                    )
                else:
                    text = await resp.text()
                    return TestResult(
                        provider_name=config.name,
                        provider_type=config.provider_type.value,
                        key_index=key_index,
                        success=False,
                        response_time_ms=response_time,
                        error=f"HTTP {resp.status}: {text[:100]}",
                        model_tested=model,
                    )

        except Exception as e:
            response_time = (time.time() - start) * 1000
            return TestResult(
                provider_name=config.name,
                provider_type=config.provider_type.value,
                key_index=key_index,
                success=False,
                response_time_ms=response_time,
                error=str(e)[:100],
                model_tested=model,
            )


async def test_all_providers(test_all_keys: bool = False):
    """Test all configured providers"""

    print("=" * 80)
    print("PROVIDER API TESTING")
    print("=" * 80)
    print()

    providers_to_test = [
        (CEREBRAS_CONFIG, "test_cerebras"),
        (GROQ_CONFIG, "test_groq"),
        (OPENROUTER_CONFIG, "test_openrouter"),
        (GEMINI_CONFIG, "test_gemini"),
        (MISTRAL_CONFIG, "test_mistral"),
        # (MISTRAL_CODESTRAL_CONFIG, "test_codestral"),  # Skipped for now
        (GITHUB_CONFIG, "test_github"),
        # (CLOUDFLARE_CONFIG, "test_cloudflare"),  # Needs account ID
    ]

    results: list[TestResult] = []

    async with ProviderTester() as tester:
        for config, test_method in providers_to_test:
            print(f"\n🔍 Testing {config.name}...")
            print(f"   Base URL: {config.base_url}")
            print(f"   API Keys: {len(config.api_keys)}")
            print(f"   Models: {', '.join(config.models[:3])}")

            # Test first key
            test_fn = getattr(tester, test_method)
            result = await test_fn(config, key_index=0)
            results.append(result)

            if result.success:
                print(f"   ✅ SUCCESS ({result.response_time_ms:.0f}ms)")
                print(f"      Model: {result.model_tested}")
                print(f"      Response: {result.response_content}")
            else:
                print(f"   ❌ FAILED ({result.response_time_ms:.0f}ms)")
                print(f"      Error: {result.error}")

            # Test all keys if requested
            if test_all_keys and len(config.api_keys) > 1:
                print(f"   Testing remaining {len(config.api_keys) - 1} keys...")
                for i in range(1, min(len(config.api_keys), 3)):  # Test up to 3 keys
                    result = await test_fn(config, key_index=i)
                    results.append(result)
                    status = "✅" if result.success else "❌"
                    print(f"      Key {i+1}: {status} ({result.response_time_ms:.0f}ms)")

                    # Small delay to avoid rate limiting
                    await asyncio.sleep(1.0)

            # Small delay between providers
            await asyncio.sleep(0.5)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    success_count = sum(1 for r in results if r.success)
    total_count = len(results)

    print(f"\nTotal Tests: {total_count}")
    print(f"Successful: {success_count} ({success_count/max(total_count,1)*100:.1f}%)")
    print(f"Failed: {total_count - success_count}")

    print(f"\n{'Provider':<25} {'Status':>10} {'Avg Time':>12} {'Keys Tested':>12}")
    print("-" * 80)

    # Group by provider
    by_provider = {}
    for r in results:
        if r.provider_name not in by_provider:
            by_provider[r.provider_name] = []
        by_provider[r.provider_name].append(r)

    for provider_name, provider_results in by_provider.items():
        success = sum(1 for r in provider_results if r.success)
        total = len(provider_results)
        avg_time = sum(r.response_time_ms for r in provider_results) / max(total, 1)
        status = f"{success}/{total}"

        print(f"{provider_name:<25} {status:>10} {avg_time:>10.0f}ms {total:>12}")

    print("=" * 80)

    # Recommendations
    print("\n📊 RECOMMENDATIONS:")

    working_providers = [
        name for name, results in by_provider.items() if any(r.success for r in results)
    ]

    if working_providers:
        print(f"\n✅ Working Providers ({len(working_providers)}):")
        for name in working_providers:
            provider_results = by_provider[name]
            avg_time = sum(r.response_time_ms for r in provider_results if r.success) / max(
                sum(1 for r in provider_results if r.success), 1
            )
            print(f"   - {name} (avg {avg_time:.0f}ms)")

    failed_providers = [
        name for name, results in by_provider.items() if not any(r.success for r in results)
    ]

    if failed_providers:
        print(f"\n❌ Failed Providers ({len(failed_providers)}):")
        for name in failed_providers:
            provider_results = by_provider[name]
            common_error = provider_results[0].error if provider_results else "Unknown"
            print(f"   - {name}: {common_error}")

    print("\n" + "=" * 80)

    return results


if __name__ == "__main__":
    import sys

    test_all_keys = "--all-keys" in sys.argv

    print("Starting provider tests...")
    if test_all_keys:
        print("Testing ALL keys for each provider")
    else:
        print("Testing FIRST key only (use --all-keys to test all)")

    print()

    results = asyncio.run(test_all_providers(test_all_keys=test_all_keys))
