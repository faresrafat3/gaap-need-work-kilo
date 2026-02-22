# TECHNICAL SPECIFICATION: Providers Evolution (Async & Native Streaming)

**Target:** `gaap/providers/`
**Auditor:** Strategic Architect
**Status:** REQUIRED FOR IMPLEMENTATION

## 1. Core Architectural Flaws
- **Blocking I/O:** Heavy reliance on `asyncio.to_thread` for sync providers (Kimi, DeepSeek Web).
- **Inert Streaming:** The `_stream_request` implementation yields the full buffer at once, losing the UX benefits of real-time tokens.
- **Maintenance Fragility:** Web-based scrapers are tightly coupled with provider UI structures.

## 2. Refactoring Requirements

### 2.1 Full Async Conversion
Migrate `UnifiedProvider` and `WebChatProvider` to native `asyncio`.
- **Action:** Replace `curl_cffi.requests` with `curl_cffi.requests.AsyncSession`.
- **Action:** Change `call()` to `async def call()`.
- **Benefit:** Allows hundreds of concurrent model calls with minimal RAM overhead.

### 2.2 Native SSE Streaming
Implement true word-by-word streaming for all providers.
- **WebChat Implementation:** Use the `stream=True` flag in `AsyncSession` and parse the SSE chunks (`data: ...`) in real-time.
- **Interface:** `async for chunk in provider.stream_chat_completion(...)`.

### 2.3 Provider-Level Prompt Caching
Integrate context caching features for supported providers (Anthropic, DeepSeek API).
- **Logic:** Detect repeating system prompts or large context blocks and enable the `cache_control` flags to save up to 90% on input tokens.

### 2.4 Structured Tool Calling (Native)
Move away from string-based tool extraction.
- **Requirement:** Implement `tools` parameter in `BaseProvider.chat_completion`.
- **Implementation:** Subclasses (Gemini, Groq) should map the GAAP tool registry to their native JSON schema requirements.

## 3. Implementation Steps
1.  **Refactor** `gaap/providers/base_provider.py` to make `_make_request` an async method.
2.  **Update** `gaap/providers/webchat_providers.py` to use `AsyncSession`.
3.  **Enable** the `streaming` loop in `Layer3_Execution` to consume tokens as they arrive.

---
**Handover Status:** Ready. Code Agent must prioritize the Async migration to enable Swarm scalability.
