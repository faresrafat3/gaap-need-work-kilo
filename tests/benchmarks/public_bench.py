"""
Public Benchmarks Runner
========================

Runs small samples from public multiple-choice benchmarks using:
- Direct Gemini API (baseline)
- GAAP system (with routing/guardrails)

Datasets used (HuggingFace):
- ai2_arc (ARC-Easy, ARC-Challenge)
- openbookqa (main)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import re
import statistics
import time
from typing import Any

from datasets import load_dataset
from detailed_logger import DetailedBenchmarkLogger, QuestionLog
from gaap_system_glm5 import GAAPRequest, create_engine

# Ensure package import works when running from file path
from gaap_system_glm5.env import load_env
from gaap_system_glm5.providers.free_tier.groq_provider import GeminiProvider

from gaap.core.exceptions import ProviderRateLimitError, ProviderResponseError
from gaap.core.types import Message, MessageRole, TaskType
from gaap.layers.layer2_tactical import AtomicTask, TaskCategory

DATASETS = {
    "arc_easy": ("ai2_arc", "ARC-Easy", "train"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "train"),
    "openbookqa": ("openbookqa", "main", "train"),
    "mmlu": ("cais/mmlu", "all", "test"),
}


def _extract_choices_arc(example: dict[str, Any]) -> tuple[str, list[tuple[str, str]], str]:
    question = example.get("question", "").strip()
    choices = example.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    answer = example.get("answerKey", "").strip()
    return question, pairs, answer


def _extract_choices_openbookqa(example: dict[str, Any]) -> tuple[str, list[tuple[str, str]], str]:
    question = example.get("question_stem", "").strip()
    choices = example.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    answer = example.get("answerKey", "").strip()
    return question, pairs, answer


def _extract_choices_mmlu(example: dict[str, Any]) -> tuple[str, list[tuple[str, str]], str]:
    """Extract MMLU multiple choice question"""
    question = example.get("question", "").strip()
    choices_list = example.get("choices", [])
    # MMLU uses 0-3 for answers, convert to A-D
    answer_idx = example.get("answer", 0)
    answer = chr(65 + answer_idx)  # 0->A, 1->B, etc

    labels = ["A", "B", "C", "D"]
    pairs = list(zip(labels[: len(choices_list)], choices_list))
    return question, pairs, answer


def _build_prompt(question: str, choices: list[tuple[str, str]]) -> str:
    lines = [
        "Choose the correct answer. Reply with ONLY the letter (A, B, C, or D).",
        "",
        f"Question: {question}",
        "",
        "Options:",
    ]
    for label, text in choices:
        lines.append(f"{label}. {text}")
    return "\n".join(lines)


def _parse_letter(text: str) -> str:
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""


def _metrics(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
    return {
        "avg": round(statistics.mean(latencies), 2),
        "p50": round(p50, 2),
        "p95": round(p95, 2),
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
    }


async def _run_direct(provider: GeminiProvider, prompt: str) -> tuple[str, float, float, int]:
    start = time.time()
    response = await provider.chat_completion(
        [
            Message(role=MessageRole.SYSTEM, content="Answer multiple-choice questions."),
            Message(role=MessageRole.USER, content=prompt),
        ],
        model=provider.default_model,
        temperature=0.0,
        max_tokens=16,
    )
    latency_ms = (time.time() - start) * 1000
    content = response.choices[0].message.content if response.choices else ""
    cost = response.metadata.get("cost_usd", 0.0)
    tokens = response.usage.total_tokens if response.usage else 0
    return content, latency_ms, cost, tokens


async def _run_gaap(engine, prompt: str) -> tuple[str, float, float, int]:
    start = time.time()
    response = await engine.process(GAAPRequest(text=prompt, budget_limit=2.0))
    latency_ms = (time.time() - start) * 1000
    content = response.output or ""
    return content, latency_ms, response.total_cost_usd, response.total_tokens


async def _run_gaap_direct(engine, prompt: str) -> tuple[str, float, float, int]:
    start = time.time()
    task = AtomicTask(
        id=f"bench_{int(time.time()*1000)}",
        name="Benchmark Direct",
        description=prompt,
        category=TaskCategory.SETUP,
        type=TaskType.ANALYSIS,
    )
    result = await engine.layer3.process(task)
    latency_ms = (time.time() - start) * 1000
    return result.output or "", latency_ms, result.cost_usd, result.tokens_used


async def _with_rate_limit_retry(
    call, max_attempts: int = 25, logger=None, context: str = ""
) -> tuple[str, float, float, int]:
    """Retry with exponential backoff for rate limits"""
    attempt = 0
    base_delay = 6.0

    while True:
        try:
            return await call()
        except ProviderRateLimitError as exc:
            attempt += 1
            if attempt >= max_attempts:
                if logger:
                    logger.log_event(
                        "max_retries_exceeded", {"context": context, "attempts": attempt}
                    )
                raise

            details = getattr(exc, "details", {}) or {}
            retry_after = details.get("retry_after_seconds")

            # Use exponential backoff after 5 attempts
            if attempt > 5:
                backoff_delay = base_delay * (1.5 ** (attempt - 5))
                delay = max(float(retry_after) if retry_after else base_delay, backoff_delay)
            else:
                delay = float(retry_after) if retry_after else base_delay

            # Cap at 120 seconds
            delay = min(delay, 120.0)

            if logger:
                logger.log_event(
                    "rate_limit_retry",
                    {
                        "context": context,
                        "attempt": attempt,
                        "delay": delay,
                        "reason": "ProviderRateLimitError",
                        "using_exponential_backoff": attempt > 5,
                    },
                )

            await asyncio.sleep(delay)

        except ProviderResponseError as exc:
            details = getattr(exc, "details", {}) or {}
            status = details.get("status_code")
            if status != 429:
                raise

            attempt += 1
            if attempt >= max_attempts:
                if logger:
                    logger.log_event(
                        "max_retries_exceeded", {"context": context, "attempts": attempt}
                    )
                raise

            response_text = details.get("response", "")
            match = re.search(r"retry in ([0-9.]+)s", response_text)
            base_retry = float(match.group(1)) if match else base_delay

            # Apply exponential backoff after 5 attempts
            if attempt > 5:
                backoff_delay = base_retry * (1.5 ** (attempt - 5))
                delay = max(base_retry, backoff_delay)
            else:
                delay = base_retry

            # Cap at 120 seconds
            delay = min(delay, 120.0)

            if logger:
                logger.log_event(
                    "rate_limit_retry",
                    {
                        "context": context,
                        "attempt": attempt,
                        "delay": delay,
                        "reason": "429_response",
                        "retry_after": base_retry,
                        "using_exponential_backoff": attempt > 5,
                    },
                )

            await asyncio.sleep(delay + 1.0)


def _apply_full_lite_settings(engine) -> None:
    # Lighter settings for full pipeline to avoid OOM in small benchmark tasks.
    engine.layer1.tot_depth = 3
    engine.layer1.tot_branching = 2
    engine.layer1.mad_rounds = 1

    engine.layer2.max_subtasks = 12
    engine.layer2.max_parallel = 2

    engine.layer3.enable_twin = False
    engine.layer3.max_parallel = 2


async def run_benchmark(
    dataset_key: str, samples: int, seed: int, gaap_mode: str
) -> dict[str, Any]:
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    load_env()

    gemini_keys_raw = os.environ.get("GEMINI_API_KEYS", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY")
    gemini_keys = [k.strip() for k in gemini_keys_raw.split(",") if k.strip()]

    if not gemini_keys and not gemini_key:
        raise RuntimeError("Missing GEMINI_API_KEYS or GEMINI_API_KEY in environment.")

    # Initialize detailed logger
    logger = DetailedBenchmarkLogger(output_dir="./benchmark_logs")
    logger.configure(
        dataset=dataset_key,
        samples=samples,
        gaap_mode=gaap_mode,
        seed=seed,
        provider="gemini",
        model="gemini-2.5-flash",
        num_api_keys=len(gemini_keys),
        enable_all=(gaap_mode != "direct_layer3"),
        budget=10.0,
    )
    logger.metadata.total_questions = samples

    dataset_name, subset, split = DATASETS[dataset_key]
    data = load_dataset(dataset_name, subset, split=split)

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[:samples]

    provider = None
    engine = None

    try:
        logger.log_event("initialization_start", {"dataset": dataset_key, "samples": samples})

        provider = GeminiProvider(api_key=gemini_key, api_keys=gemini_keys)
        logger.log_event("provider_created", {"num_keys": len(gemini_keys)})

        if gaap_mode == "direct_layer3":
            engine = create_engine(
                gemini_api_key=gemini_key,
                gemini_api_keys=gemini_keys,
                budget=10.0,
                enable_all=False,
            )
            logger.log_event("engine_created", {"mode": "direct_layer3"})
        else:
            engine = create_engine(
                gemini_api_key=gemini_key,
                gemini_api_keys=gemini_keys,
                budget=10.0,
                enable_all=True,
            )
            logger.log_event("engine_created", {"mode": gaap_mode, "enable_all": True})

            if gaap_mode == "full_lite":
                _apply_full_lite_settings(engine)
                logger.configure(
                    layer_config={
                        "layer1": {"tot_depth": 3, "tot_branching": 2, "mad_rounds": 1},
                        "layer2": {"max_subtasks": 12, "max_parallel": 2},
                        "layer3": {"enable_twin": False, "max_parallel": 2},
                    }
                )

        direct_correct = 0
        gaap_correct = 0
        direct_lat = []
        gaap_lat = []
        direct_cost = 0.0
        gaap_cost = 0.0
        direct_tokens = 0
        gaap_tokens = 0

        logger.log_event("benchmark_start", {"total_questions": len(indices)})

        for question_num, idx in enumerate(indices, start=1):
            logger.log_event("question_start", {"question_num": question_num, "dataset_idx": idx})
            example = data[idx]

            if dataset_key.startswith("arc"):
                question, choices, answer = _extract_choices_arc(example)
            elif dataset_key == "openbookqa":
                question, choices, answer = _extract_choices_openbookqa(example)
            elif dataset_key == "mmlu":
                question, choices, answer = _extract_choices_mmlu(example)
            else:
                raise ValueError(f"Unsupported dataset: {dataset_key}")

            prompt = _build_prompt(question, choices)

            # Track errors and retries for this question
            question_errors = []
            question_warnings = []
            direct_attempts = 0
            gaap_attempts = 0

            # Direct
            try:
                logger.log_event("direct_start", {"question_num": question_num})
                direct_attempts_start = 0

                async def direct_with_tracking():
                    nonlocal direct_attempts
                    direct_attempts += 1
                    return await _run_direct(provider, prompt)

                direct_text, d_lat, d_cost, d_tokens = await _with_rate_limit_retry(
                    direct_with_tracking, logger=logger, context=f"Q{question_num}_direct"
                )
                direct_lat.append(d_lat)
                direct_cost += d_cost
                direct_tokens += d_tokens
                direct_is_correct = _parse_letter(direct_text) == answer
                if direct_is_correct:
                    direct_correct += 1

                logger.log_event(
                    "direct_complete",
                    {
                        "question_num": question_num,
                        "attempts": direct_attempts,
                        "correct": direct_is_correct,
                        "latency_ms": d_lat,
                    },
                )
            except Exception as e:
                question_errors.append(f"Direct error: {str(e)}")
                logger.log_error(e, f"Question {question_num} (Direct)")
                direct_text = ""
                d_lat = 0.0
                d_cost = 0.0
                d_tokens = 0
                direct_is_correct = False

            # GAAP
            try:
                logger.log_event("gaap_start", {"question_num": question_num})

                if gaap_mode == "direct_layer3":
                    gaap_call_base = lambda: _run_gaap_direct(engine, prompt)
                else:
                    gaap_call_base = lambda: _run_gaap(engine, prompt)

                async def gaap_with_tracking():
                    nonlocal gaap_attempts
                    gaap_attempts += 1
                    return await gaap_call_base()

                gaap_text, g_lat, g_cost, g_tokens = await _with_rate_limit_retry(
                    gaap_with_tracking, logger=logger, context=f"Q{question_num}_gaap"
                )
                gaap_lat.append(g_lat)
                gaap_cost += g_cost
                gaap_tokens += g_tokens
                gaap_is_correct = _parse_letter(gaap_text) == answer
                if gaap_is_correct:
                    gaap_correct += 1

                logger.log_event(
                    "gaap_complete",
                    {
                        "question_num": question_num,
                        "attempts": gaap_attempts,
                        "correct": gaap_is_correct,
                        "latency_ms": g_lat,
                    },
                )
            except Exception as e:
                question_errors.append(f"GAAP error: {str(e)}")
                logger.log_error(e, f"Question {question_num} (GAAP)")
                gaap_text = ""
                g_lat = 0.0
                g_cost = 0.0
                g_tokens = 0
                gaap_is_correct = False

            # Log complete question
            question_log = QuestionLog(
                question_id=question_num,
                dataset=dataset_key,
                question_text=question,
                choices=choices,
                correct_answer=answer,
                direct_answer=_parse_letter(direct_text),
                direct_correct=direct_is_correct,
                direct_latency_ms=d_lat,
                direct_cost_usd=d_cost,
                direct_tokens=d_tokens,
                direct_attempts=direct_attempts,
                gaap_answer=_parse_letter(gaap_text),
                gaap_correct=gaap_is_correct,
                gaap_latency_ms=g_lat,
                gaap_cost_usd=g_cost,
                gaap_tokens=g_tokens,
                gaap_attempts=gaap_attempts,
                errors=question_errors,
                warnings=question_warnings,
            )
            logger.log_question(question_log)

            print(
                f"Q{question_num}/{samples}: Direct={'✅' if direct_is_correct else '❌'} GAAP={'✅' if gaap_is_correct else '❌'} (D:{direct_attempts} G:{gaap_attempts} attempts)"
            )

            # Larger delay between questions to avoid exhausting all keys
            # With 7 keys at 20 RPM each, we need ~9 seconds per question minimum
            # Using 12s to be safe + small jitter to avoid thundering herd
            jitter = random.uniform(0, 2.0)
            delay = 12.0 + jitter
            logger.log_event("inter_question_delay", {"delay": delay, "question": question_num})
            await asyncio.sleep(delay)

        logger.log_event(
            "benchmark_complete",
            {"direct_correct": direct_correct, "gaap_correct": gaap_correct, "total": samples},
        )

        result = {
            "dataset": dataset_key,
            "samples": samples,
            "direct": {
                "accuracy": round(direct_correct / max(samples, 1), 4),
                "latency_ms": _metrics(direct_lat),
                "total_cost_usd": round(direct_cost, 6),
                "total_tokens": direct_tokens,
            },
            "gaap": {
                "accuracy": round(gaap_correct / max(samples, 1), 4),
                "latency_ms": _metrics(gaap_lat),
                "total_cost_usd": round(gaap_cost, 6),
                "total_tokens": gaap_tokens,
            },
        }

        # Finalize logger and generate reports
        logger.finalize()
        print(f"\n📊 Detailed logs saved to: {logger.run_dir}")
        print(f"📁 Run ID: {logger.run_id}")

        return result
    except Exception as e:
        logger.log_error(e, "benchmark_run")
        raise
    finally:
        logger.log_event("cleanup_start", {})

        if engine is not None:
            for prov in getattr(engine, "providers", []):
                close_fn = getattr(prov, "close", None)
                if callable(close_fn):
                    try:
                        await close_fn()
                    except Exception as cleanup_err:
                        logger.log_error(cleanup_err, "provider_cleanup")
            engine.shutdown()

        if provider is not None:
            await provider.close()

        logger.log_event("cleanup_complete", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="GAAP Public Benchmarks")
    parser.add_argument("--dataset", default="arc_easy", choices=DATASETS.keys())
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gaap-mode",
        choices=["direct_layer3", "full", "full_lite"],
        default="direct_layer3",
        help="direct_layer3 reduces memory/CPU; full_lite runs full pipeline with lighter settings",
    )

    args = parser.parse_args()
    result = asyncio.run(run_benchmark(args.dataset, args.samples, args.seed, args.gaap_mode))
    print(result)


if __name__ == "__main__":
    main()
