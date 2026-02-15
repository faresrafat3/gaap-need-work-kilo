"""
Unified Free Benchmark Runner
=============================

Benchmarks models using a unified primary+fallback provider chain
(g4f + optional registered free API providers).

Supported Datasets:
  - mmlu        : 4-option multiple choice, general knowledge (14K questions)
  - mmlu_pro    : 10-option multiple choice, requires reasoning (12K questions)
  - arc_easy    : Science questions (easy)
  - arc_challenge: Science questions (hard)
  - openbookqa  : Open book science QA

Default profile: quality (highest capability first)
Other profiles: balanced, throughput

MMLU-Pro specifics:
  - 10 options (A-J) instead of 4 (A-D)
  - Chain-of-Thought (CoT) prompting for reasoning
  - GPT-4o scores ~72.6% (vs 88.7% on MMLU) → much harder
  - 14 categories: biology, business, chemistry, CS, economics, 
    engineering, health, history, law, math, other, philosophy, physics, psychology
"""

import sys
import os

import argparse
import random
import re
import time
from typing import Dict, Any, List, Tuple
from datasets import load_dataset

from gaap_system_glm5.benchmarks.detailed_logger import DetailedBenchmarkLogger, QuestionLog
from gaap_system_glm5.providers.unified_provider import (
    UnifiedProvider, check_lmarena_auth, warmup_lmarena_auth, invalidate_lmarena_cache,
)


# ============================================================================
# Dataset Handlers
# ============================================================================

DATASETS = {
    "arc_easy": ("ai2_arc", "ARC-Easy", "train"),
    "arc_challenge": ("ai2_arc", "ARC-Challenge", "train"),
    "openbookqa": ("openbookqa", "main", "train"),
    "mmlu": ("cais/mmlu", "all", "test"),
    "mmlu_pro": ("TIGER-Lab/MMLU-Pro", None, "test"),
}


def _extract_choices_arc(example):
    question = example.get("question", "").strip()
    choices = example.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    answer = example.get("answerKey", "").strip()
    return question, pairs, answer


def _extract_choices_mmlu(example):
    question = example.get("question", "").strip()
    choices_list = example.get("choices", [])
    answer_idx = example.get("answer", 0)
    answer = chr(65 + answer_idx)
    labels = ["A", "B", "C", "D"]
    pairs = list(zip(labels[:len(choices_list)], choices_list))
    return question, pairs, answer


def _extract_choices_openbookqa(example):
    question = example.get("question_stem", "").strip()
    choices = example.get("choices", {})
    labels = choices.get("label", [])
    texts = choices.get("text", [])
    pairs = list(zip(labels, texts))
    pairs.sort(key=lambda x: x[0])
    answer = example.get("answerKey", "").strip()
    return question, pairs, answer


def _extract_choices_mmlu_pro(example):
    """MMLU-Pro: 10 options (A-J), harder questions requiring reasoning."""
    question = example.get("question", "").strip()
    options_list = example.get("options", [])
    answer = example.get("answer", "").strip()  # Already a letter (A-J)
    labels = [chr(65 + i) for i in range(len(options_list))]  # A, B, C, ... J
    pairs = list(zip(labels, options_list))
    return question, pairs, answer


def extract_choices(dataset_key, example):
    if dataset_key.startswith("arc"):
        return _extract_choices_arc(example)
    elif dataset_key == "openbookqa":
        return _extract_choices_openbookqa(example)
    elif dataset_key == "mmlu":
        return _extract_choices_mmlu(example)
    elif dataset_key == "mmlu_pro":
        return _extract_choices_mmlu_pro(example)
    raise ValueError(f"Unknown dataset: {dataset_key}")


def build_prompt(question: str, choices: List[Tuple[str, str]], use_cot: bool = False) -> str:
    """Build prompt. use_cot=True for MMLU-Pro style Chain-of-Thought."""
    num_options = len(choices)
    last_letter = choices[-1][0]  # e.g. 'D' or 'J'
    
    if use_cot:
        lines = [
            f"Answer the following multiple-choice question. There are {num_options} options (A through {last_letter}).",
            "Think step by step, then give your final answer as: Answer: X (where X is the letter).",
            "",
            f"Question: {question}",
            "",
            "Options:",
        ]
    else:
        lines = [
            f"Choose the correct answer. Reply with ONLY the letter (A through {last_letter}).",
            "",
            f"Question: {question}",
            "",
            "Options:",
        ]
    for label, text in choices:
        lines.append(f"{label}. {text}")
    return "\n".join(lines)


def parse_letter(text: str, max_letter: str = "D") -> str:
    """Parse answer letter from model response. Supports A-D (MMLU) or A-J (MMLU-Pro)."""
    valid_range = f"A-{max_letter}"
    
    # First try explicit "Answer: X" pattern (CoT responses)
    match = re.search(r"(?:answer|Answer|ANSWER)\s*[:=]\s*\(?([" + valid_range + r"])\)?", text)
    if match:
        return match.group(1).upper()
    
    # Try "The answer is X" pattern
    match = re.search(r"the answer is\s*\(?([" + valid_range + r"])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: last standalone letter in valid range
    matches = re.findall(r"\b([" + valid_range + r"])\b", text.upper())
    if matches:
        return matches[-1]
    
    return ""


def run_preflight(provider: UnifiedProvider, timeout: int = 20) -> Dict[str, Any]:
    """Quick health check for each slot in the provider chain."""
    prompt = "Reply with only: OK"
    results = []
    print("\n🩺 Running provider preflight...")
    for slot in provider.chain:
        if not slot.enabled:
            results.append({"slot": slot.name, "ok": False, "reason": "disabled"})
            print(f"  ⏸️  {slot.name:<35} disabled")
            continue
        start = time.time()
        try:
            content, latency_ms = provider._call_slot(slot, prompt, timeout=timeout)
            ok = content.strip().upper().startswith("OK")
            elapsed = (time.time() - start) * 1000
            results.append({"slot": slot.name, "ok": ok, "latency_ms": latency_ms})
            mark = "✅" if ok else "⚠️"
            print(f"  {mark} {slot.name:<35} {elapsed:.0f}ms")
        except Exception as e:
            provider._apply_cooldown(slot, e)
            elapsed = (time.time() - start) * 1000
            msg = str(e)[:80]
            results.append({"slot": slot.name, "ok": False, "latency_ms": elapsed, "reason": msg})
            print(f"  ❌ {slot.name:<35} {elapsed:.0f}ms  {msg}")
    healthy = sum(1 for x in results if x.get("ok"))
    print(f"\n🩺 Preflight summary: {healthy}/{len(results)} healthy slots\n")
    return {"healthy": healthy, "total": len(results), "results": results}


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(
    dataset_key: str,
    samples: int,
    seed: int,
    delay: float = 4.0,
    category: str = None,
    profile: str = "quality",
    model: str = None,
) -> Dict[str, Any]:
    """
    Run benchmark using Unified Provider (Primary + Fallback).
    
    Args:
        dataset_key: Which dataset to use
        samples: Number of questions
        seed: Random seed for reproducibility
        delay: Min seconds between requests (rate limit protection)
        category: Filter by category (MMLU-Pro only)
        profile: Provider strategy profile (premium|quality|balanced|throughput)
        model: Force a specific model (creates a single-slot chain via LMArena)
    """
    
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_key}. Available: {list(DATASETS.keys())}")
    
    # Load dataset
    print(f"📦 Loading dataset: {dataset_key}...")
    dataset_name, subset, split = DATASETS[dataset_key]
    if subset is not None:
        data = load_dataset(dataset_name, subset, split=split)
    else:
        data = load_dataset(dataset_name, split=split)
    
    # Category filter (MMLU-Pro)
    if category and 'category' in data.column_names:
        data = data.filter(lambda x: x['category'] == category)
        print(f"🏷️  Category filter: {category} ({len(data)} questions)")
        if len(data) == 0:
            cats = sorted(set(load_dataset(dataset_name, split=split)['category']))
            raise ValueError(f"No questions for category '{category}'. Available: {cats}")
    
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    indices = indices[:samples]
    
    # Initialize Unified Provider (Primary + Fallback)
    if model:
        # Single-model mode: create a chain with just the requested model
        from gaap_system_glm5.providers.unified_provider import ModelSlot, BackendType, build_default_chain
        forced_slot = ModelSlot(
            name=f"{model} (LMArena)",
            model_id=model,
            backend=BackendType.G4F_PROVIDER,
            g4f_provider="LMArena",
            rpm_per_key=3,
            context_window=128_000,
            priority=999,
        )
        # Put the forced model first, then the rest of the chain as fallback
        fallback_chain = build_default_chain(profile=profile)
        provider = UnifiedProvider(
            chain=[forced_slot] + fallback_chain,
            min_delay=delay, verbose=True, profile=profile,
        )
        print(f"\n🎯 Forced model: {model} via LMArena")
    else:
        provider = UnifiedProvider(min_delay=delay, verbose=True, profile=profile)
    print(provider.get_config_summary())
    
    print(f"📊 Questions: {samples}")
    est_time = samples * (delay + 8)
    print(f"⏳ Estimated time: {est_time/60:.1f} minutes")
    
    # Initialize detailed logger
    logger = DetailedBenchmarkLogger(output_dir="./benchmark_logs")
    logger.configure(
        dataset=dataset_key,
        samples=samples,
        gaap_mode="unified_fallback",
        seed=seed,
        provider="unified",
        model=provider.primary.name,
        num_api_keys=sum(len(s.api_keys) for s in provider.chain),
        enable_all=False,
        budget=0.0,
    )
    logger.metadata.total_questions = samples
    
    # Run benchmark
    correct = 0
    total = 0
    errors = 0
    all_latencies = []
    model_correct = {}
    model_total = {}
    
    start_time = time.time()
    
    for q_num, idx in enumerate(indices, start=1):
        example = data[idx]
        question, choices, answer = extract_choices(dataset_key, example)
        use_cot = (dataset_key == "mmlu_pro")
        prompt = build_prompt(question, choices, use_cot=use_cot)
        max_letter = choices[-1][0] if choices else "D"
        
        try:
            content, model_used, latency_ms = provider.call(prompt)
            tokens = len(prompt.split()) + len(content.split())
            parsed = parse_letter(content, max_letter=max_letter)
            is_correct = (parsed == answer)
            
            if is_correct:
                correct += 1
            total += 1
            all_latencies.append(latency_ms)
            
            # Track per-model accuracy
            base_model = model_used
            model_correct[base_model] = model_correct.get(base_model, 0) + (1 if is_correct else 0)
            model_total[base_model] = model_total.get(base_model, 0) + 1
            
            # Log
            question_log = QuestionLog(
                question_id=q_num,
                dataset=dataset_key,
                question_text=question,
                choices=choices,
                correct_answer=answer,
                direct_answer=parsed,
                direct_correct=is_correct,
                direct_latency_ms=latency_ms,
                direct_cost_usd=0.0,
                direct_tokens=tokens,
                gaap_answer=parsed,
                gaap_correct=is_correct,
                gaap_latency_ms=latency_ms,
                gaap_cost_usd=0.0,
                gaap_tokens=tokens,
            )
            logger.log_question(question_log)
            
            mark = "✅" if is_correct else "❌"
            acc = correct / total * 100
            print(f"  Q{q_num:3d}/{samples} {mark} [{base_model[:20]:<20}] "
                  f"Got:{parsed or '?'} Want:{answer} | "
                  f"Acc:{acc:.1f}% | {latency_ms:.0f}ms")
            
        except Exception as e:
            errors += 1
            total += 1
            logger.log_error(e, f"Q{q_num}")
            print(f"  Q{q_num:3d}/{samples} 💥 ERROR: {str(e)[:60]}")
    
    elapsed = time.time() - start_time
    
    # Finalize logger
    logger.finalize()
    
    # Results
    accuracy = correct / max(total, 1) * 100
    avg_latency = sum(all_latencies) / max(len(all_latencies), 1)
    
    print("\n" + "=" * 80)
    print("🏆 UNIFIED BENCHMARK RESULTS (Primary + Fallback)")
    print("=" * 80)
    print(f"📊 Dataset:    {dataset_key}")
    print(f"📊 Samples:    {samples}")
    print(f"✅ Correct:    {correct}/{total}")
    print(f"🎯 Accuracy:   {accuracy:.1f}%")
    print(f"💥 Errors:     {errors}")
    print(f"⏱️  Total Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"⚡ Avg Latency: {avg_latency:.0f}ms")
    print(f"💰 Cost:       $0.00 (FREE!)")
    
    if model_correct:
        print(f"\n📊 Per-Model/Slot Accuracy:")
        for model, count in sorted(model_total.items(), key=lambda x: -model_correct.get(x[0], 0)/max(x[1], 1)):
            mc = model_correct.get(model, 0)
            mt = count
            print(f"   {model[:35]:<35} | {mc}/{mt} ({mc/mt*100:.1f}%)")
    
    print(f"\n📊 Provider Stats:")
    print(provider.get_stats())
    
    print(f"\n📁 Logs saved to: {logger.run_dir}")
    print("=" * 80)
    
    return {
        "dataset": dataset_key,
        "samples": samples,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": avg_latency,
        "cost_usd": 0.0,
        "per_model": {m: {"correct": model_correct.get(m, 0), "total": model_total.get(m, 0)} for m in model_total},
    }


def _run_battle_mode(prompt: str):
    """
    Send a request in battle mode to LMArena.
    
    Battle mode randomly assigns a model (including private/unreleased ones
    like gemini-3-pro, gemini-3-flash that aren't available in direct mode).
    After the response, the model identity is revealed.
    """
    import asyncio
    import json
    
    print("\n⚔️  LMArena Battle Mode")
    print("=" * 60)
    print(f"  Prompt: {prompt[:100]}...")
    print(f"  Note: Model will be randomly assigned (may be private/unreleased)")
    print("=" * 60)
    
    try:
        from g4f.Provider.needs_auth.LMArena import LMArena
        from g4f.cookies import get_cookies_dir
        from pathlib import Path
        import uuid
        
        cache_file = Path(get_cookies_dir()) / "auth_LMArena.json"
        if not cache_file.exists():
            print("  ❌ No auth cache - run --auth first")
            return
        
        with cache_file.open("r") as f:
            args = json.load(f)
        
        # Ensure models are loaded
        if not LMArena._models_loaded:
            models = LMArena.get_models(timeout=30)
            print(f"  📋 Loaded {len(models)} models")
        
        async def battle_request():
            from g4f.requests import StreamSession, raise_for_status
            from g4f.Provider.needs_auth.LMArena import uuid7
            
            # Get fresh recaptcha token
            print("  🔑 Getting recaptcha token...")
            try:
                updated_args, grecaptcha = await LMArena.get_grecaptcha(args, proxy=None)
            except Exception as e:
                print(f"  ⚠️  Recaptcha failed ({e}), trying full auth...")
                updated_args, grecaptcha = await LMArena.get_args_from_nodriver(proxy=None, force=True)
            
            evaluationSessionId = str(uuid7())
            userMessageId = str(uuid7())
            modelAMessageId = str(uuid7())
            modelBMessageId = str(uuid7())
            
            # Battle mode payload: no modelAId/modelBId, server picks randomly
            data = {
                "id": evaluationSessionId,
                "mode": "battle",
                "userMessageId": userMessageId,
                "modelAMessageId": modelAMessageId,
                "modelBMessageId": modelBMessageId,
                "userMessage": {
                    "content": prompt,
                    "experimental_attachments": [],
                    "metadata": {}
                },
                "modality": "chat",
                "recaptchaV3Token": grecaptcha,
            }
            
            url = "https://arena.ai/nextjs-api/stream/create-evaluation"
            print(f"  📡 Sending battle request...")
            
            response_a = []
            response_b = []
            model_a = "Unknown"
            model_b = "Unknown"
            
            async with StreamSession(**updated_args, timeout=300) as session:
                try:
                    async with session.post(url, json=data, proxy=None) as response:
                        if response.status != 200:
                            body = await response.text()
                            print(f"  ❌ HTTP {response.status}: {body[:300]}")
                            return None
                        async for chunk in response.iter_lines():
                            line = chunk.decode()
                            if line.startswith("a0:"):
                                # Model A text chunk
                                text = json.loads(line[3:])
                                if isinstance(text, str) and text != "hasArenaError":
                                    response_a.append(text)
                            elif line.startswith("b0:"):
                                # Model B text chunk
                                text = json.loads(line[3:])
                                if isinstance(text, str):
                                    response_b.append(text)
                            elif line.startswith("ad:"):
                                # Model A finish
                                finish = json.loads(line[3:])
                                model_a = finish.get("modelId", finish.get("model", "Hidden"))
                            elif line.startswith("bd:"):
                                # Model B finish
                                finish = json.loads(line[3:])
                                model_b = finish.get("modelId", finish.get("model", "Hidden"))
                            elif line.startswith("ag:"):
                                # Model A reasoning/thinking
                                pass
                            elif line.startswith("bg:"):
                                # Model B reasoning/thinking
                                pass
                            elif line.startswith("a3:"):
                                error = json.loads(line[3:])
                                print(f"  ❌ Error: {error}")
                                return
                except Exception as e:
                    print(f"  ❌ Stream error: {e}")
                    return None
            
            # Note: In battle mode, model identities are hidden until voting via UI.
            # Models like gemini-3-pro and gemini-3-flash may be assigned randomly.
            # To see which models were used, check arena.ai evaluation history.

            return {
                "model_a": model_a,
                "model_b": model_b,
                "response_a": "".join(response_a),
                "response_b": "".join(response_b),
            }
        
        start = time.time()
        result = asyncio.run(battle_request())
        elapsed = time.time() - start
        
        if result:
            print(f"\n⚔️  BATTLE RESULTS ({elapsed:.1f}s)")
            print("=" * 60)
            
            # Reverse-lookup model names from UUIDs
            model_a_name = result["model_a"]
            model_b_name = result["model_b"]
            if hasattr(LMArena, 'text_models'):
                uuid_to_name = {v: k for k, v in LMArena.text_models.items()}
                model_a_name = uuid_to_name.get(result["model_a"], result["model_a"])
                model_b_name = uuid_to_name.get(result["model_b"], result["model_b"])
            
            print(f"\n  🅰️  Model A: {model_a_name}")
            print(f"  {'─'*50}")
            print(f"  {result['response_a'][:500]}")
            if len(result['response_a']) > 500:
                print(f"  ... ({len(result['response_a'])} chars total)")
            
            print(f"\n  🅱️  Model B: {model_b_name}")
            print(f"  {'─'*50}")
            print(f"  {result['response_b'][:500]}")
            if len(result['response_b']) > 500:
                print(f"  ... ({len(result['response_b'])} chars total)")
            
            print(f"\n  ⏱️  Time: {elapsed:.1f}s")
            print("=" * 60)
            
            # Check if we got any private/unreleased models
            private_hits = []
            for name in [model_a_name, model_b_name]:
                if any(p in name.lower() for p in ["gemini-3", "gemini-2.5-pro-preview", "unreleased"]):
                    private_hits.append(name)
            if private_hits:
                print(f"\n  🎯 Private/Unreleased model detected: {', '.join(private_hits)}")
        
    except Exception as e:
        print(f"  ❌ Battle mode failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Unified Free Benchmark (g4f + optional free registered providers)"
    )
    parser.add_argument("--dataset", default="mmlu", choices=DATASETS.keys(),
                        help="Dataset to benchmark (default: mmlu)")
    parser.add_argument("--samples", type=int, default=20,
                        help="Number of samples (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--delay", type=float, default=4.0,
                        help="Min seconds between requests (default: 4.0)")
    parser.add_argument("--profile", type=str, default="quality",
                        choices=["premium", "quality", "balanced", "throughput"],
                        help="Provider profile: premium|quality|balanced|throughput (default: quality)")
    parser.add_argument("--model", type=str, default=None,
                        help="Force a specific model (e.g. 'claude-opus-4-5-20251101' via LMArena)")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (MMLU-Pro only, e.g. math, physics, biology)")
    parser.add_argument("--list-chain", action="store_true",
                        help="Show the fallback chain configuration and exit")
    parser.add_argument("--show-onboarding", action="store_true",
                        help="Show free-registration requirements and exit")
    parser.add_argument("--preflight", action="store_true",
                        help="Run provider health checks before benchmark")
    parser.add_argument("--auth", action="store_true",
                        help="Warm up LMArena auth (opens browser for captcha)")
    parser.add_argument("--auth-status", action="store_true",
                        help="Check LMArena auth cache status")
    parser.add_argument("--auth-reset", action="store_true",
                        help="Delete LMArena auth cache and re-authenticate")
    parser.add_argument("--battle", type=str, default=None,
                        help="Run a battle-mode query against LMArena (for private/unreleased models)")
    
    # WebChat provider auth commands
    parser.add_argument("--webchat-login", type=str, default=None,
                        metavar="PROVIDER",
                        help="Login to webchat provider (glm | kimi | deepseek) via browser")
    parser.add_argument("--webchat-status", action="store_true",
                        help="Show auth status for all webchat providers (GLM, Kimi, DeepSeek)")
    parser.add_argument("--webchat-reset", type=str, default=None,
                        metavar="PROVIDER",
                        help="Reset webchat provider auth cache (glm | kimi | deepseek)")
    parser.add_argument("--webchat-test", type=str, default=None,
                        metavar="PROVIDER",
                        help="Quick test webchat provider (glm | kimi | deepseek)")
    parser.add_argument("--webchat-account", type=str, default="default",
                        help="Account label for webchat commands (default: default)")
    
    args = parser.parse_args()
    
    if args.list_chain:
        p = UnifiedProvider(profile=args.profile)
        print(p.get_config_summary())
        return

    if args.show_onboarding:
        p = UnifiedProvider(profile=args.profile)
        print(p.get_onboarding_summary())
        return

    if args.preflight:
        p = UnifiedProvider(profile=args.profile, min_delay=args.delay)
        print(p.get_config_summary())
        run_preflight(p)
        return

    if args.auth_status:
        print("\n🔐 LMArena Auth Status")
        print("=" * 50)
        status = check_lmarena_auth()
        print(f"  Cache file: {status['cache_file']}")
        print(f"  Valid:      {status['valid']}")
        print(f"  Anonymous:  {status['is_anonymous']}")
        if status['expires_in_sec'] > 0:
            print(f"  Expires in: {status['expires_in_sec']//60}m {status['expires_in_sec']%60}s")
        elif status['expires_in_sec'] < 0:
            print(f"  Expired:    {abs(status['expires_in_sec'])//60}m ago")
        print(f"  Status:     {status['message']}")
        print("=" * 50)
        return

    if args.auth_reset:
        print("\n🔐 Resetting LMArena Auth")
        print("=" * 50)
        invalidate_lmarena_cache()
        warmup_lmarena_auth(force=True)
        print("=" * 50)
        return

    if args.auth:
        print("\n🔐 LMArena Auth Warmup")
        print("=" * 50)
        warmup_lmarena_auth(force=False)
        print("=" * 50)
        return

    if args.battle:
        _run_battle_mode(args.battle)
        return

    # ── WebChat provider commands ──
    if args.webchat_status:
        from gaap_system_glm5.providers.webchat_providers import check_all_webchat_auth
        print("\n🌐 WebChat Provider Auth Status")
        print("=" * 60)
        all_status = check_all_webchat_auth()
        for pname, statuses in all_status.items():
            for s in statuses:
                icon = "✅" if s["valid"] else "❌"
                print(f"  {icon} {s['provider']}[{s['account']}]: {s['message']}")
        print("=" * 60)
        return

    if args.webchat_login:
        from gaap_system_glm5.providers.webchat_providers import get_provider
        pname = args.webchat_login
        account = args.webchat_account
        print(f"\n🌐 WebChat Login: {pname} [{account}]")
        print("=" * 60)
        provider = get_provider(pname, account)
        provider.warmup(force=True)
        print("=" * 60)
        return

    if args.webchat_reset:
        from gaap_system_glm5.providers.webchat_providers import invalidate_auth
        pname = args.webchat_reset
        account = args.webchat_account
        print(f"\n🌐 Resetting WebChat Auth: {pname} [{account}]")
        if invalidate_auth(pname, account):
            print(f"  🗑️  Cleared {pname} [{account}] auth cache")
        else:
            print(f"  ℹ️  No cache to clear for {pname} [{account}]")
        return

    if args.webchat_test:
        from gaap_system_glm5.providers.webchat_providers import get_provider
        pname = args.webchat_test
        account = args.webchat_account
        print(f"\n🧪 Testing WebChat Provider: {pname} [{account}]")
        print("=" * 60)
        provider = get_provider(pname, account)
        status = provider.check_auth()
        if not status["valid"]:
            print(f"  ❌ Not authenticated: {status['message']}")
            print(f"  💡 Run: --webchat-login {pname}")
            return
        print(f"  Auth: {status['message']}")
        try:
            result = provider.call(
                [{"role": "user", "content": "Reply with only: OK"}],
                timeout=60,
            )
            print(f"  ✅ Response: '{result[:200]}'")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print("=" * 60)
        return
    
    # Show LMArena auth status before benchmark if using premium/quality
    if args.profile in ("premium", "quality"):
        status = check_lmarena_auth()
        if status["valid"]:
            print(f"  🔐 LMArena auth: {status['message']}")
        else:
            print(f"  ⚠️  LMArena auth: {status['message']}")
            print(f"      Run --auth to authenticate first")

    result = run_benchmark(
        dataset_key=args.dataset,
        samples=args.samples,
        seed=args.seed,
        delay=args.delay,
        category=args.category,
        profile=args.profile,
        model=args.model,
    )


if __name__ == "__main__":
    main()
