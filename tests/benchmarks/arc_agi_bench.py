#!/usr/bin/env python3
"""
ARC-AGI Benchmark for GAAP System (v2 — Enhanced)
====================================================

Tests abstract reasoning on ARC-AGI grid puzzles.
Each task: given input→output training pairs, predict the output for a test input.

v2 improvements over v1:
  - Official o3-style prompt (simpler, proven better)
  - Multi-attempt: 3 tries per task (ARC rules allow 3)
  - Refinement: verify answer against training, then fix
  - Better response parsing (text grids + backscan)
  - Space-separated grid output (matches official format)

Usage:
    python -m gaap.benchmarks.arc_agi_bench --provider kimi --tasks 10
    python -m gaap.benchmarks.arc_agi_bench --provider kimi --tasks 10 --strategy direct
    python -m gaap.benchmarks.arc_agi_bench --provider deepseek --tasks 5 --attempts 3
"""

import argparse
import json
import os
import random
import re
import time
from typing import Any

from gaap.providers.account_manager import bootstrap_pools
from gaap.providers.webchat_providers import webchat_call

# =============================================================================
# ARC Color Map (for visual grid rendering)
# =============================================================================
COLOR_NAMES = {
    0: "⬛",
    1: "🟦",
    2: "🟥",
    3: "🟩",
    4: "🟨",
    5: "⬜",
    6: "🟪",
    7: "🟧",
    8: "🩵",
    9: "🟫",
}


def grid_to_text(grid: list[list[int]]) -> str:
    """Convert grid to space-separated text (official ARC format)."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def grid_to_visual(grid: list[list[int]]) -> str:
    """Convert grid to emoji visual."""
    return "\n".join("".join(COLOR_NAMES.get(c, "❓") for c in row) for row in grid)


# =============================================================================
# Prompt Construction — v2 (Official o3-style)
# =============================================================================


def build_arc_prompt(task: dict[str, Any], use_visual: bool = False) -> str:
    """Build ARC prompt using official o3-style format (proven more effective)."""
    parts = []
    parts.append(
        "Find the common rule that maps an input grid to an output grid, "
        "given the examples below.\n"
    )

    # Training examples — clean, minimal
    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i+1}:")
        parts.append("Input:")
        parts.append(grid_to_text(ex["input"]))
        parts.append("Output:")
        parts.append(grid_to_text(ex["output"]))
        parts.append("")

    # Test input
    test_input = task["test"][0]["input"]
    parts.append(
        "Below is a test input grid. Predict the corresponding output grid "
        "by applying the rule you found. Your final answer should just be "
        "the text output grid itself.\n"
    )
    parts.append("Input:")
    parts.append(grid_to_text(test_input))

    return "\n".join(parts)


def build_refinement_prompt(
    task: dict[str, Any],
    previous_answer: list[list[int]],
) -> str:
    """Build a refinement prompt — asks model to verify and fix its answer."""
    parts = []
    parts.append(
        "You previously answered an ARC-AGI puzzle. Here is the full puzzle "
        "and your previous answer. Please verify your answer by checking it "
        "against EACH training example's transformation rule. If your answer "
        "is wrong, explain why and provide the corrected output grid.\n"
    )

    # Show training examples again
    for i, ex in enumerate(task["train"]):
        parts.append(f"Example {i+1}:")
        parts.append("Input:")
        parts.append(grid_to_text(ex["input"]))
        parts.append("Output:")
        parts.append(grid_to_text(ex["output"]))
        parts.append("")

    # Show test input
    test_input = task["test"][0]["input"]
    parts.append("Test Input:")
    parts.append(grid_to_text(test_input))
    parts.append("")

    # Show previous answer
    parts.append("Your previous answer:")
    parts.append(grid_to_text(previous_answer))
    parts.append("")
    parts.append(
        "Carefully verify: does your answer follow the same transformation "
        "rule as every training example? If not, provide the corrected output "
        "grid. Your final answer should just be the text output grid itself."
    )

    return "\n".join(parts)


# =============================================================================
# Response Parsing
# =============================================================================


def parse_grid_response(response: str) -> list[list[int]] | None:
    """
    Parse a grid from model response. Uses backscan strategy (answer is usually
    at the END of the response). Tries multiple extraction strategies.
    """
    # Strategy 1: Backscan for text grid (space-separated numbers) — official format
    # This is the most common output format for the o3-style prompt
    lines = response.strip().split("\n")
    grid_lines = []
    found_end = False

    for line in reversed(lines):
        clean = line.strip()
        # Skip empty lines at the end
        if not clean and not found_end:
            continue
        # Skip common non-grid endings
        if clean.startswith("```") or clean.lower().startswith(
            ("output", "answer", "result", "the ")
        ):
            if found_end:
                break
            continue
        # Try to parse as a row of numbers
        # Match lines like "0 5 0 0 0" or "0, 5, 0, 0, 0" or "[0, 5, 0]"
        clean_nums = clean.strip("[]").replace(",", " ")
        nums = clean_nums.split()
        if nums and all(n.strip().isdigit() for n in nums if n.strip()):
            grid_lines.insert(0, [int(n.strip()) for n in nums if n.strip()])
            found_end = True
        elif found_end:
            break  # We were reading a grid and hit a non-grid line

    if len(grid_lines) >= 1:
        # Validate: all rows should have same width
        widths = [len(r) for r in grid_lines]
        if len(set(widths)) == 1 and widths[0] >= 1:
            return grid_lines

    # Strategy 2: Find JSON array in response (backscan — last match)
    matches = re.findall(r"\[\s*\[[\d\s,\[\]]+\]\s*\]", response, re.DOTALL)
    if matches:
        for match in reversed(matches):
            try:
                grid = json.loads(match)
                if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
                    return grid
            except json.JSONDecodeError:
                continue

    # Strategy 3: Look for grid in code blocks (last code block)
    code_blocks = re.findall(r"```(?:json|text|)?\s*(.*?)```", response, re.DOTALL)
    for block in reversed(code_blocks):
        block = block.strip()
        # Try JSON parse
        try:
            grid = json.loads(block)
            if isinstance(grid, list) and all(isinstance(r, list) for r in grid):
                return grid
        except (json.JSONDecodeError, ValueError):
            pass
        # Try text grid parse from code block
        block_lines = block.split("\n")
        grid_from_block = []
        for bline in block_lines:
            clean_nums = bline.strip().strip("[]").replace(",", " ")
            nums = clean_nums.split()
            if nums and all(n.strip().isdigit() for n in nums if n.strip()):
                grid_from_block.append([int(n.strip()) for n in nums if n.strip()])
        if len(grid_from_block) >= 1:
            widths = [len(r) for r in grid_from_block]
            if len(set(widths)) == 1:
                return grid_from_block

    return None


# =============================================================================
# Grid Comparison
# =============================================================================


def compare_grids(predicted: list[list[int]], expected: list[list[int]]) -> dict[str, Any]:
    """Compare predicted vs expected grid."""
    if predicted is None:
        return {"match": False, "reason": "parse_failed", "cell_accuracy": 0.0}

    if len(predicted) != len(expected):
        return {
            "match": False,
            "reason": f"row_mismatch ({len(predicted)} vs {len(expected)})",
            "cell_accuracy": 0.0,
        }

    total_cells = 0
    correct_cells = 0
    for r, (pred_row, exp_row) in enumerate(zip(predicted, expected)):
        if len(pred_row) != len(exp_row):
            return {
                "match": False,
                "reason": f"col_mismatch row {r} ({len(pred_row)} vs {len(exp_row)})",
                "cell_accuracy": 0.0,
            }
        for c, (p, e) in enumerate(zip(pred_row, exp_row)):
            total_cells += 1
            if p == e:
                correct_cells += 1

    accuracy = correct_cells / total_cells if total_cells > 0 else 0.0
    return {
        "match": accuracy == 1.0,
        "cell_accuracy": round(accuracy, 4),
        "correct_cells": correct_cells,
        "total_cells": total_cells,
    }


# =============================================================================
# Task Loading
# =============================================================================


def load_arc_tasks(n_tasks: int = 10, split: str = "evaluation", seed: int = 42) -> list[dict]:
    """Load ARC-AGI tasks from HuggingFace, filtering for small grids."""
    from datasets import load_dataset

    print(f"📥 Loading ARC-AGI tasks from HuggingFace (split={split})...")
    ds = load_dataset("lordspline/arc-agi", streaming=True)

    tasks = []
    random.seed(seed)
    candidates = []

    for i, sample in enumerate(ds[split]):
        train = sample["train"]
        test = sample["test"]
        # Only small grids (≤10×10)
        max_dim = 0
        for ex in train + test:
            for g in [ex["input"], ex["output"]]:
                max_dim = max(max_dim, len(g), max(len(r) for r in g))
        if max_dim <= 10 and len(train) >= 2:
            candidates.append({"idx": i, "train": train, "test": test, "max_dim": max_dim})
        if len(candidates) >= n_tasks * 3:  # collect extra for random selection
            break

    # Random sample
    if len(candidates) > n_tasks:
        tasks = random.sample(candidates, n_tasks)
    else:
        tasks = candidates[:n_tasks]

    print(f"   Selected {len(tasks)} tasks (max grid ≤10×10)")
    return tasks


# =============================================================================
# Main Benchmark
# =============================================================================


def run_arc_benchmark(
    provider: str = "kimi",
    model: str = "kimi-k2.5-thinking",
    n_tasks: int = 10,
    timeout: int = 180,
    use_visual: bool = False,
    split: str = "evaluation",
    seed: int = 42,
    max_attempts: int = 3,
    strategy: str = "refine",  # "direct" | "refine" | "multi"
) -> dict[str, Any]:
    """
    Run ARC-AGI benchmark on a provider.

    Strategies:
      - direct: Single attempt per task (v1 behavior)
      - refine: First attempt + refinement pass if wrong (recommended)
      - multi:  Up to max_attempts independent attempts, any correct = solved
    """

    bootstrap_pools()
    tasks = load_arc_tasks(n_tasks=n_tasks, split=split, seed=seed)

    print(f"\n{'='*70}")
    print(f"🧩 ARC-AGI Benchmark v2 — {provider}/{model}")
    print(f"   Tasks: {len(tasks)} | Strategy: {strategy} | Attempts: {max_attempts}")
    print(f"   Timeout: {timeout}s | Split: {split} | Seed: {seed}")
    print(f"{'='*70}\n")

    results = []
    correct = 0
    total = 0

    for i, task in enumerate(tasks):
        total += 1
        task_id = task["idx"]
        n_train = len(task["train"])
        test_output = task["test"][0]["output"]
        grid_size = f"{len(test_output)}×{len(test_output[0])}"

        print(f"🧩 Task {i+1}/{len(tasks)} (idx={task_id}, {n_train} train, grid {grid_size})")

        task_result = {
            "task_idx": task_id,
            "grid_size": grid_size,
            "n_train": n_train,
            "match": False,
            "cell_accuracy": 0,
            "attempts": [],
        }

        solved = False
        best_accuracy = 0.0
        best_predicted = None
        total_latency = 0

        # ── Attempt loop ──────────────────────────────────────
        for attempt_num in range(1, max_attempts + 1):
            if solved:
                break

            # Determine what to send
            if attempt_num == 1 or strategy == "multi":
                # Fresh attempt
                prompt = build_arc_prompt(task, use_visual=use_visual)
                attempt_type = "direct"
            elif strategy == "refine" and best_predicted is not None:
                # Refinement attempt — ask model to verify & fix
                prompt = build_refinement_prompt(task, best_predicted)
                attempt_type = "refine"
            else:
                # Fallback: fresh attempt
                prompt = build_arc_prompt(task, use_visual=use_visual)
                attempt_type = "direct"

            t0 = time.time()
            try:
                messages = [{"role": "user", "content": prompt}]
                response = webchat_call(provider, messages, model=model, timeout=timeout)
                latency_ms = (time.time() - t0) * 1000
                total_latency += latency_ms

                predicted = parse_grid_response(response)
                comparison = compare_grids(predicted, test_output)

                attempt_data = {
                    "attempt": attempt_num,
                    "type": attempt_type,
                    "latency_ms": round(latency_ms),
                    "match": comparison["match"],
                    "cell_accuracy": comparison.get("cell_accuracy", 0),
                    "predicted": predicted,
                    "response_len": len(response),
                    "parse_ok": predicted is not None,
                }
                task_result["attempts"].append(attempt_data)

                if comparison["match"]:
                    solved = True
                    best_accuracy = 1.0
                    best_predicted = predicted
                    print(f"   ✅ Attempt {attempt_num} ({attempt_type}) | {latency_ms:.0f}ms")
                elif predicted is not None:
                    acc = comparison["cell_accuracy"]
                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_predicted = predicted
                    print(
                        f"   ❌ Attempt {attempt_num} ({attempt_type}) | {acc:.0%} cells | {latency_ms:.0f}ms"
                    )
                else:
                    print(
                        f"   ❌ Attempt {attempt_num} ({attempt_type}) | parse failed | {latency_ms:.0f}ms"
                    )

                # For direct strategy, only 1 attempt
                if strategy == "direct":
                    break

                # For refine strategy: attempt 1 = direct, attempt 2 = refine, done
                if strategy == "refine" and attempt_num >= 2:
                    break

            except Exception as e:
                latency_ms = (time.time() - t0) * 1000
                total_latency += latency_ms
                attempt_data = {
                    "attempt": attempt_num,
                    "type": attempt_type,
                    "latency_ms": round(latency_ms),
                    "match": False,
                    "cell_accuracy": 0,
                    "error": str(e)[:200],
                }
                task_result["attempts"].append(attempt_data)
                print(f"   ❌ Attempt {attempt_num} ERROR: {str(e)[:60]} ({latency_ms:.0f}ms)")

            time.sleep(2)  # Rate limit spacing

        # ── Task summary ──────────────────────────────────
        if solved:
            correct += 1

        task_result["match"] = solved
        task_result["cell_accuracy"] = best_accuracy
        task_result["total_latency_ms"] = round(total_latency)
        task_result["predicted"] = best_predicted
        task_result["expected"] = test_output
        task_result["n_attempts_used"] = len(task_result["attempts"])
        results.append(task_result)

        if not solved and best_predicted:
            print(f"   📊 Best cell accuracy: {best_accuracy:.0%}")

        time.sleep(2)  # Spacing between tasks

    # ── Overall summary ───────────────────────────────────
    accuracy = correct / total if total > 0 else 0
    avg_cell_acc = sum(r.get("cell_accuracy", 0) for r in results) / len(results) if results else 0
    avg_latency = sum(r["total_latency_ms"] for r in results) / len(results) if results else 0
    total_attempts = sum(r["n_attempts_used"] for r in results)

    summary = {
        "provider": provider,
        "model": model,
        "benchmark": "ARC-AGI",
        "version": "v2",
        "strategy": strategy,
        "max_attempts": max_attempts,
        "split": split,
        "seed": seed,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_tasks": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "avg_cell_accuracy": round(avg_cell_acc, 4),
        "avg_latency_ms": round(avg_latency),
        "total_attempts": total_attempts,
        "results": results,
    }

    print(f"\n{'='*70}")
    print(f"📊 ARC-AGI v2 Results — {provider}/{model}")
    print(f"{'='*70}")
    print(f"  Strategy:        {strategy} (max {max_attempts} attempts)")
    print(f"  Tasks:           {total}")
    print(f"  Correct:         {correct}/{total} ({accuracy:.1%})")
    print(f"  Cell accuracy:   {avg_cell_acc:.1%}")
    print(f"  Total attempts:  {total_attempts}")
    print(f"  Avg latency:     {avg_latency:.0f}ms")
    print(f"{'='*70}")

    # Save results
    outfile = f"benchmark_logs/arc_agi_{provider}_{int(time.time())}.json"
    os.makedirs("benchmark_logs", exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📁 {outfile}")

    return summary


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="ARC-AGI Benchmark v2 for GAAP")
    parser.add_argument("--provider", default="kimi", help="Provider: kimi, deepseek, glm")
    parser.add_argument("--model", default="", help="Model name (default: auto)")
    parser.add_argument("--tasks", type=int, default=10, help="Number of tasks")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout per attempt (seconds)")
    parser.add_argument("--visual", action="store_true", help="Include emoji visuals in prompt")
    parser.add_argument(
        "--split", default="evaluation", help="Dataset split: training, evaluation, trial"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--attempts", type=int, default=3, help="Max attempts per task")
    parser.add_argument(
        "--strategy",
        default="refine",
        choices=["direct", "refine", "multi"],
        help="Strategy: direct (1 shot), refine (1 + verify/fix), multi (N independent)",
    )
    args = parser.parse_args()

    # Default models
    model_defaults = {
        "kimi": "kimi-k2.5-thinking",
        "deepseek": "deepseek",
        "glm": "GLM-5",
    }
    model = args.model or model_defaults.get(args.provider, args.provider)

    run_arc_benchmark(
        provider=args.provider,
        model=model,
        n_tasks=args.tasks,
        timeout=args.timeout,
        use_visual=args.visual,
        split=args.split,
        seed=args.seed,
        max_attempts=args.attempts,
        strategy=args.strategy,
    )


if __name__ == "__main__":
    main()
