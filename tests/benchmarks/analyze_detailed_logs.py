"""
Detailed Benchmark Log Analyzer
================================

Analyzes the comprehensive logs from benchmark runs to extract insights:
- Question-by-question performance
- Rate limiting patterns
- Key rotation effectiveness
- Error analysis
- Performance trends
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_run_data(run_dir: Path) -> dict[str, Any]:
    """Load all data from a benchmark run"""
    data = {}

    # Metadata
    with open(run_dir / "metadata.json") as f:
        data["metadata"] = json.load(f)

    # Questions
    with open(run_dir / "questions.json") as f:
        data["questions"] = json.load(f)

    # Events
    with open(run_dir / "events.json") as f:
        data["events"] = json.load(f)

    # Errors (if exists)
    errors_file = run_dir / "errors.json"
    if errors_file.exists():
        with open(errors_file) as f:
            content = f.read().strip()
            if content:
                # Parse multiple JSON objects
                data["errors"] = []
                for line in content.split("\n"):
                    if line.strip():
                        data["errors"].append(json.loads(line))
            else:
                data["errors"] = []
    else:
        data["errors"] = []

    return data


def analyze_rate_limiting(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze rate limiting patterns"""
    retry_events = [e for e in events if e["type"] == "rate_limit_retry"]

    if not retry_events:
        return {
            "total_retries": 0,
            "avg_delay": 0.0,
            "max_delay": 0.0,
            "retries_per_question": {},
        }

    # Group by context (question)
    retries_by_question = defaultdict(list)
    for event in retry_events:
        context = event["details"].get("context", "unknown")
        retries_by_question[context].append(event["details"]["delay"])

    all_delays = [e["details"]["delay"] for e in retry_events]

    return {
        "total_retries": len(retry_events),
        "avg_delay": sum(all_delays) / len(all_delays),
        "max_delay": max(all_delays),
        "min_delay": min(all_delays),
        "retries_per_question": {
            q: {
                "count": len(delays),
                "total_delay": sum(delays),
                "avg_delay": sum(delays) / len(delays),
            }
            for q, delays in retries_by_question.items()
        },
        "questions_with_retries": len(retries_by_question),
    }


def analyze_accuracy_patterns(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze accuracy patterns"""

    # Both correct
    both_correct = sum(1 for q in questions if q["direct_correct"] and q["gaap_correct"])

    # Only GAAP correct (improvement)
    gaap_only = sum(1 for q in questions if q["gaap_correct"] and not q["direct_correct"])

    # Only Direct correct (regression)
    direct_only = sum(1 for q in questions if q["direct_correct"] and not q["gaap_correct"])

    # Both wrong
    both_wrong = sum(1 for q in questions if not q["direct_correct"] and not q["gaap_correct"])

    # Questions where GAAP had different answer
    different_answers = sum(1 for q in questions if q["direct_answer"] != q["gaap_answer"])

    return {
        "both_correct": both_correct,
        "gaap_only_correct": gaap_only,
        "direct_only_correct": direct_only,
        "both_wrong": both_wrong,
        "different_answers": different_answers,
        "gaap_improvements": gaap_only,
        "gaap_regressions": direct_only,
        "net_improvement": gaap_only - direct_only,
    }


def analyze_performance_by_attempts(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze performance based on number of retry attempts"""

    # Group questions by number of attempts
    by_direct_attempts = defaultdict(list)
    by_gaap_attempts = defaultdict(list)

    for q in questions:
        by_direct_attempts[q["direct_attempts"]].append(q)
        by_gaap_attempts[q["gaap_attempts"]].append(q)

    def calc_stats(questions_list):
        if not questions_list:
            return {"count": 0, "accuracy": 0.0, "avg_latency": 0.0}

        correct = sum(1 for q in questions_list if q["gaap_correct"])
        avg_lat = sum(q["gaap_latency_ms"] for q in questions_list) / len(questions_list)

        return {
            "count": len(questions_list),
            "accuracy": correct / len(questions_list),
            "avg_latency_ms": avg_lat,
        }

    return {
        "by_direct_attempts": {
            attempts: calc_stats(qs) for attempts, qs in sorted(by_direct_attempts.items())
        },
        "by_gaap_attempts": {
            attempts: calc_stats(qs) for attempts, qs in sorted(by_gaap_attempts.items())
        },
    }


def analyze_errors(errors: list[dict[str, Any]], questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze error patterns"""

    if not errors:
        return {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_context": {},
        }

    errors_by_type = defaultdict(int)
    errors_by_context = defaultdict(int)

    for error in errors:
        errors_by_type[error["error_type"]] += 1
        errors_by_context[error["context"]] += 1

    # Also check embedded errors in questions
    question_errors = defaultdict(int)
    for q in questions:
        for err in q.get("errors", []):
            if "Direct" in err:
                question_errors["direct"] += 1
            elif "GAAP" in err:
                question_errors["gaap"] += 1

    return {
        "total_errors": len(errors),
        "errors_by_type": dict(errors_by_type),
        "errors_by_context": dict(errors_by_context),
        "question_errors": dict(question_errors),
    }


def analyze_timeline(events: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Analyze timeline of the benchmark"""

    if not events:
        return {}

    start_time = datetime.fromisoformat(metadata["start_time"])
    end_time = datetime.fromisoformat(metadata.get("end_time", metadata["start_time"]))

    # Find question completion times
    question_complete_events = [
        e for e in events if e["type"] in ["direct_complete", "gaap_complete"]
    ]

    if not question_complete_events:
        return {}

    # Calculate time between questions
    question_times = []
    prev_time = None

    for event in question_complete_events:
        event_time = datetime.fromisoformat(event["timestamp"])
        if prev_time:
            delta = (event_time - prev_time).total_seconds()
            question_times.append(delta)
        prev_time = event_time

    return {
        "total_duration_seconds": (end_time - start_time).total_seconds(),
        "avg_time_between_events": (
            sum(question_times) / len(question_times) if question_times else 0
        ),
        "min_time_between_events": min(question_times) if question_times else 0,
        "max_time_between_events": max(question_times) if question_times else 0,
    }


def generate_analysis_report(run_dir: Path) -> str:
    """Generate comprehensive analysis report"""

    data = load_run_data(run_dir)

    metadata = data["metadata"]
    questions = data["questions"]
    events = data["events"]
    errors = data["errors"]

    lines = []
    lines.append("=" * 80)
    lines.append("DETAILED BENCHMARK ANALYSIS")
    lines.append(f"Run ID: {metadata['run_id']}")
    lines.append(f"Dataset: {metadata['dataset']}")
    lines.append("=" * 80)
    lines.append("")

    # Basic stats
    lines.append("## BASIC STATISTICS")
    lines.append(f"Total Questions: {metadata['total_questions']}")
    lines.append(f"Completed: {metadata['completed_questions']}")
    lines.append(f"Failed: {metadata['failed_questions']}")
    lines.append(
        f"Duration: {metadata['total_duration_seconds']:.2f}s ({metadata['total_duration_seconds']/60:.2f} min)"
    )
    lines.append(f"Avg per Question: {metadata['avg_time_per_question']:.2f}s")
    lines.append("")

    # Accuracy analysis
    lines.append("## ACCURACY ANALYSIS")
    acc_analysis = analyze_accuracy_patterns(questions)

    direct_acc = sum(1 for q in questions if q["direct_correct"]) / max(len(questions), 1)
    gaap_acc = sum(1 for q in questions if q["gaap_correct"]) / max(len(questions), 1)

    lines.append(f"Direct Accuracy: {direct_acc*100:.1f}%")
    lines.append(f"GAAP Accuracy: {gaap_acc*100:.1f}%")
    lines.append(f"Improvement: {(gaap_acc - direct_acc)*100:.1f}%")
    lines.append("")

    lines.append("### Answer Patterns")
    lines.append(f"  Both Correct: {acc_analysis['both_correct']}")
    lines.append(f"  GAAP Only Correct (Improvements): {acc_analysis['gaap_only_correct']} ✅")
    lines.append(f"  Direct Only Correct (Regressions): {acc_analysis['direct_only_correct']} ⚠️")
    lines.append(f"  Both Wrong: {acc_analysis['both_wrong']}")
    lines.append(f"  Different Answers: {acc_analysis['different_answers']}")
    lines.append(f"  Net Improvement: {acc_analysis['net_improvement']} questions")
    lines.append("")

    # Rate limiting analysis
    lines.append("## RATE LIMITING ANALYSIS")
    rl_analysis = analyze_rate_limiting(events)

    lines.append(f"Total Retries: {rl_analysis['total_retries']}")
    lines.append(f"Questions with Retries: {rl_analysis['questions_with_retries']}")
    lines.append(f"Avg Retry Delay: {rl_analysis.get('avg_delay', 0):.2f}s")
    lines.append(f"Max Retry Delay: {rl_analysis.get('max_delay', 0):.2f}s")

    if rl_analysis["total_retries"] > 0:
        total_retry_time = sum(
            info["total_delay"] for info in rl_analysis["retries_per_question"].values()
        )
        lines.append(
            f"Total Time Waiting for Rate Limits: {total_retry_time:.2f}s ({total_retry_time/60:.2f} min)"
        )
        lines.append(
            f"  {total_retry_time / metadata['total_duration_seconds'] * 100:.1f}% of total time"
        )
    lines.append("")

    # Top 5 questions with most retries
    if rl_analysis["retries_per_question"]:
        lines.append("### Questions with Most Retries")
        sorted_retries = sorted(
            rl_analysis["retries_per_question"].items(), key=lambda x: x[1]["count"], reverse=True
        )[:5]
        for context, info in sorted_retries:
            lines.append(
                f"  {context}: {info['count']} retries, {info['total_delay']:.1f}s total delay"
            )
        lines.append("")

    # Performance by attempts
    lines.append("## PERFORMANCE BY RETRY ATTEMPTS")
    perf_analysis = analyze_performance_by_attempts(questions)

    lines.append("### GAAP Performance by Attempts")
    for attempts, stats in perf_analysis["by_gaap_attempts"].items():
        lines.append(
            f"  {attempts} attempt(s): {stats['count']} questions, "
            f"{stats['accuracy']*100:.1f}% accuracy, "
            f"{stats['avg_latency_ms']:.0f}ms avg latency"
        )
    lines.append("")

    # Error analysis
    if errors or any(q.get("errors") for q in questions):
        lines.append("## ERROR ANALYSIS")
        err_analysis = analyze_errors(errors, questions)

        lines.append(f"Total System Errors: {err_analysis['total_errors']}")

        if err_analysis["errors_by_type"]:
            lines.append("### Errors by Type")
            for err_type, count in err_analysis["errors_by_type"].items():
                lines.append(f"  {err_type}: {count}")
            lines.append("")

        if err_analysis["question_errors"]:
            lines.append("### Question-Level Errors")
            for context, count in err_analysis["question_errors"].items():
                lines.append(f"  {context}: {count}")
            lines.append("")

    # Cost analysis
    lines.append("## COST ANALYSIS")
    total_direct_cost = sum(q["direct_cost_usd"] for q in questions)
    total_gaap_cost = sum(q["gaap_cost_usd"] for q in questions)

    lines.append(f"Total Direct Cost: ${total_direct_cost:.6f}")
    lines.append(f"Total GAAP Cost: ${total_gaap_cost:.6f}")
    lines.append(f"Cost Multiplier: {total_gaap_cost / max(total_direct_cost, 0.000001):.2f}x")
    lines.append(f"Per Question (Direct): ${total_direct_cost / max(len(questions), 1):.6f}")
    lines.append(f"Per Question (GAAP): ${total_gaap_cost / max(len(questions), 1):.6f}")
    lines.append("")

    # Token analysis
    lines.append("## TOKEN USAGE")
    total_direct_tokens = sum(q["direct_tokens"] for q in questions)
    total_gaap_tokens = sum(q["gaap_tokens"] for q in questions)

    lines.append(f"Total Direct Tokens: {total_direct_tokens:,}")
    lines.append(f"Total GAAP Tokens: {total_gaap_tokens:,}")
    lines.append(f"Token Multiplier: {total_gaap_tokens / max(total_direct_tokens, 1):.2f}x")
    lines.append(f"Per Question (Direct): {total_direct_tokens / max(len(questions), 1):.0f}")
    lines.append(f"Per Question (GAAP): {total_gaap_tokens / max(len(questions), 1):.0f}")
    lines.append("")

    # Latency analysis
    lines.append("## LATENCY ANALYSIS")
    direct_latencies = [q["direct_latency_ms"] for q in questions]
    gaap_latencies = [q["gaap_latency_ms"] for q in questions]

    lines.append(f"Direct Avg: {sum(direct_latencies)/len(direct_latencies):.0f}ms")
    lines.append(f"GAAP Avg: {sum(gaap_latencies)/len(gaap_latencies):.0f}ms")
    lines.append(f"Latency Increase: {sum(gaap_latencies)/sum(direct_latencies):.2f}x")
    lines.append("")

    # System configuration
    lines.append("## SYSTEM CONFIGURATION")
    lines.append(f"Provider: {metadata.get('provider', 'unknown')}")
    lines.append(f"Model: {metadata.get('model', 'unknown')}")
    lines.append(f"GAAP Mode: {metadata.get('gaap_mode', 'unknown')}")
    lines.append(f"API Keys: {metadata.get('num_api_keys', 0)}")
    lines.append(f"Enable All Layers: {metadata.get('enable_all', False)}")

    if metadata.get("layer_config"):
        lines.append("")
        lines.append("Layer Configuration:")
        for layer, config in metadata["layer_config"].items():
            lines.append(f"  {layer}:")
            for key, value in config.items():
                lines.append(f"    {key}: {value}")
    lines.append("")

    # Files reference
    lines.append("## DATA FILES")
    lines.append(f"Run Directory: {run_dir}")
    lines.append("  - metadata.json: Run configuration")
    lines.append(f"  - questions.json: All {len(questions)} questions with full details")
    lines.append(f"  - events.json: {len(events)} system events")
    lines.append("  - SUMMARY.md: High-level summary")
    lines.append("  - DETAILED.md: Question-by-question report")
    lines.append("  - PERFORMANCE.md: Performance metrics")
    lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_detailed_logs.py <run_dir>")
        print("\nExample:")
        print("  python analyze_detailed_logs.py ./benchmark_logs/run_1739123456")
        print("\nOr use the latest run:")
        print("  python analyze_detailed_logs.py ./benchmark_logs/latest")
        sys.exit(1)

    run_path = sys.argv[1]

    # Handle "latest" shortcut
    if run_path.endswith("latest"):
        logs_dir = Path("./benchmark_logs")
        if not logs_dir.exists():
            print(f"Error: {logs_dir} does not exist")
            sys.exit(1)

        # Find latest run directory
        run_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
        if not run_dirs:
            print("Error: No benchmark runs found")
            sys.exit(1)

        run_dir = max(run_dirs, key=lambda d: d.stat().st_mtime)
        print(f"Using latest run: {run_dir.name}\n")
    else:
        run_dir = Path(run_path)

    if not run_dir.exists():
        print(f"Error: {run_dir} does not exist")
        sys.exit(1)

    # Generate and display report
    report = generate_analysis_report(run_dir)
    print(report)

    # Save report
    report_file = run_dir / "ANALYSIS.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print(f"\n📄 Analysis saved to: {report_file}")


if __name__ == "__main__":
    main()
