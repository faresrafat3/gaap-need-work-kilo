#!/usr/bin/env python3
"""
Cost Monitor - Token Usage & Cost Tracking
============================================

Tracks token usage across LLM calls and generates reports.

Implements: docs/evolution_plan_2026/27_OPS_AND_CI.md

Usage:
    python scripts/cost_monitor.py --report daily
    python scripts/cost_monitor.py --alert-threshold 10.0
    python scripts/cost_monitor.py --export report.json
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


@dataclass
class UsageRecord:
    """Token record for a single LLM call"""

    timestamp: datetime
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }


@dataclass
class CostReport:
    """Cost report for a time period"""

    period_start: datetime
    period_end: datetime
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_calls: int = 0
    by_provider: dict[str, dict[str, float | int]] = field(default_factory=dict)
    by_model: dict[str, dict[str, float | int]] = field(default_factory=dict)
    avg_cost_per_call: float = 0.0
    avg_latency_estimate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "total_calls": self.total_calls,
            "by_provider": self.by_provider,
            "by_model": self.by_model,
            "avg_cost_per_call": round(self.avg_cost_per_call, 4),
            "avg_latency_estimate": round(self.avg_latency_estimate, 2),
        }


class CostMonitor:
    """
    Monitors and reports on LLM token usage and costs.

    Features:
    - Track usage records
    - Generate daily/weekly/monthly reports
    - Alert on threshold breaches
    - Export reports to JSON
    - Integration with CI guardrails
    """

    DEFAULT_STORAGE_PATH = ".gaap/costs"
    DEFAULT_ALERT_THRESHOLD = 50.0

    MODEL_PRICING = {
        "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4-turbo": {"input": 10.00 / 1_000_000, "output": 30.00 / 1_000_000},
        "gpt-3.5-turbo": {"input": 0.50 / 1_000_000, "output": 1.50 / 1_000_000},
        "claude-3-opus": {"input": 15.00 / 1_000_000, "output": 75.00 / 1_000_000},
        "claude-3-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
        "gemini-1.5-pro": {"input": 1.25 / 1_000_000, "output": 5.00 / 1_000_000},
        "gemini-1.5-flash": {"input": 0.075 / 1_000_000, "output": 0.30 / 1_000_000},
    }

    def __init__(self, storage_path: str | None = None):
        self.storage_path = Path(storage_path or self.DEFAULT_STORAGE_PATH)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._records: list[UsageRecord] = []
        self._load()

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UsageRecord:
        """
        Record a usage event.

        Args:
            provider: LLM provider name
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            task_id: Optional associated task ID
            metadata: Optional additional metadata

        Returns:
            Created UsageRecord
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        record = UsageRecord(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_id=task_id,
            metadata=metadata or {},
        )

        self._records.append(record)
        self._save()

        return record

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model pricing."""
        model_key = model.lower()

        sorted_pricing = sorted(self.MODEL_PRICING.items(), key=lambda x: len(x[0]), reverse=True)

        for key, pricing in sorted_pricing:
            if key.lower() == model_key or key.lower() in model_key:
                return input_tokens * pricing["input"] + output_tokens * pricing["output"]

        default_input = 1.0 / 1_000_000
        default_output = 3.0 / 1_000_000
        return input_tokens * default_input + output_tokens * default_output

    def generate_report(
        self,
        period: str = "daily",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> CostReport:
        """
        Generate a cost report for a time period.

        Args:
            period: 'daily', 'weekly', or 'monthly'
            start: Optional start datetime
            end: Optional end datetime

        Returns:
            CostReport for the period
        """
        end = end or datetime.now()

        if start is None:
            if period == "daily":
                start = end - timedelta(days=1)
            elif period == "weekly":
                start = end - timedelta(weeks=1)
            elif period == "monthly":
                start = end - timedelta(days=30)
            else:
                start = end - timedelta(days=1)

        filtered = [r for r in self._records if start <= r.timestamp <= end]

        report = CostReport(period_start=start, period_end=end)

        for record in filtered:
            report.total_input_tokens += record.input_tokens
            report.total_output_tokens += record.output_tokens
            report.total_cost_usd += record.cost_usd
            report.total_calls += 1

            if record.provider not in report.by_provider:
                report.by_provider[record.provider] = {
                    "calls": 0,
                    "cost": 0.0,
                    "tokens": 0,
                }
            report.by_provider[record.provider]["calls"] += 1
            report.by_provider[record.provider]["cost"] += record.cost_usd
            report.by_provider[record.provider]["tokens"] += (
                record.input_tokens + record.output_tokens
            )

            if record.model not in report.by_model:
                report.by_model[record.model] = {"calls": 0, "cost": 0.0, "tokens": 0}
            report.by_model[record.model]["calls"] += 1
            report.by_model[record.model]["cost"] += record.cost_usd
            report.by_model[record.model]["tokens"] += record.input_tokens + record.output_tokens

        if report.total_calls > 0:
            report.avg_cost_per_call = report.total_cost_usd / report.total_calls
            report.avg_latency_estimate = 2.0

        return report

    def check_alerts(self, threshold: float | None = None) -> list[str]:
        """
        Check for alert conditions.

        Args:
            threshold: Cost threshold in USD

        Returns:
            List of alert messages
        """
        threshold = threshold or self.DEFAULT_ALERT_THRESHOLD
        alerts = []

        daily_report = self.generate_report("daily")

        if daily_report.total_cost_usd > threshold:
            alerts.append(
                f"DAILY_COST_ALERT: ${daily_report.total_cost_usd:.2f} "
                f"exceeds threshold ${threshold:.2f}"
            )

        if daily_report.avg_cost_per_call > 0.10:
            alerts.append(
                f"HIGH_AVG_COST: Average cost per call ${daily_report.avg_cost_per_call:.3f} "
                f"is unusually high"
            )

        return alerts

    def export(self, filepath: str | Path | None = None) -> Path:
        """
        Export usage data to JSON.

        Args:
            filepath: Optional export path

        Returns:
            Path to exported file
        """
        filepath = Path(filepath or self.storage_path / "usage_export.json")

        data = {
            "exported_at": datetime.now().isoformat(),
            "total_records": len(self._records),
            "records": [r.to_dict() for r in self._records[-1000:]],
            "summary": self.generate_report("monthly").to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return filepath

    def get_stats(self) -> dict[str, Any]:
        """Get quick statistics."""
        return {
            "total_records": len(self._records),
            "daily_report": self.generate_report("daily").to_dict(),
            "weekly_report": self.generate_report("weekly").to_dict(),
        }

    def _save(self) -> None:
        """Save records to disk."""
        filepath = self.storage_path / "usage_records.json"

        data = {
            "records": [r.to_dict() for r in self._records[-10000:]],
            "last_updated": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load records from disk."""
        filepath = self.storage_path / "usage_records.json"

        if not filepath.exists():
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            for r in data.get("records", []):
                self._records.append(
                    UsageRecord(
                        timestamp=datetime.fromisoformat(r["timestamp"]),
                        provider=r["provider"],
                        model=r["model"],
                        input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"],
                        cost_usd=r["cost_usd"],
                        task_id=r.get("task_id"),
                        metadata=r.get("metadata", {}),
                    )
                )
        except Exception:
            pass


def print_report(report: CostReport) -> None:
    """Print a formatted report to stdout."""
    print("\n" + "=" * 60)
    print(
        f"  COST REPORT: {report.period_start.strftime('%Y-%m-%d')} to {report.period_end.strftime('%Y-%m-%d')}"
    )
    print("=" * 60)

    print(f"\n  Total Calls:        {report.total_calls}")
    print(f"  Total Input Tokens: {report.total_input_tokens:,}")
    print(f"  Total Output Tokens:{report.total_output_tokens:,}")
    print(f"  Total Cost:         ${report.total_cost_usd:.4f}")
    print(f"  Avg Cost/Call:      ${report.avg_cost_per_call:.4f}")

    if report.by_provider:
        print("\n  By Provider:")
        for provider, stats in sorted(
            report.by_provider.items(), key=lambda x: x[1]["cost"], reverse=True
        ):
            print(f"    - {provider}: ${stats['cost']:.4f} ({stats['calls']} calls)")

    if report.by_model:
        print("\n  By Model:")
        for model, stats in sorted(
            report.by_model.items(), key=lambda x: x[1]["cost"], reverse=True
        )[:5]:
            print(f"    - {model}: ${stats['cost']:.4f} ({stats['calls']} calls)")

    print("\n" + "=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="GAAP Cost Monitor")
    parser.add_argument(
        "--report",
        choices=["daily", "weekly", "monthly"],
        default="daily",
        help="Report period",
    )
    parser.add_argument("--alert-threshold", type=float, help="Cost alert threshold (USD)")
    parser.add_argument("--export", type=str, help="Export to JSON file")
    parser.add_argument(
        "--record",
        nargs=4,
        metavar=("PROVIDER", "MODEL", "INPUT_TOKENS", "OUTPUT_TOKENS"),
        help="Record a new usage event",
    )

    args = parser.parse_args()

    monitor = CostMonitor()

    if args.record:
        provider, model, input_tok, output_tok = args.record
        record = monitor.record(
            provider=provider,
            model=model,
            input_tokens=int(input_tok),
            output_tokens=int(output_tok),
        )
        print(f"Recorded: ${record.cost_usd:.6f}")
        return

    if args.export:
        path = monitor.export(args.export)
        print(f"Exported to: {path}")
        return

    report = monitor.generate_report(args.report)
    print_report(report)

    threshold = args.alert_threshold or CostMonitor.DEFAULT_ALERT_THRESHOLD
    alerts = monitor.check_alerts(threshold)

    if alerts:
        print("\n⚠️  ALERTS:")
        for alert in alerts:
            print(f"  - {alert}")
        sys.exit(1)


if __name__ == "__main__":
    main()
