"""
GAAP Dashboard Module - Dashboard Data Provider

Provides dashboard data for monitoring and visualization:
- Token usage per task type
- Cost per success
- Average thought depth
- Failure rate per tool
- Grafana-compatible JSON output

Usage:
    from gaap.observability import DashboardProvider

    dashboard = DashboardProvider()
    dashboard.record_task_completion(task_type="code_generation", tokens=1000, cost=0.01, success=True)
    dashboard.record_tool_call(tool_name="read_file", success=True)

    # Get dashboard data
    data = dashboard.get_dashboard_data()

    # Get Grafana-compatible output
    grafana_json = dashboard.to_grafana_format()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

logger = logging.getLogger("gaap.observability.dashboard")


class TimeGranularity(Enum):
    """Time granularity for aggregations."""

    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


@dataclass
class TaskMetrics:
    """Metrics for a single task type."""

    task_type: str
    total_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    total_thought_depth: int = 0
    samples_with_depth: int = 0

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def avg_tokens_input(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_tokens_input / self.total_count

    @property
    def avg_tokens_output(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_tokens_output / self.total_count

    @property
    def avg_cost_usd(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_cost_usd / self.total_count

    @property
    def cost_per_success(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_cost_usd / self.success_count

    @property
    def avg_latency_ms(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.total_latency_ms / self.total_count

    @property
    def avg_thought_depth(self) -> float:
        if self.samples_with_depth == 0:
            return 0.0
        return self.total_thought_depth / self.samples_with_depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "total_count": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "avg_tokens_input": round(self.avg_tokens_input, 2),
            "avg_tokens_output": round(self.avg_tokens_output, 2),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_cost_usd": round(self.avg_cost_usd, 6),
            "cost_per_success": round(self.cost_per_success, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "avg_thought_depth": round(self.avg_thought_depth, 2),
        }


@dataclass
class ToolMetrics:
    """Metrics for a single tool."""

    tool_name: str
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.success_count / self.total_calls

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_ms / self.total_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""

    provider_name: str
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": round(self.success_rate, 4),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


@dataclass
class DashboardData:
    """Complete dashboard data snapshot."""

    timestamp: str
    uptime_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    overall_success_rate: float
    total_tokens_input: int
    total_tokens_output: int
    total_cost_usd: float
    active_sessions: int
    tasks: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    providers: list[dict[str, Any]]
    time_series: dict[str, list[dict[str, Any]]]
    alerts: list[dict[str, Any]]
    summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "overall_success_rate": round(self.overall_success_rate, 4),
            "total_tokens_input": self.total_tokens_input,
            "total_tokens_output": self.total_tokens_output,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "active_sessions": self.active_sessions,
            "tasks": self.tasks,
            "tools": self.tools,
            "providers": self.providers,
            "time_series": self.time_series,
            "alerts": self.alerts,
            "summary": self.summary,
        }


class DashboardProvider:
    """
    Dashboard data provider for GAAP monitoring.

    Collects and aggregates metrics for dashboard visualization:
    - Token usage per task type
    - Cost analysis
    - Tool performance
    - Provider performance
    - Time series data
    - Alert generation

    Usage:
        dashboard = DashboardProvider()

        # Record events
        dashboard.record_task_completion(
            task_type="code_generation",
            tokens_input=500,
            tokens_output=200,
            cost=0.015,
            latency_ms=1500,
            success=True,
            thought_depth=3
        )

        dashboard.record_tool_call("read_file", success=True, latency_ms=50)

        # Get data
        data = dashboard.get_dashboard_data()
        grafana = dashboard.to_grafana_format()
    """

    def __init__(
        self,
        history_window_hours: int = 24,
        alert_thresholds: dict[str, float] | None = None,
    ):
        self.history_window = timedelta(hours=history_window_hours)
        self.alert_thresholds = alert_thresholds or {
            "failure_rate": 0.2,
            "avg_latency_ms": 30000,
            "cost_per_hour": 10.0,
        }

        self._lock = threading.RLock()
        self._start_time = datetime.now()

        self._task_metrics: dict[str, TaskMetrics] = defaultdict(lambda: TaskMetrics(task_type=""))
        self._tool_metrics: dict[str, ToolMetrics] = defaultdict(lambda: ToolMetrics(tool_name=""))
        self._provider_metrics: dict[str, ProviderMetrics] = defaultdict(
            lambda: ProviderMetrics(provider_name="")
        )

        self._time_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._active_sessions = 0
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0

    def record_task_completion(
        self,
        task_type: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        success: bool = True,
        thought_depth: int | None = None,
        provider: str = "unknown",
    ) -> None:
        """
        Record a task completion event.

        Args:
            task_type: Type of task
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            cost: Cost in USD
            latency_ms: Latency in milliseconds
            success: Whether the task succeeded
            thought_depth: Optional thought depth (for ToT/GoT)
            provider: Provider name
        """
        with self._lock:
            metrics = self._task_metrics[task_type]
            metrics.task_type = task_type
            metrics.total_count += 1
            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1
            metrics.total_tokens_input += tokens_input
            metrics.total_tokens_output += tokens_output
            metrics.total_cost_usd += cost
            metrics.total_latency_ms += latency_ms

            if thought_depth is not None:
                metrics.total_thought_depth += thought_depth
                metrics.samples_with_depth += 1

            self._total_requests += 1
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            self._record_time_point(
                "task_completions",
                {
                    "task_type": task_type,
                    "success": success,
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "cost": cost,
                    "latency_ms": latency_ms,
                },
            )

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        latency_ms: float = 0.0,
    ) -> None:
        """
        Record a tool call event.

        Args:
            tool_name: Name of the tool
            success: Whether the call succeeded
            latency_ms: Latency in milliseconds
        """
        with self._lock:
            metrics = self._tool_metrics[tool_name]
            metrics.tool_name = tool_name
            metrics.total_calls += 1
            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1
            metrics.total_latency_ms += latency_ms

            self._record_time_point(
                "tool_calls",
                {
                    "tool_name": tool_name,
                    "success": success,
                    "latency_ms": latency_ms,
                },
            )

    def record_provider_request(
        self,
        provider_name: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
        success: bool = True,
    ) -> None:
        """
        Record a provider request.

        Args:
            provider_name: Name of the provider
            tokens_input: Input tokens
            tokens_output: Output tokens
            cost: Cost in USD
            latency_ms: Latency in milliseconds
            success: Whether the request succeeded
        """
        with self._lock:
            metrics = self._provider_metrics[provider_name]
            metrics.provider_name = provider_name
            metrics.total_requests += 1
            if success:
                metrics.success_count += 1
            else:
                metrics.failure_count += 1
            metrics.total_tokens_input += tokens_input
            metrics.total_tokens_output += tokens_output
            metrics.total_cost_usd += cost
            metrics.total_latency_ms += latency_ms

            self._record_time_point(
                "provider_requests",
                {
                    "provider": provider_name,
                    "success": success,
                    "tokens": tokens_input + tokens_output,
                    "cost": cost,
                    "latency_ms": latency_ms,
                },
            )

    def set_active_sessions(self, count: int) -> None:
        """Set the number of active sessions."""
        with self._lock:
            self._active_sessions = count

    def _record_time_point(self, series_name: str, data: dict[str, Any]) -> None:
        """Record a time series data point."""
        point = {
            "timestamp": datetime.now().isoformat(),
            **data,
        }
        self._time_series[series_name].append(point)
        self._prune_time_series(series_name)

    def _prune_time_series(self, series_name: str) -> None:
        """Remove old data points from time series."""
        cutoff = datetime.now() - self.history_window
        self._time_series[series_name] = [
            p
            for p in self._time_series[series_name]
            if datetime.fromisoformat(p["timestamp"]) >= cutoff
        ]

    def get_dashboard_data(self) -> DashboardData:
        """
        Get complete dashboard data.

        Returns:
            DashboardData with all metrics and aggregations
        """
        with self._lock:
            total_tokens_input = sum(m.total_tokens_input for m in self._task_metrics.values())
            total_tokens_output = sum(m.total_tokens_output for m in self._task_metrics.values())
            total_cost = sum(m.total_cost_usd for m in self._task_metrics.values())

            overall_success_rate = (
                self._successful_requests / self._total_requests
                if self._total_requests > 0
                else 0.0
            )

            alerts = self._generate_alerts()

            summary = self._generate_summary()

            return DashboardData(
                timestamp=datetime.now().isoformat(),
                uptime_seconds=(datetime.now() - self._start_time).total_seconds(),
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                overall_success_rate=overall_success_rate,
                total_tokens_input=total_tokens_input,
                total_tokens_output=total_tokens_output,
                total_cost_usd=total_cost,
                active_sessions=self._active_sessions,
                tasks=[
                    m.to_dict()
                    for m in sorted(
                        self._task_metrics.values(), key=lambda x: x.total_count, reverse=True
                    )
                ],
                tools=[
                    m.to_dict()
                    for m in sorted(
                        self._tool_metrics.values(), key=lambda x: x.total_calls, reverse=True
                    )
                ],
                providers=[
                    m.to_dict()
                    for m in sorted(
                        self._provider_metrics.values(),
                        key=lambda x: x.total_requests,
                        reverse=True,
                    )
                ],
                time_series=dict(self._time_series),
                alerts=alerts,
                summary=summary,
            )

    def _generate_alerts(self) -> list[dict[str, Any]]:
        """Generate alerts based on thresholds."""
        alerts = []

        for task_type, metrics in self._task_metrics.items():
            if metrics.failure_rate > self.alert_thresholds["failure_rate"]:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "high_failure_rate",
                        "source": f"task:{task_type}",
                        "message": f"High failure rate for {task_type}: {metrics.failure_rate:.2%}",
                        "value": metrics.failure_rate,
                        "threshold": self.alert_thresholds["failure_rate"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            if metrics.avg_latency_ms > self.alert_thresholds["avg_latency_ms"]:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "high_latency",
                        "source": f"task:{task_type}",
                        "message": f"High latency for {task_type}: {metrics.avg_latency_ms:.0f}ms",
                        "value": metrics.avg_latency_ms,
                        "threshold": self.alert_thresholds["avg_latency_ms"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        for tool_name, metrics in self._tool_metrics.items():
            if metrics.failure_rate > self.alert_thresholds["failure_rate"]:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "tool_failures",
                        "source": f"tool:{tool_name}",
                        "message": f"High failure rate for tool {tool_name}: {metrics.failure_rate:.2%}",
                        "value": metrics.failure_rate,
                        "threshold": self.alert_thresholds["failure_rate"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        return alerts

    def _generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        tasks = list(self._task_metrics.values())

        avg_thought_depth = 0.0
        total_depth_samples = sum(m.samples_with_depth for m in tasks)
        if total_depth_samples > 0:
            avg_thought_depth = sum(m.total_thought_depth for m in tasks) / total_depth_samples

        most_used_task = max(tasks, key=lambda x: x.total_count).task_type if tasks else "none"
        most_expensive_task = (
            max(tasks, key=lambda x: x.total_cost_usd).task_type if tasks else "none"
        )

        tools = list(self._tool_metrics.values())
        most_used_tool = max(tools, key=lambda x: x.total_calls).tool_name if tools else "none"
        failing_tools = [t.tool_name for t in tools if t.failure_rate > 0.1]

        providers = list(self._provider_metrics.values())
        best_provider = (
            max(providers, key=lambda x: x.success_rate).provider_name if providers else "none"
        )

        return {
            "avg_thought_depth": round(avg_thought_depth, 2),
            "most_used_task_type": most_used_task,
            "most_expensive_task_type": most_expensive_task,
            "most_used_tool": most_used_tool,
            "failing_tools": failing_tools,
            "best_provider": best_provider,
            "total_task_types": len(tasks),
            "total_tools_used": len(tools),
            "total_providers_used": len(providers),
        }

    def to_grafana_format(self) -> dict[str, Any]:
        """
        Convert dashboard data to Grafana-compatible format.

        Returns:
            Dictionary in Grafana annotation/panel format
        """
        data = self.get_dashboard_data()

        panels = []

        panels.append(
            {
                "title": "Request Overview",
                "type": "stat",
                "targets": [
                    {
                        "refId": "A",
                        "datapoints": [
                            [data.total_requests, int(time.time() * 1000)],
                        ],
                    }
                ],
                "fieldConfig": {"defaults": {"displayName": "Total Requests"}},
            }
        )

        panels.append(
            {
                "title": "Success Rate",
                "type": "gauge",
                "targets": [
                    {
                        "refId": "A",
                        "datapoints": [
                            [data.overall_success_rate * 100, int(time.time() * 1000)],
                        ],
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "unit": "percent",
                        "min": 0,
                        "max": 100,
                        "thresholds": {
                            "mode": "absolute",
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "yellow", "value": 80},
                                {"color": "green", "value": 95},
                            ],
                        },
                    }
                },
            }
        )

        panels.append(
            {
                "title": "Token Usage",
                "type": "timeseries",
                "targets": [
                    {
                        "refId": "A",
                        "legendFormat": "Input Tokens",
                        "datapoints": [
                            [
                                p.get("tokens_input", 0),
                                int(datetime.fromisoformat(p["timestamp"]).timestamp() * 1000),
                            ]
                            for p in data.time_series.get("task_completions", [])[-100:]
                        ],
                    },
                    {
                        "refId": "B",
                        "legendFormat": "Output Tokens",
                        "datapoints": [
                            [
                                p.get("tokens_output", 0),
                                int(datetime.fromisoformat(p["timestamp"]).timestamp() * 1000),
                            ]
                            for p in data.time_series.get("task_completions", [])[-100:]
                        ],
                    },
                ],
            }
        )

        panels.append(
            {
                "title": "Cost Over Time",
                "type": "timeseries",
                "targets": [
                    {
                        "refId": "A",
                        "legendFormat": "Cost (USD)",
                        "datapoints": [
                            [
                                p.get("cost", 0),
                                int(datetime.fromisoformat(p["timestamp"]).timestamp() * 1000),
                            ]
                            for p in data.time_series.get("task_completions", [])[-100:]
                        ],
                    }
                ],
                "fieldConfig": {"defaults": {"unit": "currencyUSD"}},
            }
        )

        panels.append(
            {
                "title": "Tool Performance",
                "type": "table",
                "targets": [
                    {
                        "refId": "A",
                        "format": "table",
                        "frames": [
                            {
                                "schema": {
                                    "fields": [
                                        {"name": "Tool", "type": "string"},
                                        {"name": "Calls", "type": "number"},
                                        {"name": "Success Rate", "type": "number"},
                                        {"name": "Avg Latency (ms)", "type": "number"},
                                    ]
                                },
                                "data": {
                                    "values": [
                                        [t["tool_name"] for t in data.tools],
                                        [t["total_calls"] for t in data.tools],
                                        [round(t["success_rate"] * 100, 1) for t in data.tools],
                                        [t["avg_latency_ms"] for t in data.tools],
                                    ]
                                },
                            }
                        ],
                    }
                ],
            }
        )

        panels.append(
            {
                "title": "Task Performance",
                "type": "table",
                "targets": [
                    {
                        "refId": "A",
                        "format": "table",
                        "frames": [
                            {
                                "schema": {
                                    "fields": [
                                        {"name": "Task Type", "type": "string"},
                                        {"name": "Count", "type": "number"},
                                        {"name": "Success Rate", "type": "number"},
                                        {"name": "Avg Tokens", "type": "number"},
                                        {"name": "Cost/Success", "type": "number"},
                                        {"name": "Avg Depth", "type": "number"},
                                    ]
                                },
                                "data": {
                                    "values": [
                                        [t["task_type"] for t in data.tasks],
                                        [t["total_count"] for t in data.tasks],
                                        [round(t["success_rate"] * 100, 1) for t in data.tasks],
                                        [
                                            t["avg_tokens_input"] + t["avg_tokens_output"]
                                            for t in data.tasks
                                        ],
                                        [t["cost_per_success"] for t in data.tasks],
                                        [t["avg_thought_depth"] for t in data.tasks],
                                    ]
                                },
                            }
                        ],
                    }
                ],
            }
        )

        annotations = []
        for alert in data.alerts:
            annotations.append(
                {
                    "time": int(datetime.fromisoformat(alert["timestamp"]).timestamp() * 1000),
                    "title": alert["type"],
                    "text": alert["message"],
                    "tags": [alert["level"], alert["source"]],
                }
            )

        return {
            "dashboard": {
                "title": "GAAP System Dashboard",
                "uid": "gaap-main",
                "panels": panels,
                "time": {
                    "from": f"now-{self.history_window.total_seconds() // 3600}h",
                    "to": "now",
                },
                "refresh": "30s",
            },
            "annotations": annotations,
            "metadata": {
                "generated_at": data.timestamp,
                "uptime_seconds": data.uptime_seconds,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Get dashboard data as JSON string."""
        return json.dumps(self.get_dashboard_data().to_dict(), indent=indent, default=str)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._task_metrics.clear()
            self._tool_metrics.clear()
            self._provider_metrics.clear()
            self._time_series.clear()
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._start_time = datetime.now()
            logger.info("Dashboard metrics reset")
