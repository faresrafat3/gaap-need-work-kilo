"""
Detailed Benchmark Logger
==========================

Comprehensive logging system for benchmark runs that captures:
- Every question and answer
- All API calls and retries
- Key rotation events
- Performance metrics per question
- System state and configuration
- Errors and warnings
- Complete timeline

Creates a structured documentation for future reference and analysis.
"""

import json
import logging
import shutil
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class QuestionLog:
    """Log entry for a single question"""

    question_id: int
    dataset: str
    question_text: str
    choices: list[tuple]
    correct_answer: str

    # Direct model results
    direct_answer: str
    direct_correct: bool
    direct_latency_ms: float
    direct_cost_usd: float
    direct_tokens: int

    # GAAP results
    gaap_answer: str
    gaap_correct: bool
    gaap_latency_ms: float
    gaap_cost_usd: float
    gaap_tokens: int

    # Metadata (with defaults)
    direct_attempts: int = 1
    gaap_attempts: int = 1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    key_rotations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkRunMetadata:
    """Metadata for the entire benchmark run"""

    run_id: str
    start_time: str
    end_time: str | None = None

    # Configuration
    dataset: str = ""
    samples: int = 0
    gaap_mode: str = ""
    seed: int = 42

    # System configuration
    python_version: str = ""
    gaap_version: str = ""

    # Provider configuration
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    num_api_keys: int = 0

    # Layer configuration (for full/full_lite modes)
    layer_config: dict[str, Any] = field(default_factory=dict)

    # Environment
    enable_all: bool = False
    budget: float = 0.0

    # Results summary
    total_questions: int = 0
    completed_questions: int = 0
    failed_questions: int = 0

    # Performance
    total_duration_seconds: float = 0.0
    avg_time_per_question: float = 0.0

    # Errors
    total_errors: int = 0
    total_warnings: int = 0
    total_key_rotations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetailedBenchmarkLogger:
    """
    Comprehensive benchmark logger that captures everything
    """

    def __init__(self, output_dir: str = "./benchmark_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = f"run_{int(time.time())}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = BenchmarkRunMetadata(
            run_id=self.run_id, start_time=datetime.now().isoformat()
        )

        self.question_logs: list[QuestionLog] = []
        self.event_log: list[dict[str, Any]] = []

        # Setup logging
        self._setup_logging()

        self.logger.info(f"Benchmark run started: {self.run_id}")

    def _setup_logging(self):
        """Setup file and console logging"""
        self.logger = logging.getLogger(f"benchmark.{self.run_id}")
        self.logger.setLevel(logging.DEBUG)

        # File handler - detailed
        fh = logging.FileHandler(self.run_dir / "detailed.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # File handler - errors only
        eh = logging.FileHandler(self.run_dir / "errors.log")
        eh.setLevel(logging.WARNING)
        eh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n"
            )
        )

        self.logger.addHandler(fh)
        self.logger.addHandler(eh)

    def configure(self, **kwargs):
        """Update metadata configuration"""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)

        self.logger.info(f"Configuration updated: {kwargs}")
        self._save_metadata()

    def log_question(self, question_log: QuestionLog):
        """Log a completed question"""
        self.question_logs.append(question_log)
        self.metadata.completed_questions += 1

        # Update error/warning counts
        self.metadata.total_errors += len(question_log.errors)
        self.metadata.total_warnings += len(question_log.warnings)
        self.metadata.total_key_rotations += len(question_log.key_rotations)

        self.logger.info(
            f"Question {question_log.question_id} completed: "
            f"Direct={question_log.direct_correct}, GAAP={question_log.gaap_correct}"
        )

        # Save incrementally every 10 questions
        if len(self.question_logs) % 10 == 0:
            self._save_all()

    def log_event(self, event_type: str, details: dict[str, Any]):
        """Log a system event"""
        event = {"timestamp": datetime.now().isoformat(), "type": event_type, "details": details}
        self.event_log.append(event)

        self.logger.debug(f"Event: {event_type} - {details}")

    def log_error(self, error: Exception, context: str = ""):
        """Log an error with full traceback"""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }

        self.metadata.total_errors += 1
        self.logger.error(f"Error in {context}: {error}", exc_info=True)

        # Save to separate error log
        with open(self.run_dir / "errors.json", "a") as f:
            json.dump(error_info, f, indent=2)
            f.write("\n")

    def log_key_rotation(self, from_index: int, to_index: int, reason: str = ""):
        """Log API key rotation event"""
        rotation = {
            "timestamp": datetime.now().isoformat(),
            "from_index": from_index,
            "to_index": to_index,
            "reason": reason,
        }

        self.log_event("key_rotation", rotation)
        self.metadata.total_key_rotations += 1

    def finalize(self):
        """Finalize the benchmark run and save all data"""
        self.metadata.end_time = datetime.now().isoformat()
        self.metadata.total_questions = len(self.question_logs)

        # Calculate durations
        start = datetime.fromisoformat(self.metadata.start_time)
        end = datetime.fromisoformat(self.metadata.end_time)
        self.metadata.total_duration_seconds = (end - start).total_seconds()

        if self.metadata.completed_questions > 0:
            self.metadata.avg_time_per_question = (
                self.metadata.total_duration_seconds / self.metadata.completed_questions
            )

        self._save_all()
        self._generate_reports()
        self._auto_archive_old_runs()

        self.logger.info(f"Benchmark run completed: {self.run_id}")
        self.logger.info(f"Total duration: {self.metadata.total_duration_seconds:.2f}s")
        self.logger.info(
            f"Completed questions: {self.metadata.completed_questions}/{self.metadata.total_questions}"
        )

    def _auto_archive_old_runs(self):
        """
        Auto-archive old run folders, keeping only the latest N runs.

        Controls via env vars:
          - GAAP_LOG_RETENTION_ENABLED: 1/0 (default: 1)
          - GAAP_LOG_RETENTION_KEEP: number of runs to keep (default: 8)
          - GAAP_LOG_ARCHIVE_DIR: archive folder name (default: _archive)
        """
        try:
            import os

            enabled = os.getenv("GAAP_LOG_RETENTION_ENABLED", "1").strip().lower() not in {
                "0",
                "false",
                "no",
            }
            if not enabled:
                return

            keep = int(os.getenv("GAAP_LOG_RETENTION_KEEP", "8"))
            keep = max(1, keep)
            archive_name = os.getenv("GAAP_LOG_ARCHIVE_DIR", "_archive").strip() or "_archive"

            archive_dir = self.output_dir / archive_name
            archive_dir.mkdir(parents=True, exist_ok=True)

            runs = sorted(
                [p for p in self.output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")],
                key=lambda p: p.name,
                reverse=True,
            )

            to_archive = runs[keep:]
            moved = 0
            for run_dir in to_archive:
                if run_dir == self.run_dir:
                    continue
                target = archive_dir / run_dir.name
                if target.exists():
                    shutil.rmtree(target)
                shutil.move(str(run_dir), str(target))
                moved += 1

            if moved:
                self.logger.info(f"Auto-archived {moved} old runs (keep={keep})")
        except Exception as e:
            self.logger.warning(f"Auto-archive skipped due to error: {e}")

    def _save_metadata(self):
        """Save metadata to JSON"""
        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def _save_question_logs(self):
        """Save all question logs to JSON"""
        with open(self.run_dir / "questions.json", "w") as f:
            json.dump([q.to_dict() for q in self.question_logs], f, indent=2)

        # Also save as JSONL for easier processing
        with open(self.run_dir / "questions.jsonl", "w") as f:
            for q in self.question_logs:
                json.dump(q.to_dict(), f)
                f.write("\n")

    def _save_event_log(self):
        """Save event log"""
        with open(self.run_dir / "events.json", "w") as f:
            json.dump(self.event_log, f, indent=2)

    def _save_all(self):
        """Save all data"""
        self._save_metadata()
        self._save_question_logs()
        self._save_event_log()

    def _generate_reports(self):
        """Generate human-readable reports"""

        # Summary report
        summary = self._generate_summary_report()
        with open(self.run_dir / "SUMMARY.md", "w") as f:
            f.write(summary)

        # Detailed report
        detailed = self._generate_detailed_report()
        with open(self.run_dir / "DETAILED.md", "w") as f:
            f.write(detailed)

        # Performance report
        performance = self._generate_performance_report()
        with open(self.run_dir / "PERFORMANCE.md", "w") as f:
            f.write(performance)

        # README with navigation
        readme = self._generate_readme()
        with open(self.run_dir / "README.md", "w") as f:
            f.write(readme)

    def _generate_summary_report(self) -> str:
        """Generate summary report"""
        lines = []
        lines.append("# Benchmark Run Summary")
        lines.append(f"**Run ID**: `{self.run_id}`")
        lines.append(f"**Date**: {self.metadata.start_time}")
        lines.append("")

        lines.append("## Configuration")
        lines.append(f"- Dataset: {self.metadata.dataset}")
        lines.append(f"- Samples: {self.metadata.samples}")
        lines.append(f"- Mode: {self.metadata.gaap_mode}")
        lines.append(f"- Model: {self.metadata.model}")
        lines.append(f"- API Keys: {self.metadata.num_api_keys}")
        lines.append("")

        lines.append("## Results")
        correct_direct = sum(1 for q in self.question_logs if q.direct_correct)
        correct_gaap = sum(1 for q in self.question_logs if q.gaap_correct)
        total = len(self.question_logs)

        lines.append(
            f"- Completed: {self.metadata.completed_questions}/{self.metadata.total_questions}"
        )
        lines.append(
            f"- Direct Accuracy: {correct_direct}/{total} ({correct_direct/max(total,1)*100:.1f}%)"
        )
        lines.append(
            f"- GAAP Accuracy: {correct_gaap}/{total} ({correct_gaap/max(total,1)*100:.1f}%)"
        )
        lines.append(f"- Improvement: {(correct_gaap-correct_direct)/max(total,1)*100:.1f}%")
        lines.append("")

        lines.append("## Performance")
        lines.append(f"- Total Duration: {self.metadata.total_duration_seconds:.2f}s")
        lines.append(f"- Avg per Question: {self.metadata.avg_time_per_question:.2f}s")
        lines.append("")

        lines.append("## System Events")
        lines.append(f"- Errors: {self.metadata.total_errors}")
        lines.append(f"- Warnings: {self.metadata.total_warnings}")
        lines.append(f"- Key Rotations: {self.metadata.total_key_rotations}")
        lines.append("")

        return "\n".join(lines)

    def _generate_detailed_report(self) -> str:
        """Generate detailed question-by-question report"""
        lines = []
        lines.append("# Detailed Question Report")
        lines.append(f"Run: `{self.run_id}`")
        lines.append("")

        for q in self.question_logs:
            lines.append(f"## Question {q.question_id}")
            lines.append(f"**Correct Answer**: {q.correct_answer}")
            lines.append("")

            lines.append(f"**Question**: {q.question_text[:200]}...")
            lines.append("")

            lines.append("### Results")
            lines.append(f"- Direct: {q.direct_answer} {'✅' if q.direct_correct else '❌'}")
            lines.append(f"- GAAP: {q.gaap_answer} {'✅' if q.gaap_correct else '❌'}")
            lines.append("")

            lines.append("### Performance")
            lines.append(f"- Direct Latency: {q.direct_latency_ms:.0f}ms")
            lines.append(f"- GAAP Latency: {q.gaap_latency_ms:.0f}ms")
            lines.append(f"- Direct Cost: ${q.direct_cost_usd:.6f}")
            lines.append(f"- GAAP Cost: ${q.gaap_cost_usd:.6f}")
            lines.append("")

            if q.errors:
                lines.append("### Errors")
                for error in q.errors:
                    lines.append(f"- {error}")
                lines.append("")

            if q.key_rotations:
                lines.append(f"### Key Rotations: {len(q.key_rotations)}")
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _generate_performance_report(self) -> str:
        """Generate performance analysis report"""
        lines = []
        lines.append("# Performance Analysis")
        lines.append(f"Run: `{self.run_id}`")
        lines.append("")

        if not self.question_logs:
            return "\n".join(lines)

        # Latency analysis
        direct_latencies = [q.direct_latency_ms for q in self.question_logs]
        gaap_latencies = [q.gaap_latency_ms for q in self.question_logs]

        lines.append("## Latency Distribution")
        lines.append(f"- Direct Avg: {sum(direct_latencies)/len(direct_latencies):.0f}ms")
        lines.append(f"- GAAP Avg: {sum(gaap_latencies)/len(gaap_latencies):.0f}ms")
        lines.append(
            f"- Direct Min/Max: {min(direct_latencies):.0f}ms / {max(direct_latencies):.0f}ms"
        )
        lines.append(f"- GAAP Min/Max: {min(gaap_latencies):.0f}ms / {max(gaap_latencies):.0f}ms")
        lines.append("")

        # Cost analysis
        total_direct_cost = sum(q.direct_cost_usd for q in self.question_logs)
        total_gaap_cost = sum(q.gaap_cost_usd for q in self.question_logs)

        lines.append("## Cost Analysis")
        lines.append(f"- Total Direct: ${total_direct_cost:.4f}")
        lines.append(f"- Total GAAP: ${total_gaap_cost:.4f}")
        lines.append(f"- Per Question (Direct): ${total_direct_cost/len(self.question_logs):.6f}")
        lines.append(f"- Per Question (GAAP): ${total_gaap_cost/len(self.question_logs):.6f}")
        lines.append("")

        # Token analysis
        total_direct_tokens = sum(q.direct_tokens for q in self.question_logs)
        total_gaap_tokens = sum(q.gaap_tokens for q in self.question_logs)

        lines.append("## Token Usage")
        lines.append(f"- Total Direct: {total_direct_tokens:,}")
        lines.append(f"- Total GAAP: {total_gaap_tokens:,}")
        lines.append(f"- Per Question (Direct): {total_direct_tokens/len(self.question_logs):.0f}")
        lines.append(f"- Per Question (GAAP): {total_gaap_tokens/len(self.question_logs):.0f}")
        lines.append("")

        return "\n".join(lines)

    def _generate_readme(self) -> str:
        """Generate README for the run"""
        lines = []
        lines.append(f"# Benchmark Run: {self.run_id}")
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append("")

        lines.append("## Files in this directory")
        lines.append("")
        lines.append("### Data Files (JSON)")
        lines.append("- `metadata.json` - Run configuration and summary statistics")
        lines.append("- `questions.json` - All question logs with full details")
        lines.append("- `questions.jsonl` - Question logs in JSONL format (one per line)")
        lines.append("- `events.json` - System events timeline")
        lines.append("- `errors.json` - Detailed error logs")
        lines.append("")

        lines.append("### Reports (Markdown)")
        lines.append("- `SUMMARY.md` - High-level summary of the run")
        lines.append("- `DETAILED.md` - Question-by-question detailed report")
        lines.append("- `PERFORMANCE.md` - Performance analysis and metrics")
        lines.append("")

        lines.append("### Logs")
        lines.append("- `detailed.log` - Complete debug-level logs")
        lines.append("- `errors.log` - Errors and warnings only")
        lines.append("")

        lines.append("## Quick Stats")
        lines.append(f"- Dataset: {self.metadata.dataset}")
        lines.append(
            f"- Questions: {self.metadata.completed_questions}/{self.metadata.total_questions}"
        )
        lines.append(f"- Duration: {self.metadata.total_duration_seconds:.2f}s")
        lines.append("")

        return "\n".join(lines)


def create_logger(output_dir: str = "./benchmark_logs") -> DetailedBenchmarkLogger:
    """Create and return a detailed benchmark logger"""
    return DetailedBenchmarkLogger(output_dir)
