"""
Rich TUI Components for GAAP CLI
===============================

Provides beautiful terminal interface with:
- Live streaming of LLM responses
- Brain activity spinner
- OODA loop status display
- Diff preview for file changes
- Steering mode (pause/resume)

Reference: docs/evolution_plan_2026/44_CLI_AUDIT_SPEC.md
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Generator

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


class OODAStage(Enum):
    """OODA Loop stages"""

    OBSERVE = auto()
    ORIENT = auto()
    DECIDE = auto()
    ACT = auto()
    LEARN = auto()


@dataclass
class BrainState:
    """Current brain activity state"""

    stage: OODAStage = OODAStage.OBSERVE
    task: str = "Initializing..."
    tokens_used: int = 0
    budget_remaining: float = 1.0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class BrainActivityDisplay:
    """
    Displays live brain activity with spinner and status
    """

    STAGE_NAMES = {
        OODAStage.OBSERVE: ("👁️ Observing", "cyan"),
        OODAStage.ORIENT: ("🧭 Orienting", "yellow"),
        OODAStage.DECIDE: ("🤔 Deciding", "magenta"),
        OODAStage.ACT: ("⚡ Acting", "green"),
        OODAStage.LEARN: ("📚 Learning", "blue"),
    }

    def __init__(self) -> None:
        self.state = BrainState()
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._frame_idx = 0

    def update(self, stage: OODAStage, task: str, **kwargs: Any) -> None:
        """Update brain state"""
        self.state.stage = stage
        self.state.task = task
        if "tokens_used" in kwargs:
            self.state.tokens_used = kwargs["tokens_used"]
        if "budget_remaining" in kwargs:
            self.state.budget_remaining = kwargs["budget_remaining"]

    def render(self) -> Panel:
        """Render brain activity panel"""
        stage_name, stage_color = self.STAGE_NAMES.get(self.state.stage, ("❓ Unknown", "white"))
        spinner = self._spinner_frames[self._frame_idx % len(self._spinner_frames)]
        self._frame_idx += 1

        content = Text()
        content.append(f"{spinner} ", style="bold")
        content.append(f"{stage_name}", style=f"bold {stage_color}")
        content.append(f"\n  {self.state.task}")

        elapsed_str = f"{self.state.elapsed:.1f}s"
        budget_str = f"${self.state.budget_remaining:.2f}"
        tokens_str = f"{self.state.tokens_used:,}"

        footer = Text()
        footer.append(f"⏱ {elapsed_str}  ", style="dim")
        footer.append(f"💰 {budget_str}  ", style="dim")
        footer.append(f"📊 {tokens_str} tokens", style="dim")

        return Panel(
            Group(content, footer),
            title="[bold]🧠 Brain Activity[/bold]",
            border_style=stage_color,
            padding=(0, 1),
        )


class StreamingResponse:
    """
    Streams LLM response word-by-word with syntax highlighting
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._complete = False

    def add_chunk(self, chunk: str) -> None:
        """Add a chunk to the buffer"""
        self._buffer.append(chunk)

    def complete(self) -> None:
        """Mark response as complete"""
        self._complete = True

    def render(self) -> Panel:
        """Render the streamed response"""
        text = "".join(self._buffer)

        if self._buffer and self._buffer[0].startswith("```"):
            lines = text.split("\n")
            if lines:
                lang = lines[0].replace("```", "").strip() or "python"
                code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                try:
                    content = Syntax(code, lang, theme="monokai", line_numbers=True)
                except Exception:
                    content = Text(text)
            else:
                content = Markdown(text)
        else:
            content = Markdown(text)

        status = "✅" if self._complete else "📡 Streaming..."
        return Panel(
            content,
            title=f"[bold]🤖 GAAP Response[/bold] {status}",
            border_style="green" if self._complete else "yellow",
            padding=(0, 1),
        )


class OODAStatusDisplay:
    """
    Displays OODA loop status in a table
    """

    def __init__(self) -> None:
        self.stages = {
            OODAStage.OBSERVE: {"status": "pending", "message": ""},
            OODAStage.ORIENT: {"status": "pending", "message": ""},
            OODAStage.DECIDE: {"status": "pending", "message": ""},
            OODAStage.ACT: {"status": "pending", "message": ""},
            OODAStage.LEARN: {"status": "pending", "message": ""},
        }

    def set_stage(self, stage: OODAStage, status: str, message: str = "") -> None:
        """Update a stage status"""
        self.stages[stage] = {"status": status, "message": message}

    def render(self) -> Table:
        """Render OODA status table"""
        table = Table(title="OODA Loop Status", show_header=True, header_style="bold")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message", style="dim")

        status_icons = {
            "pending": ("⏳", "dim"),
            "active": ("🔄", "yellow"),
            "complete": ("✅", "green"),
            "failed": ("❌", "red"),
        }

        stage_names = {
            OODAStage.OBSERVE: "Observe",
            OODAStage.ORIENT: "Orient",
            OODAStage.DECIDE: "Decide",
            OODAStage.ACT: "Act",
            OODAStage.LEARN: "Learn",
        }

        for stage in OODAStage:
            info = self.stages.get(stage, {"status": "pending", "message": ""})
            icon, color = status_icons.get(info["status"], ("❓", "white"))
            table.add_row(
                stage_names[stage],
                f"[{color}]{icon}[/{color}]",
                info["message"][:40] if info["message"] else "-",
            )

        return table


class DiffPreview:
    """
    Shows side-by-side diff before file changes
    """

    @staticmethod
    def render(old_content: str, new_content: str, filename: str) -> Panel:
        """Render a diff preview"""
        from difflib import unified_diff

        diff_lines = list(
            unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"{filename} (current)",
                tofile=f"{filename} (new)",
            )
        )

        if not diff_lines:
            return Panel("No changes", title=f"[bold]📄 {filename}[/bold]")

        diff_text = Text()
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                diff_text.append(line, style="green")
            elif line.startswith("-") and not line.startswith("---"):
                diff_text.append(line, style="red")
            elif line.startswith("@"):
                diff_text.append(line, style="cyan")
            else:
                diff_text.append(line, style="dim")

        return Panel(
            diff_text,
            title=f"[bold]📄 Diff: {filename}[/bold]",
            border_style="yellow",
            padding=(0, 1),
        )


class SteeringMode:
    """
    Handles pause/resume with Ctrl+C
    """

    def __init__(self) -> None:
        self.paused = False
        self._callback: Callable | None = None

    def toggle(self) -> None:
        """Toggle pause state"""
        self.paused = not self.paused

    def set_callback(self, callback: Callable) -> None:
        """Set callback for steering mode"""
        self._callback = callback

    def render_prompt(self) -> Panel:
        """Render steering mode prompt"""
        return Panel(
            Text.from_markup(
                "[bold yellow]⏸️ Task Paused[/bold yellow]\n\n"
                "Options:\n"
                "  • Type an adjustment (e.g. 'Use FastAPI instead of Flask')\n"
                "  • Type 'resume' to continue\n"
                "  • Type 'abort' to stop"
            ),
            title="[bold]🎮 Steering Mode[/bold]",
            border_style="yellow",
        )


class LiveChatUI:
    """
    Complete live chat UI with all components
    """

    def __init__(self) -> None:
        self.brain = BrainActivityDisplay()
        self.response = StreamingResponse()
        self.ooda = OODAStatusDisplay()
        self.steering = SteeringMode()
        self._live: Live | None = None

    def _render(self) -> Group:
        """Render all components"""
        return Group(
            self.brain.render(),
            self.response.render(),
        )

    @contextmanager
    def live_display(self) -> Generator[None, None, None]:
        """Context manager for live display"""
        with Live(self._render(), console=console, refresh_per_second=10) as live:
            self._live = live
            try:
                yield
            finally:
                self._live = None

    def update_brain(self, stage: OODAStage, task: str, **kwargs: Any) -> None:
        """Update brain activity"""
        self.brain.update(stage, task, **kwargs)
        if self._live:
            self._live.update(self._render())

    def add_chunk(self, chunk: str) -> None:
        """Add response chunk"""
        self.response.add_chunk(chunk)
        if self._live:
            self._live.update(self._render())

    def complete_response(self) -> None:
        """Mark response complete"""
        self.response.complete()
        if self._live:
            self._live.update(self._render())


@dataclass
class TaskReceipt:
    """Summary card for a completed task"""

    task_id: str
    description: str
    status: str
    duration_seconds: float
    files_changed: list[str] = field(default_factory=list)
    quality_score: float = 0.0
    quality_breakdown: dict[str, float] = field(default_factory=dict)
    tokens_used: int = 0
    cost: float = 0.0
    layer_times: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def render(self) -> Panel:
        """Render task receipt as a panel"""
        lines = []
        lines.append(f"[bold]Task:[/bold] {self.description}")
        lines.append(f"[bold]Status:[/bold] {self._status_display()}")
        lines.append(f"[bold]Duration:[/bold] {self.duration_seconds:.1f}s")
        lines.append("")

        if self.files_changed:
            lines.append("[bold]Files Changed:[/bold]")
            for f in self.files_changed[:10]:
                lines.append(f"  • {f}")
            if len(self.files_changed) > 10:
                lines.append(f"  ... and {len(self.files_changed) - 10} more")

        if self.quality_breakdown:
            lines.append("")
            lines.append("[bold]Quality Score:[/bold] " + self._quality_bar())
            for name, score in self.quality_breakdown.items():
                lines.append(f"  • {name}: {self._score_bar(score)}")

        if self.layer_times:
            lines.append("")
            lines.append("[bold]Time by Layer:[/bold]")
            for layer, time_s in self.layer_times.items():
                lines.append(f"  • {layer}: {time_s:.2f}s")

        if self.warnings:
            lines.append("")
            lines.append("[yellow]Warnings:[/yellow]")
            for w in self.warnings[:3]:
                lines.append(f"  ⚠ {w}")

        if self.errors:
            lines.append("")
            lines.append("[red]Errors:[/red]")
            for e in self.errors[:3]:
                lines.append(f"  ✗ {e}")

        lines.append("")
        lines.append(f"[dim]Tokens: {self.tokens_used:,} | Cost: ${self.cost:.4f}[/dim]")

        return Panel(
            Text.from_markup("\n".join(lines)),
            title=f"[bold]📋 Task Receipt: {self.task_id[:8]}[/bold]",
            border_style="green" if self.status == "success" else "yellow",
            padding=(0, 1),
        )

    def _status_display(self) -> str:
        if self.status == "success":
            return "[green]✅ Success[/green]"
        elif self.status == "partial":
            return "[yellow]⚠️ Partial[/yellow]"
        else:
            return "[red]❌ Failed[/red]"

    def _quality_bar(self) -> str:
        score = self.quality_score
        if score >= 0.8:
            return f"[green]{self._score_bar(score)}[/green]"
        elif score >= 0.6:
            return f"[yellow]{self._score_bar(score)}[/yellow]"
        else:
            return f"[red]{self._score_bar(score)}[/red]"

    def _score_bar(self, score: float, width: int = 10) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled) + f" {score * 100:.0f}%"


def print_welcome() -> None:
    """Print welcome banner"""
    console.print(
        Panel(
            Text.from_markup(
                "[bold cyan]🧠 GAAP v2.0[/bold cyan]\n"
                "[dim]General-purpose AI Architecture Platform[/dim]\n"
                "[dim]Type 'help' for commands, 'exit' to quit[/dim]"
            ),
            border_style="cyan",
        )
    )


def print_error(message: str) -> None:
    """Print error message"""
    console.print(
        Panel(
            Text(message, style="red"),
            title="[bold red]❌ Error[/bold red]",
            border_style="red",
        )
    )


def print_success(message: str) -> None:
    """Print success message"""
    console.print(
        Panel(
            Text(message, style="green"),
            title="[bold green]✅ Success[/bold green]",
            border_style="green",
        )
    )


def print_stats(stats: dict[str, Any]) -> None:
    """Print enhanced session stats with breakdowns"""
    table = Table(title="📊 Session Stats", show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Requests", f"{stats.get('requests_processed', 0)}")
    table.add_row("Success Rate", f"{stats.get('success_rate', 0):.1%}")
    table.add_row("Tokens Used", f"{stats.get('total_tokens', 0):,}")
    table.add_row("Cost", f"${stats.get('total_cost', 0):.2f}")

    console.print(table)

    if stats.get("files_changed"):
        console.print()
        files_table = Table(title="📁 Files Changed", show_header=False)
        files_table.add_column("File", style="yellow")
        for f in stats["files_changed"][:15]:
            files_table.add_row(f)
        if len(stats["files_changed"]) > 15:
            files_table.add_row(f"... and {len(stats['files_changed']) - 15} more")
        console.print(files_table)

    if stats.get("quality_breakdown"):
        console.print()
        quality_table = Table(title="🎯 Quality Score Breakdown", show_header=True)
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Score", justify="right")
        quality_table.add_column("Bar", justify="left")

        for name, score in stats["quality_breakdown"].items():
            bar = _render_bar(score)
            score_color = "green" if score >= 0.8 else "yellow" if score >= 0.6 else "red"
            quality_table.add_row(name, f"[{score_color}]{score * 100:.0f}%[/{score_color}]", bar)

        overall = stats.get("quality_score", 0)
        overall_bar = _render_bar(overall)
        quality_table.add_row(
            "[bold]Overall[/bold]",
            f"[bold]{overall * 100:.0f}%[/bold]",
            f"[bold]{overall_bar}[/bold]",
        )
        console.print(quality_table)

    if stats.get("layer_times"):
        console.print()
        time_table = Table(title="⏱️ Time Breakdown by Layer", show_header=True)
        time_table.add_column("Layer", style="cyan")
        time_table.add_column("Time", justify="right")
        time_table.add_column("Percentage", justify="right")

        total_time = sum(stats["layer_times"].values())
        for layer, time_s in stats["layer_times"].items():
            pct = (time_s / total_time * 100) if total_time > 0 else 0
            layer_color = (
                "purple"
                if "L1" in layer or "strategy" in layer.lower()
                else "blue" if "L2" in layer or "tactic" in layer.lower() else "green"
            )
            time_table.add_row(
                f"[{layer_color}]{layer}[/{layer_color}]",
                f"{time_s:.2f}s",
                f"{pct:.1f}%",
            )
        time_table.add_row(
            "[bold]Total[/bold]", f"[bold]{total_time:.2f}s[/bold]", "[bold]100%[/bold]"
        )
        console.print(time_table)


def _render_bar(score: float, width: int = 20) -> str:
    """Render a progress bar for a score"""
    filled = int(score * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def print_task_receipt(receipt: TaskReceipt) -> None:
    """Print a task receipt summary card"""
    console.print(receipt.render())


def print_help() -> None:
    """Print help message"""
    table = Table(title="📖 Commands", show_header=False)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="dim")

    commands = [
        ("exit, quit, q", "Exit interactive mode"),
        ("clear", "Clear conversation history"),
        ("help", "Show this help"),
        ("stats", "Show session stats"),
        ("budget", "Show remaining budget"),
        ("models", "List available models"),
        ("Ctrl+C", "Enter steering mode (pause)"),
    ]

    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console.print(table)
