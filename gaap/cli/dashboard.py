"""
GAAP TUI Dashboard
==================

Full-screen Terminal User Interface for monitoring GAAP.

Features:
- OODA Loop status
- Real-time logs
- Budget monitor
- Task graph progress

Reference: docs/evolution_plan_2026/44_CLI_AUDIT_SPEC.md
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class DashboardState:
    """Dashboard state"""

    ooda_stage: str = "OBSERVE"
    current_task: str = "Idle"
    budget_used: float = 0.0
    budget_limit: float = 100.0
    tokens_used: int = 0
    tasks_completed: int = 0
    tasks_total: int = 0
    success_rate: float = 1.0
    logs: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    @property
    def budget_remaining(self) -> float:
        return self.budget_limit - self.budget_used

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


class GAAPDashboard:
    """
    Full-screen TUI dashboard for GAAP
    """

    def __init__(self, budget: float = 100.0):
        self.state = DashboardState(budget_limit=budget)
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        self._task_id: TaskID | None = None
        self._running = False

    def _render_ooda_panel(self) -> Panel:
        """Render OODA loop status"""
        stages = ["OBSERVE", "ORIENT", "DECIDE", "ACT", "LEARN"]
        stage_icons = {
            "OBSERVE": ("👁️", "cyan"),
            "ORIENT": ("🧭", "yellow"),
            "DECIDE": ("🤔", "magenta"),
            "ACT": ("⚡", "green"),
            "LEARN": ("📚", "blue"),
        }

        lines = []
        for stage in stages:
            icon, color = stage_icons.get(stage, ("❓", "white"))
            if stage == self.state.ooda_stage:
                lines.append(f"[bold {color}]→ {icon} {stage}[/bold {color}]")
            else:
                lines.append(f"[dim]  {icon} {stage}[/dim]")

        content = "\n".join(lines)
        return Panel(content, title="[bold]OODA Loop[/bold]", border_style="cyan")

    def _render_budget_panel(self) -> Panel:
        """Render budget monitor"""
        percent = (self.state.budget_used / self.state.budget_limit) * 100
        bar_filled = int(percent / 5)
        bar_empty = 20 - bar_filled

        if percent < 50:
            color = "green"
        elif percent < 80:
            color = "yellow"
        else:
            color = "red"

        bar = f"[{color}]{'█' * bar_filled}{'░' * bar_empty}[/{color}]"

        content = Text.from_markup(
            f"Budget: ${self.state.budget_used:.2f} / ${self.state.budget_limit:.2f}\n"
            f"{bar} {percent:.0f}%\n\n"
            f"Remaining: [bold green]${self.state.budget_remaining:.2f}[/bold green]\n"
            f"Tokens: {self.state.tokens_used:,}"
        )

        return Panel(content, title="[bold]💰 Budget[/bold]", border_style="green")

    def _render_tasks_panel(self) -> Panel:
        """Render task progress"""
        if self.state.tasks_total == 0:
            content = Text("No tasks running", style="dim")
        else:
            percent = (self.state.tasks_completed / self.state.tasks_total) * 100
            content = Text.from_markup(
                f"Current: [cyan]{self.state.current_task}[/cyan]\n\n"
                f"Progress: {self.state.tasks_completed}/{self.state.tasks_total}\n"
                f"Success Rate: [bold green]{self.state.success_rate:.0%}[/bold green]"
            )

        return Panel(content, title="[bold]📋 Tasks[/bold]", border_style="magenta")

    def _render_logs_panel(self) -> Panel:
        """Render recent logs"""
        if not self.state.logs:
            content = Text("No logs yet", style="dim")
        else:
            content = Text()
            for log in self.state.logs[-10:]:
                content.append(log + "\n")

        return Panel(content, title="[bold]📜 Logs[/bold]", border_style="blue")

    def _render_stats_panel(self) -> Panel:
        """Render session stats"""
        elapsed_min = int(self.state.elapsed // 60)
        elapsed_sec = int(self.state.elapsed % 60)

        table = Table(show_header=False, padding=0)
        table.add_column("Stat", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Session", f"{elapsed_min:02d}:{elapsed_sec:02d}")
        table.add_row("Requests", f"{self.state.tasks_completed}")
        table.add_row("Success", f"{self.state.success_rate:.0%}")
        table.add_row(
            "Avg Cost", f"${self.state.budget_used / max(1, self.state.tasks_completed):.3f}"
        )

        return Panel(table, title="[bold]📊 Stats[/bold]", border_style="yellow")

    def render(self) -> Layout:
        """Render full dashboard"""
        layout = Layout()

        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=3),
        )

        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="center", ratio=2),
            Layout(name="right", ratio=1),
        )

        layout["left"].split(
            Layout(name="ooda"),
            Layout(name="budget"),
        )

        layout["center"].split(
            Layout(name="tasks"),
            Layout(name="logs"),
        )

        layout["right"].split(
            Layout(name="stats"),
        )

        elapsed_min = int(self.state.elapsed // 60)
        elapsed_sec = int(self.state.elapsed % 60)

        layout["header"].update(
            Panel(
                Text.from_markup(
                    "[bold cyan]🧠 GAAP Dashboard[/bold cyan] "
                    f"[dim]Session: {elapsed_min:02d}:{elapsed_sec:02d}[/dim]"
                ),
                border_style="cyan",
            )
        )

        layout["ooda"].update(self._render_ooda_panel())
        layout["budget"].update(self._render_budget_panel())
        layout["tasks"].update(self._render_tasks_panel())
        layout["logs"].update(self._render_logs_panel())
        layout["stats"].update(self._render_stats_panel())

        layout["footer"].update(
            Panel(
                Text.from_markup("[dim]Press 'q' to quit | 'r' to refresh | 'h' for help[/dim]"),
                border_style="dim",
            )
        )

        return layout

    def update(self, **kwargs: Any) -> None:
        """Update dashboard state"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def add_log(self, message: str) -> None:
        """Add a log message"""
        timestamp = time.strftime("%H:%M:%S")
        self.state.logs.append(f"[{timestamp}] {message}")

    def run(self) -> None:
        """Run dashboard (non-async)"""
        self._running = True

        def get_key() -> str:
            import sys
            import tty
            import termios

            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        with Live(self.render(), console=console, refresh_per_second=4) as live:
            while self._running:
                try:
                    import select
                    import sys as sys_mod

                    if select.select([sys_mod.stdin], [], [], 0.1)[0]:
                        key = get_key().lower()
                        if key == "q":
                            self._running = False
                        elif key == "r":
                            live.update(self.render())
                        elif key == "h":
                            console.print("\n[q] Quit  [r] Refresh  [h] Help")
                    live.update(self.render())
                except Exception:
                    live.update(self.render())

    async def run_async(self) -> None:
        """Run dashboard (async)"""
        self._running = True

        with Live(self.render(), console=console, refresh_per_second=4) as live:
            while self._running:
                live.update(self.render())
                await asyncio.sleep(0.25)


async def cmd_dashboard(args: Any) -> None:
    """Launch TUI dashboard"""
    dashboard = GAAPDashboard(budget=getattr(args, "budget", 100.0))

    dashboard.add_log("Dashboard started")
    dashboard.update(
        ooda_stage="OBSERVE",
        current_task="Monitoring...",
    )

    await dashboard.run_async()
