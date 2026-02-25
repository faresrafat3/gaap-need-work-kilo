"""
Chat Commands with Rich TUI
===========================

Enhanced chat interface with live streaming and brain activity display.
Integrates NativeStreamer for real-time token streaming.
"""

import asyncio
import contextlib
import os
import signal
from typing import Any

from rich.console import Console

from gaap.cli.tui import (
    LiveChatUI,
    OODAStage,
    print_error,
    print_help,
    print_stats,
    print_welcome,
)
from gaap.providers.streaming import TokenChunk

console = Console()


def load_env() -> None:
    """Load environment variables from .gaap_env"""
    from pathlib import Path

    def parse_lines(lines: list[str]) -> None:
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    paths = [
        Path.home() / ".gaap_env",
        Path.cwd() / ".gaap_env",
    ]
    for path in paths:
        if path.exists() and path.is_file():
            try:
                parse_lines(path.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue


async def _close_providers(engine: Any) -> None:
    """Safely close all providers."""
    for provider in getattr(engine, "providers", []):
        close_fn = getattr(provider, "close", None)
        if callable(close_fn):
            with contextlib.suppress(Exception):
                result = close_fn()
                if asyncio.iscoroutine(result):
                    await result


async def cmd_chat(args: Any) -> None:
    """Quick chat command with rich UI"""
    load_env()

    from gaap.gaap_engine import create_engine

    engine = create_engine(
        budget=getattr(args, "budget", 1.0),
        enable_all=False,
    )

    ui = LiveChatUI()
    response = ""

    try:
        message = args.message if hasattr(args, "message") else args[0] if args else ""

        print_welcome()

        with ui.live_display():
            ui.update_brain(OODAStage.OBSERVE, "Reading your message...")

            ui.update_brain(OODAStage.ORIENT, "Understanding context...")

            ui.update_brain(OODAStage.DECIDE, "Planning response...")

            ui.update_brain(OODAStage.ACT, "Generating response...")

            try:
                response = await engine.chat(message)

                for word in str(response).split():
                    ui.add_chunk(word + " ")
                    await asyncio.sleep(0.02)

                ui.complete_response()

            except Exception as e:
                ui.add_chunk(f"Error: {e}")
                ui.complete_response()

    finally:
        await _close_providers(engine)
        engine.shutdown()


async def cmd_interactive(args: Any) -> None:
    """Interactive chat REPL with rich UI and steering mode"""
    load_env()

    from gaap.gaap_engine import create_engine

    engine = create_engine(
        budget=getattr(args, "budget", 10.0),
        enable_all=True,
    )

    print_welcome()
    console.print("[dim]Type 'help' for commands, 'exit' to quit, Ctrl+C to pause[/dim]")
    console.print()

    paused = False
    adjustment: str | None = None

    def handle_sigint(signum: int, frame: Any) -> None:
        nonlocal paused
        paused = True

    original_handler = signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            try:
                if paused:
                    console.print()
                    console.print("[bold yellow]⏸️ Paused[/bold yellow]")
                    console.print(
                        "[dim]Type 'resume' to continue, 'abort' to stop, or enter adjustment[/dim]"
                    )
                    adj = console.input("[cyan]Adjustment> [/cyan]").strip()

                    if adj.lower() == "abort":
                        console.print("[red]Aborted[/red]")
                        break
                    elif adj.lower() == "resume":
                        paused = False
                        console.print("[green]Resumed[/green]")
                        continue
                    else:
                        adjustment = adj
                        paused = False
                        console.print(f"[green]Adjustment: {adj}[/green]")
                        continue

                user_input = console.input("[bold cyan]👤 You:[/] ").strip()

            except (EOFError, KeyboardInterrupt):
                if not paused:
                    paused = True
                    continue
                console.print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("\n👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                console.print("\n🗑️ History cleared")
                continue

            if user_input.lower() == "help":
                print_help()
                continue

            if user_input.lower() == "stats":
                stats = engine.get_stats()
                print_stats(stats)
                continue

            if user_input.lower() == "budget":
                stats = engine.get_stats()
                remaining = getattr(args, "budget", 10.0) - stats.get("total_cost", 0)
                console.print(f"\n💰 Budget remaining: ${remaining:.2f}")
                continue

            if adjustment:
                user_input = f"{user_input}\n\nNote: User requested adjustment: {adjustment}"
                adjustment = None

            ui = LiveChatUI()

            try:
                with ui.live_display():
                    ui.update_brain(OODAStage.OBSERVE, "Analyzing your request...")

                    ui.update_brain(OODAStage.ORIENT, "Searching memory...")

                    ui.update_brain(OODAStage.DECIDE, "Planning approach...")

                    ui.update_brain(OODAStage.ACT, "Generating response...")

                    response = await engine.chat(user_input)

                    for word in str(response).split():
                        if paused:
                            break
                        ui.add_chunk(word + " ")
                        await asyncio.sleep(0.02)

                    ui.complete_response()

            except Exception as e:
                print_error(str(e))

    finally:
        signal.signal(signal.SIGINT, original_handler)
        await _close_providers(engine)
        engine.shutdown()
