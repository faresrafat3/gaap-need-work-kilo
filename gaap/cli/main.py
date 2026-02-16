# mypy: ignore-errors
# CLI Main
import argparse
import asyncio
import os
import sys

from gaap.cli.commands import (
    cmd_chat,
    cmd_config,
    cmd_doctor,
    cmd_history,
    cmd_interactive,
    cmd_models,
    cmd_providers,
    cmd_status,
    cmd_version,
    cmd_web,
)


def load_env() -> None:
    """Load env vars from .gaap_env file"""
    from pathlib import Path

    def parse_lines(lines):
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


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        prog="gaap",
        description="GAAP - General-purpose AI Architecture Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Chat command
    chat_p = subparsers.add_parser("chat", help="Quick chat with GAAP")
    chat_p.add_argument("message", help="Message to send")
    chat_p.add_argument("--budget", type=float, default=1.0, help="Budget limit")

    # Interactive command
    interactive_p = subparsers.add_parser(
        "interactive", aliases=["i"], help="Interactive chat mode"
    )
    interactive_p.add_argument("--budget", type=float, default=10.0, help="Budget limit")

    # Providers command
    providers_p = subparsers.add_parser("providers", help="Manage providers")
    providers_p.add_argument(
        "action", nargs="?", default="list", choices=["list", "status", "test", "enable", "disable"]
    )
    providers_p.add_argument("provider", nargs="?", help="Provider name")

    # Models command
    models_p = subparsers.add_parser("models", help="Manage models")
    models_p.add_argument("action", nargs="?", default="list", choices=["list", "tiers", "info"])
    models_p.add_argument("model", nargs="?", help="Model name")
    models_p.add_argument("--tier", help="Filter by tier")

    # Config command
    config_p = subparsers.add_parser("config", help="Manage configuration")
    config_p.add_argument(
        "action",
        nargs="?",
        default="show",
        choices=["show", "get", "set", "reset", "export", "import"],
    )
    config_p.add_argument("key", nargs="?", help="Config key")
    config_p.add_argument("value", nargs="?", help="Config value")
    config_p.add_argument("--file", help="File path for import")

    # History command
    history_p = subparsers.add_parser("history", help="Manage conversation history")
    history_p.add_argument(
        "action", nargs="?", default="list", choices=["list", "show", "clear", "search", "export"]
    )
    history_p.add_argument("id", nargs="?", help="Item ID or query")
    history_p.add_argument("--limit", type=int, default=20, help="Limit results")
    history_p.add_argument("--file", help="Export file path")

    # Status command
    subparsers.add_parser("status", help="Show system status")

    # Version command
    subparsers.add_parser("version", help="Show version")

    # Doctor command
    subparsers.add_parser("doctor", help="Run diagnostics")

    # Web command
    web_p = subparsers.add_parser("web", help="Launch web UI")
    web_p.add_argument("--port", type=int, default=8501, help="Port number")

    return parser


def main():
    """Main entry point"""
    load_env()
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command in ["interactive", "i"]:
        asyncio.run(cmd_interactive(args))
    elif args.command == "providers":
        cmd_providers(args)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "history":
        cmd_history(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "version":
        cmd_version(args)
    elif args.command == "doctor":
        cmd_doctor(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
