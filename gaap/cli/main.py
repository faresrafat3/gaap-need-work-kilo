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

)


def load_env() -> None:
    """Load env vars from .gaap_env file"""
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

    # --- New Sovereign Commands (v2.1) ---
    
    # Research command
    research_p = subparsers.add_parser("research", help="Run Deep Research (STORM Augmented)")
    research_p.add_argument("query", help="Research query or topic")
    research_p.add_argument("--depth", type=int, default=3, help="Research depth (steps)")

    # Debug command
    debug_p = subparsers.add_parser("debug", help="Diagnostic & Root Cause Analysis")
    debug_p.add_argument("issue", help="Description of the error or issue")

    # Maintenance commands
    subparsers.add_parser("dream", help="Trigger Memory Consolidation (REM Sleep)")
    subparsers.add_parser("audit", help="Run Constitutional Integrity Audit")


    return parser


def main() -> None:
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
    elif args.command == "research":
        # Dynamic execution using the sovereign engine
        from gaap.cli.commands import cmd_research
        asyncio.run(cmd_research(args))
    elif args.command == "debug":
        from gaap.cli.commands import cmd_debug
        asyncio.run(cmd_debug(args))
    elif args.command == "dream":
        from gaap.cli.commands import cmd_dream
        cmd_dream(args)
    elif args.command == "audit":
        from gaap.cli.commands import cmd_audit
        cmd_audit(args)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
