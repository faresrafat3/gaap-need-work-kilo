"""
Chat Commands
"""

import os


def load_env() -> None:
    """Load environment variables from .gaap_env"""
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


async def cmd_chat(args):
    """Quick chat command"""
    load_env()

    from gaap.gaap_engine import create_engine
    from gaap.storage import save_history

    engine = create_engine(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        budget=getattr(args, "budget", 1.0),
        enable_all=False,
    )

    try:
        message = args.message if hasattr(args, "message") else args[0] if args else ""

        print("\n🤖 GAAP v1.0.0")
        print("-" * 40)

        save_history("user", message)

        response = await engine.chat(message)

        save_history("assistant", str(response), provider="gaap", model="default")

        print(f"\n{response}\n")

    finally:
        for provider in getattr(engine, "providers", []):
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                try:
                    await close_fn()
                except Exception:
                    pass
        engine.shutdown()


async def cmd_interactive(args):
    """Interactive chat REPL"""
    load_env()

    from gaap.gaap_engine import create_engine
    from gaap.storage import save_history

    engine = create_engine(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        budget=getattr(args, "budget", 10.0),
        enable_all=True,
    )

    print("\n🤖 GAAP Interactive Mode")
    print("Type 'exit' or 'quit' to exit")
    print("Type 'clear' to clear history")
    print("Type 'help' for commands")
    print("-" * 40)

    try:
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break

            if user_input.lower() == "clear":
                print("\n🗑️ History cleared")
                continue

            if user_input.lower() == "help":
                print("\nCommands:")
                print("  exit, quit, q  - Exit interactive mode")
                print("  clear          - Clear conversation history")
                print("  help           - Show this help")
                print("  stats          - Show session stats")
                continue

            if user_input.lower() == "stats":
                stats = engine.get_stats()
                print("\n📊 Stats:")
                print(f"  Requests: {stats.get('requests_processed', 0)}")
                print(f"  Success Rate: {stats.get('success_rate', 0):.1%}")
                continue

            save_history("user", user_input)

            try:
                response = await engine.chat(user_input)
                save_history("assistant", str(response), provider="gaap", model="default")
                print(f"\n🤖 GAAP: {response}")
            except Exception as e:
                print(f"\n❌ Error: {e}")

    finally:
        for provider in getattr(engine, "providers", []):
            close_fn = getattr(provider, "close", None)
            if callable(close_fn):
                try:
                    await close_fn()
                except Exception:
                    pass
        engine.shutdown()
