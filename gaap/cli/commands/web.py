"""
Web Command - Start both frontend and backend servers
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

web_cli = typer.Typer(help="Start web interface")


def get_venv_python() -> str:
    root_dir = Path(__file__).parent.parent.parent.parent
    venv_python = root_dir / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@web_cli.command()
def start(
    backend_port: int = typer.Option(8000, "--backend-port", "-b", help="Backend port"),
    frontend_port: int = typer.Option(3000, "--frontend-port", "-f", help="Frontend port"),
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't open browser"),
):
    """Start both frontend and backend servers"""

    project_root = Path(__file__).parent.parent.parent.parent

    backend_process: Optional[subprocess.Popen] = None
    frontend_process: Optional[subprocess.Popen] = None

    def cleanup():
        """Stop all processes"""
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()

    def signal_handler(sig, frame):
        cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        typer.echo(f"🚀 Starting backend on port {backend_port}...")
        backend_env = os.environ.copy()
        backend_env["PORT"] = str(backend_port)

        backend_process = subprocess.Popen(
            [
                "uvicorn",
                "gaap.api.main:app",
                "--reload",
                "--port",
                str(backend_port),
                "--host",
                host,
            ],
            cwd=str(project_root),
            env=backend_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        typer.echo(f"🚀 Starting frontend on port {frontend_port}...")
        frontend_env = os.environ.copy()
        frontend_env["PORT"] = str(frontend_port)

        frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(project_root / "frontend"),
            env=frontend_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        typer.echo("\n" + "=" * 50)
        typer.echo("🌐 GAAP Web Interface")
        typer.echo("=" * 50)
        typer.echo(f"  Backend:  http://{host}:{backend_port}")
        typer.echo(f"  Frontend: http://{host}:{frontend_port}")
        typer.echo(f"  API Docs: http://{host}:{backend_port}/docs")
        typer.echo("=" * 50)
        typer.echo("\n✅ Press Ctrl+C to stop all servers\n")

        while True:
            if backend_process.poll() is not None:
                typer.echo("❌ Backend stopped unexpectedly!")
                break
            if frontend_process.poll() is not None:
                typer.echo("❌ Frontend stopped unexpectedly!")
                break
            time.sleep(1)

    except Exception as e:
        typer.echo(f"❌ Error: {e}")
    finally:
        cleanup()


def cmd_web(args) -> None:
    """Entry point for argparse-based CLI (gaap web)"""
    backend_port = getattr(args, "backend_port", 8000)
    frontend_port = getattr(args, "frontend_port", 3000)
    host = getattr(args, "host", "localhost")

    root_dir = Path(__file__).parent.parent.parent.parent
    frontend_dir = root_dir / "frontend"

    python_exe = get_venv_python()

    print(f"Starting GAAP servers...")
    print(f"  Backend:  http://{host}:{backend_port}")
    print(f"  Frontend: http://{host}:{frontend_port}")
    print(f"\nPress Ctrl+C to stop both servers\n")

    backend_process = None
    frontend_process = None

    def cleanup(signum=None, frame=None):
        print("\n\nShutting down servers...")
        if backend_process and backend_process.poll() is None:
            backend_process.terminate()
            try:
                backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
        if frontend_process and frontend_process.poll() is None:
            frontend_process.terminate()
            try:
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                frontend_process.kill()
        print("Servers stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    env = {**os.environ.copy(), "PYTHONUNBUFFERED": "1"}

    try:
        backend_process = subprocess.Popen(
            [
                python_exe,
                "-m",
                "uvicorn",
                "gaap.api.main:app",
                "--host",
                host,
                "--port",
                str(backend_port),
            ],
            cwd=root_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

        time.sleep(2)

        frontend_process = subprocess.Popen(
            ["npm", "run", "dev", "--", "-p", str(frontend_port)],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**env, "PORT": str(frontend_port)},
        )

        print("Servers started successfully!")
        print(f"  Backend:  http://{host}:{backend_port}")
        print(f"  Frontend: http://{host}:{frontend_port}")
        print(f"\nWaiting for servers... (Ctrl+C to stop)")

        while True:
            time.sleep(1)
            if backend_process.poll() is not None:
                print("Backend process died unexpectedly!")
                cleanup()
            if frontend_process.poll() is not None:
                print("Frontend process died unexpectedly!")
                cleanup()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure uvicorn and npm are installed.")
        cleanup()
    except Exception as e:
        print(f"Error starting servers: {e}")
        cleanup()
