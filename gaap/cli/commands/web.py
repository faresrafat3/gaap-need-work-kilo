"""
Web Command - Launch Streamlit UI
"""

import subprocess
import sys
from pathlib import Path
from typing import Any


def cmd_web(args: Any) -> None:
    """Launch GAAP Web UI"""
    port = getattr(args, "port", 8501) if args else 8501

    web_dir = Path(__file__).parent.parent / "web"
    app_file = web_dir / "app.py"

    if not app_file.exists():
        print(f"Error: Web app not found at {app_file}")
        return

    print(f"\nStarting GAAP Web UI on http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_file), "--server.port", str(port)],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nWeb UI stopped")
    except FileNotFoundError:
        print("Error: Streamlit not installed. Run: pip install streamlit")
    except Exception as e:
        print(f"Error: {e}")
