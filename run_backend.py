#!/usr/bin/env python3
"""Quick backend starter - Portable version"""
import os
import sys
from pathlib import Path

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Set environment with defaults
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./gaap.db")
os.environ.setdefault("REDIS_URL", "")
os.environ.setdefault("PYTHON_API_URL", "http://localhost:8000")
os.environ.setdefault("GAAP_ENVIRONMENT", "development")

# Add project to path
sys.path.insert(0, str(SCRIPT_DIR))

# Try to detect and use virtual environment
venv_site_packages = SCRIPT_DIR / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
if venv_site_packages.exists():
    sys.path.insert(0, str(venv_site_packages))

# Disable prometheus to avoid conflicts
try:
    import gaap.observability.metrics as m
    m.PROMETHEUS_AVAILABLE = False
except ImportError:
    pass

# Import and run
try:
    from gaap.api.main import app
    import uvicorn
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to activate your virtual environment:")
    print("   source venv/bin/activate")
    sys.exit(1)

if __name__ == "__main__":
    print("🚀 Starting GAAP Backend...")
    print("📊 Database: SQLite")
    print("🌐 URL: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
