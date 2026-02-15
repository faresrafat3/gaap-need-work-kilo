import asyncio
import logging
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.getcwd())

def load_env():
    """Simple env loader for GAAP."""
    paths = [
        Path.home() / ".gaap_env",
        Path.cwd() / ".gaap_env",
    ]
    for path in paths:
        if path.exists() and path.is_file():
            print(f"📁 Loading env from {path}")
            for line in path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                key, value = raw.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

load_env()

from gaap import GAAPEngine, GAAPRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test():
    print("\n🚀 Initializing GAAP Engine...")
    engine = GAAPEngine()
    
    prompt = "Write a short hello world in Python"
    print(f"📡 Processing Request: '{prompt}'...")
    request = GAAPRequest(text=prompt)
    
    try:
        response = await engine.process(request)
        print("\n" + "="*50)
        print(f"✅ Success: {response.success}")
        if response.intent:
            print(f"📝 Intent: {response.intent.intent_type}")
        print(f"⏱️  Time: {response.total_time_ms:.0f}ms")
        print(f"💰 Cost: ${response.total_cost_usd:.4f}")
        print("-" * 50)
        print(f"📄 Output:\n{response.output}")
        print("="*50)
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        engine.shutdown()

if __name__ == "__main__":
    asyncio.run(test())
