"""
GAAP Sovereign Mission Launcher
Building the Autonomous JS Research Lab
"""
import asyncio
import logging
from gaap.gaap_engine import create_engine

async def main():
    print("
" + "="*60)
    print("🚀 GAAP SOVEREIGN MISSION CONTROL")
    print("Mission: Building the Global JS Research Lab (Indeed.com Focus)")
    print("="*60 + "
")

    # 1. Start Engine
    engine = create_engine()
    
    # 2. Configure Target Context
    target_data = {
        "domain": "Indeed.com",
        "focus": "Client-side logic, API leaks, IDOR patterns in JS bundles",
        "scope": ["https://indeed.com", "https://employer.indeed.com", "https://apply.indeed.com"]
    }

    # 3. Launch Mission (Autonomous Role: JS_BOUNTY_LAB_LEADER)
    # This will follow the SOP defined in roles/js_bounty_researcher.yaml
    try:
        await engine.execute_mission(
            role_id="JS_BOUNTY_LAB_LEADER", 
            target_context=target_data
        )
    except KeyboardInterrupt:
        print("
🛑 Mission paused. All state and memory preserved.")
    except Exception as e:
        print(f"
❌ Mission aborted due to critical error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
