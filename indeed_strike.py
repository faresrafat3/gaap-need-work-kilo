import sys
import asyncio
from gaap.tools.sovereign_scanner import run

async def main():
    prefixes = ["dev", "staging", "qa", "internal", "admin", "legacy", "v1", "old", "uat", "test", "api", "api-dev", "sandbox", "portal", "corp"]
    targets = [f"https://{p}.indeed.com" for p in prefixes]
    
    print("\n🎯 Sovereign Strike v2: Massive Subdomain Scan (Indeed.com)")
    print("==========================================================\n")
    
    for target in targets:
        print(f"🔍 Scanning {target}...")
        try:
            result = run(target_url=target)
            if "FOUND" in result or "ACCESSIBLE" in result:
                print("\n" + "!" * 50)
                print(f"🔥 VULNERABILITY FOUND AT: {target}")
                print(result)
                print("!" * 50 + "\n")
            else:
                print(f"   [-] No easy find: {target}")
        except:
            print(f"   [x] Target unreachable: {target}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())
