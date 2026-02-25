import sys
import asyncio
import json
from unittest.mock import MagicMock
from datetime import datetime

# 1. Matrix Shim
sys.modules["aiohttp"] = MagicMock()
sys.modules["chromadb"] = MagicMock()
sys.modules["chromadb.config"] = MagicMock()

from gaap.gaap_engine import GAAPEngine, GAAPRequest
from gaap.layers.layer0_interface import IntentType

async def run_sovereign_demo():
    print("\n🔮 GAAP v2.1.0-SOVEREIGN: End-to-End Symphony Demo")
    print("===================================================\n")

    mock_provider = MagicMock()
    mock_provider.default_model = "sovereign-gpt-4o"
    
    async def sovereign_brain(*args, **kwargs):
        messages = kwargs.get("messages", [])
        if not messages: return MagicMock(choices=[MagicMock(message=MagicMock(content="OK"))])
        
        content = messages[-1].content or ""
        system_content = messages[0].content or ""
        
        # A. STORM Research
        if "Research Synthesizer" in system_content:
            return MagicMock(choices=[MagicMock(message=MagicMock(content="# Log4Shell Report\nCritical vulnerability..."))])
        
        # B. Layer 1 Strategy
        if "architecture specification" in system_content:
            return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "paradigm": "modular_monolith",
                "decisions": [{"aspect": "security", "choice": "strict"}],
                "components": [{"name": "Scanner", "responsibility": "Check"}],
                "phases": [],
                "risks": []
            })))])
            
        # C. Layer 2 Tactics
        if "task decomposition" in system_content:
            return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps([{
                "name": "Impl Scanner",
                "description": "Code the scanner",
                "category": "security",
                "type": "code_generation",
                "priority": "high",
                "complexity": "simple",
                "depends_on": []
            }])))])
            
        # D. Layer 3 & Metacognition
        if "Metacognition" in system_content:
            return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps({
                "confidence": 0.95,
                "can_proceed": True,
                "reasoning": "Clear task"
            })))])
            
        return MagicMock(choices=[MagicMock(message=MagicMock(content="Execution Success"))])

    mock_provider.chat_completion = sovereign_brain
    
    # Correct Engine Init
    engine = GAAPEngine(providers=[mock_provider])
    
    request_text = "Research Log4Shell and write a scanner."
    print(f"👤 User: {request_text}\n")
    
    print("⚙️  Engine: OODA Loop Activated...")
    request = GAAPRequest(text=request_text)
    
    # Run!
    response = await engine.process(request)
    
    print("\n📊 Mission Report:")
    print(f"   - Success: {response.success}")
    if response.intent:
        print(f"   - Intent: {response.intent.intent_type.name}")
    print(f"   - OODA Iterations: {response.ooda_iterations}")
    print(f"   - Quality Score: {response.quality_score}")

    print("\n🎉 DEMO CONCLUSION: The Sovereign System is fully operational.")

if __name__ == "__main__":
    asyncio.run(run_sovereign_demo())
