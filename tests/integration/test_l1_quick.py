"""
Quick L1 Strategic Test — LLM-powered architecture planning
Tests that L1 generates request-specific architecture instead of static defaults
"""
import asyncio
import sys
import os
import time
import json


async def main():
    print("=" * 60)
    print("L1 LLM Strategic — Quick Test")
    print("=" * 60)
    
    # 1. Import
    from providers.webchat_bridge import create_kimi_provider
    from layers.layer0_interface import Layer0Interface
    from layers.layer1_strategic import Layer1Strategic
    
    # 2. Create provider
    print("\n[1] Creating provider...")
    provider = create_kimi_provider(timeout=120)
    
    # 3. Create L0 + L1
    layer0 = Layer0Interface(
        firewall_strictness="high",
        enable_behavioral_analysis=True
    )
    
    layer1 = Layer1Strategic(
        tot_depth=5,
        tot_branching=4,
        mad_rounds=3,
        provider=provider
    )
    
    # 4. Test with a specific request
    test_request = "Build a REST API for a todo list app with user authentication and PostgreSQL"
    
    print(f"\n[2] Test request: {test_request}")
    print("-" * 60)
    
    # L0: Parse intent
    print("\n[3] L0: Parsing intent...")
    t0 = time.time()
    intent = await layer0.process(test_request)
    print(f"    Intent type: {intent.intent_type.name}")
    print(f"    Routing: {intent.routing_target.value}")
    print(f"    Goals: {intent.explicit_goals}")
    print(f"    Original text preserved: {'original_text' in intent.metadata}")
    print(f"    Time: {(time.time()-t0)*1000:.0f}ms")
    
    # L1: Strategic planning
    print("\n[4] L1: Strategic planning (LLM)...")
    t1 = time.time()
    spec = await layer1.process(intent)
    l1_time = time.time() - t1
    
    print(f"\n--- Architecture Spec ---")
    print(f"    Source: {spec.metadata.get('strategy_source', 'unknown')}")
    print(f"    Paradigm: {spec.paradigm.value}")
    print(f"    Data Strategy: {spec.data_strategy.value}")
    print(f"    Communication: {spec.communication.value}")
    print(f"    Tech Stack: {json.dumps(spec.tech_stack, indent=6)}")
    print(f"    Components: {len(spec.components)}")
    for c in spec.components[:5]:
        if isinstance(c, dict):
            print(f"      - {c.get('name', '?')}: {c.get('responsibility', c.get('type', ''))}")
    print(f"    Decisions: {len(spec.decisions)}")
    for d in spec.decisions[:3]:
        print(f"      - {d.aspect}: {d.choice}")
        print(f"        Why: {d.reasoning[:80]}")
    print(f"    Risks: {len(spec.risks)}")
    for r in spec.risks[:3]:
        if isinstance(r, dict):
            print(f"      - [{r.get('severity','?')}] {r.get('issue', r.get('source',''))}")
    
    # Plan/phases
    plan = spec.metadata.get('plan', {})
    phases = plan.get('phases', [])
    print(f"    Phases: {len(phases)}")
    for p in phases[:4]:
        if isinstance(p, dict):
            tasks_list = p.get('tasks', [])
            print(f"      - {p.get('name', '?')} ({p.get('duration', '?')})")
            for t in tasks_list[:3]:
                print(f"          • {t}")
    
    print(f"    Resources: {spec.estimated_resources}")
    print(f"    Time: {l1_time:.1f}s")
    
    # L1 Stats
    stats = layer1.get_stats()
    print(f"\n--- L1 Stats ---")
    print(f"    {json.dumps(stats, indent=4)}")
    
    # Verify original_intent is propagated for L2
    oi = spec.metadata.get('original_intent', {})
    print(f"\n--- L2 Handoff Data ---")
    print(f"    original_text: {oi.get('original_text', 'MISSING')[:60]}...")
    print(f"    intent_type: {oi.get('intent_type', 'MISSING')}")
    print(f"    goals: {oi.get('explicit_goals', 'MISSING')}")
    
    print(f"\n{'=' * 60}")
    print(f"RESULT: {'✓ LLM STRATEGY' if spec.metadata.get('strategy_source') == 'llm' else '✗ FALLBACK'}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    asyncio.run(main())
