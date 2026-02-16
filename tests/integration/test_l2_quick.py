"""
Quick L2 Decomposer Test — Minimal pressure
Single request, no loops, no parallel
"""

import asyncio
import time


async def main():
    print("=" * 50)
    print("L2 LLM Decomposer — Quick Test")
    print("=" * 50)

    # 1. Import
    from gaap_engine import GAAPEngine, GAAPRequest
    from providers.webchat_bridge import create_kimi_provider

    # 2. Create provider
    provider = create_kimi_provider(timeout=180)

    # 3. Create engine — minimal config
    engine = GAAPEngine(
        providers=[provider],
        budget=10.0,
        enable_context=False,
        enable_healing=False,
        enable_memory=False,
        enable_security=True,
    )

    # 4. Reduce load
    engine.layer2.decomposer.max_subtasks = 3
    engine.layer3.twin_system.enabled = False
    engine.layer3.executor_pool.max_parallel = 1

    # 5. Single request — should go through L1→L2→L3
    print("\nSending request...")
    request = GAAPRequest(
        text="Write a Python function that implements binary search on a sorted list"
    )

    start = time.time()
    response = await engine.process(request)
    elapsed = time.time() - start

    # 6. Report
    print(f"\n{'=' * 50}")
    print(f"Success: {response.success}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Intent: {response.intent.intent_type.name if response.intent else 'N/A'}")
    print(f"Routing: {response.intent.routing_target.value if response.intent else 'N/A'}")

    if response.task_graph:
        graph = response.task_graph
        print("\n--- L2 Decomposition ---")
        print(f"Total tasks: {graph.total_tasks}")
        print(f"Max depth: {graph.max_depth}")
        for node_id, node in graph.all_nodes.items():
            t = node.task
            src = t.metadata.get("source", "unknown")
            print(f"  [{src}] {t.name}")
            print(f"    → {t.description[:100]}...")
    else:
        print("\n(No task graph — went DIRECT)")

    print(f"\nExecution results: {len(response.execution_results)}")
    for i, r in enumerate(response.execution_results):
        print(f"  R{i+1}: success={r.success}, quality={r.quality_score:.1f}")

    # L2 stats
    stats = engine.layer2.get_stats()
    print(f"\nL2 Stats: {stats}")

    if response.output:
        print("\n--- Output (first 500 chars) ---")
        print(str(response.output)[:500])

    print(f"\n{'=' * 50}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
