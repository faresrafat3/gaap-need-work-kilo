"""
GAAP Observability Example

This example shows how to use the observability features
for tracing and metrics collection.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap.core.observability import (
    Observability,
    TracingConfig,
    MetricsConfig,
    traced,
)


async def traced_function():
    """A function with automatic tracing"""
    await asyncio.sleep(0.1)
    return "result"


async def main():
    """Main example"""

    obs = Observability(
        tracing_config=TracingConfig(
            service_name="gaap-example",
            environment="development",
            enable_console_export=True,
        ),
        metrics_config=MetricsConfig(
            namespace="gaap_example",
        ),
    )

    # Method 1: Using context manager
    with obs.trace_span("example_operation", layer="example"):
        result = await traced_function()
        print(f"Result: {result}")

    # Method 2: Using decorator
    @obs.traced(layer="example")
    async def another_function(x: int):
        return x * 2

    result = await another_function(21)
    print(f"Decorated result: {result}")

    # Record LLM metrics
    obs.record_llm_call(
        provider="groq",
        model="llama-3.3-70b",
        input_tokens=100,
        output_tokens=50,
        cost=0.001,
        latency=0.5,
        success=True,
    )

    # Record healing attempt
    obs.record_healing("L1_RETRY", success=True)

    # Record error
    obs.record_error("example", "TestError", "warning")

    print("\nObservability example complete!")


if __name__ == "__main__":
    asyncio.run(main())
