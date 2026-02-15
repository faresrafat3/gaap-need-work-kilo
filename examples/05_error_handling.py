"""
GAAP Error Handling and Healing Example

This example demonstrates the self-healing system and error handling.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gaap.core.types import HealingLevel, TaskPriority
from gaap.core.exceptions import (
    GAAPException,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    TaskError,
)


def exception_handling_example():
    """Example of handling GAAP exceptions"""
    print("\n=== Exception Handling Example ===")

    # Create a rate limit error
    try:
        raise ProviderRateLimitError(
            provider_name="groq",
            retry_after=60,
        )
    except GAAPException as e:
        print(f"Error Code: {e.error_code}")
        print(f"Message: {e.message}")
        print(f"Recoverable: {e.recoverable}")
        print(f"Suggestions: {e.suggestions}")
        print(f"Details: {e.details}")

    # Create a timeout error
    try:
        raise ProviderTimeoutError(
            provider_name="gemini",
            timeout_seconds=30.0,
        )
    except GAAPException as e:
        print(f"\nTimeout Error: {e.message}")
        print(f"Recoverable: {e.recoverable}")


def healing_levels_example():
    """Example of healing levels"""
    print("\n=== Healing Levels Example ===")

    healing_flow = {
        HealingLevel.L1_RETRY: "Simple retry with exponential backoff",
        HealingLevel.L2_REFINE: "Refine the prompt/request",
        HealingLevel.L3_PIVOT: "Switch to a different model/provider",
        HealingLevel.L4_STRATEGY_SHIFT: "Change the execution strategy",
        HealingLevel.L5_HUMAN_ESCALATION: "Escalate to human intervention",
    }

    for level, description in healing_flow.items():
        print(f"{level.name}: {description}")


async def retry_logic_example():
    """Example of retry logic"""
    print("\n=== Retry Logic Example ===")

    attempts = 0
    max_attempts = 3

    async def flaky_operation():
        nonlocal attempts
        attempts += 1
        print(f"Attempt {attempts}...")

        if attempts < 2:
            raise ProviderRateLimitError(provider_name="test", retry_after=1)

        return "Success!"

    # Retry loop
    result = None
    last_error = None

    while attempts < max_attempts:
        try:
            result = await flaky_operation()
            break
        except ProviderRateLimitError as e:
            last_error = e
            if e.recoverable:
                wait_time = e.details.get("retry_after_seconds", 1)
                print(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(0.1)  # Shortened for example
            else:
                raise

    print(f"Final result: {result}")
    print(f"Total attempts: {attempts}")


def error_severity_example():
    """Example of error severity levels"""
    print("\n=== Error Severity Example ===")

    errors = [
        ProviderError(message="API error", details={"code": 500}),
        ProviderRateLimitError(provider_name="test"),
        TaskError(message="Task failed"),
    ]

    for error in errors:
        print(f"{error.__class__.__name__}:")
        print(f"  Severity: {error.severity}")
        print(f"  Category: {error.error_category}")
        print(f"  Recoverable: {error.recoverable}")


def to_dict_example():
    """Example of serializing exceptions"""
    print("\n=== Exception Serialization Example ===")

    error = ProviderError(
        message="Connection failed",
        details={"host": "api.example.com"},
        suggestions=["Check network", "Try again later"],
    )

    error_dict = error.to_dict()

    print("Serialized error:")
    for key, value in error_dict.items():
        print(f"  {key}: {value}")


async def main():
    """Run all examples"""
    exception_handling_example()
    healing_levels_example()
    await retry_logic_example()
    error_severity_example()
    to_dict_example()

    print("\nError handling example complete!")


if __name__ == "__main__":
    asyncio.run(main())
