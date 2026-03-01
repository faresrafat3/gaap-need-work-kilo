#!/usr/bin/env python3
"""
Demo script for GAAP Secrets Management System.

This demonstrates the secure handling of API keys and secrets.
"""

import os
from gaap.core.secrets import (
    SecretsManager,
    SecretsProvider,
    mask_secret,
    validate_api_key_format,
    audit_codebase_for_secrets,
    generate_env_template,
    get_secrets,
    init_secrets,
)


def demo_masking():
    """Demonstrate secret masking."""
    print("=" * 60)
    print("DEMO: Secret Masking")
    print("=" * 60)

    secret = "sk-live-12345678901234567890"
    print(f"Original: {secret}")
    print(f"Masked:   {mask_secret(secret)}")
    print()

    gemini_key = "AIzaSyA1234567890123456789012345678901234"
    print(f"Gemini Key: {gemini_key}")
    print(f"Masked:     {mask_secret(gemini_key, visible_prefix=4, visible_suffix=4)}")
    print()


def demo_validation():
    """Demonstrate API key validation."""
    print("=" * 60)
    print("DEMO: API Key Validation")
    print("=" * 60)

    test_keys = [
        ("gsk_valid_key_12345678901234567890", "groq"),
        ("invalid_key", "groq"),
        ("AIzaSyA1234567890123456789012345678901234", "gemini"),
        ("test_key_12345678901234567890", "generic"),
    ]

    for key, provider in test_keys:
        is_valid, msg = validate_api_key_format(key, provider)
        status = "✓" if is_valid else "✗"
        print(f"{status} {provider}: {msg}")
    print()


def demo_secrets_manager():
    """Demonstrate SecretsManager."""
    print("=" * 60)
    print("DEMO: SecretsManager")
    print("=" * 60)

    # Create a secrets manager
    manager = SecretsManager(auto_load=False)

    # Set some example environment variables
    os.environ["GAAP_GEMINI_API_KEY"] = "AIzaDemoKey123456789012345678901234"
    os.environ["GAAP_GITHUB_TOKEN"] = "ghp_demo_token_1234567890abcdef1234"

    # Load secrets
    manager.load()

    # Access secrets safely
    print(f"Gemini API Key (masked): {manager.get_masked('gemini_api_key')}")
    print(f"GitHub Token (masked):   {manager.get_masked('github_token')}")

    # Get full masked summary
    print("\nFull Summary:")
    print(manager.get_masked_summary())
    print()


def demo_audit():
    """Demonstrate codebase audit."""
    print("=" * 60)
    print("DEMO: Codebase Audit for Secrets")
    print("=" * 60)

    # Audit the current codebase
    findings = audit_codebase_for_secrets("gaap/core")

    if findings:
        print(f"⚠️  Found {len(findings)} potential issues (may be false positives)")
        for finding in findings[:3]:  # Show first 3
            print(f"  - {finding['file']}:{finding['line']} ({finding['pattern']})")
    else:
        print("✅ No hardcoded secrets detected in gaap/core")
    print()


def demo_config_integration():
    """Demonstrate integration with ConfigManager."""
    print("=" * 60)
    print("DEMO: ConfigManager Integration")
    print("=" * 60)

    from gaap.core.config import GAAPConfig, ConfigBuilder

    # Method 1: Using ConfigBuilder with secrets
    config = (
        ConfigBuilder()
        .with_system(name="MyApp", environment="development")
        .with_secrets(auto_load=True)
        .build()
    )

    print("Config created with secrets support")
    if config.secrets:
        print(f"Secrets available: {config.secrets.get_available_providers()}")
    else:
        print("No secrets loaded (set environment variables to test)")
    print()

    # Method 2: Initialize secrets on existing config
    config2 = GAAPConfig()
    config2.init_secrets(auto_load=True)
    print("Secrets initialized on existing config")
    print()


def demo_env_template():
    """Demonstrate generating .env template."""
    print("=" * 60)
    print("DEMO: Generate .env Template")
    print("=" * 60)

    template = generate_env_template()
    lines = template.split("\n")

    print(f"Generated template with {len(lines)} lines")
    print("First 20 lines:")
    for line in lines[:20]:
        print(line)
    print("...")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GAAP Secrets Management System Demo")
    print("=" * 60)
    print()

    demo_masking()
    demo_validation()
    demo_secrets_manager()
    demo_audit()
    demo_config_integration()
    demo_env_template()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
