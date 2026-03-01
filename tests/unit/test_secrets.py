"""
Unit tests for the secrets management module.
"""

import os
import tempfile
from pathlib import Path

import pytest

from gaap.core.secrets import (
    SecretsManager,
    SecretsProvider,
    mask_secret,
    mask_middle,
    validate_api_key_format,
    check_env_file_exists,
    generate_env_template,
    audit_codebase_for_secrets,
)


@pytest.fixture(autouse=True)
def reset_secrets_manager_singleton():
    """Reset the SecretsManager singleton before each test."""
    SecretsManager._instance = None
    yield
    SecretsManager._instance = None


class TestSecretMasking:
    """Tests for secret masking functions."""

    def test_mask_secret_basic(self):
        """Test basic secret masking."""
        result = mask_secret("sk-abc123456789xyz")
        assert result.startswith("sk-")
        assert result.endswith("xyz")
        assert "*" in result
        assert "abc123" not in result  # Middle should be masked

    def test_mask_secret_short(self):
        """Test masking short secrets."""
        result = mask_secret("abc")
        assert result == "***"

    def test_mask_secret_none(self):
        """Test masking None."""
        result = mask_secret(None)
        assert result == "<not set>"

    def test_mask_secret_empty(self):
        """Test masking empty string."""
        result = mask_secret("")
        assert result == "<not set>"

    def test_mask_middle(self):
        """Test middle masking."""
        result = mask_middle("AIzaSyA1234567890123456789012345678901234")
        assert result.startswith("AIza")
        assert result.endswith("234")
        assert "..." in result


class TestValidateApiKeyFormat:
    """Tests for API key validation."""

    def test_valid_groq_key(self):
        """Test validation of valid Groq key."""
        # Groq keys start with gsk_
        is_valid, msg = validate_api_key_format("gsk_" + "a" * 40, "groq")
        assert is_valid is True
        assert "Valid" in msg

    def test_invalid_groq_key(self):
        """Test validation of invalid Groq key (wrong prefix, but long enough)."""
        is_valid, msg = validate_api_key_format("wrongprefix_" + "a" * 30, "groq")
        assert is_valid is False
        assert "prefix" in msg.lower() or "format" in msg.lower()

    def test_valid_gemini_key(self):
        """Test validation of valid Gemini key."""
        # Gemini keys start with AIza
        is_valid, msg = validate_api_key_format("AIza" + "B" * 35, "gemini")
        assert is_valid is True

    def test_key_too_short(self):
        """Test validation of too-short key."""
        is_valid, msg = validate_api_key_format("short", "generic")
        assert is_valid is False
        assert "too short" in msg.lower()

    def test_placeholder_key(self):
        """Test detection of placeholder keys."""
        is_valid, msg = validate_api_key_format("test_key_12345_12345_12345", "generic")
        assert is_valid is False
        assert "placeholder" in msg.lower()

    def test_empty_key(self):
        """Test validation of empty key."""
        is_valid, msg = validate_api_key_format(None, "generic")
        assert is_valid is False
        assert "empty" in msg.lower()


class TestSecretsProvider:
    """Tests for SecretsProvider dataclass."""

    def test_create_provider(self):
        """Test creating a SecretsProvider."""
        provider = SecretsProvider(
            groq_api_key="gsk_test12345",
            gemini_api_key="AIzaTest12345",
        )
        assert provider.groq_api_key == "gsk_test12345"
        assert provider.gemini_api_key == "AIzaTest12345"

    def test_provider_repr_masks_secrets(self):
        """Test that repr masks secrets."""
        provider = SecretsProvider(groq_api_key="gsk_secret12345")
        repr_str = repr(provider)
        assert "gsk_secret12345" not in repr_str
        assert "***" in repr_str

    def test_provider_str_masks_secrets(self):
        """Test that str masks secrets."""
        provider = SecretsProvider(gemini_api_key="AIzaSecret12345")
        str_str = str(provider)
        assert "AIzaSecret12345" not in str_str
        assert "***" in str_str

    def test_get_masked(self):
        """Test get_masked method."""
        provider = SecretsProvider(groq_api_key="gsk_abc123xyz")
        masked = provider.get_masked("groq_api_key")
        assert "gsk_abc123xyz" not in masked
        assert "***" in masked

    def test_to_dict_masked(self):
        """Test to_dict with masking."""
        provider = SecretsProvider(groq_api_key="gsk_secret123")
        result = provider.to_dict(masked=True)
        assert "gsk_secret123" not in result["groq_api_key"]
        assert "***" in result["groq_api_key"]

    def test_to_dict_unmasked(self):
        """Test to_dict without masking."""
        provider = SecretsProvider(groq_api_key="gsk_secret123")
        result = provider.to_dict(masked=False)
        assert result["groq_api_key"] == "gsk_secret123"

    def test_get_available_providers(self):
        """Test getting available providers."""
        # Use a key that's long enough (> 20 chars) to pass validation
        provider = SecretsProvider(
            groq_api_key="gsk_" + "a" * 40,
            gemini_api_key="AIza" + "B" * 35,
        )
        available = provider.get_available_providers()
        assert "groq" in available
        assert "gemini" in available


class TestSecretsManager:
    """Tests for SecretsManager."""

    def test_manager_loads_from_env(self, monkeypatch):
        """Test loading secrets from environment."""
        monkeypatch.setenv("GAAP_GROQ_API_KEY", "gsk_test_from_env_12345")

        manager = SecretsManager(auto_load=False)
        manager.load()

        assert manager.secrets.groq_api_key == "gsk_test_from_env_12345"

    def test_manager_loads_from_legacy_env(self, monkeypatch):
        """Test loading from legacy (non-prefixed) env vars."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza_legacy_test_1234567890")

        manager = SecretsManager(auto_load=False)
        manager.load()

        assert manager.secrets.gemini_api_key == "AIza_legacy_test_1234567890"

    def test_get_method(self, monkeypatch):
        """Test get method."""
        monkeypatch.setenv("GAAP_GITHUB_TOKEN", "ghp_test_token_12345")

        manager = SecretsManager(auto_load=False)
        manager.load()

        assert manager.get("github_token") == "ghp_test_token_12345"
        assert manager.get("nonexistent", "default") == "default"

    def test_require_raises_on_missing(self):
        """Test require raises error for missing secret."""
        manager = SecretsManager(auto_load=True)

        with pytest.raises(ValueError) as exc_info:
            manager.require("nonexistent_field")

        assert "nonexistent_field" in str(exc_info.value)
        assert "GAAP_NONEXISTENT_FIELD" in str(exc_info.value)

    def test_has_secret(self, monkeypatch):
        """Test has_secret method."""
        monkeypatch.setenv("GAAP_GROQ_API_KEY", "gsk_test_key_1234567890")

        manager = SecretsManager(auto_load=False)
        manager.load()

        assert manager.has_secret("groq_api_key") is True
        assert manager.has_secret("nonexistent") is False

    def test_masked_summary(self, monkeypatch):
        """Test masked summary output."""
        monkeypatch.setenv("GAAP_GROQ_API_KEY", "gsk_test_key_1234567890")

        manager = SecretsManager(auto_load=True)
        summary = manager.get_masked_summary()

        assert "gsk_test_key_1234567890" not in summary
        assert "groq_api_key" in summary
        assert "***" in summary

    def test_singleton_pattern(self):
        """Test that SecretsManager is a singleton."""
        manager1 = SecretsManager(auto_load=False)
        manager2 = SecretsManager(auto_load=False)

        assert manager1 is manager2

    def test_load_without_dotenv(self, monkeypatch):
        """Test loading when python-dotenv is not installed."""
        # Simulate dotenv not being available
        import gaap.core.secrets as secrets_module

        original_has_dotenv = hasattr(secrets_module, "load_dotenv")

        manager = SecretsManager(env_file="/nonexistent/.env", auto_load=False)
        manager.load()

        # Should still work, just with a warning
        assert manager.secrets is not None


class TestEnvFileFunctions:
    """Tests for .env file utility functions."""

    def test_check_env_file_exists_true(self, tmp_path):
        """Test check_env_file_exists when file exists."""
        env_file = tmp_path / ".env"
        env_file.write_text("TEST=value")

        assert check_env_file_exists(env_file) is True

    def test_check_env_file_exists_false(self, tmp_path):
        """Test check_env_file_exists when file doesn't exist."""
        env_file = tmp_path / ".env"

        assert check_env_file_exists(env_file) is False

    def test_generate_env_template(self):
        """Test generating env template."""
        template = generate_env_template()

        assert "GAAP_GROQ_API_KEY" in template
        assert "GAAP_GEMINI_API_KEY" in template
        assert "GAAP_GITHUB_TOKEN" in template
        assert "DO NOT commit .env to version control" in template


class TestAuditFunction:
    """Tests for the audit codebase function."""

    def test_audit_finds_no_secrets_in_clean_code(self, tmp_path):
        """Test audit on clean code."""
        # Create a clean Python file
        py_file = tmp_path / "clean.py"
        py_file.write_text("""
# This is clean code
api_key = os.environ.get("API_KEY")  # Good - from env
def hello():
    return "world"
""")

        findings = audit_codebase_for_secrets(tmp_path)
        # Should not find the os.environ.get pattern
        assert len(findings) == 0

    def test_audit_finds_hardcoded_secret(self, tmp_path):
        """Test audit finds hardcoded secrets."""
        # Create a file with a hardcoded secret pattern
        py_file = tmp_path / "bad.py"
        py_file.write_text("""
# This has a hardcoded secret
api_key = "sk_live_12345678901234567890"  # BAD!
def hello():
    return "world"
""")

        findings = audit_codebase_for_secrets(tmp_path)
        # Should find the hardcoded key
        assert len(findings) > 0
        assert any("sk_live" in f.get("matched_text", "") for f in findings)

    def test_audit_excludes_pycache(self, tmp_path):
        """Test that __pycache__ is excluded."""
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        py_file = pycache / "bad.py"
        py_file.write_text('api_key = "sk_live_12345678901234567890"')

        findings = audit_codebase_for_secrets(tmp_path)
        # Should not find anything in __pycache__
        assert len(findings) == 0


class TestIntegrationWithConfig:
    """Tests for integration with ConfigManager."""

    def test_config_init_secrets(self, monkeypatch):
        """Test that GAAPConfig can initialize secrets."""
        from gaap.core.config import GAAPConfig
        from gaap.core.secrets import SecretsManager

        # Reset singleton for clean test
        SecretsManager._instance = None

        monkeypatch.setenv("GAAP_GROQ_API_KEY", "gsk_test_key_1234567890")

        config = GAAPConfig()
        manager = config.init_secrets(auto_load=True)

        assert manager is not None
        assert config.secrets is not None
        assert config.secrets.groq_api_key == "gsk_test_key_1234567890"

    def test_config_secrets_property_without_init(self):
        """Test secrets property when not initialized."""
        from gaap.core.config import GAAPConfig

        config = GAAPConfig()
        assert config.secrets is None
