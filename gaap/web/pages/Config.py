# mypy: ignore-errors
"""
Config Page - Manage Configuration
"""

import json
from typing import Any

import streamlit as st

DEFAULT_CONFIG = {
    "default_provider": "groq",
    "default_model": "llama-3.3-70b",
    "default_budget": 10.0,
    "temperature": 0.7,
    "max_tokens": 4096,
    "timeout": 120,
    "enable_healing": True,
    "enable_memory": True,
    "enable_security": True,
    "log_level": "INFO",
}


def load_config() -> dict:
    """Load configuration"""
    try:
        from gaap.storage import load_config as _load

        config = _load()
        return config if config else DEFAULT_CONFIG.copy()
    except Exception:
        return DEFAULT_CONFIG.copy()


def save_config(key: str, value: Any):
    """Save configuration value"""
    from gaap.storage import save_config as _save

    _save(key, value)


def main():
    st.title(":gear: Configuration")
    st.markdown("---")

    config = load_config()

    tab1, tab2, tab3 = st.tabs(["Settings", "Advanced", "Export/Import"])

    with tab1:
        st.markdown("### General Settings")

        col1, col2 = st.columns(2)

        with col1:
            default_provider = st.selectbox(
                "Default Provider",
                ["groq", "gemini", "cerebras", "mistral", "g4f"],
                index=["groq", "gemini", "cerebras", "mistral", "g4f"].index(
                    config.get("default_provider", "groq")
                ),
            )
            if default_provider != config.get("default_provider"):
                save_config("default_provider", default_provider)
                st.success("Saved!")

            default_model = st.text_input(
                "Default Model", config.get("default_model", "llama-3.3-70b")
            )
            if default_model != config.get("default_model"):
                save_config("default_model", default_model)
                st.success("Saved!")

        with col2:
            default_budget = st.number_input(
                "Default Budget ($)", min_value=0.1, value=config.get("default_budget", 10.0)
            )
            if default_budget != config.get("default_budget"):
                save_config("default_budget", default_budget)
                st.success("Saved!")

            timeout = st.number_input(
                "Timeout (seconds)", min_value=10, value=config.get("timeout", 120)
            )
            if timeout != config.get("timeout"):
                save_config("timeout", timeout)
                st.success("Saved!")

        st.markdown("### Model Parameters")

        col1, col2 = st.columns(2)

        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, config.get("temperature", 0.7), 0.1)
            if temperature != config.get("temperature"):
                save_config("temperature", temperature)
                st.success("Saved!")

        with col2:
            max_tokens = st.number_input(
                "Max Tokens", min_value=100, value=config.get("max_tokens", 4096)
            )
            if max_tokens != config.get("max_tokens"):
                save_config("max_tokens", max_tokens)
                st.success("Saved!")

    with tab2:
        st.markdown("### Feature Flags")

        enable_healing = st.checkbox("Enable Self-Healing", config.get("enable_healing", True))
        if enable_healing != config.get("enable_healing"):
            save_config("enable_healing", enable_healing)
            st.success("Saved!")

        enable_memory = st.checkbox("Enable Memory System", config.get("enable_memory", True))
        if enable_memory != config.get("enable_memory"):
            save_config("enable_memory", enable_memory)
            st.success("Saved!")

        enable_security = st.checkbox(
            "Enable Security Firewall", config.get("enable_security", True)
        )
        if enable_security != config.get("enable_security"):
            save_config("enable_security", enable_security)
            st.success("Saved!")

        st.markdown("### Logging")

        log_level = st.selectbox(
            "Log Level",
            ["DEBUG", "INFO", "WARNING", "ERROR"],
            index=["DEBUG", "INFO", "WARNING", "ERROR"].index(config.get("log_level", "INFO")),
        )
        if log_level != config.get("log_level"):
            save_config("log_level", log_level)
            st.success("Saved!")

    with tab3:
        st.markdown("### Export Configuration")

        if st.button("Export to JSON", use_container_width=True):
            st.download_button(
                "Download config.json",
                json.dumps(config, indent=2),
                file_name="gaap_config.json",
                mime="application/json",
            )

        st.markdown("### Import Configuration")

        uploaded_file = st.file_uploader("Upload config file", type=["json"])
        if uploaded_file:
            try:
                imported = json.load(uploaded_file)
                if st.button("Apply Imported Config"):
                    from gaap.storage import get_store

                    store = get_store()
                    store.save("config", imported)
                    st.success("Configuration imported!")
                    st.rerun()
            except Exception as e:
                st.error(f"Invalid JSON: {e}")

        st.markdown("### Reset")

        if st.button("Reset to Defaults", type="secondary"):
            from gaap.storage import get_store

            store = get_store()
            store.save("config", DEFAULT_CONFIG.copy())
            st.success("Configuration reset!")
            st.rerun()


if __name__ == "__main__":
    main()
