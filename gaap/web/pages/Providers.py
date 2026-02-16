"""
Providers Page - Manage LLM Providers
"""

import os

import streamlit as st

PROVIDERS_INFO = {
    "groq": {
        "name": "Groq",
        "type": "Free Tier",
        "models": ["llama-3.3-70b", "llama-3.1-8b", "mixtral-8x7b"],
        "env_key": "GROQ_API_KEY",
        "status": "available",
    },
    "gemini": {
        "name": "Google Gemini",
        "type": "Free Tier",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro"],
        "env_key": "GEMINI_API_KEY",
        "status": "available",
    },
    "cerebras": {
        "name": "Cerebras",
        "type": "Free Tier",
        "models": ["llama-3.3-70b"],
        "env_key": "CEREBRAS_API_KEY",
        "status": "available",
    },
    "mistral": {
        "name": "Mistral",
        "type": "Free Tier",
        "models": ["mistral-small", "mistral-medium"],
        "env_key": "MISTRAL_API_KEY",
        "status": "available",
    },
    "g4f": {
        "name": "G4F (Free)",
        "type": "Free",
        "models": ["auto", "gpt-4", "claude-3"],
        "env_key": None,
        "status": "available",
    },
}


def get_provider_status(env_key: str) -> tuple[bool, str]:
    """Check if provider is configured"""
    if env_key is None:
        return True, "No key required"
    value = os.environ.get(env_key)
    if value:
        masked = value[:8] + "..." if len(value) > 8 else value
        return True, masked
    return False, "Not configured"


def main():
    st.title(":outbox_tray: Providers")
    st.markdown("---")

    st.markdown("### Available Providers")

    for key, info in PROVIDERS_INFO.items():
        has_key, key_display = get_provider_status(info["env_key"])
        status_icon = ":white_check_mark:" if has_key else ":x:"

        with st.container(border=True):
            col1, col2, col3 = st.columns([2, 2, 1])

            with col1:
                st.markdown(f"### {status_icon} {info['name']}")
                st.caption(f"Type: {info['type']}")

            with col2:
                st.markdown("**Models:**")
                st.caption(", ".join(info["models"][:3]))
                if len(info["models"]) > 3:
                    st.caption(f"+{len(info['models']) - 3} more")

            with col3:
                st.metric("Status", "Ready" if has_key else "Not Configured")

            with st.expander("Details"):
                st.markdown(f"**Provider ID:** `{key}`")
                if info["env_key"]:
                    st.markdown(f"**Env Key:** `{info['env_key']}`")
                    st.markdown(f"**Key Status:** `{key_display}`")
                else:
                    st.markdown("**No API key required**")

                st.markdown("**All Models:**")
                for model in info["models"]:
                    st.markdown(f"- {model}")

                if has_key and info["env_key"]:
                    if st.button("Test Connection", key=f"test_{key}"):
                        st.info("Connection test coming soon!")

    with st.sidebar:
        st.markdown("### Quick Actions")

        if st.button("Refresh Status", use_container_width=True):
            st.rerun()

        st.markdown("### Setup Guide")
        st.markdown("""
        To configure providers:

        1. Create `.gaap_env` file
        2. Add your API keys:
           ```
           GROQ_API_KEY=gsk_...
           GEMINI_API_KEY=...
           ```
        3. Restart the app
        """)


if __name__ == "__main__":
    main()
