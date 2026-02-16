"""
Models Page - Browse Available Models
"""

from typing import Any

import streamlit as st

MODELS_INFO: dict[str, dict[str, Any]] = {
    "strategic": {
        "tier": "Tier 1 - Strategic",
        "description": "For complex planning, architecture decisions, and critical tasks",
        "models": [
            {"name": "gpt-4o", "context": 128000, "cost": "$2.50/1M tokens"},
            {"name": "claude-opus", "context": 200000, "cost": "$15/1M tokens"},
            {"name": "o1-preview", "context": 128000, "cost": "$15/1M tokens"},
        ],
    },
    "tactical": {
        "tier": "Tier 2 - Tactical",
        "description": "For code generation, analysis, and moderate complexity tasks",
        "models": [
            {"name": "llama-3.3-70b", "context": 128000, "cost": "Free (Groq)"},
            {"name": "gemini-1.5-pro", "context": 1000000, "cost": "Free tier"},
            {"name": "claude-sonnet", "context": 200000, "cost": "$3/1M tokens"},
        ],
    },
    "efficient": {
        "tier": "Tier 3 - Efficient",
        "description": "For simple tasks, quick responses, and high-volume operations",
        "models": [
            {"name": "llama-3.1-8b", "context": 128000, "cost": "Free (Groq)"},
            {"name": "gemini-1.5-flash", "context": 1000000, "cost": "Free tier"},
            {"name": "mistral-small", "context": 32000, "cost": "Free tier"},
        ],
    },
    "private": {
        "tier": "Tier 4 - Private/Local",
        "description": "For sensitive data, offline usage, and privacy requirements",
        "models": [
            {"name": "llama-3.2-local", "context": 8192, "cost": "Free (local)"},
            {"name": "mistral-local", "context": 32768, "cost": "Free (local)"},
        ],
    },
}


def format_context(tokens: int) -> str:
    """Format context size"""
    if tokens >= 1_000_000:
        return f"{tokens // 1_000_000}M"
    elif tokens >= 1_000:
        return f"{tokens // 1_000}K"
    return str(tokens)


def main() -> None:
    st.title(":books: Models")
    st.markdown("---")

    st.markdown("### Model Tiers")
    st.markdown("GAAP uses a 4-tier model selection system based on task complexity")

    selected_tier = st.selectbox(
        "Filter by Tier",
        ["All", "Strategic", "Tactical", "Efficient", "Private"],
        index=0,
    )

    for tier_key, tier_info in MODELS_INFO.items():
        if selected_tier != "All" and selected_tier.lower() not in tier_key.lower():
            continue

        with st.expander(
            f"**{tier_info['tier']}** ({len(tier_info['models'])} models)",
            expanded=(selected_tier != "All"),
        ):
            st.markdown(f"*{tier_info['description']}*")
            st.markdown("")

            for model in tier_info["models"]:
                col1, col2, col3 = st.columns([3, 2, 2])

                with col1:
                    st.markdown(f"**{model['name']}**")

                with col2:
                    ctx = format_context(model["context"])
                    st.metric("Context", f"{ctx} tokens")

                with col3:
                    st.metric("Cost", model["cost"])

                st.markdown("---")

    with st.sidebar:
        st.markdown("### Quick Stats")

        total_models = sum(len(t["models"]) for t in MODELS_INFO.values())
        st.metric("Total Models", total_models)

        free_models = sum(
            1 for t in MODELS_INFO.values() for m in t["models"] if "Free" in m["cost"]
        )
        st.metric("Free Models", free_models)

        st.markdown("### Tier Guide")
        st.markdown("""
        | Tier | Use Case |
        |------|----------|
        | Strategic | Complex planning, architecture |
        | Tactical | Code gen, analysis |
        | Efficient | Simple tasks, quick responses |
        | Private | Sensitive data, offline |
        """)


if __name__ == "__main__":
    main()
