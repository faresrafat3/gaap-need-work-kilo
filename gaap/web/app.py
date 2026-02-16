"""
GAAP Web UI - Main Dashboard

Run with: streamlit run gaap/web/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="GAAP",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.title("GAAP - General-purpose AI Architecture Platform")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### :speech_balloon: Chat")
        st.markdown("Interactive chat with AI models")
        if st.button("Open Chat", key="chat_btn", use_container_width=True):
            st.switch_page("pages/Chat.py")

    with col2:
        st.markdown("### :bar_chart: Statistics")
        st.markdown("View usage and performance")
        if st.button("View Stats", key="stats_btn", use_container_width=True):
            st.switch_page("pages/Stats.py")

    with col3:
        st.markdown("### :gear: Configuration")
        st.markdown("Manage settings")
        if st.button("Configure", key="config_btn", use_container_width=True):
            st.switch_page("pages/Config.py")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### :outbox_tray: Providers")
        st.markdown("Manage LLM providers")
        if st.button("Manage Providers", key="providers_btn", use_container_width=True):
            st.switch_page("pages/Providers.py")

    with col2:
        st.markdown("### :books: Models")
        st.markdown("Browse available models")
        if st.button("View Models", key="models_btn", use_container_width=True):
            st.switch_page("pages/Models.py")

    st.markdown("---")

    with st.expander(":clipboard: Conversation History"):
        if st.button("View History", key="history_btn"):
            st.switch_page("pages/History.py")

    with st.expander(":wrench: System Diagnostics"):
        if st.button("Run Diagnostics", key="doctor_btn"):
            st.switch_page("pages/Doctor.py")

    st.sidebar.markdown("## GAAP v1.0.0")
    st.sidebar.markdown("---")

    try:
        from gaap.storage import load_stats

        stats = load_stats()
        st.sidebar.metric("Total Requests", stats.get("total_requests", 0))
        st.sidebar.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")
        st.sidebar.metric("Total Cost", f"${stats.get('total_cost', 0):.4f}")
    except Exception:
        st.sidebar.info("No stats available yet")


if __name__ == "__main__":
    main()
