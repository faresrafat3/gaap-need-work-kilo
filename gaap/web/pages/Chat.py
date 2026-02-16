"""
Chat Page - Interactive AI Chat with Streaming
"""

import asyncio
import contextlib
import os
from typing import Any

import streamlit as st


def load_env() -> None:
    """Load environment variables"""
    from pathlib import Path

    def parse_lines(lines: list[str]) -> None:
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

    paths = [Path.home() / ".gaap_env", Path.cwd() / ".gaap_env"]
    for path in paths:
        if path.exists() and path.is_file():
            try:
                parse_lines(path.read_text(encoding="utf-8").splitlines())
            except OSError:
                continue


def init_session_state() -> None:
    """Initialize session state"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "engine" not in st.session_state:
        st.session_state.engine = None


async def get_engine() -> Any:
    """Get or create GAAP engine"""
    if st.session_state.engine is None:
        from gaap.gaap_engine import create_engine

        st.session_state.engine = create_engine(
            groq_api_key=os.environ.get("GROQ_API_KEY"),
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            budget=st.session_state.get("budget", 10.0),
            enable_all=True,
        )
    return st.session_state.engine


def main() -> None:
    st.title(":speech_balloon: Chat")
    st.markdown("---")

    load_env()
    init_session_state()

    with st.sidebar:
        st.markdown("### Settings")
        st.session_state.budget = st.number_input("Budget ($)", min_value=0.1, value=10.0, step=0.5)

        st.markdown("### Options")
        show_stats = st.checkbox("Show Stats", value=True)
        save_to_history = st.checkbox("Save History", value=True)

        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("New Session", use_container_width=True):
            if st.session_state.engine:
                engine = st.session_state.engine
                for provider in getattr(engine, "providers", []):
                    close_fn = getattr(provider, "close", None)
                    if callable(close_fn):
                        with contextlib.suppress(Exception):
                            asyncio.run(close_fn())
                engine.shutdown()
            st.session_state.engine = None
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                engine = asyncio.run(get_engine())
                response = asyncio.run(engine.chat(prompt))
                full_response = str(response)
                message_placeholder.markdown(full_response)

                if save_to_history:
                    from gaap.storage import save_history

                    save_history("user", prompt)
                    save_history("assistant", full_response, provider="gaap", model="default")

                if show_stats:
                    stats = engine.get_stats()
                    st.caption(
                        f"Requests: {stats.get('requests_processed', 0)} | Success: {stats.get('success_rate', 0):.1%}"
                    )

            except Exception as e:
                full_response = f":red[Error: {e}]"
                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()
