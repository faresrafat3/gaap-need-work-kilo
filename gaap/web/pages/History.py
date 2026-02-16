"""
History Page - View Conversation History
"""

import streamlit as st


def load_history(limit: int = 100):
    """Load conversation history"""
    try:
        from gaap.storage import load_history as _load

        return _load(limit=limit)
    except Exception:
        return []


def main():
    st.title(":clipboard: Conversation History")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("### Options")
        limit = st.slider("Messages to show", 10, 200, 50)
        search_query = st.text_input("Search", placeholder="Search history...")

        if st.button("Refresh", use_container_width=True):
            st.rerun()

        if st.button("Clear History", type="secondary", use_container_width=True):
            if st.session_state.get("confirm_clear"):
                from gaap.storage import clear_history

                clear_history()
                st.session_state.confirm_clear = False
                st.success("History cleared!")
                st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm")

        st.markdown("### Stats")
        history = load_history(limit=10000)
        st.metric("Total Messages", len(history))

        user_msgs = sum(1 for h in history if h.get("role") == "user")
        st.metric("User Messages", user_msgs)

        assistant_msgs = sum(1 for h in history if h.get("role") == "assistant")
        st.metric("Assistant Messages", assistant_msgs)

    with col1:
        history = load_history(limit=limit)

        if not history:
            st.info("No conversation history yet. Start chatting!")
            return

        if search_query:
            history = [h for h in history if search_query.lower() in h.get("content", "").lower()]
            st.info(f"Found {len(history)} matching messages")

        for _i, item in enumerate(reversed(history[-limit:])):
            timestamp = item.get("timestamp", "Unknown")
            role = item.get("role", "unknown")
            content = item.get("content", "")

            with st.chat_message(role):
                st.caption(f"{timestamp[:16]} | {role}")
                st.markdown(content)

                if item.get("provider") or item.get("model"):
                    st.caption(
                        f"Provider: {item.get('provider', 'N/A')} | Model: {item.get('model', 'N/A')}"
                    )

                if item.get("tokens") or item.get("cost"):
                    st.caption(
                        f"Tokens: {item.get('tokens', 'N/A')} | Cost: ${item.get('cost', 0):.4f}"
                    )


if __name__ == "__main__":
    main()
