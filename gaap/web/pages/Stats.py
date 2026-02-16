# mypy: ignore-errors
"""
Stats Page - Usage Statistics and Charts
"""

from typing import Any

import pandas as pd
import streamlit as st


def load_stats() -> dict[str, Any]:
    """Load statistics"""
    try:
        from gaap.storage import load_stats as _load

        return _load()
    except Exception:
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "total_errors": 0,
            "by_provider": {},
            "by_model": {},
            "daily": {},
        }


def main():
    st.title(":bar_chart: Statistics")
    st.markdown("---")

    stats = load_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Requests", stats.get("total_requests", 0))

    with col2:
        st.metric("Total Tokens", f"{stats.get('total_tokens', 0):,}")

    with col3:
        st.metric("Total Cost", f"${stats.get('total_cost', 0):.4f}")

    with col4:
        errors = stats.get("total_errors", 0)
        total = max(stats.get("total_requests", 1), 1)
        error_rate = (errors / total) * 100
        st.metric("Error Rate", f"{error_rate:.1f}%")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["By Provider", "By Model", "Daily"])

    with tab1:
        st.markdown("### Usage by Provider")

        by_provider = stats.get("by_provider", {})

        if not by_provider:
            st.info("No provider data yet")
        else:
            providers = list(by_provider.keys())
            requests = [by_provider[p].get("requests", 0) for p in providers]
            [by_provider[p].get("tokens", 0) for p in providers]
            costs = [by_provider[p].get("cost", 0) for p in providers]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Requests by Provider**")
                df = pd.DataFrame({"Provider": providers, "Requests": requests})
                st.bar_chart(df.set_index("Provider"))

            with col2:
                st.markdown("**Cost by Provider**")
                df = pd.DataFrame({"Provider": providers, "Cost": costs})
                st.bar_chart(df.set_index("Provider"))

            st.markdown("**Provider Details**")
            for provider, data in by_provider.items():
                with st.expander(f"{provider}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Requests", data.get("requests", 0))
                    col2.metric("Tokens", f"{data.get('tokens', 0):,}")
                    col3.metric("Cost", f"${data.get('cost', 0):.4f}")

    with tab2:
        st.markdown("### Usage by Model")

        by_model = stats.get("by_model", {})

        if not by_model:
            st.info("No model data yet")
        else:
            models = list(by_model.keys())
            requests = [by_model[m].get("requests", 0) for m in models]

            df = pd.DataFrame({"Model": models, "Requests": requests})
            st.bar_chart(df.set_index("Model"))

            st.markdown("**Model Details**")
            for model, data in by_model.items():
                st.markdown(
                    f"- **{model}**: {data.get('requests', 0)} requests, {data.get('tokens', 0):,} tokens, ${data.get('cost', 0):.4f}"
                )

    with tab3:
        st.markdown("### Daily Usage")

        daily = stats.get("daily", {})

        if not daily:
            st.info("No daily data yet")
        else:
            dates = sorted(daily.keys())
            requests_data = [daily[d].get("requests", 0) for d in dates]
            tokens_data = [daily[d].get("tokens", 0) for d in dates]
            cost_data = [daily[d].get("cost", 0) for d in dates]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Daily Requests**")
                df = pd.DataFrame({"Date": dates, "Requests": requests_data})
                st.line_chart(df.set_index("Date"))

            with col2:
                st.markdown("**Daily Cost**")
                df = pd.DataFrame({"Date": dates, "Cost": cost_data})
                st.line_chart(df.set_index("Date"))

            st.markdown("**Daily Details**")
            df = pd.DataFrame(
                {
                    "Date": dates,
                    "Requests": requests_data,
                    "Tokens": tokens_data,
                    "Cost": cost_data,
                }
            )
            st.dataframe(df, use_container_width=True)

    with st.sidebar:
        st.markdown("### Actions")

        if st.button("Refresh Stats", use_container_width=True):
            st.rerun()

        if st.button("Export Stats", use_container_width=True):
            import json

            st.download_button(
                "Download stats.json",
                json.dumps(stats, indent=2),
                file_name="gaap_stats.json",
                mime="application/json",
            )

        st.markdown("### Info")
        st.caption("Statistics are stored locally in ~/.gaap/stats.json")
