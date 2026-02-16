"""
Doctor Page - System Diagnostics
"""

import os
import platform
import sys

import streamlit as st


def run_diagnostics() -> tuple[list[str], list[tuple[str, bool, str]]]:
    """Run all diagnostics"""
    issues = []
    results = []

    py_version = sys.version_info
    if py_version < (3, 10):
        issues.append("Python 3.10+ required")
        results.append(("Python Version", False, f"{py_version.major}.{py_version.minor}"))
    else:
        results.append(("Python Version", True, f"{py_version.major}.{py_version.minor}"))

    required_packages = ["aiohttp", "httpx", "pyyaml", "structlog", "streamlit"]
    for pkg in required_packages:
        try:
            __import__(pkg)
            results.append((f"Package: {pkg}", True, "Installed"))
        except ImportError:
            issues.append(f"{pkg} not installed")
            results.append((f"Package: {pkg}", False, "Not installed"))

    try:
        from gaap import GAAPEngine

        results.append(("GAAP Core", True, "OK"))
    except ImportError as e:
        issues.append(f"GAAP import error: {e}")
        results.append(("GAAP Core", False, str(e)))

    try:
        from gaap.storage import get_store

        store = get_store()
        test_data = {"test": "data"}
        store.save("test", test_data)
        loaded = store.load("test")
        if loaded == test_data:
            results.append(("Storage", True, "OK"))
        else:
            issues.append("Storage read/write mismatch")
            results.append(("Storage", False, "Mismatch"))
    except Exception as e:
        issues.append(f"Storage error: {e}")
        results.append(("Storage", False, str(e)))

    api_keys = {
        "GROQ_API_KEY": os.environ.get("GROQ_API_KEY"),
        "GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY"),
        "CEREBRAS_API_KEY": os.environ.get("CEREBRAS_API_KEY"),
        "MISTRAL_API_KEY": os.environ.get("MISTRAL_API_KEY"),
    }

    for key, value in api_keys.items():
        if value:
            masked = value[:8] + "..." if len(value) > 8 else value
            results.append((key, True, masked))
        else:
            results.append((key, False, "Not set"))

    has_any_key = any(api_keys.values())
    if not has_any_key:
        issues.append("No API keys configured")

    return issues, results


def main() -> None:
    st.title(":wrench: System Diagnostics")
    st.markdown("---")

    st.markdown("### System Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Python", sys.version.split()[0])
        st.metric("Platform", f"{platform.system()} {platform.release()}")

    with col2:
        st.metric("GAAP Version", "1.0.0")
        st.metric("Architecture", platform.machine())

    st.markdown("---")

    st.markdown("### Run Diagnostics")

    if st.button("Run Full Diagnostics", use_container_width=True, type="primary"):
        with st.spinner("Running diagnostics..."):
            issues, results = run_diagnostics()

        st.markdown("### Results")

        for name, success, detail in results:
            icon = ":white_check_mark:" if success else ":x:"

            with st.container(border=True):
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"**{icon} {name}**")
                with col2:
                    st.caption(detail)

        st.markdown("---")

        if issues:
            st.markdown(f"### :warning: Found {len(issues)} Issue(s)")
            for issue in issues:
                st.error(issue)
        else:
            st.markdown("### :white_check_mark: All Checks Passed!")

    st.markdown("---")

    st.markdown("### Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Check API Keys", use_container_width=True):
            st.markdown("### API Keys Status")
            keys = ["GROQ_API_KEY", "GEMINI_API_KEY", "CEREBRAS_API_KEY", "MISTRAL_API_KEY"]
            for key in keys:
                value = os.environ.get(key)
                if value:
                    masked = value[:8] + "..."
                    st.success(f"{key}: {masked}")
                else:
                    st.warning(f"{key}: Not set")

    with col2:
        if st.button("Test Storage", use_container_width=True):
            try:
                from gaap.storage import get_store

                store = get_store()
                store.save("test", {"test": True})
                store.load("test")
                st.success("Storage OK")
            except Exception as e:
                st.error(f"Storage Error: {e}")

    with col3:
        if st.button("Check Modules", use_container_width=True):
            modules = ["gaap.core", "gaap.providers", "gaap.layers", "gaap.routing"]
            for mod in modules:
                try:
                    __import__(mod)
                    st.success(f"{mod}")
                except ImportError as e:
                    st.error(f"{mod}: {e}")

    with st.sidebar:
        st.markdown("### Help")
        st.markdown("""
        **Common Issues:**

        1. **No API keys**: Create `.gaap_env` file with your keys

        2. **Import errors**: Run `pip install -e .`

        3. **Storage errors**: Check `~/.gaap/` permissions

        4. **Connection errors**: Check your internet connection
        """)


if __name__ == "__main__":
    main()
