"""Airport Investment Intelligence Agent — Streamlit Chat UI."""

import os
import json
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Airport Investment Intelligence",
    page_icon="✈️",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.title("✈️ Airport Investment Intelligence")
    st.markdown("**AI-powered airport investment analysis**")
    st.divider()

    st.subheader("About")
    st.markdown(
        """
        This agent helps analysts identify US airports
        with the highest need for modernization investment.

        **Scoring KPIs:**
        - 🔴 Congestion (25%)
        - 📈 Growth (20%)
        - ⏱️ Delay (20%)
        - 🏗️ Facility Constraint (20%)
        - 📊 Demand Gap (15%)

        All scores are **deterministic** — computed from
        real data with no AI involvement in calculations.
        """
    )

    st.divider()

    # Data status
    st.subheader("Data Status")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    files = {
        "airports.csv": "Airport Master Data",
        "runways.csv": "Runway Infrastructure",
        "t100_data.csv": "Passenger Volumes (T-100)",
        "ontime_data.csv": "On-Time Performance",
    }
    for fname, label in files.items():
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            st.markdown(f"✅ {label} ({size_mb:.1f} MB)")
        else:
            st.markdown(f"❌ {label} — missing")

    # API key status
    st.divider()
    st.subheader("API Status")
    if os.environ.get("MISTRAL_API_KEY"):
        st.markdown("✅ Mistral API key configured")
    else:
        st.markdown("❌ Mistral API key missing")

    if os.environ.get("GEMINI_API_KEY"):
        st.markdown("✅ Gemini API key configured")
    else:
        st.markdown("❌ Gemini API key missing")

    if not os.environ.get("MISTRAL_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        st.markdown("⚠️ Set `MISTRAL_API_KEY` or `GEMINI_API_KEY` in `.env`")

    # Show active provider if agent is initialized
    if "llm_provider" in st.session_state:
        st.markdown(f"🤖 **Active LLM:** {st.session_state.llm_provider.title()}")

    if os.environ.get("AVIATIONSTACK_API_KEY"):
        st.markdown("✅ AviationStack API (live flights)")
    else:
        st.markdown("ℹ️ AviationStack API not configured (optional)")

    st.divider()
    st.markdown(
        """
        **Example Questions:**
        - *Which airports in New England need investment?*
        - *Compare LAX and SNA congestion levels*
        - *What % of ANC flights are long-haul?*
        - *Is SFO over capacity?*
        """
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.pop("agent", None)
        st.session_state.pop("agent_config", None)
        st.rerun()


# --- Main Chat Area ---
st.header("Airport Investment Intelligence Agent")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about airport investment opportunities..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check for at least one API key
    if not os.environ.get("MISTRAL_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        error_msg = (
            "⚠️ **No LLM API key configured.**\n\n"
            "Please set at least one key in a `.env` file in the project root:\n"
            "```\nMISTRAL_API_KEY=your_key_here\nGEMINI_API_KEY=your_key_here\n```"
        )
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.markdown(error_msg)
        st.stop()

    # Initialize agent (lazy, cached in session state)
    if "agent" not in st.session_state:
        with st.spinner("Initializing agent and loading data..."):
            try:
                from agent.agent import create_agent
                agent, config, provider = create_agent(thread_id="streamlit_session")
                st.session_state.agent = agent
                st.session_state.agent_config = config
                st.session_state.llm_provider = provider
            except Exception as e:
                error_msg = f"⚠️ **Failed to initialize agent:** {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)
                st.stop()

    # Invoke agent
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                from agent.agent import invoke_agent
                response, new_agent, new_provider = invoke_agent(
                    st.session_state.agent,
                    st.session_state.agent_config,
                    prompt,
                )
                # If fallback was triggered, update cached agent
                if new_agent is not None:
                    st.session_state.agent = new_agent
                    st.session_state.llm_provider = new_provider
                    st.toast(f"Switched to {new_provider.title()} (fallback)")
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"⚠️ **Error:** {e}"
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
