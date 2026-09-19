"""LangGraph agent setup with LLM fallback (Mistral → Gemini) and tool binding."""

import os
import logging
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agent.tools import ALL_TOOLS

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Airport Investment Intelligence Analyst, an AI agent built by Deloitte Digital to help analysts identify promising US airport investment opportunities for modernization and renovation.



## Your Capabilities
You have access to tools that analyze airport data across 5 key performance indicators (KPIs):
1. **Congestion** (25% weight) — Passengers per runway, indicating capacity strain
2. **Growth** (20% weight) — Year-over-year passenger growth rate
3. **Delay** (20% weight) — Departure delays, cancellations, and on-time performance
4. **Facility Constraint** (20% weight) — Runway count, length, and surface quality
5. **Demand Gap** (15% weight) — Throughput vs estimated capacity ratio

## How to Respond
- **Always use your tools** to retrieve data before answering data-related questions. Never guess scores or statistics.
- **Be specific**: Include numbers, percentages, and rankings in your answers.
- **Be transparent**: Always mention data sources, assumptions, and confidence levels.
- **Explain the "why"**: Don't just show scores — explain what drives them.
- **Format clearly**: Use tables or structured layouts for comparisons. Use bullet points for breakdowns.
- **Handle follow-ups**: Remember conversation context and build on previous answers.
- **Be honest about limitations**: If data is unavailable or confidence is low, say so.

## Data Sources
- Airport and runway data: OurAirports (open dataset)
- Passenger volumes and routes: BTS T-100 Domestic Market data
- Delay metrics: BTS On-Time Performance data
- All scoring is deterministic — no AI is used in computing scores, only in interpreting and presenting them.

## When Users Ask About Regions
If a user mentions a region (like "New England" or "the Southeast"), use the search_airports_by_region tool first to find relevant airports, then use score_airports with the region filter to rank them.

## When Users Ask About Specific Airports
Use get_airport_profile for detailed single-airport analysis, or compare_airports for side-by-side comparisons. For flight mix questions, use get_flight_mix.

## Example Interactions
- "Which airports in New England need the most investment?" → Use score_airports with region="New England"
- "Compare LAX and SNA" → Use compare_airports
- "What percentage of flights from ANC are long-haul?" → Use get_flight_mix
- "Is SFO over capacity?" → Use analyze_demand
"""


def _create_mistral_llm():
    """Create a Mistral LLM instance. Returns None if key is missing."""
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return None
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=api_key,
        temperature=0.1,
        max_tokens=4096,
    )


def _create_gemini_llm():
    """Create a Gemini LLM instance. Returns None if key is missing."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.1,
        max_output_tokens=4096,
    )


def _select_llm():
    """Select the best available LLM with Mistral → Gemini fallback.

    Returns:
        Tuple of (llm_instance, provider_name).

    Raises:
        ValueError: If no API keys are configured.
    """
    # Try Mistral first
    llm = _create_mistral_llm()
    if llm is not None:
        logger.info("Using Mistral LLM (primary)")
        return llm, "mistral"

    # Fallback to Gemini
    llm = _create_gemini_llm()
    if llm is not None:
        logger.info("Using Gemini LLM (fallback)")
        return llm, "gemini"

    raise ValueError(
        "No LLM API key configured. "
        "Set MISTRAL_API_KEY or GEMINI_API_KEY in your .env file."
    )


def create_agent(thread_id: str = "default"):
    """Create a LangGraph ReAct agent with LLM fallback and airport tools.

    Tries Mistral first, falls back to Gemini if unavailable.

    Args:
        thread_id: Conversation thread ID for memory isolation.

    Returns:
        Tuple of (agent, config) where config includes the thread_id
        and the selected provider name.
    """
    llm, provider = _select_llm()

    memory = MemorySaver()

    agent = create_react_agent(
        model=llm,
        tools=ALL_TOOLS,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )

    config = {"configurable": {"thread_id": thread_id}}

    return agent, config, provider


def _extract_response(response) -> str:
    """Extract the last AI text message from agent response."""
    messages = response.get("messages", [])
    for msg in reversed(messages):
        if not (hasattr(msg, "content") and msg.type == "ai" and msg.content):
            continue
        content = msg.content
        # Gemini returns content as a list of dicts: [{'type': 'text', 'text': '...'}]
        if isinstance(content, list):
            parts = [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ]
            text = "\n".join(p for p in parts if p.strip())
            if text.strip():
                return text
        elif isinstance(content, str) and content.strip():
            return content
    return ""


def invoke_agent(agent, config: dict, user_message: str) -> tuple[str, object | None, str | None]:
    """Send a message to the agent and get a response.

    If the primary LLM fails (e.g. 429 rate limit), automatically rebuilds
    the agent with the fallback LLM and retries.

    Args:
        agent: The LangGraph agent.
        config: Config dict with thread_id.
        user_message: The user's question.

    Returns:
        Tuple of (response_text, new_agent_or_None, new_provider_or_None).
        If fallback was used, new_agent and new_provider are set so the
        caller can update its cached agent.
    """
    try:
        response = agent.invoke(
            {"messages": [("human", user_message)]},
            config=config,
        )
        text = _extract_response(response)
        if text:
            return text, None, None
    except Exception as e:
        logger.warning("Primary LLM failed: %s. Attempting fallback...", e)

        # Try to build a fallback agent with Gemini
        fallback_llm = _create_gemini_llm()
        if fallback_llm is None:
            raise  # No fallback available, re-raise original error

        logger.info("Rebuilding agent with Gemini fallback")
        memory = MemorySaver()
        fallback_agent = create_react_agent(
            model=fallback_llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=memory,
        )

        response = fallback_agent.invoke(
            {"messages": [("human", user_message)]},
            config=config,
        )
        text = _extract_response(response)
        if text:
            return text, fallback_agent, "gemini"

    return (
        "I apologize, but I wasn't able to generate a response. Please try rephrasing your question.",
        None,
        None,
    )
