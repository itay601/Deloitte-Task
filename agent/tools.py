"""Agent tools — bridge between LLM agent and scoring engine."""

import json
from langchain_core.tools import tool
from scoring.engine import ScoringEngine
from data_layer.regions import resolve_region
from data_layer.loader import get_airport_info

# Singleton scoring engine (loaded once, reused)
_engine: ScoringEngine | None = None


def _get_engine() -> ScoringEngine:
    global _engine
    if _engine is None:
        _engine = ScoringEngine()
    return _engine


@tool
def score_airports(region: str = "", top_n: int = 10) -> str:
    """Score and rank US airports by investment need. Optionally filter by region or state name.

    Args:
        region: Optional region name (e.g., "New England", "California", "TX") or empty for all US airports.
        top_n: Number of top airports to return (default 10).

    Returns:
        JSON string with ranked airports and their composite scores.
    """
    engine = _get_engine()

    state_filter = None
    if region:
        states = resolve_region(region)
        if states is None:
            return json.dumps({
                "error": f"Region '{region}' not recognized. Try a US region name (e.g., 'New England', 'Southeast') or state name/abbreviation."
            })
        state_filter = states

    results = engine.get_top_airports(n=top_n, state_filter=state_filter)

    if not results:
        return json.dumps({
            "message": f"No scored airports found for region '{region}'.",
            "suggestion": "Try a broader region or check the region name."
        })

    return json.dumps({
        "region_filter": region if region else "All US",
        "airports_scored": len(engine.scored_airports),
        "results": results,
        "methodology": "Composite score based on 5 KPIs: congestion (25%), growth (20%), delay (20%), facility constraint (20%), demand gap (15%). Each KPI is percentile-ranked 0-100.",
    }, indent=2)


@tool
def compare_airports(iata_a: str, iata_b: str) -> str:
    """Compare two airports side-by-side on all investment KPIs.

    Args:
        iata_a: IATA code of the first airport (e.g., "LAX").
        iata_b: IATA code of the second airport (e.g., "SNA").

    Returns:
        JSON string with detailed comparison including scores, raw values, and data sources.
    """
    engine = _get_engine()
    result = engine.compare_airports(iata_a, iata_b)
    return json.dumps(result, indent=2)


@tool
def get_airport_profile(iata_code: str) -> str:
    """Get a comprehensive investment profile for a single airport.

    Args:
        iata_code: IATA code of the airport (e.g., "SFO").

    Returns:
        JSON string with full scoring breakdown, raw values, confidence level, and methodology.
    """
    engine = _get_engine()
    result = engine.score_airport(iata_code)
    return json.dumps(result, indent=2)


@tool
def get_flight_mix(iata_code: str) -> str:
    """Analyze the flight distance mix for an airport (short-haul, medium-haul, long-haul).

    Args:
        iata_code: IATA code of the airport (e.g., "ANC").

    Returns:
        JSON string with percentage breakdown of short/medium/long haul flights and top routes.
    """
    engine = _get_engine()
    result = engine.get_flight_mix(iata_code)
    return json.dumps(result, indent=2)


@tool
def analyze_demand(iata_code: str) -> str:
    """Analyze demand vs capacity for an airport to identify unmet demand.

    Args:
        iata_code: IATA code of the airport (e.g., "SFO").

    Returns:
        JSON string with throughput, estimated capacity, utilization ratio, and interpretation.
    """
    engine = _get_engine()
    iata_code = iata_code.upper()

    # Get demand gap KPI details
    raw = engine.compute_raw_kpis(iata_code)
    demand = raw["demand_gap"]
    congestion = raw["congestion"]
    growth = raw["growth"]

    # Also get the full score for context
    score = engine.score_airport(iata_code)

    result = {
        "iata_code": iata_code,
        "demand_analysis": demand,
        "congestion_context": congestion,
        "growth_context": growth,
        "composite_score": score.get("composite_score"),
        "rank": score.get("rank"),
        "total_airports": score.get("total_airports_scored"),
        "interpretation": (
            f"Airport {iata_code} has a capacity utilization of {demand.get('utilization_pct', 0)}%. "
            f"{'Demand exceeds estimated capacity.' if demand.get('raw_value', 0) > 1.0 else 'Operating within estimated capacity.'} "
            f"Passenger growth rate: {growth.get('raw_value', 0):.1%}."
        ),
    }
    return json.dumps(result, indent=2)


@tool
def search_airports_by_region(region: str) -> str:
    """Find airport IATA codes in a given US region or state.

    Args:
        region: Region name (e.g., "New England", "Pacific Northwest") or state name/abbreviation.

    Returns:
        JSON string with matching airports and their basic info.
    """
    states = resolve_region(region)
    if states is None:
        return json.dumps({
            "error": f"Region '{region}' not recognized.",
            "available_regions": [
                "New England", "Mid-Atlantic", "Southeast", "Midwest",
                "Southwest", "Rocky Mountain", "Pacific", "Pacific Northwest",
                "West Coast", "Alaska", "Hawaii", "Great Lakes", "Gulf Coast",
                "Northeast", "South", "West", "East Coast",
            ],
        })

    engine = _get_engine()
    airports_df = engine.airports_df
    matches = airports_df[
        airports_df["iso_region"].apply(
            lambda r: str(r).split("-")[1] if "-" in str(r) else ""
        ).isin(states)
    ]

    # Filter to only airports in our scored set
    matches = matches[matches["iata_code"].isin(engine.scored_airports)]

    results = []
    for _, row in matches.iterrows():
        results.append({
            "iata_code": row["iata_code"],
            "name": row["name"],
            "municipality": row.get("municipality", ""),
            "state": str(row.get("iso_region", "")).split("-")[1] if "-" in str(row.get("iso_region", "")) else "",
        })

    return json.dumps({
        "region": region,
        "states": states,
        "airport_count": len(results),
        "airports": results,
    }, indent=2)


# List of all tools for agent binding
ALL_TOOLS = [
    score_airports,
    compare_airports,
    get_airport_profile,
    get_flight_mix,
    analyze_demand,
    search_airports_by_region,
]
