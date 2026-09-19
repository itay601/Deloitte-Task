"""Individual KPI computation functions.

Each function computes a raw metric value for an airport.
The ScoringEngine handles percentile ranking and normalization.
"""

import pandas as pd
import numpy as np
from scoring.config import (
    ESTIMATED_CAPACITY_PER_RUNWAY,
    MIN_RUNWAY_LENGTH_FT,
    IDEAL_RUNWAY_LENGTH_FT,
    SURFACE_QUALITY,
)


def compute_congestion(
    iata_code: str,
    t100_df: pd.DataFrame,
    runway_count: int,
) -> dict:
    """Compute congestion metric: passengers per runway.

    Higher value = more congested = higher investment need.
    """
    # Total passengers (most recent year)
    latest_year = t100_df["YEAR"].max()
    airport_data = t100_df[
        (t100_df["ORIGIN"] == iata_code) & (t100_df["YEAR"] == latest_year)
    ]
    total_pax = airport_data["PASSENGERS"].sum()

    if runway_count == 0:
        runway_count = 1  # avoid division by zero

    pax_per_runway = total_pax / runway_count

    return {
        "raw_value": pax_per_runway,
        "total_passengers": int(total_pax),
        "runway_count": runway_count,
        "year": int(latest_year),
        "unit": "passengers/runway/year",
        "interpretation": "Higher values indicate more congestion per runway",
    }


def compute_growth(
    iata_code: str,
    t100_df: pd.DataFrame,
) -> dict:
    """Compute year-over-year passenger growth rate.

    Higher growth = more urgent investment need.
    """
    years = sorted(t100_df["YEAR"].unique())
    if len(years) < 2:
        return {
            "raw_value": 0.0,
            "note": "Insufficient years of data for growth calculation",
        }

    year_old, year_new = years[-2], years[-1]

    pax_old = t100_df[
        (t100_df["ORIGIN"] == iata_code) & (t100_df["YEAR"] == year_old)
    ]["PASSENGERS"].sum()

    pax_new = t100_df[
        (t100_df["ORIGIN"] == iata_code) & (t100_df["YEAR"] == year_new)
    ]["PASSENGERS"].sum()

    if pax_old == 0:
        growth_rate = 0.0
    else:
        growth_rate = (pax_new - pax_old) / pax_old

    return {
        "raw_value": growth_rate,
        "passengers_old": int(pax_old),
        "passengers_new": int(pax_new),
        "year_old": int(year_old),
        "year_new": int(year_new),
        "unit": "ratio (e.g., 0.05 = 5% growth)",
        "interpretation": "Higher growth rates indicate increasing demand",
    }


def compute_delay(
    iata_code: str,
    ontime_df: pd.DataFrame,
) -> dict:
    """Compute delay score from on-time performance data.

    Combines average departure delay, % flights delayed >15 min, and cancellation rate.
    Higher = worse delays = more investment need.
    """
    if ontime_df.empty:
        return {
            "raw_value": 0.0,
            "note": "No on-time performance data available",
            "data_available": False,
        }

    latest_year = ontime_df["YEAR"].max()
    airport_data = ontime_df[
        (ontime_df["ORIGIN"] == iata_code) & (ontime_df["YEAR"] == latest_year)
    ]

    if airport_data.empty:
        return {
            "raw_value": 0.0,
            "note": f"No delay data for {iata_code}",
            "data_available": False,
        }

    avg_delay = airport_data["DEP_DELAY"].mean()
    pct_delayed = airport_data["DEP_DEL15"].mean()
    cancel_rate = airport_data["CANCELLED"].mean()

    # Combined delay metric (weighted)
    # Normalize each component to roughly 0-1 range then combine
    # avg_delay: typical range 5-25 min → divide by 30
    # pct_delayed: already 0-1
    # cancel_rate: typically 0-0.05 → multiply by 10
    delay_score = (
        0.4 * min(avg_delay / 30.0, 1.0)
        + 0.4 * pct_delayed
        + 0.2 * min(cancel_rate * 10, 1.0)
    )

    return {
        "raw_value": delay_score,
        "avg_departure_delay_min": round(avg_delay, 1),
        "pct_flights_delayed_15min": round(pct_delayed * 100, 1),
        "cancellation_rate_pct": round(cancel_rate * 100, 2),
        "year": int(latest_year),
        "data_available": True,
        "interpretation": "Higher scores indicate worse delay performance",
    }


def compute_facility_constraint(
    iata_code: str,
    runways: pd.DataFrame,
) -> dict:
    """Compute facility constraint score based on runway infrastructure.

    Fewer runways, shorter lengths, and poorer surface types = more constrained.
    Higher score = more constrained = more investment need.
    """
    if runways.empty:
        return {
            "raw_value": 0.5,
            "note": f"No runway data for {iata_code}",
            "runway_count": 0,
        }

    runway_count = len(runways)

    # Runway length score (avg length vs ideal)
    lengths = runways["length_ft"].dropna()
    if lengths.empty:
        avg_length = MIN_RUNWAY_LENGTH_FT
    else:
        avg_length = lengths.mean()

    length_score = 1.0 - min(avg_length / IDEAL_RUNWAY_LENGTH_FT, 1.0)

    # Surface quality score
    surfaces = runways["surface"].fillna("").str.upper().str[:3]
    surface_scores = surfaces.map(
        lambda s: SURFACE_QUALITY.get(s, 0.5)
    )
    avg_surface = 1.0 - surface_scores.mean()  # Invert: poor surface = high constraint

    # Runway count score (fewer = more constrained)
    # 1 runway = 1.0, 2 = 0.5, 3 = 0.33, 4+ = 0.25
    count_score = 1.0 / runway_count

    # Combined constraint score
    facility_score = 0.4 * count_score + 0.35 * length_score + 0.25 * avg_surface

    return {
        "raw_value": facility_score,
        "runway_count": runway_count,
        "avg_runway_length_ft": round(avg_length, 0),
        "surface_types": list(surfaces.unique()),
        "interpretation": "Higher scores indicate more infrastructure constraints",
    }


def compute_demand_gap(
    iata_code: str,
    t100_df: pd.DataFrame,
    runway_count: int,
) -> dict:
    """Compute demand gap: actual throughput vs estimated capacity.

    Ratio > 1.0 means demand exceeds estimated capacity.
    Higher ratio = bigger gap = more investment need.
    """
    latest_year = t100_df["YEAR"].max()
    airport_data = t100_df[
        (t100_df["ORIGIN"] == iata_code) & (t100_df["YEAR"] == latest_year)
    ]
    total_pax = airport_data["PASSENGERS"].sum()

    if runway_count == 0:
        runway_count = 1

    estimated_capacity = runway_count * ESTIMATED_CAPACITY_PER_RUNWAY
    utilization_ratio = total_pax / estimated_capacity if estimated_capacity > 0 else 0

    return {
        "raw_value": utilization_ratio,
        "total_passengers": int(total_pax),
        "estimated_capacity": int(estimated_capacity),
        "runway_count": runway_count,
        "utilization_pct": round(utilization_ratio * 100, 1),
        "year": int(latest_year),
        "unit": "ratio (throughput / estimated capacity)",
        "interpretation": "Values > 1.0 suggest demand exceeds capacity; higher = bigger gap",
        "assumption": f"Estimated capacity = {ESTIMATED_CAPACITY_PER_RUNWAY:,} pax/runway/year (FAA planning guideline proxy)",
    }
