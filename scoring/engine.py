"""Scoring engine — orchestrates KPI computation and ranking.

All scoring is deterministic (no LLM involvement).
"""

import pandas as pd
import numpy as np
from scoring.config import KPI_WEIGHTS, CONFIDENCE_THRESHOLDS
from scoring.kpis import (
    compute_congestion,
    compute_growth,
    compute_delay,
    compute_facility_constraint,
    compute_demand_gap,
)
from data_layer.loader import (
    load_airports,
    load_runways,
    load_t100,
    load_ontime,
    get_airport_runways,
)


class ScoringEngine:
    """Deterministic airport investment scoring engine.

    Computes 5 KPIs for each airport, applies percentile ranking,
    and produces a weighted composite score.
    """

    def __init__(self):
        self.airports_df = load_airports()
        self.runways_df = load_runways()
        self.t100_df = load_t100()
        self.ontime_df = load_ontime()

        # Build set of airports that appear in T-100 data
        self.scored_airports = set(self.t100_df["ORIGIN"].unique()) & set(
            self.airports_df["iata_code"].unique()
        )

    def _get_runway_count(self, iata_code: str) -> int:
        """Get number of runways for an airport."""
        rwys = get_airport_runways(iata_code, self.airports_df, self.runways_df)
        return len(rwys) if not rwys.empty else 1

    def _get_runways(self, iata_code: str) -> pd.DataFrame:
        """Get runway data for an airport."""
        return get_airport_runways(iata_code, self.airports_df, self.runways_df)

    def compute_raw_kpis(self, iata_code: str) -> dict:
        """Compute all raw KPI values for a single airport."""
        iata_code = iata_code.upper()
        runway_count = self._get_runway_count(iata_code)
        runways = self._get_runways(iata_code)

        return {
            "congestion": compute_congestion(iata_code, self.t100_df, runway_count),
            "growth": compute_growth(iata_code, self.t100_df),
            "delay": compute_delay(iata_code, self.ontime_df),
            "facility_constraint": compute_facility_constraint(iata_code, runways),
            "demand_gap": compute_demand_gap(iata_code, self.t100_df, runway_count),
        }

    def score_all_airports(self) -> pd.DataFrame:
        """Score all airports and return ranked DataFrame.

        Returns DataFrame with columns: iata_code, name, state,
        composite_score, and individual KPI scores (0-100).
        """
        records = []
        for iata_code in self.scored_airports:
            raw = self.compute_raw_kpis(iata_code)
            airport_info = self.airports_df[
                self.airports_df["iata_code"] == iata_code
            ]
            if airport_info.empty:
                continue

            info = airport_info.iloc[0]
            region = info.get("iso_region", "")
            state = region.split("-")[1] if "-" in str(region) else ""

            records.append({
                "iata_code": iata_code,
                "name": info.get("name", ""),
                "municipality": info.get("municipality", ""),
                "state": state,
                "raw_congestion": raw["congestion"]["raw_value"],
                "raw_growth": raw["growth"]["raw_value"],
                "raw_delay": raw["delay"]["raw_value"],
                "raw_facility": raw["facility_constraint"]["raw_value"],
                "raw_demand_gap": raw["demand_gap"]["raw_value"],
            })

        df = pd.DataFrame(records)
        if df.empty:
            return df

        # Percentile rank each KPI (0-100 scale)
        for kpi in ["congestion", "growth", "delay", "facility", "demand_gap"]:
            col = f"raw_{kpi}"
            df[f"score_{kpi}"] = df[col].rank(pct=True) * 100

        # Composite score (weighted sum of percentile scores)
        df["composite_score"] = (
            KPI_WEIGHTS["congestion"] * df["score_congestion"]
            + KPI_WEIGHTS["growth"] * df["score_growth"]
            + KPI_WEIGHTS["delay"] * df["score_delay"]
            + KPI_WEIGHTS["facility_constraint"] * df["score_facility"]
            + KPI_WEIGHTS["demand_gap"] * df["score_demand_gap"]
        )

        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def get_top_airports(
        self, n: int = 10, state_filter: list[str] | None = None
    ) -> list[dict]:
        """Get top N airports by composite score, optionally filtered by state."""
        df = self.score_all_airports()
        if state_filter:
            state_filter = [s.upper() for s in state_filter]
            df = df[df["state"].isin(state_filter)]
            # Re-rank within filter
            df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
            df["rank"] = range(1, len(df) + 1)

        top = df.head(n)
        results = []
        for _, row in top.iterrows():
            results.append({
                "rank": int(row["rank"]),
                "iata_code": row["iata_code"],
                "name": row["name"],
                "municipality": row["municipality"],
                "state": row["state"],
                "composite_score": round(row["composite_score"], 1),
                "scores": {
                    "congestion": round(row["score_congestion"], 1),
                    "growth": round(row["score_growth"], 1),
                    "delay": round(row["score_delay"], 1),
                    "facility_constraint": round(row["score_facility"], 1),
                    "demand_gap": round(row["score_demand_gap"], 1),
                },
            })
        return results

    def score_airport(self, iata_code: str) -> dict:
        """Get full scoring breakdown for a single airport."""
        iata_code = iata_code.upper()
        if iata_code not in self.scored_airports:
            return {"error": f"Airport {iata_code} not found in scored dataset."}

        df = self.score_all_airports()
        row = df[df["iata_code"] == iata_code]
        if row.empty:
            return {"error": f"Airport {iata_code} could not be scored."}

        row = row.iloc[0]
        raw = self.compute_raw_kpis(iata_code)

        # Determine confidence level
        data_points = sum([
            1 if raw["congestion"]["raw_value"] > 0 else 0,
            1 if raw["growth"]["raw_value"] != 0 else 0,
            1 if raw["delay"].get("data_available", False) else 0,
            1,  # facility always has data
            1 if raw["demand_gap"]["raw_value"] > 0 else 0,
        ])
        completeness = data_points / 5.0
        if completeness >= CONFIDENCE_THRESHOLDS["high"]:
            confidence = "high"
        elif completeness >= CONFIDENCE_THRESHOLDS["medium"]:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "iata_code": iata_code,
            "name": row["name"],
            "municipality": row["municipality"],
            "state": row["state"],
            "rank": int(row["rank"]),
            "total_airports_scored": len(df),
            "composite_score": round(row["composite_score"], 1),
            "confidence": confidence,
            "scores": {
                "congestion": {
                    "percentile_score": round(row["score_congestion"], 1),
                    "weight": KPI_WEIGHTS["congestion"],
                    **raw["congestion"],
                },
                "growth": {
                    "percentile_score": round(row["score_growth"], 1),
                    "weight": KPI_WEIGHTS["growth"],
                    **raw["growth"],
                },
                "delay": {
                    "percentile_score": round(row["score_delay"], 1),
                    "weight": KPI_WEIGHTS["delay"],
                    **raw["delay"],
                },
                "facility_constraint": {
                    "percentile_score": round(row["score_facility"], 1),
                    "weight": KPI_WEIGHTS["facility_constraint"],
                    **raw["facility_constraint"],
                },
                "demand_gap": {
                    "percentile_score": round(row["score_demand_gap"], 1),
                    "weight": KPI_WEIGHTS["demand_gap"],
                    **raw["demand_gap"],
                },
            },
            "data_sources": [
                "OurAirports (airport and runway data)",
                "BTS T-100 Domestic Market (passenger volumes)",
                "BTS On-Time Performance (delay metrics)",
            ],
            "methodology": (
                "Each KPI is computed from raw data, then percentile-ranked across all "
                "scored US commercial airports. The composite score is a weighted sum "
                "of percentile scores. Higher scores indicate greater investment need."
            ),
        }

    def compare_airports(self, iata_a: str, iata_b: str) -> dict:
        """Side-by-side comparison of two airports."""
        score_a = self.score_airport(iata_a)
        score_b = self.score_airport(iata_b)

        if "error" in score_a or "error" in score_b:
            return {
                "error": f"Could not compare: {score_a.get('error', '')} {score_b.get('error', '')}".strip()
            }

        return {
            "airport_a": score_a,
            "airport_b": score_b,
            "comparison_summary": {
                "higher_investment_need": (
                    iata_a.upper()
                    if score_a["composite_score"] > score_b["composite_score"]
                    else iata_b.upper()
                ),
                "score_difference": round(
                    abs(score_a["composite_score"] - score_b["composite_score"]), 1
                ),
            },
        }

    def get_flight_mix(self, iata_code: str) -> dict:
        """Analyze flight distance mix (short/medium/long haul) for an airport."""
        iata_code = iata_code.upper()
        latest_year = self.t100_df["YEAR"].max()
        airport_data = self.t100_df[
            (self.t100_df["ORIGIN"] == iata_code) & (self.t100_df["YEAR"] == latest_year)
        ]

        if airport_data.empty:
            return {"error": f"No flight data for {iata_code}"}

        # Aggregate by route
        routes = airport_data.groupby("DEST").agg({
            "PASSENGERS": "sum",
            "DISTANCE": "first",
            "DEPARTURES_PERFORMED": "sum",
        }).reset_index()

        total_pax = routes["PASSENGERS"].sum()

        # Categorize by distance
        short_haul = routes[routes["DISTANCE"] <= 500]["PASSENGERS"].sum()
        medium_haul = routes[
            (routes["DISTANCE"] > 500) & (routes["DISTANCE"] <= 1500)
        ]["PASSENGERS"].sum()
        long_haul = routes[routes["DISTANCE"] > 1500]["PASSENGERS"].sum()

        return {
            "iata_code": iata_code,
            "year": int(latest_year),
            "total_passengers": int(total_pax),
            "total_routes": len(routes),
            "flight_mix": {
                "short_haul_pct": round(short_haul / total_pax * 100, 1) if total_pax > 0 else 0,
                "medium_haul_pct": round(medium_haul / total_pax * 100, 1) if total_pax > 0 else 0,
                "long_haul_pct": round(long_haul / total_pax * 100, 1) if total_pax > 0 else 0,
            },
            "definitions": {
                "short_haul": "≤ 500 miles",
                "medium_haul": "501–1500 miles",
                "long_haul": "> 1500 miles",
            },
            "top_routes": [
                {
                    "destination": row["DEST"],
                    "passengers": int(row["PASSENGERS"]),
                    "distance_miles": int(row["DISTANCE"]),
                    "category": (
                        "short_haul" if row["DISTANCE"] <= 500
                        else "medium_haul" if row["DISTANCE"] <= 1500
                        else "long_haul"
                    ),
                }
                for _, row in routes.nlargest(10, "PASSENGERS").iterrows()
            ],
        }
