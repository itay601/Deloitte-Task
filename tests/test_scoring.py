"""Tests for the deterministic scoring engine."""

import pytest
import pandas as pd
import numpy as np
from scoring.engine import ScoringEngine
from scoring.kpis import (
    compute_congestion,
    compute_growth,
    compute_delay,
    compute_facility_constraint,
    compute_demand_gap,
)
from scoring.config import KPI_WEIGHTS


@pytest.fixture(scope="module")
def engine():
    """Create a scoring engine instance (shared across tests in this module)."""
    return ScoringEngine()


class TestKPIWeights:
    """Test that KPI weights are valid."""

    def test_weights_sum_to_one(self):
        total = sum(KPI_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self):
        for name, weight in KPI_WEIGHTS.items():
            assert weight > 0, f"Weight for {name} is {weight}, expected > 0"

    def test_expected_kpis_present(self):
        expected = {"congestion", "growth", "delay", "facility_constraint", "demand_gap"}
        assert set(KPI_WEIGHTS.keys()) == expected


class TestKPIFunctions:
    """Test individual KPI computation functions."""

    def test_congestion_basic(self):
        t100 = pd.DataFrame({
            "ORIGIN": ["TST"] * 12,
            "YEAR": [2023] * 12,
            "MONTH": list(range(1, 13)),
            "PASSENGERS": [100_000] * 12,
        })
        result = compute_congestion("TST", t100, runway_count=2)
        assert result["raw_value"] == 600_000  # 1.2M pax / 2 runways
        assert result["runway_count"] == 2

    def test_congestion_zero_runways_handled(self):
        t100 = pd.DataFrame({
            "ORIGIN": ["TST"] * 12,
            "YEAR": [2023] * 12,
            "MONTH": list(range(1, 13)),
            "PASSENGERS": [10_000] * 12,
        })
        result = compute_congestion("TST", t100, runway_count=0)
        assert result["runway_count"] == 1  # Defaults to 1

    def test_growth_calculation(self):
        t100 = pd.DataFrame({
            "ORIGIN": ["TST"] * 24,
            "YEAR": [2022] * 12 + [2023] * 12,
            "MONTH": list(range(1, 13)) * 2,
            "PASSENGERS": [100_000] * 12 + [110_000] * 12,
        })
        result = compute_growth("TST", t100)
        expected_growth = (110_000 * 12 - 100_000 * 12) / (100_000 * 12)
        assert abs(result["raw_value"] - expected_growth) < 1e-6

    def test_growth_single_year(self):
        t100 = pd.DataFrame({
            "ORIGIN": ["TST"] * 12,
            "YEAR": [2023] * 12,
            "MONTH": list(range(1, 13)),
            "PASSENGERS": [100_000] * 12,
        })
        result = compute_growth("TST", t100)
        assert result["raw_value"] == 0.0

    def test_delay_empty_data(self):
        result = compute_delay("TST", pd.DataFrame())
        assert result["raw_value"] == 0.0
        assert result.get("data_available") is False

    def test_delay_calculation(self):
        ontime = pd.DataFrame({
            "ORIGIN": ["TST"] * 100,
            "YEAR": [2023] * 100,
            "DEP_DELAY": [15.0] * 100,
            "DEP_DEL15": [0] * 50 + [1] * 50,
            "CANCELLED": [0] * 95 + [1] * 5,
        })
        result = compute_delay("TST", ontime)
        assert result["data_available"] is True
        assert result["avg_departure_delay_min"] == 15.0
        assert result["pct_flights_delayed_15min"] == 50.0
        assert result["cancellation_rate_pct"] == 5.0

    def test_facility_constraint_empty(self):
        result = compute_facility_constraint("TST", pd.DataFrame())
        assert result["raw_value"] == 0.5
        assert result["runway_count"] == 0

    def test_facility_constraint_basic(self):
        runways = pd.DataFrame({
            "length_ft": [10000, 8000],
            "surface": ["ASP", "CON"],
        })
        result = compute_facility_constraint("TST", runways)
        assert result["runway_count"] == 2
        assert 0 <= result["raw_value"] <= 1

    def test_demand_gap_basic(self):
        t100 = pd.DataFrame({
            "ORIGIN": ["TST"] * 12,
            "YEAR": [2023] * 12,
            "MONTH": list(range(1, 13)),
            "PASSENGERS": [2_000_000] * 12,
        })
        result = compute_demand_gap("TST", t100, runway_count=1)
        # 24M pax / 25M capacity = 0.96
        assert abs(result["raw_value"] - 0.96) < 0.01


class TestScoringEngine:
    """Test the scoring engine end-to-end."""

    def test_score_all_airports_returns_dataframe(self, engine):
        df = engine.score_all_airports()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "composite_score" in df.columns
        assert "rank" in df.columns

    def test_scores_are_deterministic(self, engine):
        """Running scoring twice produces identical results."""
        df1 = engine.score_all_airports()
        df2 = engine.score_all_airports()
        pd.testing.assert_frame_equal(df1, df2)

    def test_composite_scores_in_range(self, engine):
        df = engine.score_all_airports()
        assert df["composite_score"].min() >= 0
        assert df["composite_score"].max() <= 100

    def test_ranks_are_unique(self, engine):
        df = engine.score_all_airports()
        assert df["rank"].is_unique

    def test_score_known_airport(self, engine):
        result = engine.score_airport("ATL")
        assert "error" not in result
        assert result["iata_code"] == "ATL"
        assert "composite_score" in result
        assert "scores" in result
        assert "confidence" in result

    def test_score_unknown_airport(self, engine):
        result = engine.score_airport("ZZZ")
        assert "error" in result

    def test_compare_airports(self, engine):
        result = engine.compare_airports("LAX", "SFO")
        assert "error" not in result
        assert "airport_a" in result
        assert "airport_b" in result
        assert "comparison_summary" in result

    def test_get_top_airports(self, engine):
        results = engine.get_top_airports(n=5)
        assert len(results) == 5
        assert results[0]["rank"] == 1

    def test_get_top_airports_with_state_filter(self, engine):
        results = engine.get_top_airports(n=10, state_filter=["CA"])
        assert len(results) > 0
        for r in results:
            assert r["state"] == "CA"

    def test_flight_mix(self, engine):
        result = engine.get_flight_mix("ATL")
        assert "error" not in result
        mix = result["flight_mix"]
        total = mix["short_haul_pct"] + mix["medium_haul_pct"] + mix["long_haul_pct"]
        assert abs(total - 100.0) < 1.0  # Should sum to ~100%

    def test_flight_mix_unknown_airport(self, engine):
        result = engine.get_flight_mix("ZZZ")
        assert "error" in result
