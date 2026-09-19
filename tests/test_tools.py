"""Tests for agent tools (without LLM dependency)."""

import json
import pytest
from agent.tools import (
    score_airports,
    compare_airports,
    get_airport_profile,
    get_flight_mix,
    analyze_demand,
    search_airports_by_region,
)


class TestScoreAirports:
    def test_all_us(self):
        result = json.loads(score_airports.invoke({"region": "", "top_n": 5}))
        assert "results" in result
        assert len(result["results"]) == 5

    def test_with_region(self):
        result = json.loads(score_airports.invoke({"region": "New England", "top_n": 10}))
        assert "results" in result
        assert len(result["results"]) > 0

    def test_invalid_region(self):
        result = json.loads(score_airports.invoke({"region": "Narnia", "top_n": 5}))
        assert "error" in result


class TestCompareAirports:
    def test_valid_comparison(self):
        result = json.loads(compare_airports.invoke({"iata_a": "LAX", "iata_b": "SFO"}))
        assert "airport_a" in result
        assert "airport_b" in result

    def test_invalid_airport(self):
        result = json.loads(compare_airports.invoke({"iata_a": "LAX", "iata_b": "ZZZ"}))
        assert "error" in result


class TestGetAirportProfile:
    def test_valid_airport(self):
        result = json.loads(get_airport_profile.invoke({"iata_code": "ATL"}))
        assert result["iata_code"] == "ATL"
        assert "scores" in result

    def test_invalid_airport(self):
        result = json.loads(get_airport_profile.invoke({"iata_code": "ZZZ"}))
        assert "error" in result


class TestGetFlightMix:
    def test_valid_airport(self):
        result = json.loads(get_flight_mix.invoke({"iata_code": "ANC"}))
        assert "flight_mix" in result
        assert "long_haul_pct" in result["flight_mix"]

    def test_invalid_airport(self):
        result = json.loads(get_flight_mix.invoke({"iata_code": "ZZZ"}))
        assert "error" in result


class TestAnalyzeDemand:
    def test_valid_airport(self):
        result = json.loads(analyze_demand.invoke({"iata_code": "SFO"}))
        assert "demand_analysis" in result
        assert "utilization_pct" in result["demand_analysis"]


class TestSearchAirportsByRegion:
    def test_new_england(self):
        result = json.loads(search_airports_by_region.invoke({"region": "New England"}))
        assert result["airport_count"] > 0
        states = {a["state"] for a in result["airports"]}
        assert states.issubset({"CT", "ME", "MA", "NH", "RI", "VT"})

    def test_invalid_region(self):
        result = json.loads(search_airports_by_region.invoke({"region": "Atlantis"}))
        assert "error" in result
        assert "available_regions" in result
