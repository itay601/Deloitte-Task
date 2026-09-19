"""Optional live API clients for flight data enrichment.

These are bonus features — the system works fully with static data.
If API keys are missing, functions return None gracefully.
"""

import os
import requests
from functools import lru_cache


def _get_aviationstack_key() -> str | None:
    return os.environ.get("AVIATIONSTACK_API_KEY")


def _get_aerodatabox_key() -> str | None:
    return os.environ.get("AERODATABOX_API_KEY")


@lru_cache(maxsize=128)
def get_live_flights(iata_code: str) -> dict | None:
    """Fetch current flight info from AviationStack. Returns None if unavailable."""
    key = _get_aviationstack_key()
    if not key:
        return None
    try:
        resp = requests.get(
            "http://api.aviationstack.com/v1/flights",
            params={"access_key": key, "dep_iata": iata_code, "limit": 25},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


@lru_cache(maxsize=128)
def get_delay_stats(iata_code: str) -> dict | None:
    """Fetch delay statistics from AeroDataBox. Returns None if unavailable."""
    key = _get_aerodatabox_key()
    if not key:
        return None
    try:
        resp = requests.get(
            f"https://aerodatabox.p.rapidapi.com/airports/iata/{iata_code}/stats/delays",
            headers={"X-RapidAPI-Key": key},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None
