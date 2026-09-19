"""Data loading and caching for airport datasets."""

import os
import functools
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@functools.lru_cache(maxsize=1)
def load_airports() -> pd.DataFrame:
    """Load airports.csv from OurAirports data.

    Filters to US airports with IATA codes (medium/large airports).
    """
    path = os.path.join(DATA_DIR, "airports.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"airports.csv not found at {path}. "
            "Run 'python scripts/download_data.py' to fetch data."
        )
    df = pd.read_csv(path)
    # Filter to US airports with IATA codes
    us = df[
        (df["iso_country"] == "US")
        & (df["iata_code"].notna())
        & (df["iata_code"] != "")
        & (df["type"].isin(["medium_airport", "large_airport"]))
    ].copy()
    us["iata_code"] = us["iata_code"].str.strip().str.upper()
    return us.reset_index(drop=True)


@functools.lru_cache(maxsize=1)
def load_runways() -> pd.DataFrame:
    """Load runways.csv from OurAirports data."""
    path = os.path.join(DATA_DIR, "runways.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"runways.csv not found at {path}. "
            "Run 'python scripts/download_data.py' to fetch data."
        )
    df = pd.read_csv(path)
    return df


@functools.lru_cache(maxsize=1)
def load_t100() -> pd.DataFrame:
    """Load T-100 domestic market data.

    Expected columns: ORIGIN, DEST, PASSENGERS, DEPARTURES_PERFORMED,
    DISTANCE, YEAR, MONTH, ORIGIN_STATE_ABR, etc.
    """
    path = os.path.join(DATA_DIR, "t100_data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"t100_data.csv not found at {path}. "
            "Run 'python scripts/download_data.py' to fetch data."
        )
    df = pd.read_csv(path)
    return df


@functools.lru_cache(maxsize=1)
def load_ontime() -> pd.DataFrame:
    """Load on-time performance data.

    Expected columns: ORIGIN, DEP_DELAY, ARR_DELAY, CANCELLED,
    DEP_DEL15, YEAR, MONTH, etc.
    """
    path = os.path.join(DATA_DIR, "ontime_data.csv")
    if not os.path.exists(path):
        # On-time data is optional — return empty DataFrame
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def get_airport_runways(airport_ident: str, airports_df: pd.DataFrame, runways_df: pd.DataFrame) -> pd.DataFrame:
    """Get runways for a specific airport by matching airport ident."""
    # Find the airport's internal ID
    airport_row = airports_df[airports_df["iata_code"] == airport_ident.upper()]
    if airport_row.empty:
        return pd.DataFrame()
    airport_id = airport_row.iloc[0]["id"]
    return runways_df[runways_df["airport_ref"] == airport_id].copy()


def get_airport_info(iata_code: str) -> dict | None:
    """Get basic info for an airport by IATA code."""
    airports = load_airports()
    row = airports[airports["iata_code"] == iata_code.upper()]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "iata_code": r["iata_code"],
        "name": r["name"],
        "municipality": r.get("municipality", ""),
        "region": r.get("iso_region", ""),
        "latitude": r.get("latitude_deg", None),
        "longitude": r.get("longitude_deg", None),
        "type": r.get("type", ""),
        "ident": r.get("ident", ""),
    }
