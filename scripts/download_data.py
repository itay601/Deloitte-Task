#!/usr/bin/env python3
"""Download and prepare airport datasets.

Downloads:
1. airports.csv from OurAirports
2. runways.csv from OurAirports
3. Generates synthetic T-100 and on-time data for demo purposes

BTS T-100 data requires manual download from https://www.transtats.bts.gov/
For the exam demo, we generate realistic synthetic data covering major US airports.
"""

import os
import sys
import random
import requests
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def download_file(url: str, dest: str) -> bool:
    """Download a file from URL to dest path."""
    print(f"  Downloading {url}...")
    try:
        resp = requests.get(url, timeout=60, stream=True)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ Saved to {dest}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_ourairports():
    """Download airports.csv and runways.csv from OurAirports."""
    print("\n[1/3] Downloading OurAirports data...")
    base = "https://davidmegginson.github.io/ourairports-data"

    airports_path = os.path.join(DATA_DIR, "airports.csv")
    runways_path = os.path.join(DATA_DIR, "runways.csv")

    download_file(f"{base}/airports.csv", airports_path)
    download_file(f"{base}/runways.csv", runways_path)


def generate_t100_data():
    """Generate realistic synthetic T-100-style data for demo.

    In production, this would be replaced with actual BTS data downloaded from
    https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FMF
    """
    print("\n[2/3] Generating T-100 sample data...")

    # Major US airports with realistic passenger volumes (annual, approximate)
    airports_traffic = {
        "ATL": 93_000_000, "LAX": 88_000_000, "ORD": 83_000_000,
        "DFW": 73_000_000, "DEN": 69_000_000, "JFK": 62_000_000,
        "SFO": 57_000_000, "SEA": 50_000_000, "LAS": 52_000_000,
        "MCO": 50_000_000, "EWR": 46_000_000, "CLT": 50_000_000,
        "PHX": 46_000_000, "IAH": 45_000_000, "MIA": 45_000_000,
        "BOS": 42_000_000, "MSP": 39_000_000, "FLL": 36_000_000,
        "DTW": 36_000_000, "PHL": 33_000_000, "LGA": 31_000_000,
        "BWI": 27_000_000, "SLC": 26_000_000, "SAN": 25_000_000,
        "IAD": 24_000_000, "DCA": 24_000_000, "MDW": 22_000_000,
        "TPA": 22_000_000, "PDX": 20_000_000, "HNL": 21_000_000,
        "STL": 16_000_000, "BNA": 19_000_000, "AUS": 18_000_000,
        "RDU": 14_000_000, "MCI": 12_000_000, "SNA": 11_000_000,
        "SMF": 12_000_000, "SJC": 15_000_000, "CLE": 10_000_000,
        "OAK": 13_000_000, "SAT": 10_000_000, "IND": 9_500_000,
        "PIT": 10_000_000, "CVG": 9_000_000, "CMH": 8_500_000,
        "MKE": 7_000_000, "JAX": 7_000_000, "ANC": 5_000_000,
        "ABQ": 5_500_000, "OMA": 5_000_000, "BUR": 5_000_000,
        "RNO": 4_500_000, "TUS": 3_800_000, "ELP": 3_500_000,
        "SDF": 4_000_000, "BDL": 6_500_000, "PVD": 4_000_000,
        "MHT": 2_000_000, "PWM": 2_200_000,
    }

    # State mapping for these airports
    airport_states = {
        "ATL": "GA", "LAX": "CA", "ORD": "IL", "DFW": "TX", "DEN": "CO",
        "JFK": "NY", "SFO": "CA", "SEA": "WA", "LAS": "NV", "MCO": "FL",
        "EWR": "NJ", "CLT": "NC", "PHX": "AZ", "IAH": "TX", "MIA": "FL",
        "BOS": "MA", "MSP": "MN", "FLL": "FL", "DTW": "MI", "PHL": "PA",
        "LGA": "NY", "BWI": "MD", "SLC": "UT", "SAN": "CA", "IAD": "VA",
        "DCA": "VA", "MDW": "IL", "TPA": "FL", "PDX": "OR", "HNL": "HI",
        "STL": "MO", "BNA": "TN", "AUS": "TX", "RDU": "NC", "MCI": "MO",
        "SNA": "CA", "SMF": "CA", "SJC": "CA", "CLE": "OH", "OAK": "CA",
        "SAT": "TX", "IND": "IN", "PIT": "PA", "CVG": "KY", "CMH": "OH",
        "MKE": "WI", "JAX": "FL", "ANC": "AK", "ABQ": "NM", "OMA": "NE",
        "BUR": "CA", "RNO": "NV", "TUS": "AZ", "ELP": "TX", "SDF": "KY",
        "BDL": "CT", "PVD": "RI", "MHT": "NH", "PWM": "ME",
    }

    np.random.seed(42)
    rows = []

    # Typical domestic route pairs and distances
    route_distances = {
        200: 0.3, 500: 0.25, 1000: 0.2, 1500: 0.1,
        2000: 0.08, 2500: 0.05, 3000: 0.02,
    }

    airport_codes = list(airports_traffic.keys())

    for year in [2022, 2023]:
        for origin in airport_codes:
            annual_pax = airports_traffic[origin]
            # Growth factor: 2023 is ~3-8% higher than 2022 for most airports
            if year == 2023:
                growth = np.random.uniform(0.02, 0.10)
                annual_pax = int(annual_pax * (1 + growth))

            # Distribute across ~20-40 destination airports
            n_routes = min(len(airport_codes) - 1, random.randint(15, 40))
            dests = random.sample([a for a in airport_codes if a != origin], n_routes)

            # Distribute passengers across routes (power law)
            weights = np.random.pareto(1.5, n_routes) + 1
            weights = weights / weights.sum()

            for i, dest in enumerate(dests):
                pax = int(annual_pax * weights[i])
                if pax < 100:
                    continue

                # Pick a realistic distance
                distance = random.choice(list(route_distances.keys()))
                distance += random.randint(-100, 100)
                distance = max(100, distance)

                # Flights = pax / ~150 load per flight
                departures = max(1, pax // random.randint(120, 180))

                for month in range(1, 13):
                    # Seasonal variation
                    seasonal = 1.0 + 0.15 * np.sin((month - 3) * np.pi / 6)
                    m_pax = max(1, int(pax / 12 * seasonal))
                    m_dep = max(1, int(departures / 12 * seasonal))

                    rows.append({
                        "YEAR": year,
                        "MONTH": month,
                        "ORIGIN": origin,
                        "DEST": dest,
                        "ORIGIN_STATE_ABR": airport_states.get(origin, ""),
                        "DEST_STATE_ABR": airport_states.get(dest, ""),
                        "PASSENGERS": m_pax,
                        "DEPARTURES_PERFORMED": m_dep,
                        "DISTANCE": distance,
                    })

    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "t100_data.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Generated {len(df)} route-month records for {len(airport_codes)} airports")
    print(f"  ✓ Saved to {path}")


def generate_ontime_data():
    """Generate realistic synthetic on-time performance data."""
    print("\n[3/3] Generating on-time performance sample data...")

    # Delay profiles: (mean_delay, pct_delayed, cancel_rate)
    # Higher = worse performance
    delay_profiles = {
        "ATL": (12, 0.22, 0.015), "LAX": (14, 0.24, 0.012),
        "ORD": (18, 0.30, 0.025), "DFW": (13, 0.22, 0.018),
        "DEN": (15, 0.25, 0.020), "JFK": (19, 0.32, 0.022),
        "SFO": (16, 0.27, 0.015), "SEA": (11, 0.20, 0.012),
        "LAS": (9, 0.18, 0.010), "MCO": (10, 0.19, 0.013),
        "EWR": (21, 0.35, 0.028), "CLT": (12, 0.21, 0.016),
        "PHX": (10, 0.18, 0.011), "IAH": (14, 0.24, 0.019),
        "MIA": (13, 0.22, 0.014), "BOS": (16, 0.28, 0.022),
        "MSP": (13, 0.23, 0.020), "FLL": (11, 0.20, 0.013),
        "DTW": (14, 0.24, 0.018), "PHL": (17, 0.29, 0.024),
        "LGA": (20, 0.33, 0.026), "BWI": (12, 0.21, 0.015),
        "SLC": (9, 0.16, 0.010), "SAN": (8, 0.15, 0.008),
        "IAD": (14, 0.24, 0.018), "DCA": (15, 0.26, 0.020),
        "MDW": (14, 0.25, 0.018), "TPA": (10, 0.19, 0.012),
        "PDX": (10, 0.18, 0.011), "HNL": (8, 0.14, 0.008),
        "STL": (11, 0.20, 0.014), "BNA": (12, 0.21, 0.015),
        "AUS": (13, 0.22, 0.016), "RDU": (11, 0.19, 0.013),
        "MCI": (11, 0.20, 0.015), "SNA": (7, 0.13, 0.007),
        "SMF": (9, 0.16, 0.010), "SJC": (10, 0.17, 0.009),
        "CLE": (13, 0.23, 0.018), "OAK": (10, 0.18, 0.011),
        "SAT": (12, 0.21, 0.015), "IND": (12, 0.21, 0.016),
        "PIT": (13, 0.22, 0.017), "CVG": (11, 0.20, 0.014),
        "CMH": (11, 0.19, 0.014), "MKE": (12, 0.21, 0.017),
        "JAX": (10, 0.18, 0.012), "ANC": (8, 0.14, 0.010),
        "ABQ": (9, 0.16, 0.011), "OMA": (11, 0.20, 0.016),
        "BUR": (8, 0.14, 0.008), "RNO": (9, 0.16, 0.010),
        "TUS": (8, 0.15, 0.009), "ELP": (9, 0.16, 0.012),
        "SDF": (11, 0.20, 0.015), "BDL": (14, 0.25, 0.020),
        "PVD": (13, 0.23, 0.018), "MHT": (12, 0.21, 0.016),
        "PWM": (11, 0.19, 0.015),
    }

    np.random.seed(42)
    rows = []

    for year in [2022, 2023]:
        for origin, (mean_delay, pct_del, cancel_rate) in delay_profiles.items():
            for month in range(1, 13):
                # Winter months have worse delays
                winter_factor = 1.0
                if month in [12, 1, 2]:
                    winter_factor = 1.3
                elif month in [6, 7, 8]:
                    winter_factor = 1.1  # thunderstorm season

                n_flights = random.randint(800, 5000)

                for _ in range(n_flights // 50):  # Sample ~2% of flights
                    delay = np.random.normal(mean_delay * winter_factor, 15)
                    delay = max(-10, delay)  # can be early
                    cancelled = 1 if random.random() < cancel_rate * winter_factor else 0
                    dep_del15 = 1 if delay > 15 else 0

                    rows.append({
                        "YEAR": year,
                        "MONTH": month,
                        "ORIGIN": origin,
                        "DEP_DELAY": round(delay, 1),
                        "ARR_DELAY": round(delay + np.random.normal(0, 5), 1),
                        "CANCELLED": cancelled,
                        "DEP_DEL15": dep_del15,
                    })

    df = pd.DataFrame(rows)
    path = os.path.join(DATA_DIR, "ontime_data.csv")
    df.to_csv(path, index=False)
    print(f"  ✓ Generated {len(df)} on-time records")
    print(f"  ✓ Saved to {path}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print("Airport Investment Intelligence — Data Download")
    print("=" * 60)

    download_ourairports()
    generate_t100_data()
    generate_ontime_data()

    print("\n" + "=" * 60)
    print("✓ All data prepared successfully!")
    print(f"  Data directory: {DATA_DIR}")
    print("  Files:")
    for f in sorted(os.listdir(DATA_DIR)):
        if f.endswith(".csv"):
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"    {f}: {size / 1024 / 1024:.1f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
