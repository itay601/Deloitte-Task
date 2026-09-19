"""Scoring configuration — weights, thresholds, constants."""

# KPI weights (must sum to 1.0)
KPI_WEIGHTS = {
    "congestion": 0.25,
    "growth": 0.20,
    "delay": 0.20,
    "facility_constraint": 0.20,
    "demand_gap": 0.15,
}

# Capacity estimation: passengers per runway per year
# Based on FAA planning guidelines, a single runway can handle ~200k-250k operations/year
# With avg ~130 pax/flight, that's ~26-32M pax/runway/year for a busy airport
# We use a conservative estimate for scoring purposes
ESTIMATED_CAPACITY_PER_RUNWAY = 25_000_000  # passengers per year per runway

# Runway quality thresholds
MIN_RUNWAY_LENGTH_FT = 5000  # Minimum for commercial jets
IDEAL_RUNWAY_LENGTH_FT = 8000  # Ideal for wide-body operations

# Surface type quality scores (higher = better)
SURFACE_QUALITY = {
    "ASP": 1.0,   # Asphalt
    "CON": 1.0,   # Concrete
    "PEM": 0.9,   # Partially concrete/asphalt
    "BIT": 0.8,   # Bituminous
    "GRS": 0.3,   # Grass
    "GVL": 0.4,   # Gravel
    "SAN": 0.2,   # Sand
    "WAT": 0.5,   # Water (seaplane)
    "": 0.5,      # Unknown
}

# Confidence levels based on data completeness
CONFIDENCE_THRESHOLDS = {
    "high": 0.8,    # >80% of expected data points available
    "medium": 0.5,  # 50-80%
    "low": 0.0,     # <50%
}
