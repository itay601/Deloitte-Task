"""Constants for airport data layer."""

# OurAirports download URLs
OURAIRPORTS_BASE = "https://davidmegginson.github.io/ourairports-data"
AIRPORTS_CSV_URL = f"{OURAIRPORTS_BASE}/airports.csv"
RUNWAYS_CSV_URL = f"{OURAIRPORTS_BASE}/runways.csv"

# BTS T-100 data is pre-downloaded as it requires form submission.
# We ship a sample in the repo and the download script can fetch more.

# US state FIPS codes to names (for region mapping)
STATE_CODES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico", "VI": "Virgin Islands", "GU": "Guam",
}

# Major US airport IATA codes (for validation / quick lookup)
# This is a convenience set — the full list comes from airports.csv
MAJOR_US_AIRPORTS = {
    "ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA", "LAS", "MCO",
    "EWR", "CLT", "PHX", "IAH", "MIA", "BOS", "MSP", "FLL", "DTW", "PHL",
    "LGA", "BWI", "SLC", "SAN", "IAD", "DCA", "MDW", "TPA", "PDX", "HNL",
    "STL", "BNA", "AUS", "RDU", "MCI", "SNA", "SMF", "SJC", "CLE", "OAK",
    "SAT", "IND", "PIT", "CVG", "CMH", "MKE", "JAX", "ANC", "ABQ", "OMA",
    "BUR", "RNO", "TUS", "ELP", "SDF", "BDL", "PVD", "MHT", "PWM",
}
