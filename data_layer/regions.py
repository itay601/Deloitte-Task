"""Region-to-state mapping for US airport regions."""

# Maps informal region names to US state abbreviations
REGION_TO_STATES: dict[str, list[str]] = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Mid-Atlantic": ["NJ", "NY", "PA"],
    "Southeast": ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "VA", "WV"],
    "Midwest": ["IL", "IN", "IA", "KS", "MI", "MN", "MO", "NE", "ND", "OH", "SD", "WI"],
    "Southwest": ["AZ", "NM", "OK", "TX"],
    "Rocky Mountain": ["CO", "ID", "MT", "UT", "WY"],
    "Pacific": ["CA", "OR", "WA"],
    "Pacific Northwest": ["OR", "WA"],
    "West Coast": ["CA", "OR", "WA"],
    "Alaska": ["AK"],
    "Hawaii": ["HI"],
    "Great Lakes": ["IL", "IN", "MI", "MN", "OH", "WI"],
    "Gulf Coast": ["AL", "FL", "LA", "MS", "TX"],
    "Northeast": ["CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"],
    "South": ["AL", "AR", "FL", "GA", "KY", "LA", "MS", "NC", "SC", "TN", "TX", "VA", "WV"],
    "West": ["AZ", "CA", "CO", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"],
    "East Coast": ["CT", "DE", "FL", "GA", "ME", "MD", "MA", "NH", "NJ", "NY", "NC", "PA", "RI", "SC", "VA", "VT", "DC"],
}

# Normalize keys for fuzzy matching
_NORMALIZED = {k.lower().strip(): v for k, v in REGION_TO_STATES.items()}


def resolve_region(region_name: str) -> list[str] | None:
    """Resolve a region name to a list of US state abbreviations.

    Returns None if region is not recognized. Supports fuzzy matching
    by lowering case and stripping whitespace.
    """
    normalized = region_name.lower().strip()

    # Direct match
    if normalized in _NORMALIZED:
        return _NORMALIZED[normalized]

    # Check if it's a state name
    from data_layer.constants import STATE_CODES
    for code, name in STATE_CODES.items():
        if normalized == name.lower() or normalized == code.lower():
            return [code]

    # Substring match as fallback
    for key, states in _NORMALIZED.items():
        if normalized in key or key in normalized:
            return states

    return None
