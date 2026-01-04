# config.py

SEASONS = [2022, 2023, 2024, 2025]

# If None => use full season schedule (ALL races)
# If list => only these races (fast debug mode)
RACES = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Canadian Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Singapore Grand Prix",
    "Japanese Grand Prix",
    "Qatar Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Abu Dhabi Grand Prix",
]
# Example debug subset:
# RACES = ["Bahrain Grand Prix", "Monaco Grand Prix", "Italian Grand Prix"]

# minimum lap time to consider (avoid outlaps/inlaps/SC chaos)
MIN_LAP_TIME_SEC = 50

# speed threshold for "corner" (km/h) for AEBI
CORNER_SPEED_THRESHOLD = 180

# cache directory for fastf1
FASTF1_CACHE_DIR = "data/raw"

# Optional approximate corner counts per circuit
# Missing circuits will default to 0 (or you can set -1)
TRACK_CORNERS = {
    "Bahrain Grand Prix": 15,
    "Saudi Arabian Grand Prix": 27,
    "Australian Grand Prix": 14,
    "Azerbaijan Grand Prix": 20,
    "Miami Grand Prix": 19,
    "Monaco Grand Prix": 19,
    "Spanish Grand Prix": 16,
    "Canadian Grand Prix": 14,
    "Austrian Grand Prix": 10,
    "British Grand Prix": 18,
    "Hungarian Grand Prix": 14,
    "Belgian Grand Prix": 19,
    "Dutch Grand Prix": 14,
    "Italian Grand Prix": 11,
    "Singapore Grand Prix": 19,
    "Japanese Grand Prix": 18,
    "Qatar Grand Prix": 16,
    "United States Grand Prix": 20,
    "Mexico City Grand Prix": 17,
    "São Paulo Grand Prix": 15,
    "Las Vegas Grand Prix": 17,
    "Abu Dhabi Grand Prix": 16,
}
