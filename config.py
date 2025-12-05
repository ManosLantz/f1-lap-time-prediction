# config.py

SEASONS = [2022, 2023, 2024, 2025]

RACES = [
    "Bahrain Grand Prix",
    "Italian Grand Prix",
    "Belgian Grand Prix",
    "Spanish Grand Prix",
    "Monaco Grand Prix",
    "Azerbaijan Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
]

# minimum laps to consider (avoid outlaps, inlaps, SC chaos for now)
MIN_LAP_TIME_SEC = 50

# speed threshold for "corner" (km/h)
CORNER_SPEED_THRESHOLD = 180

# cache directory for fastf1
FASTF1_CACHE_DIR = "data/raw"

# approximate corner counts per circuit (for metadata)
TRACK_CORNERS = {
    "Bahrain Grand Prix": 15,
    "Italian Grand Prix": 11,
    "Belgian Grand Prix": 19,
    "Spanish Grand Prix": 16,
    "Monaco Grand Prix": 19,
    "Azerbaijan Grand Prix": 20,
    "British Grand Prix": 18,
    "Hungarian Grand Prix": 14,
}
