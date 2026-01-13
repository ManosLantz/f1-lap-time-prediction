# config.py
"""
Configuration module for the F1 Lap Time Prediction project.

This module defines global constants, race schedules, and physics parameters used
across the data loading, feature engineering, and modeling pipelines.
"""

# ==============================================================================
# 1. SCHEDULE & RACES
# ==============================================================================
SEASONS = [2022, 2023, 2024, 2025]

# RACES: List of race names to process.
# If None, the system processes the full season schedule.
# If a list is provided, it filters execution to only these races (useful for debugging).
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

# ==============================================================================
# 2. DATA FILTERING THRESHOLDS
# ==============================================================================

# MIN_LAP_TIME_SEC: Absolute minimum lap time to be considered valid.
# Used to filter out anomalies, out-laps, or severe safety car laps.
MIN_LAP_TIME_SEC = 50

# CORNER_SPEED_THRESHOLD: Speed in km/h below which a sample is considered "cornering".
# Used for computing AEBI (Aerodynamic Efficiency Brake Index).
CORNER_SPEED_THRESHOLD = 180

# FASTF1_CACHE_DIR: Local directory to store FastF1 cache files.
FASTF1_CACHE_DIR = "data/raw"

# ==============================================================================
# 3. TRACK METADATA
# ==============================================================================

# TRACK_CORNERS: Approximate corner counts per circuit used for track complexity proxies.
# If a circuit is missing, feature engineering handles defaults.
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
