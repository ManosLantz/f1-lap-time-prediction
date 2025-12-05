# f1_data.py

import os
import fastf1
import pandas as pd

from config import (
    SEASONS,
    RACES,
    FASTF1_CACHE_DIR,
    MIN_LAP_TIME_SEC,
    TRACK_CORNERS,
)
from features import compute_aebi_for_session, build_features_from_laps
from utils import save_df


def init_fastf1():
    os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
    print("Using FastF1 cache at:", FASTF1_CACHE_DIR)
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)


def load_session(season: int, race_name: str):
    session = fastf1.get_session(season, race_name, "R")  # Race session
    session.load(laps=True, telemetry=True, weather=True)
    return session


def get_weather_df(session):
    """
    Try to get weather data in a version-agnostic way.
    Returns a DataFrame with 'Time', 'AirTemp', 'TrackTemp' if possible.
    """
    # Newer versions
    if hasattr(session, "weather_data"):
        w = session.weather_data
    elif hasattr(session, "get_weather_data"):
        try:
            w = session.get_weather_data()
        except Exception:
            w = None
    elif hasattr(session, "_weather_data"):
        w = session._weather_data
    else:
        w = None

    if w is None:
        print("  No weather data available.")
        return None

    # Ensure DataFrame
    if not isinstance(w, pd.DataFrame):
        try:
            w = pd.DataFrame(w)
        except Exception:
            print("  Weather data is not a usable DataFrame.")
            return None

    # Make sure 'Time' is a column
    if "Time" not in w.columns:
        try:
            w = w.reset_index()
            if "Time" not in w.columns:
                # assume index is time-like
                w = w.rename(columns={w.columns[0]: "Time"})
        except Exception:
            print("  Failed to reset index for weather data.")
            return None

    # Keep only the relevant columns if they exist
    cols = ["Time"]
    if "AirTemp" in w.columns:
        cols.append("AirTemp")
    if "TrackTemp" in w.columns:
        cols.append("TrackTemp")

    w = w[cols].copy()
    return w

def parse_track_length_km(session):
    """
    Parse track length from session metadata.
    Try multiple locations depending on FastF1 version.
    Returns float in km or 0.0 if not available.
    """

    # Newer FastF1 versions
    if hasattr(session, "info"):
        info = session.info
    else:
        # Fallback: try private metadata or event
        info = {}
        if hasattr(session, "_session_info") and isinstance(session._session_info, dict):
            info = session._session_info
        elif hasattr(session, "event") and hasattr(session.event, "get"):
            info = session.event
        else:
            # No usable metadata, just return 0
            return 0.0

    length = info.get("CourseLength", None)

    if length is None:
        return 0.0

    # Sometimes it's already numeric
    if isinstance(length, (int, float)):
        return float(length)

    # Sometimes it's a string like "5.412 km"
    if isinstance(length, str):
        try:
            return float(length.split()[0])
        except Exception:
            return 0.0

    return 0.0


def build_dataset():
    init_fastf1()

    all_rows = []

    for season in SEASONS:
        for race_name in RACES:
            print(f"Processing {season} - {race_name}")
            session = load_session(season, race_name)

            # Reference driver for AEBI/avg speed: fastest lap of the race
            fastest_lap = session.laps.pick_fastest()
            ref_driver = fastest_lap["Driver"]
            print(f"  Reference driver for AEBI: {ref_driver}")

            telemetry = fastest_lap.get_telemetry()
            aebi = compute_aebi_for_session(telemetry)
            print(f"  AEBI = {aebi:.4f}")

            # Track metadata
            track_length_km = parse_track_length_km(session)
            track_corners = TRACK_CORNERS.get(race_name, 0)
            track_avg_speed_kph = float(telemetry["Speed"].mean()) if telemetry is not None and not telemetry.empty else 0.0

            # Get laps
            laps = session.laps.copy()

            # Remove pit in / pit out laps if columns exist
            for col in ["PitOutTime", "PitInTime"]:
                if col in laps.columns:
                    laps = laps[laps[col].isna()]

            # Proper weather interpolation: merge nearest weather reading based on Time
            weather = get_weather_df(session)
            if weather is not None and "Time" in weather.columns:
                try:
                    laps = laps.sort_values("Time")
                    weather = weather.sort_values("Time")

                    laps = pd.merge_asof(
                        laps,
                        weather,
                        on="Time",
                        direction="nearest",
                        suffixes=("", "_weather"),
                    )
                except Exception as e:
                    print("  Weather merge_asof failed:", e)
            else:
                print("  Skipping weather merge for this session.")

            # Filter out very slow laps
            laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
            laps = laps[laps["LapTimeSec"] > MIN_LAP_TIME_SEC]

            # Build per-lap features (PrevLapTimeSec, etc.)
            df_features = build_features_from_laps(laps, aebi).copy()
            # === Detect driver mistakes via sector anomalies ===

            # Rolling mean & std per sector (per driver)
            for sec in ["Sector1Sec", "Sector2Sec", "Sector3Sec"]:
                if sec in df_features.columns:
                    df_features[f"{sec}_roll_mean"] = (
                        df_features.groupby("Driver")[sec].rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
                    )
                    df_features[f"{sec}_roll_std"] = (
                        df_features.groupby("Driver")[sec].rolling(5, min_periods=3).std().reset_index(level=0, drop=True)
                    )

                    # Standardized Z-score for anomaly detection
                    df_features[f"{sec}_z"] = (
                        (df_features[sec] - df_features[f"{sec}_roll_mean"])
                        / df_features[f"{sec}_roll_std"].replace(0, 1e-6)
                    )

            # Mistake if ANY sector z-score > threshold
            Z_THRESHOLD = 2.5  # aggressive flagging
            df_features["IsMistakeLap"] = (
                (df_features.get("Sector1Sec_z", 0) > Z_THRESHOLD)
                | (df_features.get("Sector2Sec_z", 0) > Z_THRESHOLD)
                | (df_features.get("Sector3Sec_z", 0) > Z_THRESHOLD)
            ).astype(int)

            # Optional: remove mistake laps directly here
            df_features = df_features[df_features["IsMistakeLap"] == 0]

                        # Add race-level metadata
            df_features.loc[:, "Season"] = season
            df_features.loc[:, "RaceName"] = race_name
            df_features.loc[:, "TrackLengthKm"] = track_length_km
            df_features.loc[:, "TrackCorners"] = track_corners
            df_features.loc[:, "TrackAvgSpeedKph"] = track_avg_speed_kph

            all_rows.append(df_features)

        if not all_rows:
            print("No data collected.")
            return

        # Concatenate ALL per-race dataframes
        full_df = pd.concat(all_rows, ignore_index=True)

        # ----- Tyre compound change detection -----
        # Make sure rows are ordered
        full_df = full_df.sort_values(["Season", "RaceName", "Driver", "LapNumber"])

        if "Compound" in full_df.columns:
            full_df["CompoundChange"] = (
                full_df.groupby(["Season", "RaceName", "Driver"])["Compound"]
                .apply(lambda s: s.ne(s.shift(1)).astype(int))
                .reset_index(level=[0, 1, 2], drop=True)
            )
            # First lap per driver has no "change" → set to 0
            full_df["CompoundChange"] = full_df["CompoundChange"].fillna(0).astype(int)
        else:
            full_df["CompoundChange"] = 0

        # ----- Track evolution features -----
        # RaceLapNorm: LapNumber normalized by max lap per race
        full_df["RaceLapNorm"] = full_df["LapNumber"] / full_df.groupby(
            ["Season", "RaceName"]
        )["LapNumber"].transform("max")

        # RaceTimeNorm: Time (seconds) normalized within race
        full_df["RaceTimeSec"] = pd.to_timedelta(full_df["Time"]).dt.total_seconds()
        full_df["RaceTimeNorm"] = full_df["RaceTimeSec"] / full_df.groupby(
            ["Season", "RaceName"]
        )["RaceTimeSec"].transform("max")

        # ----- Fuel load proxy: laps remaining -----
        full_df["FuelLapsRemaining"] = (
            full_df.groupby(["Season", "RaceName"])["LapNumber"].transform("max")
            - full_df["LapNumber"]
        )

        # ----- Car performance factor (CarPaceIndex) -----
        # Step 1: per Season/Race/Driver, median of first 10 laps
        car_pace = (
            full_df.sort_values("LapNumber")
            .groupby(["Season", "RaceName", "Driver"])
            .apply(lambda g: g.head(10)["LapTimeSec"].median())
            .rename("CarPace")
            .reset_index()
        )

        full_df = full_df.merge(car_pace, on=["Season", "RaceName", "Driver"], how="left")

        # Step 2: per race median lap time
        race_median = full_df.groupby(["Season", "RaceName"])["LapTimeSec"].transform("median")

        # Step 3: CarPaceIndex = car pace relative to race median
        full_df["CarPaceIndex"] = full_df["CarPace"] - race_median
        full_df["CarPaceIndex"] = full_df["CarPaceIndex"].fillna(0.0)

        # Optional: drop raw CarPace
        full_df = full_df.drop(columns=["CarPace"])

        os.makedirs("data", exist_ok=True)
        save_df(full_df, "data/f1_lap_dataset.csv")
        print("Saved dataset to data/f1_lap_dataset.csv")


if __name__ == "__main__":
    build_dataset()
