# features.py

import numpy as np
import pandas as pd
from utils import safe_divide
from config import CORNER_SPEED_THRESHOLD


def compute_aebi_for_session(telemetry) -> float:
    """
    Compute Aero–Engine Balance Index (AEBI) for a session:
    AEBI = (time in corners) / (time at full throttle).
    Uses speed + throttle samples for one reference driver.
    """
    if telemetry is None or telemetry.empty:
        return 0.0

    speeds = telemetry["Speed"].values  # km/h
    throttle = telemetry["Throttle"].values  # 0–100

    total_samples = len(speeds)
    if total_samples == 0:
        return 0.0

    corner_samples = np.sum(speeds < CORNER_SPEED_THRESHOLD)
    full_throttle_samples = np.sum(throttle == 100)

    return safe_divide(corner_samples, full_throttle_samples)


def _to_seconds_series(series):
    s = pd.to_timedelta(series, errors="coerce")
    return s.dt.total_seconds().fillna(0.0)


def build_features_from_laps(laps: pd.DataFrame, aebi: float) -> pd.DataFrame:
    """
    Input: laps DataFrame from fastf1 (optionally merged with weather).
    Output: DataFrame with features and targets (NextLapTimeSec, DeltaNextLapSec).
    """

    # Columns we want if they exist
    base_cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife",
        "IsAccurate",
        "Time",
        "AirTemp",
        "TrackTemp",
        "GapToLeader",
        "IntervalToPositionAhead",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "TrackStatus",
    ]

    cols = [c for c in base_cols if c in laps.columns]
    df = laps[cols].copy()

    # Filter invalid laps
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]

    df = df.dropna(subset=["LapTime"])

    # Lap time in seconds
    df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    # Compound → category + one-hot
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype("category")
        compound_dummies = pd.get_dummies(df["Compound"], prefix="compound")
        df = pd.concat([df, compound_dummies], axis=1)

    # Tyre age
    if "TyreLife" in df.columns:
        df["TyreAge"] = df["TyreLife"].fillna(0).astype(float)
    else:
        df["TyreAge"] = 0.0

    # Weather
    if "AirTemp" in df.columns:
        df["AirTemp"] = df["AirTemp"].astype(float)
    else:
        df["AirTemp"] = 0.0

    if "TrackTemp" in df.columns:
        df["TrackTemp"] = df["TrackTemp"].astype(float)
    else:
        df["TrackTemp"] = 0.0

    # Gaps
    if "GapToLeader" in laps.columns:
        df["GapToLeaderSec"] = _to_seconds_series(laps.loc[df.index, "GapToLeader"])
    else:
        df["GapToLeaderSec"] = 0.0

    if "IntervalToPositionAhead" in laps.columns:
        df["GapAheadSec"] = _to_seconds_series(
            laps.loc[df.index, "IntervalToPositionAhead"]
        )
    else:
        df["GapAheadSec"] = 0.0

    # Sector times → in seconds
    if "Sector1Time" in laps.columns:
        df["Sector1Sec"] = _to_seconds_series(laps.loc[df.index, "Sector1Time"])
    else:
        df["Sector1Sec"] = 0.0

    if "Sector2Time" in laps.columns:
        df["Sector2Sec"] = _to_seconds_series(laps.loc[df.index, "Sector2Time"])
    else:
        df["Sector2Sec"] = 0.0

    if "Sector3Time" in laps.columns:
        df["Sector3Sec"] = _to_seconds_series(laps.loc[df.index, "Sector3Time"])
    else:
        df["Sector3Sec"] = 0.0

    # Safety Car / VSC / Yellow flags from TrackStatus (string codes)
    if "TrackStatus" in laps.columns:
        status = laps.loc[df.index, "TrackStatus"].astype(str).fillna("")
        df["IsSC"] = status.str.contains("4").astype(int)
        df["IsVSC"] = status.str.contains("5|6").astype(int)
        df["IsYellow"] = status.str.contains("2|3").astype(int)
    else:
        df["IsSC"] = 0
        df["IsVSC"] = 0
        df["IsYellow"] = 0

    # AEBI as constant feature
    df["AEBI"] = aebi

    # Sort by driver + lap
    df = df.sort_values(["Driver", "LapNumber"])

    # Previous lap time (per driver)
    df["PrevLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(1)
    df["PrevLapTimeSec"] = df["PrevLapTimeSec"].fillna(df["LapTimeSec"])

    # Next lap time (target baseline)
    df["NextLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(-1)

    # Delta target: next lap relative to previous lap
    df["DeltaNextLapSec"] = df["NextLapTimeSec"] - df["PrevLapTimeSec"]

    # ===== NEW FEATURES =====

    # 1) PushIndex: how hard the previous lap was vs 5-lap rolling min pace
    df["Rolling5MinLap"] = (
        df.groupby("Driver")["LapTimeSec"]
        .rolling(5, min_periods=3)
        .min()
        .reset_index(level=0, drop=True)
    )
    df["PushIndex"] = df["PrevLapTimeSec"] - df["Rolling5MinLap"]
    df["PushIndex"] = df["PushIndex"].fillna(0.0)

    # 2) Tyre degradation rate and smoothed version
    df["TyreDegRate"] = df["LapTimeSec"] - df["PrevLapTimeSec"]
    df["TyreDegRate"] = df["TyreDegRate"].fillna(0.0)

    df["TyreDegSmooth"] = (
        df.groupby("Driver")["TyreDegRate"]
        .rolling(3, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["TyreDegSmooth"] = df["TyreDegSmooth"].fillna(0.0)

    # 3) Delta gap ahead: traffic pressure proxy
    df["DeltaGapAhead"] = df.groupby("Driver")["GapAheadSec"].diff()
    df["DeltaGapAhead"] = df["DeltaGapAhead"].fillna(0.0)

    # 4) ERSProxy from Sector3: how slow sector 3 is vs recent best
    rolling_s3_min = (
        df.groupby("Driver")["Sector3Sec"]
        .rolling(3, min_periods=2)
        .min()
        .reset_index(level=0, drop=True)
    )
    df["ERSProxy"] = df["Sector3Sec"] - rolling_s3_min
    df["ERSProxy"] = df["ERSProxy"].fillna(0.0)

    # Drop helper column to keep dataset clean
    df = df.drop(columns=["Rolling5MinLap"])

    # Remove rows where there is no next lap
    df = df.dropna(subset=["NextLapTimeSec", "DeltaNextLapSec"])

    return df

