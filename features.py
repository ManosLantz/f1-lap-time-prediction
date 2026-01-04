# features.py

import numpy as np
import pandas as pd
from utils import safe_divide
from config import CORNER_SPEED_THRESHOLD


def compute_aebi_for_session(telemetry) -> float:
    """
    AEBI = (time in corners) / (time at full throttle)
    Uses speed + throttle samples for one reference driver.
    """
    if telemetry is None or telemetry.empty:
        return 0.0

    speeds = telemetry["Speed"].values  # km/h
    throttle = telemetry["Throttle"].values  # 0–100

    if len(speeds) == 0:
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

    base_cols = [
        "Driver",
        "LapNumber",
        "LapTime",
        "Compound",
        "TyreLife",
        "IsAccurate",
        "Time",
        "AirTemp",
        "Position",
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

    # Filter invalid laps early
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]

    df = df.dropna(subset=["LapTime"])

    # Lap time in seconds
    df["LapTimeSec"] = df["LapTime"].dt.total_seconds()

    # Ensure LapNumber numeric
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(0).astype(int)

    # Position
    if "Position" in df.columns:
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(0).astype(int)
    else:
        df["Position"] = 0

    # Race time seconds for within-lap ordering
    df["RaceTimeSec"] = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds().fillna(0.0)

    # --- Gap behind proxy (same-lap ordering by RaceTimeSec) ---
    df["GapBehindSec"] = 0.0
    for lap_no, g in df.groupby("LapNumber", sort=False):
        g_sorted = g.sort_values("RaceTimeSec")
        behind_gap = g_sorted["RaceTimeSec"].shift(-1) - g_sorted["RaceTimeSec"]
        df.loc[g_sorted.index, "GapBehindSec"] = behind_gap.fillna(0.0).clip(lower=0.0)

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
    df["AirTemp"] = pd.to_numeric(df.get("AirTemp", 0.0), errors="coerce").fillna(0.0)
    df["TrackTemp"] = pd.to_numeric(df.get("TrackTemp", 0.0), errors="coerce").fillna(0.0)

    # Gaps
    if "GapToLeader" in laps.columns:
        df["GapToLeaderSec"] = _to_seconds_series(laps.loc[df.index, "GapToLeader"])
    else:
        df["GapToLeaderSec"] = 0.0

    if "IntervalToPositionAhead" in laps.columns:
        df["GapAheadSec"] = _to_seconds_series(laps.loc[df.index, "IntervalToPositionAhead"])
    else:
        df["GapAheadSec"] = 0.0

    # Sector times → seconds
    df["Sector1Sec"] = _to_seconds_series(laps.loc[df.index, "Sector1Time"]) if "Sector1Time" in laps.columns else 0.0
    df["Sector2Sec"] = _to_seconds_series(laps.loc[df.index, "Sector2Time"]) if "Sector2Time" in laps.columns else 0.0
    df["Sector3Sec"] = _to_seconds_series(laps.loc[df.index, "Sector3Time"]) if "Sector3Time" in laps.columns else 0.0

    # Safety flags from TrackStatus
    if "TrackStatus" in laps.columns:
        status = laps.loc[df.index, "TrackStatus"].astype(str).fillna("")
        df["IsSC"] = status.str.contains("4").astype(int)
        df["IsVSC"] = status.str.contains("5|6").astype(int)
        df["IsYellow"] = status.str.contains("2|3").astype(int)
    else:
        df["IsSC"] = 0
        df["IsVSC"] = 0
        df["IsYellow"] = 0

    # AEBI constant per session
    df["AEBI"] = float(aebi)

    # Sort by driver + lap (CRITICAL)
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # =========================
    # Rolling LapTime stats (PAST-only)
    # =========================
    df["LapTime_roll_mean"] = (
        df.groupby("Driver")["LapTimeSec"]
          .rolling(5, min_periods=3).mean()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
    )

    df["LapTime_roll_std"] = (
        df.groupby("Driver")["LapTimeSec"]
          .rolling(5, min_periods=3).std()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
    )

    df["LapTime_z"] = (
        (df["LapTimeSec"] - df["LapTime_roll_mean"])
        / df["LapTime_roll_std"].replace(0, 1e-6)
    ).fillna(0.0)

    df["LapSpikeFast"] = (df["LapTime_z"] < -1.5).astype(int)
    df["PrevLapSpikeFast"] = df.groupby("Driver")["LapSpikeFast"].shift(1).fillna(0).astype(int)

    # =========================
    # Targets + sequential baseline features
    # =========================
    df["PrevLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(1)
    df["PrevLapTimeSec"] = df["PrevLapTimeSec"].fillna(df["LapTimeSec"])

    df["NextLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(-1)
    df["DeltaNextLapSec"] = df["NextLapTimeSec"] - df["PrevLapTimeSec"]

    # =========================
    # Engineered features (history only)
    # =========================

    # PushIndex: prev lap vs rolling 5-lap min (history)
    rolling5min = (
        df.groupby("Driver")["LapTimeSec"]
          .rolling(5, min_periods=3).min()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
    )
    df["PushIndex"] = (df["PrevLapTimeSec"] - rolling5min).fillna(0.0)

    # Tyre degradation proxy
    df["TyreDegRate"] = (df["LapTimeSec"] - df["PrevLapTimeSec"]).fillna(0.0)

    df["TyreDegSmooth"] = (
        df.groupby("Driver")["TyreDegRate"]
          .rolling(3, min_periods=2).mean()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
          .fillna(0.0)
    )

    # Delta gap ahead: traffic pressure proxy
    df["DeltaGapAhead"] = df.groupby("Driver")["GapAheadSec"].diff().fillna(0.0)

    # ERSProxy from Sector3 vs recent best (history only)
    rolling_s3_min = (
        df.groupby("Driver")["Sector3Sec"]
          .rolling(3, min_periods=2).min()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
    )
    df["ERSProxy"] = (df["Sector3Sec"] - rolling_s3_min).fillna(0.0)

    df["ERSVolatility"] = (
        df.groupby("Driver")["ERSProxy"]
          .rolling(5, min_periods=3).std()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
          .fillna(0.0)
    )

    # ✅ FIXED PushTrend: must be per-driver rolling (was mixing drivers)
    df["PushTrend"] = (
        df.groupby("Driver")["LapTimeSec"]
          .apply(lambda s: s.diff().rolling(3, min_periods=2).mean().shift(1))
          .reset_index(level=0, drop=True)
          .reindex(df.index)
          .fillna(0.0)
    )

    # DRS / Under-attack flags
    df["IsDRSRange"] = ((df["GapAheadSec"] > 0) & (df["GapAheadSec"] <= 1.0)).astype(int)
    df["IsUnderAttack"] = ((df["GapBehindSec"] > 0) & (df["GapBehindSec"] <= 1.0)).astype(int)

    # Remove rows with no next lap target
    df = df.dropna(subset=["NextLapTimeSec", "DeltaNextLapSec"])

    return df
