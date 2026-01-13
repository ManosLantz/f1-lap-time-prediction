# features.py
"""
Feature engineering module for F1 Lap Time Prediction.

This module assumes all laps have been pre-filtered for validity.
It transforms raw telemetry and lap data into a structured dataset suitable for
supervised learning, including:
- Physics-based features (AEBI, Cornering metrics)
- History-based features (Rolling means, Tyre Age)
- Context-based features (Traffic, Track Status)
"""

import numpy as np
import pandas as pd
from utils import safe_divide
from config import CORNER_SPEED_THRESHOLD


def compute_aebi_for_session(telemetry: pd.DataFrame) -> float:
    """
    Computes the Aerodynamic Efficiency Brake Index (AEBI) for a session.
    
    AEBI is a proxy for how "track-limited" vs "power-limited" a circuit is.
    Formula: (Samples in Corners) / (Samples at Full Throttle)
    
    Args:
        telemetry: FastF1 telemetry object containing 'Speed' and 'Throttle'.
        
    Returns:
        float: AEBI ratio. Higher values indicate more twisty/technical tracks.
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


def _to_seconds_series(series: pd.Series) -> pd.Series:
    """Helper to convert Timedelta series to total seconds (float)."""
    s = pd.to_timedelta(series, errors="coerce")
    return s.dt.total_seconds().fillna(0.0)


def build_features_from_laps(laps: pd.DataFrame, aebi: float) -> pd.DataFrame:
    """
    Main feature engineering pipeline.
    
    Transforms the FastF1 laps DataFrame into a feature-rich dataset.
    
    Args:
        laps: Raw laps DataFrame from FastF1.
        aebi: Pre-computed AEBI scalar for this session.
        
    Returns:
        pd.DataFrame: DataFrame containing Features (X) and Targets (y).
    """

    base_cols = [
        "Driver", "LapNumber", "LapTime", "Compound", "TyreLife",
        "IsAccurate", "Time", "AirTemp", "Position", "TrackTemp",
        "GapToLeader", "IntervalToPositionAhead", "Sector1Time",
        "Sector2Time", "Sector3Time", "TrackStatus",
    ]

    cols = [c for c in base_cols if c in laps.columns]
    df = laps[cols].copy()

    # Filter invalid laps early (Basic data integrity)
    if "IsAccurate" in df.columns:
        df = df[df["IsAccurate"] == True]

    df = df.dropna(subset=["LapTime"])

    # 1. Base Conversions
    df["LapTimeSec"] = df["LapTime"].dt.total_seconds()
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce").fillna(0).astype(int)

    # Position handling
    if "Position" in df.columns:
        df["Position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(0).astype(int)
    else:
        df["Position"] = 0

    # Race time seconds for within-lap ordering
    df["RaceTimeSec"] = pd.to_timedelta(df["Time"], errors="coerce").dt.total_seconds().fillna(0.0)

    # 2. Gap Behind Proxy
    # Logic: Next car's race time - My race time (Same lap)
    df["GapBehindSec"] = 0.0
    for lap_no, g in df.groupby("LapNumber", sort=False):
        g_sorted = g.sort_values("RaceTimeSec")
        behind_gap = g_sorted["RaceTimeSec"].shift(-1) - g_sorted["RaceTimeSec"]
        df.loc[g_sorted.index, "GapBehindSec"] = behind_gap.fillna(0.0).clip(lower=0.0)

    # 3. Compound One-Hot Encoding
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].astype("category")
        compound_dummies = pd.get_dummies(df["Compound"], prefix="compound")
        df = pd.concat([df, compound_dummies], axis=1)

    # 4. Tyre Age
    if "TyreLife" in df.columns:
        df["TyreAge"] = df["TyreLife"].fillna(0).astype(float)
    else:
        df["TyreAge"] = 0.0

    # 5. Weather & Gaps
    
    if "AirTemp" in df.columns:
        df["AirTemp"] = pd.to_numeric(df["AirTemp"], errors="coerce").fillna(0.0)
    else:
        df["AirTemp"] = 0.0

    if "TrackTemp" in df.columns:
        df["TrackTemp"] = pd.to_numeric(df["TrackTemp"], errors="coerce").fillna(0.0)
    else:
        df["TrackTemp"] = 0.0

    if "GapToLeader" in laps.columns:
        df["GapToLeaderSec"] = _to_seconds_series(laps.loc[df.index, "GapToLeader"])
    else:
        df["GapToLeaderSec"] = 0.0

    if "IntervalToPositionAhead" in laps.columns:
        df["GapAheadSec"] = _to_seconds_series(laps.loc[df.index, "IntervalToPositionAhead"])
    else:
        df["GapAheadSec"] = 0.0

    # 6. Sector Times
    df["Sector1Sec"] = _to_seconds_series(laps.loc[df.index, "Sector1Time"]) if "Sector1Time" in laps.columns else 0.0
    df["Sector2Sec"] = _to_seconds_series(laps.loc[df.index, "Sector2Time"]) if "Sector2Time" in laps.columns else 0.0
    df["Sector3Sec"] = _to_seconds_series(laps.loc[df.index, "Sector3Time"]) if "Sector3Time" in laps.columns else 0.0

    # 7. Safety Flags
    if "TrackStatus" in laps.columns:
        status = laps.loc[df.index, "TrackStatus"].astype(str).fillna("")
        df["IsSC"] = status.str.contains("4").astype(int)
        df["IsVSC"] = status.str.contains("5|6").astype(int)
        df["IsYellow"] = status.str.contains("2|3").astype(int)
    else:
        df["IsSC"] = 0
        df["IsVSC"] = 0
        df["IsYellow"] = 0

    # 8. Session-Level Features
    df["AEBI"] = float(aebi)

    # CRITICAL: Sort by driver + lap for all history calculations
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    # ==========================================================================
    # HISTORY FEATURES (Rolling Statistics)
    # ==========================================================================
    
    # Rolling Mean/Std of LapTime (Last 5 laps)
    # Note: Shifted by 1 to prevent data leakage (using CURRENT lap to predict CURRENT lap)
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

    # Z-Score: How weird is this lap time compared to recent history?
    df["LapTime_z"] = (
        (df["LapTimeSec"] - df["LapTime_roll_mean"])
        / df["LapTime_roll_std"].replace(0, 1e-6)
    ).fillna(0.0)

    # Spikes: Detect sudden improvements
    df["LapSpikeFast"] = (df["LapTime_z"] < -1.5).astype(int)
    df["PrevLapSpikeFast"] = df.groupby("Driver")["LapSpikeFast"].shift(1).fillna(0).astype(int)

    # ==========================================================================
    # TARGETS & BASELINES
    # ==========================================================================
    df["PrevLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(1)
    df["PrevLapTimeSec"] = df["PrevLapTimeSec"].fillna(df["LapTimeSec"])

    df["NextLapTimeSec"] = df.groupby("Driver")["LapTimeSec"].shift(-1)
    # DELTA TARGET: Predict Change, not Absolute Time
    df["DeltaNextLapSec"] = df["NextLapTimeSec"] - df["PrevLapTimeSec"]

    # ==========================================================================
    # ADVANCED ENGINEERED FEATURES
    # ==========================================================================

    # PushIndex: Current pace vs Rolling Minimum
    # Positive = Slower than best (Saving?). Negative = Faster (Pushing?)
    rolling5min = (
        df.groupby("Driver")["LapTimeSec"]
          .rolling(5, min_periods=3).min()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
    )
    df["PushIndex"] = (df["PrevLapTimeSec"] - rolling5min).fillna(0.0)

    # Tyre Degradation Proxy: Simple Lap-to-Lap decay
    df["TyreDegRate"] = (df["LapTimeSec"] - df["PrevLapTimeSec"]).fillna(0.0)

    df["TyreDegSmooth"] = (
        df.groupby("Driver")["TyreDegRate"]
          .rolling(3, min_periods=2).mean()
          .reset_index(level=0, drop=True)
          .shift(1)
          .reindex(df.index)
          .fillna(0.0)
    )

    # Traffic Pressure
    df["DeltaGapAhead"] = df.groupby("Driver")["GapAheadSec"].diff().fillna(0.0)

    # ERS Usage Proxy: Sector 3 Performance
    # Reasoning: S3 usually requires ERS deployment. Drop in S3 vs Best S3 = Low ERS?
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

    # PushTrend: Is the driver speeding up or slowing down over 3 laps?
    df["PushTrend"] = (
        df.groupby("Driver")["LapTimeSec"]
          .apply(lambda s: s.diff().rolling(3, min_periods=2).mean().shift(1))
          .reset_index(level=0, drop=True)
          .reindex(df.index)
          .fillna(0.0)
    )

    # Interactive Flags
    df["IsDRSRange"] = ((df["GapAheadSec"] > 0) & (df["GapAheadSec"] <= 1.0)).astype(int)
    df["IsUnderAttack"] = ((df["GapBehindSec"] > 0) & (df["GapBehindSec"] <= 1.0)).astype(int)

    # Cleanup: Remove rows where we can't train (no target)
    df = df.dropna(subset=["NextLapTimeSec", "DeltaNextLapSec"])

    return df
