# data_loader.py (STRICT CLEAN GREEN ONLY)

import pandas as pd
import numpy as np

DATA_PATH = "data/f1_lap_dataset.csv"

BASE_FEATURES = [
    "PrevLapTimeSec", "TyreAge", "AirTemp", "TrackTemp", "AEBI",
    "Sector1Sec", "Sector2Sec", "Sector3Sec",
    "PushIndex", "TyreDegSmooth", "FuelLapsRemaining",
    "CarPaceIndex", "IsDRSRange", "GapAheadSec"
]


def is_clean_trackstatus(ts) -> bool:
    """
    FastF1 TrackStatus is a string that can contain multiple codes in one lap (e.g. '67').
    Clean racing only if it is exactly '1' (track clear). :contentReference[oaicite:6]{index=6}
    """
    if pd.isna(ts):
        return False
    s = str(ts).strip()
    return s == "1"


def generate_missing_features(df):
    # Group by Season+RaceName (DO NOT group by RaceName alone; mixes seasons)
    grp = ['Season', 'RaceName']

    if 'FuelLapsRemaining' not in df.columns and 'LapNumber' in df.columns:
        max_laps = df.groupby(grp)['LapNumber'].transform('max')
        df['FuelLapsRemaining'] = max_laps - df['LapNumber']

    if 'CarPaceIndex' not in df.columns and 'LapTimeSec' in df.columns:
        race_medians = df.groupby(grp)['LapTimeSec'].transform('median')
        df['CarPaceIndex'] = df['LapTimeSec'] / race_medians

    # If you have LapsSinceRestart already created, keep it; if not, fill 0
    if 'LapsSinceRestart' not in df.columns:
        df['LapsSinceRestart'] = 0

    return df


def clean_data_strict(df):
    # 1) Slicks only (helps but "dry" already enforced by f1_data.py skipping rainy races)
    if 'Compound' in df.columns:
        df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])].copy()

    # 2) Hard filter by TrackStatus (only exact "1" survives)
    if 'TrackStatus' not in df.columns:
        raise ValueError("TrackStatus column missing. Rebuild dataset ensuring TrackStatus is exported.")

    df['IsCleanStatus'] = df['TrackStatus'].apply(is_clean_trackstatus)

    # 3) Protect transitions: current/prev/next must all be clean
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber']).copy()

    df['PrevClean'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsCleanStatus'].shift(1).fillna(False)
    df['NextClean'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsCleanStatus'].shift(-1).fillna(False)

    df = df[(df['IsCleanStatus']) & (df['PrevClean']) & (df['NextClean'])].copy()

    df.drop(columns=['IsCleanStatus', 'PrevClean', 'NextClean'], inplace=True)

    return df


def load_features():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Strict cleaning
    print("Applying STRICT filters: no rain races + clean green only laps...")
    df = clean_data_strict(df)

    print(f"After strict filters: {len(df)} laps.")

    df = generate_missing_features(df)

    # One-hot encoding
    driver_backup = df['Driver'].copy() if 'Driver' in df.columns else None
    if 'Driver' in df.columns:
        df = pd.get_dummies(df, columns=['Driver'], dummy_na=False)
    if 'Compound' in df.columns:
        df = pd.get_dummies(df, columns=['Compound'], dummy_na=False)
    if driver_backup is not None:
        df['Driver'] = driver_backup

    dummy_cols = [c for c in df.columns if c.startswith('Driver_') or c.startswith('Compound_')]
    model_features = BASE_FEATURES + dummy_cols

    # Keep only features that actually exist (defensive)
    model_features = [f for f in model_features if f in df.columns]
    # Force all model features to numeric
    for c in model_features:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    model_features = [
        "PrevLapTimeSec",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "PushIndex",
        "TyreAge",
        "FuelLapsRemaining",
        "TyreDegSmooth",
        "CarPaceIndex",
    ]

    return df, model_features
