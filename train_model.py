# train_model.py

import pandas as pd
from math import sqrt
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor


DATA_PATH = "data/f1_lap_dataset.csv"

def prepare_data_for_training():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Loaded dataset:", df.shape)

    # Required columns
    if "NextLapTimeSec" not in df.columns or "PrevLapTimeSec" not in df.columns:
        raise ValueError("Dataset is missing required target columns.")

    # === 1. Numeric cleanup ===
    numeric_to_fill_zero = [
        "GapToLeaderSec",
        "GapAheadSec",
        "TrackLengthKm",
        "TrackCorners",
        "TrackAvgSpeedKph",
        "RaceLapNorm",
        "RaceTimeNorm",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "CompoundChange",
        "IsSC",
        "IsVSC",
        "IsYellow",
        # new features:
        "PushIndex",
        "TyreDegRate",
        "TyreDegSmooth",
        "DeltaGapAhead",
        "ERSProxy",
        "CarPaceIndex",
        "FuelLapsRemaining",
    ]


    for col in numeric_to_fill_zero:
        df[col] = df.get(col, 0).fillna(0.0)

    # Temps → median fill
    for col in ["AirTemp", "TrackTemp"]:
        df[col] = df.get(col, 0).fillna(df[col].median())

    df["Season"] = df.get("Season", 0).astype(int)
    df["TyreAge"] = df.get("TyreAge", 0).fillna(0.0)

    # === 2. One-hot encoding ===
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["Driver"], prefix="driver"),
            pd.get_dummies(df["RaceName"], prefix="circuit"),
        ],
        axis=1,
    )

    compound_cols = [c for c in df.columns if c.startswith("compound_")]
    driver_cols   = [c for c in df.columns if c.startswith("driver_")]
    circuit_cols  = [c for c in df.columns if c.startswith("circuit_")]

    # Fix booleans / strings in compound_* columns
    for c in compound_cols:
        df[c] = (
            df[c]
            .astype(str)
            .map({"True": 1.0, "False": 0.0, "1": 1.0, "0": 0.0})
            .fillna(0.0)
        )

    # === 3. Feature list ===
    base_features = [
        "LapNumber",
        "TyreAge",
        "AirTemp",
        "TrackTemp",
        "AEBI",
        "GapToLeaderSec",
        "GapAheadSec",
        "PrevLapTimeSec",
        "Season",
        "TrackLengthKm",
        "TrackCorners",
        "TrackAvgSpeedKph",
        "RaceLapNorm",
        "RaceTimeNorm",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "IsSC",
        "IsVSC",
        "IsYellow",
        "CompoundChange",
        # new features:
        "PushIndex",
        "TyreDegRate",
        "TyreDegSmooth",
        "DeltaGapAhead",
        "ERSProxy",
        "CarPaceIndex",
        "FuelLapsRemaining",
    ]


    feature_cols = base_features + compound_cols + driver_cols + circuit_cols

    # Ensure numeric dtype
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # === 4. Split (same as master script) ===
    races_2025 = sorted(df.loc[df["Season"] == 2025, "RaceName"].unique())
    if len(races_2025) < 2:
        raise ValueError("Need at least 2 races from 2025.")

    random.seed(42)
    test_races = random.sample(races_2025, 2)

    test_mask = (df["Season"] == 2025) & (df["RaceName"].isin(test_races))
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df  = df[test_mask].copy()

    # === 5. Filters: remove SC/VSC/yellow, wet tyres, mistakes ===
    for d in (train_df, test_df):
        # Remove SC/VSC/Yellow laps
        if set(["IsSC", "IsVSC", "IsYellow"]).issubset(d.columns):
            d.query("IsSC == 0 and IsVSC == 0 and IsYellow == 0", inplace=True)

        # Dry tyres only
        if "Compound" in d.columns:
            d.query("Compound in ['SOFT', 'MEDIUM', 'HARD']", inplace=True)

        # Remove mistake laps
        if "IsMistakeLap" in d.columns:
            d.query("IsMistakeLap == 0", inplace=True)

    X_train = train_df[feature_cols]
    y_train = train_df["NextLapTimeSec"]

    X_test = test_df[feature_cols]
    y_test = test_df["NextLapTimeSec"]

    print("Training samples:", len(X_train))
    print("Testing samples: ", len(X_test))
    print("Test races:", test_races)

    return X_train, y_train, X_test, y_test


def main():
    # === 1. Load data ===
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    print("Loaded dataset:", df.shape)
    print("Columns:", df.columns.tolist())

    # Required columns check
    for col in ["NextLapTimeSec", "PrevLapTimeSec"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col}, check f1_data.py and features.py")

    # === 2. Basic numeric cleanup ===

    numeric_to_fill_zero = [
        "GapToLeaderSec",
        "GapAheadSec",
        "TrackLengthKm",
        "TrackCorners",
        "TrackAvgSpeedKph",
        "RaceLapNorm",
        "RaceTimeNorm",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "CompoundChange",
        "IsSC",
        "IsVSC",
        "IsYellow",
        # new features:
        "PushIndex",
        "TyreDegRate",
        "TyreDegSmooth",
        "DeltaGapAhead",
        "ERSProxy",
        "CarPaceIndex",
        "FuelLapsRemaining",
    ]


    for col in numeric_to_fill_zero:
        df[col] = df.get(col, 0).fillna(0.0)

    # Temps → fill with median
    for col in ["AirTemp", "TrackTemp"]:
        df[col] = df.get(col, 0).fillna(df[col].median())

    df["Season"] = df.get("Season", 0).astype(int)
    df["TyreAge"] = df.get("TyreAge", 0).fillna(0.0)

    # === 3. One-hot driver & circuit ===
    df = pd.concat(
        [
            df,
            pd.get_dummies(df["Driver"], prefix="driver"),
            pd.get_dummies(df["RaceName"], prefix="circuit"),
        ],
        axis=1,
    )

    compound_cols = [c for c in df.columns if c.startswith("compound_")]
    driver_cols = [c for c in df.columns if c.startswith("driver_")]
    circuit_cols = [c for c in df.columns if c.startswith("circuit_")]

    # --- FIX: convert compound_* from "True"/"False"/bool to 0.0/1.0 ---
    for c in compound_cols:
        # if already numeric, this is cheap; if "True"/"False", map to 1/0
        df[c] = (
            df[c]
            .astype(str)
            .map({"True": 1.0, "False": 0.0, "1": 1.0, "0": 0.0})
            .fillna(0.0)
        )

    # === 4. Feature list ===
    base_features = [
        "LapNumber",
        "TyreAge",
        "AirTemp",
        "TrackTemp",
        "AEBI",
        "GapToLeaderSec",
        "GapAheadSec",
        "PrevLapTimeSec",
        "Season",
        "TrackLengthKm",
        "TrackCorners",
        "TrackAvgSpeedKph",
        "RaceLapNorm",
        "RaceTimeNorm",
        "Sector1Sec",
        "Sector2Sec",
        "Sector3Sec",
        "IsSC",
        "IsVSC",
        "IsYellow",
        "CompoundChange",
        # new features:
        "PushIndex",
        "TyreDegRate",
        "TyreDegSmooth",
        "DeltaGapAhead",
        "ERSProxy",
        "CarPaceIndex",
        "FuelLapsRemaining",
    ]


    feature_cols = base_features + compound_cols + driver_cols + circuit_cols

    # ensure all exist and are numeric
    feature_cols = [c for c in feature_cols if c in df.columns]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # === 5. Choose test races (2 random 2025 races) ===
    races_2025 = sorted(df.loc[df["Season"] == 2025, "RaceName"].unique())
    print("2025 races:", races_2025)

    if len(races_2025) < 2:
        raise ValueError("Need at least 2 races in 2025 for test split.")

    random.seed(42)
    test_races = random.sample(races_2025, 2)
    print("Chosen 2025 test races:", test_races)

    test_mask = (df["Season"] == 2025) & (df["RaceName"].isin(test_races))
    train_mask = ~test_mask

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    # === 6. Filters: Keep only clean, dry, green laps ===
    for d_name, d in [("train", train_df), ("test", test_df)]:
        # No SC/VSC/Yellow
        if set(["IsSC", "IsVSC", "IsYellow"]).issubset(d.columns):
            d.query("IsSC == 0 and IsVSC == 0 and IsYellow == 0", inplace=True)

        # Only dry compounds (based on original 'Compound' column)
        if "Compound" in d.columns:
            d.query("Compound in ['SOFT', 'MEDIUM', 'HARD']", inplace=True)

        # No mistake laps if column exists
        if "IsMistakeLap" in d.columns:
            d.query("IsMistakeLap == 0", inplace=True)

        print(f"After filters, {d_name} samples: {len(d)}")

    X_train = train_df[feature_cols]
    y_train = train_df["NextLapTimeSec"]

    X_test = test_df[feature_cols]
    y_test = test_df["NextLapTimeSec"]
    # === Naive baseline: predict next lap as previous lap ===
    if "PrevLapTimeSec" not in test_df.columns:
        raise ValueError("PrevLapTimeSec not found in test_df; check feature generation.")

    y_pred_naive = test_df["PrevLapTimeSec"].values

    baseline_mae = mean_absolute_error(y_test, y_pred_naive)
    baseline_rmse = sqrt(mean_squared_error(y_test, y_pred_naive))

    print("\n==== NAIVE BASELINE (NextLap = PrevLap) ====")
    print(f"Baseline MAE:  {baseline_mae:.3f} sec")
    print(f"Baseline RMSE: {baseline_rmse:.3f} sec\n")


    print(f"Final training samples: {len(X_train)}")
    print(f"Final testing samples:  {len(X_test)}")

    # === 7. Train XGBoost model ===
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=3,
        gamma=0.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        n_jobs=-1,
        tree_method="hist",
    )



    print("\nTraining XGBoost...")
    model.fit(X_train, y_train)

    # === 8. Evaluate ===
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n==== XGBoost PERFORMANCE ====")
    print(f"MAE:  {mae:.3f} sec")
    print(f"RMSE: {rmse:.3f} sec")

    # === 9. Feature importances ===
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type="gain")

    importance_list = sorted(
        [(k, v) for k, v in importance_dict.items()],
        key=lambda x: -x[1],
    )

    print("\nTop 25 Most Important Features:")
    for feat, imp in importance_list[:25]:
        print(f"{feat:30s} {imp:.5f}")


if __name__ == "__main__":
    main()
