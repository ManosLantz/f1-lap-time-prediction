# tune_xgboost.py

import pandas as pd
import random
from math import sqrt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor


DATA_PATH = "data/f1_lap_dataset.csv"


def prepare_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("Loaded dataset:", df.shape)

    # === Basic checks ===
    for col in ["NextLapTimeSec", "PrevLapTimeSec"]:
        if col not in df.columns:
            raise ValueError(f"Missing {col}, check f1_data.py / features.py")

    # === Numeric cleanup ===
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
    ]

    for col in numeric_to_fill_zero:
        df[col] = df.get(col, 0).fillna(0.0)

    for col in ["AirTemp", "TrackTemp"]:
        df[col] = df.get(col, 0).fillna(df[col].median())

    df["Season"] = df.get("Season", 0).astype(int)
    df["TyreAge"] = df.get("TyreAge", 0).fillna(0.0)

    # === One-hot driver & circuit ===
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

    # Fix compound_* types: map True/False/"1"/"0" -> 1.0/0.0
    for c in compound_cols:
        df[c] = (
            df[c]
            .astype(str)
            .map({"True": 1.0, "False": 0.0, "1": 1.0, "0": 0.0})
            .fillna(0.0)
        )

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
    ]

    feature_cols = base_features + compound_cols + driver_cols + circuit_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # === Same 2025 split logic as train_model.py ===
    races_2025 = sorted(df.loc[df["Season"] == 2025, "RaceName"].unique())
    print("2025 races:", races_2025)

    if len(races_2025) < 2:
        raise ValueError("Need at least 2 races in 2025 for test split.")

    random.seed(42)
    test_races = random.sample(races_2025, 2)
    print("Chosen 2025 test races (held-out):", test_races)

    test_mask = (df["Season"] == 2025) & (df["RaceName"].isin(test_races))
    train_mask = ~test_mask

    train_df = df[train_mask].copy()   # tuning will use ONLY this
    # test_df  = df[test_mask].copy()  # ignored for tuning, used only in final eval

    # === Apply same filters as final model: clean, dry, no mistakes ===
    if set(["IsSC", "IsVSC", "IsYellow"]).issubset(train_df.columns):
        train_df = train_df.query("IsSC == 0 and IsVSC == 0 and IsYellow == 0")

    if "Compound" in train_df.columns:
        train_df = train_df[train_df["Compound"].isin(["SOFT", "MEDIUM", "HARD"])]

    if "IsMistakeLap" in train_df.columns:
        train_df = train_df.query("IsMistakeLap == 0")

    X_train = train_df[feature_cols]
    y_train = train_df["NextLapTimeSec"]

    print("Tuning on training samples:", len(X_train))

    return X_train, y_train, feature_cols


def main():
    X_train, y_train, feature_cols = prepare_data()

    # === 2. Define XGBoost model & search space ===
    base_model = XGBRegressor(
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
    )

    param_distributions = {
        "n_estimators": [200, 300, 400, 600],
        "max_depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.05, 0.07, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "gamma": [0.0, 0.1, 0.3],
    }

    # MAE as scoring (we want to MINIMISE it, so use negative MAE)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=25,           # increase for more thorough search
        scoring=mae_scorer,
        cv=3,                # 3-fold CV on training data
        verbose=2,
        random_state=42,
        n_jobs=-1,
    )

    print("\nStarting RandomizedSearchCV for XGBoost...")
    search.fit(X_train, y_train)

    print("\nBest params found:")
    print(search.best_params_)

    print("\nBest CV score (negative MAE):", search.best_score_)
    print("Best CV MAE:", -search.best_score_)


if __name__ == "__main__":
    main()
