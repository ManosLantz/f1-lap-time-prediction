import os
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import fastf1

def get_chronological_racekeys_2025():
    schedule = fastf1.get_event_schedule(2025)
    races = schedule[schedule["EventFormat"].isin(["conventional", "sprint"])]
    races = races.sort_values("EventDate")

    racekeys = []
    for _, r in races.iterrows():
        racekeys.append(f"2025_{r['EventName']}")
    return racekeys


# ================= CONFIG =================
DATA_PATH = "data/f1_lap_dataset.csv"
META_PATH = "model_v3_meta.pkl"   # for features + best params
SEASON_EVAL = 2025

OUT_DIR = "report_artifacts"
OUT_CSV = os.path.join(OUT_DIR, "per_race_2025_forward_loro.csv")
OUT_TXT = os.path.join(OUT_DIR, "eval_2025_forward_loro_summary.txt")

# =========================================

def mae(y, yhat):
    return float(mean_absolute_error(y, yhat))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- Load data ----------------
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = ["Season", "RaceName", "LapNumber",
                "PrevLapTimeSec", "NextLapTimeSec"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["Season"] = df["Season"].astype(int)
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["PrevLapTimeSec"] = pd.to_numeric(df["PrevLapTimeSec"], errors="coerce")
    df["NextLapTimeSec"] = pd.to_numeric(df["NextLapTimeSec"], errors="coerce")

    df = df.dropna(subset=["LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]).copy()

    df["RaceKey"] = df["Season"].astype(str) + "_" + df["RaceName"].astype(str)

    # ---------------- Load meta ----------------
    meta = joblib.load(META_PATH)
    FEATURES = meta["features"]
    BEST_PARAMS = meta["best_params"]

    # ---------------- Feature patches ----------------
    if "FuelLapsRemaining" in FEATURES and "FuelLapsRemaining" not in df.columns:
        max_laps = df.groupby("RaceKey")["LapNumber"].transform("max")
        df["FuelLapsRemaining"] = max_laps - df["LapNumber"]

    if "CarPaceIndex" in FEATURES and "CarPaceIndex" not in df.columns:
        race_med = df.groupby("RaceKey")["PrevLapTimeSec"].transform("median")
        df["CarPaceIndex"] = df["PrevLapTimeSec"] / race_med

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Still missing features: {missing}")

    # ---------------- Split races ----------------
    races_2025 = get_chronological_racekeys_2025()

    # Keep only races that actually exist in the dataset
    races_2025 = [r for r in races_2025 if r in df["RaceKey"].unique()]

    if len(races_2025) == 0:
        raise ValueError("No 2025 races found.")

    print("2025 race order:")
    for r in races_2025:
        print(" ", r)

    results = []

    # ---------------- Forward LORO ----------------
    for i, test_race in enumerate(races_2025):
        print(f"\nEvaluating {test_race} ({i+1}/{len(races_2025)})")

        # Training races: all < test_race chronologically
        train_races = (
            df[
                (df["Season"] < SEASON_EVAL) |
                ((df["Season"] == SEASON_EVAL) &
                 (df["RaceKey"].isin(races_2025[:i])))
            ]["RaceKey"]
            .unique()
            .tolist()
        )

        train_df = df[df["RaceKey"].isin(train_races)]
        test_df = df[df["RaceKey"] == test_race]

        if len(test_df) < 50 or len(train_df) < 1000:
            print("  Skipped (insufficient data)")
            continue

        X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
        y_train = (
            train_df["NextLapTimeSec"] - train_df["PrevLapTimeSec"]
        ).to_numpy(dtype=float)

        X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
        y_test = test_df["NextLapTimeSec"].to_numpy(dtype=float)
        base = test_df["PrevLapTimeSec"].to_numpy(dtype=float)

        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)

        model = xgb.XGBRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        pred_delta = model.predict(X_test)
        pred = base + pred_delta

        base_mae = mae(y_test, base)
        model_mae = mae(y_test, pred)
        gain = base_mae - model_mae

        print(f"  Baseline: {base_mae:.4f} | Model: {model_mae:.4f} | Gain: {gain:.4f}")

        results.append({
            "RaceKey": test_race,
            "n_samples": len(test_df),
            "baseline_mae": base_mae,
            "model_mae": model_mae,
            "mae_gain": gain
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT_CSV, index=False)

    mean_gain = res["mae_gain"].mean()
    median_gain = res["mae_gain"].median()
    pct_pos = (res["mae_gain"] > 0).mean() * 100

    summary = (
        f"Forward LORO evaluation – Season {SEASON_EVAL}\n"
        f"Races evaluated: {len(res)}\n\n"
        f"Mean MAE gain (race-level): {mean_gain:.6f} s\n"
        f"Median MAE gain: {median_gain:.6f} s\n"
        f"% races improved: {pct_pos:.2f}%\n\n"
        f"Saved per-race table: {OUT_CSV}"
    )

    print("\n" + summary)

    with open(OUT_TXT, "w") as f:
        f.write(summary)

    print(f"\nSaved summary to {OUT_TXT}")


if __name__ == "__main__":
    main()
