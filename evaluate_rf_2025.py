import os
import numpy as np
import pandas as pd
import joblib
import fastf1

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def get_chronological_racekeys_2025():
    schedule = fastf1.get_event_schedule(2025)
    races = schedule[schedule["EventFormat"].isin(["conventional", "sprint"])]
    races = races.sort_values("EventDate")
    return [f"2025_{r['EventName']}" for _, r in races.iterrows()]


# ================= CONFIG =================
DATA_PATH = "data/f1_lap_dataset.csv"
PARAMS_PATH = "best_params_rf_leave_k_races_out.pkl"  # produced by tune_rf_loro.py
SEASON_EVAL = 2025

OUT_DIR = "report_artifacts"
OUT_CSV = os.path.join(OUT_DIR, "per_race_2025_forward_loro_rf.csv")
OUT_TXT = os.path.join(OUT_DIR, "eval_2025_forward_loro_rf_summary.txt")

MIN_TEST_SAMPLES = 100
MIN_TRAIN_SAMPLES = 5000
# =========================================


def mae(y, yhat):
    return float(mean_absolute_error(np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)))


def impute_with_train_medians(X_train, X_test):
    # Replace inf with nan then impute nan with train medians
    X_train = np.where(np.isfinite(X_train), X_train, np.nan)
    X_test = np.where(np.isfinite(X_test), X_test, np.nan)

    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    idx = np.where(np.isnan(X_train))
    X_train[idx] = np.take(med, idx[1])

    idx = np.where(np.isnan(X_test))
    X_test[idx] = np.take(med, idx[1])

    return X_train, X_test


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = ["Season", "RaceName", "LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["Season"] = df["Season"].astype(int)
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["PrevLapTimeSec"] = pd.to_numeric(df["PrevLapTimeSec"], errors="coerce")
    df["NextLapTimeSec"] = pd.to_numeric(df["NextLapTimeSec"], errors="coerce")
    df = df.dropna(subset=["LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]).copy()

    df["RaceKey"] = df["Season"].astype(str) + "_" + df["RaceName"].astype(str)

    # Load params + feature list
    cfg = joblib.load(PARAMS_PATH)
    FEATURES = cfg["features"]
    BEST_PARAMS = cfg["best_params"]

    # Feature patches (if your dataset doesn't contain them as stored columns)
    if "FuelLapsRemaining" in FEATURES and "FuelLapsRemaining" not in df.columns:
        max_laps = df.groupby("RaceKey")["LapNumber"].transform("max")
        df["FuelLapsRemaining"] = max_laps - df["LapNumber"]

    if "CarPaceIndex" in FEATURES and "CarPaceIndex" not in df.columns:
        race_med = df.groupby("RaceKey")["PrevLapTimeSec"].transform("median")
        df["CarPaceIndex"] = df["PrevLapTimeSec"] / race_med

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing model features: {missing}")

    races_2025 = get_chronological_racekeys_2025()
    races_2025 = [r for r in races_2025 if r in df["RaceKey"].unique()]

    print("\n2025 race order:")
    for r in races_2025:
        print(" ", r)

    results = []

    for i, test_race in enumerate(races_2025):
        print(f"\nEvaluating {test_race} ({i+1}/{len(races_2025)})")

        train_races = (
            df[
                (df["Season"] < SEASON_EVAL) |
                ((df["Season"] == SEASON_EVAL) & (df["RaceKey"].isin(races_2025[:i])))
            ]["RaceKey"]
            .unique()
            .tolist()
        )

        train_df = df[df["RaceKey"].isin(train_races)]
        test_df = df[df["RaceKey"] == test_race]

        if len(test_df) < MIN_TEST_SAMPLES or len(train_df) < MIN_TRAIN_SAMPLES:
            print("  Skipped (insufficient data)")
            continue

        X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
        y_train = (train_df["NextLapTimeSec"] - train_df["PrevLapTimeSec"]).to_numpy(dtype=float)

        X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
        y_test = test_df["NextLapTimeSec"].to_numpy(dtype=float)
        base = test_df["PrevLapTimeSec"].to_numpy(dtype=float)

        X_train, X_test = impute_with_train_medians(X_train, X_test)

        model = RandomForestRegressor(
            **BEST_PARAMS,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)

        pred_delta = model.predict(X_test)
        pred = base + pred_delta

        base_mae = mae(y_test, base)
        model_mae = mae(y_test, pred)
        gain = base_mae - model_mae
        pct_gain = 100.0 * gain / base_mae if base_mae > 1e-9 else np.nan

        print(f"  Baseline: {base_mae:.4f} | RF: {model_mae:.4f} | Gain: {gain:.4f} ({pct_gain:.2f}%)")

        results.append({
            "RaceKey": test_race,
            "n_samples": int(len(test_df)),
            "baseline_mae": float(base_mae),
            "model_mae": float(model_mae),
            "mae_gain": float(gain),
            "pct_gain": float(pct_gain),
        })

    res = pd.DataFrame(results)
    res.to_csv(OUT_CSV, index=False)

    mean_base = res["baseline_mae"].mean()
    mean_model = res["model_mae"].mean()
    mean_gain = mean_base - mean_model
    mean_pct = 100.0 * mean_gain / mean_base if mean_base > 1e-9 else np.nan
    pct_pos = (res["mae_gain"] > 0).mean() * 100.0

    summary = (
        f"Forward LORO evaluation – Season {SEASON_EVAL} (RandomForest)\n"
        f"Races evaluated: {len(res)}\n\n"
        f"Mean Baseline MAE : {mean_base:.6f} s\n"
        f"Mean RF MAE       : {mean_model:.6f} s\n"
        f"Mean Gain         : {mean_gain:.6f} s\n"
        f"Pct Gain          : {mean_pct:.2f}%\n"
        f"% races improved  : {pct_pos:.2f}%\n\n"
        f"Saved per-race table: {OUT_CSV}\n"
    )

    print("\n" + summary)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(summary)

    print("==================== PER-CIRCUIT RESULTS ====================")
    for _, r in res.iterrows():
        print(
            f"{r['RaceKey']:<35} | "
            f"Baseline: {r['baseline_mae']:.4f}s | "
            f"RF: {r['model_mae']:.4f}s | "
            f"Gain: {r['mae_gain']:.4f}s | "
            f"Gain %: {r['pct_gain']:.2f}%"
        )
    print("=============================================================")


if __name__ == "__main__":
    main()
