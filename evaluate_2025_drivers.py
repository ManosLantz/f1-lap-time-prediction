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
META_PATH = "model_v3_meta.pkl"   # contains features + best_params
SEASON_EVAL = 2025

# Driver-level filters (to avoid junk)
MIN_LAPS_PER_DRIVER_IN_RACE = 30   # require at least N samples for a driver in a test race
MIN_TOTAL_RACES_PER_DRIVER = 5     # require at least N races overall to report driver summary

OUT_DIR = "report_artifacts"
OUT_DRIVER_CSV = os.path.join(OUT_DIR, "per_driver_2025_forward_loro.csv")
OUT_DRIVER_TXT = os.path.join(OUT_DIR, "eval_2025_forward_loro_by_driver_summary.txt")

OUT_DRIVER_RACE_CSV = os.path.join(OUT_DIR, "per_driver_per_race_2025_forward_loro.csv")
# =========================================


def mae(y, yhat):
    return float(mean_absolute_error(np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---------------- Load data ----------------
    df = pd.read_csv(DATA_PATH, low_memory=False)

    required = ["Season", "RaceName", "Driver", "LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    df["Season"] = df["Season"].astype(int)
    df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
    df["PrevLapTimeSec"] = pd.to_numeric(df["PrevLapTimeSec"], errors="coerce")
    df["NextLapTimeSec"] = pd.to_numeric(df["NextLapTimeSec"], errors="coerce")

    df = df.dropna(subset=["Driver", "LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]).copy()
    df["Driver"] = df["Driver"].astype(str)

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

    # ---------------- Chronological 2025 races ----------------
    races_2025 = get_chronological_racekeys_2025()
    races_2025 = [r for r in races_2025 if r in df["RaceKey"].unique()]

    if len(races_2025) == 0:
        raise ValueError("No 2025 races found in the dataset.")

    print("2025 race order:")
    for r in races_2025:
        print(" ", r)

    # Will store per-driver-per-race metrics
    driver_race_rows = []

    # ---------------- Forward LORO by race, evaluate per-driver ----------------
    for i, test_race in enumerate(races_2025):
        print(f"\nEvaluating race {test_race} ({i+1}/{len(races_2025)})")

        # Train on all previous data (2022-2024 + earlier 2025 races)
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

        if len(test_df) < 200 or len(train_df) < 5000:
            print("  Skipped (insufficient data)")
            continue

        X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
        y_train = (train_df["NextLapTimeSec"] - train_df["PrevLapTimeSec"]).to_numpy(dtype=float)

        X_test = test_df[FEATURES].to_numpy(dtype=np.float32)
        y_test = test_df["NextLapTimeSec"].to_numpy(dtype=float)
        base = test_df["PrevLapTimeSec"].to_numpy(dtype=float)

        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)

        model = xgb.XGBRegressor(**BEST_PARAMS)
        model.fit(X_train, y_train, verbose=False)

        pred_delta = model.predict(X_test)
        pred = base + pred_delta

        # Attach predictions to test_df for grouping
        tmp = test_df[["Driver"]].copy()
        tmp["y_true"] = y_test
        tmp["baseline"] = base
        tmp["model"] = pred

        # Per-driver MAE within this test race
        for drv, g in tmp.groupby("Driver"):
            if len(g) < MIN_LAPS_PER_DRIVER_IN_RACE:
                continue

            base_mae = mae(g["y_true"].to_numpy(), g["baseline"].to_numpy())
            model_mae = mae(g["y_true"].to_numpy(), g["model"].to_numpy())
            gain = base_mae - model_mae
            pct_gain = 100.0 * gain / base_mae if base_mae > 1e-9 else np.nan

            driver_race_rows.append({
                "RaceKey": test_race,
                "Driver": drv,
                "n_laps": int(len(g)),
                "baseline_mae": float(base_mae),
                "model_mae": float(model_mae),
                "mae_gain": float(gain),
                "pct_gain": float(pct_gain)
            })

        # Race-level quick progress print (overall, not per-driver)
        overall_base_mae = mae(y_test, base)
        overall_model_mae = mae(y_test, pred)
        print(f"  Overall Baseline: {overall_base_mae:.4f} | Overall Model: {overall_model_mae:.4f} | Gain: {(overall_base_mae - overall_model_mae):.4f}")

    if len(driver_race_rows) == 0:
        raise ValueError("No driver-level results produced. Try lowering MIN_LAPS_PER_DRIVER_IN_RACE.")

    driver_race = pd.DataFrame(driver_race_rows)
    driver_race.to_csv(OUT_DRIVER_RACE_CSV, index=False)
    print(f"\nSaved per-driver-per-race table: {OUT_DRIVER_RACE_CSV}")

    # ---------------- Aggregate across races per driver (race-level averaging) ----------------
    # Each (driver, race) entry is one data point -> no lap pooling leakage.
    rows = []
    for drv, g in driver_race.groupby("Driver"):
        n_races = g["RaceKey"].nunique()
        if n_races < MIN_TOTAL_RACES_PER_DRIVER:
            continue

        mean_base = g["baseline_mae"].mean()
        mean_model = g["model_mae"].mean()
        mean_gain = mean_base - mean_model
        mean_pct = 100.0 * mean_gain / mean_base if mean_base > 1e-9 else np.nan

        rows.append({
            "Driver": drv,
            "races_used": int(n_races),
            "laps_used": int(g["n_laps"].sum()),
            "mean_baseline_mae": float(mean_base),
            "mean_model_mae": float(mean_model),
            "mean_gain": float(mean_gain),
            "mean_pct_gain": float(mean_pct)
        })

    driver_summary = pd.DataFrame(rows).sort_values("mean_gain", ascending=False)
    driver_summary.to_csv(OUT_DRIVER_CSV, index=False)

    # ---------------- Print results ----------------
    print("\n==================== DRIVER SUMMARY (2025 forward LORO) ====================")
    for _, r in driver_summary.iterrows():
        print(
            f"{r['Driver']:<8} | races: {int(r['races_used']):>2} | laps: {int(r['laps_used']):>5} | "
            f"Base: {r['mean_baseline_mae']:.4f}s | Model: {r['mean_model_mae']:.4f}s | "
            f"Gain: {r['mean_gain']:.4f}s | Gain%: {r['mean_pct_gain']:.2f}%"
        )
    print("============================================================================")

    # Overall across drivers (unweighted mean across drivers)
    overall_base = driver_summary["mean_baseline_mae"].mean()
    overall_model = driver_summary["mean_model_mae"].mean()
    overall_gain = overall_base - overall_model
    overall_pct = 100.0 * overall_gain / overall_base if overall_base > 1e-9 else np.nan

    summary_txt = (
        f"Driver-level forward LORO evaluation – Season {SEASON_EVAL}\n"
        f"Drivers reported: {len(driver_summary)}\n"
        f"Min laps/driver/race: {MIN_LAPS_PER_DRIVER_IN_RACE}\n"
        f"Min races/driver: {MIN_TOTAL_RACES_PER_DRIVER}\n\n"
        f"Unweighted mean across drivers:\n"
        f"Mean Baseline MAE: {overall_base:.6f} s\n"
        f"Mean Model MAE:    {overall_model:.6f} s\n"
        f"Mean Gain:         {overall_gain:.6f} s\n"
        f"Pct Gain:          {overall_pct:.2f}%\n\n"
        f"Saved driver summary: {OUT_DRIVER_CSV}\n"
        f"Saved per-driver-per-race: {OUT_DRIVER_RACE_CSV}\n"
    )

    with open(OUT_DRIVER_TXT, "w", encoding="utf-8") as f:
        f.write(summary_txt)

    print("\n" + summary_txt)
    print(f"Saved summary to: {OUT_DRIVER_TXT}")


if __name__ == "__main__":
    main()
