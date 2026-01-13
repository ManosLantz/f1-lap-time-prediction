import os
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error

# Optional SciPy. If not installed, we fall back to bootstrap-only.
try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


DATA_PATH = "data/f1_lap_dataset.csv"
MODEL_PATH = "model_v3.pkl"
META_PATH  = "model_v3_meta.pkl"

OUT_CSV = "report_artifacts/tables/significance_per_race.csv"
OUT_TXT = "report_artifacts/significance_summary.txt"


def ensure_dirs():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_TXT), exist_ok=True)


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def bootstrap_ci_mean(x, n_boot=5000, alpha=0.05, seed=42):
    rng = np.random.RandomState(seed)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (np.nan, np.nan, np.nan)
    means = []
    n = len(x)
    for _ in range(n_boot):
        samp = x[rng.randint(0, n, size=n)]
        means.append(np.mean(samp))
    means = np.sort(means)
    lo = means[int((alpha/2) * len(means))]
    hi = means[int((1 - alpha/2) * len(means))]
    return (float(np.mean(x)), float(lo), float(hi))


def cohen_d_paired(diff):
    # diff = baseline_mae - model_mae per race
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if len(diff) < 5:
        return np.nan
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


def main():
    ensure_dirs()

    # Load dataset
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Required columns
    for c in ["Season", "RaceName", "Driver", "LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]:
        if c not in df.columns:
            raise ValueError(f"Dataset missing required column: {c}")

    df["RaceKey"] = df["Season"].astype(str) + "_" + df["RaceName"].astype(str)

    df["PrevLapTimeSec"] = safe_numeric(df["PrevLapTimeSec"])
    df["NextLapTimeSec"] = safe_numeric(df["NextLapTimeSec"])
    df["LapNumber"] = safe_numeric(df["LapNumber"])

    df = df.dropna(subset=["PrevLapTimeSec", "NextLapTimeSec", "LapNumber"]).copy()

    # Load model + meta
    if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
        raise ValueError("Missing model files. Need model_v3.pkl and model_v3_meta.pkl in this folder.")

    model = joblib.load(MODEL_PATH)
    meta  = joblib.load(META_PATH)

    features = meta.get("features", None)
    if not features:
        raise ValueError("model_v3_meta.pkl must contain 'features' list.")

    # Patch missing engineered features if needed (common)
    # FuelLapsRemaining proxy
    if "FuelLapsRemaining" in features and "FuelLapsRemaining" not in df.columns:
        max_laps = df.groupby("RaceKey")["LapNumber"].transform("max")
        df["FuelLapsRemaining"] = max_laps - df["LapNumber"]

    # CarPaceIndex proxy
    if "CarPaceIndex" in features and "CarPaceIndex" not in df.columns:
        race_median = df.groupby("RaceKey")["PrevLapTimeSec"].transform("median")
        df["CarPaceIndex"] = df["PrevLapTimeSec"] / race_median

    # Verify features now exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Still missing model features: {missing}")

    # Build X and predict delta (your setup)
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    X = np.where(np.isfinite(X), X, np.nan)

    pred_delta = model.predict(X)
    pred_next = df["PrevLapTimeSec"].to_numpy(dtype=float) + pred_delta

    df["BaselinePred"] = df["PrevLapTimeSec"]
    df["ModelPred"] = pred_next

    # Per-race MAE (this is the right unit for significance under LORO)
    rows = []
    for rk, g in df.groupby("RaceKey"):
        y = g["NextLapTimeSec"].to_numpy(dtype=float)
        base = g["BaselinePred"].to_numpy(dtype=float)
        mod  = g["ModelPred"].to_numpy(dtype=float)

        base_mae = mean_absolute_error(y, base)
        mod_mae  = mean_absolute_error(y, mod)
        gain = base_mae - mod_mae  # positive = model better

        rows.append({
            "RaceKey": rk,
            "n_samples": int(len(g)),
            "baseline_mae": float(base_mae),
            "model_mae": float(mod_mae),
            "mae_gain": float(gain)
        })

    per_race = pd.DataFrame(rows).sort_values("mae_gain", ascending=False)
    per_race.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)

    gains = per_race["mae_gain"].to_numpy(dtype=float)

    # Summary stats
    mean_gain = float(np.mean(gains))
    median_gain = float(np.median(gains))
    pct_positive = float(np.mean(gains > 0) * 100.0)

    # Bootstrap CI for mean gain
    mean_est, ci_lo, ci_hi = bootstrap_ci_mean(gains)

    # Paired tests across races
    t_p = np.nan
    w_p = np.nan
    if HAVE_SCIPY and len(gains) >= 10:
        # Paired t-test on gains vs 0
        t_stat, t_p = stats.ttest_1samp(gains, popmean=0.0, nan_policy="omit")
        # Wilcoxon signed-rank test (non-parametric)
        # Use zero_method='wilcox' to drop zeros safely
        try:
            w_stat, w_p = stats.wilcoxon(gains, zero_method="wilcox", alternative="greater")
        except Exception:
            w_p = np.nan

    d = cohen_d_paired(gains)

    # Write a clean text report
    lines = []
    lines.append("Significance test: Model vs Baseline (paired by Race, LORO-style)\n")
    lines.append(f"Races evaluated: {len(per_race)}")
    lines.append(f"Mean MAE gain (baseline - model): {mean_gain:.6f} s")
    lines.append(f"Median MAE gain: {median_gain:.6f} s")
    lines.append(f"% races where model beats baseline (gain > 0): {pct_positive:.2f}%")
    lines.append(f"Bootstrap 95% CI for mean gain: [{ci_lo:.6f}, {ci_hi:.6f}] s")
    lines.append(f"Effect size (paired Cohen's d): {d:.3f}\n")

    if HAVE_SCIPY:
        lines.append(f"Paired t-test (H0: mean gain = 0): p = {t_p:.3e}")
        lines.append(f"Wilcoxon signed-rank (H0: median gain <= 0, alt: gain > 0): p = {w_p:.3e}")
    else:
        lines.append("SciPy not installed -> skipping formal p-values (t-test/Wilcoxon).")
        lines.append("Bootstrap CI is still valid and reportable.")

    report = "\n".join(lines)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print("\nSaved:", OUT_TXT)


if __name__ == "__main__":
    main()
