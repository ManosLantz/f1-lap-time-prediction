import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
DATA_PATH = "data/f1_lap_dataset.csv"

MODEL_PATH = "model_v3.pkl"          # optional
META_PATH  = "model_v3_meta.pkl"     # optional

OUT_DIR = "report_artifacts"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

# If you want to focus only on your final features:
# You can leave None and it will use meta["features"] if available.
FORCE_FEATURES = None

# =========================
# HELPERS
# =========================
def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def save_df(df, name):
    path = os.path.join(TAB_DIR, name)
    df.to_csv(path, index=False)
    print("Saved table:", path)

def save_fig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print("Saved figure:", path)

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.nanmean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    # safe MAPE: ignore tiny denominators
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

def r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.nanmean((y_true - y_pred) ** 2)
    ss_tot = np.nanmean((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def qq_plot(residuals, title, fname):
    # Simple QQ plot vs Normal
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    if len(res) < 50:
        return
    res_sorted = np.sort(res)
    n = len(res_sorted)
    # theoretical quantiles from normal
    # Use inverse error function approximation via scipy? not allowed.
    # We'll approximate using numpy and math with a simple normal quantile approximation.
    # Here we use a quick rational approximation for inverse CDF (Acklam).
    def norm_ppf(p):
        # Peter J. Acklam approximation
        # valid for p in (0,1)
        a = [-3.969683028665376e+01,  2.209460984245205e+02,
             -2.759285104469687e+02,  1.383577518672690e+02,
             -3.066479806614716e+01,  2.506628277459239e+00]
        b = [-5.447609879822406e+01,  1.615858368580409e+02,
             -1.556989798598866e+02,  6.680131188771972e+01,
             -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01,
             -2.400758277161838e+00, -2.549732539343734e+00,
              4.374664141464968e+00,  2.938163982698783e+00]
        d = [ 7.784695709041462e-03,  3.224671290700398e-01,
              2.445134137142996e+00,  3.754408661907416e+00]
        plow = 0.02425
        phigh = 1 - plow
        if p < plow:
            q = math.sqrt(-2*math.log(p))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        if p > phigh:
            q = math.sqrt(-2*math.log(1-p))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)

    p = (np.arange(1, n+1) - 0.5) / n
    theo = np.array([norm_ppf(pi) for pi in p], dtype=float)

    plt.figure()
    plt.scatter(theo, res_sorted, s=10)
    # reference line
    m = np.nanmean(res_sorted)
    s = np.nanstd(res_sorted) + 1e-9
    line = m + s*theo
    plt.plot(theo, line)
    plt.title(title)
    plt.xlabel("Theoretical Quantiles (Normal)")
    plt.ylabel("Residual Quantiles")
    save_fig(fname)

def plot_hist(data, title, xlabel, fname, bins=60):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return
    plt.figure()
    plt.hist(x, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    save_fig(fname)

def plot_cdf(data, title, xlabel, fname):
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 50:
        return
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    save_fig(fname)

def boxplot_by_group(df, value_col, group_col, title, fname, top_k=12):
    # show top_k groups by count
    temp = df[[group_col, value_col]].dropna()
    counts = temp[group_col].value_counts().head(top_k).index.tolist()
    temp = temp[temp[group_col].isin(counts)]
    groups = []
    labels = []
    for g in counts:
        vals = safe_numeric(temp.loc[temp[group_col] == g, value_col]).dropna().values
        if len(vals) >= 20:
            groups.append(vals)
            labels.append(str(g))
    if len(groups) < 2:
        return
    plt.figure(figsize=(12, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(value_col)
    save_fig(fname)

def scatter_with_trend(x, y, title, xlabel, ylabel, fname, max_points=25000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if len(x) < 200:
        return
    if len(x) > max_points:
        idx = np.random.RandomState(42).choice(len(x), size=max_points, replace=False)
        x = x[idx]; y = y[idx]
    plt.figure()
    plt.scatter(x, y, s=5)
    # simple binned trend line
    bins = 30
    edges = np.linspace(np.nanmin(x), np.nanmax(x), bins+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    med = []
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i+1])
        if np.sum(m) > 30:
            med.append(np.nanmedian(y[m]))
        else:
            med.append(np.nan)
    plt.plot(centers, np.array(med))
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(fname)

# =========================
# MAIN
# =========================
def main():
    ensure_dirs()

    print("Loading dataset:", DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Basic required cols
    required = ["Season", "RaceName", "Driver", "LapNumber", "PrevLapTimeSec", "NextLapTimeSec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Dataset missing required columns: " + str(missing))

    # Make RaceKey
    df["RaceKey"] = df["Season"].astype(str) + "_" + df["RaceName"].astype(str)

    # Force numeric for core times
    df["PrevLapTimeSec"] = safe_numeric(df["PrevLapTimeSec"])
    df["NextLapTimeSec"] = safe_numeric(df["NextLapTimeSec"])
    
    # --- PATCH: generate missing engineered features if needed ---

    # FuelLapsRemaining (race progress proxy)
    if "FuelLapsRemaining" not in df.columns:
        max_laps = df.groupby("RaceKey")["LapNumber"].transform("max")
        df["FuelLapsRemaining"] = max_laps - df["LapNumber"]

    # CarPaceIndex (lap pace relative to race median)
    if "CarPaceIndex" not in df.columns:
        race_median = df.groupby("RaceKey")["PrevLapTimeSec"].transform("median")
        df["CarPaceIndex"] = df["PrevLapTimeSec"] / race_median

    # Drop rows missing core times
    df = df.dropna(subset=["PrevLapTimeSec", "NextLapTimeSec"]).copy()

    # Derived columns
    df["DeltaTrue"] = df["NextLapTimeSec"] - df["PrevLapTimeSec"]
    df["BaselinePredNext"] = df["PrevLapTimeSec"]
    df["BaselineAbsErr"] = np.abs(df["NextLapTimeSec"] - df["BaselinePredNext"])
    df["BaselineErr"] = df["NextLapTimeSec"] - df["BaselinePredNext"]

    # =========================
    # Load model if available
    # =========================
    have_model = os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)
    model = None
    meta = None

    if have_model:
        print("Loading model:", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        meta = joblib.load(META_PATH)

        # Determine feature list
        features = FORCE_FEATURES
        if features is None:
            features = meta.get("features", None)
        if not features:
            raise ValueError("Model meta does not contain 'features'. Set FORCE_FEATURES in the script.")

        # ensure all features exist
        missing_feat = [c for c in features if c not in df.columns]
        if missing_feat:
            raise ValueError("Dataset missing model features: " + str(missing_feat))

        # numeric matrix
        X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)

        # model predicts delta (in your setup)
        print("Computing model predictions...")
        pred_delta = model.predict(X)
        df["ModelPredDelta"] = pred_delta
        df["ModelPredNext"] = df["PrevLapTimeSec"] + df["ModelPredDelta"]
        df["ModelAbsErr"] = np.abs(df["NextLapTimeSec"] - df["ModelPredNext"])
        df["ModelErr"] = df["NextLapTimeSec"] - df["ModelPredNext"]

    # =========================
    # Global summary tables
    # =========================
    summary = {
        "n_rows": int(len(df)),
        "n_races": int(df["RaceKey"].nunique()),
        "n_drivers": int(df["Driver"].nunique()),
        "prev_mean": float(df["PrevLapTimeSec"].mean()),
        "next_mean": float(df["NextLapTimeSec"].mean()),
        "delta_mean": float(df["DeltaTrue"].mean()),
        "delta_std": float(df["DeltaTrue"].std()),
    }

    # Baseline metrics
    summary["baseline_mae"] = mae(df["NextLapTimeSec"], df["BaselinePredNext"])
    summary["baseline_rmse"] = rmse(df["NextLapTimeSec"], df["BaselinePredNext"])
    summary["baseline_mape_pct"] = mape(df["NextLapTimeSec"], df["BaselinePredNext"])
    summary["baseline_r2"] = r2(df["NextLapTimeSec"], df["BaselinePredNext"])

    if have_model:
        summary["model_mae"] = mae(df["NextLapTimeSec"], df["ModelPredNext"])
        summary["model_rmse"] = rmse(df["NextLapTimeSec"], df["ModelPredNext"])
        summary["model_mape_pct"] = mape(df["NextLapTimeSec"], df["ModelPredNext"])
        summary["model_r2"] = r2(df["NextLapTimeSec"], df["ModelPredNext"])
        summary["mae_improvement_sec"] = summary["baseline_mae"] - summary["model_mae"]

    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved:", os.path.join(OUT_DIR, "summary.json"))
    print(json.dumps(summary, indent=2))

    # Missingness report (top 30)
    miss = df.isna().mean().sort_values(ascending=False).head(30).reset_index()
    miss.columns = ["column", "missing_rate"]
    save_df(miss, "missingness_top30.csv")

    # Per-race metrics
    race_rows = []
    for rk, g in df.groupby("RaceKey"):
        row = {
            "RaceKey": rk,
            "n": int(len(g)),
            "baseline_mae": mae(g["NextLapTimeSec"], g["BaselinePredNext"]),
            "baseline_rmse": rmse(g["NextLapTimeSec"], g["BaselinePredNext"]),
        }
        if have_model:
            row["model_mae"] = mae(g["NextLapTimeSec"], g["ModelPredNext"])
            row["model_rmse"] = rmse(g["NextLapTimeSec"], g["ModelPredNext"])
            row["mae_gain"] = row["baseline_mae"] - row["model_mae"]
        race_rows.append(row)
    race_tbl = pd.DataFrame(race_rows).sort_values(by=("model_mae" if have_model else "baseline_mae"))
    save_df(race_tbl, "per_race_metrics.csv")

    # Per-driver metrics (top drivers by count)
    drv_rows = []
    for d, g in df.groupby("Driver"):
        row = {
            "Driver": d,
            "n": int(len(g)),
            "baseline_mae": mae(g["NextLapTimeSec"], g["BaselinePredNext"]),
        }
        if have_model:
            row["model_mae"] = mae(g["NextLapTimeSec"], g["ModelPredNext"])
            row["mae_gain"] = row["baseline_mae"] - row["model_mae"]
        drv_rows.append(row)
    drv_tbl = pd.DataFrame(drv_rows).sort_values(by=("model_mae" if have_model else "baseline_mae"))
    save_df(drv_tbl, "per_driver_metrics.csv")

    # =========================
    # Figures: Dataset distributions
    # =========================
    plot_hist(df["PrevLapTimeSec"], "Prev Lap Time Distribution", "PrevLapTimeSec (s)", "hist_prev_lap.png")
    plot_hist(df["NextLapTimeSec"], "Next Lap Time Distribution", "NextLapTimeSec (s)", "hist_next_lap.png")
    plot_hist(df["DeltaTrue"], "True Delta Distribution (Next - Prev)", "DeltaTrue (s)", "hist_delta_true.png")
    plot_cdf(df["BaselineAbsErr"], "Baseline Absolute Error CDF", "Abs Error (s)", "cdf_baseline_abs_err.png")

    # Boxplots across top races/drivers
    boxplot_by_group(df, "BaselineAbsErr", "RaceKey",
                     "Baseline Abs Error by Race (top races by samples)",
                     "box_baseline_err_by_race.png", top_k=12)
    boxplot_by_group(df, "BaselineAbsErr", "Driver",
                     "Baseline Abs Error by Driver (top drivers by samples)",
                     "box_baseline_err_by_driver.png", top_k=12)

    # Residual vs baseline lap time
    scatter_with_trend(df["PrevLapTimeSec"], df["BaselineErr"],
                       "Baseline Error vs PrevLapTimeSec",
                       "PrevLapTimeSec (s)", "Error (Next-Prev) (s)",
                       "scatter_baseline_err_vs_prev.png")

    # QQ plot for baseline error
    qq_plot(df["BaselineErr"], "QQ Plot: Baseline Residuals", "qq_baseline_residuals.png")

    # =========================
    # Model vs baseline plots (if model exists)
    # =========================
    if have_model:
        plot_cdf(df["ModelAbsErr"], "Model Absolute Error CDF", "Abs Error (s)", "cdf_model_abs_err.png")
        plot_hist(df["ModelErr"], "Model Residual Distribution", "Residual (s)", "hist_model_residuals.png")
        qq_plot(df["ModelErr"], "QQ Plot: Model Residuals", "qq_model_residuals.png")

        # Improvement distribution
        df["AbsErrGain"] = df["BaselineAbsErr"] - df["ModelAbsErr"]
        plot_hist(df["AbsErrGain"], "Abs Error Gain (Baseline - Model)", "Gain (s)", "hist_abs_err_gain.png")
        plot_cdf(df["AbsErrGain"], "Abs Error Gain CDF (Baseline - Model)", "Gain (s)", "cdf_abs_err_gain.png")

        # Scatter: baseline abs err vs model abs err
        scatter_with_trend(df["BaselineAbsErr"], df["ModelAbsErr"],
                           "Model AbsErr vs Baseline AbsErr",
                           "Baseline AbsErr (s)", "Model AbsErr (s)",
                           "scatter_model_vs_baseline_abs.png")

        # Per-race MAE gains bar plot (top gains)
        top = race_tbl.sort_values("mae_gain", ascending=False).head(20)
        plt.figure(figsize=(12, 6))
        plt.bar(top["RaceKey"], top["mae_gain"])
        plt.xticks(rotation=45, ha="right")
        plt.title("Top 20 Races by MAE Gain (Baseline - Model)")
        plt.ylabel("MAE Gain (s)")
        save_fig("bar_top20_race_mae_gain.png")

        # Boxplot model errors by race/driver
        boxplot_by_group(df, "ModelAbsErr", "RaceKey",
                         "Model Abs Error by Race (top races by samples)",
                         "box_model_err_by_race.png", top_k=12)
        boxplot_by_group(df, "ModelAbsErr", "Driver",
                         "Model Abs Error by Driver (top drivers by samples)",
                         "box_model_err_by_driver.png", top_k=12)

        # Residuals vs important features (if present)
        for col in ["TyreAge", "FuelLapsRemaining", "RaceLapsRemaining", "TyreDegSmooth", "PushIndex",
                    "Sector1Sec", "Sector2Sec", "Sector3Sec", "CarPaceIndex"]:
            if col in df.columns:
                scatter_with_trend(df[col], df["ModelErr"],
                                   f"Model Residual vs {col}",
                                   col, "Residual (s)",
                                   f"scatter_model_resid_vs_{col}.png")

        # Quantile table: how often within thresholds (presentation-friendly)
        thresholds = [0.1, 0.2, 0.3, 0.5, 1.0]
        rows = []
        for t in thresholds:
            rows.append({
                "threshold_sec": t,
                "baseline_pct_within": float(np.mean(df["BaselineAbsErr"] <= t) * 100.0),
                "model_pct_within": float(np.mean(df["ModelAbsErr"] <= t) * 100.0),
            })
        thr_tbl = pd.DataFrame(rows)
        save_df(thr_tbl, "within_thresholds.csv")

    # =========================
    # “Scientific looking” correlation snapshot for selected columns
    # =========================
    # Keep this modest: too big and it becomes noise.
    cols = ["PrevLapTimeSec", "NextLapTimeSec", "DeltaTrue", "BaselineAbsErr"]
    candidates = ["TyreAge", "FuelLapsRemaining", "RaceLapsRemaining", "TyreDegSmooth",
                  "PushIndex", "CarPaceIndex", "Sector1Sec", "Sector2Sec", "Sector3Sec"]
    for c in candidates:
        if c in df.columns:
            cols.append(c)
    if have_model:
        cols += ["ModelAbsErr"]

    corr_df = df[cols].apply(pd.to_numeric, errors="coerce").corr()
    corr_df = corr_df.reset_index().rename(columns={"index": "feature"})
    save_df(corr_df, "correlation_matrix_selected.csv")

    # Heatmap-style plot using imshow (matplotlib only)
    mat = df[cols].apply(pd.to_numeric, errors="coerce").corr().values
    plt.figure(figsize=(10, 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(cols)), cols, rotation=45, ha="right")
    plt.yticks(range(len(cols)), cols)
    plt.title("Correlation Heatmap (Selected Features)")
    save_fig("heatmap_correlation_selected.png")

    print("\nDONE. Check:", OUT_DIR)
    print("Figures:", FIG_DIR)
    print("Tables:", TAB_DIR)

if __name__ == "__main__":
    main()
