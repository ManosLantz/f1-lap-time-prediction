# tune_v3_full_history.py
# Ultra-strict tuning (LORO) for clean-green + no-rain dataset
# - Loads data via data_loader.load_features()
# - Leave-One-Race-Out evaluation
# - Race-aware validation split inside training races (deterministic, stable hash)
# - Early stopping
# - Saves best params to best_params_v3_ultra_fidelity.pkl

import os
import sys
import hashlib
import numpy as np
import pandas as pd
import optuna
import joblib
import xgboost as xgb

from sklearn.metrics import mean_absolute_error

# Ensure local imports work when running as a script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_features  # uses your strict cleaning rules


HOLDOUT_RACEKEY = "2025_British Grand Prix"  # keep as final untouched holdout (if present)
N_TRIALS = 100

RANDOM_SEED = 42


def stable_hash_0_999(s: str) -> int:
    """Stable hash across runs (unlike Python's built-in hash())."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 1000


def objective(trial):
    # Optuna-sampled params
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 3500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),

        # speed + reproducibility
        "tree_method": "hist",
        "max_bin": 256,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbosity": 0,

        # early stopping (works when passed in constructor for many xgboost versions)
        "early_stopping_rounds": 50,
        "eval_metric": "mae",
    }

    fold_maes = []
    baseline_maes = []

    # LORO across races
    for race_key in ALL_RACES:
        test_mask = (df_global["RaceKey"] == race_key)
        test_df = df_global[test_mask]
        train_df = df_global[~test_mask]

        # skip tiny races
        if len(test_df) < 5:
            continue

        # deterministic race-aware validation split INSIDE training races
        unique_train_races = train_df["RaceKey"].unique()
        if len(unique_train_races) < 5:
            continue

        n_val = max(1, int(0.10 * len(unique_train_races)))

        fold_seed = RANDOM_SEED + trial.number * 1000 + stable_hash_0_999(race_key)
        rng = np.random.RandomState(fold_seed)

        val_races = rng.choice(unique_train_races, size=n_val, replace=False)
        val_mask = train_df["RaceKey"].isin(val_races)

        train_sub = train_df[~val_mask]
        val_sub = train_df[val_mask]

        # guards against unstable training
        if len(train_sub) < 150:
            continue
        if len(val_sub) < 30:
            continue

        # Build X/y (predict delta)
        X_train = train_sub[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y_train = (train_sub["NextLapTimeSec"] - train_sub["PrevLapTimeSec"]).to_numpy(dtype=np.float32)

        X_val = val_sub[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y_val = (val_sub["NextLapTimeSec"] - val_sub["PrevLapTimeSec"]).to_numpy(dtype=np.float32)

        X_test = test_df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        y_test_true = test_df["NextLapTimeSec"].to_numpy(dtype=np.float32)
        base_test = test_df["PrevLapTimeSec"].to_numpy(dtype=np.float32)

        # Replace inf/-inf with NaN
        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        X_val   = np.where(np.isfinite(X_val),   X_val,   np.nan)
        X_test  = np.where(np.isfinite(X_test),  X_test,  np.nan)

        model = xgb.XGBRegressor(**params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        preds_diff = model.predict(X_test)
        preds_lap = base_test + preds_diff

        mae = mean_absolute_error(y_test_true, preds_lap)
        fold_maes.append(mae)

        baseline_mae = mean_absolute_error(y_test_true, base_test)
        baseline_maes.append(baseline_mae)

    # Stability enforcement (don’t let Optuna win by evaluating 3 easy races)
    # With strict filtering, you may have fewer races, so keep this realistic.
    min_required = max(10, int(len(ALL_RACES) * 0.70))
    if len(fold_maes) < min_required:
        return 99.0

    avg_model_mae = float(np.mean(fold_maes))
    avg_base_mae = float(np.mean(baseline_maes))

    print(
        f"   [Trial {trial.number:03d}] "
        f"Model: {avg_model_mae:.4f}s | "
        f"Baseline: {avg_base_mae:.4f}s | "
        f"Delta: {avg_base_mae - avg_model_mae:.4f}s | "
        f"Folds: {len(fold_maes)}/{len(ALL_RACES)}"
    )

    return avg_model_mae


if __name__ == "__main__":
    print("🚀 LOADING DATASET (STRICT CLEAN GREEN + NO-RAIN RACES)...")
    df_global, features = load_features()

    # Sanity checks
    required_cols = ["Season", "RaceName", "Driver", "PrevLapTimeSec", "NextLapTimeSec"]
    missing = [c for c in required_cols if c not in df_global.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # RaceKey (deterministic ID per race)
    if "RaceKey" not in df_global.columns:
        df_global["RaceKey"] = df_global["Season"].astype(str) + "_" + df_global["RaceName"].astype(str)
    else:
        # ensure string type
        df_global["RaceKey"] = df_global["RaceKey"].astype(str)

    # Exclude holdout race (if present) to keep it as a clean final test
    all_racekeys = sorted(df_global["RaceKey"].unique().tolist())
    if HOLDOUT_RACEKEY in all_racekeys:
        ALL_RACES = sorted([r for r in all_racekeys if r != HOLDOUT_RACEKEY])
        print(f"🧊 Holdout race excluded from tuning: {HOLDOUT_RACEKEY}")
    else:
        ALL_RACES = all_racekeys
        print(f"⚠️ Holdout race not found in dataset: {HOLDOUT_RACEKEY}")
        print("   Tuning will use all available races. Consider choosing another holdout for final testing.")

    print(f"📊 Dataset shape: {df_global.shape}")
    print(f"🏁 LORO races: {len(ALL_RACES)}")
    print(f"🔍 Num features: {len(features)}")

    # Optuna study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 70)
    print(f"🏆 BEST LORO MAE: {study.best_value:.4f}s")
    print("-" * 70)
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    joblib.dump(study.best_params, "best_params_v3_ultra_fidelity.pkl")
    joblib.dump(study, "optuna_study_v3_ultra_fidelity.pkl")
    print("💾 Saved best params to 'best_params_v3_ultra_fidelity.pkl'")
    print("💾 Saved full study to 'optuna_study_v3_ultra_fidelity.pkl'")
