import os
import sys
import numpy as np
import optuna
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_features, clean_data_strict


# ---------------- CONFIG ----------------
N_TRIALS = 30
RANDOM_SEED = 66

HOLDOUT_RACEKEY = None   # optional clean holdout race
K_TEST_RACES = 4         # set to 3 or 4 (leave-k-races-out)
MIN_LAPS_PER_RACE = 50   # skip tiny races to avoid noisy folds
# ---------------------------------------


def mae(y, yhat):
    return float(mean_absolute_error(y, yhat))


def make_leave_k_races_out_folds(race_keys, k, seed):
    """
    Makes non-overlapping folds by shuffling race keys and chunking into groups of size k.
    Each fold uses one chunk as test races, rest as train races.
    """
    rng = np.random.RandomState(seed)
    race_keys = np.array(sorted(race_keys), dtype=object)
    rng.shuffle(race_keys)

    chunks = []
    for i in range(0, len(race_keys), k):
        chunk = race_keys[i:i + k]
        if len(chunk) < k:
            # drop last incomplete chunk (keeps fold sizes consistent)
            break
        chunks.append(list(chunk))

    return chunks


def objective(trial):
    # Keep RF search space reasonable for speed.
    # (Huge n_estimators * many folds is the main killer.)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900),
        "max_depth": trial.suggest_int("max_depth", 8, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_float("max_features", 0.4, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
    }

    fold_maes = []
    baseline_maes = []

    # Train/eval per fold (NOT per race).
    for fold_id, (train_idx, test_idx) in enumerate(FOLDS_IDX):
        # Build train/test arrays
        X_train = X_GLOBAL[train_idx].copy()
        y_train = Y_DELTA_GLOBAL[train_idx]

        X_test = X_GLOBAL[test_idx].copy()
        y_test_true = Y_NEXT_GLOBAL[test_idx]
        base_test = Y_PREV_GLOBAL[test_idx]

        # Impute NaN/inf using train medians (per fold)
        X_train = np.where(np.isfinite(X_train), X_train, np.nan)
        X_test = np.where(np.isfinite(X_test), X_test, np.nan)

        col_medians = np.nanmedian(X_train, axis=0)
        col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)

        tr_nan = np.isnan(X_train)
        if tr_nan.any():
            X_train[tr_nan] = np.take(col_medians, np.where(tr_nan)[1])

        te_nan = np.isnan(X_test)
        if te_nan.any():
            X_test[te_nan] = np.take(col_medians, np.where(te_nan)[1])

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        pred_delta = model.predict(X_test)
        pred_lap = base_test + pred_delta

        fold_mae = mae(y_test_true, pred_lap)
        base_mae = mae(y_test_true, base_test)

        fold_maes.append(fold_mae)
        baseline_maes.append(base_mae)

        # Optuna pruning: kill bad trials early
        trial.report(float(np.mean(fold_maes)), step=fold_id)
        if trial.should_prune():
            raise optuna.TrialPruned()

    avg_model = float(np.mean(fold_maes))
    avg_base = float(np.mean(baseline_maes))

    print(
        f"   [Trial {trial.number:03d}] RF: {avg_model:.4f}s | "
        f"Base: {avg_base:.4f}s | Delta: {avg_base - avg_model:.4f}s | "
        f"Folds: {len(fold_maes)} (leave-{K_TEST_RACES}-races-out)"
    )

    return avg_model


if __name__ == "__main__":
    print("🚀 LOADING FULL DATASET FOR RF TUNING...")
    df_raw, features = load_features()

    print("🧹 Cleaning dataset globally...")
    df = clean_data_strict(df_raw)
    df["RaceKey"] = df["Season"].astype(str) + "_" + df["RaceName"].astype(str)

    # Optional: remove a final clean holdout race from tuning
    all_keys = sorted(df["RaceKey"].unique().tolist())
    if HOLDOUT_RACEKEY is not None:
        all_keys = [r for r in all_keys if r != HOLDOUT_RACEKEY]

    # Drop races with too few rows (so folds aren’t garbage)
    race_counts = df["RaceKey"].value_counts()
    keep_keys = [rk for rk in all_keys if race_counts.get(rk, 0) >= MIN_LAPS_PER_RACE]
    df = df[df["RaceKey"].isin(keep_keys)].copy()

    print(f"📊 Cleaned Dataset: {df.shape}")
    print(f"🏁 Eligible races: {len(keep_keys)}")
    print(f"🔍 Num features: {len(features)}")

    # Precompute global arrays (major speed win)
    X_GLOBAL = df[features].to_numpy(dtype=np.float32)
    Y_PREV_GLOBAL = df["PrevLapTimeSec"].to_numpy(dtype=np.float32)
    Y_NEXT_GLOBAL = df["NextLapTimeSec"].to_numpy(dtype=np.float32)
    Y_DELTA_GLOBAL = (Y_NEXT_GLOBAL - Y_PREV_GLOBAL).astype(np.float32)
    GROUPS_GLOBAL = df["RaceKey"].to_numpy(dtype=object)

    # Build leave-k-races-out folds
    fold_test_races = make_leave_k_races_out_folds(keep_keys, K_TEST_RACES, RANDOM_SEED)

    # Convert folds to index arrays once
    FOLDS_IDX = []
    for test_races in fold_test_races:
        test_mask = np.isin(GROUPS_GLOBAL, test_races)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]

        # basic sanity: skip tiny folds
        if len(test_idx) < 500 or len(train_idx) < 5000:
            continue
        FOLDS_IDX.append((train_idx, test_idx))

    if len(FOLDS_IDX) < 3:
        raise RuntimeError("Not enough valid folds built. Reduce K_TEST_RACES or relax size thresholds.")

    print(f"🧪 CV folds: {len(FOLDS_IDX)} (each fold holds out {K_TEST_RACES} races)")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 60)
    print(f"🏆 BEST RF MAE: {study.best_value:.4f}s")
    print("-" * 60)
    print("Params:", study.best_params)
    print("=" * 60)

    out = {
        "model_type": "RandomForestRegressor",
        "best_params": study.best_params,
        "features": features,
        "holdout_racekey": HOLDOUT_RACEKEY,
        "cv": f"Leave-{K_TEST_RACES}-Races-Out",
        "min_laps_per_race": MIN_LAPS_PER_RACE,
        "n_folds": len(FOLDS_IDX),
        "random_seed": RANDOM_SEED,
    }

    joblib.dump(out, "best_params_rf_leave_k_races_out.pkl")
    joblib.dump(study, "optuna_study_rf_leave_k_races_out.pkl")

    print("💾 Saved: best_params_rf_leave_k_races_out.pkl")
    print("💾 Saved: optuna_study_rf_leave_k_races_out.pkl")
