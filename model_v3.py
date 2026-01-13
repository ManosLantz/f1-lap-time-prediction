# model_v3.py
# Train final model on 2022–2024 and save it for reuse.

import os
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_features

BEST_PARAMS_PATH = "best_params_v3_ultra_fidelity.pkl"
MODEL_OUT_PATH = "model_v3.pkl"
META_OUT_PATH = "model_v3_meta.pkl"
RANDOM_SEED = 42


def main():
    print("🏁 Loading cleaned dataset...")
    df, features = load_features()

    required = ["Season", "RaceName", "PrevLapTimeSec", "NextLapTimeSec"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Train only on seasons < 2025
    train_df = df[df["Season"] < 2025].copy()
    if train_df.empty:
        raise RuntimeError("Training set is empty (Season < 2025).")

    train_df = train_df.reset_index(drop=True)

    print(f"📊 Training laps: {len(train_df)}")
    print(f"🔍 Features used ({len(features)}): {features}")

    print("📦 Loading best hyperparameters...")
    best_params = joblib.load(BEST_PARAMS_PATH)

    # Ensure stable defaults (in case your params file only contains tuned values)
    best_params.update({
        "tree_method": "hist",
        "eval_metric": "mae",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0
    })

    # Prepare training arrays
    X_train = train_df[features].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    y_train = (train_df["NextLapTimeSec"] - train_df["PrevLapTimeSec"]).to_numpy(np.float32)

    X_train = np.where(np.isfinite(X_train), X_train, np.nan)

    print("🚀 Training model...")
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, verbose=False)

    print(f"💾 Saving model to: {MODEL_OUT_PATH}")
    joblib.dump(model, MODEL_OUT_PATH)

    meta = {
        "features": features,
        "train_seasons": sorted(train_df["Season"].unique().tolist()),
        "best_params": best_params
    }

    print(f"💾 Saving metadata to: {META_OUT_PATH}")
    joblib.dump(meta, META_OUT_PATH)

    print("✅ Done.")


if __name__ == "__main__":
    main()
