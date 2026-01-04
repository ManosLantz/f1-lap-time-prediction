import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from train_model import load_and_prep, FEATURES

# === CONFIG ===
TEST_SEASON = 2025

def test_rf():
    print(f"Loading Data to test Random Forest on {TEST_SEASON}...")
    df, features = load_and_prep()
    
    # 1. Split Data (Train on 2022-2024, Test on 2025)
    test_mask = (df['Season'] == TEST_SEASON)
    train_df = df[~test_mask]
    test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
    
    print(f"Training Set: {len(train_df)} laps")
    print(f"Test Set:     {len(test_df)} laps")

    # 2. Train Random Forest
    # n_jobs=-1 uses all CPU cores (RF is parallel, so this is fast)
    print("Training Random Forest (this might take a minute)...")
    rf_model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Don't let trees grow too deep (prevents overfitting)
        min_samples_leaf=5,    # Require 5 laps in a leaf to make a prediction
        n_jobs=-1,             
        random_state=42,
        verbose=1
    )
    rf_model.fit(train_df[features], train_df['NextLapTimeSec'])
    
    # 3. Predict (Pure Physics)
    print("Generating Predictions...")
    preds_rf = rf_model.predict(test_df[features])
    preds_base = test_df['PrevLapTimeSec'].values
    y_true = test_df['NextLapTimeSec'].values
    
    # Safety Clip (prevent negative or crazy times)
    preds_rf = np.maximum(preds_rf, preds_base - 2.0)
    
    # 4. Run Adaptive Ensemble (RF Version)
    # We swap XGBoost for RF in the exact same loop
    print("Running Adaptive Ensemble with Random Forest...")
    adaptive_preds = []
    drivers = test_df['Driver'].unique()
    current_idx = 0
    alpha = 0.5  # Using your best alpha
    
    for driver in drivers:
        n_laps = len(test_df[test_df['Driver'] == driver])
        d_p_model = preds_rf[current_idx : current_idx + n_laps]
        d_p_base = preds_base[current_idx : current_idx + n_laps]
        d_y_true = y_true[current_idx : current_idx + n_laps]
        
        w = 0.5
        e_m = 0.5
        e_b = 0.5
        
        for i in range(n_laps):
            pred = (w * d_p_model[i]) + ((1 - w) * d_p_base[i])
            adaptive_preds.append(pred)
            
            err_m = abs(d_y_true[i] - d_p_model[i])
            err_b = abs(d_y_true[i] - d_p_base[i])
            
            e_m = (alpha * err_m) + ((1 - alpha) * e_m)
            e_b = (alpha * err_b) + ((1 - alpha) * e_b)
            
            if (e_m + e_b) > 0: w = e_b / (e_m + e_b)
            else: w = 0.5
            
        current_idx += n_laps

    # 5. Results
    mae_base = mean_absolute_error(y_true, preds_base)
    mae_rf = mean_absolute_error(y_true, preds_rf)
    mae_adaptive_rf = mean_absolute_error(y_true, adaptive_preds)
    
    print("\n" + "="*50)
    print(f"RANDOM FOREST RESULTS ({TEST_SEASON})")
    print("="*50)
    print(f"{'Model':<25} | {'MAE (s)':<10} | {'vs Base'}")
    print("-" * 50)
    print(f"{'Baseline (PrevLap)':<25} | {mae_base:.4f}s    | -")
    print(f"{'RF (Physics Only)':<25} | {mae_rf:.4f}s    | {((mae_base - mae_rf)/mae_base)*100:.2f}%")
    print(f"{'RF (Adaptive Ensemble)':<25} | {mae_adaptive_rf:.4f}s    | {((mae_base - mae_adaptive_rf)/mae_base)*100:.2f}%")
    print("="*50)
    
    # Comparison Logic
    print("\nCompare this to your XGBoost results:")
    print("If RF is BETTER -> Your data is noisy, and XGBoost was overfitting.")
    print("If RF is WORSE  -> XGBoost successfully learned the complex trends (tyre deg curves).")

if __name__ == "__main__":
    test_rf()