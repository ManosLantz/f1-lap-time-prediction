import pandas as pd
import numpy as np
import xgboost as xgb
from train_model import load_and_prep, FEATURES

def run_sensitivity_analysis():
    print("Loading data for Sensitivity Analysis...")
    df, features = load_and_prep()
    
    # 1. Train the Model ONCE (No need to retrain for every alpha)
    print("Training Physics Model (Base)...")
    test_mask = (df['Season'] == 2025)
    train_df = df[~test_mask]
    test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
    
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6, 
        subsample=0.7, colsample_bytree=0.8, min_child_weight=3, 
        reg_lambda=5.0, n_jobs=-1
    )
    model.fit(train_df[features], train_df['NextLapTimeSec'])
    
    # Get Raw Predictions
    preds_model = model.predict(test_df[features])
    preds_base = test_df['PrevLapTimeSec'].values
    y_true = test_df['NextLapTimeSec'].values
    # Safety Clip
    preds_model = np.maximum(preds_model, preds_base - 2.0)
    
    # 2. Test Different Alphas
    alphas_to_test = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.7, 0.9]
    results = []
    
    print("\nRunning Adaptive Loop for different Alphas...")
    
    for alpha in alphas_to_test:
        adaptive_preds = []
        drivers = test_df['Driver'].unique()
        current_idx = 0
        
        for driver in drivers:
            n_laps = len(test_df[test_df['Driver'] == driver])
            d_p_model = preds_model[current_idx : current_idx + n_laps]
            d_p_base = preds_base[current_idx : current_idx + n_laps]
            d_y_true = y_true[current_idx : current_idx + n_laps]
            
            # Reset Weights per driver
            w = 0.5
            e_m = 0.5
            e_b = 0.5
            
            for i in range(n_laps):
                # Predict
                pred = (w * d_p_model[i]) + ((1 - w) * d_p_base[i])
                adaptive_preds.append(pred)
                
                # Update with current alpha
                err_m = abs(d_y_true[i] - d_p_model[i])
                err_b = abs(d_y_true[i] - d_p_base[i])
                
                # The Alpha is used here
                e_m = (alpha * err_m) + ((1 - alpha) * e_m)
                e_b = (alpha * err_b) + ((1 - alpha) * e_b)
                
                if (e_m + e_b) > 0: w = e_b / (e_m + e_b)
                else: w = 0.5
            
            current_idx += n_laps
            
        mae = np.mean(np.abs(y_true - np.array(adaptive_preds)))
        results.append({'Alpha': alpha, 'MAE': mae})
        print(f"Alpha: {alpha} -> MAE: {mae:.4f}s")

    # 3. Print Summary
    print("\n" + "="*40)
    print(f"{'ALPHA SENSITIVITY':^40}")
    print("="*40)
    print(f"{'Alpha':<10} | {'MAE (s)':<15} | {'Behavior'}")
    print("-" * 40)
    for r in results:
        behavior = ""
        if r['Alpha'] <= 0.05: behavior = "Extremely Sluggish"
        elif r['Alpha'] == 0.1: behavior = "Sluggish (Slow to adapt)"
        elif r['Alpha'] == 0.15: behavior = "Somewhat Sluggish"
        elif r['Alpha'] == 0.2: behavior = "Somewhat Sluggish"
        elif r['Alpha'] == 0.25: behavior = "Moderate"
        elif r['Alpha'] == 0.3: behavior = "Balanced"
        elif r['Alpha'] >= 0.7: behavior = "Jittery (Over-reactive)"
        
        print(f"{r['Alpha']:<10} | {r['MAE']:.4f}s        | {behavior}")
    print("="*40)

if __name__ == "__main__":
    run_sensitivity_analysis()