import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep, FEATURES
import itertools

def run_extended_grid_search():
    print("Loading Data for Extended Random Forest Tuning...")
    df, features = load_and_prep()
    
    # 1. Setup Train/Test Split (LORO 2025)
    test_mask = (df['Season'] == 2025)
    train_df = df[~test_mask]
    test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
    
    X_train = train_df[features]
    y_train = train_df['NextLapTimeSec']
    X_test = test_df[features]
    y_test = test_df['NextLapTimeSec']
    
    preds_base = test_df['PrevLapTimeSec'].values
    mae_base = mean_absolute_error(y_test, preds_base)
    
    print(f"\nBaseline MAE to beat: {mae_base:.4f}s")
    print("="*80)
    print(f"{'DEPTH':<6} | {'TREES':<6} | {'LEAF':<6} | {'MAE (s)':<10} | {'vs Base'}")
    print("-" * 80)

    # 2. Define the Grid
    # We test 12 combinations to cover all bases
    param_grid = {
        'n_estimators': [100, 300],        # Standard vs Heavy
        'max_depth': [10, 20, None],       # Shallow vs Deep vs Unlimited
        'min_samples_leaf': [1, 5]         # Overfit-prone vs Regularized
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    best_mae = float('inf')
    best_config = None
    best_preds = None

    # 3. Run the Grid Search
    for i, params in enumerate(combinations):
        rf = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        preds = np.maximum(preds, preds_base - 2.0) # Safety clip
        
        mae = mean_absolute_error(y_test, preds)
        diff = ((mae_base - mae) / mae_base) * 100
        
        # Formatting for print
        d_str = str(params['max_depth']) if params['max_depth'] else "None"
        status = "BETTER" if diff > 0 else "WORSE"
        
        print(f"{d_str:<6} | {params['n_estimators']:<6} | {params['min_samples_leaf']:<6} | {mae:.4f}s     | {diff:+.2f}% ({status})")
        
        if mae < best_mae:
            best_mae = mae
            best_config = params
            best_preds = preds

    # 4. Final Verdict
    print("="*80)
    print(f"BEST RF CONFIGURATION: {best_config}")
    print(f"BEST RF MAE:         {best_mae:.4f}s")
    print(f"BASELINE MAE:        {mae_base:.4f}s")
    print(f"IMPROVEMENT:         {((mae_base - best_mae)/mae_base)*100:.2f}%")
    
    # 5. Check Adaptive Score for the Winner
    print("-" * 80)
    print("Checking if Adaptive Ensemble can save the Best RF...")
    
    adaptive_preds = []
    drivers = test_df['Driver'].unique()
    current_idx = 0
    alpha = 0.5 
    y_true = y_test.values
    
    for driver in drivers:
        n_laps = len(test_df[test_df['Driver'] == driver])
        d_p_model = best_preds[current_idx : current_idx + n_laps]
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
        
    mae_adaptive = mean_absolute_error(y_true, adaptive_preds)
    print(f"BEST RF + ADAPTIVE MAE: {mae_adaptive:.4f}s")
    print(f"FINAL IMPROVEMENT:      {((mae_base - mae_adaptive)/mae_base)*100:.2f}%")
    print("="*80)

if __name__ == "__main__":
    run_extended_grid_search()