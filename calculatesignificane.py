import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
from train_model import load_and_prep, FEATURES

def calculate_p_values():
    print("Loading data for Statistical Significance Test...")
    df, features = load_and_prep()
    
    # Filter for Test Set (2025)
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    
    # Store per-race MAEs (samples for the t-test)
    mae_baseline = []
    mae_physics = []
    mae_adaptive = []
    
    print(f"Evaluating {len(test_races)} races to build error distribution...")
    
    for race in test_races:
        # 1. Prepare Race Data
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask]
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # 2. Train Model
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6, 
            subsample=0.7, colsample_bytree=0.8, min_child_weight=3, 
            reg_lambda=5.0, n_jobs=-1
        )
        model.fit(train_df[features], train_df['NextLapTimeSec'])
        
        # 3. Generate Predictions
        preds_model = model.predict(test_df[features])
        if hasattr(preds_model, "values"): preds_model = preds_model.values
        
        preds_base = test_df['PrevLapTimeSec'].values
        y_true = test_df['NextLapTimeSec'].values
        
        # Safety Clip
        preds_model = np.maximum(preds_model, preds_base - 2.0)
        
        # 4. Adaptive Loop
        adaptive_preds = []
        drivers = test_df['Driver'].unique()
        current_idx = 0
        
        for driver in drivers:
            n_laps = len(test_df[test_df['Driver'] == driver])
            d_p_model = preds_model[current_idx : current_idx + n_laps]
            d_p_base = preds_base[current_idx : current_idx + n_laps]
            d_y_true = y_true[current_idx : current_idx + n_laps]
            
            # Adaptive logic
            w = 0.5
            alpha = 0.3
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
            
        # 5. Calculate Mean Absolute Error for this Race
        mae_base_race = np.mean(np.abs(y_true - preds_base))
        mae_phys_race = np.mean(np.abs(y_true - preds_model))
        mae_adap_race = np.mean(np.abs(y_true - np.array(adaptive_preds)))
        
        mae_baseline.append(mae_base_race)
        mae_physics.append(mae_phys_race)
        mae_adaptive.append(mae_adap_race)

    # --- STATISTICAL TESTS ---
    print("\n" + "="*60)
    print(f"{'STATISTICAL SIGNIFICANCE REPORT (Paired T-Test)':^60}")
    print("="*60)
    
    # Test 1: Adaptive vs Baseline
    t_stat, p_val = stats.ttest_rel(mae_baseline, mae_adaptive)
    significant = "YES" if p_val < 0.05 else "NO"
    
    print(f"\n1. Adaptive vs Baseline")
    print(f"   Mean Improvement: {(np.mean(mae_baseline) - np.mean(mae_adaptive)):.4f}s")
    print(f"   T-Statistic:      {t_stat:.4f}")
    print(f"   P-Value:          {p_val:.5f}")
    print(f"   Significant?      {significant} (at p < 0.05)")

    # Test 2: Adaptive vs Physics Model
    t_stat_2, p_val_2 = stats.ttest_rel(mae_physics, mae_adaptive)
    significant_2 = "YES" if p_val_2 < 0.05 else "NO"
    
    print(f"\n2. Adaptive vs Physics Model")
    print(f"   Mean Improvement: {(np.mean(mae_physics) - np.mean(mae_adaptive)):.4f}s")
    print(f"   T-Statistic:      {t_stat_2:.4f}")
    print(f"   P-Value:          {p_val_2:.5f}")
    print(f"   Significant?      {significant_2} (at p < 0.05)")
    print("="*60)
    
    # Interpretation Helper
    if p_val < 0.05:
        print("\nCONCLUSION: The improvement is statistically significant.")
        print("You can reject the Null Hypothesis.")
    else:
        print("\nCONCLUSION: The improvement is NOT statistically significant.")
        print("You cannot reject the Null Hypothesis (results might be chance).")

if __name__ == "__main__":
    calculate_p_values()