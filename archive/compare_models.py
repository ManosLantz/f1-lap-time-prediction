import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep
from models import train_physics_model # Imports your "Heavy" XGBoost

# === CONFIGURATION ===
# Winning Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': 42
}

# Adaptive Logic Parameters
ADAPTIVE_ALPHA = 0.5

def run_adaptive_logic(preds_model, preds_base, y_true, alpha=0.5):
    """
    Applies the adaptive smoothing logic to ANY model's predictions.
    """
    adaptive_preds = []
    w = 0.5
    e_m = 0.5
    e_b = 0.5
    
    for i in range(len(preds_model)):
        # Predict
        pred = (w * preds_model[i]) + ((1 - w) * preds_base[i])
        adaptive_preds.append(pred)
        
        # Update Errors
        err_m = abs(y_true[i] - preds_model[i])
        err_b = abs(y_true[i] - preds_base[i])
        
        # Smooth Errors
        e_m = (alpha * err_m) + ((1 - alpha) * e_m)
        e_b = (alpha * err_b) + ((1 - alpha) * e_b)
        
        # Update Weight
        if (e_m + e_b) > 0:
            w = e_b / (e_m + e_b)
        else:
            w = 0.5
            
    return np.array(adaptive_preds)

def compare_all_models():
    print("🚀 STARTING ULTIMATE MODEL SHOWDOWN (XGB vs RF vs Baseline)...")
    df, features = load_and_prep()
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    
    results = {
        'Race': [],
        'Baseline': [],
        'XGB_Pure': [],
        'XGB_Adaptive': [],
        'RF_Pure': [],
        'RF_Adaptive': []
    }
    
    for race in test_races:
        # LORO Split
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # Prepare Targets
        y_true = test_df['NextLapTimeSec'].values
        preds_base = test_df['PrevLapTimeSec'].values
        
        # --- 1. XGBOOST (The Heavyweight) ---
        # Train on Delta
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        xgb_model = train_physics_model(train_df[features], train_df['DeltaTarget'])
        
        preds_delta_xgb = xgb_model.predict(test_df[features])
        preds_xgb_pure = test_df['PrevLapTimeSec'].values + preds_delta_xgb
        preds_xgb_pure = np.maximum(preds_xgb_pure, preds_base - 2.0) # Safety Clip
        
        # Apply Adaptive to XGB
        preds_xgb_adapt = run_adaptive_logic(preds_xgb_pure, preds_base, y_true, alpha=ADAPTIVE_ALPHA)

        # --- 2. RANDOM FOREST (The New Champion) ---
        # Train on Absolute Time (RF prefers this usually, but let's stick to Delta for fair comparison)
        # Actually, your previous RF success likely used Absolute Time? 
        # Let's use Absolute Time for RF as that's standard for it, but Delta is safer.
        # Let's stick to Delta for RF too to be scientifically fair.
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(train_df[features], train_df['DeltaTarget'])
        
        preds_delta_rf = rf_model.predict(test_df[features])
        preds_rf_pure = test_df['PrevLapTimeSec'].values + preds_delta_rf
        preds_rf_pure = np.maximum(preds_rf_pure, preds_base - 2.0)
        
        # Apply Adaptive to RF
        preds_rf_adapt = run_adaptive_logic(preds_rf_pure, preds_base, y_true, alpha=ADAPTIVE_ALPHA)

        # --- 3. CALCULATE METRICS ---
        mae_base = mean_absolute_error(y_true, preds_base)
        mae_xgb = mean_absolute_error(y_true, preds_xgb_pure)
        mae_xgb_ada = mean_absolute_error(y_true, preds_xgb_adapt)
        mae_rf = mean_absolute_error(y_true, preds_rf_pure)
        mae_rf_ada = mean_absolute_error(y_true, preds_rf_adapt)
        
        results['Race'].append(race.replace(" Grand Prix", ""))
        results['Baseline'].append(mae_base)
        results['XGB_Pure'].append(mae_xgb)
        results['XGB_Adaptive'].append(mae_xgb_ada)
        results['RF_Pure'].append(mae_rf)
        results['RF_Adaptive'].append(mae_rf_ada)
        
        print(f"  > {race:<15} | Base: {mae_base:.3f} | XGB: {mae_xgb:.3f} | RF: {mae_rf:.3f}")

    # --- FINAL SUMMARY ---
    df_res = pd.DataFrame(results)
    means = df_res.mean(numeric_only=True)
    
    print("\n" + "="*80)
    print(f"{'STRATEGY':<20} | {'MAE (s)':<10} | {'IMPROVEMENT':<10}")
    print("-" * 80)
    
    base_score = means['Baseline']
    strategies = ['Baseline', 'XGB_Pure', 'XGB_Adaptive', 'RF_Pure', 'RF_Adaptive']
    
    for strat in strategies:
        score = means[strat]
        imp = ((base_score - score) / base_score) * 100
        print(f"{strat:<20} | {score:.4f}s    | {imp:+.2f}%")
    print("="*80)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    melted = df_res.melt(id_vars='Race', var_name='Model', value_name='MAE')
    
    # Custom Palette: Gray for Base, Blues for XGB, Greens for RF
    palette = {
        'Baseline': '#999999',
        'XGB_Pure': '#6baed6',
        'XGB_Adaptive': '#2171b5',
        'RF_Pure': '#74c476',
        'RF_Adaptive': '#238b45'
    }
    
    sns.barplot(data=melted, x='Model', y='MAE', palette=palette, estimator=np.mean, errorbar=None)
    plt.title("Final Model Comparison: Average Error Across 2025 Season")
    plt.ylabel("Mean Absolute Error (s)")
    plt.ylim(0.4, 0.55) # Zoom in to see differences
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("ultimate_comparison.png")
    print("📸 Saved chart to 'ultimate_comparison.png'")

if __name__ == "__main__":
    compare_all_models()