
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

# --- PARAMETERS ---
# 1. Default (Previous Best)
XGB_DEFAULT = {
    'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.05, 'n_jobs': -1, 'random_state': 42
}

# 2. Tuned (From fast_tune_causal.py)
XGB_TUNED = {
    'colsample_bytree': 0.92, 
    'learning_rate': 0.025, 
    'max_depth': 9, 
    'n_estimators': 108, 
    'reg_alpha': 0.77, 
    'reg_lambda': 0.20, 
    'subsample': 0.60,
    'n_jobs': -1, 'random_state': 42
}

RF_TUNED = {
    'max_depth': 30, 
    'max_features': 'sqrt', 
    'min_samples_leaf': 1, 
    'min_samples_split': 2, # Standard for leaf=1
    'n_estimators': 152, # Keep same tree count
    'n_jobs': 1, 'random_state': 42
}

def combine_predictions(p1, p2, y_true, alpha=0.5):
    # Simple adaptive-like (or just weighted average)
    # Since we lack the complex structure here, let's just do 50/50 static
    # Or implement the EWMA logic quickly?
    # Optimized Alpha from tune_ensemble_rf.py = 0.2 (20% XGB, 80% RF)
    return (0.2 * p1) + (0.8 * p2)

def run_final_comparison():
    print("🚀 STARTING FINAL CAUSAL COMPARISON...")
    df_global, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    races_2025 = df_global[
        (df_global['Season'] == 2025) & 
        (df_global['RaceName'] != 'British Grand Prix')
    ]['RaceName'].unique()
    
    results = []
    
    for race in races_2025:
        test_mask = (df_global['RaceName'] == race) & (df_global['Season'] == 2025)
        if test_mask.sum() < 15: continue
        
        train_df = df_global[~test_mask]
        test_df = df_global[test_mask].sort_values(['Driver', 'LapNumber'])
        
        X_train = train_df[CAUSAL_FEATURES]
        y_train = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        
        X_test = test_df[CAUSAL_FEATURES]
        y_test_true = test_df['NextLapTimeSec'].values
        base_test = test_df['PrevLapTimeSec'].values
        
        # 1. Baseline
        mae_base = mean_absolute_error(y_test_true, base_test)
        
        # 2. XGB Default
        m_xgb_def = xgb.XGBRegressor(**XGB_DEFAULT)
        m_xgb_def.fit(X_train, y_train)
        pred_xgb_def = base_test + m_xgb_def.predict(X_test)
        mae_xgb_def = mean_absolute_error(y_test_true, pred_xgb_def)
        
        # 3. XGB Tuned
        m_xgb_tun = xgb.XGBRegressor(**XGB_TUNED)
        m_xgb_tun.fit(X_train, y_train)
        pred_xgb_tun = base_test + m_xgb_tun.predict(X_test)
        mae_xgb_tun = mean_absolute_error(y_test_true, pred_xgb_tun)
        
        # 4. RF Tuned
        m_rf_tun = RandomForestRegressor(**RF_TUNED)
        m_rf_tun.fit(X_train, y_train)
        pred_rf_tun = base_test + m_rf_tun.predict(X_test)
        mae_rf_tun = mean_absolute_error(y_test_true, pred_rf_tun)
        
        # 5. Hybrid (Tuned XGB + Tuned RF)
        # Optimized Alpha: 0.2
        pred_hybrid = (0.2 * pred_xgb_tun) + (0.8 * pred_rf_tun)
        mae_hybrid = mean_absolute_error(y_test_true, pred_hybrid)
        
        print(f"  🏁 {race[:15]:<15}: Base={mae_base:.4f} | XGB_Def={mae_xgb_def:.4f} | XGB_Trn={mae_xgb_tun:.4f} | RF_Trn={mae_rf_tun:.4f} | Hybrid={mae_hybrid:.4f}")
        
        results.append({
            'Race': race,
            'Base': mae_base,
            'XGB_Def': mae_xgb_def,
            'XGB_Trn': mae_xgb_tun,
            'RF_Trn': mae_rf_tun,
            'Hybrid': mae_hybrid
        })
        
    df_res = pd.DataFrame(results)
    print("\n" + "="*80)
    print(f"{'MODEL':<20} | {'MEAN MAE':<10} | {'WIN RATE %':<10}")
    print("-" * 80)
    
    means = df_res.mean(numeric_only=True)
    # Count wins
    wins = df_res[['XGB_Def', 'XGB_Trn', 'RF_Trn', 'Hybrid']].idxmin(axis=1).value_counts()
    
    for col in ['XGB_Def', 'XGB_Trn', 'RF_Trn', 'Hybrid']:
        mu = means[col]
        win_pct = (wins.get(col, 0) / len(df_res)) * 100
        print(f"{col:<20} | {mu:.4f}s    | {win_pct:.1f}%")
        
    print("="*80)
    print(f"Baseline MAE: {means['Base']:.4f}s")
    
    # Check Verdict
    best_model = means[['XGB_Def', 'XGB_Trn', 'RF_Trn', 'Hybrid']].idxmin()
    print(f"🏆 OVERALL WINNER: {best_model} ({means[best_model]:.4f}s)")

if __name__ == "__main__":
    try:
        run_final_comparison()
    except Exception as e:
        print(e)
