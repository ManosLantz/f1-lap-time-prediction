
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CONFIGURATIONS ---

# 1. THE DEFENDING CHAMPION (Solo XGB)
# Tuned via fast_tune_causal.py
XGB_SOLO_PARAMS = {
    'colsample_bytree': 0.92, 
    'learning_rate': 0.025, 
    'max_depth': 9, 
    'n_estimators': 108, 
    'reg_alpha': 0.77, 
    'reg_lambda': 0.20, 
    'subsample': 0.60,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

# 2. THE CHALLENGER (Mystery Hybrid)
# Found by run_optuna_hybrid.py (Step 1523 Log)
HYBRID_XGB_PARAMS = {
    'n_estimators': 288,
    'max_depth': 8,
    'learning_rate': 0.06434928666576482,
    'subsample': 0.6441069553228377,
    'colsample_bytree': 0.9538271902136684,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

HYBRID_RF_PARAMS = {
    'n_estimators': 68,
    'max_depth': 7,
    'min_samples_leaf': 9,
    'n_jobs': 1, 'random_state': 42
}
HYBRID_ALPHA = 0.6

# 3. THE SOLO RF (Benchmark)
RF_SOLO_PARAMS = {
    'max_depth': 20, 
    'max_features': 1.0, 
    'min_samples_leaf': 8, 
    'min_samples_split': 6, 
    'n_estimators': 152,
    'n_jobs': 1, 'random_state': 42
}

CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

def run_showdown():
    print("🚀 STARTING FINAL SHOWDOWN...")
    df_global, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    races_2025 = df_global[
        (df_global['Season'] == 2025) & 
        (df_global['RaceName'] != 'British Grand Prix')
    ]['RaceName'].unique()
    
    print(f"🏁 Racing across {len(races_2025)} tracks...")
    
    results = []
    
    for race in races_2025:
        test_mask = (df_global['RaceName'] == race) & (df_global['Season'] == 2025)
        if test_mask.sum() < 10: continue
        
        train_df = df_global[~test_mask]
        test_df = df_global[test_mask].sort_values(['Driver', 'LapNumber'])
        
        X_train = train_df[CAUSAL_FEATURES]
        y_train = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        
        X_test = test_df[CAUSAL_FEATURES]
        y_test_true = test_df['NextLapTimeSec'].values
        base_test = test_df['PrevLapTimeSec'].values
        
        # 1. SOLO XGB
        m1 = xgb.XGBRegressor(**XGB_SOLO_PARAMS)
        m1.fit(X_train, y_train)
        p1 = base_test + m1.predict(X_test)
        e1 = mean_absolute_error(y_test_true, p1)
        
        # 2. SOLO RF
        m2 = RandomForestRegressor(**RF_SOLO_PARAMS)
        m2.fit(X_train, y_train)
        p2 = base_test + m2.predict(X_test)
        e2 = mean_absolute_error(y_test_true, p2)
        
        # 3. MYSTERY HYBRID
        # Train specific components
        h_xgb = xgb.XGBRegressor(**HYBRID_XGB_PARAMS)
        h_xgb.fit(X_train, y_train)
        ph_xgb = base_test + h_xgb.predict(X_test)
        
        h_rf = RandomForestRegressor(**HYBRID_RF_PARAMS)
        h_rf.fit(X_train, y_train)
        ph_rf = base_test + h_rf.predict(X_test)
        
        # Mix
        p3 = (HYBRID_ALPHA * ph_xgb) + ((1 - HYBRID_ALPHA) * ph_rf)
        e3 = mean_absolute_error(y_test_true, p3)
        
        print(f"  🏁 {race[:15]:<15}: XGB={e1:.4f} | RF={e2:.4f} | Hybrid={e3:.4f}")
        
        results.append({'Race': race, 'XGB_Solo': e1, 'RF_Solo': e2, 'Hybrid_New': e3})
        
    # Summary
    df_res = pd.DataFrame(results)
    means = df_res.mean(numeric_only=True)
    
    print("\n" + "="*60)
    print(f"{'MODEL':<20} | {'MEAN MAE':<10} | {'WIN RATE':<10}")
    print("-" * 60)
    
    wins = df_res[['XGB_Solo', 'RF_Solo', 'Hybrid_New']].idxmin(axis=1).value_counts()
    
    for col in ['XGB_Solo', 'RF_Solo', 'Hybrid_New']:
        mu = means[col]
        pct = (wins.get(col, 0) / len(df_res)) * 100
        print(f"{col:<20} | {mu:.4f}s    | {pct:.1f}%")
        
    print("="*60)
    best = means.idxmin()
    print(f"🏆 CHAMPION: {best} ({means[best]:.4f}s)")

if __name__ == "__main__":
    run_showdown()
