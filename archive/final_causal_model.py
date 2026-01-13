
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys
import os

# Import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
# Based on Graphical Lasso Discovery
CAUSAL_FEATURES = [
    'PrevLapTimeSec',     # The Baseline
    'TyreAge',            # Physical Wear
    'PushIndex',          # Driver Intent
    'FuelLapsRemaining',  # Mass
    'CarPaceIndex',       # Car Performance Proxy (Replaces Driver IDs)
    'FieldDelta'          # Track Evolution
]

def run_causal_evaluation():
    print("🚀 STARTING CAUSAL MODEL EVALUATION (Pure Physics)...")
    print(f"Features: {CAUSAL_FEATURES}")
    
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
        
        # Train simple XGBoost using typical params
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        
        preds = base_test + model.predict(X_test)
        mae = mean_absolute_error(y_test_true, preds)
        
        results.append(mae)
        print(f"  🏁 {race:<25}: Causal MAE={mae:.4f}s")
        
    avg_mae = np.mean(results)
    print("\n" + "="*50)
    print(f"🏆 CAUSAL MODEL GLOBAL MAE: {avg_mae:.4f}s")
    print("="*50)

if __name__ == "__main__":
    run_causal_evaluation()
