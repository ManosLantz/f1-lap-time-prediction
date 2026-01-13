
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CONFIGURATIONS ---

# 1. CLEAR AIR MODEL (Standard)
# Uses Original 6 Features + Original Optim Params
FEATURES_CLEAR = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]
PARAMS_CLEAR = {
    'colsample_bytree': 0.92, 
    'learning_rate': 0.025, 
    'max_depth': 9, 
    'n_estimators': 108, 
    'reg_alpha': 0.77, 
    'reg_lambda': 0.20, 
    'subsample': 0.60,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

# 2. TRAFFIC MODEL (Hardcore)
# Uses 7 Features (inc TrackDiff) + Tuned V2 Params
FEATURES_TRAFFIC = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta', 'TrackDifficulty', 'PaceDelta'
]
PARAMS_TRAFFIC = {
    'n_estimators': 462, 
    'max_depth': 3, 
    'learning_rate': 0.01037218193150847, 
    'subsample': 0.8066780781052801, 
    'colsample_bytree': 0.8214893435262108, 
    'reg_alpha': 2.622705806176596, 
    'reg_lambda': 6.679210346593906,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

def validate_dual():
    print("🚀 VALIDATING DUAL-MODEL SYSTEM (Traffic Expert + Clear Air Expert)...")
    
    df, _ = load_and_prep()
    # British GP Exclusion
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    races = df['RaceName'].unique()
    scores = []
    
    print(f"   Evaluating across {len(races)} races...")
    
    for race in races:
        test_mask = (df['RaceName'] == race)
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        if len(test_df) < 10: continue
        
        # --- TRAIN BOTH ON FULL DATASET ---
        # 1. Clear Model (Standard Features)
        X_train_c = train_df[FEATURES_CLEAR]
        y_train_c = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        
        model_clear = xgb.XGBRegressor(**PARAMS_CLEAR)
        model_clear.fit(X_train_c, y_train_c)
        
        # 2. Traffic Model (Difficulty Features)
        X_train_t = train_df[FEATURES_TRAFFIC]
        y_train_t = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        
        model_traffic = xgb.XGBRegressor(**PARAMS_TRAFFIC)
        model_traffic.fit(X_train_t, y_train_t)
        
        # --- PREDICT ---
        # We must predict row-by-row based on the condition
        # Or split test set, predict, and merge.
        
        test_clear_mask = test_df['GapToAhead'] >= 1.0
        test_traffic_mask = ~test_clear_mask
        
        preds = np.zeros(len(test_df))
        
        # Predict Clear
        if test_clear_mask.sum() > 0:
            p_clear = model_clear.predict(test_df.loc[test_clear_mask, FEATURES_CLEAR])
            preds[test_clear_mask] = test_df.loc[test_clear_mask, 'PrevLapTimeSec'] + p_clear
            
        # Predict Traffic
        if test_traffic_mask.sum() > 0:
            p_traffic = model_traffic.predict(test_df.loc[test_traffic_mask, FEATURES_TRAFFIC])
            preds[test_traffic_mask] = test_df.loc[test_traffic_mask, 'PrevLapTimeSec'] + p_traffic
            
        y_true = test_df['NextLapTimeSec'].values
        mae = mean_absolute_error(y_true, preds)
        scores.append(mae)
        
        print(f"   🏁 {race[:15]:<15}: MAE {mae:.4f}s")
        
    avg_mae = np.mean(scores)
    
    print("\n" + "="*50)
    print(f"👯 DUAL-MODEL SYSTEM RESULT")
    print(f"   Avg MAE: {avg_mae:.4f} s")
    print("="*50)

if __name__ == "__main__":
    validate_dual()
