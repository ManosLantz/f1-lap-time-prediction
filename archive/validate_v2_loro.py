
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

XGB_PARAMS = {
    'colsample_bytree': 0.92, 
    'learning_rate': 0.025, 
    'max_depth': 9, 
    'n_estimators': 108, 
    'reg_alpha': 0.77, 
    'reg_lambda': 0.20, 
    'subsample': 0.60,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

FEATURES_V2 = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

def validate_loro():
    print("🧪 VALIDATING MODEL v2.0 (LORO Strategy)...")
    
    df, _ = load_and_prep()
    
    # Filter for 2025 only as requested by the user
    df = df[df['Season'] == 2025].copy()
    
    # Exclude British GP for scoring
    test_races_2025 = df[df['RaceName'] != 'British Grand Prix']['RaceName'].unique()
    
    races = test_races_2025
    scores = []
    
    print(f"   Evaluating across {len(races)} races (2025 ONLY)...")
    
    for race in races:
        if race == 'British Grand Prix':
            continue
            
        test_mask = (df['RaceName'] == race)
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        if len(test_df) < 10: continue
        
        X_train = train_df[FEATURES_V2].values
        y_train = (train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']).values
        
        X_test = test_df[FEATURES_V2].values
        y_test_true = test_df['NextLapTimeSec'].values
        base_test = test_df['PrevLapTimeSec'].values
        
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        
        pred_diff = model.predict(X_test)
        pred_lap = base_test + pred_diff
        
        mae = mean_absolute_error(y_test_true, pred_lap)
        scores.append(mae)
        print(f"   🏁 {race[:15]:<15}: MAE {mae:.4f}s")
        
    avg_mae = np.mean(scores)
    
    print("\n" + "="*50)
    print(f"🏆 MODEL v2.0 LORO RESULT")
    print(f"   Avg MAE: {avg_mae:.4f} s")
    print("="*50)

if __name__ == "__main__":
    validate_loro()
