
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# Features identified as "KEEP" by Permutation Importance
FEATURES_PERMUTATION = [
    'FieldDelta', 'CarPaceIndex', 'PrevLapTimeSec', 'PushIndex', 'TyreAge', 
    'FuelLapsRemaining', 'Sector2Sec', 'Sector3Sec', 'PrevFieldMedian', 
    'TrackDifficulty', 'PaceDelta', 'AirTemp', 'Sector1Sec', 'TyreDegSmooth', 
    'TrackTemp', 'Driver_VER', 'LapsSinceRestart', 'Driver_COL', 'GapToAhead', 
    'Driver_NOR', 'Driver_PIA', 'DifficultyInteraction', 'Compound_HARD'
]

# Using same params as the v1 Champion for a fair "Feature-only" comparison
# (Or maybe default params if v1 params were specifically for 6 features?)
# Let's use the v1 Champion params first.
XGB_PARAMS = {
    'n_estimators': 368,
    'max_depth': 4, 
    'learning_rate': 0.014134247715989, 
    'subsample': 0.5053056191976778, 
    'colsample_bytree': 0.8484497288832383, 
    'reg_alpha': 4.277056587799488, 
    'reg_lambda': 7.196434250850857,
    'n_jobs': -1, 'random_state': 42, 'verbosity': 0
}

def validate_permutation_set():
    print("🧪 VALIDATING PERMUTATION-BASED MODEL (23 Features)...")
    
    df, _ = load_and_prep()
    
    # Target races for validation: 2025 (excluding British GP)
    test_races_2025 = df[(df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')]['RaceName'].unique()
    
    races = test_races_2025
    scores = []
    
    print(f"   Evaluating across {len(races)} races (2025 ONLY)...")
    
    for race in races:
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        if len(test_df) < 10: continue
        
        X_train = train_df[FEATURES_PERMUTATION].values
        y_train = (train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']).values
        
        X_test = test_df[FEATURES_PERMUTATION].values
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
    print(f"🏆 PERMUTATION MODEL LORO RESULT")
    print(f"   Avg MAE: {avg_mae:.4f} s")
    print("="*50)
    print(f"   Compare to Causal Champion: 0.4167 s")

if __name__ == "__main__":
    validate_permutation_set()
