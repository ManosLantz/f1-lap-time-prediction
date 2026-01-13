
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def validate_v3():
    print("🧪 VALIDATING MODEL V3 (Refined Features + New Flags)...")
    
    df, features = load_and_prep()
    
    # Target races for validation: 2025 (excluding British GP)
    test_races_2025 = df[(df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')]['RaceName'].unique()
    
    races = test_races_2025
    scores = []
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🔍 Features ({len(features)}): {features}")
    
    # Using Champion params as a baseline
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
    
    print(f"   Evaluating across {len(races)} races (2025 ONLY)...")
    
    for race in races:
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask]
        test_df = df[test_mask]
        
        if len(test_df) < 10: continue
        
        X_train = train_df[features].values
        y_train = (train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']).values
        
        X_test = test_df[features].values
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
    print(f"🏆 MODEL V3 LORO RESULT")
    print(f"   Avg MAE: {avg_mae:.4f} s")
    print("="*50)
    print(f"   Compare to v1 Champion: 0.4167 s")
    print(f"   Compare to v2 (23 features): 0.4094 s")

if __name__ == "__main__":
    validate_v3()
