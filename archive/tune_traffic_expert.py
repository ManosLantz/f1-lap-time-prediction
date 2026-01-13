
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# Features for Traffic Expert (Includes Track Difficulty)
FEATURES_TRAFFIC = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta', 'TrackDifficulty'
]

def objective(trial):
    # Suggest Params
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    kf = GroupKFold(n_splits=3)
    scores = []
    
    for train_idx, val_idx in kf.split(X_global, y_global, groups=groups_global):
        X_train = X_global[train_idx]
        y_train = diff_global[train_idx]
        
        X_val = X_global[val_idx]
        y_val_target = y_global[val_idx] # True Next Lap Time
        base_val = base_global[val_idx]  # Prev Lap Time
        gap_val = gap_global[val_idx]    # Gap To Ahead
        
        # Train on EVERYTHING (Learn Physics)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        # Predict on VAL
        preds_diff = model.predict(X_val)
        preds_total = base_val + preds_diff
        
        # VALIDATE ONLY ON TRAFFIC (Gap < 1.0)
        # This forces the optimized params to prioritize Traffic Accuracy
        mask_traffic = (gap_val < 1.0)
        
        if mask_traffic.sum() == 0: continue
        
        mae_traffic = mean_absolute_error(y_val_target[mask_traffic], preds_total[mask_traffic])
        scores.append(mae_traffic)
            
    return np.mean(scores)

if __name__ == "__main__":
    print("🚀 TUNING TRAFFIC EXPERT (Target: Gap < 1.0s)...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Pre-calculate globals
    global X_global, y_global, diff_global, base_global, groups_global, gap_global
    
    X_global = df[FEATURES_TRAFFIC].values
    y_global = df['NextLapTimeSec'].values    
    base_global = df['PrevLapTimeSec'].values 
    diff_global = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    groups_global = df['RaceName'].factorize()[0]
    gap_global = df['GapToAhead'].values # Crucial for masking
    
    print(f"📊 Dataset: {X_global.shape}")
    print(f"   Traffic Samples: {(gap_global < 1.0).sum()} / {len(df)}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*60)
    print(f"🏆 BEST TRAFFIC MAE: {study.best_value:.4f}s")
    print("-" * 60)
    print(f"Params: {study.best_params}")
    print("="*60)
    
    joblib.dump(study.best_params, 'best_params_traffic.pkl')
