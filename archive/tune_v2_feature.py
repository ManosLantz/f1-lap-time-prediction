
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

# Features including the new Interaction term
FEATURES_V2 = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex'
]

def objective(trial):
    # Suggest Params
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # Global data access (loaded in main)
    
    kf = GroupKFold(n_splits=3)
    scores = []
    
    for train_idx, val_idx in kf.split(X_global, y_global, groups=groups_global):
        X_train, X_val = X_global[train_idx], X_global[val_idx]
        # Target: Difference
        y_train, y_val = diff_global[train_idx], diff_global[val_idx]
        
        # We model the DIFFERENCE
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds_diff = model.predict(X_val)
        
        # Evaluate on FULL LAP TIME MAE
        # Actual Next = Prev + Actual Diff
        # Pred Next = Prev + Pred Diff
        # Error = |Pred Next - Actual Next| = |Pred Diff - Actual Diff|
        # So evaluating on residuals is correct.
        
        mae = mean_absolute_error(y_val, preds_diff)
        scores.append(mae)
            
    return np.mean(scores)

if __name__ == "__main__":
    print("🚀 TUNING MODEL v2.0 (New Feature)...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP (Standard Clean Set)
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Pre-calculate globals for speed
    global X_global, y_global, diff_global, groups_global
    
    X_global = df[FEATURES_V2].values
    y_global = df['NextLapTimeSec'].values    # Actual Lap
    base_global = df['PrevLapTimeSec'].values # Previous Lap
    diff_global = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    groups_global = df['RaceName'].factorize()[0]
    
    print(f"📊 Dataset: {X_global.shape}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*60)
    print(f"🏆 BEST CV MAE: {study.best_value:.4f}s")
    print("-" * 60)
    print(f"Params: {study.best_params}")
    print("="*60)
    
    # Save best params
    joblib.dump(study.best_params, 'best_params_v2.pkl')
