
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def objective(trial):
    # 1. Suggest Params
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 600),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # 2. LORO Validation on Globals
    races = np.unique(groups_global)
    fold_maes = []
    
    for race_idx in races:
        val_mask = (groups_global == race_idx)
        train_mask = ~val_mask
        
        X_train, X_val = X_global[train_mask], X_global[val_mask]
        y_train, y_val = y_global[train_mask], y_global[val_mask]
        
        base_val = base_global[val_mask]
        y_true_val = y_val + base_val
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        preds_diff = model.predict(X_val)
        preds_lap = base_val + preds_diff
        
        mae = mean_absolute_error(y_true_val, preds_lap)
        fold_maes.append(mae)
        
    return np.mean(fold_maes)

if __name__ == "__main__":
    print("🚀 LOADING DATA FOR V3 LORO TUNING...")
    df, features = load_and_prep()
    
    # Filter 2025 (excluding British GP used for leaderboard bench)
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Set Globals for speed in Optuna
    global X_global, y_global, base_global, groups_global
    X_global = df[features].values
    y_global = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    base_global = df['PrevLapTimeSec'].values
    groups_global = df['RaceName'].factorize()[0]
    
    print(f"📊 Dataset: {X_global.shape}, {len(np.unique(groups_global))} Races")
    print(f"🔍 Features: {features}")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    
    print("\n" + "="*60)
    print(f"🏆 BEST V3 LORO MAE: {study.best_value:.4f}s")
    print("-" * 60)
    print(f"Params: {study.best_params}")
    print("="*60)
    
    # Save best params
    joblib.dump(study.best_params, 'best_params_v3_loro.pkl')
    print("💾 Saved best params to 'best_params_v3_loro.pkl'")
