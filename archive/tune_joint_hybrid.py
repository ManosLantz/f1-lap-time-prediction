
import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

def objective(trial):
    # 1. Suggest XGB Params
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 300),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        'n_jobs': -1,
        'verbosity': 0,
        'random_state': 42
    }
    
    # 2. Suggest RF Params
    rf_params = {
        'n_estimators': trial.suggest_int('rf_n_estimators', 50, 200),
        'max_depth': trial.suggest_int('rf_max_depth', 5, 25),
        'min_samples_leaf': trial.suggest_int('rf_leaf', 1, 20),
        'max_features': trial.suggest_categorical('rf_features', [1.0, 'sqrt', 0.5]),
        'n_jobs': 1, # Safety first
        'random_state': 42
    }
    
    # 3. Suggest Alpha
    alpha = trial.suggest_float('alpha', 0.0, 1.0)
    
    # 5. Validation Strategy: GroupKFold by Race
    gkf = GroupKFold(n_splits=3)
    
    scores = []
    
    # Use Global Data
    for train_idx, val_idx in gkf.split(X_global, y_global, groups=groups_global):
        X_train, X_val = X_global[train_idx], X_global[val_idx]
        y_train, y_val = y_global[train_idx], y_global[val_idx]
        base_val = base_global[val_idx]
        y_true_val = y_val + base_val
        
        # Train XGB
        m_xgb = xgb.XGBRegressor(**xgb_params)
        m_xgb.fit(X_train, y_train)
        p_xgb = base_val + m_xgb.predict(X_val)
        
        # Train RF
        m_rf = RandomForestRegressor(**rf_params)
        m_rf.fit(X_train, y_train)
        p_rf = base_val + m_rf.predict(X_val)
        
        # Hybrid Mix
        p_hybrid = (alpha * p_xgb) + ((1 - alpha) * p_rf)
        
        mae = mean_absolute_error(y_true_val, p_hybrid)
        scores.append(mae)
        
    return np.mean(scores)

if __name__ == "__main__":
    print("🚀 LOADING DATA FOR EXTENDED JOINT TUNING (50 Trials)...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Set Globals for Optuna
    global X_global, y_global, base_global, groups_global
    X_global = df[CAUSAL_FEATURES].values
    y_global = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    base_global = df['PrevLapTimeSec'].values
    groups_global = df['RaceName'].factorize()[0] 
    
    print(f"📊 Data: {X_global.shape}, Groups: {len(np.unique(groups_global))} Races")
    print("🔄 STARTING OPTUNA STUDY (50 Trials)...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("\n" + "="*80)
    print(f"🏆 BEST JOINT RESULT: {study.best_value:.4f}s")
    print("-" * 80)
    print(f"Params: {study.best_params}")
    print("="*80)
