
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

def run_fast_tuning():
    print("🚀 Loading Data for Fast Causal Tuning...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    X = df[CAUSAL_FEATURES]
    y = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    print(f"📊 Dataset: {X.shape}")
    
    # ==========================================
    # 1. Tune XGBoost
    # ==========================================
    print("\n🔬 Tuning XGBoost (Causal)...")
    xgb_params = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.2),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }
    
    xgb_search = RandomizedSearchCV(
        xgb.XGBRegressor(n_jobs=-1, random_state=42),
        param_distributions=xgb_params,
        n_iter=20,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    xgb_search.fit(X, y)
    print(f"✅ Best XGB Params: {xgb_search.best_params_}")
    print(f"   Best CV MAE: {-xgb_search.best_score_:.4f}")
    
    # ==========================================
    # 2. Tune Random Forest
    # ==========================================
    print("\n🔬 Tuning Random Forest (Causal)...")
    # For RF, use fewer estimators during tuning to be fast
    rf_params = {
        'n_estimators': randint(50, 200),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 10),
        'max_features': [1.0, 'sqrt', 0.5]
    }
    
    rf_search = RandomizedSearchCV(
        RandomForestRegressor(n_jobs=1, random_state=42), # n_jobs=1 for safety
        param_distributions=rf_params,
        n_iter=15,
        cv=3,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1, # Parallelize the search, not the fit
        verbose=1
    )
    rf_search.fit(X, y)
    print(f"✅ Best RF Params: {rf_search.best_params_}")
    print(f"   Best CV MAE: {-rf_search.best_score_:.4f}")
    
    return xgb_search.best_params_, rf_search.best_params_

if __name__ == "__main__":
    run_fast_tuning()
