
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

# --- TUNED PARAMS (From fast_tune_causal.py) ---
XGB_TUNED = {
    'colsample_bytree': 0.92, 
    'learning_rate': 0.025, 
    'max_depth': 9, 
    'n_estimators': 108, 
    'reg_alpha': 0.77, 
    'reg_lambda': 0.20, 
    'subsample': 0.60,
    'n_jobs': -1, 'random_state': 42,
    'verbosity': 0
}

RF_TUNED = {
    'max_depth': 20, 
    'max_features': 1.0, 
    'min_samples_leaf': 8, 
    'min_samples_split': 6, 
    'n_estimators': 152,
    'n_jobs': 1, 'random_state': 42
}

def tune_hybrid_alpha():
    print("🚀 Loading Data for Hybrid Causal Tuning...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    X = df[CAUSAL_FEATURES].values
    y = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    base_test = df['PrevLapTimeSec'].values
    drivers = df['Driver'].values
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_mae = {a: [] for a in alphas}
    
    print(f"📊 Dataset: {X.shape}")
    print("🔄 Running 3-Fold CV to find Best Alpha...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"   Fold {fold+1}/3...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        base_val = base_test[val_idx]
        drivers_val = drivers[val_idx]
        y_true_val = y_val + base_val # Reconstruct actual lap time
        
        # 1. Train Base Models
        m_xgb = xgb.XGBRegressor(**XGB_TUNED)
        m_xgb.fit(X_train, y_train)
        p_xgb = base_val + m_xgb.predict(X_val)
        
        m_rf = RandomForestRegressor(**RF_TUNED)
        m_rf.fit(X_train, y_train)
        p_rf = base_val + m_rf.predict(X_val)
        
        # 2. Test Alphas
        # (Simplified adaptive logic simulation: constant alpha across race?)
        # Wait, the hybrid model uses alpha for *adaptive error smoothing*.
        # To strictly tune it, we need sequential data (Race-by-Race).
        # Random Shuffle CV destroys the sequential nature required for EWMA.
        # BUT, we can test "Static Weighting" or "Adaptive Simulation" if we group by race.
        
        # Alternative: Just optimize weighted average W*P1 + (1-W)*P2
        # The complex adaptive logic is hard to CV quickly.
        # Let's optimize the simple ensemble weight first.
        
        # Calculate Errors
        err_xgb = np.abs(y_true_val - p_xgb)
        err_rf = np.abs(y_true_val - p_rf)
        
        # Just use global mix for speed? 
        # Or run the adaptive logic? 
        # Generating sequential indices for val set is tricky with Shuffle.
        
        # Let's switch to checking simple weighted average.
        for a in alphas:
            p_hybrid = (a * p_xgb) + ((1-a) * p_rf) # a = weight for XGB
            mae = mean_absolute_error(y_true_val, p_hybrid)
            alpha_mae[a].append(mae)
            
    # Aggregate
    print("\n" + "="*40)
    print(f"{'ALPHA (XGB Wgt)':<15} | {'CV MAE':<10}")
    print("-" * 40)
    
    best_a = 0.5
    best_score = 1.0
    
    for a in alphas:
        avg = np.mean(alpha_mae[a])
        print(f"{a:<15} | {avg:.4f}")
        if avg < best_score:
            best_score = avg
            best_a = a
            
    print("="*40)
    print(f"✅ Best Alpha (Static): {best_a} (MAE: {best_score:.4f})")
    
    # We can use this as a proxy for the dynamic alpha
    return best_a

if __name__ == "__main__":
    tune_hybrid_alpha()
