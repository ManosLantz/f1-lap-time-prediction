
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import itertools
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

# --- CAUSAL FEATURE SET ---
CAUSAL_FEATURES = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta'
]

# --- FIXED ANCHOR (Tuned XGB) ---
XGB_FINAL = {
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

def tune_ensemble_rf():
    print("🚀 Loading Data for Ensemble Synergy Tuning...")
    df, _ = load_and_prep()
    
    # Filter 2025 & Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    X = df[CAUSAL_FEATURES].values
    y = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    base_test = df['PrevLapTimeSec'].values
    y_true = df['NextLapTimeSec'].values
    
    # Pre-calculate XGB Predictions (Fixed Anchor)
    # We do this inside CV loop to avoid leakage
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    # Grid Search for RF Synergy
    # We want to see if a shallower/deeper or more random RF helps
    rf_depths = [10, 20, 30]
    rf_features = [1.0, 'sqrt', 0.5] # sqrt/0.5 increases diversity (decorrelation)
    rf_leaves = [1, 5, 10]
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5] # Focus on the mix range
    
    best_mae = 1.0
    best_config = {}
    
    print(f"📊 Dataset: {X.shape}")
    print("🔄 Sweeping RF Params to find best Hybrid Complement...")
    
    # Iterate over grid
    # To save time, we run all RF configs per fold
    
    results = [] # Store (MAE, Config)
    
    for depth, max_feat, min_leaf in itertools.product(rf_depths, rf_features, rf_leaves):
        
        # Params for this candidate RF
        rf_params = {
            'n_estimators': 100, # Fixed decent size
            'max_depth': depth,
            'max_features': max_feat,
            'min_samples_leaf': min_leaf,
            'n_jobs': 1, 'random_state': 42
        }
        
        fold_maes = {a: [] for a in alphas}
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            base_val = base_test[val_idx]
            y_val_true = y_val + base_val
            
            # Train XGB (Anchor)
            m_xgb = xgb.XGBRegressor(**XGB_FINAL)
            m_xgb.fit(X_train, y_train)
            p_xgb = base_val + m_xgb.predict(X_val)
            
            # Train RF (Candidate)
            m_rf = RandomForestRegressor(**rf_params)
            m_rf.fit(X_train, y_train)
            p_rf = base_val + m_rf.predict(X_val)
            
            # Eval Alphas
            for a in alphas:
                p_hybrid = (a * p_xgb) + ((1-a) * p_rf)
                mae = mean_absolute_error(y_val_true, p_hybrid)
                fold_maes[a].append(mae)
                
        # Average across folds
        for a in alphas:
            avg_mae = np.mean(fold_maes[a])
            
            results.append((avg_mae, a, rf_params))
            
            # Simple print if good
            if avg_mae < 0.3994: # Previous best CV score
                print(f"⭐ New Best: MAE={avg_mae:.5f} | Alpha={a} | RF={rf_params}")

    # Find total best
    results.sort(key=lambda x: x[0])
    top_mae, top_alpha, top_rf = results[0]
    
    print("\n" + "="*80)
    print(f"🏆 BEST HYBRID CONFIGURATION")
    print("-" * 80)
    print(f"Global Best CV MAE: {top_mae:.5f}s")
    print(f"Optimal Alpha: {top_alpha} (XGB Weight)")
    print(f"Optimal RF Params: {top_rf}")
    print("="*80)

if __name__ == "__main__":
    tune_ensemble_rf()
