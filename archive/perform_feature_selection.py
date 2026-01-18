
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import sys
import os
import joblib

# Import data_loader (from parent directory)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import load_features

def run_feature_selection():
    print("⏳ Loading Data for Feature Selection...")
    df, features = load_features()
    
    # Filter 2025 and Exclude British GP (Clean Schema)
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    print(f"📊 Dataset Size: {len(df)} rows")
    print(f"🔍 Analyzing {len(features)} features...")
    
    X = df[features]
    y = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    DATA_PATH = "data/f1_lap_dataset.csv"
    META_PATH = "model_v3_meta.pkl"  
    # Train XGBoost (Fast but decent)
    print("🚀 Training XGBoost Probe Model...")
    meta = joblib.load(META_PATH)
    BEST_PARAMS = meta["best_params"]

    model = xgb.XGBRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)
    
    baseline_score = model.score(X_val, y_val)
    print(f"✅ Baseline R² Score: {baseline_score:.4f}")
    
    # Calculate Permutation Importance
    print("🔄 Calculating Permutation Importance (this may take a minute)...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
        scoring='neg_mean_absolute_error' # We want to see impact on MAE
    )
    
    # Rank Features
    perm_sorted_idx = result.importances_mean.argsort()
    
    print("\n" + "="*80)
    print(f"{'RANK':<5} | {'FEATURE':<30} | {'MAE IMPACT (s)':<15} | {'STATUS'}")
    print("-" * 80)
    
    top_features = []
    
    # Print in descending order (Most important first)
    for i in perm_sorted_idx[::-1]:
        feature = features[i]
        importance = result.importances_mean[i]
        std = result.importances_std[i]
        
        # Importance is negative MAE change? No, sklearn flips it for 'neg_'. 
        # Actually default is usually score reduction. 
        # Let's trust the magnitude. Higher is better.
        
        status = "🟢 KEEP"
        if importance <= 0: status = "🔴 REMOVE (Useless)"
        elif importance < 0.001: status = "🟡 WEAK (Consider Removing)"
        
        print(f"{len(features)-i:<5} | {feature:<30} | {importance:.6f}       | {status}")
        
    print("="*80)
    print("NOTE: 'MAE IMPACT' means how much worse the MAE gets if we randomize this feature.")

if __name__ == "__main__":
    run_feature_selection()
