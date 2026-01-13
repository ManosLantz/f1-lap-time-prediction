
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

# Copied from deploy_final_model.py
XGB_PARAMS = {
    'n_estimators': 1500,
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_jobs': -1,
    'random_state': 42
}

# New Feature Set including TrackDifficulty
FEATURES_V2 = [
    'PrevLapTimeSec', 'TyreAge', 'PushIndex', 
    'FuelLapsRemaining', 'CarPaceIndex', 'FieldDelta', 'TrackDifficulty'
]

def validate_v2():
    print("🧪 VALIDATING MODEL v2.0 (with Track Difficulty)...")
    
    df, _ = load_and_prep()
    
    # 2025 Season
    mask = (df['Season'] == 2025)
    df = df[mask].copy()
    
    X = df[FEATURES_V2].values
    y = (df['NextLapTimeSec'] - df['PrevLapTimeSec']).values
    groups = df['RaceName'].factorize()[0]
    
    print(f"   Data: {X.shape}")
    
    gkf = GroupKFold(n_splits=5)
    scores = []
    
    print("   Running GroupKFold (5 Splits)...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # We model the DIFFERENCE.
        # But we evaluate MAE on the (Prev + Diff) vs (TrueNext).
        # Since TrueNext = Prev + TrueDiff,
        # Error = |(Prev + PredDiff) - (Prev + TrueDiff)| = |PredDiff - TrueDiff|.
        # So simpler MAE(y_true, y_pred) is valid here.
        
        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        mae = mean_absolute_error(y_val, preds)
        scores.append(mae)
        print(f"   Fold {fold+1}: MAE = {mae:.4f}s")
        
    avg_mae = np.mean(scores)
    
    print("\n" + "="*50)
    print(f"🏁 MODEL v2.0 RESULTS")
    print(f"   Avg MAE: {avg_mae:.4f} s")
    print("="*50)
    
    # Compare with Baseline (approx 0.4119 / however that was LORO, this is KFold)
    # The previous best LORO was 0.4119. KFold might differ slightly.
    # But this number is the TRUTH for the current feature set.

if __name__ == "__main__":
    validate_v2()
