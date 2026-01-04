import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import random
from train_model import DATA_PATH, FEATURES

# === CONFIGURATION ===
N_ITER = 15
TUNING_RACES = [
    "Bahrain Grand Prix", "Saudi Arabian Grand Prix", 
    "Spanish Grand Prix", "Italian Grand Prix"
]

PARAM_DIST = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [300, 500],
    'subsample': [0.6, 0.7, 0.8],
    'colsample_bytree': [0.6, 0.7, 0.8],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [1.0, 5.0, 10.0]
}

def load_data_strict():
    print("Loading full history for tuning...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 1. Keep ALL SEASONS for the Tuning Races
    # (Fixes the "Unseen Track" issue: Model can learn Italy pace from 2022-2024)
    df = df[df['RaceName'].isin(TUNING_RACES)]
    
    if 'Compound' in df.columns:
        df = df[df['Compound'].isin(['SOFT','MEDIUM','HARD'])]

    # 2. Generate Missing Features
    if 'FuelLapsRemaining' not in df.columns:
        max_laps = df.groupby('RaceName')['LapNumber'].transform('max')
        df['FuelLapsRemaining'] = max_laps - df['LapNumber']
    
    if 'CarPaceIndex' not in df.columns:
        race_medians = df.groupby('RaceName')['LapTimeSec'].transform('median')
        df['CarPaceIndex'] = df['LapTimeSec'] / race_medians
        
    if 'LapsSinceRestart' not in df.columns:
        df['LapsSinceRestart'] = df['LapNumber'] 

    # 3. Triple Filter (Prev/Curr/Next must be Fast)
    median_pace = df.groupby('RaceName')['LapTimeSec'].transform('median')
    threshold = median_pace * 1.07
    
    if 'PrevLapTimeSec' not in df.columns:
        df['PrevLapTimeSec'] = df.groupby('Driver')['LapTimeSec'].shift(1)
        
    # Calculate Delta Target (The ChatGPT Fix)
    df['DeltaNextLapSec'] = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    mask_prev = df['PrevLapTimeSec'] < threshold
    mask_curr = df['LapTimeSec'] < threshold
    
    if 'NextLapTimeSec' in df.columns:
        mask_next = df['NextLapTimeSec'] < threshold
        final_mask = mask_prev & mask_curr & mask_next
    else:
        final_mask = mask_prev & mask_curr

    df = df[final_mask].copy()
    
    # 4. Encoding (With RaceName!)
    # We include RaceName dummies so model knows "This is Monza"
    df = pd.get_dummies(df, columns=['Driver', 'Compound', 'RaceName'], dummy_na=False)
    
    # 5. Feature Alignment
    available_feats = [f for f in FEATURES if f in df.columns]
    dummy_cols = [c for c in df.columns if c.startswith('Driver_') or c.startswith('Compound_') or c.startswith('RaceName_')]
    final_feats = list(set(available_feats + dummy_cols))
    
    # Ensure PrevLapTimeSec is in features
    if 'PrevLapTimeSec' not in final_feats:
         final_feats.append('PrevLapTimeSec')

    print(f"Data Loaded: {len(df)} laps across multiple seasons.")
    return df, final_feats

def run_tuning():
    df, features = load_data_strict()
    
    if df.empty:
        print("Error: No data.")
        return

    best_mae = float('inf')
    best_params = {}
    
    print(f"\nStarting Delta-Based Search...")
    
    for i in range(1, N_ITER + 1):
        params = {k: random.choice(v) for k, v in PARAM_DIST.items()}
        params['n_jobs'] = -1
        
        mae_scores = []
        
        # LORO on 2025 races only (Training on 2022-2024 + others)
        for race in TUNING_RACES:
            # Test: 2025 version of this race
            test_mask = (df['RaceName_' + race] == 1) & (df['Season'] == 2025)
            
            # If the dummy column approach fails (rare), fallback to string matching if RaceName column preserved
            # But get_dummies usually drops it. We trust the dummy column exists.
            if f'RaceName_{race}' not in df.columns: continue
            
            # Train: Everything else (including 2022-24 of this race!)
            train_df = df[~test_mask]
            test_df = df[test_mask]
            
            if len(test_df) < 10: continue
            
            # Align cols
            train_cols = [c for c in train_df.columns if c in features]
            
            model = xgb.XGBRegressor(**params)
            
            # === THE CHATGPT FIX: TRAIN ON DELTA ===
            model.fit(train_df[train_cols], train_df['DeltaNextLapSec'])
            delta_preds = model.predict(test_df[train_cols])
            
            # Reconstruct Absolute Time for MAE
            abs_preds = test_df['PrevLapTimeSec'] + delta_preds
            
            mae = mean_absolute_error(test_df['NextLapTimeSec'], abs_preds)
            mae_scores.append(mae)
        
        if not mae_scores: continue
        avg_mae = np.mean(mae_scores)
        
        print(f"Iter {i:02d} | MAE: {avg_mae:.4f}s | LR: {params['learning_rate']} Depth: {params['max_depth']}")
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params

    print("\n" + "=" * 60)
    print(f"BEST MAE: {best_mae:.4f}s")
    print(best_params)
    print("=" * 60)

if __name__ == "__main__":
    run_tuning()