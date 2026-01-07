import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import mean_absolute_error

#Paths
DATA_PATH = "data/f1_lap_dataset.csv"

# === CONFIGURATION ===
N_ITER = 120  # Good balance of speed/quality
N_FOLDS = 5 

# Expanded Search Space for Clean Data
PARAM_DIST = {
    # Since data is cleaner, we might tolerate slightly higher learning rates
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
    'n_estimators': [300, 500, 700, 1000],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'subsample': [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5],
    'reg_lambda': [1.0, 5.0, 10.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0]
}

FEATURES = [
    'LapNumber', 'TyreAge', 'FuelLapsRemaining', 
    'AirTemp', 'TrackTemp', 'Humid', 'Rainfall', 
    'CarPaceIndex', 'LapsSinceRestart'
]

def load_tuning_data_clean():
    print("Loading history (2022-2024) for tuning...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 1. Filter Years
    df = df[df['Season'].isin([2022, 2023, 2024])].copy()
    if 'Compound' in df.columns:
        df = df[df['Compound'].isin(['SOFT','MEDIUM','HARD'])]

    # 2. Feature Gen
    if 'FuelLapsRemaining' not in df.columns:
        max_laps = df.groupby('RaceName')['LapNumber'].transform('max')
        df['FuelLapsRemaining'] = max_laps - df['LapNumber']
    
    if 'CarPaceIndex' not in df.columns:
        race_medians = df.groupby('RaceName')['LapTimeSec'].transform('median')
        df['CarPaceIndex'] = df['LapTimeSec'] / race_medians
        
    if 'LapsSinceRestart' not in df.columns:
        df['LapsSinceRestart'] = df['LapNumber']

    if 'PrevLapTimeSec' not in df.columns:
        df['PrevLapTimeSec'] = df.groupby('Driver')['LapTimeSec'].shift(1)

    # 3. APPLY THE "JAPAN FIX" FILTER (Strict Cleaning)
    print("   Applying Dynamic Race Status Filter (Removing VSC/Slow Laps)...")
    
    # A. Calculate Global Benchmark per race (Top 25% pace)
    race_benchmarks = df.groupby(['Season', 'RaceName'])['LapTimeSec'].quantile(0.25)
    df = df.merge(race_benchmarks.rename('RaceBenchmark'), on=['Season', 'RaceName'], how='left')
    
    # B. Identify Slow Laps (> 105% of benchmark)
    df['IsSlowLap'] = df['LapTimeSec'] > (df['RaceBenchmark'] * 1.05)
    
    # C. Shift to find broken chains
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['PrevIsSlow'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsSlowLap'].shift(1).fillna(True)
    df['NextIsSlow'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsSlowLap'].shift(-1).fillna(True)
    
    # D. Keep only Pure Chains (Fast -> Fast -> Fast)
    clean_mask = (~df['IsSlowLap']) & (~df['PrevIsSlow']) & (~df['NextIsSlow'])
    df = df[clean_mask].copy()
    
    # 4. Create Target (Delta)
    df['DeltaNextLapSec'] = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    # 5. Backup RaceName for Grouping
    race_name_backup = df['RaceName'].copy() 
    
    # 6. Encode
    df = pd.get_dummies(df, columns=['Driver', 'Compound', 'RaceName'], dummy_na=False)
    df['RaceName'] = race_name_backup 

    # 7. Final Features
    available_feats = [f for f in FEATURES if f in df.columns]
    dummy_cols = [c for c in df.columns if c.startswith('Driver_') or c.startswith('Compound_') or c.startswith('RaceName_')]
    final_feats = list(set(available_feats + dummy_cols))
    if 'PrevLapTimeSec' not in final_feats: final_feats.append('PrevLapTimeSec')

    print(f"Tuning Data: {len(df)} clean laps. (Seasons: {df['Season'].unique()})")
    return df, final_feats

def run_tuning():
    df, features = load_tuning_data_clean()
    
    X = df[features]
    y = df['DeltaNextLapSec']
    groups = df['RaceName']  # Group by Race to prevent leakage
    
    model = xgb.XGBRegressor(objective='reg:absoluteerror', n_jobs=-1, random_state=42)
    
    cv_strategy = GroupKFold(n_splits=N_FOLDS)
    
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring='neg_mean_absolute_error',
        cv=cv_strategy,
        verbose=1,
        n_jobs=-1,
    )
    
    print(f"Starting Random Search ({N_ITER} iterations)...")
    random_search.fit(X, y, groups=groups)
    
    print("\n============================================================")
    print(f"🏆 CHAMPION CONFIGURATION (CV MAE: {-random_search.best_score_:.4f}s)")
    print("------------------------------------------------------------")
    print(random_search.best_params_)
    print("============================================================")

if __name__ == "__main__":
    run_tuning()