import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA_PATH = "data/f1_lap_dataset.csv"

# The features we want to train on
FEATURES = [
    "PrevLapTimeSec", "LapsSinceRestart", "TyreAge", 
    "AirTemp", "TrackTemp", "AEBI",
    "Sector1Sec", "Sector2Sec", "Sector3Sec",
    "PushIndex", "TyreDegSmooth", "FuelLapsRemaining",
    "CarPaceIndex", "IsDRSRange", "GapAheadSec"
]

def compute_laps_since_restart(df):
    if 'TrackStatus' not in df.columns: return np.zeros(len(df))
    status = df['TrackStatus'].astype(str)
    is_sc = status.str.contains('4|5|6').astype(int)
    sc_change = is_sc.diff().fillna(0)
    restart_trigger = (sc_change == -1)
    stint_id = restart_trigger.cumsum()
    df_temp = df.copy()
    df_temp['StintID'] = stint_id
    laps_since = df_temp.groupby(['RaceName', 'Driver', 'StintID']).cumcount() + 1
    laps_since = np.where(is_sc == 1, 0, laps_since)
    return laps_since

def generate_missing_features(df):
    print("Checking for missing features...")
    if 'FuelLapsRemaining' not in df.columns:
        max_laps = df.groupby('RaceName')['LapNumber'].transform('max')
        df['FuelLapsRemaining'] = max_laps - df['LapNumber']
    if 'CarPaceIndex' not in df.columns:
        race_medians = df.groupby('RaceName')['LapTimeSec'].transform('median')
        df['CarPaceIndex'] = df['LapTimeSec'] / race_medians
    if 'LapsSinceRestart' not in df.columns:
        df['LapsSinceRestart'] = compute_laps_since_restart(df)
    return df


def add_global_context(df):
    print("  Adding Global Field Context (The 'Japan Fix')...")
    
    # 1. Calculate the Median Pace of the FIELD for every lap in every race
    # Group by [Season, RaceName, LapNumber] -> Get Median LapTime
    field_stats = df.groupby(['Season', 'RaceName', 'LapNumber'])['LapTimeSec'].median().reset_index()
    field_stats.rename(columns={'LapTimeSec': 'FieldMedian_Current'}, inplace=True)
    
    # 2. Merge this back into the main dataframe
    df = df.merge(field_stats, on=['Season', 'RaceName', 'LapNumber'], how='left')
    
    # 3. CRITICAL: Shift by 1 to avoid Leakage
    # To predict Lap 20, we can only look at the Field Median of Lap 19.
    # We group by Driver to ensure the shift happens within each driver's timeline
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['PrevFieldMedian'] = df.groupby(['Season', 'RaceName', 'Driver'])['FieldMedian_Current'].shift(1)
    
    # 4. Create the "Track Offset" Feature
    # How much faster is the field running compared to the driver's own previous lap?
    # (Optional interaction feature, but PrevFieldMedian alone is usually enough)
    df['FieldDelta'] = df['PrevLapTimeSec'] - df['PrevFieldMedian']
    
    # Fill NA for Lap 1 (use current lap or 0)
    df['PrevFieldMedian'] = df['PrevFieldMedian'].fillna(df['PrevLapTimeSec'])
    df['FieldDelta'] = df['FieldDelta'].fillna(0)
    
    return df

def add_traffic_context(df):
    print("  Adding Local Traffic Context (The 'Azerbaijan Fix')...")
    
    # 1. Calculate Cumulative Race Time for every driver
    # We assume 'LapTimeSec' captures the full loop. 
    # This approximates the "Race Position" graph.
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['RaceTime'] = df.groupby(['Season', 'RaceName', 'Driver'])['LapTimeSec'].cumsum()
    
    # 2. Rank drivers per lap to find "Car Ahead"
    # Group by Race/Lap and sort by Total Race Time (Fastest cumulative time is P1)
    df = df.sort_values(['Season', 'RaceName', 'LapNumber', 'RaceTime'])
    
    # 3. Calculate Gap to the car immediately ahead
    # Shift(1) in this sorted order gives the RaceTime of the car in front
    df['TimeAhead'] = df.groupby(['Season', 'RaceName', 'LapNumber'])['RaceTime'].shift(1)
    
    # The Gap is simply MyTime - TimeAhead
    df['GapToAhead'] = df['RaceTime'] - df['TimeAhead']
    
    # 4. Handle P1 (Leader has no car ahead) and Outliers
    # Fill NA with a large number (Clean Air) e.g., 20 seconds
    df['GapToAhead'] = df['GapToAhead'].fillna(20.0)
    
    # 5. Create "InTraffic" Feature (The DRS Train Indicator)
    # If gap is < 1.5s, you are likely stuck in dirty air/DRS train
    # We use a soft curve or binary flag. Let's use the raw gap but clipped.
    df['CloseTraffic'] = (df['GapToAhead'] < 1.5).astype(int)
    
    # Clean up temporary columns to save memory
    df.drop(columns=['RaceTime', 'TimeAhead'], inplace=True)
    
    return df

def load_and_prep():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # 1. Standard Compound Filter
    if 'Compound' in df.columns:
        df = df[df['Compound'].isin(['SOFT','MEDIUM','HARD'])]
    df = add_global_context(df)   # The Japan Fix (Global)
    df = add_traffic_context(df)  # The Azerbaijan Fix (Local) <--- NEW
    # --- NEW: TEMPORAL CONTEXT (The British GP Fix) ---
    # 2. CALCULATE LIMITS
    # Define "Slow" as > 107% of the median pace (Standard F1 Rule)
    median_pace = df.groupby('RaceName')['LapTimeSec'].transform('median')
    pace_threshold = median_pace * 1.07
    
    # 3. CRITICAL FIX: Filter CURRENT Lap AND NEXT Lap
    # We only want to train on:
    #  (a) Fast Laps...
    #  (b) ...that are followed by Fast Laps.
    
    mask_current_fast = df['LapTimeSec'] < pace_threshold
    
    # Check if 'NextLapTimeSec' exists and filter it too
    if 'NextLapTimeSec' in df.columns:
        mask_next_fast = df['NextLapTimeSec'] < pace_threshold
        df = df[mask_current_fast & mask_next_fast]
    else:
        # Fallback if column missing (shouldn't happen)
        df = df[mask_current_fast]
        
    print(f"Filtered dataset to {len(df)} pure racing transitions.")

    # ... (Rest of function: Generate missing features, One-Hot, etc.) ...
    df = generate_missing_features(df)
    
    # ... inside load_and_prep ...
    
    # 1. Backup the 'Driver' column before it gets deleted
    driver_backup = df['Driver'].copy()
    
    # 2. Perform One-Hot Encoding (This deletes 'Driver')
    df = pd.get_dummies(df, columns=['Driver', 'Compound'], dummy_na=False)
    
    # 3. RESTORE the 'Driver' column so the evaluation loop can use it
    df['Driver'] = driver_backup
    
    global FEATURES
    dummy_cols = [c for c in df.columns if c.startswith('Driver_') or c.startswith('Compound_')]
    
    # Add new features to the list
    new_features = ['PrevFieldMedian', 'FieldDelta', 'GapToAhead', 'CloseTraffic']
    model_features = FEATURES + dummy_cols + new_features
    
    return df, model_features

def run_loro_evaluation(df, features):
    print(f"\n{'RACE NAME':<30} | {'MODEL':<8} | {'BASE':<8} | {'STATIC':<8} | {'ADAPTIVE':<8} | {'WINNER':<10}")
    print("-" * 115)
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    
    # Store scores for ALL strategies to calculate averages later
    scores_model = []
    scores_base = []
    scores_static = []
    scores_adaptive = []
    
    for race in test_races:
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask]
        
        # Sort strictly for the history loop
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # Train Model
        model = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6, 
            subsample=0.7, colsample_bytree=0.8, min_child_weight=3, 
            reg_lambda=5.0, n_jobs=-1
        )
        model.fit(train_df[features], train_df['NextLapTimeSec'])
        
        # --- PREDICTIONS (Force Numpy) ---
        preds_model = model.predict(test_df[features])
        if hasattr(preds_model, "values"): preds_model = preds_model.values
            
        preds_base = test_df['PrevLapTimeSec'].values
        y_true = test_df['NextLapTimeSec'].values
        
        # Safety Clip
        preds_model = np.maximum(preds_model, preds_base - 2.0)

        # --- ADAPTIVE LOOP ---
        adaptive_preds = np.zeros(len(test_df))
        drivers = test_df['Driver'].unique()
        current_idx = 0
        
        for driver in drivers:
            n_laps = len(test_df[test_df['Driver'] == driver])
            
            d_preds_model = preds_model[current_idx : current_idx + n_laps]
            d_preds_base = preds_base[current_idx : current_idx + n_laps]
            d_y_true = y_true[current_idx : current_idx + n_laps]
            
            d_adaptive_preds = []
            
            w_model = 0.5
            alpha = 0.5
            err_model_smooth = 0.5
            err_base_smooth = 0.5
            
            for i in range(n_laps):
                # Predict
                pred = (w_model * d_preds_model[i]) + ((1 - w_model) * d_preds_base[i])
                d_adaptive_preds.append(pred)
                
                # Update Weights
                true_val = d_y_true[i]
                raw_err_model = abs(true_val - d_preds_model[i])
                raw_err_base = abs(true_val - d_preds_base[i])
                
                err_model_smooth = (alpha * raw_err_model) + ((1 - alpha) * err_model_smooth)
                err_base_smooth = (alpha * raw_err_base) + ((1 - alpha) * err_base_smooth)
                
                total_error = err_model_smooth + err_base_smooth
                if total_error > 0:
                    w_model = err_base_smooth / total_error
                else:
                    w_model = 0.5
            
            adaptive_preds[current_idx : current_idx + n_laps] = d_adaptive_preds
            current_idx += n_laps

        # Calculate Metrics
        mae_model = mean_absolute_error(y_true, preds_model)
        mae_base = mean_absolute_error(y_true, preds_base)
        mae_static = mean_absolute_error(y_true, (0.5*preds_model + 0.5*preds_base))
        mae_adaptive = mean_absolute_error(y_true, adaptive_preds)
        
        # Append to lists
        scores_model.append(mae_model)
        scores_base.append(mae_base)
        scores_static.append(mae_static)
        scores_adaptive.append(mae_adaptive)
        
        race_scores = {'MODEL': mae_model, 'BASE': mae_base, 'STATIC': mae_static, 'ADAPTIVE': mae_adaptive}
        winner = min(race_scores, key=race_scores.get)
        
        print(f"{race:<30} | {mae_model:.3f}s  | {mae_base:.3f}s  | {mae_static:.3f}s  | {mae_adaptive:.3f}s  | {winner}")

    # --- FINAL SUMMARY CALCULATION ---
    avg_model = np.mean(scores_model)
    avg_base = np.mean(scores_base)
    avg_static = np.mean(scores_static)
    avg_adaptive = np.mean(scores_adaptive)
    
    # Calculate improvements vs Baseline
    imp_model = ((avg_base - avg_model) / avg_base) * 100
    imp_static = ((avg_base - avg_static) / avg_base) * 100
    imp_adaptive = ((avg_base - avg_adaptive) / avg_base) * 100
    
    print("=" * 115)
    print(f"{'FINAL STRATEGY COMPARISON':^115}")
    print("=" * 115)
    print(f"{'STRATEGY':<20} | {'AVG MAE (s)':<15} | {'DIFF VS BASE':<15} | {'% IMPROVEMENT':<15}")
    print("-" * 75)
    print(f"{'Baseline':<20} | {avg_base:.4f}s        | {'-':<15} | {'-':<15}")
    print(f"{'XGBoost (Physics)':<20} | {avg_model:.4f}s        | {avg_base - avg_model:+.4f}s        | {imp_model:+.2f}%")
    print(f"{'Static Hybrid':<20} | {avg_static:.4f}s        | {avg_base - avg_static:+.4f}s        | {imp_static:+.2f}%")
    print(f"{'Adaptive Ensemble':<20} | {avg_adaptive:.4f}s        | {avg_base - avg_adaptive:+.4f}s        | {imp_adaptive:+.2f}%")
    print("=" * 115)

if __name__ == "__main__":
    df, feats = load_and_prep()
    run_loro_evaluation(df, feats)