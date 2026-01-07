import pandas as pd
import numpy as np

# Configuration
DATA_PATH = "data/f1_lap_dataset.csv"

# Base Features
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
    print("   Checking for missing features...")
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
    print("   Adding Global Field Context (The 'Japan Fix')...")
    field_stats = df.groupby(['Season', 'RaceName', 'LapNumber'])['LapTimeSec'].median().reset_index()
    field_stats.rename(columns={'LapTimeSec': 'FieldMedian_Current'}, inplace=True)
    df = df.merge(field_stats, on=['Season', 'RaceName', 'LapNumber'], how='left')
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['PrevFieldMedian'] = df.groupby(['Season', 'RaceName', 'Driver'])['FieldMedian_Current'].shift(1)
    df['FieldDelta'] = df['PrevLapTimeSec'] - df['PrevFieldMedian']
    df['PrevFieldMedian'] = df['PrevFieldMedian'].fillna(df['PrevLapTimeSec'])
    df['FieldDelta'] = df['FieldDelta'].fillna(0)
    return df

def add_traffic_context(df):
    print("   Adding Local Traffic Context (The 'Azerbaijan Fix')...")
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['RaceTime'] = df.groupby(['Season', 'RaceName', 'Driver'])['LapTimeSec'].cumsum()
    df = df.sort_values(['Season', 'RaceName', 'LapNumber', 'RaceTime'])
    df['TimeAhead'] = df.groupby(['Season', 'RaceName', 'LapNumber'])['RaceTime'].shift(1)
    df['GapToAhead'] = df['RaceTime'] - df['TimeAhead']
    df['GapToAhead'] = df['GapToAhead'].fillna(20.0)
    df['CloseTraffic'] = (df['GapToAhead'] < 1.5).astype(int)
    df.drop(columns=['RaceTime', 'TimeAhead'], inplace=True)
    return df

def load_and_prep():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    if 'Compound' in df.columns:
        df = df[df['Compound'].isin(['SOFT','MEDIUM','HARD'])]
    
    df = add_global_context(df)
    df = add_traffic_context(df)
    
    # Filter for Racing Transitions
    median_pace = df.groupby('RaceName')['LapTimeSec'].transform('median')
    pace_threshold = median_pace * 1.07
    mask_current_fast = df['LapTimeSec'] < pace_threshold
    
    if 'NextLapTimeSec' in df.columns:
        mask_next_fast = df['NextLapTimeSec'] < pace_threshold
        df = df[mask_current_fast & mask_next_fast]
    else:
        df = df[mask_current_fast]
        
    print(f"Filtered dataset to {len(df)} pure racing transitions.")

    df = generate_missing_features(df)
    
    # One-Hot Encoding
    driver_backup = df['Driver'].copy()
    df = pd.get_dummies(df, columns=['Driver', 'Compound'], dummy_na=False)
    df['Driver'] = driver_backup
    
    # Update Feature List
    dummy_cols = [c for c in df.columns if c.startswith('Driver_') or c.startswith('Compound_')]
    new_features = ['PrevFieldMedian', 'FieldDelta', 'GapToAhead', 'CloseTraffic']
    model_features = FEATURES + dummy_cols + new_features

    # 3. DYNAMIC RACE STATUS FILTER (The "Japan Fix 2.0")
    print("   Applying Dynamic Race Status Filter...")
    
    # A. Calculate the 'Green Flag Threshold' per race
    # We find the median pace of the Top 50% of laps (to ignore SCs)
    race_benchmarks = df.groupby(['Season', 'RaceName'])['LapTimeSec'].quantile(0.25)
    
    # Map benchmark back to the dataframe
    df = df.merge(race_benchmarks.rename('RaceBenchmark'), on=['Season', 'RaceName'], how='left')
    
    # B. Define "Slow Lap" Threshold (e.g., 107% of the benchmark)
    # If a lap is > 107% of the benchmark, it's likely an In-Lap, Out-Lap, or VSC.
    df['IsSlowLap'] = df['LapTimeSec'] > (df['RaceBenchmark'] * 1.07)
    
    # C. STRICT FILTER:
    # We remove the lap if:
    # 1. It is slow (Current Lap)
    # 2. The PREVIOUS lap was slow (Baseline is corrupted)
    # 3. The NEXT lap is slow (Target is corrupted)
    
    # Get Prev/Next Slow Flags
    # We group by Driver to ensure we don't shift data between drivers
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['PrevIsSlow'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsSlowLap'].shift(1).fillna(True)
    df['NextIsSlow'] = df.groupby(['Season', 'RaceName', 'Driver'])['IsSlowLap'].shift(-1).fillna(True)
    
    # Keep only Pure Racing Chains (Fast -> Fast -> Fast)
    clean_mask = (~df['IsSlowLap']) & (~df['PrevIsSlow']) & (~df['NextIsSlow'])
    
    df = df[clean_mask].copy()
    
    # Drop temp columns
    df.drop(columns=['RaceBenchmark', 'IsSlowLap', 'PrevIsSlow', 'NextIsSlow'], inplace=True)
    
    print(f"Filtered dataset to {len(df)} PURE racing transitions (Removed VSC/Rain/Pits).")
    return df, model_features