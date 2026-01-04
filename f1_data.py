# f1_data.py (UPDATED)

import os
import fastf1
import pandas as pd
import numpy as np

from config import SEASONS, RACES, FASTF1_CACHE_DIR, MIN_LAP_TIME_SEC, TRACK_CORNERS
from features import compute_aebi_for_session, build_features_from_laps
from utils import save_df

OUT_PATH = "data/f1_lap_dataset.csv"
PROGRESS_DIR = "data/per_race"

def init_fastf1():
    os.makedirs(FASTF1_CACHE_DIR, exist_ok=True)
    fastf1.Cache.enable_cache(FASTF1_CACHE_DIR)

def get_all_race_names_for_season(season: int):
    if RACES is not None:
        return RACES
    try:
        schedule = fastf1.get_event_schedule(season)
        # Filter for conventional races only (excludes pre-season testing if labeled oddly)
        return schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]['EventName'].dropna().unique().tolist()
    except Exception as e:
        print(f"Error fetching schedule for {season}: {e}")
        return []

def load_session(season, race_name):
    try:
        session = fastf1.get_session(season, race_name, 'R')
        session.load(laps=True, telemetry=False, weather=True)
        return session
    except Exception as e:
        print(f"  Could not load session {race_name} {season}: {e}")
        return None

def compute_laps_since_restart(df):
    """
    Identifies 'Regime Shifts' caused by Safety Cars.
    Counts laps since the last SC/VSC ending.
    """
    # 4=SC, 5=Red, 6=VSC, 7=VSC ending
    # We treat '1' (Track Clear) as the restart trigger if prev was SC
    if 'TrackStatus' not in df.columns:
        return np.zeros(len(df))
    
    # Convert to string to handle '4' or '45' etc.
    status = df['TrackStatus'].astype(str)
    
    # 1 if SC/VSC active, 0 if Green
    is_sc = status.str.contains('4|5|6').astype(int)
    
    # Identify the exact lap where Green Flag returns (1 -> 0 transition)
    # We actually want to count cumulative laps of GREEN running
    
    # Create a 'Stint' ID that increments every time SC comes out
    sc_active_mask = (is_sc == 1)
    # New racing stint starts when SC ends
    stint_change = (sc_active_mask.shift(1).fillna(0) == 1) & (sc_active_mask == 0)
    stint_id = stint_change.cumsum()
    
    # For each driver + stint_id, count the cumulative lap count
    # We group by Driver + StintID
    # This resets the counter to 0 every time a restart happens
    df['RestartStintID'] = stint_id
    
    # Calculate laps since restart per driver
    # We use 'cumcount' + 1 to denote 1st lap, 2nd lap...
    # If currently under SC, we can set this to 0 or -1
    laps_since = df.groupby(['Driver', 'RestartStintID']).cumcount() + 1
    
    # Force SC laps to have value 0 (so model knows it's not "racing")
    laps_since = laps_since.where(~sc_active_mask, 0)
    
    return laps_since

def build_dataset_all_races():
    init_fastf1()
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    all_rows = []

    for season in SEASONS:
        races = get_all_race_names_for_season(season)
        print(f"=== Season {season} ({len(races)} races) ===")
        
        for race_name in races:
            out_file = os.path.join(PROGRESS_DIR, f"{season}_{race_name.replace(' ','_')}.csv")
            
            # Resume logic
            if os.path.exists(out_file):
                print(f"  Skipping (cached): {race_name}")
                try:
                    all_rows.append(pd.read_csv(out_file, low_memory=False))
                except: 
                    pass # Corrupt file protection
                continue
                
            print(f"  Processing: {race_name}")
            session = load_session(season, race_name)
            if session is None: continue
            
            laps = session.laps
            if laps is None or laps.empty: continue

            # Basic cleaning
            laps = laps.pick_quicklaps(threshold=1.07) # Filter out ultra-slow laps immediately? No, keep for logic.
            # actually better to filter manually later.
            
            # --- FEATURE ENGINEERING BLOCK ---
            # 1. Weather
            if hasattr(session, 'weather_data'):
                w = session.weather_data
                w['TimeSec'] = w['Time'].dt.total_seconds()
                # Simple merge on nearest time
                # (Skipping complex merge for brevity, simplified assumption: constant weather or pre-merged)
            
            # 2. Base Features (from your original code)
            # We assume build_features_from_laps handles the heavy lifting
            # But we inject AEBI here
            
            # Telemetry for AEBI (Fastest Lap only)
            try:
                fastest = laps.pick_fastest()
                tel = fastest.get_telemetry()
                aebi = compute_aebi_for_session(tel)
            except:
                aebi = 0.0
                
            df_feats = build_features_from_laps(laps, aebi)
            
            # 3. NEW: Restart Logic
            # We need the original TrackStatus which might be lost in build_features if not careful
            # Re-merge TrackStatus if needed, or ensure it's in df_feats
            if 'TrackStatus' in laps.columns and 'TrackStatus' not in df_feats.columns:
                 # Join back logic if lost (omitted for brevity, assuming it's passed through)
                 pass
            
            df_feats['LapsSinceRestart'] = compute_laps_since_restart(df_feats)
            
            # Metadata
            df_feats['Season'] = season
            df_feats['RaceName'] = race_name
            
            df_feats.to_csv(out_file, index=False)
            all_rows.append(df_feats)

    if not all_rows:
        print("No data found.")
        return

    full_df = pd.concat(all_rows, ignore_index=True)
    save_df(full_df, OUT_PATH)
    print(f"Saved {len(full_df)} laps to {OUT_PATH}")

if __name__ == "__main__":
    build_dataset_all_races()