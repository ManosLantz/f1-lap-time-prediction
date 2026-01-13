
import pandas as pd
import numpy as np
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

DATA_PATH = "data/f1_lap_dataset.csv"

def calculate_difficulty():
    print("🏎️  CALCULATING OVERTAKING DIFFICULTY INDEX...")
    
    # Load raw data directly to avoid filtering context issues
    # We want ALL laps to count overtakes
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # We need Position. If not present, we must infer from cumulative time.
    # The processed loader filters too much (out laps etc), so we use raw for counting.
    
    if 'Position' not in df.columns:
        print("   ⚠️ 'Position' column not found. Inferring from Lap Times...")
        # Sort by Race, Lap, Cumulative Time
        df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
        df['RaceTime'] = df.groupby(['Season', 'RaceName', 'Driver'])['LapTimeSec'].cumsum()
        
        # Rank drivers per lap to get position
        df['Position'] = df.groupby(['Season', 'RaceName', 'LapNumber'])['RaceTime'].rank()
        
    # Calculate Position Changes
    # Sort by Driver, Race, Lap
    df = df.sort_values(['Season', 'RaceName', 'Driver', 'LapNumber'])
    df['PrevPosition'] = df.groupby(['Season', 'RaceName', 'Driver'])['Position'].shift(1)
    
    # Overtake = Position Change (Simple approximation)
    # Note: Pit stops cause "fake" overtakes, but they happen on every track, 
    # so the relative difficulty should still hold (hard tracks have fewer pit-passes too).
    df['PosChange'] = (df['Position'] != df['PrevPosition']).astype(int)
    
    # Group by RaceName
    difficulty = df.groupby('RaceName')['PosChange'].mean().reset_index()
    difficulty.rename(columns={'PosChange': 'OvertakeFreq'}, inplace=True)
    
    # Invert: High Freq = Easy. Low Freq = Hard.
    # Difficulty Score: 1.0 (Hardest) to 0.0 (Easiest)
    # We normalize: Score = 1 - (Freq - Min) / (Max - Min)
    min_freq = difficulty['OvertakeFreq'].min()
    max_freq = difficulty['OvertakeFreq'].max()
    
    difficulty['DifficultyScore'] = 1.0 - ((difficulty['OvertakeFreq'] - min_freq) / (max_freq - min_freq))
    
    print("\n" + "="*60)
    print(f"{'TRACK':<25} | {'FREQ':<10} | {'DIFFICULTY (0-1)':<15}")
    print("-" * 60)
    
    mapping = {}
    
    sorted_diff = difficulty.sort_values('DifficultyScore', ascending=False)
    
    for _, row in sorted_diff.iterrows():
        track = row['RaceName']
        score = row['DifficultyScore']
        print(f"{track:<25} | {row['OvertakeFreq']:.4f}     | {score:.4f}")
        mapping[track] = round(score, 4)
        
    print("="*60)
    
    # Save mapping to JSON
    with open('track_difficulty_map.json', 'w') as f:
        json.dump(mapping, f, indent=4)
        
    print("✅ Saved to track_difficulty_map.json")

if __name__ == "__main__":
    calculate_difficulty()
