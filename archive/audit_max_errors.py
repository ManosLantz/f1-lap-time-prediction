
import pandas as pd
import numpy as np
import sys
import os

# Import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def audit_errors():
    print("🔍 Loading Data for Error Audit...")
    df, _ = load_and_prep()
    
    # Filter for 2025 and Exclude British GP (already done in main, but explicit here)
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Calculate Baseline Error (Proxy for prediction difficulty)
    # The Model Error will be close to this for huge outliers
    df['BaselineError'] = abs(df['NextLapTimeSec'] - df['PrevLapTimeSec'])
    
    # Find Top 10 Worst Errors
    worst_laps = df.sort_values('BaselineError', ascending=False).head(10)
    
    print("\n🚨 TOP 10 WORST ERRORS (Baseline Proxy) 🚨")
    print(f"{'Race':<25} | {'Driver':<5} | {'Lap':<4} | {'PrevTime':<10} | {'NextTime':<10} | {'Delta':<10}")
    print("-" * 80)
    
    for _, row in worst_laps.iterrows():
        print(f"{row['RaceName']:<25} | {row['Driver']:<5} | {row['LapNumber']:<4} | {row['PrevLapTimeSec']:.1f}s       | {row['NextLapTimeSec']:.1f}s       | {row['BaselineError']:.1f}s")
        
    # Group by Race
    print("\n📊 MAX ERROR PER RACE")
    race_stats = df.groupby('RaceName')['BaselineError'].max().sort_values(ascending=False).head(5)
    print(race_stats)

if __name__ == "__main__":
    audit_errors()
