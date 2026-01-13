
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def check_traffic_impact():
    print("🕵️ INVESTIGATING MONACO TRAFFIC ANOMALY...")
    df, _ = load_and_prep()
    
    # 2025 Data Only
    df = df[df['Season'] == 2025]
    
    # Define Target: Lap Time difference (Slower/Faster than prev)
    df['Delta'] = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    # Tracks to compare
    tracks = ['Monaco Grand Prix', 'Italian Grand Prix', 'Bahrain Grand Prix']
    
    print("\n" + "="*70)
    print(f"{'TRACK':<20} | {'CORRELATION (FieldDelta vs Slowdown)':<35}")
    print("-" * 70)
    
    stats_list = []
    
    for track in tracks:
        track_df = df[df['RaceName'] == track]
        if len(track_df) == 0: continue
        
        # We look at cases where there IS traffic (FieldDelta < 3.0s)
        # to see the impact of close racing.
        traffic_df = track_df[track_df['FieldDelta'] < 3.0]
        
        if len(traffic_df) < 20:
             corr = 0
        else:
            # Correlation between FieldDelta and LapTime Delta
            # Positive FieldDelta (Further away) -> Should be Faster (Negative LapDelta)
            # So we expect NEGATIVE Correlation normally (More Space = Faster).
            # If correlation is near zero, it means distance doesn't matter much.
            corr = traffic_df['FieldDelta'].corr(traffic_df['Delta'])
            
        avg_loss = traffic_df['Delta'].mean()
        
        print(f"{track:<20} | {corr:.4f} (Avg Loss in Traffic: {avg_loss:.3f}s)")
        stats_list.append({'Track': track, 'Corr': corr})

    print("="*70)
    print("\nanalysis:")
    print("- Monaco: Traffic (Low FieldDelta) usually means huge time loss (High Delta).")
    print("  So we expect a strong relationship.")
    print("- Monza: Traffic might provide a slipstream.")

if __name__ == "__main__":
    check_traffic_impact()
