import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep
from models import train_physics_model, run_adaptive_ensemble

def run_loro_evaluation(df, features):
    """
    Performs Leave-One-Race-Out (LORO) Cross-Validation using DELTA TARGETS.
    """
    print(f"\n{'RACE NAME':<27} | {'MODEL':<8} | {'BASE':<8} | {'STATIC':<8} | {'ADAPTIVE':<8} | {'WINNER':<9}")
    print("-" * 115)
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    
    scores_model, scores_base, scores_static, scores_adaptive = [], [], [], []
    
    for race in test_races:
        # 1. Split Data (Leave One Race Out)
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # === THE FIX: CREATE DELTA TARGET ===
        # We train the model to predict CHANGE in time, not absolute time.
        # This makes the model "Track Agnostic".
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        
        # 2. Train Physics Model on DELTA
        # Note: We pass 'DeltaTarget' instead of 'NextLapTimeSec'
        model = train_physics_model(train_df[features], train_df['DeltaTarget'])
        
        # 3. Predict DELTA
        preds_delta = model.predict(test_df[features])
        if hasattr(preds_delta, "values"): preds_delta = preds_delta.values
            
        # 4. Convert Delta back to Absolute Time
        # Prediction = Previous Lap + Predicted Change
        preds_model = test_df['PrevLapTimeSec'].values + preds_delta
        
        preds_base = test_df['PrevLapTimeSec'].values
        y_true = test_df['NextLapTimeSec'].values
        
        # Safety Clip (Physics sanity check)
        preds_model = np.maximum(preds_model, preds_base - 2.0)

        # 5. Run Adaptive Ensemble Logic
        adaptive_preds = run_adaptive_ensemble(test_df, preds_model, preds_base, y_true)

        # 6. Calculate Metrics
        mae_model = mean_absolute_error(y_true, preds_model)
        mae_base = mean_absolute_error(y_true, preds_base)
        mae_static = mean_absolute_error(y_true, (0.5*preds_model + 0.5*preds_base))
        mae_adaptive = mean_absolute_error(y_true, adaptive_preds)
        
        scores_model.append(mae_model)
        scores_base.append(mae_base)
        scores_static.append(mae_static)
        scores_adaptive.append(mae_adaptive)
        
        race_scores = {'MODEL': mae_model, 'BASE': mae_base, 'STATIC': mae_static, 'ADAPTIVE': mae_adaptive}
        winner = min(race_scores, key=race_scores.get)
        
        print(f"{race:<27} | {mae_model:.3f}s   | {mae_base:.3f}s   | {mae_static:.3f}s   | {mae_adaptive:.3f}s   | {winner}")

    # --- FINAL SUMMARY ---
    avg_model = np.mean(scores_model)
    avg_base = np.mean(scores_base)
    avg_static = np.mean(scores_static)
    avg_adaptive = np.mean(scores_adaptive)
    
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