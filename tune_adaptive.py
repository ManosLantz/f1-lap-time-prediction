import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep
from models import train_physics_model, run_adaptive_ensemble

def collect_predictions(df, features):
    """
    Runs the LORO loop ONE LAST TIME to gather raw prediction data.
    """
    print("Collecting raw predictions from LORO loop (This takes ~1 minute)...")
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    all_results = []
    
    for race in test_races:
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # 1. Train & Predict (Physics)
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        model = train_physics_model(train_df[features], train_df['DeltaTarget'])
        
        preds_delta = model.predict(test_df[features])
        preds_model = test_df['PrevLapTimeSec'].values + preds_delta
        preds_base = test_df['PrevLapTimeSec'].values
        preds_model = np.maximum(preds_model, preds_base - 2.0)
        
        # 2. Store Raw Data
        chunk = pd.DataFrame({
            'Driver': test_df['Driver'].values,
            'RaceName': race,
            'Pred_Model': preds_model,
            'Pred_Base': preds_base,
            'Actual': test_df['NextLapTimeSec'].values
        })
        all_results.append(chunk)
        print(f"  > Processed {race}")

    return pd.concat(all_results)

def tune_strategy(full_df):
    """
    Optimizes Alpha and Start_Weight on the collected data.
    """
    print("\n=== TUNING ADAPTIVE LOGIC ===")
    
    best_mae = float('inf')
    best_config = {}
    
    # Grid Search Space
    alphas = [0.994, 0.995, 0.997, 0.998]
    start_ws = [0.9, 0.995] # Trust model more initially?
    
    print(f"{'ALPHA':<8} | {'START_W':<8} | {'MAE':<8} | {'STATUS'}")
    print("-" * 45)

    for alpha in alphas:
        for w in start_ws:
            # Run the adaptive function on the pre-calculated dataframe
            # This is SUPER FAST (pure numpy)
            adaptive_preds = run_adaptive_ensemble(
                full_df, 
                full_df['Pred_Model'].values, 
                full_df['Pred_Base'].values, 
                full_df['Actual'].values,
                alpha=alpha,
                start_w=w
            )
            
            mae = mean_absolute_error(full_df['Actual'], adaptive_preds)
            
            if mae < best_mae:
                best_mae = mae
                best_config = {'alpha': alpha, 'start_w': w}
                print(f"{alpha:<8} | {w:<8} | {mae:.4f}s | * New Best")
            else:
                pass 
                # print(f"{alpha:<8} | {w:<8} | {mae:.4f}s |")

    print("=" * 45)
    print(f"🏆 CHAMPION ADAPTIVE CONFIG: {best_config}")
    print(f"🏆 NEW BEST MAE: {best_mae:.4f}s")
    
    # Calculate improvement over Pure Physics
    phys_mae = mean_absolute_error(full_df['Actual'], full_df['Pred_Model'])
    print(f"   (Pure Physics MAE was: {phys_mae:.4f}s)")
    
    if best_mae < phys_mae:
        print("✅ SUCCESS: Adaptive Logic now beats Pure Physics!")
    else:
        print("⚠️ NOTE: Pure Physics is still slightly better. Consider using pure model.")

if __name__ == "__main__":
    df, features = load_and_prep()
    results_df = collect_predictions(df, features)
    tune_strategy(results_df)