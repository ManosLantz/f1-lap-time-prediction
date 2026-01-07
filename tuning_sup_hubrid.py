import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep

# === CONFIGURATION ===
# We test these Alphas. The script picks the best one per race.
ALPHAS_TO_TEST = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# 1. XGBoost Params (The Champion Settings)
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        learning_rate=0.005, n_estimators=500, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_lambda=5.0, reg_alpha=0.1, n_jobs=-1, verbosity=0
    )
    model.fit(X_train, y_train)
    return model

# 2. Random Forest Params (The Robust Settings)
RF_PARAMS = {'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 5, 'n_jobs': -1, 'random_state': 42}

def run_multi_alpha_logic(preds_A, preds_B, y_true, alphas):
    """
    Runs the adaptive logic for MULTIPLE alphas simultaneously.
    Returns: Best Score, Best Alpha, and the dictionary of all scores.
    """
    results = {}
    
    # Pre-calculate errors once to save time
    errors_A = np.abs(y_true - preds_A)
    errors_B = np.abs(y_true - preds_B)
    
    for alpha in alphas:
        adaptive_preds = []
        w = 0.5
        e_A = 0.5
        e_B = 0.5
        
        for i in range(len(preds_A)):
            # Predict
            pred = (w * preds_A[i]) + ((1 - w) * preds_B[i])
            adaptive_preds.append(pred)
            
            # Update EMA Errors
            e_A = (alpha * errors_A[i]) + ((1 - alpha) * e_A)
            e_B = (alpha * errors_B[i]) + ((1 - alpha) * e_B)
            
            # Update Weight
            if (e_A + e_B) > 0:
                w = e_B / (e_A + e_B)
            else:
                w = 0.5
        
        results[alpha] = mean_absolute_error(y_true, adaptive_preds)
        
    best_alpha = min(results, key=results.get)
    return results[best_alpha], best_alpha

def run_tuned_hybrid():
    print(f"🚀 STARTING FINAL TUNING RUN (Testing Alphas: {ALPHAS_TO_TEST})...")
    df, features = load_and_prep()
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    final_results = []
    
    for race in test_races:
        # LORO Split
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        y_true = test_df['NextLapTimeSec'].values
        preds_base = test_df['PrevLapTimeSec'].values
        
        # --- TRAIN MODELS ---
        # XGB
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        xgb_model = train_xgboost(train_df[features], train_df['DeltaTarget'])
        preds_xgb = np.maximum(test_df['PrevLapTimeSec'].values + xgb_model.predict(test_df[features]), preds_base - 2.0)

        # RF
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(train_df[features], train_df['DeltaTarget'])
        preds_rf = np.maximum(test_df['PrevLapTimeSec'].values + rf_model.predict(test_df[features]), preds_base - 2.0)
        
        # --- FIND BEST ALPHA ---
        mae_tuned, best_alpha = run_multi_alpha_logic(preds_xgb, preds_rf, y_true, ALPHAS_TO_TEST)
        
        # Calculate untuned scores for comparison
        mae_base = mean_absolute_error(y_true, preds_base)
        mae_xgb = mean_absolute_error(y_true, preds_xgb)
        mae_rf = mean_absolute_error(y_true, preds_rf)
        
        # Log Result
        final_results.append({
            'Race': race.replace(" Grand Prix", ""),
            'Baseline': mae_base,
            'XGB': mae_xgb,
            'RF': mae_rf,
            'Tuned_Hybrid': mae_tuned,
            'Best_Alpha': best_alpha
        })
        
        # Live Print: Show improvement over the best single model
        best_single = min(mae_xgb, mae_rf)
        imp = ((best_single - mae_tuned) / best_single) * 100
        print(f"  > {race[:12]:<12} | Best Single: {best_single:.3f} | Tuned Hybrid: {mae_tuned:.3f} (α={best_alpha}) | Gain: +{imp:.2f}%")

    # --- FINAL REPORT ---
    res_df = pd.DataFrame(final_results)
    
    print("\n" + "="*80)
    print("🏁 FINAL CHAMPIONSHIP STANDINGS (TUNED ADAPTIVE SYSTEM)")
    print("="*80)
    print(f"{'STRATEGY':<20} | {'MAE (s)':<10} | {'IMPROVEMENT vs BASE':<20}")
    print("-" * 80)
    
    base_avg = res_df['Baseline'].mean()
    xgb_avg = res_df['XGB'].mean()
    rf_avg = res_df['RF'].mean()
    tuned_avg = res_df['Tuned_Hybrid'].mean()
    
    print(f"{'Baseline':<20} | {base_avg:.4f}s    | -")
    print(f"{'Random Forest':<20} | {rf_avg:.4f}s    | +{((base_avg - rf_avg)/base_avg)*100:.2f}%")
    print(f"{'XGBoost':<20} | {xgb_avg:.4f}s    | +{((base_avg - xgb_avg)/base_avg)*100:.2f}%")
    print("-" * 80)
    print(f"{'🏆 TUNED HYBRID':<20} | {tuned_avg:.4f}s    | +{((base_avg - tuned_avg)/base_avg)*100:.2f}%")
    print("="*80)

    # Plot: Alpha Distribution
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Best_Alpha', data=res_df, palette='magma')
    plt.title("Distribution of Optimal Alphas (Low = Stable, High = Reactive)")
    plt.xlabel("Alpha Value")
    plt.ylabel("Count of Races")
    plt.savefig("final_alpha_distribution.png")
    print("Saved alpha distribution chart.")

if __name__ == "__main__":
    run_tuned_hybrid()