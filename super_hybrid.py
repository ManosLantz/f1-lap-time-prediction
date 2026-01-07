import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from data_loader import load_and_prep
from models import train_physics_model

# === CONFIGURATION ===
# 1. The Heavyweight XGBoost (Trend/Physics Expert)
# Uses the params that got you 0.4753s
def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor(
        learning_rate=0.005,
        n_estimators=500,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.1,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model

# 2. The Robust Random Forest (Texture/Noise Expert)
RF_PARAMS = {
    'n_estimators': 300,
    'max_depth': None,
    'min_samples_leaf': 5,
    'n_jobs': -1,
    'random_state': 42
}

# 3. Adaptive Settings
ALPHA = 1  # How fast we forget past errors (0.5 = Balanced)

def run_dual_adaptive_logic(preds_A, preds_B, y_true, alpha=1):
    """
    Dynamically weighs Model A vs Model B based on recent performance.
    """
    adaptive_preds = []
    w = 0.5      # Start neutral (50/50 split)
    e_A = 0.5    # Error history for Model A
    e_B = 0.5    # Error history for Model B
    
    # Trace for visualization (optional debugging)
    weights_trace = []

    for i in range(len(preds_A)):
        # 1. Make Combined Prediction using current weights
        pred = (w * preds_A[i]) + ((1 - w) * preds_B[i])
        adaptive_preds.append(pred)
        weights_trace.append(w)
        
        # 2. Measure who was actually right
        err_A = abs(y_true[i] - preds_A[i])
        err_B = abs(y_true[i] - preds_B[i])
        
        # 3. Update Error History (Exponential Moving Average)
        e_A = (alpha * err_A) + ((1 - alpha) * e_A)
        e_B = (alpha * err_B) + ((1 - alpha) * e_B)
        
        # 4. Update Weight for NEXT lap
        # If Model A has lower error, w increases towards 1.0
        # If Model B has lower error, w decreases towards 0.0
        if (e_A + e_B) > 0:
            w = e_B / (e_A + e_B) # Invert: High error A -> Low weight A
        else:
            w = 0.5
            
    return np.array(adaptive_preds), weights_trace

def run_super_hybrid():
    print("🚀 STARTING SUPER-HYBRID TEST (XGBoost vs RandomForest)...")
    print(f"   Adaptive Alpha: {ALPHA}")
    
    df, features = load_and_prep()
    
    # We will test on 2025 data (Hold-out year)
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    
    results = []
    
    for race in test_races:
        # LORO Split
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # Prepare Targets
        y_true = test_df['NextLapTimeSec'].values
        preds_base = test_df['PrevLapTimeSec'].values
        
        # --- MODEL 1: XGBOOST ---
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        xgb_model = train_xgboost(train_df[features], train_df['DeltaTarget'])
        
        preds_delta_xgb = xgb_model.predict(test_df[features])
        preds_xgb = test_df['PrevLapTimeSec'].values + preds_delta_xgb
        # Safety Clip (prevent impossible lap times)
        preds_xgb = np.maximum(preds_xgb, preds_base - 2.0)

        # --- MODEL 2: RANDOM FOREST ---
        # Note: RF usually likes absolute targets, but to be fair we train on Delta too
        rf_model = RandomForestRegressor(**RF_PARAMS)
        rf_model.fit(train_df[features], train_df['DeltaTarget'])
        
        preds_delta_rf = rf_model.predict(test_df[features])
        preds_rf = test_df['PrevLapTimeSec'].values + preds_delta_rf
        preds_rf = np.maximum(preds_rf, preds_base - 2.0)
        
        # --- ADAPTIVE FUSION ---
        preds_hybrid, _ = run_dual_adaptive_logic(preds_xgb, preds_rf, y_true, alpha=ALPHA)
        
        # --- METRICS ---
        mae_base = mean_absolute_error(y_true, preds_base)
        mae_xgb = mean_absolute_error(y_true, preds_xgb)
        mae_rf = mean_absolute_error(y_true, preds_rf)
        mae_hybrid = mean_absolute_error(y_true, preds_hybrid)
        
        results.append({
            'Race': race.replace(" Grand Prix", ""),
            'Baseline': mae_base,
            'XGB_Solo': mae_xgb,
            'RF_Solo': mae_rf,
            'Super_Hybrid': mae_hybrid
        })
        
        # Print live update (Cleaner format)
        winner = "HYBRID" if mae_hybrid < min(mae_xgb, mae_rf) else ("XGB" if mae_xgb < mae_rf else "RF")
        print(f"  > {race[:15]:<15} | XGB: {mae_xgb:.3f} | RF: {mae_rf:.3f} | HYB: {mae_hybrid:.3f} | Winner: {winner}")

    # --- FINAL REPORT ---
    res_df = pd.DataFrame(results)
    avg_base = res_df['Baseline'].mean()
    avg_xgb = res_df['XGB_Solo'].mean()
    avg_rf = res_df['RF_Solo'].mean()
    avg_hyb = res_df['Super_Hybrid'].mean()
    
    print("\n" + "="*60)
    print("🏁 FINAL SEASON RESULTS (MAE)")
    print("="*60)
    print(f"1. Baseline:      {avg_base:.4f}s")
    print(f"2. Random Forest: {avg_rf:.4f}s  (Improvement: {((avg_base-avg_rf)/avg_base)*100:.2f}%)")
    print(f"3. XGBoost:       {avg_xgb:.4f}s  (Improvement: {((avg_base-avg_xgb)/avg_base)*100:.2f}%)")
    print("-" * 60)
    print(f"🏆 SUPER HYBRID:  {avg_hyb:.4f}s  (Improvement: {((avg_base-avg_hyb)/avg_base)*100:.2f}%)")
    print("="*60)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Baseline', 'RF', 'XGB', 'Super Hybrid'], y=[avg_base, avg_rf, avg_xgb, avg_hyb], palette='viridis')
    plt.title("Model Showdown: Can the Hybrid Beat the Specialists?")
    plt.ylabel("Mean Absolute Error (s)")
    plt.ylim(0.45, 0.55)
    plt.savefig("super_hybrid_results.png")
    print("Saved plot to super_hybrid_results.png")

if __name__ == "__main__":
    run_super_hybrid()