import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from train_model import load_and_prep, FEATURES

# === CONFIG ===
TEST_SEASON = 2025
ALPHA = 0.5  # Your optimal alpha

def generate_graphs():
    print("Loading data for visualization...")
    df, features = load_and_prep()
    
    # 1. Setup Train/Test Split (LORO 2025)
    test_mask = (df['Season'] == TEST_SEASON)
    train_df = df[~test_mask]
    test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
    
    # 2. Train Physics Model
    print("Training Model...")
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6, 
        subsample=0.7, colsample_bytree=0.8, min_child_weight=3, 
        reg_lambda=5.0, n_jobs=-1
    )
    model.fit(train_df[features], train_df['NextLapTimeSec'])
    
    # 3. Generate Raw Predictions
    print("Generating Predictions...")
    preds_model = model.predict(test_df[features])
    preds_base = test_df['PrevLapTimeSec'].values
    y_true = test_df['NextLapTimeSec'].values
    preds_model = np.maximum(preds_model, preds_base - 2.0)
    
    # 4. Run Adaptive Ensemble
    adaptive_preds = []
    drivers = test_df['Driver'].unique()
    current_idx = 0
    
    for driver in drivers:
        n_laps = len(test_df[test_df['Driver'] == driver])
        d_p_model = preds_model[current_idx : current_idx + n_laps]
        d_p_base = preds_base[current_idx : current_idx + n_laps]
        d_y_true = y_true[current_idx : current_idx + n_laps]
        
        w = 0.5
        e_m = 0.5
        e_b = 0.5
        
        for i in range(n_laps):
            pred = (w * d_p_model[i]) + ((1 - w) * d_p_base[i])
            adaptive_preds.append(pred)
            
            err_m = abs(d_y_true[i] - d_p_model[i])
            err_b = abs(d_y_true[i] - d_p_base[i])
            e_m = (ALPHA * err_m) + ((1 - ALPHA) * e_m)
            e_b = (ALPHA * err_b) + ((1 - ALPHA) * e_b)
            if (e_m + e_b) > 0: w = e_b / (e_m + e_b)
            else: w = 0.5
        
        current_idx += n_laps

    test_df['Adaptive_Pred'] = adaptive_preds
    test_df['Actual'] = y_true

    # === GRAPH 1: Average Pace Comparison per Circuit ===
    print("Generating Bar Chart...")
    # Group by Race and get mean pace
    race_stats = test_df.groupby('RaceName')[['Actual', 'Adaptive_Pred']].mean().reset_index()
    
    # Melt for easier plotting with seaborn
    race_stats_melt = race_stats.melt(id_vars='RaceName', var_name='Type', value_name='LapTime')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(data=race_stats_melt, x='RaceName', y='LapTime', hue='Type', palette=['#1f77b4', '#ff7f0e'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Average Actual vs. Predicted Pace by Circuit ({TEST_SEASON})", fontsize=14)
    plt.ylabel("Average Lap Time (s)")
    plt.xlabel("")
    plt.legend(title="Legend")
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('average_pace_comparison.png')
    print("Saved 'average_pace_comparison.png'")

    # === GRAPH 2: Error Distribution by Circuit (Boxplot) ===
    print("Generating Boxplot...")
    test_df['Abs_Error'] = abs(test_df['Actual'] - test_df['Adaptive_Pred'])
    
    # Sort races by median error to see hardest/easiest tracks
    order = test_df.groupby('RaceName')['Abs_Error'].median().sort_values().index
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=test_df, x='RaceName', y='Abs_Error', order=order, showfliers=False, palette="viridis")
    plt.xticks(rotation=45, ha='right')
    plt.title(f"Prediction Error Distribution by Circuit ({TEST_SEASON})", fontsize=14)
    plt.ylabel("Absolute Error (s)")
    plt.xlabel("")
    plt.axhline(0.55, color='red', linestyle='--', label='Global MAE (0.55s)')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('error_distribution_by_circuit.png')
    print("Saved 'error_distribution_by_circuit.png'")
    
    # === GRAPH 3: Actual vs Predicted Scatter (Global) ===
    print("Generating Scatter Plot...")
    plt.figure(figsize=(10, 10))
    # Sample 5000 points to avoid overplotting if dataset is huge
    sample_df = test_df.sample(min(5000, len(test_df)), random_state=42)
    
    sns.scatterplot(x=sample_df['Actual'], y=sample_df['Adaptive_Pred'], alpha=0.3, color='purple')
    
    # Draw perfect prediction line
    lims = [min(sample_df['Actual']), max(sample_df['Actual'])]
    plt.plot(lims, lims, color='red', linestyle='--', label='Perfect Prediction')
    
    plt.title("Actual vs. Predicted Lap Times (Global Sample)", fontsize=14)
    plt.xlabel("Actual Lap Time (s)")
    plt.ylabel("Predicted Lap Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('global_accuracy_scatter.png')
    print("Saved 'global_accuracy_scatter.png'")

if __name__ == "__main__":
    generate_graphs()