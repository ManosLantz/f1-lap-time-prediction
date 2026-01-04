import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Import your existing pipeline
from train_model import load_and_prep, FEATURES

# === CONFIG ===
TARGET_RACE = "Japanese Grand Prix"
TARGET_SEASON = 2025
FOCUS_DRIVERS = ["VER", "HAM", "NOR", "ALO"] # Drivers to plot specifically

def diagnose_specific_race():
    print(f"Loading data to diagnose {TARGET_RACE}...")
    df, features = load_and_prep()
    
    # 1. Split Data
    # Train on everything EXCEPT the target race (LORO)
    test_mask = (df['RaceName'] == TARGET_RACE) & (df['Season'] == TARGET_SEASON)
    train_df = df[~test_mask]
    test_df = df[test_mask]
    
    if len(test_df) == 0:
        print(f"Error: No data found for {TARGET_RACE} {TARGET_SEASON}")
        return

    print(f"Training Model on {len(train_df)} laps...")
    print(f"Testing on {len(test_df)} laps from {TARGET_RACE}...")

    # 2. Train with your BEST parameters
    model = xgb.XGBRegressor(
        n_estimators=300, 
        learning_rate=0.03, 
        max_depth=6, 
        subsample=0.7, 
        colsample_bytree=0.8, 
        min_child_weight=3, 
        reg_lambda=5.0, 
        n_jobs=-1
    )
    model.fit(train_df[features], train_df['NextLapTimeSec'])
    
    # 3. Predict
    preds = model.predict(test_df[features])
    
    # Add predictions to dataframe for plotting
    plot_df = test_df.copy()
    plot_df['PredictedLap'] = preds
    plot_df['ActualLap'] = plot_df['NextLapTimeSec']
    plot_df['Residual'] = plot_df['ActualLap'] - plot_df['PredictedLap']
    
    # Calculate Metrics
    mae = mean_absolute_error(plot_df['ActualLap'], plot_df['PredictedLap'])
    bias = np.mean(plot_df['Residual'])
    print(f"\nResults for {TARGET_RACE}:")
    print(f"MAE: {mae:.3f}s")
    print(f"Bias: {bias:.3f}s (Positive = Model is too fast / Actual is slower)")

    # === PLOTTING ===
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Race Trace (Actual vs Predicted) for specific drivers
    ax = axes[0]
    for driver in FOCUS_DRIVERS:
        d_data = plot_df[plot_df['Driver_' + driver] == 1] if f'Driver_{driver}' in plot_df.columns else pd.DataFrame()
        
        # Fallback if dummy column logic differs, try manual string match if available
        if d_data.empty and 'Driver' in plot_df.columns:
             d_data = plot_df[plot_df['Driver'] == driver]
             
        if not d_data.empty:
            d_data = d_data.sort_values('LapNumber')
            ax.plot(d_data['LapNumber'], d_data['ActualLap'], label=f"{driver} Actual", linestyle='-', alpha=0.7)
            ax.plot(d_data['LapNumber'], d_data['PredictedLap'], label=f"{driver} Model", linestyle='--', linewidth=2)
            
    ax.set_ylabel("Lap Time (s)")
    ax.set_title(f"Race Pace Trace: Actual (Solid) vs Model (Dashed)\nWatch for spikes that the model misses!")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals over time (All Drivers)
    ax = axes[1]
    sns.scatterplot(data=plot_df, x='LapNumber', y='Residual', hue='TyreAge', palette='viridis', ax=ax, alpha=0.6)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_ylabel("Error (Actual - Pred)")
    ax.set_title("Where does the error happen? (Color = Tyre Age)")
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Distribution of Errors
    ax = axes[2]
    sns.histplot(plot_df['Residual'], bins=50, kde=True, ax=ax, color='salmon')
    ax.axvline(0, color='black', linestyle='--')
    ax.set_xlabel("Lap Number") # Shared X axis implies this, but for hist it's actually Error
    ax.set_xlabel("Prediction Error (seconds)")
    ax.set_title(f"Error Distribution (Bias = {bias:.2f}s)")
    
    plt.tight_layout()
    plt.savefig("japan_deep_dive.png")
    print("\nSaved diagnostic plot to 'japan_deep_dive.png'")

if __name__ == "__main__":
    diagnose_specific_race()