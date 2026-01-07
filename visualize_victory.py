import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from data_loader import load_and_prep
from models import train_physics_model
from sklearn.metrics import mean_absolute_error

def visualize_season_performance():
    print("🚀 Simulating Full 2025 Season (This may take ~2 minutes)...")
    df, features = load_and_prep()
    
    test_races = df[df['Season'] == 2025]['RaceName'].unique()
    race_results = []
    
    # 1. Collect Data Race-by-Race
    for race in test_races:
        test_mask = (df['RaceName'] == race) & (df['Season'] == 2025)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy().sort_values(['Driver', 'LapNumber'])
        
        if len(test_df) < 50: continue
        
        # Train (Delta)
        train_df['DeltaTarget'] = train_df['NextLapTimeSec'] - train_df['PrevLapTimeSec']
        model = train_physics_model(train_df[features], train_df['DeltaTarget'])
        
        # Predict
        preds_delta = model.predict(test_df[features])
        preds_model = test_df['PrevLapTimeSec'] + preds_delta
        preds_base = test_df['PrevLapTimeSec']
        y_true = test_df['NextLapTimeSec']
        
        # Safety Clip
        preds_model = np.maximum(preds_model, preds_base - 2.0)
        
        # Calculate MAE for this race
        mae_model = mean_absolute_error(y_true, preds_model)
        mae_base = mean_absolute_error(y_true, preds_base)
        
        # Calculate Total Error Savings (Sum of errors avoided)
        total_err_model = np.sum(np.abs(y_true - preds_model))
        total_err_base = np.sum(np.abs(y_true - preds_base))
        saved_seconds = total_err_base - total_err_model
        
        race_results.append({
            'Race': race.replace(" Grand Prix", ""), # Shorten name for plot
            'Model MAE': mae_model,
            'Baseline MAE': mae_base,
            'Saved Seconds': saved_seconds
        })
        print(f"  > Finished {race}")

    results_df = pd.DataFrame(race_results)

    # === PLOT 1: RACE-BY-RACE BATTLE ===
    plt.figure(figsize=(16, 8))
    
    # Transform for plotting (Melt)
    plot_df = results_df.melt(id_vars='Race', value_vars=['Model MAE', 'Baseline MAE'], var_name='Strategy', value_name='Error (s)')
    
    sns.barplot(data=plot_df, x='Race', y='Error (s)', hue='Strategy', palette=['blue', 'gray'])
    plt.title("2025 Season: AI vs Baseline Error per Race", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Mean Absolute Error (Lower is Better)")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig('season_comparison_bar.png')
    print("✅ Saved 'season_comparison_bar.png'")

    # === PLOT 2: CUMULATIVE SAVINGS ===
    # This shows the "Total Value" of the AI over the season
    plt.figure(figsize=(16, 8))
    
    # Calculate cumulative savings
    results_df['Cumulative Saved'] = results_df['Saved Seconds'].cumsum()
    
    sns.lineplot(data=results_df, x='Race', y='Cumulative Saved', marker='o', color='green', linewidth=3)
    plt.fill_between(results_df['Race'], results_df['Cumulative Saved'], color='green', alpha=0.1)
    
    plt.title("Cumulative Prediction Accuracy Gained by AI over 2025", fontsize=16)
    plt.ylabel("Total Seconds Saved vs Baseline (Cumulative)")
    plt.xticks(rotation=45, ha='right')
    plt.axhline(0, color='black', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('season_cumulative_value.png')
    print("✅ Saved 'season_cumulative_value.png'")

if __name__ == "__main__":
    visualize_season_performance()