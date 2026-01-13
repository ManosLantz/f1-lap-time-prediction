
import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import mean_absolute_error

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def analyze_drivers():
    print("🏎️  ANALYZING DRIVER ACCURACY...")
    
    # 1. Load Model
    try:
        model = joblib.load('final_f1_model.pkl')
        features = joblib.load('final_model_features.pkl')
        print("   ✅ Model loaded.")
    except FileNotFoundError:
        print("   ❌ Model not found. Run deploy_final_model.py first.")
        return

    # 2. Load Data
    df, _ = load_and_prep()
    mask = (df['Season'] == 2025)
    df = df[mask].copy()
    
    # 3. Predict
    X = df[features]
    y_true_diff = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    print(f"   📊 Predicting {len(df)} laps...")
    pred_diff = model.predict(X)
    
    # 4. Calculate Errors
    # We care about the absolute error of the predicted lap time vs actual lap time.
    # Prediction: NextLap = PrevLap + pred_diff
    # Actual: NextLap
    # Error = |(Prev + pred_diff) - Actual| = |pred_diff - (Actual - Prev)| = |pred_diff - y_true_diff|
    # This is equivalent to MAE on the residuals.
    
    df['Residual'] = np.abs(pred_diff - y_true_diff)
    
    # 5. Group by Driver
    driver_stats = df.groupby('Driver')['Residual'].agg(['mean', 'count', 'std'])
    driver_stats = driver_stats.rename(columns={'mean': 'MAE', 'count': 'Laps', 'std': 'StdDev'})
    
    # Filter drivers with too few laps (e.g. reserves) to avoid skew
    driver_stats = driver_stats[driver_stats['Laps'] > 50].sort_values('MAE')
    
    # 6. Generate Report
    print("\n" + "="*60)
    print(f"{'DRIVER':<15} | {'MAE (s)':<10} | {'LAPS':<6} | {'CONSISTENCY (Std)':<15}")
    print("-" * 60)
    
    for driver, row in driver_stats.iterrows():
        print(f"{driver:<15} | {row['MAE']:.4f}     | {int(row['Laps']):<6} | {row['StdDev']:.4f}")
        
    print("="*60)
    
    # Save to Markdown
    with open('DRIVER_RANKING.md', 'w', encoding='utf-8') as f:
        f.write("# 🏎️ Driver Prediction Accuracy Ranking (2025)\n\n")
        f.write("Which drivers are the most predictable? Which are the 'Wildcards'?\n\n")
        f.write("| Rank | Driver | MAE (s) | Laps | Verdict |\n")
        f.write("|:---:|:---|:---:|:---:|:---|\n")
        
        for i, (driver, row) in enumerate(driver_stats.iterrows(), 1):
            mae = row['MAE']
            if i <= 3: verdict = "🎯 Predictable"
            elif i >= len(driver_stats) - 3: verdict = "🎲 Unpredictable"
            else: verdict = "Average"
            
            f.write(f"| {i} | **{driver}** | `{mae:.4f}` | {int(row['Laps'])} | {verdict} |\n")
            
    print("\n✅ Report saved to DRIVER_RANKING.md")

if __name__ == "__main__":
    analyze_drivers()
