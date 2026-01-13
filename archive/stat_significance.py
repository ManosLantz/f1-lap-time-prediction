
import pandas as pd
import numpy as np
from scipy import stats

def check_significance():
    csv_path = "final_residuals.csv"
    print(f"📊 Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("❌ Error: final_residuals.csv not found. Run final_hybrid.py first.")
        return

    # Pivot to get paired samples: Index (implicit) | Baseline | XGB_Solo | RF_Solo | Hybrid
    # We assume the order of residuals is preserved per model (which it is in the generation script)
    # To be safe, we just reshape assuming equal lengths and aligned indices
    
    models = df['Model'].unique()
    print(f"Models found: {models}")
    
    # Extract arrays
    err_hybrid = abs(df[df['Model'] == 'Hybrid']['Residual'].values)
    err_xgb = abs(df[df['Model'] == 'XGB_Solo']['Residual'].values)
    err_rf = abs(df[df['Model'] == 'RF_Solo']['Residual'].values)
    err_base = abs(df[df['Model'] == 'Baseline']['Residual'].values)
    
    # Ensure lengths match
    min_len = min(len(err_hybrid), len(err_xgb), len(err_rf), len(err_base))
    err_hybrid = err_hybrid[:min_len]
    err_xgb = err_xgb[:min_len]
    err_rf = err_rf[:min_len]
    err_base = err_base[:min_len]
    
    comparisons = [
        ("Hybrid vs XGB Solo", err_hybrid, err_xgb),
        ("Hybrid vs RF Solo", err_hybrid, err_rf),
        ("XGB Solo vs RF Solo", err_xgb, err_rf),
        ("Hybrid vs Baseline", err_hybrid, err_base)
    ]
    
    print("\n" + "="*80)
    print(f"{'COMPARISON':<25} | {'MEAN DIFF':<12} | {'T-TEST p-val':<12} | {'WILCOXON p-val':<12} | {'SIGNIFICANT?'}")
    print("-" * 80)
    
    for name, series_a, series_b in comparisons:
        # A is usually the "new" model (Hybrid)
        # Difference > 0 means A has HIGHER error (worse)
        # Difference < 0 means A has LOWER error (better)
        diff = series_a - series_b
        mean_diff = np.mean(diff)
        
        # 1. Paired T-Test (Parametric, assumes normality of mean diff)
        t_stat, p_t = stats.ttest_rel(series_a, series_b)
        
        # 2. Wilcoxon Signed-Rank (Non-parametric, robust to outliers)
        # using 'pratt' method to handle zeros if any, though unlikely to have exact ties
        w_stat, p_w = stats.wilcoxon(series_a, series_b)
        
        # Significance Threshold
        sig = "✅ YES" if p_w < 0.05 else "❌ NO"
        if p_w < 0.001: sig = "✅✅ YES!!"
        
        print(f"{name:<25} | {mean_diff:+.5f}s    | {p_t:.2e}     | {p_w:.2e}     | {sig}")
        
    print("="*80)
    print("\ninterpretation:")
    print("- Mean Diff: Negative means the first model has lower error (Better).")
    print("- P-Value < 0.05: The difference is statistically significant (95% confidence).")
    print("- P-Value < 0.001: Extremely significant.")

if __name__ == "__main__":
    check_significance()
