
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_error_distributions():
    csv_path = "final_residuals.csv"
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found. Run final_hybrid.py first.")
        return

    print(f"📊 Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # 1. Box Plot
    print("📈 Generating Box Plot...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Model', y='Residual', data=df, showfliers=False, palette="viridis")
    plt.title('Error Distribution by Model (Outliers Hidden)')
    plt.ylabel('Residual Error (s)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_boxplot.png', dpi=150)
    print("✅ Saved 'error_boxplot.png'")
    
    # 2. Histogram (KDE)
    print("📈 Generating Histogram (KDE)...")
    plt.figure(figsize=(12, 6))
    
    # Filter extreme outliers for visualization clarity (-5s to +5s)
    df_zoom = df[(df['Residual'] > -5) & (df['Residual'] < 5)]
    
    sns.histplot(data=df_zoom, x='Residual', hue='Model', element="step", stat="density", common_norm=False, alpha=0.3, kde=True, palette="viridis")
    plt.title('Residual Distribution (Zoomed ±5s)')
    plt.xlabel('Residual Error (True - Pred) in Seconds')
    plt.xlim(-5, 5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('error_histogram.png', dpi=150)
    print("✅ Saved 'error_histogram.png'")

if __name__ == "__main__":
    try:
        plot_error_distributions()
    except Exception as e:
        print(f"❌ Error plotting: {e}")
