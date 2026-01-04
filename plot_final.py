import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_thesis_summary():
    # Data from your milestones
    data = {
        'Stage': ['1. Physics Only', '1. Physics Only', '1. Physics Only',
                  '2. + Global Context', '2. + Global Context', '2. + Global Context',
                  '3. + Local Traffic', '3. + Local Traffic', '3. + Local Traffic'],
        'Metric': ['Japan (Regime)', 'Baku (Traffic)', 'Global Avg',
                   'Japan (Regime)', 'Baku (Traffic)', 'Global Avg',
                   'Japan (Regime)', 'Baku (Traffic)', 'Global Avg'],
        'MAE': [0.73, 1.60, 0.67,  # Stage 1 (Approx from your early runs)
                0.47, 1.50, 0.54,  # Stage 2 (After Japan Fix)
                0.44, 1.10, 0.51]  # Stage 3 (Final Result)
    }
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    
    # Custom Palette: Red (Bad) -> Yellow (Better) -> Green (Best)
    palette = sns.color_palette("RdYlGn", 3)
    
    ax = sns.barplot(data=df, x='Metric', y='MAE', hue='Stage', palette='viridis')
    
    # Add numbers on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs', padding=3)
        
    plt.title("The Evolution of Model Accuracy: Curing 'State Blindness'", fontsize=14, fontweight='bold')
    plt.ylabel("Prediction Error (MAE seconds)", fontsize=12)
    plt.xlabel("Problem Category", fontsize=12)
    plt.ylim(0, 1.8)
    plt.legend(title='Development Stage')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("final_thesis_results.png")
    print("Saved final plot to final_thesis_results.png")

if __name__ == "__main__":
    plot_thesis_summary()