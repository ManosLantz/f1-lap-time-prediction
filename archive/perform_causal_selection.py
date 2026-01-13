
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import GraphicalLassoCV
import sys
import os

# Import data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_and_prep

def run_causal_discovery():
    print("⏳ Loading Data for Causal Discovery...")
    df, features = load_and_prep()
    
    # Filter 2025 and Exclude British GP
    mask = (df['Season'] == 2025) & (df['RaceName'] != 'British Grand Prix')
    df = df[mask].copy()
    
    # Define Target: Delta (Next - Prev)
    target_col = 'Target_Delta'
    df[target_col] = df['NextLapTimeSec'] - df['PrevLapTimeSec']
    
    # Prepare Matrix X containing Features + Target
    # We include ALL features to see which ones directly connect to Target
    cols_to_analyze = features + [target_col]
    
    # 1. Preprocessing
    # Drop constant columns if any
    X = df[cols_to_analyze].copy()
    X = X.loc[:, (X != X.iloc[0]).any()] 
    
    # Handle NaNs (Drop rows)
    X = X.dropna()
    
    print(f"📊 Dataset for Causal Graph: {X.shape}")
    
    # Standardize (Required for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 2. Graphical Lasso (Sparse Inverse Covariance)
    # This finds the Conditional Independence structure (Markov Network)
    # If Entry[i, j] is non-zero, variables i and j are directly connected given all others.
    print("🕸️ Estimating Causal Graph (Graphical Lasso)...")
    
    # Alpha controls sparsity. CV automatically finds best alpha.
    model = GraphicalLassoCV(cv=5, n_jobs=-1, max_iter=200)
    model.fit(X_scaled)
    
    print(f"✅ Best Alpha (Sparsity Penalty): {model.alpha_:.4f}")
    
    # 3. Analyze Connections to Target
    precision_matrix = model.precision_
    
    # Find index of target
    target_idx = list(X.columns).index(target_col)
    
    # Get partial correlations (normalized precision)
    # P_ij_norm = -P_ij / sqrt(P_ii * P_jj)
    target_precision = precision_matrix[target_idx, :]
    diag = np.diag(precision_matrix)
    
    partial_corrs = []
    
    print("\n" + "="*80)
    print(f"{'FEATURE':<30} | {'PARTIAL CORR':<15} | {'CAUSAL STATUS'}")
    print("-" * 80)
    
    for i, col in enumerate(X.columns):
        if i == target_idx: continue
        
        # Calculate Partial Correlation
        p_val = target_precision[i]
        scale = np.sqrt(diag[i] * diag[target_idx])
        partial_r = -p_val / scale
        
        # Threshold for "Direct Connection"
        # In standardized Lasso, non-zero implies connection.
        # We use a small epsilon to filter numerical noise.
        is_connected = abs(partial_r) > 0.005 # Strict cutoff
        
        status = "🔗 DIRECT CAUSE" if is_connected else "❌ COND. INDEP. (Indirect)"
        
        if is_connected:
            partial_corrs.append((col, partial_r))
            print(f"{col:<30} | {partial_r:>8.4f}        | {status}")
            
    # Print Disconnected
    # for i, col in enumerate(X.columns):
    #     if i == target_idx: continue
    #     if col not in [x[0] for x in partial_corrs]:
    #         print(f"{col:<30} | {'0.0000':>8}        | ❌ COND. INDEP. (Indirect)")
            
    print("="*80)
    print("INTERPRETATION:")
    print("- 'DIRECT CAUSE': Feature has information about Target NOT explained by any other feature.")
    print("- 'COND. INDEP.': Feature is redundant; its info is already carried by the Direct Causes.")
    print("- Positive Part. Corr: Increases Lap Time (Slower)")
    print("- Negative Part. Corr: Decreases Lap Time (Faster)")

if __name__ == "__main__":
    run_causal_discovery()
